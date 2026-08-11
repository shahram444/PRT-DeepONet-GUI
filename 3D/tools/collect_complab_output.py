#!/usr/bin/env python3
"""
collect_complab_output.py — turn a finished CompLaB campaign into ONE training-ready HDF5 file,
and report every run that failed and why.

    python collect_complab_output.py --campaign ./campaign --geometries ./geometries \
                      --out ./dataset --mode steady

Outputs
-------
    dataset/dataset.h5   every usable simulation, in the layout
                               dataset_reader.py (and therefore the DeepONet)
                               consumes directly
    dataset/failures.csv       one row per failed run, with the reason
    dataset/campaign_report.md human-readable summary

Failure handling
----------------
Nothing here aborts on a bad run.  Runs are validated in three stages and every
rejection is recorded with a specific reason:
    stage 1  run_one.sh already marked it failed (crash, timeout, no output)
    stage 2  expected .vti files missing or unreadable
    stage 3  the data itself is unusable (NaN, all-zero, wrong shape,
             concentration above a settings_and_units bound)

HDF5 layout
-----------
    /geom/gid          (G,)   int32     geometry id
    /geom/material     (G,nx,ny,nz) uint8    0 solid, 1 wall, 2 pore
    /geom/gdf          (G,nx,ny,nz) float32  geodesic distance from inlet
    /geom/edt          (G,nx,ny,nz) float32  Euclidean distance (ablation control)
    /samples/geom_index(S,)   int32     row into /geom
    /samples/run_id    (S,)   int32
    /samples/params    (S,6)  float32   pe, da_bio, da_abio, ks_ac, ks_a, Y
    /samples/t_norm    (S,T)  float32   normalised snapshot times
    /samples/conc      (S,T,C,nx,ny,nz) float16
    /samples/velocity  (S,3,nx,ny,nz)   float16   (omit with --no-velocity)
    attrs: species=['Ac','A','P','Bio'], param_names=[...], conc_scale=(C,)
"""

import argparse, base64, csv, json, os, re, struct, sys, zlib
from collections import Counter, OrderedDict
import numpy as np


# ======================================================================= VTI
def _read_vti_pyvista(path):
    import pyvista as pv
    m = pv.read(path)
    ext = m.extent                      # (x0,x1,y0,y1,z0,z1) of points
    dims = (ext[1] - ext[0] + 1, ext[3] - ext[2] + 1, ext[5] - ext[4] + 1)
    out = {}
    for name in m.array_names:
        a = np.asarray(m[name])
        n = int(np.prod(dims))
        if a.size == n:                                  # scalar
            out[name] = a.reshape(dims[::-1]).transpose(2, 1, 0)
        elif a.size == 3 * n:                            # vector
            out[name] = a.reshape(dims[::-1] + (3,)).transpose(3, 2, 1, 0)
    return out, dims


_HDR = {"UInt32": ("<I", 4), "UInt64": ("<Q", 8)}


def _decode_inline(b64, dtype, header_type, compressed):
    """Decode an inline <DataArray format="binary"> payload.

    There are TWO conventions in the wild and a reader has to survive both.

      (a) one base64 stream over  header + data      -- what some writers emit
      (b) the header base64-encoded SEPARATELY from the data, and the two
          strings concatenated                        -- what the VTK library
                                                         itself writes

    Under (b) the header is 16 bytes for UInt32, which is not a multiple of 3,
    so it carries base64 padding in the middle of the string.  Decoding the
    concatenation as a single stream then yields garbage and zlib reports
    "incomplete or truncated stream".  That failure only appears once the data
    is compressed, which is why it can sit unnoticed until a real campaign
    comes back.

    We try (a) first and fall back to (b).
    """
    b64 = re.sub(r"\s", "", b64)
    try:
        return _decode(base64.b64decode(b64), dtype, header_type, compressed)
    except Exception:                                          # noqa: BLE001
        if not compressed:
            raise
    # convention (b): work out how long the header is, decode it on its own
    fmt, hs = _HDR[header_type]
    probe = base64.b64decode(b64[:24] + "==")           # >= 3 words for UInt32
    nblocks = struct.unpack(fmt, probe[:hs])[0]
    hbytes = (3 + nblocks) * hs
    hchars = 4 * ((hbytes + 2) // 3)
    head = base64.b64decode(b64[:hchars])
    data = base64.b64decode(b64[hchars:])
    return _decode(head + data, dtype, header_type, compressed)


def _decode(raw, dtype, header_type, compressed):
    fmt, hs = _HDR[header_type]
    if not compressed:
        n = struct.unpack(fmt, raw[:hs])[0]
        return np.frombuffer(raw[hs:hs + n], dtype=dtype)
    nblocks = struct.unpack(fmt, raw[:hs])[0]
    off = hs * 3
    sizes = [struct.unpack(fmt, raw[off + i * hs: off + (i + 1) * hs])[0]
             for i in range(nblocks)]
    off += nblocks * hs
    buf = b""
    for s in sizes:
        buf += zlib.decompress(raw[off:off + s]); off += s
    return np.frombuffer(buf, dtype=dtype)


def _read_vti_manual(path):
    """Minimal VTK ImageData reader: ascii, inline base64 (compressed or not),
    and appended raw/base64.  Used when pyvista/vtk are unavailable."""
    with open(path, "rb") as f:
        blob = f.read()
    head_end = blob.find(b"<AppendedData")
    header = (blob if head_end < 0 else blob[:head_end]).decode("utf-8", "replace")

    m = re.search(r'WholeExtent="([-\d\s]+)"', header)
    ext = [int(v) for v in m.group(1).split()]
    dims = (ext[1] - ext[0] + 1, ext[3] - ext[2] + 1, ext[5] - ext[4] + 1)
    n = int(np.prod(dims))

    compressed = "vtkZLibDataCompressor" in header
    ht = re.search(r'header_type="(\w+)"', header)
    header_type = ht.group(1) if ht else "UInt32"

    appended = None
    if head_end >= 0:
        enc = re.search(r'<AppendedData\s+encoding="(\w+)"', blob[head_end:head_end + 200]
                        .decode("utf-8", "replace")).group(1)
        start = blob.find(b"_", head_end) + 1
        stop = blob.find(b"</AppendedData>", start)
        appended = (enc, blob[start:stop])

    np_of = {"Float64": np.float64, "Float32": np.float32,
             "Int32": np.int32, "UInt8": np.uint8, "Int64": np.int64}
    out = {}
    for mt in re.finditer(
            r'<DataArray\b([^>]*)>(.*?)</DataArray>|<DataArray\b([^>]*)/>',
            header, re.S):
        attrs = mt.group(1) or mt.group(3)
        body = mt.group(2) or ""
        gn = re.search(r'Name="([^"]+)"', attrs)
        gt = re.search(r'type="(\w+)"', attrs)
        gf = re.search(r'format="(\w+)"', attrs)
        if not (gn and gt and gf):
            continue
        name, dtype, fmt = gn.group(1), np_of.get(gt.group(1), np.float64), gf.group(1)
        ncomp = int((re.search(r'NumberOfComponents="(\d+)"', attrs) or
                     re.match(r'(1)', "1")).group(1))
        if fmt == "ascii":
            arr = np.fromstring(body.strip(), sep=" ", dtype=np.float64).astype(dtype)
        elif fmt == "binary":
            arr = _decode_inline(body, dtype, header_type, compressed)
        elif fmt == "appended":
            off = int(re.search(r'offset="(\d+)"', attrs).group(1))
            enc, data = appended
            raw = data[off:] if enc == "raw" else base64.b64decode(
                re.sub(rb"\s", b"", data))[off:]
            arr = _decode(raw, dtype, header_type, compressed)
        else:
            continue
        if arr.size == n and ncomp == 1:
            out[name] = arr.reshape(dims[::-1]).transpose(2, 1, 0).astype(np.float32)
        elif arr.size == 3 * n:
            out[name] = arr.reshape(dims[::-1] + (3,)).transpose(3, 2, 1, 0).astype(np.float32)
    return out, dims


def read_vti(path):
    try:
        return _read_vti_pyvista(path)
    except Exception:
        return _read_vti_manual(path)


# ================================================================ collection
def snapshots(out_dir, name):
    """[(iteration, path)] for every <name>_<7digit>.vti, in iteration order.

    The ITERATION NUMBER is kept, not just the ordering. Runs stop at their own
    convergence iteration, so run A's 5th snapshot and run B's 5th snapshot are
    not the same settings_and_units time. Ordering alone would silently mislabel time."""
    rx = re.compile(r"^%s_(\d{7})\.vti$" % re.escape(name))
    hits = []
    for f in os.listdir(out_dir):
        m = rx.match(f)
        if m:
            hits.append((int(m.group(1)), os.path.join(out_dir, f)))
    return sorted(hits)


def _pick_scalar(arrs, species_name, shape):
    """Find the concentration array inside a .vti.

    CompLaB writes it as 'Density', which is what we look for first.  But a
    hard-coded name is a brittle contract with somebody else's writer: a
    different Palabos version, or a hand-exported file, may call it after the
    species, or may simply hold one scalar array with some other name.  Falling
    back in that order costs nothing and turns a total failure into a
    successful read.
    """
    for key in ("Density", species_name, species_name.lower(),
                species_name.upper(), "concentration", "Concentration"):
        if key in arrs and arrs[key].shape == shape:
            return arrs[key]
    cands = [v for v in arrs.values() if v.shape == shape]
    return cands[0] if len(cands) == 1 else None


def load_run(rdir, species, microbes, shape, mode, want_velocity, max_conc,
             n_times=None):
    """Returns (conc (T,C,nx,ny,nz) float32, t_norm (T,), velocity or None) or
    raises ValueError with a specific, reportable reason.

    mode='steady'      the last snapshot only, T=1, t_norm=[1.0]
    mode='transient'   T snapshots on a COMMON normalised time axis.

    Why the common axis matters: <ade_converge_iT> lets each run stop when it
    reaches steady state, so runs end at different iterations and hold different
    numbers of snapshots. Stacking those raw would either crash on the shape or,
    worse, put different settings_and_units times in the same slot. With --n-times K each
    run is resampled onto t_norm = 0, 1/(K-1), ..., 1 of ITS OWN run length, so
    slot k means the same fraction of the approach to steady state in every run,
    which is what the trunk's t input has to mean."""
    od = os.path.join(rdir, "output")
    if not os.path.isdir(od):
        raise ValueError("no output directory")

    fields = list(species) + list(microbes)
    per = OrderedDict()
    for nm in fields:
        s = snapshots(od, nm)
        if not s:
            raise ValueError("no .vti for species '%s'" % nm)
        per[nm] = s

    ntimes = min(len(v) for v in per.values())
    if ntimes == 0:
        raise ValueError("no snapshots")

    # the common time axis, expressed in this run's own iterations
    iters = [it for it, _ in per[fields[0]][:ntimes]]
    span = max(iters[-1], 1)
    if mode == "steady":
        idx, t_norm = [ntimes - 1], np.array([1.0], np.float32)
    elif n_times is None:
        idx = list(range(ntimes))
        t_norm = np.array(iters, np.float32) / span
    else:
        if ntimes < 2:
            raise ValueError("only %d snapshot(s); --mode transient needs at least 2 "
                             "(raise <ade_max_iT> or lower --vtk-interval)" % ntimes)
        want = np.linspace(0.0, 1.0, int(n_times))
        have = np.array(iters, np.float32) / span
        idx = [int(np.abs(have - w).argmin()) for w in want]   # nearest snapshot
        t_norm = want.astype(np.float32)

    conc = np.zeros((len(idx), len(fields)) + shape, np.float32)
    for ci, nm in enumerate(fields):
        for ti, k in enumerate(idx):
            path = per[nm][k][1]
            arrs, dims = read_vti(path)
            a = _pick_scalar(arrs, nm, shape)
            if a is None:
                raise ValueError(
                    "no usable scalar array in %s; the file contains %s"
                    % (os.path.basename(path),
                       ", ".join("%s%s" % (k2, tuple(v.shape))
                                 for k2, v in arrs.items()) or "nothing"))
            if a.shape != shape:
                raise ValueError("shape %s != expected %s in %s"
                                 % (a.shape, shape, os.path.basename(path)))
            conc[ti, ci] = a

    if not np.all(np.isfinite(conc)):
        raise ValueError("non-finite values (NaN/Inf) in concentration fields")
    if float(np.abs(conc).max()) == 0.0:
        raise ValueError("all concentration fields are identically zero")
    if float(conc.max()) > max_conc:
        raise ValueError("concentration %.3g exceeds bound %.3g (blow-up)"
                         % (float(conc.max()), max_conc))
    frac_neg = float((conc < -1e-12).mean())
    if frac_neg > 0.05:
        raise ValueError("%.1f%% of voxels negative (unstable kinetics)" % (100 * frac_neg))

    vel = None
    vel_is_magnitude_only = False
    if want_velocity:
        # CompLaB writes the flow field as nsLattice_*.vti, but a single
        # hard-coded prefix is a brittle contract with somebody else's writer.
        # Missing the velocity is not a cosmetic loss: it makes --flow-proxy and
        # --dim-free IMPOSSIBLE later, and it used to happen silently, so the
        # first sign of trouble was a training run refusing to start days later.
        s = []
        for prefix in ("nsLattice", "velocity", "Velocity", "vel", "flow"):
            s = snapshots(od, prefix)
            if s:
                break
        if s:
            arrs, _ = read_vti(s[-1][1])
            for key in ("velocity", "Velocity", "u", "U"):
                if key in arrs and arrs[key].shape == (3,) + shape:
                    vel = arrs[key].astype(np.float32); break
            if vel is None:
                vecs = [v for v in arrs.values() if v.shape == (3,) + shape]
                if len(vecs) == 1:
                    vel = vecs[0].astype(np.float32)
            if vel is None and "velocityNorm" in arrs:
                # A magnitude is NOT a vector field. Putting |u| into the
                # x-component makes the flow look purely axial everywhere, and
                # travel_time then integrates that fiction into a plausible but
                # wrong tau. We still take it -- it is better than nothing for
                # the branch -- but the run is FLAGGED so the caller can refuse
                # to use it for --flow-proxy.
                vel = np.zeros((3,) + shape, np.float32)
                vel[0] = arrs["velocityNorm"]
                vel_is_magnitude_only = True
    return conc, t_norm, vel, vel_is_magnitude_only


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign", required=True, nargs="+",
                   help="one campaign dir, or several (e.g. campaign/batch_*). "
                        "Batches merge into a single dataset; run ids are globally "
                        "unique so there are no collisions.")
    p.add_argument("--geometries", required=True)
    p.add_argument("--out", default="./dataset")
    p.add_argument("--mode", choices=["steady", "transient"], default="steady",
                   help="steady: the final snapshot only (T=1). transient: a time "
                        "series, which is what makes the trunk's t input mean "
                        "something. Use --n-times with it.")
    p.add_argument("--n-times", type=int, default=None, metavar="K",
                   help="transient only: resample every run onto K equally spaced "
                        "normalised times in [0,1]. REQUIRED in practice, because "
                        "runs converge at different iterations and otherwise hold "
                        "different snapshot counts. K=6 is a good start; the file "
                        "grows linearly in K.")
    p.add_argument("--no-velocity", action="store_true")
    p.add_argument("--max-conc", type=float, default=1.0,
                   help="reject a run whose max concentration exceeds this (mol/L)")
    p.add_argument("--compress", default="gzip", choices=["gzip", "lzf", "none"])
    args = p.parse_args()

    import h5py
    os.makedirs(args.out, exist_ok=True)
    # expand each --campaign entry: a campaign dir, or a parent holding batch_*
    camps = []
    for c in args.campaign:
        if os.path.isdir(os.path.join(c, "runs")):
            camps.append(c)
        else:
            sub = sorted(os.path.join(c, d) for d in os.listdir(c)
                         if d.startswith("batch_") and
                         os.path.isdir(os.path.join(c, d, "runs")))
            if not sub:
                sys.exit("no runs/ and no batch_*/runs under %s" % c)
            camps += sub
    run_names = []          # (label, full path)
    for c in camps:
        rd = os.path.join(c, "runs")
        for d in sorted(os.listdir(rd)):
            if d.startswith("run_"):
                lbl = d if len(camps) == 1 else "%s/%s" % (os.path.basename(c), d)
                run_names.append((lbl, os.path.join(rd, d)))
    if not run_names:
        sys.exit("no runs found under: %s" % ", ".join(camps))
    print("collecting %d runs from %d campaign dir(s)" % (len(run_names), len(camps)))

    # ---- stage 1+2: triage ------------------------------------------------
    good, failures = [], []
    for d, rd in run_names:
        try:
            with open(os.path.join(rd, "params.json")) as f:
                pr = json.load(f)
        except Exception as e:
            failures.append(dict(run=d, gid=-1, stage="params", reason="unreadable params.json: %s" % e))
            continue
        sp = os.path.join(rd, "status.json")
        if not os.path.isfile(sp):
            failures.append(dict(run=d, gid=pr["gid"], stage="run", reason="never ran (no status.json)"))
            continue
        try:
            with open(sp) as f:
                st = json.load(f)
        except Exception as e:
            failures.append(dict(run=d, gid=pr["gid"], stage="run", reason="unreadable status.json: %s" % e))
            continue
        if st.get("state") != "ok":
            failures.append(dict(run=d, gid=pr["gid"], stage="run",
                                 reason=st.get("reason") or "marked failed",
                                 exit_code=st.get("exit_code"), wall_s=st.get("wall_s")))
            continue
        good.append((d, rd, pr, st))

    if not good:
        _report(args, run_names, [], failures, None)
        sys.exit("no successful runs to collect_complab_output")

    shape = (good[0][2]["nx"], good[0][2]["ny"], good[0][2]["nz"])
    species, microbes = good[0][2]["species"], good[0][2]["microbes"]
    fields = species + microbes

    # ---- geometries -------------------------------------------------------
    gids = sorted({pr["gid"] for _, _, pr, _ in good})
    gindex = {g: i for i, g in enumerate(gids)}
    G = len(gids)
    mat = np.zeros((G,) + shape, np.uint8)
    gdf = np.zeros((G,) + shape, np.float32)
    edt = np.zeros((G,) + shape, np.float32)
    for g in gids:
        f = os.path.join(args.geometries, "geom_%04d" % g, "geom_%04d.npz" % g)
        z = np.load(f)
        i = gindex[g]
        mat[i] = z["material"]; gdf[i] = np.nan_to_num(z["gdf"]); edt[i] = z["edt"]

    # ---- stage 3: read the fields ----------------------------------------
    T = 1 if args.mode == "steady" else (args.n_times if args.n_times else None)
    C = len(fields)
    recs = []
    for d, rd, pr, st in good:
        try:
            conc, tnorm, vel, vmag = load_run(rd, species, microbes, shape, args.mode,
                                        not args.no_velocity, args.max_conc,
                                        args.n_times)
        except Exception as e:
            failures.append(dict(run=d, gid=pr["gid"], stage="data", reason=str(e),
                                 wall_s=st.get("wall_s")))
            continue
        if T is None:
            T = conc.shape[0]
        if conc.shape[0] != T:
            failures.append(dict(run=d, gid=pr["gid"], stage="data",
                                 reason="snapshot count %d != %d; runs stop at their "
                                        "own convergence iteration, so pass --n-times "
                                        "to put every run on one time axis"
                                        % (conc.shape[0], T)))
            continue
        recs.append((d, pr, conc, tnorm, vel, vmag))
        print("  %s  gid=%d  Pe=%.3g  Da_bio=%.3g  max=%.3g"
              % (d, pr["gid"], pr["pe"], pr["da_bio"], float(conc.max())))

    if not recs:
        _report(args, run_names, [], failures, None)
        sys.exit("every run failed validation; see campaign_report.md")

    # ---- write HDF5 -------------------------------------------------------
    S = len(recs)
    comp = None if args.compress == "none" else args.compress
    h5p = os.path.join(args.out, "dataset.h5")
    with h5py.File(h5p, "w") as h:
        gg = h.create_group("geom")
        gg.create_dataset("gid", data=np.array(gids, np.int32))
        gg.create_dataset("material", data=mat, compression=comp)
        gg.create_dataset("gdf", data=gdf, compression=comp)
        gg.create_dataset("edt", data=edt, compression=comp)

        sg = h.create_group("samples")
        sg.create_dataset("geom_index", data=np.array([gindex[r[1]["gid"]] for r in recs], np.int32))
        sg.create_dataset("run_id", data=np.array([r[1]["run_id"] for r in recs], np.int32))
        pnames = ["pe", "da_bio", "da_abio", "ks_ac_norm", "ks_a_norm", "y_norm"]
        sg.create_dataset("params", data=np.array(
            [[r[1][k] for k in pnames] for r in recs], np.float32))
        sg.create_dataset("t_norm", data=np.array(
            [np.asarray(r[3], np.float32) for r in recs], np.float32))

        dc = sg.create_dataset("conc", shape=(S, T, C) + shape, dtype=np.float16,
                               compression=comp, chunks=(1, 1, 1) + shape)
        for i, r in enumerate(recs):
            dc[i] = r[2].astype(np.float16)

        n_vel = sum(1 for r in recs if r[4] is not None)
        n_mag = sum(1 for r in recs if len(r) > 5 and r[5])
        if n_mag:
            print("\n" + "!" * 72)
            print("%d of %d runs supplied only the velocity MAGNITUDE, not the"
                  % (n_mag, len(recs)))
            print("vector field. It has been stored in the x-component, which")
            print("makes the flow look purely axial everywhere. The branch input")
            print("is usable; the TRAVEL TIME derived from it is not, so treat")
            print("--flow-proxy and --dim-free results on this dataset as")
            print("unreliable. Check what your CompLaB build writes to")
            print("nsLattice_*.vti.")
            print("!" * 72 + "\n")
        if not args.no_velocity and n_vel == 0:
            print("\n" + "!" * 72)
            print("NO FLOW FIELD WAS FOUND IN ANY RUN.")
            print("The dataset will have no /samples/velocity, and that means the")
            print("--flow-proxy and --dim-free switches CANNOT be used with it: they")
            print("replace the geometry with the flow, so they need it present.")
            print("Looked for nsLattice_*.vti, velocity_*.vti, vel_*.vti and")
            print("flow_*.vti inside each run's output/ folder, holding a 3-component")
            print("array. Check what your CompLaB build actually writes.")
            print("Pass --no-velocity to silence this if you do not need the flow.")
            print("!" * 72 + "\n")
        elif not args.no_velocity and n_vel < len(recs):
            print("\nWARNING: only %d of %d runs had a readable flow field. The "
                  "others will hold zeros." % (n_vel, len(recs)))
        if not args.no_velocity and any(r[4] is not None for r in recs):
            dv = sg.create_dataset("velocity", shape=(S, 3) + shape, dtype=np.float16,
                                   compression=comp, chunks=(1, 3) + shape)
            for i, r in enumerate(recs):
                if r[4] is not None:
                    dv[i] = r[4].astype(np.float16)

        # Per-species scale so the ML side normalises identically every time.
        # Accumulated run by run. The old one-liner stacked EVERY run for a
        # species into one array -- (S, T, nx, ny, nz) float32, over 10 GB for a
        # realistic 240-run 128x64x64 campaign -- and did it at the very end,
        # after the HDF5 was already written. A multi-day collection would die
        # with MemoryError on its last statement.
        scale = np.full(C, 1e-30, np.float64)
        for r in recs:
            for ci in range(C):
                m = float(np.abs(r[2][:, ci]).max())
                if m > scale[ci]:
                    scale[ci] = m
        scale = scale.astype(np.float32)
        sg.attrs["conc_scale"] = scale
        h.attrs["species"] = np.array(fields, dtype="S8")
        h.attrs["param_names"] = np.array(pnames, dtype="S16")
        h.attrs["mode"] = args.mode
        h.attrs["shape"] = np.array(shape, np.int32)
        h.attrs["n_samples"] = S
        h.attrs["n_geometries"] = G
        h.attrs["velocity_magnitude_only_runs"] = int(
            sum(1 for r in recs if len(r) > 5 and r[5]))

    _report(args, run_names, recs, failures, (h5p, S, G, T, C, shape))


def _report(args, run_names, recs, failures, ds):
    fcsv = os.path.join(args.out, "failures.csv")
    if failures:
        keys = sorted({k for f in failures for k in f})
        with open(fcsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(failures)

    L = []
    L.append("# Campaign report\n")
    L.append("Total runs built: **%d**  |  usable: **%d**  |  failed: **%d**  (%.1f%% yield)\n"
             % (len(run_names), len(recs), len(failures),
                100.0 * len(recs) / max(len(run_names), 1)))
    if ds:
        h5p, S, G, T, C, shape = ds
        sz = os.path.getsize(h5p) / 1e9
        L.append("\n## Dataset\n")
        L.append("`%s` — %.2f GB\n" % (os.path.basename(h5p), sz))
        L.append("\n%d samples over %d geometries, %d snapshot(s), %d field(s), grid %dx%dx%d.\n"
                 % (S, G, T, C, *shape))
    if failures:
        L.append("\n## Failures\n")
        by_stage = Counter(f["stage"] for f in failures)
        L.append("By stage: " + ", ".join("%s=%d" % kv for kv in sorted(by_stage.items())) + "\n")
        L.append("\n| count | reason |\n|---:|---|\n")
        for reason, k in Counter(f["reason"][:110] for f in failures).most_common(30):
            L.append("| %d | %s |\n" % (k, reason.replace("|", "/")))
        L.append("\n### Every failed run\n")
        L.append("\n| run | geometry | stage | reason |\n|---|---:|---|---|\n")
        for f in failures:
            L.append("| %s | %s | %s | %s |\n"
                     % (f["run"], f.get("gid", "?"), f["stage"],
                        str(f["reason"])[:140].replace("|", "/")))
        L.append("\nFull detail in `failures.csv`.\n")
        L.append("\n**stage=run** failures (crash, timeout, missing input) are worth "
                 "re-queueing as-is: `python complab_campaign.py retry --out <campaign>`.\n")
        L.append("\n**stage=data** failures are physics failures — the run completed but "
                 "produced unusable fields (blow-up, all-zero, negative concentrations). "
                 "Re-running the same parameters reproduces them exactly. Fix the cause "
                 "instead: check that `PRT_DT` matches the printed `[ADE] dt`, lower "
                 "`PRT_MAXFRAC`, or exclude that corner of the Pe/Da space. `complab_campaign.py "
                 "retry` deliberately does NOT re-queue these.\n")
    else:
        L.append("\nNo failures.\n")

    with open(os.path.join(args.out, "campaign_report.md"), "w") as f:
        f.writelines(L)
    print("\n" + "".join(L[:4]))
    print("wrote %s" % os.path.join(args.out, "campaign_report.md"))


if __name__ == "__main__":
    main()
