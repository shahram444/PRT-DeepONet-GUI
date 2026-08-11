#!/usr/bin/env python3
"""
import_2d_simulations.py — turn SOMEBODY ELSE'S 2D simulation output into a dataset we
can train on.

WHY THIS IS SEPARATE FROM build_dataset_2d.py
    build_dataset_2d.py takes geometries and solves the physics ITSELF.  That is
    what you use with the published release, because the release ships
    geometries, trained weights and notebooks -- and NO simulation fields.

    This file is for the other case: a collaborator sends actual simulation
    output, concentration fields and all, and we want to train on THEIR physics
    rather than re-solving it.  That is the case Christof's todo describes:
    "discuss with Heewon about adding his simulation results" to the database.

START WITH --dry-run
    Ask for ONE run first, point this at it with --dry-run, and it will report
    exactly what it found and what is missing, without writing anything.  That
    one round trip is much cheaper than receiving a hundred runs in a layout we
    cannot read.

        python import_2d_simulations.py --src /path/to/one_run --dry-run

WHAT IT ACCEPTS
    A folder of .npz files, one per run, or a single .npz holding stacked
    arrays.  Array names are matched case-insensitively against a list of
    aliases, so 'C', 'conc', 'concentration' and 'sol' are all understood.
    Concentration may be (nx,ny), (T,nx,ny), (C,nx,ny) or (T,C,nx,ny); the
    ambiguous (T or C, nx, ny) case is resolved with --n-species.

WHAT IT NEEDS, AT MINIMUM
    geometry       the pore mask, so the network knows the structure
    concentration  the field to be predicted
    Pe and Da      the conditions each run was carried out at

WHAT IT WOULD LIKE
    velocity       the flow field. Without it the --flow-proxy and --dim-free
                   switches cannot be used at all.
    time           more than one snapshot, so the trunk's time input means
                   something. A single snapshot gives a steady-state dataset.

See HEEWON_DATA.md for the request to send, written out.
"""

import argparse
import glob
import json
import os
import sys
import numpy as np
import h5py
from scipy import ndimage

SOLID, WALL, PORE = 0, 1, 2

GEOM_KEYS = ["material", "domain", "geometry", "geom", "mask", "m", "phase", "img"]
CONC_KEYS = ["conc", "concentration", "c", "sol", "solution", "field", "u", "y"]
VEL_KEYS = ["velocity", "vel", "u_vec", "uv", "flow"]
GDF_KEYS = ["gdf", "geodesic", "dist_inlet", "distance"]
PE_KEYS = ["pe", "peclet", "pe_number"]
DA_KEYS = ["da", "damkohler", "da_number", "da_bio"]
T_KEYS = ["t", "time", "t_norm", "times"]


def _find(d, keys, ndim=None):
    low = {str(k).lower(): k for k in d}
    for k in keys:
        if k in low:
            v = d[low[k]]
            if ndim is None or getattr(v, "ndim", 0) in ndim:
                return low[k], np.asarray(v)
    if ndim is not None:
        for k, orig in low.items():
            v = np.asarray(d[orig])
            if v.ndim in ndim:
                return orig, v
    return None, None


def to_mask(a):
    """Whatever coding the sender used -> CompLaB's 0 solid, 1 wall, 2 pore."""
    a = np.asarray(a)
    u = np.unique(a)
    if set(u.tolist()) <= {0, 1}:
        # which value is pore? the majority of a percolating medium is usually
        # not the pore phase, so decide by connectivity instead of by count
        best, bestn = None, -1
        for v in (1, 0):
            lab, _ = ndimage.label(a == v)
            top = set(np.unique(lab[0][a[0] == v]).tolist()) - {0}
            bot = set(np.unique(lab[-1][a[-1] == v]).tolist()) - {0}
            span = len(top & bot)
            if span > bestn:
                best, bestn = v, span
        return np.where(a == best, PORE, SOLID).astype(np.uint8)
    if set(u.tolist()) <= {0, 1, 2}:
        if (a == 1).mean() > 0.25:               # published convention
            return np.where(a == 0, SOLID, PORE).astype(np.uint8)
        return a.astype(np.uint8)                # already ours
    return np.where(a > 0, PORE, SOLID).astype(np.uint8)


def geodesic_2d(g):
    INF = np.float32(1e9)
    d = np.where(g == PORE, INF, np.nan).astype(np.float32)
    d[0][g[0] == PORE] = 0.0
    for _ in range(4 * g.shape[0]):
        prev = d.copy()
        for ax in (0, 1):
            for sh in (1, -1):
                nb = np.roll(d, sh, axis=ax)
                sl = [slice(None)] * 2
                sl[ax] = 0 if sh > 0 else -1
                nb[tuple(sl)] = INF
                d = np.where(g == PORE,
                             np.fmin(d, np.nan_to_num(nb, nan=INF) + 1.0), np.nan)
        if np.nanmax(np.abs(np.nan_to_num(d - prev, nan=0.0))) < 1e-6:
            break
    return np.nan_to_num(d, nan=0.0).astype(np.float32)


def shape_conc(c, n_species):
    """Normalise the concentration array to (T, C, nx, ny)."""
    c = np.asarray(c, np.float32)
    if c.ndim == 2:
        return c[None, None]
    if c.ndim == 4:
        return c
    if c.ndim == 3:
        lead = c.shape[0]
        if n_species and lead == n_species:
            return c[None]                       # (C, nx, ny) -> one snapshot
        return c[:, None]                        # (T, nx, ny) -> one species
    raise ValueError("concentration has %d dimensions, expected 2, 3 or 4" % c.ndim)


def read_run(path, n_species):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    r = {"path": path, "keys": list(d)}
    gk, g = _find(d, GEOM_KEYS, ndim=(2,))
    ck, c = _find(d, CONC_KEYS, ndim=(2, 3, 4))
    if ck == gk:
        ck, c = None, None
    vk, v = _find(d, VEL_KEYS, ndim=(3, 4))
    dk, gd = _find(d, GDF_KEYS, ndim=(2,))
    r.update(geom_key=gk, conc_key=ck, vel_key=vk, gdf_key=dk)
    r["geom"] = to_mask(g) if g is not None else None
    r["conc"] = shape_conc(c, n_species) if c is not None else None
    r["vel"] = np.asarray(v, np.float32) if v is not None else None
    r["gdf"] = np.asarray(gd, np.float32) if gd is not None else None
    for name, keys in (("pe", PE_KEYS), ("da", DA_KEYS)):
        k, val = _find(d, keys)
        r[name] = float(np.asarray(val).ravel()[0]) if val is not None else None
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="a folder of per-run .npz files, or a single .npz")
    ap.add_argument("--out", default=None, help="the dataset to write")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what was found and write nothing. USE THIS "
                         "FIRST, on a single example run.")
    ap.add_argument("--n-species", type=int, default=1)
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--pe", type=float, default=None,
                    help="use this Peclet for every run, if the files do not "
                         "carry it")
    ap.add_argument("--da", type=float, default=None)
    ap.add_argument("--params-json", default=None,
                    help='a json mapping filename -> {"pe":..,"da":..}, for when '
                         "the conditions live outside the arrays")
    a = ap.parse_args()

    if os.path.isdir(a.src):
        files = sorted(glob.glob(os.path.join(a.src, "**", "*.npz"),
                                 recursive=True))
    else:
        files = [a.src]
    if not files:
        sys.exit("no .npz files under %s" % a.src)

    ext = {}
    if a.params_json:
        ext = json.load(open(a.params_json))

    print("=" * 74)
    print("READING %d file(s) from %s" % (len(files), a.src))
    print("=" * 74)

    runs, problems = [], []
    for f in files:
        try:
            r = read_run(f, a.n_species)
        except Exception as e:                                 # noqa: BLE001
            problems.append((f, str(e)))
            continue
        base = os.path.basename(f)
        if base in ext:
            r["pe"] = ext[base].get("pe", r["pe"])
            r["da"] = ext[base].get("da", r["da"])
        if r["pe"] is None:
            r["pe"] = a.pe
        if r["da"] is None:
            r["da"] = a.da
        runs.append(r)
        if len(runs) <= 5 or a.dry_run:
            print("\n%s" % base)
            print("   arrays present : %s" % ", ".join(map(str, r["keys"])))
            print("   geometry       : %s%s"
                  % (("%s %s" % (r["geom_key"], r["geom"].shape))
                     if r["geom"] is not None else "MISSING",
                     ("   porosity %.3f" % float((r["geom"] == PORE).mean()))
                     if r["geom"] is not None else ""))
            print("   concentration  : %s"
                  % (("%s -> (T=%d, C=%d, %d, %d)"
                      % ((r["conc_key"],) + r["conc"].shape))
                     if r["conc"] is not None else "MISSING"))
            print("   velocity       : %s"
                  % (("%s %s" % (r["vel_key"], r["vel"].shape))
                     if r["vel"] is not None else
                     "missing  -> flow-proxy and dim-free will NOT be usable"))
            print("   Pe, Da         : %s, %s"
                  % (r["pe"] if r["pe"] is not None else "MISSING",
                     r["da"] if r["da"] is not None else "MISSING"))

    print("\n" + "=" * 74)
    ok = [r for r in runs
          if r["geom"] is not None and r["conc"] is not None
          and r["pe"] is not None and r["da"] is not None]
    print("USABLE: %d of %d" % (len(ok), len(files)))
    for f, e in problems:
        print("   unreadable: %s  (%s)" % (os.path.basename(f), e))
    missing = {}
    for r in runs:
        for what, val in (("geometry", r["geom"]), ("concentration", r["conc"]),
                          ("Pe", r["pe"]), ("Da", r["da"])):
            if val is None:
                missing[what] = missing.get(what, 0) + 1
    for what, n in missing.items():
        print("   %d run(s) have no %s" % (n, what))
        if what in ("Pe", "Da"):
            print("      -> pass --pe / --da, or supply --params-json")
    n_vel = sum(1 for r in ok if r["vel"] is not None)
    if ok and n_vel == 0:
        print("\n   NO FLOW FIELD in any run. The dataset will train fine, but")
        print("   --flow-proxy and --dim-free cannot be used with it, because")
        print("   they replace the geometry with the flow. Worth asking for.")
    if ok and max(r["conc"].shape[0] for r in ok) == 1:
        print("\n   Only ONE snapshot per run, so this is a steady-state dataset.")
        print("   The trunk's time input will be dropped automatically. Ask for a")
        print("   time series if the transient behaviour matters.")

    if a.dry_run:
        print("\ndry run: nothing written.")
        return 0 if ok else 1
    if not ok:
        sys.exit("nothing usable to write; see above")
    if not a.out:
        sys.exit("--out is required unless --dry-run")

    shapes = {r["geom"].shape for r in ok}
    if len(shapes) > 1:
        sys.exit("runs have different grids: %s. They must all match." % shapes)
    nx, ny = ok[0]["geom"].shape
    T = max(r["conc"].shape[0] for r in ok)
    C = max(r["conc"].shape[1] for r in ok)
    species = a.species or (["C"] if C == 1 else ["s%d" % i for i in range(C)])
    species = list(species)[:C] + ["s%d" % i for i in range(len(species), C)]
    # collect_complab_output.py's names, so every writer in this project agrees and the rate
    # figures do not silently fall back to their defaults.
    pnames = ["pe", "da_bio", "da_abio", "ks_ac_norm", "ks_a_norm", "y_norm"]

    # geometries: identical masks are shared, so the train/test split by
    # geometry stays meaningful
    keys, gindex = [], []
    for r in ok:
        b = r["geom"].tobytes()
        if b not in keys:
            keys.append(b)
        gindex.append(keys.index(b))
    G = len(keys)
    print("\n%d distinct geometries across %d runs" % (G, len(ok)))
    if G == 1:
        print("   WARNING: every run uses the SAME structure, so a split by")
        print("   geometry is impossible and any score will be optimistic.")

    shape = (nx, ny, 1)
    mat = np.zeros((G,) + shape, np.uint8)
    gdf = np.zeros((G,) + shape, np.float32)
    edt = np.zeros((G,) + shape, np.float32)
    seen = {}
    for r, gi in zip(ok, gindex):
        if gi in seen:
            continue
        seen[gi] = True
        g = r["geom"]
        mat[gi] = g[:, :, None]
        gdf[gi] = (r["gdf"] if r["gdf"] is not None else geodesic_2d(g))[:, :, None]
        edt[gi] = ndimage.distance_transform_edt(g == PORE)[:, :, None]

    S = len(ok)
    conc = np.zeros((S, T, C) + shape, np.float32)
    vel = np.zeros((S, 3) + shape, np.float32)
    par = np.zeros((S, len(pnames)), np.float32)
    tn = np.zeros((S, T), np.float32)
    # Runs with FEWER snapshots than the maximum are held at their last state
    # rather than padded with zeros. Zero padding invented snapshots that were
    # identically empty and then labelled them t = 0.1 ... 1.0, and the network
    # was trained to predict nothing there.
    short = 0
    for i, r in enumerate(ok):
        c = r["conc"]
        nt_i = c.shape[0]
        conc[i, :nt_i, :c.shape[1]] = c[..., None]
        if nt_i < T:
            short += 1
            conc[i, nt_i:, :c.shape[1]] = c[-1][None, ..., None]
        # the true normalised times of THIS run, not an assumed even grid
        tn[i] = np.linspace(0, 1, T) if nt_i == T else np.concatenate(
            [np.linspace(0, 1, nt_i), np.full(T - nt_i, 1.0)]).astype(np.float32)
        if r["vel"] is not None:
            v = r["vel"]
            if v.ndim == 3 and v.shape[0] in (2, 3):
                vel[i, :v.shape[0]] = v[..., None]
            elif v.ndim == 3 and v.shape[-1] in (2, 3):
                vel[i, :v.shape[-1]] = np.moveaxis(v, -1, 0)[..., None]
        par[i] = [r["pe"], r["da"], r["da"], 0.1, 0.1, 0.05]

    scale = np.maximum(
        conc.reshape(S * T * C, -1).max(1).reshape(S, T, C).max((0, 1)), 1e-6)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with h5py.File(a.out, "w") as h:
        h.attrs["n_samples"] = S
        h.attrs["shape"] = np.array(shape, np.int32)
        h.attrs["species"] = np.array([s.encode() for s in species])
        h.attrs["param_names"] = np.array([s.encode() for s in pnames])
        h.attrs["dimension"] = 2
        h.attrs["source"] = b"external_2d_simulations"
        gg = h.create_group("geom")
        gg.create_dataset("gid", data=np.arange(G, dtype=np.int32))
        gg.create_dataset("material", data=mat, compression="gzip")
        gg.create_dataset("gdf", data=gdf, compression="gzip")
        gg.create_dataset("edt", data=edt, compression="gzip")
        sg = h.create_group("samples")
        sg.create_dataset("geom_index", data=np.array(gindex, np.int32))
        sg.create_dataset("run_id", data=np.arange(S, dtype=np.int32))
        sg.create_dataset("params", data=par)
        sg.create_dataset("t_norm", data=tn)
        d = sg.create_dataset("conc", data=conc.astype(np.float16),
                              compression="gzip")
        d.attrs["conc_scale"] = scale.astype(np.float32)
        sg.attrs["conc_scale"] = scale.astype(np.float32)
        if n_vel:
            sg.create_dataset("velocity", data=vel.astype(np.float16),
                              compression="gzip")

    if short:
        print("\n   NOTE: %d of %d runs had fewer than %d snapshots. They are "
              "held at their final state for the remainder rather than padded "
              "with zeros, and their t_norm reflects that." % (short, S, T))
    print("\nwrote %s" % a.out)
    print("  %d runs, %d geometries, %d snapshots, %d species, grid %d x %d"
          % (S, G, T, C, nx, ny))
    print("  flow field included: %s" % ("yes" if n_vel else "NO"))
    print("\nnext:")
    print("  python ../model/train.py --data %s --out runs/external" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
