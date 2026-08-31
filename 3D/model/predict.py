#!/usr/bin/env python3
"""
predict.py — run a trained PRT-DeepONet-3D on a NEW geometry. No simulation.

This is the thing the whole project is for: give it a pore geometry and a set of
dimensionless numbers, get back 3D concentration fields in milliseconds instead
of running CompLaB for an hour.

    python predict.py \
        --checkpoint ../runs/gdf/best.pt \
        --geometry   ../geometries/geom_0007/geom_0007.npz \
        --pe 12.5 --da-bio 3.2 --da-abio 0.8 \
        --ks-ac 0.1 --ks-a 0.15 --y 0.04 \
        --out ./pred_geom0007

INPUT
    A geometry, either as the .npz written by build_geometry_3d.py (preferred,
    because it already carries the geodesic field) or as a raw CompLaB
    geometry.dat plus --nx/--ny/--nz, in which case the geodesic field is
    computed here with scikit-fmm.
    Six dimensionless numbers, the same ones the complab_campaign varied.

PROCESS
    geometry  -> branch 1 (3D CNN)          \\
    6 numbers -> branch 2 (FNN)              >-- dot product -> concentration
    (x,y,z,t,geodesic) at every pore voxel -> trunk (FNN)   /

    The trunk is evaluated at EVERY pore voxel, in chunks, so the output is a
    dense volume rather than the 8192-point subsample used during training.

OUTPUT
    pred.npz            all 4 fields as (4, nx, ny, nz) float32, plus metadata
    <species>.vti       one VTK ImageData per species, openable in ParaView
                        directly alongside CompLaB's own .vti output
    pred_slices.png     mid-plane slices of each field
    timing printed to stdout, which is your speedup number for the paper
"""

import argparse, base64, json, os, struct, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from deeponet_model import PRT_DeepONet3D                                  # noqa: E402
from make_figures import standard_panels, render_2d, render_3d               # noqa: E402
import make_figures                                                          # noqa: E402

PORE = 2


# ------------------------------------------------------------------ geometry
def load_geometry(args):
    """Returns (material, gdf, edt). Computes the geodesic field if absent."""
    p = args.geometry
    if p.endswith(".npz"):
        z = np.load(p, allow_pickle=True)
        mat = z["material"].astype(np.uint8)
        gdf = np.nan_to_num(z["gdf"]).astype(np.float32) if "gdf" in z else None
        edt = z["edt"].astype(np.float32) if "edt" in z else None
    else:
        if not (args.nx and args.ny and args.nz):
            sys.exit("a raw .dat needs --nx --ny --nz")
        mat = np.loadtxt(p, dtype=np.uint8).reshape(args.nx, args.ny, args.nz)
        gdf = edt = None

    if gdf is None:
        print("  geodesic field not in file, computing with scikit-fmm ...")
        import skfmm
        pore = (mat == PORE)
        phi = np.ones(mat.shape, np.float64)
        phi[0][pore[0]] = 0.0
        d = skfmm.distance(np.ma.MaskedArray(phi, mask=~pore))
        gdf = np.nan_to_num(np.array(d.filled(np.nan), np.float32))
    if edt is None:
        from scipy import ndimage
        edt = ndimage.distance_transform_edt(mat == PORE).astype(np.float32)
    return mat, gdf, edt


# ----------------------------------------------------------------- VTI write
def _dwall_scale(ny, nz):
    """Transverse half-width, used to normalise the wall distance.

    With nz == 1 the old expression min(ny,nz)/2 collapsed to 0.5 and was then
    clamped to 1, so a 2D dataset carried dwall in RAW VOXELS (0..~32) while a
    3D one carried it in 0..1. That defeats the whole point of switch C: a trunk
    transferred between 2D and 3D would see the same settings_and_units feature on scales
    differing by about thirty.
    """
    dims = [d for d in (ny, nz) if d > 1] or [max(ny, nz)]
    return max(min(dims) / 2.0, 1.0)


def write_time_series(series, ts, mat, species, raw, pnames, args):
    """One row of mid-plane slices per quantity, one column per time."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rates = [make_figures.reaction_rates(series[j], species, raw, pnames) for j in range(len(ts))]
    rows = [("BIOTIC  R_bio", [r[0] for r in rates], "YlGn"),
            ("ABIOTIC  R_abio", [r[1] for r in rates], "OrRd")]
    rows += [(sp, [series[j][i] for j in range(len(ts))], "viridis")
             for i, sp in enumerate(species)]
    rows = [r for r in rows if all(a is not None for a in r[1])]

    z = mat.shape[2] // 2
    solid = (mat[:, :, z] != PORE).T
    fig, ax = plt.subplots(len(rows), len(ts),
                           figsize=(2.9 * len(ts), 2.9 * len(rows)), squeeze=False)
    for i, (nm, vols, cm) in enumerate(rows):
        lim = make_figures.shared_limits(*vols)       # one scale across time, so the row reads as evolution
        for j in range(len(ts)):
            img = np.where(solid, np.nan, vols[j][:, :, z].T)
            kw = dict(origin="lower", cmap=cm, interpolation="nearest")
            if lim: kw.update(vmin=lim[0], vmax=lim[1])
            im = ax[i, j].imshow(img, **kw)
            ax[i, j].imshow(np.where(solid, 1.0, np.nan), origin="lower", cmap="Greys",
                            vmin=0, vmax=1.6, interpolation="nearest")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0: ax[i, j].set_title("t = %.2f" % ts[j], fontsize=11)
            if j == 0: ax[i, j].set_ylabel(nm, fontsize=10)
        fig.colorbar(im, ax=ax[i, :].tolist(), fraction=.02, pad=.01)
    fig.suptitle("time evolution   Pe=%.3g  Da_bio=%.3g  Da_abio=%.3g   (mid-plane slice, "
                 "one colour scale per row)" % (args.pe, args.da_bio, args.da_abio), fontsize=12)
    fig.savefig(os.path.join(args.out, "time_series.png"), dpi=115, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(os.path.join(args.out, "time_series.npz"),
                        concentration=series, t_norm=ts,
                        species=np.array(species, dtype="S8"), material=mat)
    print("  time     : %d snapshots written to time_series.png / .npz" % len(ts))


def write_vti(path, array, name):
    """VTK ImageData, inline base64, so it opens in ParaView next to CompLaB's."""
    nx, ny, nz = array.shape
    flat = array.transpose(2, 1, 0).ravel().astype(np.float64)
    raw = flat.tobytes()
    b64 = base64.b64encode(struct.pack("<I", len(raw)) + raw).decode()
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n'
                '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" '
                'header_type="UInt32">\n'
                '<ImageData WholeExtent="0 %d 0 %d 0 %d" Origin="0 0 0" Spacing="1 1 1">\n'
                '<Piece Extent="0 %d 0 %d 0 %d">\n<PointData>\n'
                % (nx - 1, ny - 1, nz - 1, nx - 1, ny - 1, nz - 1))
        f.write('<DataArray type="Float64" Name="%s" NumberOfComponents="1" '
                'format="binary">%s</DataArray>\n' % (name, b64))
        f.write("</PointData><CellData></CellData></Piece></ImageData></VTKFile>\n")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--geometry", required=True, help=".npz from build_geometry_3d.py, or a raw geometry.dat")
    ap.add_argument("--nx", type=int); ap.add_argument("--ny", type=int); ap.add_argument("--nz", type=int)
    ap.add_argument("--pe", type=float, required=True)
    ap.add_argument("--da-bio", type=float, required=True)
    ap.add_argument("--da-abio", type=float, required=True)
    ap.add_argument("--ks-ac", type=float, default=0.1)
    ap.add_argument("--ks-a", type=float, default=0.15)
    ap.add_argument("--y", type=float, default=0.04)
    ap.add_argument("--t-norm", type=float, default=1.0, help="1.0 = final/steady state")
    ap.add_argument("--t-series", type=int, default=0, metavar="K",
                    help="time-dependent models only: also evaluate at K times "
                         "spanning t=0..1 and write time_series.png plus "
                         "time_series.npz. This is the 3D analogue of the snapshot "
                         "strip on the right of the 2D paper's figure.")
    ap.add_argument("--velocity", default=None,
                    help=".npy of shape (3,nx,ny,nz). Required for a --with-velocity, "
                         "--flow-proxy or --dim-free checkpoint. Produce it with a "
                         "Stokes solve, or with a flow surrogate such as Geo-ONet.")
    ap.add_argument("--out", default="./prediction")
    ap.add_argument("--chunk", type=int, default=65536, help="trunk points per forward pass")
    ap.add_argument("--no-vti", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ta = ck["args"]; species = ck["species"]; pnames = ck["param_names"]
    print("checkpoint : %s" % args.checkpoint)
    print("  species  : %s" % species)
    # trunk width comes from the checkpoint. Older checkpoints predate the field,
    # and those always carried t, hence the fallback.
    with_time = ck.get("with_time", True)
    from dataset_reader import dataset_kwargs_from_ckpt
    _kw, _cfg = dataset_kwargs_from_ckpt(ck)
    trunk_in_dim = ck.get("trunk_in_dim", _cfg["trunk_dim"])
    if _cfg["needs_flow"] and not args.velocity:
        sys.exit(
            "This checkpoint was trained with:  %s\n"
            "That configuration replaces the geometry with the FLOW FIELD, so a "
            "geometry file alone is not enough to predict on a new sample.\n"
            "Pass --velocity <vel.npy>, shape (3, nx, ny, nz).  Get it either from\n"
            "  a Stokes solve   :  python ../tools/demo/stokes_lbm.py geom.npz 2000 vel.npy\n"
            "  or a flow surrogate (Geo-ONet), which is the whole point of the switch:\n"
            "  somebody else owns geometry -> flow, we own flow -> concentration."
            % _cfg["label"])
    print("  switches : %s" % _cfg["label"])
    print("  branch1  : %s" % ", ".join(_cfg["branch_ch"]))
    print("  trunk    : %s" % ", ".join(_cfg["trunk_cols"]))
    if with_time:
        print("  time     : this is a TIME-DEPENDENT model; --t-norm %.3g selects the "
              "snapshot" % args.t_norm)

    t_load = time.time()
    mat, gdf, edt = load_geometry(args)
    nx, ny, nz = mat.shape
    pore_idx = np.argwhere(mat == PORE).astype(np.int64)
    print("  grid     : %dx%dx%d, %d pore voxels (porosity %.3f)"
          % (nx, ny, nz, len(pore_idx), len(pore_idx) / mat.size))
    t_prep = time.time() - t_load

    in_ch = ck.get("in_channels", _cfg["in_channels"])
    model = PRT_DeepONet3D(in_channels=in_ch, n_params=len(pnames),
                           n_species=len(species),
                           trunk_in_dim=trunk_in_dim,
                           grid=(nx, ny, nz),
                           inject_every=(ta.get("inject_every", 3)
                                         if _cfg["film"] else 0)
                           ).to(device).eval()
    # SAY WHAT IS WRONG, IN WORDS, BEFORE TORCH SAYS IT IN TENSOR SHAPES.
    #
    # The geometry encoder's depth and its fully-connected width both follow
    # from the grid, so a checkpoint trained at one size cannot read a
    # geometry of another. Left to itself, torch reports this as "Missing
    # key(s) branch1.features.12.weight" and "size mismatch ... [128, 128] vs
    # [128, 256]", which is true and tells the reader nothing they can act on.
    trained_grid = ck.get("grid")
    if trained_grid is not None and tuple(int(v) for v in trained_grid) != (nx, ny, nz):
        sys.exit(
            "this checkpoint was trained on a %s grid, but the geometry you "
            "gave is %s.\n"
            "The geometry encoder is sized by the grid, so the two cannot be "
            "mixed. Either predict on a geometry of the trained size, or "
            "train a model on the size you want to predict at."
            % ("x".join(str(int(v)) for v in trained_grid),
               "%dx%dx%d" % (nx, ny, nz)))
    model.load_state_dict(ck["model"])

    # ---- flow-space fields, if the switches asked for them ------------------
    vel = tau = None
    if args.velocity:
        vel = np.load(args.velocity).astype(np.float32)
        if vel.shape != (3, nx, ny, nz):
            sys.exit("--velocity has shape %s but the geometry is %s"
                     % (vel.shape, (3, nx, ny, nz)))
        leak = float((np.abs(vel)[:, mat != PORE] > 1e-12).mean())
        if leak > 0.02:
            sys.exit("%.1f%% of solid voxels have non-zero velocity: this velocity "
                     "field does not belong to this geometry" % (100 * leak))
    if _cfg["needs_flow"]:
        from flow_coordinates import travel_time, normalize_velocity, squash, stats
        t_tau = time.time()
        tau_raw, stag = travel_time(vel, mat, u_floor=ta.get("u_floor", 0.01))
        tau = squash(np.nan_to_num(tau_raw, nan=0.0))
        st = stats(tau_raw, mat)
        print("  travel time: computed in %.1fs   median %.2f  p99 %.2f  "
              "stagnant %.1f%% of pore voxels"
              % (time.time() - t_tau, st.get("median", 0), st.get("p99", 0),
                 100 * float(stag[mat == PORE].mean())))
        if float(stag[mat == PORE].mean()) > 0.25:
            print("  WARNING: more than a quarter of the pore space is stagnant, so "
                  "the flow field carries little information there. This is the "
                  "known low-Peclet weakness of --flow-proxy; a --flow-mode both "
                  "model, which also sees the geodesic field, would be safer here.")

    # ---- branch inputs -----------------------------------------------------
    b1 = []
    for name in _cfg["branch_ch"]:
        if name == "material":
            b1.append((mat == PORE).astype(np.float32)[None])
        elif name == "ux":
            if vel is None:
                sys.exit("this checkpoint needs a velocity field: pass --velocity")
            v3 = normalize_velocity(vel, mat) if _cfg["needs_flow"] else vel
            # A 2D checkpoint has only ux, uy -- its first convolution was built
            # for two channels. Appending all three made every 2D flow-proxy
            # model unusable for prediction.
            nvel = sum(1 for c in _cfg["branch_ch"] if c in ("ux", "uy", "uz"))
            b1.append(v3[:nvel])
        elif name in ("uy", "uz"):
            continue
    b1 = torch.from_numpy(np.concatenate(b1, 0))[None].to(device)

    raw = np.array([args.pe, args.da_bio, args.da_abio,
                    args.ks_ac, args.ks_a, args.y], np.float32)

    # ---- is this a question the network was ever taught to answer? --------
    #
    # The parameters go into the branch unnormalised, so a value outside the
    # trained range is not clipped or rescaled -- it is simply a place the
    # network has never been, and the answer it gives there is an
    # extrapolation with no error bar. Worse, the flags this script does NOT
    # expose fall back on defaults that were never checked against the data:
    # Ks_A defaulted to 0.15 and Y to 0.04 against a dataset built with 0.10
    # and 0.05.
    ranges = ck.get("param_ranges")
    if ranges:
        for i, nm in enumerate(pnames):
            nm = nm.decode() if isinstance(nm, bytes) else str(nm)
            if nm not in ranges:
                continue
            lo, med, hi = ranges[nm]
            if hi - lo < 1e-12:
                # constant during training: the network cannot respond to it,
                # and any other value is off the map for no benefit
                if abs(float(raw[i]) - med) > 1e-9:
                    print("  note: %s was fixed at %g in training, so the "
                          "network cannot respond to it. Using %g instead of "
                          "the %g requested." % (nm, med, med, raw[i]))
                raw[i] = med
            elif not (lo <= raw[i] <= hi):
                print("  WARNING: %s = %g is outside the trained range "
                      "%g to %g. This is an extrapolation, not a prediction."
                      % (nm, raw[i], lo, hi))
    else:
        print("  note: this checkpoint predates the recording of parameter "
              "ranges, so nothing can be checked against what it was trained "
              "on. Retrain to get that check.")

    b2 = raw.copy()
    b2[:3] = np.log10(np.maximum(b2[:3], 1e-12))     # same transform as training
    b2 = torch.from_numpy(b2)[None].to(device)

    # ---- trunk over every pore voxel, in chunks ----------------------------
    xi, yi, zi = pore_idx[:, 0], pore_idx[:, 1], pore_idx[:, 2]
    cols = []
    for name in _cfg["trunk_cols"]:            # same order as dataset_reader.py
        if name == "x":
            cols.append(xi / (nx - 1))
        elif name == "y":
            cols.append(yi / (ny - 1))
        elif name == "z":
            cols.append(zi / max(nz - 1, 1))
        elif name == "t":
            cols.append(np.full(len(pore_idx), args.t_norm, np.float32))
        elif name == "gdf":
            cols.append(gdf[xi, yi, zi] / max(nx - 1, 1))
        elif name == "edt":
            cols.append(edt[xi, yi, zi] / max(nx - 1, 1))
        elif name == "dwall":
            cols.append(edt[xi, yi, zi] / _dwall_scale(ny, nz))
        elif name == "tau":
            cols.append(tau[xi, yi, zi])
        elif name == "speed":
            vn = normalize_velocity(vel, mat)
            cols.append(np.sqrt((vn[:, xi, yi, zi] ** 2).sum(0)))
        else:
            sys.exit("unknown trunk column %r" % name)
    trunk = np.stack([np.asarray(c, np.float32) for c in cols], 1)

    t0 = time.time()
    out = np.empty((len(pore_idx), len(species)), np.float32)
    with torch.no_grad():
        for s in range(0, len(trunk), args.chunk):
            tk = torch.from_numpy(trunk[s:s + args.chunk])[None].to(device)
            out[s:s + args.chunk] = model(b1, b2, tk)[0].cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf = time.time() - t0

    # ---- back to dense volumes --------------------------------------------
    vol = np.full((len(species), nx, ny, nz), np.nan, np.float32)
    vol[:, xi, yi, zi] = out.T

    # ---- derived FLOW / BIOTIC / ABIOTIC fields ----------------------------
    velfield = None
    if args.velocity:
        velfield = np.load(args.velocity).astype(np.float32)
        if velfield.shape != (3,) + mat.shape:
            sys.exit("velocity shape %s does not match geometry %s"
                     % (velfield.shape, mat.shape))
        # A velocity field from a DIFFERENT geometry is silently wrong but looks
        # plausible, so check that it is zero exactly where the solid is.
        solid = (mat != PORE)
        leak = float((np.abs(velfield).sum(0)[solid] > 1e-12).mean())
        if leak > 0.02:
            sys.exit("velocity field does not match this geometry: %.1f%% of solid "
                     "voxels carry non-zero velocity. It is probably from another "
                     "geometry." % (100 * leak))
    panels, R_bio, R_abio = standard_panels(vol, species, velfield, raw, pnames)

    # ---- optional time series ---------------------------------------------
    series = None
    if args.t_series and args.t_series > 1:
        if not with_time:
            print("note: --t-series ignored, this checkpoint has no time input "
                  "(it was trained on a steady dataset)")
        else:
            # ASK AT THE TIMES THE MODEL WAS TAUGHT, NOT AT EVENLY SPACED ONES.
            #
            # The training snapshots are log-spaced, because filling is fast at
            # first and then asymptotic: on the 2D toy set the ladder is
            # 0, 0.002, 0.007, 0.018, 0.050, 0.134, 0.364, 1.0, and 62% of every
            # snapshot the network ever saw is below t = 0.1.
            #
            # np.linspace(0, 1, 8) puts seven of its eight queries above 0.14 --
            # in the region where the field has already stopped changing. The
            # resulting picture is eight near-identical frames, which reads as
            # "the model predicts nothing happens" when what actually happened
            # is that the entire transient fell between the first two frames.
            k = int(args.t_series)
            ladder = ck.get("t_norm_median")
            if ladder is not None and len(ladder) >= 2:
                ladder = np.asarray(ladder, np.float32)
                if k == len(ladder):
                    ts = ladder                      # exactly the training times
                else:
                    ts = np.interp(np.linspace(0, len(ladder) - 1, k),
                                   np.arange(len(ladder)), ladder).astype(np.float32)
                print("  times   : taken from the training ladder, so the "
                      "transient is actually sampled")
            else:
                ts = np.concatenate(
                    [[0.0], np.geomspace(1.0 / 400.0, 1.0, k - 1)]).astype(np.float32)
                print("  times   : log-spaced (this checkpoint predates the "
                      "recording of the training time ladder). Evenly spaced "
                      "times would put almost every frame after the transient.")
            series = np.empty((len(ts), len(species), nx, ny, nz), np.float32)
            series[:] = np.nan
            with torch.no_grad():
                # The time column is NOT always index 3. resolve_switches puts
                # it at 2 for a 2D Cartesian trunk (x,y,t,gdf) and at 0 for the
                # dimension-free trunk (t,dwall,tau). Writing to 3 blindly
                # overwrote the GEODESIC column in 2D -- every frame computed
                # with a corrupted geometry input and an unchanged time -- and
                # raised IndexError under --dim-free.
                t_col = _cfg["trunk_cols"].index("t")
                for j, tv in enumerate(ts):
                    tr = trunk.copy(); tr[:, t_col] = tv
                    o = np.empty((len(pore_idx), len(species)), np.float32)
                    for c in range(0, len(tr), args.chunk):
                        tk = torch.from_numpy(tr[c:c + args.chunk])[None].to(device)
                        o[c:c + args.chunk] = model(b1, b2, tk)[0].cpu().numpy()
                    series[j][:, xi, yi, zi] = o.T
            write_time_series(series, ts, mat, species, raw, pnames, args)

    save = dict(concentration=vol, material=mat, gdf=gdf,
                species=np.array(species, dtype="S8"),
                params=raw, param_names=np.array(pnames, dtype="S16"),
                t_norm=args.t_norm)
    if R_bio is not None:  save["R_bio"] = R_bio
    if R_abio is not None: save["R_abio"] = R_abio
    if velfield is not None: save["velocity"] = velfield
    np.savez_compressed(os.path.join(args.out, "pred.npz"), **save)

    if not args.no_vti:
        for i, sp in enumerate(species):
            write_vti(os.path.join(args.out, "%s_pred.vti" % sp), np.nan_to_num(vol[i]), sp)
        if R_bio is not None:
            write_vti(os.path.join(args.out, "R_bio_pred.vti"), np.nan_to_num(R_bio), "R_bio")
        if R_abio is not None:
            write_vti(os.path.join(args.out, "R_abio_pred.vti"), np.nan_to_num(R_abio), "R_abio")

    # ---- 2D and 3D figures -------------------------------------------------
    ttl = ("predicted   Pe=%.3g  Da_bio=%.3g  Da_abio=%.3g  t=%.2f"
           % (args.pe, args.da_bio, args.da_abio, args.t_norm))
    try:
        render_2d(mat, panels, os.path.join(args.out, "pred_2d.png"), ttl + "   (mid-plane slice)")
        render_3d(mat, panels[:3], os.path.join(args.out, "pred_3d.png"),
                  ttl + "   (3D, half cut away)")
        render_3d(mat, panels[3:], os.path.join(args.out, "pred_3d_species.png"),
                  ttl + "   (species, 3D, half cut away)")
        print("  figures  : pred_2d.png, pred_3d.png, pred_3d_species.png")
    except Exception as e:
        print("  (figures skipped: %s)" % e)

    print("\nper-species range (normalised units):")
    for i, sp in enumerate(species):
        v = vol[i][~np.isnan(vol[i])]
        print("  %-5s min %+.4f  mean %+.4f  max %+.4f" % (sp, v.min(), v.mean(), v.max()))
    print("\ntiming on %s" % device)
    print("  geometry prep (incl. geodesic) : %7.3f s   <- once per geometry, cacheable" % t_prep)
    print("  network inference              : %7.3f s   <- this is the number to quote" % t_inf)
    print("     %.1f microseconds per pore voxel, %d voxels" % (1e6 * t_inf / len(pore_idx), len(pore_idx)))
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
