#!/usr/bin/env python3
# =============================================================================
# CHANGED FROM THE 2D VERSION
#
#   WHERE IT CAME FROM
#     github.com/hjunglab/PRT-DeepONet   branch/folder: velocity-informed
#     flow/models/PRT-DeepONet_Velocity_load.ipynb, code cells 7 and 8
#     (their predict_velocity() helper)
#
#   WHAT THEIR 2D CODE DOES
#     Loads the released weights, rebuilds the features for one bundled
#     example domain using constants written into the notebook, runs the
#     network for three Reynolds conditions, and draws a quiver plot.
#
#   WHAT WE CHANGED, AND WHY
#     1. THE SCALING TRAVELS WITH THE WEIGHTS. Their constants live in the
#        notebook; ours live in the checkpoint and are reapplied here. The
#        commonest way a surrogate produces confident nonsense is being handed
#        inputs scaled differently from the ones it trained on, and the only
#        defence is that the scaling cannot be separated from the model. A
#        bare state_dict is REFUSED for exactly this reason.
#     2. IT RUNS OVER A WHOLE DATASET. --data with --write-back stores the
#        fields as samples/velocity_pred, BESIDE samples/velocity and never
#        over it, so the predicted and the simulated field can be compared.
#        That is the second half of their two-stage pipeline, which their
#        notebook does one domain at a time.
#     3. IT REPORTS THE DIVERGENCE RESIDUAL. The field is divergence
#        PENALISED, not divergence free. Printing how far short it fell makes
#        that visible instead of assumed, which matters before upscaling.
#     4. IT REFUSES A GRID MISMATCH WITH THE REASON. The branch's fully
#        connected layer is tied to the grid it was trained on.
# =============================================================================
"""Run a trained velocity operator on a geometry, with no simulation.

Two ways in.

    --geometry rock.npz     one pore structure, one flow condition, one field out.
    --data dataset.h5       every rock in a dataset, written back into it as
                            geom/velocity_pred so the concentration model can be
                            trained on the predicted field rather than the simulated
                            one. This is the second half of their two stage pipeline.

The checkpoint carries the grid, the feature scaling, the velocity standardisation and
the name of the conditioning column, and all of them are reapplied here. That is
deliberate: the single most common way a surrogate produces confident nonsense is being
handed inputs scaled differently from the ones it trained on, and the only defence is
that the scaling travels with the weights rather than living in a script.

WHAT COMES OUT IS NOT A SIMULATION. The field is a prediction. It is divergence
penalised, not divergence free, and the residual is reported so the number is visible
rather than assumed. If you are going to integrate fluxes across an interface, read
that number first.

Run it like this.

    python predict_velocity.py --checkpoint runs/vel/best.pt \
        --geometry work/geometries/geom_0007/geom_0007.npz --pe 10 --out pred/

    python predict_velocity.py --checkpoint runs/vel/best.pt \
        --data work/demo/dataset.h5 --write-back

What you get back. With --out, pred/velocity.npz holding the components and the
geometry, plus velocity.png when matplotlib is present, plus a .vti per component when
--vti is given, which opens in ParaView beside CompLaB's own output. With --write-back,
a new geom/velocity_pred dataset inside the HDF5 file and an attribute recording which
checkpoint wrote it.

Worth knowing. The features are recomputed here from the geometry, using the buffer
width and the scaling constants the checkpoint records. Handing it a geometry from a
campaign with different padding will silently shift the MIS map, so the checkpoint's
buffer is printed at the top of every run and it is worth reading.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

try:
    import torch
except ImportError:                                                    # pragma: no cover
    sys.exit("predict_velocity.py needs torch. Run: python gui/install_requirements.py")

from velocity_model import (build_velocity_model, interior_pore_mask,
                            PressureComponentUNet, PressureComponentUNet3D,
                            load_reference_weights)
import flow_features as ff
from harmonic_pressure import harmonic_gradient


def load_checkpoint(path, device="cpu"):
    # weights_only=False on purpose: the checkpoint carries the grid, the
    # scaling and the held-out list beside the tensors, and all of it is needed.
    ck = torch.load(path, map_location=device, weights_only=False)
    if "model" not in ck or "stats" not in ck:
        raise SystemExit(
            "%s does not look like a velocity checkpoint. It must carry the weights, "
            "the grid and the scaling constants; a bare state_dict is not enough, "
            "because nothing would say how its inputs were scaled." % path)
    grid = tuple(int(g) for g in ck["grid"])
    model, trunk_in = build_velocity_model(grid)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck, trunk_in


def features_for(pore, ck, device="cpu", unet=None):
    """Rebuild the branch and trunk inputs for one geometry, the checkpoint's way."""
    st = ck["stats"]
    grid = tuple(int(g) for g in ck["grid"])
    if tuple(pore.shape) != grid:
        raise SystemExit(
            "This checkpoint was trained on a %s grid and the geometry is %s.\n"
            "The branch's fully connected layer is tied to the grid, so there is no "
            "way to run it here. Retrain, or resample the geometry."
            % (grid, tuple(pore.shape)))
    ndim = len(grid)

    f = ff.all_features(pore, buf=int(ck.get("buffer", 10)))
    b1 = np.empty((1, 3) + grid, np.float32)
    b1[0, 0] = pore.astype(np.float32)
    # UPRM before MIS. Channel order is theirs and is baked into the weights.
    b1[0, 1] = (f["uprm"] - st["uprm_mu"]) / st["uprm_sd"]
    b1[0, 2] = (f["mis"] - st["mis_mu"]) / st["mis_sd"]

    e = ff.distance_transform_edt(pore).astype(np.float32)
    e2 = e * e
    dw2 = np.zeros(grid, np.float32)
    lo, hi = float(st["dw2_min"]), float(st["dw2_max"])
    # Clipped, not rescaled: a rock wider than anything in training saturates
    # at 1 rather than shifting every other voxel's value.
    dw2[pore] = np.clip((e2[pore] - lo) / max(hi - lo, 1e-12), 0, 1)

    # The pressure prior must be produced the SAME way it was at training time.
    # The checkpoint records which, so a model trained against an exact solve
    # cannot be run against a U-Net approximation without anyone noticing.
    mode = ck.get("pressure", "solve")
    if mode == "none":
        grads = np.zeros((ndim,) + grid, np.float32)
    elif mode == "unet":
        if not unet:
            raise SystemExit("this checkpoint used --pressure unet; pass --unet here too")
        net = (PressureComponentUNet(1, 2) if ndim == 2
               else PressureComponentUNet3D(1, 3)).to(device)
        load_reference_weights(net, unet, verbose=False)
        net.eval()
        with torch.no_grad():
            o = net(torch.from_numpy(pore.astype(np.float32)[None, None]).to(device))[0]
        grads = o.cpu().numpy()
        grads[:, ~pore] = 0.0
    else:
        grads = harmonic_gradient(pore)

    Lp = int(np.prod(grid))
    # Coordinates normalised to [0, 1] per axis, so the trunk sees the same
    # numbers whatever the grid size.
    coords = np.stack(np.meshgrid(*[np.arange(g, dtype=np.float32) / max(g - 1, 1)
                                    for g in grid], indexing="ij"), axis=-1).reshape(Lp, ndim)
    tr = np.empty((1, Lp, ndim + ndim + 1), np.float32)
    tr[0, :, :ndim] = coords
    for c in range(ndim):
        tr[0, :, ndim + c] = grads[c].reshape(-1)
    tr[0, :, -1] = dw2.reshape(-1)
    return b1, tr, f


@torch.no_grad()
def predict(model, ck, pore, condition, device="cpu", unet=None):
    """Physical velocity for one geometry at one flow condition."""
    st = ck["stats"]
    # The features are rebuilt from the CHECKPOINT, not from anything typed on
    # the command line: same buffer, same scaling, same pressure prior. Nothing
    # about the inputs is left to be remembered correctly by the user.
    b1, tr, _ = features_for(pore, ck, device, unet)
    # Scaled with the TRAINING statistics carried in the checkpoint, never with
    # anything measured here. A field predicted at a different scaling from the
    # one the weights were fitted at is wrong in a way that looks plausible.
    b2 = np.array([[(float(condition) - st["cond_mu"]) / st["cond_sd"]]], np.float32)
    out = model(torch.from_numpy(b1).to(device),
                torch.from_numpy(b2).to(device),
                torch.from_numpy(tr).to(device))[0].cpu().numpy()
    mu = np.asarray(st["vel_mu"], np.float32)
    sd = np.asarray(st["vel_sd"], np.float32)
    n = out.shape[-1]
    # Back to PHYSICAL units. The network works in standardised ones, and every
    # consumer downstream, the divergence residual included, expects lattice
    # velocity.
    vel = np.stack([out[..., c] * sd[c] + mu[c] for c in range(n)])
    vel[:, ~pore] = 0.0                  # a grain carries no velocity
    return vel.astype(np.float32)


def divergence_residual(vel, pore):
    """Mean absolute divergence over the interior, relative to the mean speed.

    A dimensionless number, so it means the same thing at every flow rate. Zero would
    be a divergence free field; the penalty during training pushes toward that without
    reaching it, and this says how far short it fell.
    """
    ndim = vel.shape[0]
    # float64 here even though the field is float32. This is a sum of small
    # signed differences that very nearly cancel, which is exactly the case
    # where single precision loses the answer in the rounding.
    div = np.zeros(vel.shape[1:], np.float64)
    for ax in range(ndim):
        # An axis under three voxels has no centre to difference about. Skipping
        # it is right for a thin slab; differencing it would read the array's
        # own edges as a gradient.
        if vel.shape[1 + ax] < 3:
            continue
        sl_c = [slice(None)] * ndim
        sl_p = [slice(None)] * ndim
        sl_m = [slice(None)] * ndim
        sl_c[ax] = slice(1, -1)
        sl_p[ax] = slice(2, None)
        sl_m[ax] = slice(None, -2)
        d = np.zeros(vel.shape[1:], np.float64)
        d[tuple(sl_c)] = (vel[ax][tuple(sl_p)] - vel[ax][tuple(sl_m)]) * 0.5
        div += d
    inner = interior_pore_mask(pore)
    speed = np.sqrt((vel ** 2).sum(0))
    scale = speed[pore].mean() if pore.any() else 1.0
    # NaN, not zero. A rock with no interior, or a field that is zero
    # everywhere, has no divergence residual to report, and returning 0.0 would
    # read as a perfectly divergence free field.
    if not inner.any() or scale <= 0:
        return float("nan")
    # Divided by the mean speed, so the number is dimensionless and means the
    # same thing at every flow rate.
    return float(np.abs(div[inner]).mean() / scale)


def _write_vti(path, field, name, spacing=(1.0, 1.0, 1.0)):
    """One scalar component as an appended-raw VTI, the layout CompLaB writes."""
    a = np.asarray(field, np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    nx, ny, nz = a.shape
    flat = a.transpose(2, 1, 0).ravel().astype("<f4")
    header = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" '
        'header_type="UInt32">\n'
        '  <ImageData WholeExtent="0 %d 0 %d 0 %d" Origin="0 0 0" Spacing="%g %g %g">\n'
        '    <Piece Extent="0 %d 0 %d 0 %d">\n'
        '      <PointData Scalars="%s">\n'
        '        <DataArray type="Float32" Name="%s" format="appended" offset="0"/>\n'
        '      </PointData>\n'
        '    </Piece>\n'
        '  </ImageData>\n'
        '  <AppendedData encoding="raw">\n   _'
        % (nx - 1, ny - 1, nz - 1, spacing[0], spacing[1], spacing[2],
           nx - 1, ny - 1, nz - 1, name, name))
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(np.uint32(flat.nbytes).tobytes())
        f.write(flat.tobytes())
        f.write(b"\n  </AppendedData>\n</VTKFile>\n")


def _render(path, vel, pore):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    v = vel
    p = pore
    if p.ndim == 3:
        k = p.shape[2] // 2
        v = vel[:, :, :, k]
        p = pore[:, :, k]
    mag = np.sqrt((v ** 2).sum(0))
    fig, ax = plt.subplots(1, v.shape[0] + 1,
                           figsize=(5 * (v.shape[0] + 1), 5 * p.shape[1] / max(p.shape[0], 1)))
    panels = [("|u|", mag)] + [("u%d" % c, v[c]) for c in range(v.shape[0])]
    for k, (name, field) in enumerate(panels):
        img = np.ma.masked_where(~p, field)
        cm = "viridis" if k == 0 else "RdBu_r"
        lim = float(np.abs(field[p]).max()) if p.any() else 1.0
        kw = dict(cmap=cm) if k == 0 else dict(cmap=cm, vmin=-lim, vmax=lim)
        im = ax[k].imshow(img.T, origin="lower", **kw)
        ax[k].set_title(name)
        ax[k].set_xticks([])
        ax[k].set_yticks([])
        fig.colorbar(im, ax=ax[k], fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict a velocity field with no simulation.")
    ap.add_argument("--checkpoint", required=True, help="best.pt from train_velocity.py")
    ap.add_argument("--geometry", help=".npz or .npy holding one pore structure")
    ap.add_argument("--data", help="dataset.h5; predict for every rock in it")
    ap.add_argument("--pe", type=float, help="the flow condition, in the checkpoint's units")
    ap.add_argument("--pore-code", type=int, default=None)
    ap.add_argument("--flow-axis", type=int, default=0)
    ap.add_argument("--unet", help="pressure U-Net checkpoint, if the model used one")
    ap.add_argument("--out", default="./prediction", help="folder to write into")
    ap.add_argument("--vti", action="store_true", help="also write a .vti per component")
    ap.add_argument("--write-back", action="store_true",
                    help="with --data, store the fields as geom/velocity_pred in the file")
    a = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck, trunk_in = load_checkpoint(a.checkpoint, device)
    st = ck["stats"]
    print("checkpoint %s" % a.checkpoint)
    print("  grid %s, %d components, conditioned on %r, buffer %d, pressure %s"
          % (tuple(ck["grid"]), ck["n_comp"], ck.get("condition", "?"),
             ck.get("buffer", 10), ck.get("pressure", "solve")))
    print("  trained with lambda %g, best epoch %s, held-out Huber %.6f"
          % (ck.get("lam", 0), ck.get("best_epoch", "?"), ck.get("heldout_huber", float("nan"))))

    if a.data:
        import h5py
        with h5py.File(a.data, "r+" if a.write_back else "r") as h:
            mat = np.asarray(h["geom/material"])
            pore_code = int(h.attrs["pore_code"]) if "pore_code" in h.attrs else None
            dim = int(h.attrs.get("dimension", 3))
            gidx = np.asarray(h["samples/geom_index"]).astype(int)
            pnames = [n.decode() if isinstance(n, bytes) else str(n)
                      for n in h.attrs.get("param_names", [])]
            cond_name = ck.get("condition", "pe")
            if cond_name not in pnames:
                raise SystemExit("this dataset has no %r column" % cond_name)
            cond = np.asarray(h["samples/params"])[:, pnames.index(cond_name)]
            if dim == 2:
                mat = mat[..., 0]

            n_comp = int(ck["n_comp"])
            fields = np.zeros((len(gidx), n_comp) + tuple(ck["grid"]), np.float32)
            res = []
            for i in range(len(gidx)):
                pore = ff.pore_mask_from_material(mat[gidx[i]], pore_code)
                v = predict(model, ck, pore, cond[i], device, a.unet)
                fields[i] = v
                res.append(divergence_residual(v, pore))
                if (i + 1) % 25 == 0 or i == len(gidx) - 1:
                    print("  %d/%d" % (i + 1, len(gidx)), flush=True)
            res = np.array(res, float)
            print("\n  divergence residual, relative to the mean speed:")
            print("    median %.4e   90th %.4e   worst %.4e"
                  % (np.nanmedian(res), np.nanpercentile(res, 90), np.nanmax(res)))
            if a.write_back:
                g = h["samples"]
                if "velocity_pred" in g:
                    del g["velocity_pred"]
                g.create_dataset("velocity_pred", data=fields, compression="gzip",
                                 chunks=(1, 1) + tuple(ck["grid"]))
                g["velocity_pred"].attrs["how_to_read_this"] = (
                    b"Predicted, not simulated. Written by predict_velocity.py from the "
                    b"checkpoint named in velocity_pred_source. Compare against "
                    b"samples/velocity, which is what the solver produced.")
                g.attrs["velocity_pred_source"] = os.path.abspath(a.checkpoint).encode()
                g.attrs["velocity_pred_divergence_median"] = float(np.nanmedian(res))
                print("\n  wrote samples/velocity_pred into", a.data)
        return 0

    if not a.geometry:
        ap.error("give --geometry or --data")
    if a.pe is None:
        ap.error("give --pe, the flow condition this prediction is for")

    if a.geometry.endswith(".npz"):
        with np.load(a.geometry) as d:
            key = next((k for k in ("material", "m", "mat", "geom", "arr_0")
                        if k in d.files), d.files[0])
            arr = np.asarray(d[key])
    else:
        arr = np.load(a.geometry)
    arr = ff.read_reference_orientation(arr, a.flow_axis)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    pore = ff.pore_mask_from_material(arr, a.pore_code)
    print("\ngeometry %s, shape %s, porosity %.3f"
          % (a.geometry, tuple(arr.shape), float(pore.mean())))

    vel = predict(model, ck, pore, a.pe, device, a.unet)
    speed = np.sqrt((vel ** 2).sum(0))
    print("  predicted speed: mean %.4g, max %.4g over the pore space"
          % (float(speed[pore].mean()), float(speed[pore].max())))
    print("  divergence residual %.4e (0 would be divergence free)"
          % divergence_residual(vel, pore))

    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "velocity.npz"),
                        velocity=vel, pore=pore, condition=a.pe,
                        checkpoint=os.path.abspath(a.checkpoint))
    print("  wrote", os.path.join(a.out, "velocity.npz"))
    png = _render(os.path.join(a.out, "velocity.png"), vel, pore)
    if png:
        print("  wrote", png)
    if a.vti:
        names = ["ux", "uy", "uz"][:vel.shape[0]]
        for c, nm in enumerate(names):
            p = os.path.join(a.out, "%s_pred.vti" % nm)
            _write_vti(p, vel[c], nm)
            print("  wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
