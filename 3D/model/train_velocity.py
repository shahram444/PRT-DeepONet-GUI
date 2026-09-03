#!/usr/bin/env python3
# =============================================================================
# CHANGED FROM THE 2D VERSION
#
#   WHERE IT CAME FROM
#     github.com/hjunglab/PRT-DeepONet   branch/folder: velocity-informed
#     flow/models/PRT-DeepONet_Velocity.ipynb, code cells 4, 5 and 7
#
#   WHAT THEIR 2D CODE DOES
#     Reads a fixed split.pt plus a folder of per-run npz files, computes the
#     features, standardises the velocity, and trains with a Huber loss over
#     the pore region plus lambda times the mean squared divergence.
#
#   WHAT WE CHANGED, AND WHY
#     1. IT READS OUR HDF5, NOT THEIR split.pt. The same campaign that trains
#        the concentration network trains this one, so the two cannot drift
#        apart about which rocks are held out.
#     2. IT CONDITIONS ON PECLET, NOT REYNOLDS. Their campaign swept Re; our
#        datasets record Pe and do not record Re. The operator does not care
#        which, so long as it is the same quantity at training and prediction.
#        --condition names the column and the CHECKPOINT RECORDS IT, so the
#        two can never silently differ.
#     3. THE SPLIT IS BY GEOMETRY, NEVER BY RUN. Two runs on one rock share
#        their entire pore structure, so splitting by run leaks the geometry
#        into the test set and inflates the score.
#     4. THE PRESSURE PRIOR IS A CHOICE. --pressure solve computes it exactly,
#        unet uses their network, none is the ablation. Theirs is always the
#        network.
#     5. EARLY STOPPING WATCHES THE DATA TERM ALONE, as theirs does, and the
#        reason is worth keeping: watching the sum would let a run getting
#        WORSE at velocity look like it is improving because the divergence
#        term fell, which is the opposite of what the model is for.
# =============================================================================
"""Train the velocity operator on a dataset this project built.

The released implementation reads a fixed split.pt and a folder of per run npz files.
This reads our own HDF5 layout instead, so the same campaign that trains the
concentration network trains this one, and neither can drift from the other's idea of
which rocks are held out.

WHAT IT LEARNS. Geometry to velocity field. The branch sees the pore mask with the MIS
and UPRM maps stacked on it; the parameter branch sees one scalar flow condition; the
trunk sees each voxel's position, the squared wall distance and the pressure gradient
components. The target is the velocity the simulator produced.

WHAT THE SCALAR MEANS. The released model conditions on the Reynolds number, because
their campaign swept it. Our datasets record the Peclet number and not the Reynolds
number, so --condition defaults to pe and the checkpoint records which column was used.
The operator does not care what the scalar means, only that it is the same quantity at
training and at prediction time, and the checkpoint is what stops those from differing.

THE PRESSURE GRADIENT. Their pipeline predicts it with a U-Net trained separately, so
it costs one forward pass per rock instead of a Laplace solve. Ours can do either:

    --pressure solve   solve Laplace's equation directly, sparse, once per rock and
                       cached. Exact, and on a 148 by 64 grid it is milliseconds.
                       This is the default, because a network is a strange way to
                       approximate something you can solve.
    --pressure unet    load a PressureComponentUNet checkpoint and use it instead.
                       Use this to reproduce their pipeline exactly, or when the
                       solve is the bottleneck at 3D campaign scale.
    --pressure none    zero the gradient inputs, which turns the trunk back into
                       (x.., dw2) and is the ablation that says how much the pressure
                       prior is worth.

THE DIVERGENCE PENALTY. Weight it with --lambda. Their sweep found the velocity error
essentially flat up to 10 and clearly degrading past 100, while the divergence error
fell throughout, so 10 is their choice and the default here. Raising it buys
incompressibility at a real cost in pointwise accuracy; if you are going to integrate
fluxes across an interface, that trade may be worth making, and if you are going to
feed the field to the concentration operator their own evidence says it is not.

Run it like this.

    python train_velocity.py --data work/demo/dataset.h5 --out runs/vel
    python train_velocity.py --data work/demo/dataset.h5 --out runs/vel \
        --lambda 0 --epochs 40 --quick
    python train_velocity.py --data d.h5 --out runs/vel --pressure unet \
        --unet path/to/Pressure_component_UNet.pt

What you get back. runs/vel/best.pt, holding the weights, the grid, the feature
scaling constants, the velocity standardisation, the held out geometry indices and the
name of the conditioning column. Plus history.csv and a one screen report.

Worth knowing. THE SPLIT IS BY GEOMETRY, NEVER BY RUN. Two runs on the same rock at
different conditions share their entire pore structure, so putting one in training and
one in test leaks the geometry and inflates the score. The concentration side of this
project has always split this way and so does this.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:                                                    # pragma: no cover
    sys.exit("train_velocity.py needs torch. Run: python gui/install_requirements.py")
try:
    import h5py
except ImportError:                                                    # pragma: no cover
    sys.exit("train_velocity.py needs h5py. Run: python gui/install_requirements.py")

from velocity_model import (build_velocity_model, ROIHuberLoss, divergence_penalty,
                            interior_pore_mask, PressureComponentUNet,
                            PressureComponentUNet3D, load_reference_weights)
import flow_features as ff
from harmonic_pressure import harmonic_gradient


# ------------------------------------------------------------------ reading the file
def _unscale(dset, group, name):
    """Read a stored field, undoing the scale attribute if the writer left one."""
    arr = np.asarray(dset)
    key = name + "_scale"
    if key in group.attrs:
        s = np.asarray(group.attrs[key], np.float64)
        if s.size == 1:
            arr = arr.astype(np.float32) * float(s)
        else:
            shape = [1] * arr.ndim
            shape[1] = s.size
            arr = arr.astype(np.float32) * s.reshape(shape).astype(np.float32)
    return arr.astype(np.float32)


def read_dataset(path, condition="pe", buffer=10, pressure="solve", unet_path=None,
                 device="cpu", limit=None, verbose=True):
    """Everything the trainer needs, read once and reported.

    Returns a dict. Absences are recorded rather than filled in: a dataset with no
    velocity cannot train this model and the error says so plainly rather than
    training on zeros.
    """
    out = {}
    with h5py.File(path, "r") as h:
        if "samples/velocity" not in h:
            raise SystemExit(
                "This dataset holds no velocity field, so there is nothing to learn.\n"
                "Collect it again without --no-velocity, or point at a dataset that has one.")
        shape = tuple(int(v) for v in h.attrs["shape"])
        dim = int(h.attrs.get("dimension", 3 if shape[2] > 1 else 2))
        grid = shape[:2] if dim == 2 else shape
        pnames = [n.decode() if isinstance(n, bytes) else str(n)
                  for n in h.attrs.get("param_names", [])]
        if condition not in pnames:
            raise SystemExit(
                "No parameter named %r in this dataset. It has: %s.\n"
                "Pick one with --condition." % (condition, ", ".join(pnames) or "none"))
        # The COLUMN INDEX, resolved here once and stored in the checkpoint by
        # name rather than by number. A dataset collected later can have its
        # columns in a different order, and a saved index would then quietly
        # feed the Damkohler number to a model trained on the Peclet number.
        cidx = pnames.index(condition)

        mat = np.asarray(h["geom/material"])
        gidx = np.asarray(h["samples/geom_index"]).astype(int)
        params = np.asarray(h["samples/params"], np.float64)
        vel = _unscale(h["samples/velocity"], h["samples"], "velocity")

        # MIS and UPRM may already be stored by the collector. Recompute only what is
        # absent, and say which, because recomputing 3000 rocks is not free.
        have_mis = "geom/mis" in h
        have_uprm = "geom/uprm" in h
        mis = np.asarray(h["geom/mis"], np.float32) if have_mis else None
        uprm = np.asarray(h["geom/uprm"], np.float32) if have_uprm else None
        pore_code = int(h.attrs["pore_code"]) if "pore_code" in h.attrs else None

    # A 2D campaign is stored one voxel deep so that one reader serves both.
    # Drop that axis here, once, rather than carrying a length-1 dimension into
    # every convolution and every difference stencil downstream.
    if dim == 2:
        mat = mat[..., 0]
        vel = vel[:, :2, :, :, 0] if vel.ndim == 5 else vel[:, :2]      # uz is zero
        if mis is not None:
            mis = mis[..., 0]
        if uprm is not None:
            uprm = uprm[..., 0]
    n_comp = dim

    if limit:
        keep = np.arange(min(int(limit), len(gidx)))
        gidx, params, vel = gidx[keep], params[keep], vel[keep]

    pore = np.stack([ff.pore_mask_from_material(m, pore_code) for m in mat])
    G = len(mat)
    if verbose:
        print("dataset  %s" % path)
        print("  %d runs over %d rocks, grid %s, %dD, porosity %.3f"
              % (len(gidx), G, grid, dim, float(pore.mean())))
        print("  conditioning on %r, range %.4g to %.4g"
              % (condition, params[:, cidx].min(), params[:, cidx].max()))

    # --- geometry descriptors
    t0 = time.time()
    if mis is None or uprm is None:
        if verbose:
            print("  computing %s for %d rocks"
                  % (" and ".join([n for n, v in (("MIS", mis), ("UPRM", uprm)) if v is None]), G))
        mis_l, uprm_l = [], []
        for i in range(G):
            f = ff.all_features(pore[i], buf=buffer)
            mis_l.append(f["mis"])
            uprm_l.append(f["uprm"])
        if mis is None:
            mis = np.stack(mis_l)
        if uprm is None:
            uprm = np.stack(uprm_l)
    elif verbose:
        print("  MIS and UPRM read from the file")

    dw2 = np.zeros_like(mis)
    e2_all = []
    for i in range(G):
        # Unscaled on purpose. The [0, 1] range has to be fitted over the
        # TRAINING rocks only, and which rocks those are is not known yet: the
        # split happens after this function returns.
        d, _ = ff.dw2_map(pore[i], 0.0, 1.0)          # unscaled for now
        e2_all.append(d)
    # one scaling for the whole campaign, from the training rocks only
    out["_e2"] = e2_all

    # --- pressure gradient
    if pressure == "none":
        grads = np.zeros((G, n_comp) + tuple(grid), np.float32)
        if verbose:
            print("  pressure gradient: OFF (ablation)")
    elif pressure == "unet":
        if not unet_path or not os.path.exists(unet_path):
            raise SystemExit("--pressure unet needs --unet pointing at a checkpoint")
        net = (PressureComponentUNet(1, 2) if dim == 2
               else PressureComponentUNet3D(1, 3)).to(device)
        load_reference_weights(net, unet_path, verbose=verbose)
        net.eval()
        grads = np.zeros((G, n_comp) + tuple(grid), np.float32)
        with torch.no_grad():
            for i in range(G):
                x = torch.from_numpy(pore[i].astype(np.float32)[None, None]).to(device)
                o = net(x)[0].cpu().numpy()
                for c in range(n_comp):
                    g = o[c]
                    # A network will happily predict a pressure gradient inside
                    # a grain. Zeroing the solid here keeps the U-Net route and
                    # the exact-solve route giving the trunk the same thing.
                    g[~pore[i]] = 0.0
                    grads[i, c] = g
        if verbose:
            print("  pressure gradient: U-Net, %s" % os.path.basename(unet_path))
    else:
        if verbose:
            print("  pressure gradient: solving Laplace directly for %d rocks" % G)
        grads = np.stack([harmonic_gradient(pore[i]) for i in range(G)])
    if verbose:
        print("  features took %.1f s" % (time.time() - t0))

    out.update(dict(grid=tuple(grid), dim=dim, n_comp=n_comp, pore=pore, mis=mis,
                    uprm=uprm, grads=grads, gidx=gidx, cond=params[:, cidx],
                    condition=condition, vel=vel, G=G, path=path, buffer=buffer))
    return out


# ---------------------------------------------------------------------- assembling
def build_tensors(D, train_g, test_g, verbose=True):
    """Turn the read arrays into two TensorDatasets, scaled once over training only.

    Every constant here is computed on the TRAINING rocks and then applied to both
    halves. Fitting the scaling on everything would let the test rocks influence the
    inputs, which is a small leak but a real one.
    """
    grid, n_comp = D["grid"], D["n_comp"]
    pore, mis, uprm, grads = D["pore"], D["mis"], D["uprm"], D["grads"]
    gidx, cond, vel = D["gidx"], D["cond"], D["vel"]

    tr_rows = np.where(np.isin(gidx, train_g))[0]
    te_rows = np.where(np.isin(gidx, test_g))[0]
    if len(tr_rows) == 0 or len(te_rows) == 0:
        raise SystemExit("the split left one side empty; use more rocks or a smaller --test-frac")

    mis_mu, mis_sd = ff.zscore_stats([mis[g] for g in train_g])
    up_mu, up_sd = ff.zscore_stats([uprm[g] for g in train_g])

    e2 = D["_e2"]
    tr_vals = np.concatenate([e2[g][pore[g]] for g in train_g])
    dw2_min, dw2_max = float(tr_vals.min()), float(tr_vals.max())
    if dw2_max <= dw2_min:
        dw2_max = dw2_min + 1.0

    # velocity standardisation, per component, over training pore voxels
    # Standardise each velocity component over the TRAINING pore voxels only.
    # Fitting on everything would let the held-out rocks influence the inputs,
    # which is a small leak but a real one.
    mu = np.zeros(n_comp, np.float64)
    sd = np.ones(n_comp, np.float64)
    for c in range(n_comp):
        vals = np.concatenate([vel[i, c][pore[gidx[i]]] for i in tr_rows])
        mu[c] = vals.mean()
        s = vals.std()
        sd[c] = s if s > 0 else 1.0
    ratios = (sd / sd[0]).astype(np.float32)

    cmu = float(cond[tr_rows].mean())
    csd = float(cond[tr_rows].std()) or 1.0

    Lp = int(np.prod(grid))
    coords = np.stack(np.meshgrid(*[np.arange(g, dtype=np.float32) / max(g - 1, 1)
                                    for g in grid], indexing="ij"), axis=-1)
    coords = coords.reshape(Lp, len(grid))

    def pack(rows):
        n = len(rows)
        b1 = np.empty((n, 3) + tuple(grid), np.float32)
        b2 = np.empty((n, 1), np.float32)
        tr = np.empty((n, Lp, len(grid) + n_comp + 1), np.float32)
        y = np.empty((n,) + tuple(grid) + (n_comp,), np.float32)
        roi = np.empty((n,) + tuple(grid), np.float32)
        inter = np.empty((n,) + tuple(grid), bool)
        for k, i in enumerate(rows):
            g = gidx[i]
            b1[k, 0] = pore[g].astype(np.float32)
            b1[k, 1] = (uprm[g] - up_mu) / up_sd
            b1[k, 2] = (mis[g] - mis_mu) / mis_sd
            b2[k, 0] = (cond[i] - cmu) / csd
            tr[k, :, :len(grid)] = coords
            for c in range(n_comp):
                tr[k, :, len(grid) + c] = grads[g, c].reshape(-1)
            d = np.zeros(grid, np.float32)
            d[pore[g]] = np.clip((e2[g][pore[g]] - dw2_min) / (dw2_max - dw2_min), 0, 1)
            tr[k, :, -1] = d.reshape(-1)
            for c in range(n_comp):
                y[k, ..., c] = (vel[i, c] - mu[c]) / sd[c]
            roi[k] = pore[g].astype(np.float32)
            inter[k] = interior_pore_mask(pore[g])
        return (torch.from_numpy(b1), torch.from_numpy(b2), torch.from_numpy(tr),
                torch.from_numpy(y), torch.from_numpy(roi), torch.from_numpy(inter))

    tr_t = pack(tr_rows)
    te_t = pack(te_rows)
    stats = dict(mis_mu=mis_mu, mis_sd=mis_sd, uprm_mu=up_mu, uprm_sd=up_sd,
                 dw2_min=dw2_min, dw2_max=dw2_max, vel_mu=mu.tolist(),
                 vel_sd=sd.tolist(), ratios=ratios.tolist(),
                 cond_mu=cmu, cond_sd=csd, condition=D["condition"])
    if verbose:
        print("  %d training runs over %d rocks, %d test runs over %d rocks"
              % (len(tr_rows), len(train_g), len(te_rows), len(test_g)))
        print("  velocity standardisation: mu %s  sd %s"
              % (np.round(mu, 8).tolist(), np.round(sd, 8).tolist()))
    return TensorDataset(*tr_t), TensorDataset(*te_t[:5]), stats


# ------------------------------------------------------------------------ training
def train(model, train_ds, test_ds, ratios, lam=10.0, epochs=300, lr=1e-3,
          batch=25, patience=15, device="cpu", out_dir=None, verbose=True):
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    el = DataLoader(test_ds, batch_size=max(batch, 25))
    crit = ROIHuberLoss(1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = copy.deepcopy(model.state_dict())
    best_v = float("inf")
    best_ep = 0
    stale = 0
    hist = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run_h = run_d = nb = 0.0
        for b1, b2, tr, y, roi, inter in tl:
            b1, b2, tr = b1.to(device), b2.to(device), tr.to(device)
            y, roi, inter = y.to(device), roi.to(device), inter.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(b1, b2, tr)
            lh = crit(pred, y, roi)
            ld = (divergence_penalty(pred, inter, ratios) if lam > 0
                  else torch.zeros((), device=pred.device))
            # ONE backward pass over the SUM. Two separate backward calls would
            # need retain_graph and would give the same gradient more slowly.
            (lh + lam * ld).backward()
            opt.step()
            # .detach() before float(): without it torch warns on every batch
            # about calling a scalar conversion on a tensor that needs grad.
            run_h += float(lh.detach())
            run_d += float(ld.detach())
            nb += 1
        # ---------------------------------------------------------------------
        # Early stopping watches the DATA term only. Watching the sum would let
        # a run that is getting worse at velocity look like it is improving
        # because the divergence term fell, which is the opposite of what the
        # model is for. Theirs does the same.
        # ---------------------------------------------------------------------
        model.eval()
        tot = 0.0
        n = 0
        with torch.no_grad():
            for b1, b2, tr, y, roi in el:
                tot += float(crit(model(b1.to(device), b2.to(device), tr.to(device)),
                                  y.to(device), roi.to(device)))
                n += 1
        vl = tot / max(n, 1)
        hist.append((ep, run_h / max(nb, 1), run_d / max(nb, 1), vl))
        # A margin, not a bare <. Float noise alone will beat the previous best
        # every few epochs, and patience would then never run out on a run that
        # has plainly stopped learning.
        if vl < best_v - 1e-6:
            best_v, best_ep, stale = vl, ep, 0
            # deepcopy, not a reference: state_dict() hands back live tensors
            # that the next optimiser step would overwrite in place.
            best = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if verbose and (ep % 5 == 0 or ep == 1 or stale >= patience):
            print("  ep%4d  train huber %.5f  div %.3e  held-out huber %.5f  best %.5f@%d  %.0fs"
                  % (ep, run_h / max(nb, 1), run_d / max(nb, 1), vl, best_v, best_ep,
                     time.time() - t0), flush=True)
        if stale >= patience:
            if verbose:
                print("  stopped early at epoch %d, %d without improvement" % (ep, patience))
            break
    model.load_state_dict(best)
    if out_dir:
        with open(os.path.join(out_dir, "history.csv"), "w") as f:
            f.write("epoch,train_huber,train_divergence,heldout_huber\n")
            for row in hist:
                f.write("%d,%.8f,%.8e,%.8f\n" % row)
    return best_ep, best_v, hist


@torch.no_grad()
def report(model, test_ds, stats, n_comp, device="cpu", n=8):
    """Per sample NRMSE of the velocity magnitude, in physical units.

    Normalised by the range of the true field over that sample's own pore space, which
    is what the paper reports, so the numbers are comparable with theirs.
    """
    mu = np.asarray(stats["vel_mu"], np.float32)
    sd = np.asarray(stats["vel_sd"], np.float32)
    model.eval()
    vals = []
    for i in range(min(n, len(test_ds))):
        b1, b2, tr, y, roi = test_ds[i]
        p = model(b1[None].to(device), b2[None].to(device), tr[None].to(device))[0].cpu().numpy()
        g = y.numpy()
        m = roi.numpy() > 0.5
        pm = np.sqrt(sum((p[..., c] * sd[c] + mu[c]) ** 2 for c in range(n_comp)))
        gm = np.sqrt(sum((g[..., c] * sd[c] + mu[c]) ** 2 for c in range(n_comp)))
        rng = gm[m].max() - gm[m].min()
        vals.append(float(np.sqrt(np.mean((pm[m] - gm[m]) ** 2)) / (rng + 1e-30)))
    vals = np.array(vals)
    print("\nheld-out velocity magnitude NRMSE over %d samples" % len(vals))
    for q in (25, 50, 75, 90):
        print("  %2dth percentile  %.4f" % (q, float(np.percentile(vals, q))))
    print("  worst            %.4f" % float(vals.max()))
    return vals


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the velocity operator.")
    ap.add_argument("--data", required=True, help="dataset.h5 built by this project")
    ap.add_argument("--out", required=True, help="folder for best.pt and history.csv")
    ap.add_argument("--condition", default="pe",
                    help="which params column conditions the flow. Default pe, because "
                         "our datasets record the Peclet number and not the Reynolds "
                         "number the published model uses.")
    ap.add_argument("--lambda", dest="lam", type=float, default=10.0,
                    help="divergence penalty weight (default 10, their choice)")
    ap.add_argument("--pressure", choices=("solve", "unet", "none"), default="solve",
                    help="how to get the harmonic pressure gradient (default solve)")
    ap.add_argument("--unet", help="PressureComponentUNet checkpoint, for --pressure unet")
    ap.add_argument("--buffer", type=int, default=10,
                    help="open buffer voxels at each end, for the MIS treatment")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="use only the first N runs, for a smoke test")
    ap.add_argument("--quick", action="store_true",
                    help="three epochs on a handful of runs, to prove the wiring only")
    a = ap.parse_args(argv)

    if a.quick:
        a.epochs = min(a.epochs, 3)
        a.limit = a.limit or 12
        a.patience = 99

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    print("device:", device)

    D = read_dataset(a.data, a.condition, a.buffer, a.pressure, a.unet, device, a.limit)

    rng = np.random.default_rng(a.seed)
    all_g = np.unique(D["gidx"])
    perm = rng.permutation(all_g)
    n_te = max(1, int(round(a.test_frac * len(all_g))))
    test_g, train_g = perm[:n_te], perm[n_te:]

    train_ds, test_ds, stats = build_tensors(D, train_g, test_g)
    model, trunk_in = build_velocity_model(D["grid"])
    model = model.to(device)
    print("  model: %d parameters, trunk takes %d inputs, %d output components"
          % (sum(p.numel() for p in model.parameters()), trunk_in, D["n_comp"]))
    print("  divergence weight lambda = %g" % a.lam)

    ep, v, _ = train(model, train_ds, test_ds, stats["ratios"], a.lam, a.epochs,
                     a.lr, a.batch_size, a.patience, device, a.out)
    print("\nbest epoch %d, held-out Huber %.6f" % (ep, v))
    report(model, test_ds, stats, D["n_comp"], device)

    # NOT a bare state_dict. Nothing in a state_dict says how its inputs were
    # scaled, which grid it expects, which parameter column drove it, or which
    # rocks it never saw, and every one of those is needed to run it correctly
    # later. predict_velocity.py refuses a bare state_dict for this reason.
    ck = dict(model=model.state_dict(), grid=D["grid"], dim=D["dim"],
              n_comp=D["n_comp"], stats=stats, trunk_in=trunk_in,
              lam=a.lam, pressure=a.pressure, buffer=a.buffer,
              test_geometries=test_g.tolist(), source=os.path.abspath(a.data),
              condition=a.condition, best_epoch=ep, heldout_huber=v)
    path = os.path.join(a.out, "best.pt")
    torch.save(ck, path)
    print("\nwrote", path)
    with open(os.path.join(a.out, "run.json"), "w") as f:
        json.dump({k: v for k, v in ck.items() if k != "model"}, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
