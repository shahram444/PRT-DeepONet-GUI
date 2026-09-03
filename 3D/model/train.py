#!/usr/bin/env python3
# =============================================================================
# CHANGED FROM THE 2D VERSION
#
#   WHAT CHANGED HERE, IN ONE LINE
#     Two new flags, --velocity-informed and --geom-features, plus a guard that
#     refuses to run switch A and the flow pipeline at the same time.
#
#   WHERE IT CAME FROM
#     github.com/hjunglab/PRT-DeepONet   branch/folder: velocity-informed
#     Their concentration model takes the velocity on the BRANCH and leaves the
#     trunk alone. Their own control is what makes this worth copying: the same
#     velocity fed to the trunk pointwise recovered almost none of the gain
#     (0.0297 against a 0.0304 baseline), so it is not the information that
#     helps, it is the information arriving as a FIELD a convolution can read.
#
#   THE TWO ROUTES, AND WHICH TO RUN FIRST
#     --velocity-informed simulated uses the flow your solver already stored.
#     It needs no velocity operator at all and costs one training run. That is
#     the ceiling a predicted field is chasing, so run it FIRST. If the true
#     flow field does not help, a network trained to approximate it will not
#     help either, and you will have saved yourself the second stage.
#     --velocity-informed predicted is the full two stage pipeline and needs
#     train_velocity.py and predict_velocity.py --write-back to have run.
#
#   WHY A AND D CANNOT BE COMBINED
#     Switch A REPLACES the trunk's geodesic distance with a flow coordinate
#     and puts the velocity in the branch. This leaves the trunk exactly as
#     it was and adds the velocity to the branch. Asking for both is asking for
#     two different branch layouts at once, and whichever won would be a coin
#     toss the log did not record. So this file raises instead of choosing.
#
#   WHAT DID NOT CHANGE
#     Every flag that existed before behaves as it did. With the two new flags
#     left at their defaults this is the v1.1 training script.
# =============================================================================
"""
train.py — train the 3D PRT-DeepONet on dataset_reader.h5.

    python train.py --data ../dataset/dataset_reader.h5 --out ./runs/gdf
    python train.py --data ... --out ./runs/edt  --distance edt      # ablation
    python train.py --data ... --out ./runs/none --distance none     # ablation
    python train.py --data ... --out ./runs/vel  --with-velocity     # pipeline 2

The three `--distance` runs are the ablation that carries the paper: they show
that the GEODESIC field, not just any distance field, is what buys the accuracy.

Same training recipe as the 2D notebook (AdamW, Huber, AMP, early stopping),
with one addition: the split is by GEOMETRY, never by sample. Splitting by
sample leaks pore structure between train and test and inflates the score.
"""

import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from dataset_reader import (PRT3DDataset, split_by_geometry,        # noqa: E402
                            resolve_switches)
from deeponet_model import PRT_DeepONet3D, count_parameters              # noqa: E402


def evaluate(model, loader, device, n_species):
    model.eval()
    se = torch.zeros(n_species, device=device)
    n = 0
    with torch.no_grad():
        for b1, b2, tk, y in loader:
            b1, b2, tk, y = (t.to(device, non_blocking=True) for t in (b1, b2, tk, y))
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                p = model(b1, b2, tk)
            se += ((p.float() - y.float()) ** 2).sum(dim=(0, 1))
            n += y.shape[0] * y.shape[1]
    return torch.sqrt(se / max(n, 1)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./runs/default")
    ap.add_argument("--distance", choices=["gdf", "edt", "none"], default="gdf")
    ap.add_argument("--with-velocity", action="store_true")

    # ---------------------------------------------------------------- switches
    # A, B and C are the feature switches. --velocity-informed is not one of
    # them: it is the flow pipeline, which needs three earlier stages to have
    # run against the same dataset, and the window gives it a page of its own
    # rather than a box beside these. It lives in this group because it is a
    # flag on this script like the rest, not because it works like them.
    #
    # All default OFF. With all of them off this script behaves exactly as it
    # did before they existed; tools/test_three_switches.py proves it bit-exactly.
    sw = ap.add_argument_group(
        "feature switches and the flow pipeline (all default OFF)")
    sw.add_argument("--flow-proxy", action="store_true",
                    help="SWITCH A. Use the FLOW FIELD instead of the geometry. "
                         "The trunk's geodesic column becomes the advective "
                         "travel time tau, and the branch is fed the normalised "
                         "velocity field instead of the binary pore mask. This "
                         "is Christof's 'can we just take the flow field "
                         "instead of the GDF?'")
    sw.add_argument("--flow-mode", choices=["tau", "speed", "both"], default="tau",
                    help="what switch A puts in the trunk. 'tau' (default) is "
                         "the travel time; 'speed' is |u|; 'both' gives the "
                         "trunk gdf AND tau so the network can choose, which is "
                         "the safe option at low Peclet where the flow field "
                         "carries no information about dead-end pores.")
    sw.add_argument("--keep-geometry-channel", action="store_true",
                    help="with switch A, keep the binary pore mask as a fourth "
                         "branch channel alongside the three velocity channels")
    sw.add_argument("--u-floor", type=float, default=0.01,
                    help="velocity floor for tau, as a fraction of the mean pore "
                         "speed. Keeps tau finite in stagnant zones.")

    sw.add_argument("--transfer-2d", default=None, metavar="H5",
                    help="SWITCH B. Mix 2D training data into the 3D run. Point "
                         "this at an h5 written by tools/build_transfer_set_2d_to_3d.py, which "
                         "extrudes Jung's 2D domains into 3D. An extruded 2D "
                         "domain is an EXACT 3D problem, so this is not an "
                         "approximation -- it is free training data.")
    sw.add_argument("--transfer-2d-frac", type=float, default=0.3,
                    help="expected fraction of each epoch drawn from the 2D "
                         "source. 0.3 is a reasonable default; too high and the "
                         "network learns that nothing ever happens in z.")
    sw.add_argument("--init-from", default=None, metavar="CKPT",
                    help="SWITCH B, two-stage form. Warm-start from a checkpoint "
                         "trained on the 2D source, then fine-tune here on 3D.")
    sw.add_argument("--freeze-trunk", action="store_true",
                    help="with --init-from, freeze the trunk and train only the "
                         "branches. The trunk carries the dimension-INDEPENDENT "
                         "reaction response; the branch carries the topology, "
                         "which is what actually differs between 2D and 3D.")

    sw.add_argument("--velocity-informed", choices=["off", "simulated", "predicted"],
                    default="off",
                    help="THE FLOW PIPELINE. Give the CONCENTRATION branch the velocity "
                         "field as "
                         "extra image channels, z scored, leaving the trunk exactly as "
                         "it was. 'simulated' uses samples/velocity, what the solver "
                         "produced, which needs no velocity operator and measures the "
                         "ceiling a predicted field is chasing. 'predicted' uses "
                         "samples/velocity_pred, which predict_velocity.py --write-back "
                         "puts there, and is the published two stage pipeline.")
    sw.add_argument("--geom-features", action="store_true",
                    help="with --velocity-informed, also give the branch the MIS and "
                         "UPRM maps. "
                         "Needs geom/mis and geom/uprm; add_flow_features.py writes "
                         "them into an existing dataset without recollecting it.")
    sw.add_argument("--dim-free", action="store_true",
                    help="SWITCH C. A and B together. Replaces the Cartesian "
                         "trunk (x,y,z,t,gdf) with the flow-space trunk "
                         "(t,d_wall,tau), which has the SAME NUMBER OF INPUTS "
                         "in 2D and 3D. That is what makes 2D and 3D data "
                         "interchangeable: the Cartesian trunk needs 4 columns "
                         "in 2D and 5 in 3D and cannot transfer at all. "
                         "Implies --flow-proxy.")
    ap.add_argument("--n-points", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inject-every", type=int, default=3,
                    help="re-inject the geometry feature every N trunk layers; "
                         "0 reproduces the plain 2D trunk")
    ap.add_argument("--with-time", dest="with_time", action="store_true", default=None,
                    help="force t into the trunk. By default it is decided by the "
                         "file: present for a transient dataset, dropped for a "
                         "steady one where it would be a constant.")
    ap.add_argument("--no-time", dest="with_time", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Switch A REPLACES the trunk's geometry feature with a flow coordinate and puts
    # the velocity in the branch. --velocity-informed leaves the trunk alone and puts it
    # in the branch. Asking for both is asking for two different branch layouts at
    # once, and whichever won would be a coin toss the log did not record.
    if args.velocity_informed != "off" and (args.flow_proxy or args.dim_free):
        raise SystemExit(
            "--velocity-informed cannot be combined with --flow-proxy or --dim-free.\n"
            "Switch A replaces the trunk's geodesic distance with a flow coordinate;\n"
            "--velocity-informed keeps the trunk unchanged and adds to the branch.\n"
            "They are two different answers to the same question, so run them "
            "separately and compare.")
    if args.geom_features and args.velocity_informed == "off":
        raise SystemExit("--geom-features only does anything with --velocity-informed")

    tr_idx, te_idx = split_by_geometry(args.data, frac=args.test_frac, seed=args.seed)
    common = dict(n_points=args.n_points, with_velocity=args.with_velocity,
                  distance=args.distance, with_time=args.with_time,
                  flow_proxy=args.flow_proxy, dim_free=args.dim_free,
                  flow_mode=args.flow_mode, u_floor=args.u_floor,
                  keep_geometry_channel=args.keep_geometry_channel,
                  velocity_informed=args.velocity_informed,
                  geom_features=args.geom_features)
    train_ds = PRT3DDataset(args.data, indices=tr_idx, **common)
    test_ds = PRT3DDataset(args.data, indices=te_idx, **common)

    dl = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=sh, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), drop_last=False)
    test_loader = dl(test_ds, False)

    # ------------------------------------------------ SWITCH B: mix in 2D data
    ds2d = None
    if args.transfer_2d:
        ds2d = PRT3DDataset(args.transfer_2d, source_tag=1, **common)
        bad = []
        if tuple(ds2d.shape) != tuple(train_ds.shape):
            bad.append("grid %s vs %s" % (tuple(ds2d.shape), tuple(train_ds.shape)))
        if ds2d.C != train_ds.C:
            bad.append("species %d vs %d" % (ds2d.C, train_ds.C))
        elif list(ds2d.species) != list(train_ds.species):
            # Matching COUNTS is not enough. Two four-species files whose
            # channel order differs would be mixed with channel 1 meaning "P"
            # in one source and "A" in the other, and nothing would complain.
            bad.append("species names/order %s vs %s"
                       % (list(ds2d.species), list(train_ds.species)))
        if ds2d.trunk_dim != train_ds.trunk_dim:
            bad.append("trunk %d vs %d" % (ds2d.trunk_dim, train_ds.trunk_dim))
        if ds2d.in_channels != train_ds.in_channels:
            bad.append("branch %d vs %d" % (ds2d.in_channels, train_ds.in_channels))
        if bad:
            raise SystemExit(
                "--transfer-2d file is not compatible with --data: " + "; ".join(bad)
                + "\n  Re-run tools/build_transfer_set_2d_to_3d.py with --target-shape %d %d %d"
                % tuple(train_ds.shape))
        n3, n2 = len(train_ds), len(ds2d)
        f = float(args.transfer_2d_frac)
        w = np.concatenate([np.full(n3, (1 - f) / max(n3, 1)),
                            np.full(n2, f / max(n2, 1))])
        mixed = torch.utils.data.ConcatDataset([train_ds, ds2d])
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double), num_samples=n3 + n2,
            replacement=True)
        train_loader = torch.utils.data.DataLoader(
            mixed, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.workers, pin_memory=(device.type == "cuda"))
        print("switch B    : mixing %d 2D samples with %d 3D at frac %.2f"
              % (n2, n3, f))
    else:
        train_loader = dl(train_ds, True)

    model = PRT_DeepONet3D(
        in_channels=train_ds.in_channels,
        n_params=len(train_ds.param_names),
        n_species=train_ds.C,
        trunk_in_dim=train_ds.trunk_dim,
        grid=train_ds.shape,
        inject_every=(args.inject_every if train_ds.cfg["film"] else 0),
    ).to(device)
    if not train_ds.cfg["film"] and args.inject_every:
        print("note: this configuration has no geometry column in the trunk, so "
              "FiLM re-injection is switched off")

    # ------------------------------------ SWITCH B, two-stage: warm start on 2D
    if args.freeze_trunk and not args.init_from:
        print("note: --freeze-trunk does nothing without --init-from; there is "
              "no pretrained trunk to freeze, so it would just cripple training")
    if args.init_from:
        pre = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(pre["model"], strict=False)
        print("switch B    : warm-started from %s" % args.init_from)
        if missing or unexpected:
            print("              %d missing, %d unexpected tensors (shape changes "
                  "are skipped, not silently reshaped)"
                  % (len(missing), len(unexpected)))
        if args.freeze_trunk:
            for q in model.trunk.parameters():
                q.requires_grad_(False)
            n_frozen = sum(q.numel() for q in model.trunk.parameters())
            print("              trunk FROZEN (%.2fM parameters). The trunk holds "
                  "the dimension-independent reaction response; only the "
                  "geometry branch is retrained." % (n_frozen / 1e6))

    print("device      : %s" % device)
    print("species     : %s" % train_ds.species)
    print("switches    : %s" % train_ds.cfg["label"])
    print("branch1 ch  : %d  (%s)   params: %.2fM"
          % (train_ds.in_channels, ", ".join(train_ds.branch_ch),
             count_parameters(model) / 1e6))
    print("trunk inputs: %s   (snapshots per run: %d)"
          % (", ".join(train_ds.trunk_cols), train_ds.T))
    print("train/test  : %d / %d samples over disjoint geometries"
          % (len(train_ds), len(test_ds)))

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.HuberLoss(delta=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best, bad, hist = float("inf"), 0, []
    for ep in range(args.epochs):
        model.train(); tot = n = 0; t0 = time.time()
        for b1, b2, tk, y in train_loader:
            b1, b2, tk, y = (t.to(device, non_blocking=True) for t in (b1, b2, tk, y))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                loss = crit(model(b1, b2, tk), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * b1.size(0); n += b1.size(0)
        sched.step()
        tr = tot / max(n, 1)

        model.eval(); tot = n = 0
        with torch.no_grad():
            for b1, b2, tk, y in test_loader:
                b1, b2, tk, y = (t.to(device, non_blocking=True) for t in (b1, b2, tk, y))
                with torch.autocast("cuda", enabled=(device.type == "cuda")):
                    loss = crit(model(b1, b2, tk), y)
                tot += loss.item() * b1.size(0); n += b1.size(0)
        te = tot / max(n, 1)
        hist.append(dict(epoch=ep, train=tr, test=te, lr=sched.get_last_lr()[0],
                         sec=time.time() - t0))
        print("epoch %3d  train %.6f  test %.6f  (%.1fs)" % (ep, tr, te, time.time() - t0))

        if te < best - 1e-6:
            best, bad = te, 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "species": train_ds.species,
                        "param_names": train_ds.param_names,
                        # What the parameter branch actually SAW. Without this,
                        # predict.py has to fall back on hardcoded defaults for
                        # the parameters it does not expose, and those defaults
                        # were not the values in the data -- it sent Ks_A 0.15
                        # and Y 0.04 to a network trained on 0.10 and 0.05. A
                        # column that never varied also cannot be learned from,
                        # so knowing which ones are constant is worth as much
                        # as knowing their range.
                        # The times the training snapshots actually sit at.
                        # They are LOG-spaced, so a prediction asked at evenly
                        # spaced times lands almost entirely after the
                        # transient and looks like a static field.
                        "t_norm_median": np.median(
                            train_ds.t_norm, axis=0).astype(float).tolist(),
                        "param_ranges": {
                            n: [float(train_ds.params[:, i].min()),
                                float(np.median(train_ds.params[:, i])),
                                float(train_ds.params[:, i].max())]
                            for i, n in enumerate(train_ds.param_names)},
                        # recorded explicitly so predict.py and evaluate.py never
                        # have to re-derive the trunk width and get it wrong
                        "trunk_in_dim": train_ds.trunk_dim,
                        "with_time": train_ds.with_time,
                        "n_times": train_ds.T,
                        "in_channels": train_ds.in_channels,
                        "trunk_cols": train_ds.trunk_cols,
                        "branch_ch": train_ds.branch_ch,
                        "switch_label": train_ds.cfg["label"],
                        "grid": list(train_ds.shape)},
                       os.path.join(args.out, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop at epoch %d" % ep); break

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"))["model"])
    rmse = evaluate(model, test_loader, device, train_ds.C)
    summary = dict(best_test_loss=best,
                   rmse_per_species={s: float(r) for s, r in zip(train_ds.species, rmse)},
                   rmse_mean=float(rmse.mean()), history=hist, args=vars(args))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nheld-out RMSE (normalised units):")
    for s, r in zip(train_ds.species, rmse):
        print("  %-6s %.4f" % (s, r))
    print("  %-6s %.4f    <- the 2D paper's bar was 0.04" % ("mean", rmse.mean()))


if __name__ == "__main__":
    main()
