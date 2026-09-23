#!/usr/bin/env python3
"""
train.py — train the 3D PRT-DeepONet on dataset_reader.h5.

WHAT CHANGED FROM THE 2D VERSION
    The recipe is the 2D notebook's recipe: AdamW, Huber loss, mixed precision,
    cosine schedule, early stopping. Four things around it changed, and each one
    changed because of a number that was wrong and did not look wrong.

    THE SPLIT IS BY GEOMETRY, AND THERE ARE THREE OF THEM. The notebook splits
    by sample, which puts snapshots of the same rock on both sides. This splits
    whole rocks, into train, validation and test: validation chooses the
    checkpoint, and the test set is opened once at the end and is the number
    that gets reported. On one real run those two numbers were 0.3446 and
    0.3590, so reporting the first would have been reporting the score the
    checkpoint was selected on.

    THE SCALINGS ARE FITTED ON THE TRAINING ROWS ONLY. The velocity mean and
    standard deviation, and the concentration scale, used to come from the whole
    file, which lets a held-out run set the size of a training target.

    WHAT THE MODEL IS FOR IS WRITTEN DOWN. Which chemistry, and which distance
    convention. Neither changes a tensor shape, which is exactly why they have
    to be recorded: without them a model trained on one reaction loads into a
    run predicting another, every shape agrees, and the field is merely wrong.

    2D AND 3D ARE THE SAME COMMAND. A dataset whose third grid number is 1 is a
    2D problem, and the branch, the trunk and the channels follow the data. There
    is no 2D fork of this script to keep in step.

    python train.py --data ../dataset/dataset_reader.h5 --out ./runs/gdf
    python train.py --data ... --out ./runs/edt  --distance edt      # ablation
    python train.py --data ... --out ./runs/none --distance none     # ablation
    python train.py --data ... --out ./runs/vel  --with-velocity     # pipeline 2
    python train.py --data ... --out ./runs/A    --species A         # pick the field

The model has ONE output field, as in the 2D release, so a dataset holding
several chemical species needs one training run per species. --species picks
it; without the flag the first species in the file is used and the run says so.

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
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from dataset_reader import (PRT3DDataset, split_by_geometry,        # noqa: E402
                            split_three_ways,
                            resolve_switches)
from deeponet_model import PRT_DeepONet3D, count_parameters              # noqa: E402
from prt_core import conventions, reactions                              # noqa: E402


def reaction_agreement(r, ds):
    """Complaints about a --reaction that does not describe this dataset.

    The dataset is authoritative for the WIDTHS: the parameter branch is built
    from the columns the file actually holds, so naming the wrong chemistry
    cannot change the shape of anything. What it changes is what the checkpoint
    CLAIMS, and a checkpoint that claims the wrong chemistry is how a model
    trained on methane and sulfate gets used to predict acetate later, with
    every shape agreeing and the numbers meaningless.

    So this reports rather than raises: the run continues, the disagreement is
    printed, and the checkpoint records what was really used.
    """
    said = []
    if len(r.params) != len(ds.param_names):
        said.append("%s has %d dimensionless numbers (%s) and this dataset "
                    "carries %d (%s)"
                    % (r.key, len(r.params), ", ".join(r.param_names),
                       len(ds.param_names), ", ".join(ds.param_names)))
    unknown = [s for s in ds.species if s not in r.species]
    if unknown:
        said.append("this dataset holds %s, and %s does not list %s"
                    % (", ".join(ds.species), r.key, ", ".join(unknown)))
    if r.steady and getattr(ds, "with_time", False):
        said.append("%s is a steady state and this dataset has a time column"
                    % r.key)
    return said


def training_target_scale(h5path, train_idx, ds):
    """The concentration scale, fitted on the TRAINING rows only.

    AUDIT DATA-04. The scale stored in the dataset is the maximum over every
    run in the file, so normalising the target with it lets a held-out run set
    the size of a training target. That is a small leak, but it is a leak, and
    it is free to remove: one pass over the training concentrations.

    The file's own scale stays untouched and is still the physical unit a
    prediction is multiplied back by. What changes is only what the target is
    divided by while training.

    Returns one positive float per channel, never zero, so a channel that is
    flat everywhere does not turn into a division by nothing.
    """
    import h5py
    with h5py.File(h5path, "r") as h:
        conc = h["samples/conc"]
        n_ch = int(conc.shape[2])
        peak = np.zeros(n_ch, np.float64)
        for i in np.asarray(train_idx, dtype=int):
            block = np.abs(np.asarray(conc[int(i)], np.float32))   # (T, C, ...)
            peak = np.maximum(peak, block.reshape(block.shape[0], n_ch, -1)
                                          .max(axis=(0, 2)))
    stored = np.asarray(ds.conc_scale, np.float64)
    out = np.where(peak > 0, peak, np.where(stored > 0, stored, 1.0))
    return out.astype(np.float32)


def evaluate(model, loader, device):
    """Held-out RMSE of the single predicted field, in normalised units."""
    model.eval()
    se = torch.zeros((), device=device)
    n = 0
    with torch.no_grad():
        for b1, b2, tk, y in loader:
            b1, b2, tk, y = (t.to(device, non_blocking=True) for t in (b1, b2, tk, y))
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                p = model(b1, b2, tk)
            se += ((p.float() - y.float()) ** 2).sum()
            n += y.shape[0] * y.shape[1]
    return float(torch.sqrt(se / max(n, 1)).cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./runs/default")
    ap.add_argument("--distance", choices=["gdf", "edt", "none"], default="gdf")
    ap.add_argument("--species", default=None, metavar="NAME",
                    help="which chemical species this model predicts. The model "
                         "has ONE output field, as in the 2D release, so one "
                         "model is trained per species. Defaults to the first "
                         "species in the dataset.")
    # Which chemistry this run is for, and how the trunk's distance column is
    # scaled. Neither changes a tensor shape: both are recorded in the
    # checkpoint so that evaluate.py and predict.py can refuse a mismatch by
    # name instead of running happily on the wrong one.
    reactions.add_argument(ap)
    ap.add_argument("--distance-convention", default=None,
                    choices=list(conventions.DISTANCE_CONVENTIONS),
                    help="how the trunk's distance column is scaled. Defaults "
                         "to the one the chosen reaction was fitted under: "
                         "'ours' is zero at the inlet, the published two are "
                         "one there, and they are anti-correlated, so a warm "
                         "start under the wrong one trains against a column "
                         "running backwards")
    ap.add_argument("--with-velocity", action="store_true")
    # AUDIT TRAIN-04. Validation chooses the checkpoint, test is opened once
    # afterwards and is what gets reported. The old arrangement used one set
    # for both, which makes the reported number optimistic by an unknown
    # amount. --two-way-split restores the old behaviour for a dataset too
    # small to divide three ways, and says so in the output and the summary.
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="share of GEOMETRIES used to choose the checkpoint")
    ap.add_argument("--two-way-split", action="store_true",
                    help="one held-out set for both selection and reporting. "
                         "Only for a dataset with too few geometries to split "
                         "three ways; the reported score is then the score the "
                         "checkpoint was chosen on")
    # AUDIT DATA-04. The scale stored in the dataset is computed over every run
    # in the file, test rows included. 'train' refits it on the training split
    # alone, which costs one pass over the training concentrations.
    ap.add_argument("--target-scale", choices=["train", "file"], default="train",
                    help="what the target is divided by: a scale fitted on the "
                         "training split, or the one stored in the file")

    # ---------------------------------------------------------------- switches
    # All three default OFF.  With all three off this script behaves exactly as
    # it did before they existed; tools/test_three_switches.py proves it bit-exactly.
    sw = ap.add_argument_group(
        "feature switches (all default OFF -> original behaviour)")
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
                    help="SWITCH D. Give the CONCENTRATION branch the velocity field as "
                         "extra image channels, z scored, leaving the trunk exactly as "
                         "it was. 'simulated' uses samples/velocity, what the solver "
                         "produced, which needs no velocity operator and measures the "
                         "ceiling a predicted field is chasing. 'predicted' uses "
                         "samples/velocity_pred, which predict_velocity.py --write-back "
                         "puts there, and is the published two stage pipeline.")
    sw.add_argument("--geom-features", action="store_true",
                    help="with switch D, also give the branch the MIS and UPRM maps. "
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
    # the velocity in the branch. Switch D leaves the trunk alone and puts the velocity
    # in the branch. Asking for both is asking for two different branch layouts at
    # once, and whichever won would be a coin toss the log did not record.
    if args.velocity_informed != "off" and (args.flow_proxy or args.dim_free):
        raise SystemExit(
            "--velocity-informed cannot be combined with --flow-proxy or --dim-free.\n"
            "Switch A replaces the trunk's geodesic distance with a flow coordinate;\n"
            "switch D keeps the trunk unchanged and adds the velocity to the branch.\n"
            "They are two different answers to the same question, so run them "
            "separately and compare.")
    if args.geom_features and args.velocity_informed == "off":
        raise SystemExit("--geom-features only does anything with --velocity-informed")

    # ---------------------------------------------------------------- the split
    # AUDIT TRAIN-04. Three disjoint sets of whole geometries. The exact
    # geometry ids go into the checkpoint, so evaluate.py can reproduce the
    # test set instead of recomputing it from a fraction and a seed that may
    # have moved since.
    if args.two_way_split:
        tr_idx, te_idx = split_by_geometry(args.data, frac=args.test_frac,
                                           seed=args.seed)
        va_idx = te_idx
        split_ids = None
        print("split       : TWO-WAY. The reported score is the score the "
              "checkpoint was chosen on.")
    else:
        tr_idx, va_idx, te_idx, split_ids = split_three_ways(
            args.data, val_frac=args.val_frac, test_frac=args.test_frac,
            seed=args.seed)

    common = dict(n_points=args.n_points, with_velocity=args.with_velocity,
                  distance=args.distance, with_time=args.with_time,
                  flow_proxy=args.flow_proxy, dim_free=args.dim_free,
                  flow_mode=args.flow_mode, u_floor=args.u_floor,
                  keep_geometry_channel=args.keep_geometry_channel,
                  velocity_informed=args.velocity_informed,
                  geom_features=args.geom_features,
                  species=args.species)
    train_ds = PRT3DDataset(args.data, indices=tr_idx, **common)

    # ------------------------------------------- statistics, fitted on train only
    # AUDIT DATA-05. The velocity scaling is fitted here, on the training
    # split, and handed to the other splits. Fitting it separately per split
    # meant a held-out rock was scaled by numbers the network had never been
    # trained under.
    fitted_vel = None
    if any(c in ("ux", "uy", "uz") for c in train_ds.branch_ch):
        mu, sd = train_ds.vel_stats
        fitted_vel = (mu, sd)
        print("velocity    : scaled by the training split, mu %s sd %s"
              % (np.array2string(mu, precision=3),
                 np.array2string(sd, precision=3)))

    # AUDIT DATA-04. Same argument for the concentration scale.
    fitted_scale = None
    if args.target_scale == "train":
        fitted_scale = training_target_scale(args.data, tr_idx, train_ds)
        print("target scale: fitted on the training split, %s"
              % np.array2string(fitted_scale, precision=4))
    else:
        print("target scale: the one stored in the file, over every run in it")

    common_fitted = dict(common, vel_stats=fitted_vel, target_scale=fitted_scale)
    # rebuild the training set so it uses the same scale it was fitted from
    train_ds = PRT3DDataset(args.data, indices=tr_idx, **common_fitted)
    val_ds = PRT3DDataset(args.data, indices=va_idx, **common_fitted)
    test_ds = (val_ds if args.two_way_split
               else PRT3DDataset(args.data, indices=te_idx, **common_fitted))

    dl = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=sh, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), drop_last=False)
    val_loader = dl(val_ds, False)
    test_loader = dl(test_ds, False)

    # ------------------------------------------------ SWITCH B: mix in 2D data
    ds2d = None
    if args.transfer_2d:
        ds2d = PRT3DDataset(args.transfer_2d, source_tag=1, **common_fitted)
        bad = []
        if tuple(ds2d.shape) != tuple(train_ds.shape):
            bad.append("grid %s vs %s" % (tuple(ds2d.shape), tuple(train_ds.shape)))
        if ds2d.target_species != train_ds.target_species:
            # The two sources must be teaching the SAME chemical. Mixing a file
            # whose selected field is "P" into one whose selected field is "A"
            # trains one output on two different quantities and nothing
            # complains.
            bad.append("species %r vs %r"
                       % (ds2d.target_species, train_ds.target_species))
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
        trunk_in_dim=train_ds.trunk_dim,
        grid=train_ds.shape,
    ).to(device)

    # ------------------------------------ SWITCH B, two-stage: warm start on 2D
    if args.freeze_trunk and not args.init_from:
        print("note: --freeze-trunk does nothing without --init-from; there is "
              "no pretrained trunk to freeze, so it would just cripple training")
    if args.init_from:
        pre = torch.load(args.init_from, map_location="cpu", weights_only=False)
        # AUDIT, found while reading the code and reproduced directly. The
        # message below used to promise that "shape changes are skipped, not
        # silently reshaped". load_state_dict does not do that: it RAISES on a
        # size mismatch whatever strict is set to, so a 2D-mode warm start fed
        # to a 3D run died on the first convolution with no explanation. The
        # mismatched tensors are dropped here, deliberately and out loud, and
        # the ones that fit are loaded.
        own = dict(model.state_dict())
        pre_sd = pre["model"]
        clash = [(k, tuple(v.shape), tuple(own[k].shape))
                 for k, v in pre_sd.items()
                 if k in own and tuple(own[k].shape) != tuple(v.shape)]
        usable = {k: v for k, v in pre_sd.items()
                  if not (k in own and tuple(own[k].shape) != tuple(v.shape))}
        missing, unexpected = model.load_state_dict(usable, strict=False)
        print("switch B    : warm-started from %s" % args.init_from)
        if clash:
            print("              %d tensor(s) had the right name and the wrong "
                  "shape and were NOT loaded:" % len(clash))
            for _k, _a, _b in clash[:6]:
                print("                %-36s %s -> %s" % (_k, _a, _b))
            print("              A checkpoint saved in 2D mode cannot warm-start "
                  "a 3D run: its convolutions")
            print("              are 3x3 and this model wants 3x3x3. The route "
                  "that does work is")
            print("              --dim-free on an extruded transfer set, which "
                  "is 3D-shaped already.")
            print("              See 3D/SWITCHES.md.")
        if missing or unexpected:
            print("              %d missing, %d unexpected tensors"
                  % (len(missing), len(unexpected)))
        if args.freeze_trunk:
            for q in model.trunk.parameters():
                q.requires_grad_(False)
            n_frozen = sum(q.numel() for q in model.trunk.parameters())
            print("              trunk FROZEN (%.2fM parameters). The trunk holds "
                  "the dimension-independent reaction response; only the "
                  "geometry branch is retrained." % (n_frozen / 1e6))

    # ------------------------------------------------- the chemistry, by name
    reaction = reactions.resolve(args.reaction)
    convention = args.distance_convention or reaction.distance_convention
    print("device      : %s" % device)
    print("reaction    : %s  (%s)" % (reaction.key, reaction.title))
    print("distance    : convention %r, %s"
          % (convention, conventions.describe(convention)))
    for line in reaction_agreement(reaction, train_ds):
        print("              NOTE: %s" % line)
    print("species     : predicting %r  (this file holds %s)"
          % (train_ds.target_species, ", ".join(train_ds.species)))
    print("switches    : %s" % train_ds.cfg["label"])
    print("branch1 ch  : %d  (%s)   params: %.2fM"
          % (train_ds.in_channels, ", ".join(train_ds.branch_ch),
             count_parameters(model) / 1e6))
    print("parameters  : %s   (layout %s%s)"
          % (", ".join(train_ds.param_names), train_ds.param_layout,
             ", da = " + train_ds.da_column if train_ds.da_column else ""))
    print("trunk inputs: %s   (snapshots per run: %d)"
          % (", ".join(train_ds.trunk_cols), train_ds.T))
    if args.two_way_split:
        print("train/test  : %d / %d samples over disjoint geometries"
              % (len(train_ds), len(test_ds)))
    else:
        print("train/val/test: %d / %d / %d samples over three disjoint "
              "sets of geometries" % (len(train_ds), len(val_ds), len(test_ds)))
        print("              geometries  train %d, val %d, test %d"
              % (len(split_ids["train"]), len(split_ids["val"]),
                 len(split_ids["test"])))

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

        # AUDIT TRAIN-04. This is the VALIDATION set. It chooses the
        # checkpoint and nothing else. The test set is not touched until the
        # loop has finished.
        model.eval(); tot = n = 0
        with torch.no_grad():
            for b1, b2, tk, y in val_loader:
                b1, b2, tk, y = (t.to(device, non_blocking=True) for t in (b1, b2, tk, y))
                with torch.autocast("cuda", enabled=(device.type == "cuda")):
                    loss = crit(model(b1, b2, tk), y)
                tot += loss.item() * b1.size(0); n += b1.size(0)
        te = tot / max(n, 1)
        hist.append(dict(epoch=ep, train=tr, val=te, lr=sched.get_last_lr()[0],
                         sec=time.time() - t0))
        print("epoch %3d  train %.6f  val %.6f  (%.1fs)" % (ep, tr, te, time.time() - t0))

        if te < best - 1e-6:
            best, bad = te, 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "species": train_ds.target_species,
                        # WHICH CHEMISTRY, AND WHICH DISTANCE CONVENTION.
                        # Neither changes a tensor shape, which is exactly why
                        # they have to be written down: without them a model
                        # trained on methane and sulfate loads into a run
                        # predicting acetate, every shape agrees, nothing
                        # raises, and the field is merely wrong. evaluate.py
                        # and predict.py compare these and say so by name.
                        "reaction": reaction.key,
                        "reaction_species": list(reaction.species),
                        "distance_convention": convention,
                        "param_names": train_ds.param_names,
                        # which layout the dataset used, so predict.py can put
                        # the numbers you type into the right columns
                        "param_layout": train_ds.param_layout,
                        "da_column": train_ds.da_column,
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
                        "grid": list(train_ds.shape),
                        # AUDIT TRAIN-04. The exact geometries in each split,
                        # so evaluate.py scores on the same test rocks instead
                        # of recomputing a split from a fraction and a seed.
                        "split_geometries": split_ids,
                        "split_kind": "two-way" if args.two_way_split else "three-way",
                        # AUDIT DATA-05 and DATA-04. The scalings the network
                        # was actually trained under, so evaluation and
                        # prediction reuse them rather than refitting.
                        "vel_stats": (None if fitted_vel is None else
                                      [fitted_vel[0].tolist(), fitted_vel[1].tolist()]),
                        "target_scale": (None if fitted_scale is None else
                                         fitted_scale.tolist())},
                       os.path.join(args.out, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop at epoch %d" % ep); break

    # AUDIT TRAIN-04. The checkpoint has been chosen. Only now is the test set
    # opened, and it is the test number that is reported and written down.
    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"))["model"])
    rmse = evaluate(model, test_loader, device)
    val_rmse = evaluate(model, val_loader, device)
    summary = dict(best_val_loss=best, species=train_ds.target_species,
                   reaction=reaction.key, distance_convention=convention,
                   rmse=rmse, rmse_mean=rmse, val_rmse=val_rmse,
                   split_kind="two-way" if args.two_way_split else "three-way",
                   split_geometries=split_ids,
                   target_scale=(None if fitted_scale is None
                                 else fitted_scale.tolist()),
                   history=hist, args=vars(args))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    if args.two_way_split:
        print("TWO-WAY SPLIT. One set chose the checkpoint and is reported "
              "here, so this number is optimistic.")
        print("held-out RMSE (normalised units):")
        print("  %-6s %.4f    <- the 2D paper's bar was 0.04"
              % (train_ds.target_species, rmse))
    else:
        print("RMSE (normalised units), on geometries held out of both "
              "training and checkpoint selection:")
        print("  %-6s test %.4f    <- the 2D paper's bar was 0.04"
              % (train_ds.target_species, rmse))
        print("  %-6s val  %.4f    (what the checkpoint was chosen on; "
              "report the test number)" % ("", val_rmse))


if __name__ == "__main__":
    main()
