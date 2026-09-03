#!/usr/bin/env python3
"""
build_toy_pack_2d.py — a small, complete 2D playground: enough data to actually
train on, small enough to finish over a coffee.

WHAT IT MAKES
    toy2d_train.h5        the training dataset
    predict_me/           geometries that are NOT in the training set, so you
                          can predict on structures the network has never seen
                          -- which is the only prediction worth looking at
    predict_me/*_vel.npy  their flow fields, so the --flow-proxy and --dim-free
                          switches can be used at prediction time too
    README.txt            the four commands, in order

WHY THE HELD-OUT GEOMETRIES ARE A SEPARATE FOLDER
    Predicting on a structure the network trained on tells you nothing. The
    geometries in predict_me/ are generated from a different random seed and
    never enter the dataset, so the prediction is a real test.

    Note this is a SECOND layer of holding-out. Training already sets aside
    whole structures for its own scoring; these are held out from that too.

SIZE AND TIME, measured
    The default 16 structures x 5 parameter sets = 80 runs takes roughly
    twenty minutes on one core. Training on it for 60 epochs is a few minutes
    on a processor, no graphics card needed.

    That is longer than it used to claim, and the reason is worth knowing.
    The transport solver runs until the field stops changing, and an explicit
    scheme's step limit and its time to steady state both scale with the grid
    width squared -- so the number of steps is set by the grid, not by how
    fast the computer is. The earlier version appeared to be four minutes
    because a periodic-boundary bug filled the domain from the wrong end
    almost immediately and the convergence test believed it. Twenty honest
    minutes is worth more than four dishonest ones.

    Use --quick for a third of the runs when you only want to see the
    machinery move.

WHY THE PECLET RANGE STARTS AT 3
    Below about Pe = 1 the problem is diffusion-dominated, and a diffusive
    run costs roughly ten times an advective one for the same grid -- Pe = 2
    measured 39 s against 4 s at Pe = 50. Those runs are perfectly valid and
    build_dataset_2d.py will do them; they just do not belong in something
    meant to be generated while you wait. For the low-Peclet end, run
    build_dataset_2d.py directly with --pe-min 0.3 and leave it going.

GRID
    148 x 64 by default, which is the grid the published 2D release uses. That
    is deliberate: it means the published trained weights can be used as a warm
    start on this data without any reshaping.

    python build_toy_pack_2d.py --out ./toy2d
"""

import argparse
import os
import subprocess
import sys
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_dataset_2d import (blob_2d, keep_spanning_cluster, percolates,   # noqa
                             geodesic_2d, stokes_d2q9)

SOLID, WALL, PORE = 0, 1, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./toy2d", help="folder to create")
    ap.add_argument("--n-geom", type=int, default=16,
                    help="structures in the training set")
    ap.add_argument("--n-sets", type=int, default=5,
                    help="Pe and Da combinations per structure")
    ap.add_argument("--n-times", type=int, default=8, help="snapshots per run")
    ap.add_argument("--n-species", type=int, default=2)
    ap.add_argument("--pe-min", type=float, default=3.0,
                    help="see the note at the top on why this is not 0.3")
    ap.add_argument("--pe-max", type=float, default=50.0)
    ap.add_argument("--quick", action="store_true",
                    help="a third of the runs, for checking the machinery "
                         "rather than for training anything you believe")
    ap.add_argument("--shape", type=int, nargs=2, default=[148, 64])
    ap.add_argument("--n-predict", type=int, default=4,
                    help="extra structures, never trained on")
    ap.add_argument("--stokes-iters", type=int, default=30000)
    ap.add_argument("--adr-steps", type=int, default=None,
                    help="LEAVE UNSET. The transport step count is derived from "
                         "Peclet and Damkohler by running until the field stops "
                         "changing. Forcing a fixed value here silently "
                         "overrode that and produced near-empty fields.")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    if a.quick:
        a.n_geom = max(4, a.n_geom // 2)
        a.n_sets = max(2, a.n_sets // 2)

    # =========================================================================
    # BLOCK 1.  WHERE EVERYTHING GOES
    #
    # Two things come out of this script and they must not be mixed: a training
    # set, and a handful of structures that are NOT in it. Predicting on a
    # structure the network trained on tells you nothing, and the easiest way
    # to do that by accident is to keep both in one folder.
    # =========================================================================
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    pred_dir = os.path.join(out, "predict_me")
    os.makedirs(pred_dir, exist_ok=True)
    nx, ny = a.shape
    train_h5 = os.path.join(out, "toy2d_train.h5")

    # ---------------------------------------------------- the training set
    print("=" * 70)
    print("1/2  training set: %d structures x %d parameter sets"
          % (a.n_geom, a.n_sets))
    print("=" * 70)
    # =========================================================================
    # BLOCK 2.  THE TRAINING SET
    #
    # Not built here. It shells out to build_dataset_2d.py, the same generator
    # a real campaign uses, so this pack exercises the real path rather than a
    # simplified copy of it that could drift.
    # =========================================================================
    cmd = [sys.executable, os.path.join(HERE, "build_dataset_2d.py"),
           "--out", train_h5, "--n-geom", str(a.n_geom),
           "--n-sets", str(a.n_sets), "--n-times", str(a.n_times),
           "--n-species", str(a.n_species), "--shape", str(nx), str(ny),
           "--stokes-iters", str(a.stokes_iters), "--seed", str(a.seed),
           "--pe-min", str(a.pe_min), "--pe-max", str(a.pe_max)]
    if a.adr_steps is not None:
        cmd += ["--adr-steps", str(a.adr_steps)]
    if subprocess.run(cmd).returncode:
        sys.exit("failed to build the training set")

    # ------------------------------------------- structures never trained on
    print()
    print("=" * 70)
    print("2/2  %d structures held out of the dataset entirely, for prediction"
          % a.n_predict)
    print("=" * 70)
    # a different seed stream, so these cannot collide with the training set
    # =========================================================================
    # BLOCK 3.  THE HELD-OUT STRUCTURES
    #
    # A DIFFERENT SEED STREAM, offset far enough that it cannot overlap the
    # training generator's. Reusing the seed would produce the same rocks and
    # quietly turn the honest test into a memorisation check.
    # =========================================================================
    rng = np.random.default_rng(a.seed + 100000)
    made = 0
    names = "ABCDEFGH"
    while made < a.n_predict:
        phi = 0.58 + 0.22 * rng.random()
        g = blob_2d((nx, ny), phi, rng, sigma=3.0)
        g, _ = keep_spanning_cluster(g)
        # Rejected, not repaired. A non-percolating rock has no flow to predict,
        # and nudging the porosity until one appears biases the whole set.
        if not percolates(g):
            continue
        nm = names[made]
        # The flow is solved and stored beside each held-out rock, so a
        # prediction run needs no solver of its own.
        v = stokes_d2q9(g, nit=a.stokes_iters)
        np.savez_compressed(
            os.path.join(pred_dir, "geom_%s.npz" % nm),
            material=g[:, :, None],
            gdf=geodesic_2d(g)[:, :, None],
            edt=ndimage.distance_transform_edt(g == PORE)[:, :, None].astype(np.float32))
        np.save(os.path.join(pred_dir, "geom_%s_vel.npy" % nm),
                v[:, :, :, None].astype(np.float32))
        sp = np.sqrt((v[:2] ** 2).sum(0))
        print("  geom_%s   porosity %.3f   mean speed %.2e" % (nm, float((g == PORE).mean()),
                                                              float(sp[g == PORE].mean())))
        made += 1

    mb = os.path.getsize(train_h5) / 1e6
    readme = """TOY 2D PACK
===========

toy2d_train.h5     the training set: %d structures, %d runs, %d snapshots each,
                   %d chemicals, %d x %d grid.  %.1f MB.
predict_me/        %d structures that are NOT in the training set, plus their
                   flow fields.  Predicting on these is a real test; predicting
                   on a structure the network trained on is not.

FOUR COMMANDS, IN ORDER
-----------------------
From this folder, with <T> = the path to 3D/tools and <M> = 3D/model:

1. train         (measured: about 30 s per epoch on two cores, so roughly
                  half an hour for 60. The test error is most of the way down
                  by epoch 15 if you want to look sooner -- the checkpoint is
                  rewritten every time it improves, so runs/toy/best.pt is
                  always usable mid-run.)
   python <M>/train.py --data toy2d_train.h5 --out runs/toy --epochs 60

2. score it on structures it never saw
   python <M>/evaluate.py --checkpoint runs/toy/best.pt \\
          --data toy2d_train.h5 --out figs/toy --no-3d

3. predict on a structure that was never simulated   (under a second)
   python <M>/predict.py --checkpoint runs/toy/best.pt \\
          --geometry predict_me/geom_A.npz \\
          --pe 10 --da-bio 1 --da-abio 1 --out prediction/A

4. try the switches -- this is what 2D is FOR, it is ~50x cheaper than 3D
   python <M>/train.py --data toy2d_train.h5 --out runs/flow --epochs 60 --flow-proxy
   python <M>/train.py --data toy2d_train.h5 --out runs/both --epochs 60 \\
          --flow-proxy --flow-mode both
   python <M>/train.py --data toy2d_train.h5 --out runs/edt  --epochs 60 --distance edt

   or all of them at once, compared on the same held-out structures:
   python <M>/run_ablation_sweep.py --data toy2d_train.h5 --out sweep

WHAT IS IN IT, AND WHAT IS NOT
------------------------------
Peclet runs from 3 to 50. Below about Pe = 1 the problem becomes diffusion
dominated and a single run costs roughly ten times as much for the same grid
(measured: 39 s at Pe = 2 against 4 s at Pe = 50), which does not belong in
something meant to be generated while you wait. For the low-Peclet end, run
build_dataset_2d.py directly with --pe-min 0.3 and leave it going.

The two chemicals are the electron donor and the electron acceptor, genuinely
coupled -- the acceptor is consumed at 0.4 per donor, which is the
stoichiometric ratio times Ac0/A0. They are similar fields but not the same
field, which is the point: an earlier version made the second species a
rescaled copy of the first, so the network reported two errors that were one
error.

Every run was checked as it was built: nothing exceeds the inlet
concentration, there is no checkerboard, nothing reaches the outlet before it
has crossed the sample, and no chemical is a copy of another. samples/settled
records which runs reached a steady state; one of the eighty ran out of steps
first, which makes it a valid transient whose last snapshot is "as far as we
got" rather than a steady state.

NOTES
-----
Prediction with --flow-proxy or --dim-free needs the flow field, because those
switches replace the geometry with it:
   python <M>/predict.py --checkpoint runs/flow/best.pt \\
          --geometry predict_me/geom_A.npz \\
          --velocity predict_me/geom_A_vel.npy \\
          --pe 10 --da-bio 1 --da-abio 1 --out prediction/A_flow

The grid is %d x %d, which is the published release's grid, so the published
trained weights can warm-start on this data:
   python <T>/load_pretrained_2d_weights.py --checkpoint <2D>/parameters/Monod.pt \\
          --n-species %d --n-params 6 --save runs/warmstart.pt
   python <M>/train.py --data toy2d_train.h5 --out runs/warm \\
          --init-from runs/warmstart.pt --freeze-trunk

Everything above is also a button in the GUI.  Working in: 2D.
""" % (a.n_geom, a.n_geom * a.n_sets, a.n_times, a.n_species, nx, ny, mb,
       a.n_predict, nx, ny, a.n_species)
    with open(os.path.join(out, "README.txt"), "w") as f:
        f.write(readme)

    print()
    print("=" * 70)
    print("DONE.  %s" % out)
    print("=" * 70)
    print("  toy2d_train.h5    %.1f MB   %d runs on %d structures"
          % (mb, a.n_geom * a.n_sets, a.n_geom))
    print("  predict_me/       %d structures never trained on, with flow fields"
          % a.n_predict)
    print("  README.txt        the four commands, in order")
    print()
    print("start with:")
    print("  python ../model/train.py --data %s --out runs/toy --epochs 60"
          % os.path.join(out, "toy2d_train.h5"))


if __name__ == "__main__":
    main()
