#!/usr/bin/env python3
"""
run_experiment.py — run the comparison that settles Christof's question.

A thin wrapper over 3D/model/run_ablation_sweep.py that knows where things live.

    python run_experiment.py --data ../3D/dataset/prt3d_dataset.h5
    python run_experiment.py --data 3d.h5 --quick        # plumbing check only

It trains one model per configuration, evaluates them all on the SAME held-out
geometries, and prints mean RMSE split by Peclet band:

    configuration      RMSE all        Pe<1    1<=Pe<10      Pe>=10
    baseline             ...            ...        ...         ...
    edt                  ...            ...        ...         ...
    flow-tau             ...            ...        ...         ...
    flow-both            ...            ...        ...         ...
    dim-free             ...            ...        ...         ...
    dim-free+2D          ...            ...        ...         ...
    dim-free+2Dpre       ...            ...        ...         ...

WHAT TO LOOK FOR
----------------
1.  flow-tau against baseline, per Peclet band.  The prediction is that the flow
    field wins at Pe >= 10 and LOSES at Pe < 1.  At low Peclet solute arrives by
    diffusion, the velocity is near zero in the dead-end pores that matter, and
    the travel time carries no information there while the geodesic field still
    does.  If flow-tau wins everywhere, the low-Peclet worry was unfounded and we
    can drop the geometry encoder outright.  If it loses everywhere, the flow
    field is not a sufficient input and switch A is dead.

2.  flow-both against the better of the two.  It sees gdf AND tau, so it should
    be at least as good as either in EVERY band.  If it is not, the extra trunk
    column is not being used and something is wrong with the setup, not with the
    idea.

3.  dim-free+2Dpre against baseline.  This is the compute argument: pretrained on
    free 2D data, fine-tuned on 3D with the trunk frozen.  If it matches baseline
    while using a fraction of the 3D geometries, we need far fewer 16-hour runs.
"""

import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# =============================================================================
# BLOCK 1.  WHERE THE SWEEP LIVES
#
# The same trap as in build_transfer_set.py, and the same fault: this pointed at
# run_switch_sweep.py long after that file had become run_ablation_sweep.py, so
# the wrapper exited on its first check every time.
# =============================================================================
SWEEP = os.path.join(ROOT, "3D", "model", "run_ablation_sweep.py")
DEFAULT_2D = os.path.join(HERE, "train2d.h5")


def main():
    # Checked before anything else, so a moved or renamed script is reported
    # instead of surfacing later as a confusing subprocess failure.
    if not os.path.exists(SWEEP):
        sys.exit("cannot find %s\nIs this script still inside PRT-DeepONet/bridge/?"
                 % SWEEP)
    # =========================================================================
    # BLOCK 2.  FILL IN WHAT THE USER DID NOT TYPE, AND SAY SO
    #
    # Two defaults are supplied here, and each prints a line saying so. A
    # wrapper that silently adds arguments is a wrapper whose output cannot be
    # reproduced from the command that was typed.
    # =========================================================================
    argv = sys.argv[1:]
    # Only if the user did not name one. An explicit --data-2d always wins.
    if "--data-2d" not in argv and os.path.exists(DEFAULT_2D):
        argv += ["--data-2d", DEFAULT_2D]
        print("using the 2D transfer set at %s" % DEFAULT_2D)
    elif "--data-2d" not in argv:
        print("no 2D transfer set found at %s\n"
              "  the 2D rows will be skipped. Build one first with:\n"
              "      python build_transfer_set.py" % DEFAULT_2D)
    # bridge/sweep, not the sweep script's own ./sweep, so a run started from
    # here leaves its results here.
    if "--out" not in argv:
        argv += ["--out", os.path.join(HERE, "sweep")]
    cmd = [sys.executable, SWEEP] + argv
    # Printed before it runs, so a wrapper is never a black box.
    print("$ " + " ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd).returncode)   # the child's status is ours


if __name__ == "__main__":
    main()
