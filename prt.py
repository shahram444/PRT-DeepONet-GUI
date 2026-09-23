#!/usr/bin/env python3
"""
prt.py -- one command line for the whole project, with no window.

WHY THIS FILE EXISTS
    Everything here already runs without the GUI: the window has never done
    anything except build a command line and start it. But the commands live in
    two directories with names that only make sense once you know the layout,
    and on a cluster, over ssh, with no display, that is exactly when you do not
    want to be remembering paths. This is one entry point that knows where
    everything is.

    It is a dispatcher and nothing else. It does not parse the arguments of the
    script it calls, it does not add defaults, and it does not reorder anything.
    Every argument after the command name is handed over untouched, so anything
    written here works verbatim in the GUI, and anything the GUI shows works
    verbatim here. That is the point: one set of behaviour, not two.

USE IT
    python prt.py                       the list of commands
    python prt.py help train            the full help for one of them
    python prt.py train --data d.h5 --out runs/a --epochs 300

    Every command takes --help:
    python prt.py dataset2d --help

THE USUAL ORDER, IN 2D AND IN 3D
    The same commands serve both. The dimension comes from the data: a dataset
    with a third grid number of 1 is 2D, and the geometry branch, the trunk and
    the channels follow it. Nothing selects it by hand.

    2D                                        3D
    --------------------------------------    --------------------------------------
    prt dataset2d --out d.h5                  prt geometry --out rocks/
                                              prt dataset3d --geom-dir rocks/ --out d.h5
    prt train --data d.h5 --out runs/a        prt train --data d.h5 --out runs/a
    prt evaluate --checkpoint runs/a/best.pt  prt evaluate --checkpoint runs/a/best.pt
             --data d.h5                               --data d.h5
    prt predict --checkpoint runs/a/best.pt   the same
             --geometry d.h5 --geom-index 3

FLOW AWARE, WHICH IS OPTIONAL
    Off unless asked for. The five steps, in this order, are what the window's
    flow pipeline panel runs:

    prt flow-features --data d.h5                       1. the pore size maps
    prt train --data d.h5 --out runs/sim \\               2. the control
              --velocity-informed simulated
    prt velocity-train --data d.h5 --out runs/vel       3. the velocity model
    prt velocity-predict --data d.h5 \\                   4. predict, write back
              --checkpoint runs/vel/best.pt --write-back
    prt train --data d.h5 --out runs/pred \\              5. the full pipeline
              --velocity-informed predicted

    Step 2 is the control and it comes first on purpose: it costs one training
    run and tells you whether step 3 is worth doing at all.

FROM COMPLAB
    prt collect --campaign <dir> --out dataset.h5          the campaign layout
    prt collect-foreign --runs <dir> --out dataset.h5      any run folders

CHECKING THE INSTALL
    prt test            the fast gate, about forty seconds
    prt test-all        everything, including the slow physics
"""

import os
import subprocess
import sys

# =============================================================================
#  WHERE EVERYTHING IS
#  Resolved from this file's own location, never from the working directory, so
#  the dispatcher works from anywhere: a home directory, a Slurm job's scratch,
#  or the repository root.
# =============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "3D", "tools")
MODEL = os.path.join(HERE, "3D", "model")
GUI = os.path.join(HERE, "gui")
BRIDGE = os.path.join(HERE, "bridge")
TWO_D = os.path.join(HERE, "2D_scripts")

# command -> (script, one line). The order is the order they are usually run in,
# because a list sorted alphabetically teaches nobody anything.
COMMANDS = [
    ("--- make the data ---", None, None),
    ("geometry", os.path.join(TOOLS, "build_geometry_3d.py"),
     "make 3D pore structures"),
    ("dataset2d", os.path.join(TOOLS, "build_dataset_2d.py"),
     "simulate a 2D training set with our own solver"),
    ("dataset3d", os.path.join(TOOLS, "build_dataset_3d.py"),
     "the same in 3D"),
    ("practice", os.path.join(TOOLS, "build_practice_dataset.py"),
     "a tiny dataset for checking the pipeline runs"),
    ("campaign", os.path.join(TOOLS, "complab_campaign.py"),
     "write the CompLaB run folders and the Slurm submission"),
    ("collect", os.path.join(TOOLS, "collect_complab_output.py"),
     "turn a CompLaB campaign into one dataset.h5"),
    ("collect-foreign", os.path.join(TOOLS, "collect_foreign_complab.py"),
     "the same for run folders that were not written by campaign"),
    ("import-2d", os.path.join(TOOLS, "import_2d_simulations.py"),
     "import somebody else's 2D simulations from .npz files"),
    ("transfer", os.path.join(TOOLS, "build_transfer_set_2d_to_3d.py"),
     "extrude 2D domains into a 3D training set, for switch B"),
    ("dataset-info", os.path.join(TOOLS, "dataset_reader.py"),
     "look inside a dataset.h5: its arrays, and what the model will see"),

    ("--- train and use a model ---", None, None),
    ("train", os.path.join(MODEL, "train.py"),
     "train the operator. 2D or 3D, decided by the data"),
    ("evaluate", os.path.join(MODEL, "evaluate.py"),
     "score a checkpoint on the geometries it never saw"),
    ("predict", os.path.join(MODEL, "predict.py"),
     "predict on a new geometry: .h5, .npz, .vti or .dat"),
    ("sweep", os.path.join(MODEL, "run_ablation_sweep.py"),
     "train every switch configuration and compare them"),
    ("warm-start", os.path.join(TOOLS, "load_pretrained_2d_weights.py"),
     "load the published 2D weights into our network"),

    ("--- the flow-aware option ---", None, None),
    ("flow-features", os.path.join(TOOLS, "add_flow_features.py"),
     "add the pore size maps to an existing dataset"),
    ("velocity-train", os.path.join(MODEL, "train_velocity.py"),
     "train the velocity model"),
    ("velocity-predict", os.path.join(MODEL, "predict_velocity.py"),
     "predict a flow field, and write it back into the dataset"),

    ("--- the published 2D model ---", None, None),
    ("predict2d", os.path.join(TWO_D, "prt2d_predict.py"),
     "predict with the published 2D weights, without Jupyter"),

    ("--- what the chemistry is ---", None, None),
    ("reactions", os.path.join(HERE, "prt_core", "reactions.py"),
     "list the chemistries, or check them with --self-test"),
    ("conventions", os.path.join(HERE, "prt_core", "conventions.py"),
     "the two ways the trunk's distance column is scaled"),
    ("geometry-info", os.path.join(HERE, "prt_core", "inputs.py"),
     "what is in a geometry file: .h5, .npz, .vti or .dat"),

    ("--- check the install ---", None, None),
    ("core", os.path.join(HERE, "prt_core", "test_core.py"),
     "check the shared core on its own"),
    ("test", os.path.join(HERE, "smoke_test.py"),
     "the fast gate: imports, shapes, switches, buttons"),
    ("test-all", os.path.join(HERE, "run_tests.py"),
     "everything, including the slow physics"),
    ("doctor", os.path.join(HERE, "check_everything.py"),
     "every self-test this project has"),
    ("install", os.path.join(GUI, "install_requirements.py"),
     "install what is missing, or --check what is there"),

    ("--- the window ---", None, None),
    ("gui", os.path.join(GUI, "prt_gui.py"),
     "open the desktop window, if this machine has a display"),
]

LOOKUP = {name: script for name, script, _ in COMMANDS if script}   # headings drop out


# =============================================================================
#  THE LISTING
#  Grouped and in the order the commands are usually run, because a list sorted
#  alphabetically teaches nobody the order of the work. A command whose script
#  is missing is shown and marked, rather than hidden, so an incomplete checkout
#  is visible at a glance.
# =============================================================================
def usage(stream=sys.stdout):
    print(__doc__.strip().splitlines()[0], file=stream)
    print(file=stream)
    print("  python prt.py <command> [arguments]", file=stream)
    print("  python prt.py <command> --help        the full help for one",
          file=stream)
    print("  python prt.py help                    this text, in full",
          file=stream)
    print(file=stream)
    for name, script, one_line in COMMANDS:
        if script is None:
            print("  %s" % name, file=stream)
            continue
        missing = "" if os.path.exists(script) else "   (script not found)"
        print("    %-18s %s%s" % (name, one_line, missing), file=stream)
    print(file=stream)
    print("  Every argument after the command is passed to the script "
          "untouched, so what", file=stream)
    print("  works here works in the window and on the cluster, unchanged.",
          file=stream)


# =============================================================================
#  DISPATCH, AND NOTHING ELSE
#  No argument of the target script is parsed here, no default is added, and
#  nothing is reordered. That is what makes a command written in this file work
#  verbatim in the window and on the cluster: there is one behaviour, not two.
# =============================================================================
def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        usage()
        return 0
    if argv[0] == "help":
        if len(argv) == 1:
            print(__doc__)
            return 0
        argv = [argv[1], "--help"]

    cmd, rest = argv[0], argv[1:]
    if cmd not in LOOKUP:
        print("no such command: %s" % cmd, file=sys.stderr)
        # A near miss is worth suggesting: most mistakes here are a remembered
        # prefix rather than an invented name.
        near = [n for n in LOOKUP if n.startswith(cmd[:3])]
        if near:
            print("did you mean: %s" % ", ".join(sorted(near)), file=sys.stderr)
        print(file=sys.stderr)
        usage(sys.stderr)
        return 2

    script = LOOKUP[cmd]
    if not os.path.exists(script):
        print("%s should be at %s and is not.\n"
              "Is this the whole repository?" % (cmd, script), file=sys.stderr)
        return 2

    # -u so the output arrives while the job is running rather than in one
    # block at the end, which is what makes a Slurm log worth watching.
    return subprocess.call([sys.executable, "-u", script] + rest,
                           cwd=os.path.dirname(script))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
