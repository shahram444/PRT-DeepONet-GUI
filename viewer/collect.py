#!/usr/bin/env python3
"""One command that turns finished work into one .h5 file.

There are two kinds of finished work in this project and each used to have its
own collector with its own flags, so the first thing anyone had to learn was
which script to run. This is the one entry point. It works out what it has been
pointed at and calls the right machinery underneath.

    python3 collect.py  <a folder>                       work it out, and say so
    python3 collect.py  <a folder>  --out my_file.h5     name the output

    python3 collect.py complab  <campaign folder>  --out dataset.h5
    python3 collect.py prt      <runs folder> --dataset dataset.h5 --out results.h5

WHAT IT RECOGNISES

  a CompLaB3D campaign     folders of .vti files written by the solver, 2D or
                           3D. Becomes the campaign dataset: the pore spaces,
                           every simulated field at every stored time, and the
                           input record of every run.

  a PRT fitting run        folders holding fitted checkpoints, from the 2D or
                           the 3D model. Becomes the results file: how accurate
                           every model is on the training, validation and test
                           sets, the predicted and simulated volumes, the
                           fitting history and the weights. This one also needs
                           the campaign dataset the models were fitted to, so
                           it can say what each run was.

2D AND 3D NEED NO FLAG. A 2D campaign is one whose grid is one voxel deep, and
every step below treats it that way, so the same command covers both and the
file that comes out is laid out identically. The only difference is nz.

Both kinds end up matching the layout in dataset_schema.py, which is checked
before this exits. Anything the campaign genuinely does not have is recorded as
absent with its reason rather than filled with zeros.
"""
from __future__ import annotations

import argparse
import glob
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# The machinery lives where it always did. This file finds it rather than
# holding a second copy of it, so there is still exactly one implementation of
# each step.
SEARCH = [
    HERE,
    os.path.join(REPO, "3D", "tools"),
    os.path.join(REPO, "tools"),
]
for p in SEARCH:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def _find(name):
    for p in SEARCH:
        f = os.path.join(p, name)
        if os.path.isfile(f):
            return f
    return None


# ===========================================================================
#  working out what a folder is
# ===========================================================================

def _any(root, pattern, depth=3):
    """Is there a file matching this pattern, within `depth` levels?"""
    for d in range(depth + 1):
        if glob.glob(os.path.join(root, *([ "*" ] * d), pattern)):
            return True
    return False


def identify(path):
    """(kind, why). kind is 'complab', 'prt' or None."""
    if not os.path.isdir(path):
        return None, "%s is not a folder" % path

    prt_marks = [("*.pt", "fitted checkpoints"),
                 ("best_*.pt", "fitted checkpoints"),
                 ("summary*.json", "a training summary")]
    complab_marks = [("subsLattice*.vti", "solver output"),
                     ("*.vti", "VTK files"),
                     ("CompLaB.xml", "a CompLaB input file")]

    for pat, why in prt_marks:
        if _any(path, pat):
            return "prt", "it holds %s (%s)" % (why, pat)
    for pat, why in complab_marks:
        if _any(path, pat):
            return "complab", "it holds %s (%s)" % (why, pat)
    return None, ("nothing in %s looks like either solver output or a fitting "
                  "run. Name the kind yourself:  collect.py complab <folder>  "
                  "or  collect.py prt <folder>" % path)


def _is_own_campaign(path):
    """Our own campaign builder leaves a campaign.json and a geometries folder.

    Anything else is somebody's folder of runs, which needs the collector that
    reads the input file of each run rather than assuming the campaign record
    is there.
    """
    return (os.path.isfile(os.path.join(path, "campaign.json"))
            or os.path.isfile(os.path.join(path, "..", "campaign.json")))


# ===========================================================================
#  running the right one
# ===========================================================================

def _run(script, argv):
    name = os.path.basename(script)
    print("   running %s" % name)
    old = sys.argv
    sys.argv = [script] + argv
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    finally:
        sys.argv = old


def collect_complab(path, out, geometries=None, extra=None):
    own = _is_own_campaign(path)
    script = _find("collect_complab_output.py" if own
                   else "collect_foreign_complab.py")
    if script is None:
        raise SystemExit(
            "cannot find the CompLaB collector. Expected it beside this file "
            "or in 3D/tools/.")
    print("CompLaB3D campaign  ->  %s" % out)
    print("   %s" % ("our own campaign builder wrote this, so the campaign "
                     "record is read directly"
                     if own else
                     "a plain folder of runs, so each run's own input file is "
                     "read"))
    outdir = os.path.dirname(os.path.abspath(out)) or "."
    if own:
        if geometries is None:
            for cand in (os.path.join(path, "geometries"),
                         os.path.join(path, "..", "geometries")):
                if os.path.isdir(cand):
                    geometries = cand
                    break
        if geometries is None:
            raise SystemExit(
                "this campaign needs its geometries folder. Pass "
                "--geometries <folder>.")
        argv = ["--campaign", path, "--geometries", geometries,
                "--out", outdir]
    else:
        argv = ["--runs", path, "--out", outdir]
    _run(script, argv + list(extra or []))


def collect_prt(path, out, dataset=None, extra=None):
    script = _find("make_results_h5.py")
    if script is None:
        raise SystemExit("cannot find make_results_h5.py beside this file.")
    if not dataset:
        raise SystemExit(
            "a PRT fitting run is scored against the campaign it was fitted "
            "to, so this needs that file too:\n"
            "   collect.py prt %s --dataset dataset.h5 --out %s"
            % (path, out))
    print("PRT fitting run  ->  %s" % out)
    print("   scored against %s" % dataset)
    _run(script, ["--data", dataset, "--runs", path, "--out", out]
         + list(extra or []))


def verify(out):
    """Check what was written against the layout, and say so either way."""
    try:
        import dataset_schema
    except ImportError:
        print("\n(dataset_schema.py is not beside this file, so the layout "
              "was not checked)")
        return
    if not os.path.isfile(out):
        print("\nno file was written at %s" % out)
        return
    print("\nchecking %s against the layout" % os.path.basename(out))
    try:
        problems = dataset_schema.validate(out)[1]
    except Exception as e:
        print("   could not read it back: %s" % e)
        return
    if problems:
        print("   %d entr%s still missing:" % (len(problems),
                                               "y" if len(problems) == 1
                                               else "ies"))
        for p in problems:
            print("      %s" % p)
        print("   dataset_schema.py upgrade %s --out fixed.h5" % out)
    else:
        print("   it matches the layout.")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="With no kind given, the folder is examined and the kind "
               "worked out from what is in it.")
    ap.add_argument("kind", nargs="?", choices=["complab", "prt"],
                    help="say what the folder is, instead of working it out")
    ap.add_argument("folder", help="the finished work to collect")
    ap.add_argument("--out", default=None, help="the .h5 to write")
    ap.add_argument("--dataset", default=None,
                    help="for a PRT run, the campaign file it was fitted to")
    ap.add_argument("--geometries", default=None,
                    help="for our own campaign, where the pore spaces are")
    ap.add_argument("--dry-run", action="store_true",
                    help="say what would happen and stop")
    a, extra = ap.parse_known_args(argv)

    # `collect.py prt somewhere` parses `prt` as the kind. `collect.py
    # somewhere` leaves kind empty and puts the folder in the right place
    # already, because folder is the only positional left.
    kind = a.kind
    folder = a.folder
    if kind is None:
        kind, why = identify(folder)
        if kind is None:
            raise SystemExit(why)
        print("%s looks like %s, because %s\n"
              % (folder,
                 "output from CompLaB3D" if kind == "complab"
                 else "a PRT fitting run", why))

    out = a.out or ("dataset.h5" if kind == "complab" else "prt_results.h5")
    if a.dry_run:
        print("would collect %s as %s into %s" % (folder, kind, out))
        return 0

    if kind == "complab":
        collect_complab(folder, out, a.geometries, extra)
    else:
        collect_prt(folder, out, a.dataset, extra)
    verify(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
