#!/usr/bin/env python3
"""
smoke_test.py -- one command that says whether this checkout works, without
training anything.

    python smoke_test.py            the deploy gate, about a minute
    python smoke_test.py --full     the above plus the slow physics self-tests
    python smoke_test.py --list     show what would run, and skip nothing

WHY THIS EXISTS, AND WHAT IT IS NOT

Training takes hours and answers a question about accuracy.  This answers a
different question: is the code wired up correctly on this machine, with these
packages, right now.  It imports every module, builds the network, builds a
small synthetic dataset from analytic fields, writes an untrained checkpoint,
and then drives evaluate.py and predict.py end to end on them.  Random weights
travel through exactly the same load, build, inference and write path that
trained weights do, so a renamed checkpoint key, a shape that stopped matching,
a figure writer that assumed a list of species, or a missing package all fail
here in seconds.

A green line means the plumbing holds.  It does not mean the science is right;
that is what the figures and the held-out error are for.

WHAT RUNS IN THE DEFAULT PASS

  every file compiles            a syntax error anywhere in the tree
  the network                    the output growing a species axis, the bias
                                 becoming a vector, FiLM coming back, the
                                 parameter branch ignoring n_params, a 2D
                                 dataset being sent through 3D convolutions
  the dataset layer              the target regrowing a species axis, --species
                                 not reaching the reader, the two-column and
                                 six-column parameter layouts stopping to
                                 coexist, train and test sharing a geometry
  evaluate and predict           either one breaking on a single-field
                                 prediction, or regrowing a per-species index
                                 in its output file names
  your own settings              a unit mistake, and a written template that is
                                 not valid XML
  flow coordinates               travel time, wall distance, the squash
  the flow descriptors           MIS, UPRM and the wall distance
  the pressure solve             the harmonic pressure proxy and its gradient
  the three switches             every switch OFF no longer reproducing the
                                 original code bit for bit
  the velocity pipeline          the flow-aware path refusing bad input quietly
  what the buttons send          a GUI button emitting a flag its script does
                                 not accept
  the sweep boxes                a range box leaking the other box's flags

WITH --full, ALSO

  the 2D simulator, the 3D simulator, the flow solvers and the documented
  numbers.  These solve real lattice-Boltzmann problems and take minutes.

The window test needs a display, and is skipped with a note when there is none.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "3D", "tools")
MODEL = os.path.join(HERE, "3D", "model")
GUI = os.path.join(HERE, "gui")
TESTS = os.path.join(HERE, "tests")

PY = sys.executable

FAST = [
    ("the network", [os.path.join(TESTS, "test_model.py")]),
    ("the dataset layer", [os.path.join(TESTS, "test_dataset.py")]),
    ("evaluate and predict", [os.path.join(TESTS, "test_pipeline.py")]),
    ("your own settings", [os.path.join(TOOLS, "settings_and_units.py"),
                           "--self-test"]),
    ("flow coordinates", [os.path.join(TOOLS, "flow_coordinates.py"),
                          "--self-test"]),
    ("the flow descriptors", [os.path.join(TOOLS, "flow_features.py"),
                              "--self-test"]),
    ("the pressure solve", [os.path.join(TOOLS, "harmonic_pressure.py"),
                            "--self-test"]),
    ("the three switches", [os.path.join(TOOLS, "test_three_switches.py")]),
    ("the velocity pipeline", [os.path.join(MODEL, "test_flow_pipeline.py")]),
    ("what the buttons send", [os.path.join(GUI, "test_gui_commands.py")]),
    ("the sweep boxes", [os.path.join(GUI, "test_gui_sweep_modes.py")]),
]

SLOW = [
    ("the 2D simulator", [os.path.join(TOOLS, "prtlb_2d.py")]),
    ("the 3D simulator", [os.path.join(TOOLS, "prtlb_3d.py")]),
    ("the flow solvers", [os.path.join(TOOLS, "test_flow_solvers.py")]),
    ("the documented numbers",
     [os.path.join(TOOLS, "test_documented_numbers.py")]),
]

NEEDED = [("numpy", "numpy"), ("scipy", "scipy"), ("h5py", "h5py"),
          ("torch", "torch"), ("matplotlib", "matplotlib")]
OPTIONAL = [("skfmm", "scikit-fmm, used by predict.py for the geodesic field"),
            ("skimage", "scikit-image, used by the 3D surface renders"),
            ("tkinter", "tkinter, needed only by the window")]


def packages():
    """Report what is importable before anything tries to use it."""
    missing = []
    for mod, pkg in NEEDED:
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    absent = []
    for mod, why in OPTIONAL:
        try:
            __import__(mod)
        except Exception:
            absent.append(why)
    return missing, absent


def compiles():
    """Every .py in the tree parses.  Cheap, and catches a bad merge.

    Compiled in memory rather than through py_compile, so nothing is written
    next to the sources and a read-only checkout still passes.
    """
    bad = []
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            try:
                with open(p, "rb") as fh:
                    compile(fh.read(), p, "exec")
            except Exception as e:
                bad.append("%s: %s" % (os.path.relpath(p, HERE), e))
    return bad


def verdict(out):
    """The line that states the outcome, not whatever was printed last."""
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        low = ln.lower()
        if any(w in low for w in ("passed", "failed", "ok", "accepts",
                                  "checked", "error")):
            return ln[:70]
    return lines[-1][:70] if lines else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="also run the slow physics self-tests")
    ap.add_argument("--list", action="store_true",
                    help="print what would run and exit")
    ap.add_argument("--window", action="store_true",
                    help="also try the window test, which needs a display")
    a = ap.parse_args()

    checks = list(FAST) + (list(SLOW) if a.full else [])
    if a.list:
        for name, cmd in checks:
            print("  %-24s %s" % (name, os.path.relpath(cmd[0], HERE)))
        return 0

    print("=" * 74)
    print("SMOKE TEST   no training, no simulation campaign")
    print("=" * 74)

    missing, absent = packages()
    if missing:
        print("  MISSING PACKAGES: %s" % ", ".join(missing))
        print("  install them with:  python gui/install_requirements.py")
        return 2
    for why in absent:
        print("  note: not installed, %s" % why)

    bad = compiles()
    rows = [("every file compiles", "FAILED" if bad else "ok",
             bad[0][:70] if bad else "all .py files parse", "\n".join(bad))]

    for name, cmd in checks:
        if not os.path.exists(cmd[0]):
            rows.append((name, "MISSING", "not found: %s" % cmd[0], ""))
            continue
        t0 = time.time()
        p = subprocess.run([PY] + cmd, capture_output=True, text=True, cwd=HERE)
        out = (p.stdout or "") + (p.stderr or "")
        rows.append((name, "ok" if p.returncode == 0 else "FAILED",
                     "%-52s %4.1fs" % (verdict(out), time.time() - t0), out))

    if a.window:
        wid = os.path.join(GUI, "test_gui_widgets.py")
        p = subprocess.run([PY, wid], capture_output=True, text=True, cwd=HERE)
        out = (p.stdout or "") + (p.stderr or "")
        if "tkinter" in out and "No module" in out or "no display" in out.lower():
            rows.append(("the window itself", "skipped",
                         "no display or no tkinter here", ""))
        else:
            rows.append(("the window itself",
                         "ok" if p.returncode == 0 else "FAILED",
                         verdict(out), out))

    print()
    w = max(len(r[0]) for r in rows)
    for name, status, line, _ in rows:
        print("  %-*s  %-8s %s" % (w, name, status, line))
    print()

    failed = [r for r in rows if r[1] == "FAILED"]
    for name, _, _, out in failed:
        print("-" * 74)
        print("FULL OUTPUT: %s" % name)
        print("-" * 74)
        print(out[-6000:])
    if failed:
        print("%d of %d checks FAILED. This checkout is not ready to deploy."
              % (len(failed), len(rows)))
        return 1
    skipped = [r for r in rows if r[1] in ("skipped", "MISSING")]
    print("Everything that could run, passed." if skipped
          else "Everything passed.")
    print("Ready to deploy. Accuracy is a separate question: train, then read "
          "the held-out error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
