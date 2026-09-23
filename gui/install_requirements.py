#!/usr/bin/env python3
"""
install_requirements.py — install everything the project needs, into the Python
that is running THIS script.

That last part matters.  The launcher starts the window with the project's own
virtual environment (3D/.venv) when one exists, so running this script with the
same interpreter puts the packages exactly where the analysis scripts will look
for them.  Installing "into Python" generally is what goes wrong: people install
into the system Python and the venv stays empty.

    python install_requirements.py                 # everything, CPU torch
    python install_requirements.py --torch cuda    # if the machine has an NVIDIA card
    python install_requirements.py --torch skip    # viewer only, no training
    python install_requirements.py --check         # report, install nothing

WHY EACH PACKAGE
    numpy         arrays, everywhere
    scipy         distance transforms, connected components, smoothing
    h5py          the dataset format
    matplotlib    every figure and the viewer
    scikit-image  marching cubes, for the 3D surface renders
    torch         the network itself: training and prediction

A NOTE ON pip.exe IN A MOVED VIRTUAL ENVIRONMENT
    On Windows the little pip.exe shim inside a venv hard-codes the absolute path
    of the venv at the moment it was created.  Move the folder and pip.exe stops
    working, with a confusing error.  python -m pip does NOT hard-code anything,
    so this script always uses that form.
"""

import argparse
import subprocess
import sys

# =============================================================================
#  WHAT THIS PROJECT NEEDS, AND WHAT IT ONLY LIKES
#  Split deliberately. A missing REQUIRED package stops the analysis; a missing
#  optional one costs a picture or a faster route. Reporting the two the same
#  way sends people installing things they do not need.
# =============================================================================
PACKAGES = [
    ("numpy", "numpy", "arrays, used by everything"),
    ("scipy", "scipy", "distance transforms, connected components, smoothing"),
    ("h5py", "h5py", "the dataset file format"),
    ("matplotlib", "matplotlib", "every figure, and the viewer"),
    ("skimage", "scikit-image", "marching cubes, for the 3D surface renders"),
    # OPTIONAL. The geometry generator can use it for a smoother geodesic
    # field, but it is NOT the default and nothing breaks without it -- the
    # datasets are built with the neighbour-relaxation solver either way, and
    # mixing the two would give a model one distance scale in training and a
    # 17% smaller one at prediction. Listed so the installer offers it rather
    # than leaving a ModuleNotFoundError to be discovered mid-run.
    ("skfmm", "scikit-fmm", "optional: a smoother geodesic field"),
]
TORCH = ("torch", "torch", "the network itself: training and prediction")

CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def have(mod):
    try:
        __import__(mod)
        return True
    except Exception:                                          # noqa: BLE001
        # Any failure to import counts as missing, not just ImportError: a
        # half-installed package raises other things and is no more usable.
        return False


def report(title):
    print("=" * 72)
    print(title)
    print("=" * 72)
    print("Python being used:")         # which interpreter, before what is in it
    print("   %s" % sys.executable)
    print("   version %s" % sys.version.split()[0])
    print()
    allok = True
    # torch last, because it is the slowest to import and the largest to
    # install, so its line is the one people wait for.
    for mod, pkg, why in PACKAGES + [TORCH]:
        ok = have(mod)
        allok &= ok
        print("   %-8s %-14s %s" % ("OK" if ok else "MISSING", pkg, why))
    print()
    return allok


def pip(args):
    # Run as a module of THIS interpreter, never as a bare "pip": on a machine
    # with several Pythons those are routinely different environments.

    cmd = [sys.executable, "-m", "pip"] + args
    print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    return r.returncode


# =============================================================================
#  THE EXIT STATUS
#  --check returns 0 only when everything required is present. It used to
#  report what was missing and return success anyway, so a setup step that
#  tested it carried on into a run that could not work.
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch", choices=["cpu", "cuda", "skip"], default="cpu",
                    help="'cpu' is the right answer unless this machine has an "
                         "NVIDIA card. The CPU build is much smaller and is fine "
                         "for the practice dataset and for prediction.")
    ap.add_argument("--check", action="store_true",
                    help="report what is installed and stop")
    ap.add_argument("--upgrade", action="store_true",
                    help="upgrade packages that are already present")
    a = ap.parse_args()

    # AUDIT GUIINST-10. --check used to return 0 whatever it found, so a
    # scripted environment check reported missing packages and still exited
    # successfully. It now returns 0 only when everything is present, which is
    # what an exit status is for.
    everything_present = report("CHECKING WHAT IS ALREADY INSTALLED")
    if everything_present and not a.upgrade:
        print("Everything the project needs is already present. Nothing to do.")
        return 0
    if a.check:
        if not everything_present:
            print("Some packages are missing. Run this again without --check "
                  "to install them.")
            return 1
        return 0

    print("=" * 72)
    print("INSTALLING")
    print("=" * 72)
    print("This downloads from the internet and may take several minutes.")
    print("torch is the big one, roughly 200 MB for the CPU build.")
    print()

    if pip(["install", "--upgrade", "pip"]):
        print("\nCould not upgrade pip. Continuing anyway.\n")

    want = [pkg for mod, pkg, _ in PACKAGES if a.upgrade or not have(mod)]
    if want:
        flags = ["install"] + (["--upgrade"] if a.upgrade else []) + want
        if pip(flags):
            print("\nFAILED while installing: %s" % " ".join(want))
            print("The usual causes are no internet connection, or a proxy that "
                  "needs configuring.")
            return 1
    else:
        print("The base packages are already present.")

    if a.torch != "skip" and (a.upgrade or not have("torch")):
        print()
        if a.torch == "cpu":
            print("Installing the CPU build of torch. Choose --torch cuda only "
                  "if this machine has an NVIDIA card.")
            rc = pip(["install", "torch", "--index-url", CPU_INDEX])
        else:
            print("Installing the default torch build, which brings CUDA support.")
            rc = pip(["install", "torch"])
        if rc:
            print("\nFAILED while installing torch.")
            print("You can still use the viewer and build datasets without it; "
                  "only training and prediction need torch.")
            return 1

    print()
    ok = report("RESULT")
    if ok:
        print("Everything is installed. Close this and the window is ready to use.")
        print("If the window is still open, use Tools > Check this computer can "
              "run everything to confirm.")
    else:
        print("Some packages are still missing, see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
