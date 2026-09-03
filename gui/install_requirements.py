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

# The CPU wheel index. The default PyPI torch pulls the CUDA build, which is
# about two gigabytes of driver code a laptop will never run.
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


# =============================================================================
#  BLOCK 2.  IS IT THERE?
#
#  By IMPORT NAME, not by package name: pip installs scikit-image and Python
#  imports skimage. Asking pip what it thinks is installed answers about a
#  different interpreter as often as not, which is the whole failure this
#  script exists to sort out.
# =============================================================================
def have(mod):
    try:
        __import__(mod)
        return True
    except Exception:                                          # noqa: BLE001
        return False


def report(title):
    print("=" * 72)
    print(title)
    print("=" * 72)
    print("Python being used:")
    print("   %s" % sys.executable)
    print("   version %s" % sys.version.split()[0])
    print()
    allok = True    # every package, not the first missing one
    # torch last, because it is the big one and the one most likely to be
    # missing, so it is the line the eye lands on.
    for mod, pkg, why in PACKAGES + [TORCH]:
        ok = have(mod)
        allok &= ok
        print("   %-8s %-14s %s" % ("OK" if ok else "MISSING", pkg, why))
    print()
    return allok


# =============================================================================
#  BLOCK 3.  INSTALLING
#
#  Always "sys.executable -m pip", never a bare "pip". A pip on PATH belongs to
#  whichever Python was installed last, and packages put there are invisible to
#  the interpreter the window actually runs its scripts with.
# =============================================================================
def pip(args):
    cmd = [sys.executable, "-m", "pip"] + args
    print("$ " + " ".join(cmd), flush=True)      # nothing installed unseen
    r = subprocess.run(cmd)
    return r.returncode


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

    if report("CHECKING WHAT IS ALREADY INSTALLED") and not a.upgrade:
        print("Everything the project needs is already present. Nothing to do.")
        return 0
    if a.check:
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
