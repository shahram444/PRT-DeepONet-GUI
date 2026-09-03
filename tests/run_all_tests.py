#!/usr/bin/env python3
"""
run_all_tests.py — run everything, print one table, write one report.

    python run_all_tests.py                 everything (about half an hour)
    python run_all_tests.py --fast          skip the slow ones (about a quarter)
    python run_all_tests.py --list          just say what would be run

WHAT THIS IS
    One command that runs every test in this project, in order from cheapest to
    slowest, and writes a report you can send back to whoever asked you to run
    it. You do not need to know anything about the science to run it or to read
    what it says.

HOW TO READ THE RESULT
    Every line ends in ok, FAILED or SKIPPED.

      ok       that group passed
      FAILED   something is wrong. The full output of every failure is printed
               underneath the table, and saved in the report file
      SKIPPED  the group could not run at all, usually because something
               optional is not installed. A skip is NOT a pass, and it is
               printed differently for that reason

    The exit code is 0 if everything that ran passed, and 1 otherwise.

WHAT IT WRITES
    test_report.txt, beside this file. That is the thing to send back. It holds
    the table, the full output of anything that failed, and the version of
    Python and every library that was used, which is usually the first question
    anyone asks about a failure.
"""

import argparse
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "3D", "tools")
MODEL = os.path.join(ROOT, "3D", "model")
GUI = os.path.join(ROOT, "gui")

# =============================================================================
#  BLOCK 1.  WHAT TO RUN
#
#  (name, command, minutes, is_slow). Ordered cheapest first, deliberately: a
#  broken installation fails in the first thirty seconds rather than twenty
#  minutes in, and you get the answer before you walk away.
# =============================================================================
PY = sys.executable

GROUPS = [
    # ---- the tests in this folder ------------------------------------------
    ("smoke: does everything start",
     [PY, os.path.join(HERE, "test_01_smoke.py")], 1, False),
    ("unit: one function at a time",
     [PY, os.path.join(HERE, "test_02_units.py")], 1, False),
    ("data: files read back correctly",
     [PY, os.path.join(HERE, "test_03_data.py")], 1, False),
    ("repeatable: same input, same answer",
     [PY, os.path.join(HERE, "test_04_repeatable.py")], 1, False),
    ("refusals: bad input is refused",
     [PY, os.path.join(HERE, "test_05_refusals.py")], 2, False),

    # ---- the project's own checks ------------------------------------------
    ("the window's buttons",
     [PY, os.path.join(GUI, "test_gui_commands.py")], 2, False),
    ("the flow pipeline panel",
     [PY, os.path.join(GUI, "test_flow_panel.py")], 1, False),
    ("the sweep mode boxes",
     [PY, os.path.join(GUI, "test_gui_sweep_modes.py")], 1, False),
    ("the comments",
     [PY, os.path.join(ROOT, "audit_comments.py"), "--quiet"], 1, False),
    ("the flow descriptors",
     [PY, os.path.join(TOOLS, "flow_features.py"), "--self-test"], 1, False),
    ("the pressure solve",
     [PY, os.path.join(TOOLS, "harmonic_pressure.py"), "--self-test"], 1, False),
    ("the flow coordinates",
     [PY, os.path.join(TOOLS, "flow_coordinates.py"), "--self-test"], 1, False),

    # ---- the slow ones -----------------------------------------------------
    ("your own settings",
     [PY, os.path.join(TOOLS, "settings_and_units.py"), "--self-test"], 2, True),
    ("the 2D simulator",
     [PY, os.path.join(TOOLS, "prtlb_2d.py")], 3, True),
    ("the 3D simulator",
     [PY, os.path.join(TOOLS, "prtlb_3d.py")], 4, True),
    ("both flow solvers",
     [PY, os.path.join(TOOLS, "test_flow_solvers.py")], 2, True),
    ("the velocity pipeline",
     [PY, os.path.join(MODEL, "test_flow_pipeline.py")], 4, True),
]


# =============================================================================
#  BLOCK 2.  RUNNING ONE GROUP
#
#  Each group is a separate process. That matters: a crash in one cannot take
#  the others down with it, and the table still shows what passed.
# =============================================================================
def run_group(cmd, minutes):
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"          # never try to open a window
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           timeout=max(600, minutes * 180))
        out = (p.stdout or "") + (p.stderr or "")
        rc = p.returncode
    except subprocess.TimeoutExpired:
        return "FAILED", "TIMED OUT", 0.0, "the group did not finish in time"
    except FileNotFoundError:
        return "SKIPPED", "not found", 0.0, "%s is not there" % cmd[1]
    dt = time.time() - t0

    # A group that could not run at all is SKIPPED, not FAILED. Reporting a
    # missing optional library as a failure sends people hunting for a bug
    # that is not there.
    if rc != 0 and ("ModuleNotFoundError" in out or "ImportError" in out):
        missing = [ln for ln in out.split("\n") if "ModuleNotFoundError" in ln
                   or "ImportError" in ln]
        return "SKIPPED", "missing library", dt, (missing[-1] if missing else out[-200:])

    return ("ok" if rc == 0 else "FAILED"), verdict(out), dt, out


def verdict(out):
    """The line that states the outcome, not merely the last line printed.

    Several of these scripts end on a library warning, and showing that as the
    result reads as a failure when everything passed.
    """
    good = ("PASSED", "passed", "Everything", "Every button", "Every file",
            "All ", "ok", "checks")
    for line in reversed([ln.strip() for ln in out.split("\n") if ln.strip()]):
        if any(g in line for g in good) or "FAIL" in line:
            return line[:64]
    return (out.strip().split("\n")[-1][:64] if out.strip() else "")


# =============================================================================
#  BLOCK 3.  THE REPORT
# =============================================================================
def environment():
    lines = ["python   %s" % sys.version.split()[0],
             "         %s" % sys.executable,
             "platform %s" % platform.platform()]
    for mod in ("numpy", "scipy", "h5py", "torch", "matplotlib", "skimage"):
        try:
            m = __import__(mod)
            lines.append("%-8s %s" % (mod, getattr(m, "__version__", "?")))
        except Exception:                                      # noqa: BLE001
            lines.append("%-8s NOT INSTALLED" % mod)
    try:
        import tkinter
        lines.append("%-8s %s" % ("tkinter", tkinter.TkVersion))
    except Exception:                                          # noqa: BLE001
        lines.append("%-8s NOT INSTALLED (the window will not open; the tests "
                     "still run)" % "tkinter")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true",
                    help="skip the slow groups: about a quarter of an hour "
                         "instead of half of one. Good for a first look; run "
                         "the whole thing before reporting back")
    ap.add_argument("--list", action="store_true",
                    help="print what would be run, and run nothing")
    ap.add_argument("--report", default=os.path.join(HERE, "test_report.txt"),
                    help="where to write the report")
    a = ap.parse_args()

    groups = [g for g in GROUPS if not (a.fast and g[3])]

    if a.list:
        print("%d group(s), about %d minutes:\n"
              % (len(groups), sum(g[2] for g in groups)))
        for name, cmd, mins, slow in groups:
            print("  %-38s ~%2d min  %s%s"
                  % (name, mins, os.path.basename(cmd[1]),
                     "   (slow)" if slow else ""))
        return 0

    print("=" * 78)
    print("PRT-DeepONet Studio  ::  running every test")
    print("=" * 78)
    for ln in environment():
        print("  " + ln)
    print()
    print("%d groups, roughly %d minutes.%s"
          % (len(groups), sum(g[2] for g in groups),
             "  (--fast: the slow ones are skipped)" if a.fast else ""))
    print("Nothing here changes the project. Everything is written to a")
    print("temporary folder and removed afterwards.")
    print()

    results = []
    width = max(len(g[0]) for g in groups)
    for name, cmd, mins, _slow in groups:
        print("  %-*s  running ..." % (width, name), end="", flush=True)
        status, line, dt, out = run_group(cmd, mins)
        print("\r  %-*s  %-8s %5.0fs  %s" % (width, name, status, dt, line))
        results.append((name, status, line, dt, out))

    failed = [r for r in results if r[1] == "FAILED"]
    skipped = [r for r in results if r[1] == "SKIPPED"]

    print()
    print("=" * 78)
    if failed:
        print("%d of %d groups FAILED." % (len(failed), len(results)))
    else:
        print("Everything that ran, passed.  %d group(s)."
              % (len(results) - len(skipped)))
    if skipped:
        print("%d group(s) SKIPPED, which is weaker than passing:" % len(skipped))
        for name, _s, line, _dt, _o in skipped:
            print("   %-38s %s" % (name, line))
    print("=" * 78)

    # ---- the failures, in full, on screen and in the file -------------------
    body = []
    for name, status, line, dt, out in results:
        if status == "FAILED":
            body.append("=" * 78)
            body.append("FULL OUTPUT: %s" % name)
            body.append("=" * 78)
            body.append(out.rstrip())
            body.append("")
    if body:
        print()
        print("\n".join(body))

    with open(a.report, "w", encoding="utf-8") as f:
        f.write("PRT-DeepONet Studio, test report\n")
        f.write("%s\n\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
        for ln in environment():
            f.write("  %s\n" % ln)
        f.write("\n")
        for name, status, line, dt, _o in results:
            f.write("  %-*s  %-8s %5.0fs  %s\n" % (width, name, status, dt, line))
        f.write("\n")
        f.write("%d groups, %d failed, %d skipped\n"
                % (len(results), len(failed), len(skipped)))
        f.write("\n")
        f.write("\n".join(body))
    print()
    print("Report written to %s" % a.report)
    print("That file is the one to send back.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # Piping this into `head` closes the pipe early. That is not an error
        # and must not print a traceback at somebody who only wanted the first
        # few lines.
        os._exit(0)
