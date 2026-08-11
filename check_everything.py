#!/usr/bin/env python3
"""
check_everything.py — run every self-check in the project and report once.

    python check_everything.py

WHAT IT RUNS, and what each one would catch

  prtlb_2d.py   (flow, transport and reactions, two dimensions)
  prtlb_3d.py   (flow, transport and reactions, three dimensions)
      The transport solver against answers that can be written down without
      it: plug flow through an open channel must leave exp(-Da) at the
      outlet, nothing may exceed the inlet concentration, a sealed half of a
      domain must stay exactly empty. Catches the whole family of boundary,
      stability and length-scale mistakes.

  settings_and_units.py --self-test
      The settings file: that a default one reproduces, exactly, the constants
      the solver used before settings files existed; that the two Damkohler
      conventions differ by exactly Peclet; that the settings_and_units-to-dimensionless
      conversion agrees with complab_campaign.py, which builds the CompLaB runs from
      the same groups; and that a file written out reads back unchanged.
      Catches the family of unit mistakes -- a factor of 86400, a length taken
      against the wrong scale -- that produce a perfectly runnable simulation
      of the wrong experiment.

  test_flow_solvers.py
      The D2Q9 and D3Q19 flow solvers against flow between two flat plates,
      which has an analytic parabola. Checks the lattice identities exactly,
      the viscosity recovered from the profile's curvature, and that the
      no-slip wall lands at the mid-link where Palabos and CompLaB put it.
      Also measures whether the default iteration counts actually converge.

  test_three_switches.py
      That every switch OFF reproduces the original code BIT FOR BIT, by
      re-implementing the original inside the test and comparing arrays.

  test_documented_numbers.py
      The numerical claims made in the documentation, against measurement.

  flow_coordinates.py --self-test
      Travel time, wall distance and the squash function.

  gui/test_gui_commands.py
      That every button emits a command line its script actually accepts,
      asked of the real argparse parsers rather than a copy of them. Includes
      its own sensitivity check, so it cannot pass by going blind.

  gui/test_gui_widgets.py
      That every page of the window can be built and every view mode drawn.
      Needs a display; skipped with a note if there is none.

Anything that fails prints its own output in full. A green line here means
the checks passed, not that the science is right -- that is what the figures
are for.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "3D", "tools")
if not os.path.isdir(TOOLS):
    TOOLS = os.path.join(os.path.dirname(HERE), "GeometryAware3D", "tools")
GUI = os.path.join(HERE, "gui")

CHECKS = [
    ("the 2D simulator", [sys.executable, os.path.join(TOOLS, "prtlb_2d.py")]),
    ("the 3D simulator", [sys.executable, os.path.join(TOOLS, "prtlb_3d.py")]),
    ("your own settings", [sys.executable, os.path.join(TOOLS, "settings_and_units.py"),
                           "--self-test"]),
    ("pictures and VTI", [sys.executable, os.path.join(TOOLS,
                                                      "write_vti_and_png.py")]),
    ("the flow solvers", [sys.executable, os.path.join(TOOLS, "test_flow_solvers.py")]),
    ("the three switches", [sys.executable, os.path.join(TOOLS, "test_three_switches.py")]),
    ("the documented numbers", [sys.executable, os.path.join(TOOLS, "test_documented_numbers.py")]),
    ("flow coordinates", [sys.executable, os.path.join(TOOLS, "flow_coordinates.py"),
                          "--self-test"]),
    ("what the buttons send", [sys.executable,
                               os.path.join(GUI, "test_gui_commands.py")]),
]


def _verdict(out):
    """The line that states the outcome, not merely the last line printed.

    Several of these scripts end on a numpy warning, and showing that as the
    result reads like a failure when nothing failed.
    """
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        low = ln.lower()
        if any(w in low for w in ("passed", "failed", "ok.", "accepts",
                                  "checks,")):
            return ln
    return lines[-1] if lines else ""


def _python_with_tkinter():
    """An interpreter that can import tkinter, or None.

    The window needs tkinter; the analysis scripts do not, so the two may live
    in different interpreters on the same machine. Reporting "skipped" when a
    usable one is sitting right there would hide a real failure.
    """
    import shutil
    cands = [sys.executable] + [shutil.which(n) for n in
                                ("python3.13", "python3.12", "python3.11",
                                 "python3.10", "python3")]
    seen = set()
    for c in cands:
        if not c or c in seen:
            continue
        seen.add(c)
        r = subprocess.run([c, "-c", "import tkinter"], capture_output=True)
        if r.returncode == 0:
            return c
    return None


def main():
    # A hard stop against running inside itself.
    #
    # This script runs the GUI command test, that test builds the window's
    # action catalogue, and the catalogue now contains a button that runs this
    # script. Nothing about that is obvious from either file, and the first
    # time the loop closed it quietly spawned ten nested processes. The child
    # test skips argument-less scripts, which is the proper fix; this is the
    # backstop for the next time something reaches this file by a route nobody
    # anticipated.
    if os.environ.get("PRT_CHECK_RUNNING"):
        print("check_everything.py is already running further up the process "
              "tree; not starting a second copy inside it.")
        return 0
    os.environ["PRT_CHECK_RUNNING"] = "1"

    results = []
    for name, cmd in CHECKS:
        if not os.path.exists(cmd[1]):
            results.append((name, "MISSING", "not found: %s" % cmd[1], ""))
            continue
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = (p.stdout or "") + (p.stderr or "")
        results.append((name, "ok" if p.returncode == 0 else "FAILED",
                        _verdict(out), out))

    # the widget test only means anything with a display
    wid = os.path.join(GUI, "test_gui_widgets.py")
    if os.path.exists(wid):
        py = _python_with_tkinter()
        cmd = [py or sys.executable, wid]
        if not os.environ.get("DISPLAY"):
            import shutil
            if shutil.which("xvfb-run"):
                cmd = ["xvfb-run", "-a"] + cmd
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = (p.stdout or "") + (p.stderr or "")
        if "No module named 'tkinter'" in out or "no display" in out.lower():
            results.append(("the window itself", "skipped",
                            "no display or no tkinter on this Python", ""))
        else:
            results.append(("the window itself",
                            "ok" if p.returncode == 0 else "FAILED",
                            _verdict(out), out))

    print("=" * 74)
    print("PROJECT SELF-CHECK")
    print("=" * 74)
    width = max(len(n) for n, *_ in results)
    for name, status, line, _ in results:
        print("  %-*s  %-8s %s" % (width, name, status, line))
    print()

    bad = [r for r in results if r[1] == "FAILED"]
    for name, _, _, out in bad:
        print("-" * 74)
        print("FULL OUTPUT: %s" % name)
        print("-" * 74)
        print(out)
    if bad:
        print("%d of %d checks FAILED." % (len(bad), len(results)))
        return 1
    skipped = [r for r in results if r[1] in ("skipped", "MISSING")]
    print("Everything that could run, passed." if skipped
          else "Everything passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
