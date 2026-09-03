#!/usr/bin/env python3
# =============================================================================
# CHANGED FROM THE 2D VERSION
#
#   WHAT CHANGED HERE, IN ONE LINE
#     Six new check groups, and one dead check removed.
#
#   THE SIX ADDED
#     the flow descriptors    3D/tools/flow_features.py --self-test
#                             15 checks, including the 2D forms against the
#                             values their notebook produces.
#     the pressure solve      3D/tools/harmonic_pressure.py --self-test
#                             14 checks on the Laplace solve that feeds the
#                             velocity operator's trunk.
#     the velocity pipeline   3D/model/test_flow_pipeline.py
#                             33 checks, end to end on a dataset it builds.
#     the flow pipeline panel gui/test_flow_panel.py
#                             35 checks on the window's flow panel: that its
#                             five steps line up with each other, and that a
#                             failed step stops the sequence instead of
#                             running the next one against nothing.
#     the comments            audit_comments.py --quiet
#                             every file carrying its top block, its section
#                             markers and its line comments.
#     the test suite          tests/, discovered with unittest
#                             smoke, unit, round trip, repeatability and the
#                             refusals. Written to be run by somebody who does
#                             not work on this project; tests/README.md is
#                             addressed to them.
#
#   WHY MODEL HAD TO BE ADDED AS A PATH
#     Every check before this lived in 3D/tools. test_flow_pipeline.py lives in
#     3D/model, beside the models it exercises, so this file now resolves both
#     directories rather than assuming one.
#
#   THE ONE REMOVED
#     "pictures and VTI" pointed at write_vti_and_png.py, which was deleted
#     from the project earlier. The check could only ever report MISSING, which
#     is noise, not information.
#
#   ABOUT THE FAILURES YOU USED TO SEE
#     Four groups failed here for a long time and were reported as inherited.
#     Three of them were real faults and have been fixed: settings_and_units.py
#     wrote an XML it could not read back, test_documented_numbers.py was
#     hardcoded to one machine, and the window completeness check was itself
#     wrong about six settings that ARE reachable. If something fails now,
#     compare against PRT-DeepONet-v1.1 before assuming it is new.
# =============================================================================
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
# Derived from TOOLS rather than from HERE, so it follows the fallback above
# into a GeometryAware3D layout instead of pointing at a folder that is not there.
MODEL = os.path.join(os.path.dirname(TOOLS), "model")
GUI = os.path.join(HERE, "gui")     # the window's own tests live beside it

CHECKS = [
    # Order matters here: cheapest first. A broken installation fails in the
    # first second on the simulators rather than eight minutes in, and the user
    # gets the answer before they walk away.
    ("the 2D simulator", [sys.executable, os.path.join(TOOLS, "prtlb_2d.py")]),
    ("the 3D simulator", [sys.executable, os.path.join(TOOLS, "prtlb_3d.py")]),
    ("your own settings", [sys.executable, os.path.join(TOOLS, "settings_and_units.py"),
                           "--self-test"]),
    ("the flow solvers", [sys.executable, os.path.join(TOOLS, "test_flow_solvers.py")]),
    # Still called "three": the flow pipeline is checked by the groups below.
    ("the three switches", [sys.executable, os.path.join(TOOLS, "test_three_switches.py")]),
    ("the documented numbers", [sys.executable, os.path.join(TOOLS, "test_documented_numbers.py")]),
    ("flow coordinates", [sys.executable, os.path.join(TOOLS, "flow_coordinates.py"),
                          "--self-test"]),
    # ---- the flow groups, new in v1.2 --------------------------------------
    # The first two are self-tests inside the modules they test, so they cost a
    # second and run everywhere. The third builds a small dataset and trains for
    # two epochs, so it is the slow one. The fourth is with the GUI checks
    # below, because it tests the window rather than the model.
    ("the flow descriptors", [sys.executable, os.path.join(TOOLS, "flow_features.py"),
                              "--self-test"]),
    ("the pressure solve", [sys.executable, os.path.join(TOOLS, "harmonic_pressure.py"),
                            "--self-test"]),
    ("the velocity pipeline", [sys.executable,
                               os.path.join(MODEL, "test_flow_pipeline.py")]),
    # No widget tests here: they need a display, or xvfb, and a check that
    # cannot run on a plain server is a check nobody runs.
    ("what the buttons send", [sys.executable,
                               os.path.join(GUI, "test_gui_commands.py")]),
    # New in v1.2. The buttons test asks whether each command parses. This one
    # asks whether the five steps of the flow pipeline line up WITH EACH OTHER,
    # which argparse cannot see: five perfectly valid commands can still have
    # step 5 reading a file step 4 never wrote.
    ("the flow pipeline panel", [sys.executable,
                                 os.path.join(GUI, "test_flow_panel.py")]),
    # Not a check on the code's behaviour but on whether it can be READ: every
    # new or changed file carrying its top block, its section markers and its
    # line comments. A documentation rule nobody checks decays on the first
    # busy afternoon.
    ("the comments", [sys.executable, os.path.join(HERE, "audit_comments.py"),
                      "--quiet"]),
    # The suite in tests/: smoke, unit, round trip, repeatability, refusals.
    # Written to be run by somebody who does not work on this project, and
    # tests/README.md is addressed to them.
    #
    # Discovered with unittest rather than through tests/run_all_tests.py,
    # which would be the obvious call and is the wrong one: that runner ALSO
    # runs several of the groups above, so this file would end up running them
    # twice and reporting each of them under two names.
    ("the test suite", [sys.executable, "-m", "unittest", "discover",
                        "-s", os.path.join(HERE, "tests"),
                        "-t", HERE, "-p", "test_*.py"]),
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
