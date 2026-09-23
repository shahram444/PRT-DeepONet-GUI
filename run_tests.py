#!/usr/bin/env python3
"""
run_tests.py -- the comprehensive test run, with a written report and a fix for
anything that fails.

    python run_tests.py                  everything, about 8 to 12 minutes
    python run_tests.py --quick          skip the slow physics, about 1 minute
    python run_tests.py --group model    just one group
    python run_tests.py --window         also test the window, needs a display
    python run_tests.py --report r.md    write the report to a file as well

WHAT THIS ANSWERS, AND WHAT IT DOES NOT

It answers: does this checkout work on this machine, with these packages.  It
does not train, and it does not answer whether the model is accurate.  Accuracy
comes from training and reading the held-out error; this is the gate you run
before you start.

HOW IT REPORTS

Every check prints its own verdict line and its time.  Anything that fails
prints three things: what the check was protecting, the tail of the real
output, and a concrete fix.  At the end you get a per-group summary and, if
anything failed, an ordered list of what to do about it.

THE GROUPS

  env        packages, versions, and what the machine has
  static     every file parses
  model      the network: shapes, the single output field, no FiLM, the 2D
             fallback, gradients reaching every parameter
  data       the dataset layer: species selection, parameter layouts, the
             geometry split, the volume writer
  pipeline   evaluate.py and predict.py driven end to end on an untrained
             checkpoint, which is what catches renamed keys and shape drift
  units      your settings, the unit conversions, and that a written template
             is valid XML
  flow       the flow descriptors, the pressure proxy, flow coordinates, and
             the velocity-informed pipeline
  switches   every feature switch OFF reproducing the original code bit for bit
  gui        every button emits a command line its script accepts
  physics    the lattice-Boltzmann solvers against answers known in closed
             form, and the numbers quoted in the documentation.  Slow.
"""

import argparse
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "3D", "tools")
MODEL = os.path.join(HERE, "3D", "model")
GUI = os.path.join(HERE, "gui")
TESTS = os.path.join(HERE, "tests")
PY = sys.executable

INSTALL = "python gui/install_requirements.py"

CORE = os.path.join(HERE, "prt_core")

# name, group, command, what it protects, what to do when it fails
CHECKS = [
    ("the shared core", "model", [os.path.join(CORE, "test_core.py"), "--quiet"],
     "the four things every script now imports instead of carrying its own "
     "copy of: which chemistry a model is for, how the trunk's distance column "
     "is scaled, one reader for every geometry format, and one definition of "
     "the network in both key layouts. It also checks the claim that makes the "
     "naming worth anything, that the convention called 'published' reproduces "
     "the released notebook's own column voxel for voxel on the released rock",
     "Read which of the four self-tests failed and run it on its own: "
     "python prt_core/reactions.py --self-test, and the same for "
     "conventions.py, inputs.py and model.py. If the crossing checks failed "
     "instead, the four still agree with themselves and have stopped agreeing "
     "with each other; the message names which pair. A failure here makes the "
     "failures below it echoes, so fix this one first."),

    # WHY THESE FIVE ARE HERE.
    #
    # tests/ holds two sets of files: the three named ones below, and a
    # numbered series run by tests/run_all_tests.py. The two sets overlap
    # heavily and NEITHER was complete, so seeing everything meant knowing to
    # run both runners, and nobody did. Ten real failures sat in the numbered
    # series for two releases because this runner never opened them.
    #
    # They are listed here individually rather than by calling run_all_tests.py,
    # because that would run the lattice Boltzmann self-tests a second time and
    # add four minutes for no new information.
    ("everything starts", "static", [os.path.join(TESTS, "test_01_smoke.py")],
     "every module importing and every script surviving its own --help. The "
     "cheapest question asked of everything, and the one that makes every "
     "other test meaningful",
     "Read which import failed. This is almost always a file that was renamed "
     "or a package that is not installed; run python gui/install_requirements.py "
     "--check first."),

    ("one function at a time", "units", [os.path.join(TESTS, "test_02_units.py")],
     "the descriptors and the scalings against answers worked out by hand, "
     "never produced by this code. A test whose expected answer came from the "
     "code under test proves only that the code is deterministic",
     "The failing assertion names the quantity and the hand-computed value. "
     "Work the small example through on paper before changing anything: these "
     "numbers are the definition, not a recorded output."),

    ("files read back correctly", "data", [os.path.join(TESTS, "test_03_data.py")],
     "the dataset layout being what everything downstream expects, a geometry "
     "surviving a round trip, and add_flow_features.py changing a file in "
     "place without touching anything it was not asked to",
     "If a fixture raised rather than a check failing, the fixture and the "
     "reader have drifted apart: tests/_common.py two_channels needs ny of at "
     "least 9, and the reader needs conc_scale on samples/conc."),

    ("same input, same answer", "data",
     [os.path.join(TESTS, "test_04_repeatable.py")],
     "the seed being obeyed, the pressure solve repeating exactly, and every "
     "scaling being undoable. Without this a result cannot be reproduced and "
     "an ablation compares nothing",
     "If the voxel sampling differs between two runs at the same seed, "
     "something in dataset_reader.py is drawing from the global numpy random "
     "state instead of its own default_rng(seed)."),

    ("bad input is refused", "data",
     [os.path.join(TESTS, "test_05_refusals.py")],
     "every wrong request failing with a sentence rather than a traceback or, "
     "worse, a plausible looking answer: a missing flow field, a geometry at "
     "the wrong size, a bare state dict, two contradictory switches",
     "A test here failing usually means a guard was removed or moved after the "
     "thing it was guarding. Read which refusal stopped happening."),

    ("the comments", "static", [os.path.join(HERE, "audit_comments.py"), "--quiet"],
     "every file carrying the top block that says what changed from the 2D "
     "version, block comments marking its sections, and line comments where "
     "the code looks wrong until explained",
     "It names each file and what it is short of. This is a documentation "
     "gate, not a correctness one, so it never blocks a result; it blocks a "
     "release."),

    ("the network", "model", [os.path.join(TESTS, "test_model.py")],
     "the architecture staying the published 2D one: one output field, a "
     "scalar bias, no FiLM, the parameter branch honouring n_params, and a "
     "dataset with nz = 1 going through 2D convolutions",
     "Read the assertion that failed. If it is test_shapes or "
     "test_scalar_bias, something reintroduced the multi-species head in "
     "3D/model/deeponet_model.py: the forward pass must end in "
     "(basis * code.unsqueeze(1)).sum(-1) + self.bias with bias of shape (1,). "
     "If it is test_no_film, the Trunk grew a film ModuleList again and must "
     "go back to a plain Linear stack."),

    ("the dataset layer", "data", [os.path.join(TESTS, "test_dataset.py")],
     "the reader handing the model one species at a time, both parameter "
     "layouts coexisting, and train and test never sharing a geometry",
     "If test_target_shape failed, the target regrew a species axis in "
     "3D/tools/dataset_reader.py: __getitem__ must return y of shape "
     "(n_points,). If test_param_layouts failed, check param_layout_names in "
     "3D/tools/settings_and_units.py and that the dataset's param_names "
     "attribute survived the write. If test_split_by_geometry failed, the "
     "split is leaking pore structure and every held-out number is optimistic."),

    ("evaluate and predict", "pipeline",
     [os.path.join(TESTS, "test_pipeline.py")],
     "the whole load, build, inference and write path, exercised with random "
     "weights so it costs seconds instead of hours",
     "This is usually a renamed checkpoint key or a shape that stopped "
     "matching. The output tail shows which script died. Compare the keys "
     "train.py writes with the keys evaluate.py and predict.py read: species, "
     "param_names, trunk_in_dim, in_channels, grid, args. If only the figure "
     "writers failed, something still expects a list of species."),

    ("your own settings", "units",
     [os.path.join(TOOLS, "settings_and_units.py"), "--self-test"],
     "the unit conversions, the two Damkohler conventions, and that a "
     "settings file written out is valid XML and reads back unchanged",
     "A failure here means a runnable simulation of the wrong experiment, so "
     "do not ignore it. If the failure is an XML ParseError, the template text "
     "contains a double hyphen inside an XML comment, which is illegal: find "
     "it in the to_xml text and reword it. If a conversion check failed, "
     "compare against complab_campaign.py, which derives the same groups."),

    ("flow coordinates", "flow",
     [os.path.join(TOOLS, "flow_coordinates.py"), "--self-test"],
     "travel time, wall distance and the squash function used by the "
     "dimension-free trunk",
     "Only the --dim-free switch depends on this. If you are not using it, the "
     "failure is not blocking, but fix it before you turn that switch on."),

    ("the flow descriptors", "flow",
     [os.path.join(TOOLS, "flow_features.py"), "--self-test"],
     "MIS, UPRM and the wall distance, the geometry features the flow-aware "
     "branch is conditioned on",
     "Check that scipy is installed and recent: these are distance transforms "
     "and connected components. UPRM also requires connectivity 1; a failure "
     "mentioning connectivity means someone passed a different value."),

    ("the pressure solve", "flow",
     [os.path.join(TOOLS, "harmonic_pressure.py"), "--self-test"],
     "the harmonic pressure proxy and its gradient, which the trunk takes per "
     "point in the velocity-informed path",
     "The solve is a plain Laplace problem on the pore space. A failure is "
     "usually a boundary condition applied to the wrong face after a change of "
     "flow axis. Check read_reference_orientation and the flow_axis argument."),

    ("the three switches", "switches",
     [os.path.join(TOOLS, "test_three_switches.py")],
     "every feature switch OFF reproducing the original code BIT FOR BIT, by "
     "re-implementing the original inside the test and comparing arrays",
     "This is the most valuable test in the tree and the least ambiguous. If "
     "it fails, a change meant to be optional is now always on. The test "
     "prints which array differed and by how much; find the switch in "
     "resolve_switches in 3D/tools/dataset_reader.py and make its OFF branch "
     "do nothing at all."),

    ("the velocity pipeline", "flow",
     [os.path.join(MODEL, "test_flow_pipeline.py")],
     "the velocity-informed path refusing bad input loudly: a geometry at the "
     "wrong grid, a bare state_dict with no scaling, a dataset with no "
     "velocity, an unknown conditioning column",
     "If a refusal stopped happening, the two-stage path will run on "
     "mismatched input and produce plausible nonsense. Restore the guard in "
     "3D/model/predict_velocity.py or train_velocity.py, whichever the output "
     "names. One skip is expected: it needs Jung's released checkpoints, which "
     "are not in this repository."),

    ("what the buttons send", "gui",
     [os.path.join(GUI, "test_gui_commands.py")],
     "every button in the window emitting a command line that its script's "
     "real argparse parser accepts",
     "The output names the button and the flag. Either the script's flag was "
     "renamed and the window still sends the old one, or a new flag was added "
     "to the window without adding it to the script. Fix the side that is "
     "wrong, not the test."),

    # AUDIT GUI-02. This test existed and was in no runner, which is how the
    # panel it describes came to be missing from the window entirely.
    ("the flow pipeline panel", "gui",
     [os.path.join(GUI, "test_flow_panel.py")],
     "the five steps of the flow capability: that they exist, that they are "
     "filled in from one set of settings so they cannot disagree, and that the "
     "sequence stops at the first failure instead of running a step against "
     "something the failed one never wrote",
     "The output names the check. A wiring failure means two steps disagree "
     "about a path; a sequencing failure means a failed step did not stop the "
     "rest. Fix the panel, not the test."),

    ("the sweep boxes", "gui",
     [os.path.join(GUI, "test_gui_sweep_modes.py")],
     "the range-or-list boxes, so neither half leaks the other half's flags "
     "into the command line",
     "Look at the page the output names and check which box wrote which flag. "
     "A leak here silently runs a different sweep from the one on screen."),

    ("the 2D simulator", "physics", [os.path.join(TOOLS, "prtlb_2d.py")],
     "the transport solver against answers that can be written down without "
     "it: plug flow leaving exp(-Da) at the outlet, nothing exceeding the "
     "inlet concentration, a sealed half staying exactly empty",
     "A failure is a boundary, stability or length-scale mistake, not a "
     "tolerance to loosen. Check PRT_DT and PRT_MAXFRAC first: an unstable "
     "time step shows up here before it shows up in a dataset."),

    ("the 3D simulator", "physics", [os.path.join(TOOLS, "prtlb_3d.py")],
     "the same closed-form checks in three dimensions",
     "If the 2D simulator passes and this fails, suspect the third axis: the "
     "flow axis convention, or a boundary applied to the wrong face."),

    ("the flow solvers", "physics",
     [os.path.join(TOOLS, "test_flow_solvers.py")],
     "D2Q9 and D3Q19 against flow between flat plates, which has an analytic "
     "parabola, including where the no-slip wall actually sits",
     "A small error in the recovered viscosity usually means the iteration "
     "count is too low to converge, and the test says so. A large one means "
     "the lattice weights or the wall treatment changed."),

    ("the documented numbers", "physics",
     [os.path.join(TOOLS, "test_documented_numbers.py")],
     "the numerical claims in the documentation, against measurement",
     "This one needs a trained checkpoint to measure against and will fail on "
     "a fresh checkout. Train something first, or read the failure as a note "
     "that the documented number is unverified here."),
]

GROUPS = ["env", "static", "model", "data", "pipeline", "units", "flow",
          "switches", "gui", "physics"]
SLOW_GROUPS = {"physics"}

NEEDED = [("numpy", "numpy"), ("scipy", "scipy"), ("h5py", "h5py"),
          ("torch", "torch"), ("matplotlib", "matplotlib")]
OPTIONAL = [("skfmm", "scikit-fmm", "predict.py uses it for the geodesic "
             "field; without it, pass a geometry that already has one"),
            ("skimage", "scikit-image", "the 3D surface renders"),
            ("tkinter", "tkinter", "the window; nothing else needs it")]


def env_report():
    """Everything about this machine that a failure might be blamed on."""
    lines = ["python      : %s" % sys.version.split()[0],
             "executable  : %s" % sys.executable,
             "platform    : %s %s" % (platform.system(), platform.release()),
             "machine     : %s, %d logical cpus"
             % (platform.machine(), os.cpu_count() or 0)]
    missing = []
    for mod, pkg in NEEDED:
        try:
            m = __import__(mod)
            lines.append("%-12s: %s" % (pkg, getattr(m, "__version__", "?")))
        except Exception as e:
            missing.append(pkg)
            lines.append("%-12s: MISSING (%s)" % (pkg, type(e).__name__))
    absent = []
    for mod, pkg, why in OPTIONAL:
        try:
            __import__(mod)
            lines.append("%-12s: present" % pkg)
        except Exception:
            absent.append((pkg, why))
            lines.append("%-12s: absent, %s" % (pkg, why))
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        lines.append("torch device: %s" % dev)
        if dev == "cuda":
            lines.append("gpu         : %s" % torch.cuda.get_device_name(0))
        lines.append("threads     : %d" % torch.get_num_threads())
    except Exception:
        pass
    return lines, missing, absent


# Directories the static check never descends into. Anything here is either
# not ours (a virtual environment, an installed package, a vendored tree) or
# not source (caches, build output). AUDIT TEST-RUNNER-03.
# A check that fails, times out, or cannot be found at all. MISSING belongs
# here: a configured check with no script behind it protects nothing.
# AUDIT TEST-RUNNER-02.
FAILING_STATUSES = ("FAILED", "TIMEOUT", "MISSING")

SKIP_DIRS = frozenset((
    "__pycache__", ".git", ".hg", ".svn", ".tox", ".mypy_cache", ".pytest_cache",
    "venv", ".venv", "env", ".env", "site-packages", "dist-packages",
    "node_modules", "build", "dist", ".eggs", "work", "runs",
))


def static_check():
    """Every .py parses.  In memory, so a read-only checkout still passes."""
    bad = []
    # AUDIT TEST-RUNNER-03. The exclusion list held only __pycache__ and .git,
    # so the walk descended into any virtual environment sitting in the
    # checkout and tried to compile third-party source. A syntax error in
    # somebody else's package is not a failure of this repository, and it is
    # what made the last static check fail on PyTorch's own files.
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                   and not d.endswith(".egg-info")]
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
    """The line that states the outcome, not whatever was printed last.

    unittest ends on a bare OK or on FAILED (errors=n), and several of the
    other scripts end on a numpy warning, so neither the first nor the last
    line is reliable.  Take unittest's summary when it is there, and otherwise
    the last line that reads like a conclusion.
    """
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln == "OK" or ln.startswith("OK (") or ln.startswith("FAILED ("):
            for prev in reversed(lines):
                if prev.startswith("Ran ") and "test" in prev:
                    return "%s  %s" % (ln, prev)
            return ln[:66]
    for ln in reversed(lines):
        low = ln.lower()
        if any(w in low for w in ("passed", "failed", "accepts", "checked",
                                  "error", "traceback")):
            return ln[:66]
    return lines[-1][:66] if lines else "(no output)"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow physics group")
    ap.add_argument("--group", action="append", choices=GROUPS,
                    help="run only these groups, repeatable")
    ap.add_argument("--window", action="store_true",
                    help="also run the window test, which needs a display")
    ap.add_argument("--report", default=None,
                    help="write the whole report to this file as well")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="seconds allowed per check, default 1800")
    a = ap.parse_args()

    want = set(a.group or GROUPS)
    if a.quick:
        want -= SLOW_GROUPS

    log = []

    def say(s=""):
        print(s)
        log.append(s)

    say("=" * 78)
    say("PRT-DeepONet-3D   FULL TEST RUN")
    say("no training, no simulation campaign")
    say("=" * 78)
    say()

    env_lines, missing, absent = env_report()
    if "env" in want:
        say("ENVIRONMENT")
        for ln in env_lines:
            say("  " + ln)
        say()
    if missing:
        say("STOP. These packages are missing: %s" % ", ".join(missing))
        say("Install them with:  %s" % INSTALL)
        say("Nothing else can be trusted until they are present.")
        if a.report:
            open(a.report, "w").write("\n".join(log) + "\n")
        return 2

    rows = []      # (name, group, status, verdict, seconds, output, protects, fix)

    if "static" in want:
        t0 = time.time()
        bad = static_check()
        rows.append(("every file parses", "static",
                     "FAILED" if bad else "ok",
                     bad[0][:66] if bad else "all .py files compile",
                     time.time() - t0, "\n".join(bad),
                     "a syntax error anywhere in the tree, including a bad "
                     "merge or a half-applied edit",
                     "The first line of the output names the file and the "
                     "line. Nothing else in this report means anything until "
                     "it parses."))

    for name, group, cmd, protects, fix in CHECKS:
        if group not in want:
            continue
        if not os.path.exists(cmd[0]):
            rows.append((name, group, "MISSING", "not found: %s" % cmd[0],
                         0.0, "", protects,
                         "This file is not in the checkout. Either the copy is "
                         "incomplete or the file was renamed."))
            continue
        t0 = time.time()
        try:
            p = subprocess.run([PY] + cmd, capture_output=True, text=True,
                               cwd=HERE, timeout=a.timeout)
            out = (p.stdout or "") + (p.stderr or "")
            # AUDIT TEST-01. Exit status 2 means a check could not run for
            # want of a fixture. That is a skip, not a failure and not a pass,
            # and the output says what to build.
            status = ("ok" if p.returncode == 0
                      else "skipped" if p.returncode == 2 else "FAILED")
            v = verdict(out)
        except subprocess.TimeoutExpired:
            out, status = "", "TIMEOUT"
            v = "no result in %d s" % a.timeout
        rows.append((name, group, status, v, time.time() - t0, out, protects,
                     fix))

    if a.window and "gui" in want:
        wid = os.path.join(GUI, "test_gui_widgets.py")
        t0 = time.time()
        p = subprocess.run([PY, wid], capture_output=True, text=True, cwd=HERE)
        out = (p.stdout or "") + (p.stderr or "")
        no_display = ("No module named 'tkinter'" in out
                      or "no display" in out.lower())
        rows.append(("the window itself", "gui",
                     "skipped" if no_display else
                     ("ok" if p.returncode == 0 else "FAILED"),
                     "no display or no tkinter here" if no_display
                     else verdict(out),
                     time.time() - t0, out,
                     "every page of the window building, and every view mode "
                     "drawing",
                     "Needs tkinter and a display. On a headless machine run "
                     "it under xvfb-run, or skip it: nothing in the analysis "
                     "path depends on the window."))

    # ---------------------------------------------------------------- results
    say("RESULTS")
    w = max([len(r[0]) for r in rows] + [10])
    for name, group, status, v, secs, _, _, _ in rows:
        say("  %-*s  %-9s %-8s %6.1fs  %s"
            % (w, name, group, status, secs, v))
    say()

    by_group = {}
    for _, g, status, *_ in rows:
        d = by_group.setdefault(g, {"ok": 0, "bad": 0, "other": 0})
        d["ok" if status == "ok" else
          ("bad" if status in FAILING_STATUSES else "other")] += 1
    say("BY GROUP")
    for g in GROUPS:
        if g in by_group:
            d = by_group[g]
            say("  %-9s  %d passed, %d failed, %d skipped or missing"
                % (g, d["ok"], d["bad"], d["other"]))
    say()

    # AUDIT TEST-RUNNER-02. MISSING rows were created when a configured check
    # had no script behind it, and then dropped from the failure set, so a
    # test that had been deleted or renamed was reported and still counted as
    # a clean run. A configured check that cannot be found is a failure.
    bad = [r for r in rows if r[2] in FAILING_STATUSES]
    for i, (name, group, status, v, secs, out, protects, fix) in enumerate(bad, 1):
        say("=" * 78)
        say("FAILURE %d of %d: %s   [%s, %s]" % (i, len(bad), name, group,
                                                 status))
        say("=" * 78)
        say("WHAT IT PROTECTS")
        say("  " + protects)
        say()
        say("WHAT TO DO")
        say("  " + fix)
        say()
        say("OUTPUT, last 60 lines")
        tail = [ln for ln in out.strip().splitlines()][-60:]
        for ln in tail:
            say("  | " + ln[:150])
        say()

    skipped = [r for r in rows if r[2] == "skipped"]
    say("=" * 78)
    if bad:
        say("VERDICT: %d of %d checks FAILED. Not ready to deploy."
            % (len(bad), len(rows)))
        say()
        say("ORDER TO FIX THEM IN")
        order = ["static", "env", "model", "data", "pipeline", "units",
                 "switches", "flow", "physics", "gui"]
        n = 1
        for g in order:
            for r in bad:
                if r[1] == g:
                    say("  %d. %s   (%s)" % (n, r[0], g))
                    n += 1
        say()
        say("Fix them from the top. A failure in static or model makes every "
            "later group unreliable, so a green run of those first is worth "
            "more than chasing the longest error message.")
    else:
        say("VERDICT: everything that could run, passed."
            if skipped else "VERDICT: everything passed.")
        say()
        say("This checkout is ready to deploy. What that means and does not "
            "mean:")
        say("  it means  the code imports, the shapes agree, the switches are "
            "off when off, the")
        say("            buttons send commands the scripts accept, and "
            "evaluate and predict run")
        say("            end to end on a checkpoint")
        say("  it does not mean anything about accuracy. Train, then read the "
            "held-out error and")
        say("            the speedup, which is what the decision metrics are "
            "about.")
        if absent:
            say()
            say("Optional packages not installed here:")
            for pkg, why in absent:
                say("  %-14s %s" % (pkg, why))
        if skipped:
            say()
            say("Skipped or missing, with the reason in the table above:")
            for r in skipped:
                say("  %s" % r[0])
    say("=" * 78)

    if a.report:
        with open(a.report, "w") as fh:
            fh.write("\n".join(log) + "\n")
        print("report written to %s" % a.report)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
