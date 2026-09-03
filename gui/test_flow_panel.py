#!/usr/bin/env python3
# =============================================================================
# NEW IN THE FLOW VERSION.  There is no 2D counterpart of this file.
#
#   WHAT IT TESTS
#     The flow pipeline panel: the five steps it offers, the commands they
#     emit, and what it does when one of them fails.
#
#   WHY IT EXISTS
#     test_gui_commands.py asks every BUTTON whether the script it calls would
#     accept its command. The pipeline panel is not a button. It is a sequence,
#     and a sequence has a second way to be wrong that argparse cannot see:
#     each command can be perfectly valid on its own while step 4 writes to one
#     place and step 5 reads from another. Nothing about that is a parse error.
#     It is a wasted afternoon.
#
#     So this file checks the WIRING. Same dataset everywhere. Step 4 told to
#     write the field back. Step 5 reading the file step 4 wrote. The two
#     training runs landing in different folders so they can be compared
#     afterwards instead of one quietly replacing the other.
#
#   AND WHAT HAPPENS WHEN A STEP FAILS
#     Carrying on would run step 5 against a field step 4 never wrote, and the
#     error the user then reads would be a missing HDF5 key rather than the
#     real failure several minutes earlier. The panel is supposed to stop. This
#     checks that it does, by failing a step on purpose.
#
#   HOW IT RUNS WITHOUT A SCREEN
#     tkinter is replaced with the same stubs test_gui_commands.py uses, and
#     the window itself with a stand-in that records what it was asked to
#     launch instead of launching it. Nothing is executed. The only thing
#     written is an empty placeholder dataset in a temporary folder, which is
#     removed again at the end; the panel refuses to start a step until the
#     dataset box points at a file that exists, and without one every check
#     below would pass by doing nothing.
#
#       python test_flow_panel.py
# =============================================================================

import os
import shutil
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import test_gui_commands as T          # noqa: E402  (the tkinter stubs)


# A real empty file in a temporary folder, deleted at the end. The panel
# refuses to start anything until the dataset box points at something that
# EXISTS, which is the right behaviour and would otherwise make every
# sequencing check below pass by doing nothing at all. Nothing reads the
# file's contents, so nought bytes is enough.
TMP = tempfile.mkdtemp(prefix="prt_flow_panel_")
DATA = os.path.join(TMP, "dataset.h5")
OUT = os.path.join(TMP, "results")
open(DATA, "wb").close()


# =============================================================================
#  BLOCK 1.  A window that records instead of running
# =============================================================================
class RecordingApp:
    """Everything the panel asks of the window, and nothing else.

    `launch` is the important one. The real window starts a subprocess; this
    one writes down what it was asked to start and hands back control, so the
    test can then say what the exit code was and watch what the panel does
    next. That is what makes the sequencing testable at all.
    """

    def __init__(self, actions):
        self.actions = actions
        self.on_run_finished = None
        self.launched = []             # (action key, argv) in the order run
        self.logged = []
        self.stopped = 0

    def status(self, _s):
        pass

    def logline(self, text, _tag=None):
        self.logged.append(text)

    def launch(self, action, cmd, _total):
        self.launched.append((action.key, list(cmd)))

    def stop_run(self):
        self.stopped += 1

    def open_action(self, _key):
        pass

    # Not part of the window's interface. The test uses it to say "that run
    # has now ended, with this code", which is what the real Runner does.
    def finish(self, rc):
        cb, self.on_run_finished = self.on_run_finished, None
        if cb is not None:
            cb(rc)


_FAILS = []
_CHECKS = [0]


def check(name, ok, extra=""):
    _CHECKS[0] += 1
    print("  %-6s %-58s %s" % ("PASS" if ok else "FAIL", name, extra))
    if not ok:
        _FAILS.append(name)


def build(gui):
    """A panel, filled in, with a recording window behind it."""
    app = RecordingApp(gui.build_actions())
    page = gui.FlowPipelinePage.__new__(gui.FlowPipelinePage)
    gui.FlowPipelinePage.__init__(page, None, app)
    page.v_data.set(DATA)
    page.v_out.set(OUT)
    page.v_buffer.set("5")
    page.v_condition.set("pe")
    page.v_epochs.set("7")
    return app, page


def argv_of(gui, page, act_key):
    _a, cmd = page._fill(act_key)
    return cmd[3:]                     # drop [python, -u, script]


def flag(argv, name):
    """The value after a flag, True for a flag with no value, None if absent."""
    if name not in argv:
        return None
    i = argv.index(name)
    if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
        return argv[i + 1]
    return True


# =============================================================================
#  BLOCK 2.  The steps are the ones the panel says they are
# =============================================================================
def group_steps(gui, page):
    print("\n[1] the five steps")
    keys = [s[0] for s in gui.FLOW_STEPS]
    check("five steps, in order", keys == ["descriptors", "control",
                                           "operator", "predict", "flowmodel"],
          " ".join(keys))
    check("every step has a card on the panel",
          set(page.cards) == set(keys))
    check("every step names an action that exists",
          all(s[4] in page.app.actions for s in gui.FLOW_STEPS))
    # The order is an argument, not a preference. The control run costs one
    # training run and decides whether the other three are worth doing.
    ci = keys.index("control")
    oi = keys.index("operator")
    check("the cheap control comes BEFORE the velocity operator", ci < oi,
          "control at %d, operator at %d" % (ci + 1, oi + 1))


# =============================================================================
#  BLOCK 3.  The wiring between the steps
# =============================================================================
def group_wiring(gui, page):
    print("\n[2] the steps line up with each other")
    a = {k: argv_of(gui, page, s[4]) for k, s in
         zip([s[0] for s in gui.FLOW_STEPS], gui.FLOW_STEPS)}

    check("every step reads the same dataset",
          all(flag(v, "--data") == DATA for v in a.values()),
          DATA)
    check("the shared buffer reaches the steps that take one",
          all(flag(v, "--buffer") in (None, "5") for v in a.values()))
    check("the shared epoch count reaches all three training steps",
          all(flag(a[k], "--epochs") == "7"
              for k in ("control", "operator", "flowmodel")))
    check("step 2 asks for the SIMULATED field",
          flag(a["control"], "--velocity-informed") == "simulated")
    check("step 5 asks for the PREDICTED field",
          flag(a["flowmodel"], "--velocity-informed") == "predicted")
    # The one that costs a whole pipeline when it is wrong.
    check("step 4 is told to write the field back",
          flag(a["predict"], "--write-back") is True)
    check("step 4 reads the checkpoint step 3 writes",
          flag(a["predict"], "--checkpoint")
          == os.path.join(OUT, "velocity_operator", "best.pt"))
    check("the two training runs land in DIFFERENT folders",
          flag(a["control"], "--out") != flag(a["flowmodel"], "--out"),
          "%s  vs  %s" % (os.path.basename(flag(a["control"], "--out") or ""),
                          os.path.basename(flag(a["flowmodel"], "--out") or "")))
    # Predicting for a whole dataset and naming one flow condition are two
    # different requests, and sending both means the file's own recorded
    # condition is contradicted by a number left in a box.
    check("step 4 does not send a single-geometry flow condition as well",
          flag(a["predict"], "--pe") is None)
    check("the pore size maps are off unless asked for",
          flag(a["control"], "--geom-features") is None)
    page.v_geom.set(True)
    b = argv_of(gui, page, "train_flow_sim")
    check("and on when they are", flag(b, "--geom-features") is True)
    page.v_geom.set(False)


# =============================================================================
#  BLOCK 4.  The flow capability is no longer a box on the training page
# =============================================================================
def group_not_a_switch(gui, page):
    print("\n[3] the flow capability is a pipeline, not a switch")
    train = page.app.actions["train"]
    keys = [f.key for f in train.fields]
    flags = [f.flag for f in train.fields]
    check("no velocity box left on the training page", "SW_D" not in keys)
    check("no --velocity-informed on the training page",
          "--velocity-informed" not in flags)
    check("no --geom-features on the training page",
          "--geom-features" not in flags)
    # It has to still be REACHABLE, or it has been removed rather than moved.
    check("the flag still reaches train.py through the pipeline",
          flag(argv_of(gui, page, "train_flow_pred"), "--velocity-informed")
          == "predicted")
    # And switches A, B and C must be exactly where they were.
    for sw in ("SW_A", "SW_B", "SW_C"):
        check("switch %s is untouched on the training page" % sw[-1],
              sw in keys)


# =============================================================================
#  BLOCK 5.  Running the sequence
# =============================================================================
def group_sequence(gui, page):
    print("\n[4] running all five, and stopping when one fails")
    app = page.app

    app.launched = []
    page.run_all()
    check("'run the whole pipeline' starts step 1 and nothing else",
          len(app.launched) == 1 and app.launched[0][0] == "flow_features",
          app.launched[0][0] if app.launched else "nothing started")

    for expect in ("train_flow_sim", "train_velocity", "predict_velocity",
                   "train_flow_pred"):
        app.finish(0)
        ok = app.launched and app.launched[-1][0] == expect
        check("after a success, the next step is %s" % expect, bool(ok),
              app.launched[-1][0] if app.launched else "nothing started")
    app.finish(0)
    check("all five ran, in order", len(app.launched) == 5,
          " -> ".join(k for k, _ in app.launched))
    check("and nothing runs after the last one",
          page.queue == [] and page.running is None)

    # The one that matters. A failure must not be followed by a step that
    # reads what the failed one was supposed to write.
    app.launched = []
    app.logged = []
    page.run_all()
    app.finish(0)                      # step 1 fine
    app.finish(1)                      # step 2 FAILS
    check("a failed step stops the pipeline", len(app.launched) == 2,
          " -> ".join(k for k, _ in app.launched))
    check("and the panel says why it stopped",
          any("stopped here" in t for t in app.logged))
    check("the failed step is marked, not left saying 'running'",
          page.running is None)

    # A single step is a single step. Finishing it must not start the rest.
    app.launched = []
    page.run_one("operator")
    check("'run this step' starts exactly one step", len(app.launched) == 1,
          app.launched[0][0] if app.launched else "nothing started")
    app.finish(0)
    check("and finishing it starts nothing else", len(app.launched) == 1)

    # Unticking a step leaves it out, rather than running it anyway.
    app.launched = []
    page.cards["descriptors"]["do"].set(False)
    page.run_all()
    check("an unticked step is skipped",
          app.launched and app.launched[0][0] == "train_flow_sim",
          app.launched[0][0] if app.launched else "nothing started")
    page.cards["descriptors"]["do"].set(True)
    page.queue = []
    page.running = None


# =============================================================================
def main():
    gui = T.load_gui_module()
    # os.path.exists is real here, and the panel refuses to start a step whose
    # script is missing. The scripts ARE present in a checkout, so nothing is
    # faked; if one is genuinely absent the sequence test says so loudly.
    _app, page = build(gui)

    print("=" * 72)
    print("THE FLOW PIPELINE PANEL")
    print("=" * 72)
    print("Nothing is executed. The window is replaced by a stand-in that")
    print("records what it was asked to run. The only thing written is an")
    print("empty placeholder dataset in a temporary folder, removed at the end:")
    print("  " + TMP)

    group_steps(gui, page)
    group_wiring(gui, page)
    group_not_a_switch(gui, page)
    group_sequence(gui, page)

    print("\n" + "-" * 72)
    if _FAILS:
        print("FAILED: %d of %d" % (len(_FAILS), _CHECKS[0]))
        for f in _FAILS:
            print("   " + f)
        return 1
    print("All %d checks passed." % _CHECKS[0])
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        rc = 2
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(rc)
