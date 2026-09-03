#!/usr/bin/env python3
"""
test_01_smoke.py — does every piece of this project start at all?

WHAT A SMOKE TEST IS
    The cheapest possible question, asked of everything: does it turn on? It
    proves nothing about whether the answers are right. It catches the faults
    that make every other test meaningless: a file with a syntax error, an
    import that no longer exists, a script that crashes before it reads its
    first argument.

    It is called a smoke test because the original version was plugging in a
    new circuit board and watching for smoke.

WHAT IS CHECKED HERE
    1. every Python file in the project compiles
    2. every library module imports
    3. every script with a command line answers --help without running anything
    4. the window builds, with the display faked

NOTHING IS SIMULATED AND NOTHING IS WRITTEN.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, TOOLS, MODEL, GUI, BRIDGE, run     # noqa: E402


# =============================================================================
#  BLOCK 1.  WHAT IS BEING SMOKE TESTED
#
#  Two lists, written out by hand rather than discovered by walking the folder.
#  That is deliberate: a script ADDED to the project and forgotten shows up here
#  as a missing entry when somebody notices, rather than being silently exempt
#  from every check in this file because nobody listed it.
# =============================================================================
# Scripts that take a command line. Each must answer --help and exit 0.
CLI_SCRIPTS = [
    (TOOLS, "build_geometry_3d.py"), (TOOLS, "build_toy_pack_2d.py"),
    (TOOLS, "build_dataset_2d.py"), (TOOLS, "build_dataset_3d.py"),
    (TOOLS, "build_practice_dataset.py"), (TOOLS, "make_demo_complab.py"),
    (TOOLS, "complab_campaign.py"), (TOOLS, "collect_complab_output.py"),
    (TOOLS, "collect_foreign_complab.py"), (TOOLS, "import_2d_simulations.py"),
    (TOOLS, "build_transfer_set_2d_to_3d.py"),
    (TOOLS, "load_pretrained_2d_weights.py"), (TOOLS, "settings_and_units.py"),
    (TOOLS, "flow_features.py"), (TOOLS, "harmonic_pressure.py"),
    (TOOLS, "add_flow_features.py"), (TOOLS, "flow_coordinates.py"),
    (MODEL, "train.py"), (MODEL, "evaluate.py"), (MODEL, "predict.py"),
    (MODEL, "run_ablation_sweep.py"), (MODEL, "train_velocity.py"),
    (MODEL, "predict_velocity.py"), (TOOLS, "test_documented_numbers.py"),
    (GUI, "install_requirements.py"),
    (BRIDGE, "build_transfer_set.py"),
]

# Modules that are libraries. Importing them must not run anything.
LIBRARIES = [
    "flow_features", "harmonic_pressure", "flow_coordinates",
    "dataset_reader", "settings_and_units",
    "deeponet_model", "velocity_model",
]


# =============================================================================
#  BLOCK 2.  DOES IT PARSE, DOES IT IMPORT
#
#  The two cheapest questions, and the two that make every later test
#  meaningless if the answer is no.
# =============================================================================
class Compiles(unittest.TestCase):
    """Every .py file in the project is valid Python."""

    def test_every_file_compiles(self):
        import py_compile
        bad = []
        for folder in (ROOT, TOOLS, MODEL, GUI, BRIDGE,
                       os.path.join(ROOT, "tests")):
            if not os.path.isdir(folder):
                continue
            for f in sorted(os.listdir(folder)):
                if not f.endswith(".py"):
                    continue
                p = os.path.join(folder, f)
                try:
                    py_compile.compile(p, doraise=True)
                except Exception as e:                         # noqa: BLE001
                    bad.append("%s: %s" % (os.path.relpath(p, ROOT), e))
        self.assertEqual(bad, [], "these files do not compile:\n  "
                                  + "\n  ".join(bad))


class Imports(unittest.TestCase):
    """Every library module imports cleanly."""

    def test_libraries_import(self):
        bad = []
        for name in LIBRARIES:
            try:
                __import__(name)
            except Exception as e:                             # noqa: BLE001
                bad.append("%s: %s: %s" % (name, type(e).__name__, e))
        self.assertEqual(bad, [], "these modules do not import:\n  "
                                  + "\n  ".join(bad))

    def test_importing_does_not_run_anything(self):
        """A library that DID something on import would make every test here
        slow and unpredictable, and would run a simulation the moment somebody
        opened a notebook. Checked by timing: an import that takes longer than
        a few seconds is doing work."""
        import time
        slow = []
        for name in LIBRARIES:
            if name in sys.modules:
                del sys.modules[name]
            t0 = time.time()
            __import__(name)
            dt = time.time() - t0
            if dt > 20.0:
                slow.append("%s took %.1f s to import" % (name, dt))
        self.assertEqual(slow, [], "\n  ".join(slow))


# =============================================================================
#  BLOCK 3.  DOES IT REACH ITS ARGUMENTS
#
#  --help is the whole test. A script that crashes on import, or on a module
#  that has moved, never prints a usage line, and this catches it in a second
#  rather than three seconds into somebody's overnight run.
# =============================================================================
class Help(unittest.TestCase):
    """Every script answers --help and exits 0, having done nothing."""

    def test_every_script_answers_help(self):
        bad = []
        for folder, name in CLI_SCRIPTS:
            p = os.path.join(folder, name)
            if not os.path.exists(p):
                bad.append("%s is missing" % name)
                continue
                # --help, never a real run: argparse prints and exits before the
            # script does anything at all.
            rc, out = run([sys.executable, p, "--help"], timeout=180)
            if rc != 0:
                bad.append("%s --help exited %s: %s"
                           % (name, rc, out.strip()[-200:]))
            elif "usage" not in out.lower():
                bad.append("%s --help printed no usage line" % name)
        self.assertEqual(bad, [], "\n  ".join(bad))


# =============================================================================
#  BLOCK 4.  THE WINDOW, WITH NO SCREEN
#
#  tkinter is replaced with stubs, so this runs on a server. It proves the
#  catalogue can be built and nothing more; whether the pages LOOK right is
#  test_gui_widgets.py's job and that one does need a display.
# =============================================================================
class Window(unittest.TestCase):
    """The window's catalogue builds with no display."""

    def test_the_window_builds_headless(self):
        # The command test already imports the window behind a fake tkinter.
        # Reusing its loader means one stub set, not two that can disagree.
        sys.path.insert(0, GUI)
        import test_gui_commands as T
        gui = T.load_gui_module()
        actions = gui.build_actions()
        self.assertGreater(len(actions), 15,
                           "the window should offer more than fifteen actions")
        for key, a in actions.items():
            self.assertTrue(a.title, "action %r has no title" % key)
            self.assertTrue(a.script, "action %r has no script" % key)

    def test_the_flow_panel_exists(self):
        sys.path.insert(0, GUI)
        import test_gui_commands as T
        gui = T.load_gui_module()
        self.assertTrue(hasattr(gui, "FlowPipelinePage"))
        self.assertEqual(len(gui.FLOW_STEPS), 5,
                         "the flow pipeline should have five steps")


if __name__ == "__main__":
    unittest.main(verbosity=2)
