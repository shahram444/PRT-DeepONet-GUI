#!/usr/bin/env python3
"""
test_gui_commands.py — check that every button in the window emits a command
line the script it calls will actually accept.

WHY THIS EXISTS
    Seven separate bugs in this window were all the same bug: the button sent
    a flag the script does not have, or omitted one it requires, or packed
    several values into a single token. None of them could be seen by reading
    the button. All of them appeared three seconds into a run, as an argparse
    error, after the user had filled in a page of settings.

    This asks each script's own parser -- not a copy of it, the real one --
    whether the command the button builds is acceptable. It therefore cannot
    drift out of date the way a hand-written list of flags would.

HOW
    Each script is imported with argparse patched so that ArgumentParser.
    parse_args does not run the script; it records the result and raises. The
    button's command is then fed to it. A SystemExit from argparse is a
    failure with the message argparse would have printed.

    Nothing is executed and nothing is written.

    python test_gui_commands.py
"""

import argparse
import importlib.util
import io
import os
import contextlib
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

os.environ.setdefault("MPLBACKEND", "Agg")


# ---------------------------------------------------------------------------
#  A headless stand-in for the Tk variables, so the action catalogue can be
#  built without a display. Only get/set are ever used by Field.
# ---------------------------------------------------------------------------
class FakeVar:
    def __init__(self, value=""):
        self._v = value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


def load_gui_module():
    """Import prt_gui with tkinter stubbed, and return it.

    Before stubbing anything, the heavy libraries the analysis scripts import
    are loaded for real. Otherwise they get imported LATER, while tkinter is a
    stub and module names are being reused, and they fail on that rather than
    on anything to do with the command line -- which this test would then have
    to report as a failure it cannot explain.
    """
    for name in ("numpy", "scipy", "h5py", "torch", "matplotlib", "skimage"):
        try:
            __import__(name)
        except Exception:                                      # noqa: BLE001
            pass
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:                                          # noqa: BLE001
        pass
    import types
    tk = types.ModuleType("tkinter")

    def _sv(value="", **kw):
        return FakeVar(value)

    def _bv(value=False, **kw):
        return FakeVar(bool(value))

    def _iv(value=0, **kw):
        return FakeVar(int(value))

    tk.StringVar = _sv
    tk.BooleanVar = _bv
    tk.IntVar = _iv
    tk.DoubleVar = _sv
    for name in ("Tk", "Frame", "Label", "Button", "Entry", "Text", "Canvas",
                 "Checkbutton", "Menu", "Toplevel", "Scrollbar", "Message",
                 "PanedWindow", "Listbox", "LabelFrame", "PhotoImage"):
        setattr(tk, name, type(name, (), {"__init__": lambda self, *a, **k: None,
                                          "__getattr__": lambda self, n: (lambda *a, **k: None)}))
    tk.END = "end"
    tk.TclError = Exception
    ttk = types.ModuleType("tkinter.ttk")
    # Scrollbar, Label and Button are here because the flow pipeline panel
    # builds a scrolling column of cards. A stub set that is missing one
    # widget fails as an AttributeError deep inside a constructor, which reads
    # like a bug in the window rather than a gap in this list.
    for name in ("Combobox", "Notebook", "Progressbar", "Style", "Treeview",
                 "Separator", "Frame", "Scrollbar", "Label", "Button",
                 "Checkbutton", "Entry"):
        setattr(ttk, name, type(name, (), {"__init__": lambda self, *a, **k: None,
                                           "__getattr__": lambda self, n: (lambda *a, **k: None)}))
    tk.ttk = ttk
    for sub in ("filedialog", "messagebox", "font", "scrolledtext",
                "simpledialog"):
        m = types.ModuleType("tkinter." + sub)
        m.__getattr__ = lambda n: (lambda *a, **k: None)
        sys.modules["tkinter." + sub] = m
        setattr(tk, sub, m)
    sys.modules["tkinter"] = tk
    sys.modules["tkinter.ttk"] = ttk

    spec = importlib.util.spec_from_file_location(
        "prt_gui_headless", os.path.join(HERE, "prt_gui.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
#  Ask a script's own parser whether it accepts an argument list.
# ---------------------------------------------------------------------------
class _Accepted(Exception):
    pass


_COUNT = [0]


def _parses_arguments(script):
    """Does this script define an argument parser at all?

    Read from the source rather than found by running it, because running it
    is the thing being avoided.
    """
    try:
        src = open(script, "r", errors="replace").read()
    except OSError:
        return False
    return "ArgumentParser" in src


def parser_accepts(script, argv):
    """Run `script`'s argparse against argv. Returns None if fine, else why not.

    The script is imported and its main() called with argparse patched: the
    first parse_args to complete raises _Accepted, so nothing after the parse
    ever runs.
    """
    real_parse = argparse.ArgumentParser.parse_args
    real_parse_known = argparse.ArgumentParser.parse_known_args
    # parse_args calls parse_known_args internally. Patching both naively means
    # the inner call raises _Accepted before parse_args gets to reject leftover
    # arguments -- so an invented flag would sail through. The depth counter
    # keeps the inner call real.
    depth = [0]

    reached = [False]

    def patched(self, args=None, namespace=None):
        reached[0] = True
        depth[0] += 1
        try:
            real_parse(self, args=argv, namespace=namespace)
        finally:
            depth[0] -= 1
        raise _Accepted()

    def patched_known(self, args=None, namespace=None):
        if depth[0]:
            return real_parse_known(self, args=args, namespace=namespace)
        reached[0] = True
        real_parse_known(self, args=argv, namespace=namespace)
        raise _Accepted()

    argparse.ArgumentParser.parse_args = patched
    argparse.ArgumentParser.parse_known_args = patched_known
    old_argv = sys.argv
    sys.argv = [script] + list(argv)
    try:
        # A fresh module name each time: re-executing a torch-importing
        # script under a name that is already in sys.modules trips torch's
        # own circular-import guard, which has nothing to do with arguments.
        _COUNT[0] += 1
        name = "_under_test_%d" % _COUNT[0]
        spec = importlib.util.spec_from_file_location(name, script)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                spec.loader.exec_module(mod)
                fn = getattr(mod, "main", None)
                if fn is None:
                    return "no main() to reach the parser through"
                fn()
        except _Accepted:
            return None
        except SystemExit:
            msg = buf.getvalue().strip().splitlines()
            return "argparse rejected it: " + (msg[-1] if msg else "(no message)")
        except Exception as e:                                 # noqa: BLE001
            # If the parser was reached, the parse passed and the script then
            # failed on something else -- a missing input file, most likely.
            # That is not this test's business.
            #
            # If it was NOT reached, the script blew up on import or before the
            # parse, and reporting that as a pass is exactly the silent-accept
            # bug this file exists to catch. Say so instead.
            if reached[0]:
                return None
            return ("never reached its argument parser: %s: %s"
                    % (type(e).__name__, e))
        if not reached[0]:
            return "never reached its argument parser (main() returned first)"
        return None
    finally:
        argparse.ArgumentParser.parse_args = real_parse
        argparse.ArgumentParser.parse_known_args = real_parse_known
        sys.argv = old_argv
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
#  What does the script REALLY expect after each flag?
# ---------------------------------------------------------------------------
def parser_shape(script):
    """Return {flag: nargs} for every option the script defines.

    nargs is normalised to 1, an integer above 1, "*", or the string "flag"
    for a switch that takes no value at all. Subparsers are walked, so
    complab_campaign.py's build/status/retry options are included.

    This exists because argparse cannot catch the commonest packing bug on its
    own: an option declared nargs="*" with no type accepts "a b c" as ONE
    string quite happily, and the mistake only shows up much later as a
    chemical named "Ac A P Bio". Comparing the two declarations catches it
    before it is ever run.
    """
    captured = []
    real_parse = argparse.ArgumentParser.parse_args
    real_known = argparse.ArgumentParser.parse_known_args

    def grab(self, args=None, namespace=None):
        captured.append(self)
        raise _Accepted()

    argparse.ArgumentParser.parse_args = grab
    argparse.ArgumentParser.parse_known_args = grab
    old_argv = sys.argv
    sys.argv = [script]
    _COUNT[0] += 1
    name = "_shape_%d" % _COUNT[0]
    try:
        spec = importlib.util.spec_from_file_location(name, script)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                spec.loader.exec_module(mod)
                fn = getattr(mod, "main", None)
                if fn is not None:
                    fn()
        except (_Accepted, SystemExit, Exception):             # noqa: BLE001
            pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse
        argparse.ArgumentParser.parse_known_args = real_known
        sys.argv = old_argv
        sys.modules.pop(name, None)

    if not captured:
        return None
    out = {}

    def walk(p):
        for act in p._actions:
            if isinstance(act, argparse._SubParsersAction):
                for sub in act.choices.values():
                    walk(sub)
                continue
            n = act.nargs
            if not act.option_strings:
                continue
            if act.const is not None and n == 0:
                shape = "flag"
            elif n is None:
                shape = 1
            elif n == 0:
                shape = "flag"
            elif n in ("*", "+"):
                shape = "*"
            elif n == "?":
                shape = 1
            else:
                shape = int(n)
            for f in act.option_strings:
                out[f] = shape
    walk(captured[0])
    return out


def nargs_mismatches(A):
    """Every field's declared nargs against the parser's.

    Returns (problems, deliberate_differences).
    """
    bad, notes = [], []
    shapes = {}
    for key in sorted(A):
        act = A[key]
        if not act.python or not os.path.exists(act.script):
            continue
        if not _parses_arguments(act.script):
            continue
        if act.script not in shapes:
            shapes[act.script] = parser_shape(act.script)
        sh = shapes[act.script]
        if not sh:
            bad.append("%s: could not read %s's parser at all"
                       % (key, os.path.basename(act.script)))
            continue
        for f in act.fields:
            if f.flag is None:
                continue
            if f.flag not in sh:
                bad.append("%s / %s: the script has no such option"
                           % (key, f.flag))
                continue
            want = sh[f.flag]
            if want == "flag":
                if f.kind != "bool":
                    bad.append("%s / %s: takes no value, but the field is a "
                               "'%s' box" % (key, f.flag, f.kind))
                continue
            if f.kind == "bool":
                bad.append("%s / %s: needs a value, but the field is a tickbox"
                           % (key, f.flag))
                continue
            if f.nargs == want:
                continue
            # A path picker deliberately sends ONE value even where the script
            # would accept a list. Splitting a folder box on spaces would
            # break every path like "C:\Users\me\My Documents\complab_campaign", which
            # is a much likelier mistake than someone typing two folders into
            # a box with a Browse button next to it.
            if want == "*" and f.nargs == 1 and f.kind in ("file", "savefile",
                                                           "dir"):
                notes.append("%s / %s: the script accepts several, the window "
                             "offers one folder. Deliberate -- splitting on "
                             "spaces would break paths that contain spaces."
                             % (key, f.flag))
                continue
            bad.append("%s / %s: the script wants %s value(s), the field "
                       "sends %s -- several values would arrive as one "
                       "token" % (key, f.flag, want, f.nargs))
    return bad, notes


# ---------------------------------------------------------------------------
#  Does the check still have teeth?
# ---------------------------------------------------------------------------
def sensitivity(A):
    """Feed the checker commands that are known to be wrong, and one that is
    known to be right.

    A test that always passes is worse than no test, because it is trusted.
    Every entry below is one of the bugs this file was written after; if the
    checker stops catching them it says so here instead of printing a
    reassuring green line. The last entry is the control: a correct command
    that must NOT be flagged, which catches the opposite failure of a checker
    that rejects everything.
    """
    cases = [
        ("complab_campaign.py with no subcommand", "complab_campaign",
         ["--geometries", "/tmp/g", "--out", "/tmp/c", "--batches", "5"], True),
        ("complab_campaign.py build with no --complab", "complab_campaign",
         ["build", "--geometries", "/tmp/g", "--out", "/tmp/c"], True),
        ("--grid \"148 64\" packed into one token", "heewon",
         ["--checkpoint", "/tmp/a.pt", "--grid", "148 64",
          "--save", "/tmp/w.pt"], True),
        ("a flag the script does not have", "train",
         ["--data", "/tmp/x.h5", "--out", "/tmp/o", "--nonsense", "1"], True),
        # NOTE there is deliberately no case here for a text field with
        # nargs="*" packed into one token -- "--compare 'a=1.pt b=2.pt'".
        # argparse cannot see that one: a list of strings happily accepts a
        # single long string. It is caught instead by comparing each field's
        # declared nargs against the parser's, below.
        ("a correct command  (the control -- must pass)", "train",
         ["--data", "/tmp/x.h5", "--out", "/tmp/o", "--epochs", "3"], False),
    ]
    blind = []
    print("Sensitivity -- can this check still see a broken command?")
    for name, key, argv, should_fail in cases:
        if key not in A:
            continue
        why = parser_accepts(A[key].script, argv)
        ok = bool(why) == should_fail
        print("  %-6s %s" % ("ok" if ok else "BLIND", name))
        if not ok:
            blind.append(name)
    print()
    return blind


def _dep_keys(dep):
    """Every field key a dependency is guarded by.

    A dependency is normally one (key, wanted value) pair, but it may also be a
    LIST of such pairs when a box is shown only if two things are true at once
    -- the biotic reaction is on AND the user asked for a range rather than
    exact numbers. Reading only dep[0] finds the key in the first case and a
    whole tuple in the second, which silently stops the choice box being
    recognised as a gate, and the combinations behind it stop being checked.
    """
    if not dep:
        return set()
    if isinstance(dep, str):
        return {dep}
    if (isinstance(dep, (tuple, list)) and dep
            and all(isinstance(d, (tuple, list)) for d in dep)):
        out = set()
        for d in dep:
            out |= _dep_keys(d)
        return out
    if isinstance(dep, (tuple, list)) and dep:
        return {dep[0]}
    return set()


# ---------------------------------------------------------------------------
#  The flow pipeline panel
# ---------------------------------------------------------------------------
def flow_pipeline(gui, A):
    """The five commands the flow pipeline panel emits, against the real parsers.

    The panel is a Tk widget, so nothing inside it can be reached here. That is
    why its command building lives in flow_step_command(), a free function that
    takes plain values: this can call it with no display and check what comes
    out, which is the whole reason it was pulled out of the class.

    Two things are checked, not one. First that each command parses, the same
    question asked of every button. Second that the pipeline is COHERENT: that
    every step points at the same dataset, that step 5 reads the field step 4
    was told to write, and that step 4 was in fact told to write it. A step
    that parses perfectly and reads the wrong file is the failure this pipeline
    is most likely to have, and argparse cannot see it.
    """
    fails = []
    data = "/tmp/x.h5"
    out = "/tmp/flowrun"
    cmds = {}
    print("The flow pipeline panel")
    for key in [s[4] for s in gui.FLOW_STEPS]:
        try:
            _a, argv = gui.flow_step_command(A, key, data=data, out=out,
                                             buffer="10", condition="pe",
                                             epochs="3", geom=True)
        except Exception as e:                                 # noqa: BLE001
            fails.append((key, "", "could not build the command: %s" % e))
            print("  FAIL   %-18s could not build the command" % key)
            continue
        cmds[key] = argv[3:]
        why = parser_accepts(A[key].script, argv[3:])
        print("  %-6s %-18s %s" % ("ok" if not why else "FAIL", key,
                                   " ".join(argv[3:])[:90]))
        if why:
            fails.append((key, " ".join(argv[3:]), why))

    def has(key, flag, value=None):
        a = cmds.get(key, [])
        if flag not in a:
            return False
        if value is None:
            return True
        i = a.index(flag)
        return i + 1 < len(a) and a[i + 1] == value

    checks = [
        ("every step reads the same dataset",
         all(has(k, "--data", data) for k in cmds)),
        ("step 4 is told to write the field back",
         has("predict_velocity", "--write-back")),
        ("step 4 reads the checkpoint step 3 writes",
         has("predict_velocity", "--checkpoint",
             os.path.join(out, "velocity_operator", "best.pt"))),
        ("step 2 asks for the simulated field",
         has("train_flow_sim", "--velocity-informed", "simulated")),
        ("step 5 asks for the predicted field",
         has("train_flow_pred", "--velocity-informed", "predicted")),
        ("the two training steps write to different folders",
         (cmds.get("train_flow_sim", []) and cmds.get("train_flow_pred", [])
          and cmds["train_flow_sim"][cmds["train_flow_sim"].index("--out") + 1]
          != cmds["train_flow_pred"][cmds["train_flow_pred"].index("--out") + 1])),
        ("the pore size maps reach both training steps",
         has("train_flow_sim", "--geom-features")
         and has("train_flow_pred", "--geom-features")),
    ]
    for name, ok in checks:
        print("  %-6s %s" % ("ok" if ok else "FAIL", name))
        if not ok:
            fails.append(("pipeline", name, "the steps do not line up"))
    print()
    return fails


# ---------------------------------------------------------------------------
def main():
    gui = load_gui_module()
    A = gui.build_actions()

    fails, checked, skipped = [], 0, []

    # Fields left blank in the catalogue would be rejected as missing rather
    # than as malformed, which is a different test. Fill required blanks with
    # something shaped right, so what is being tested is the FLAGS.
    filler = {"file": "/tmp/x.h5", "savefile": "/tmp/x.h5", "dir": "/tmp/x",
              "int": "1", "float": "1.0", "text": "x", "files": "/tmp/x"}

    for act in A.values():
        for f in act.fields:
            if f.var is None:
                f.make_var()

    for key in sorted(A):
        act = A[key]
        if not act.python:
            skipped.append((key, "not a python script"))
            continue
        if not os.path.exists(act.script):
            skipped.append((key, "script not found: %s" % act.script))
            continue
        # A script with no argument parser has nothing to check here, and
        # calling its main() would RUN it. For the "check everything" button
        # that means this test runs every test in the project, including
        # itself -- which is not hypothetical: it spawned ten nested processes
        # the first time the loop closed.
        if not _parses_arguments(act.script):
            argv = act.command()[3:]
            if argv:
                fails.append((key, " ".join(argv),
                              "the script parses no arguments, but the button "
                              "sends some"))
            skipped.append((key, "takes no arguments; nothing to check"))
            continue

        # every combination of the choice fields that gate other fields
        gates = [f for f in act.fields
                 if f.kind in ("choice", "subcommand")
                 and any(f.key in _dep_keys(g.depends) for g in act.fields)]

        combos = [{}]
        for g in gates:
            combos = [dict(c, **{g.key: v}) for c in combos for v in g.choices]

        for combo in combos:
            saved = {}
            for f in act.fields:
                saved[f.key] = f.var.get()
                if f.key in combo:
                    f.var.set(combo[f.key])
                elif f.kind != "bool" and not str(f.var.get()).strip():
                    if not f.optional:
                        f.var.set(filler.get(f.kind, "x"))
            argv = act.command()[3:]          # drop [python, -u, script]
            why = parser_accepts(act.script, argv)
            checked += 1
            label = key + (("  [%s]" % ", ".join("%s=%s" % kv
                                                 for kv in combo.items()))
                           if combo else "")
            if why:
                fails.append((label, " ".join(argv), why))
            for f in act.fields:
                f.var.set(saved[f.key])

    # ---------------------------------------------------------- multi-value
    narg_fails, narg_notes = nargs_mismatches(A)
    flow_fails = flow_pipeline(gui, A)
    fails += flow_fails

    print("=" * 72)
    print("GUI COMMAND CHECK")
    print("=" * 72)
    blind = sensitivity(A)
    print("%d command lines checked against the real parsers" % checked)
    for k, why in skipped:
        print("  skipped %-14s %s" % (k, why))
    print()
    if narg_fails:
        print("MULTI-VALUE FIELDS THAT WOULD BE SENT AS ONE TOKEN")
        for m in narg_fails:
            print("  " + m)
        print()
    if narg_notes:
        print("Deliberate differences, checked and accepted:")
        for m in narg_notes:
            print("  " + m)
        print()
    if fails:
        print("REJECTED")
        for label, cmd, why in fails:
            print("  %s" % label)
            print("      %s" % cmd)
            print("      %s" % why)
        print()
        print("FAILED: %d of %d" % (len(fails), checked))
        return 1
    if blind or narg_fails:
        print("FAILED: the check itself is blind to %d known-bad command(s)"
              % len(blind))
        return 1
    print("Every button emits a command its script accepts.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                          # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
