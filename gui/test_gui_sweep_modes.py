#!/usr/bin/env python3
"""Both branches of the range-or-exact-numbers boxes emit a command the
generator accepts.

test_gui_commands.py checks every button once, with every box at its default.
The porosity, Peclet and Damkohler boxes now have two states each, and only the
default one -- 'a range' -- is covered by that check. This file flips each of
them to 'exact numbers' and checks the other half.

It also checks the thing argparse cannot see: that switching to exact numbers
actually REMOVES the range flags from the command, rather than sending both and
letting one silently win.

    python test_gui_sweep_modes.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from test_gui_commands import load_gui_module, parser_accepts     # noqa: E402

# =============================================================================
#  BLOCK 1.  WHAT IS BEING CHECKED
#
#  Three boxes on the dataset pages each have two states: a RANGE (a minimum and
#  a maximum) or EXACT NUMBERS (a list). They are alternatives, and the command
#  must carry one form or the other, never both and never neither.
#
#  Each row below is (the mode box, the list box, the two range flags, the list
#  flag), so a fourth such box is one row here and no other change.
# =============================================================================
PAGES = ("build_dataset_3d.py", "build_dataset_2d.py")
MODES = (("phi_mode", "phi_values", ("--phi-min", "--phi-max"), "--phi-values"),
         ("pe_mode", "pe_values", ("--pe-min", "--pe-max"), "--pe-values"),
         ("da_mode", "da_values", ("--da-min", "--da-max"), "--da-values"))


def actions_with_modes(gui):
    out = []
    for key, act in sorted(gui.build_actions().items()):
        keys = {f.key for f in act.fields}
        if any(m[0] in keys for m in MODES):
            out.append((key, act))
    return out


# =============================================================================
#  BLOCK 2.  DRIVING A PAGE THAT WAS NEVER DRAWN
#
#  The catalogue is built with tkinter stubbed, so a field has no variable until
#  something makes one. The stand-in below is the smallest object Field.value()
#  will accept: get and set, nothing else.
# =============================================================================
def set_field(act, key, value):
    f = next((x for x in act.fields if x.key == key), None)
    if f is None:
        return False
    if getattr(f, "var", None) is None:
        f.var = type("V", (), {"_v": value,
                               "get": lambda s: s._v,
                               "set": lambda s, v: setattr(s, "_v", v)})()
    f.var.set(value)
    return True


def main():
    gui = load_gui_module()
    acts = actions_with_modes(gui)
    # Zero pages is a pass, not a failure: it means no page has such a box.
    if not acts:
        print("no page has a range-or-exact-numbers box; nothing to check")
        return 0

    # =========================================================================
    # BLOCK 3.  BOTH HALVES OF EVERY BOX, ON EVERY PAGE THAT HAS ONE
    #
    # Four assertions per state, and the last two are the ones argparse cannot
    # make. Switching to exact numbers must REMOVE --phi-min and --phi-max, not
    # merely add --phi-values beside them: a command carrying both parses
    # perfectly and then silently ignores one of them.
    # =========================================================================
    bad = []
    checked = 0
    for key, act in acts:
        script = act.script
        for mode_key, val_key, range_flags, list_flag in MODES:
            if not any(f.key == mode_key for f in act.fields):
                continue
            for choice in ("a range", "exact numbers"):
                set_field(act, mode_key, choice)
                argv = [str(t) for t in act.command()]
                # act.command() returns the whole line, interpreter and script
                # included. The parser wants only what comes after the script.
                for n, tok in enumerate(argv):
                    if tok.endswith(".py"):
                        argv = argv[n + 1:]
                        break
                checked += 1
                want_list = choice == "exact numbers"
                has_list = list_flag in argv
                has_range = any(r in argv for r in range_flags)
                where = "%s / %s = %r" % (key, mode_key, choice)
                if want_list and not has_list:
                    bad.append("%s: %s is missing" % (where, list_flag))
                if want_list and has_range:
                    bad.append("%s: %s sent as well as %s; only one of them "
                               "can be meant" % (where, range_flags[0], list_flag))
                if not want_list and has_list:
                    bad.append("%s: %s sent although a range was chosen"
                               % (where, list_flag))
                if not want_list and not has_range:
                    bad.append("%s: %s is missing" % (where, range_flags[0]))
                why = parser_accepts(act.script, argv)
                if why:
                    bad.append("%s: %s" % (where, why))
            # Put it back. The action objects are shared across this loop, so a
            # box left in the other state would change the next page's command.
            set_field(act, mode_key, "a range")

    print("=" * 68)
    print("SWEEP MODE CHECK")
    print("=" * 68)
    print("%d command lines checked, over %d page(s)" % (checked, len(acts)))
    if bad:
        print()
        for b in bad:
            print("  FAILED  " + b)
        return 1
    print()
    print("Both halves of every range-or-exact-numbers box emit a command the")
    print("generator accepts, and neither half leaks the other one's flags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
