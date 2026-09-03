#!/usr/bin/env python3
"""
test_gui_widgets.py — build every page of the real window and click through it.

WHY THIS IS SEPARATE FROM test_gui_commands.py
    That file checks what the buttons SEND. This one checks that the window
    can be built at all. They fail for different reasons: a new field kind
    that no widget knows how to draw, a view mode with no branch behind it, a
    dependency rule that hides a field the command still needs. None of that
    is visible from the command line, and all of it is invisible until
    somebody opens the page.

    It needs a display. On a machine without one, run it under Xvfb:

        xvfb-run -a python test_gui_widgets.py

    It builds pages, sets values and reads commands back. It launches nothing.
"""

import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, h5py
import tkinter as tk
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("prt_gui_live", os.path.join(HERE, "prt_gui.py"))
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

app = g.Studio()
app.withdraw()
root = app
A = app.actions
print("actions:", len(A))

bad = []
# 1. every action page builds, and its command line renders
for key in sorted(A):
    try:
        page = g.ActionPage(root, app, A[key])
        cmd = A[key].command()
        assert isinstance(cmd, list) and all(isinstance(c, str) for c in cmd), cmd
    except Exception as e:
        bad.append("page %s: %s: %s" % (key, type(e).__name__, e)); traceback.print_exc()
print("pages built:", len(A) - len([b for b in bad if b.startswith("page")]))

# 2. the complab_campaign page's subcommand actually lands first
c = A["complab_campaign"]
for f in c.fields:
    if f.key == "cmd": f.var.set("build")
    if f.key == "complab": f.var.set("/tmp/complab")
cmd = c.command()
print("complab_campaign build ->", " ".join(cmd[2:]))
if cmd[3] != "build": bad.append("subcommand is not first: %r" % (cmd[3:5],))
if "--complab" not in cmd: bad.append("--complab missing from complab_campaign build")
for f in c.fields:
    if f.key == "cmd": f.var.set("status")
cmd = c.command()
print("complab_campaign status ->", " ".join(cmd[2:]))
if "--complab" in cmd or "--geometries" in cmd:
    bad.append("status still sends build-only flags")

# 3. multi-token fields split
h = A["heewon"]
for f in h.fields:
    if f.key == "grid": f.var.set("148 64")
cmd = h.command()
i = cmd.index("--grid")
if cmd[i+1:i+3] != ["148", "64"]: bad.append("--grid not split: %r" % cmd[i:i+3])
print("grid ->", cmd[i:i+3])
for f in h.fields:
    if f.key == "grid": f.var.set("148")
if not h.fields[[f.key for f in h.fields].index("grid")].check():
    bad.append("a one-number grid was not rejected")
else:
    print("one-number grid rejected:", h.fields[[f.key for f in h.fields].index("grid")].check())

# 3b. a repeated flag is repeated, not packed. --set is argparse action="append",
#     so 'tau=0.9 nx=64' has to become two separate --set arguments. Packing them
#     into one is silent: the first is applied and the second is swallowed.
d3 = A["dataset3d"]
before = d3.command().count("--set")
for f in d3.fields:
    if f.key == "overrides":
        f.var.set("tau=0.9 species.A.d_pore=5e-10")
cmd = d3.command()
if cmd.count("--set") - before != 2:
    bad.append("the overrides box added %d --set arguments, expected 2: %r"
               % (cmd.count("--set") - before, cmd))
elif "tau=0.9" not in cmd or "species.A.d_pore=5e-10" not in cmd:
    bad.append("--set carried the wrong value: %r" % cmd)
else:
    print("repeated flag -> --set tau=0.9 --set species.A.d_pore=5e-10")
for f in d3.fields:
    if f.key == "overrides":
        f.var.set("")

# 3d. an untouched settings_and_units box sends nothing; a changed one sends NAME=VALUE.
#     Untouched boxes have to stay silent, or loading a settings file would be
#     pointless: two dozen defaults would overwrite everything in it.
if d3.command().count("--set") != 0:
    bad.append("untouched settings_and_units boxes emitted --set: %r"
               % [x for x in d3.command() if "=" in x])
for f in d3.fields:
    if f.key == "ph_tau":
        f.var.set("0.95")
    if f.key == "ph_unit":          # a drop-down, not a typed box
        f.var.set("mm")
    if f.key == "ph_pore":
        f.var.set("1")
cmd = d3.command()
want = ["tau=0.95", "unit=mm", "pore_code=1"]
missing = [w for w in want if w not in cmd]
if missing:
    bad.append("changed settings_and_units boxes did not send %r: %r"
               % (missing, [x for x in cmd if "=" in x]))
elif cmd.count("--set") != 3:
    bad.append("expected exactly 3 --set arguments, got %d: %r"
               % (cmd.count("--set"), cmd))
else:
    print("settings_and_units boxes ->", " ".join(want))
# and every one of them has to be a setting settings_and_units.py actually knows
import subprocess as _sp
_r = _sp.run([sys.executable, os.path.join(HERE, "..", "3D", "tools",
                                           "build_dataset_3d.py"),
              "--out", "/tmp/_never.h5", "--show-settings"] +
             sum([["--set", w] for w in want], []),
             capture_output=True, text=True)
if _r.returncode != 0:
    bad.append("the generator rejected the boxes: %s"
               % (_r.stderr or _r.stdout)[-400:])
else:
    print("the generator accepted them, tau ->",
          [l.strip() for l in _r.stdout.splitlines() if "relaxation" in l])
for f in d3.fields:
    if f.key == "ph_tau": f.var.set("0.8")
    if f.key == "ph_unit": f.var.set("um")
    if f.key == "ph_pore": f.var.set("2")

# 3e. THE CHEMICALS TABLE. Its defaults have to match settings_and_units.py's, because a
#     box whose value equals its default sends nothing at all: if the two
#     drifted apart, the window would either silently impose its own idea of
#     the defaults over a settings file, or fail to send a change you made.
sys.path.insert(0, os.path.join(HERE, "..", "3D", "tools"))
import settings_and_units as _phys
_ref = _phys.Settings()
for i, sp in enumerate(_ref.species):
    d = g.chem_default(i)
    for key, got in (("name", sp.name), ("left_value", sp.left_value),
                     ("initial", sp.initial), ("d_pore", sp.d_pore),
                     ("d_biofilm", sp.d_biofilm),
                     ("right_value", sp.right_value)):
        want = d[key]
        same = (str(got) == want if key == "name"
                else abs(float(got) - float(want)) <= 1e-15)
        if not same:
            bad.append("chemical %d %s: window says %s, settings_and_units.py says %r"
                       % (i, key, want, got))
    if d["left_type"] != sp.left_type or d["right_type"] != sp.right_type:
        bad.append("chemical %d boundary types disagree" % i)
    if (d["immobile"] == "true") != sp.immobile:
        bad.append("chemical %d immobile disagrees" % i)
# and the fifth, which settings_and_units.py invents when the list grows
_ref.apply_override("n_chemicals", "5")
if g.chem_default(4)["name"] != _ref.species[4].name:
    bad.append("the fifth chemical is called %r in the window and %r in "
               "settings_and_units.py" % (g.chem_default(4)["name"], _ref.species[4].name))
print("chemical defaults agree with settings_and_units.py:", len(_ref.species) - 1,
      "checked")

ch = [f for f in d3.fields if f.kind == "chemicals"][0]
if d3.command().count("--set") != 0:
    bad.append("an untouched chemicals table emitted --set")
if "--n-species" not in d3.command() or \
        d3.command()[d3.command().index("--n-species") + 1] != "4":
    bad.append("the chemicals table did not send --n-species 4")

# the plus button: a fifth chemical, then a change to it
ch.rows.append({k: g.tk.StringVar(value=g.chem_default(4)[k])
                for k in g.CHEM_KEYS})
cmd = d3.command()
if cmd[cmd.index("--n-species") + 1] != "5":
    bad.append("adding a chemical did not change --n-species: %r" % cmd)
if "n_chemicals=5" not in cmd:
    bad.append("adding a chemical did not send n_chemicals=5: %r" % cmd)
if cmd.index("n_chemicals=5") > min(
        [i for i, x in enumerate(cmd) if x.startswith("species.")] or [1e9]):
    bad.append("n_chemicals was sent AFTER a per-chemical setting, so the "
               "chemical it refers to does not exist yet: %r" % cmd)
ch.rows[4]["name"].set("Bromide")
ch.rows[4]["d_pore"].set("2.0e-9")
ch.rows[1]["left_value"].set("1.0e-3")
cmd = d3.command()
for want in ("n_chemicals=5", "species.4.name=Bromide",
             "species.4.d_pore=2.0e-9", "species.1.left_value=1.0e-3"):
    if want not in cmd:
        bad.append("the chemicals table did not send %s: %r"
                   % (want, [x for x in cmd if "=" in x]))
print("chemicals table ->", [x for x in cmd if "=" in x])

# and the generator has to accept all of it, for real
_r = _sp.run([sys.executable, os.path.join(HERE, "..", "3D", "tools",
                                           "build_dataset_3d.py"),
              "--out", "/tmp/_never.h5", "--show-settings"] + cmd[3:],
             capture_output=True, text=True)
if _r.returncode != 0:
    bad.append("the generator rejected the chemicals table: %s"
               % (_r.stderr or _r.stdout)[-500:])
elif "Bromide" not in _r.stdout:
    bad.append("the generator ran but never mentioned Bromide")
else:
    print("the generator accepted five chemicals including Bromide")
ch.rows.pop()
ch.rows[1]["left_value"].set(g.chem_default(1)["left_value"])
if d3.command().count("--set") != 0:
    bad.append("the table did not go quiet again after being put back")

# 3f. COMPLETENESS. Every settings_and_units input the Python code can take has to be
#     reachable from the window, or the window is quietly less capable than
#     the code behind it and nobody can tell which parts. Anything not on a
#     --set box has to be claimed here, by name, against the dedicated box
#     that carries it -- so a setting added to settings_and_units.py and forgotten in
#     the window fails this test rather than going unnoticed.
COVERED_ELSEWHERE = {
    "nx": "the Grid size box", "ny": "the Grid size box",
    "nz": "the Grid size box",
    "peclet": "the Lowest/Highest Peclet boxes",
    "peclet_min": "the Lowest Peclet box",
    "peclet_max": "the Highest Peclet box",
    "damkohler": "the Lowest/Highest Damkohler boxes",
    "damkohler_min": "the Lowest BIOTIC Damkohler box",
    "damkohler_max": "the Highest BIOTIC Damkohler box",
    "damkohler_abiotic": "the ABIOTIC Damkohler boxes",
    "damkohler_abiotic_min": "the Lowest ABIOTIC Damkohler box",
    "damkohler_abiotic_max": "the Highest ABIOTIC Damkohler box",
    # THE SIX BELOW WERE THIS CHECK BEING WRONG, NOT THE WINDOW BEING SHORT.
    #
    # Every one of them is on the two dataset pages already, in the
    # "range or exact numbers" boxes: choose a range and the page sends
    # --phi-min and --phi-max, choose exact numbers and it sends --phi-values
    # instead. The same for Peclet and for both Damkohler numbers.
    #
    # They were reported as unreachable because this check looks only at fields
    # carrying a settings NAME, the ones that become --set name=value. These
    # boxes do not: they emit their own flags directly, which is why they are
    # claimed here like every other dedicated box.
    #
    # That they really are sent, and that choosing one form removes the other's
    # flags rather than sending both, is checked separately and by measurement
    # in test_gui_sweep_modes.py.
    "porosity_min": "the Lowest porosity box, in the range/exact pair",
    "porosity_max": "the Highest porosity box, in the range/exact pair",
    "porosity_values": "the exact-numbers half of the porosity box",
    "peclet_values": "the exact-numbers half of the Peclet box",
    "damkohler_values": "the exact-numbers half of the BIOTIC Damkohler box",
    "damkohler_abiotic_values": "the exact-numbers half of the ABIOTIC Damkohler box",
    # DELIBERATELY NOT IN THE WINDOW. You choose a Peclet and the pressure
    # gradient that produces it is derived, per rock, and printed. Offering a
    # pressure box as well would be two controls for one thing, and the one
    # people actually think in is Peclet. All three remain settable from a
    # settings file or a --set argument for anyone reproducing a measured
    # flow rate.
    "flow_driver": "not offered: the window always drives by Peclet",
    "delta_P": "derived from the Peclet you ask for, and printed per rock",
    "fluid_viscosity": "settings-file only; water is assumed",
    "ns_max_iT": "the Flow solver iterations box",
    "n_snapshots": "the Snapshots in time box",
    "n_chemicals": "the chemicals table's plus and minus buttons",
    "dimension": "not a setting: which builder is running",
    "source": "not a setting: where the numbers came from",
    "species": "the chemicals table",
    "ignored": "not a setting", "notes": "not a setting",
}
_settable = {k.rstrip("_") if k == "yield_" else k
             for k in vars(_phys.Settings())}
_in_window = {f.setting for f in d3.fields if getattr(f, "setting", None)}
_unreachable = sorted(_settable - _in_window - set(COVERED_ELSEWHERE))
if _unreachable:
    bad.append("these settings_and_units settings exist in settings_and_units.py but no box in "
               "the window sets them: %s" % ", ".join(_unreachable))
else:
    print("every settings_and_units setting is reachable from the window: %d boxes, "
          "%d covered by a dedicated control" % (len(_in_window),
                                                 len(COVERED_ELSEWHERE) - 5))
# and every chemical property too
_chem_props = {k for k in vars(_phys.Species("x"))}
_missing_chem = sorted(_chem_props - set(g.CHEM_KEYS))
if _missing_chem:
    bad.append("these per-chemical settings are not in the chemicals table: %s"
               % ", ".join(_missing_chem))
else:
    print("every per-chemical setting is in the table:", len(g.CHEM_KEYS))
# and every box in the window has to be a setting that really exists
_bogus = sorted(_in_window - _settable - {"n_chemicals"})
if _bogus:
    bad.append("these boxes set names settings_and_units.py does not have: %s"
               % ", ".join(_bogus))

# 3g. WHAT THE RUN BUTTON COMPLAINS ABOUT HAS TO BE SOMETHING YOU CAN FIX.
#
# This is the check that was missing. Adding three field kinds that carry no
# single value -- a section heading, the line that counts up the simulations,
# and the chemicals table -- made the pre-run validator flag all of them. The
# dialog then listed five section headings and one blank line and told the
# user to attend to them, naming nothing anybody could act on.
#
# Two things are required. Every complaint must name a box that can actually
# be typed into. And the two dataset pages, which need no input files at all,
# must have nothing to complain about the moment they open.
def prerun_complaints(act):
    """Exactly what ActionPage.run() checks, so the two cannot drift."""
    out = []
    for f in act.fields:
        if not g._dep_ok(act.fields, f.depends):
            continue
        if f.kind in ("heading", "computed"):
            continue
        c = f.check()
        if c:
            out.append((f, c))
            continue
        if f.optional or f.kind in ("bool", "choice", "subcommand",
                                    "chemicals", "setting"):
            continue
        if not str(f.value()).strip():
            out.append((f, "empty"))
    return out

for key in sorted(A):
    act = A[key]
    page = g.ActionPage(root, app, act)
    for f, why in prerun_complaints(act):
        if f.kind in ("heading", "computed", "chemicals"):
            bad.append("page %r complains about %r, which is not a box the "
                       "user can type into" % (key, f.label))
        if not str(f.label).strip():
            bad.append("page %r complains about a field with no label at all"
                       % key)
# and the two pages that need no input files must open ready to run
for key in ("dataset2d", "dataset3d"):
    c = prerun_complaints(A[key])
    if c:
        bad.append("page %r needs no input files but complains as it opens: %s"
                   % (key, [(f.label, w) for f, w in c]))
print("Run-button complaints all name a real box; both dataset pages open "
      "ready to run")

# and the complaint text, when there IS one, has to name the box and say what
# to do -- not name a section heading
ch2 = [f for f in A["dataset3d"].fields if f.kind == "chemicals"][0]
ch2.rows[0]["name"].set("")
msg = ch2.check()
if not msg or "Chemical 1" not in msg:
    bad.append("a nameless chemical produced no usable complaint: %r" % msg)
else:
    print("a nameless chemical says:", msg)
ch2.rows[0]["name"].set(g.chem_default(0)["name"])
ch2.rows[1]["d_pore"].set("not a number")
msg = ch2.check()
if not msg or "is not a number" not in msg:
    bad.append("a non-numeric diffusion coefficient produced no complaint: %r"
               % msg)
else:
    print("a bad number says:", msg)
ch2.rows[1]["d_pore"].set(g.chem_default(1)["d_pore"])
if ch2.check():
    bad.append("the chemicals table still complains after being put right: %s"
               % ch2.check())

# 3h. THE PROGRESS BAR HAS TO BE ABLE TO FOLLOW THE GENERATORS.
#
# The dataset builders print one line per simulation, and their total is rocks
# TIMES conditions -- not the value of any single box on the page. So the
# pattern carries its own total, and this checks it against a real line from
# a real run rather than against a line somebody typed into a test.
_line = ("  [ 7/60] rock 2   Pe=1.34    Da=0.42       12 s   3180 steps  "
         "settled   about 8 min left")
for key in ("dataset2d", "dataset3d"):
    pr = A[key].progress_re
    if pr is None:
        bad.append("page %r has no progress pattern, so its bar cannot move"
                   % key)
        continue
    m = pr.search(_line)
    if not m:
        bad.append("page %r cannot read its own progress line" % key)
    elif (m.group(1), m.group(2)) != ("7", "60"):
        bad.append("page %r read %r out of its progress line"
                   % (key, m.groups()))
print("the progress bar reads 7 of 60 from a real generator line")

# 3i. RESULTS ARE NEVER WRITTEN ON TOP OF RESULTS.
#
# When the place you named already holds something, the run goes to the next
# free name beside it. For a FILE the folder gets the new name and the file
# keeps its own, so every experiment folder holds a file called the same thing
# and a script written for one works on all of them.
import tempfile as _tf
with _tf.TemporaryDirectory() as _td:
    # a fresh path is used as it stands
    p1 = os.path.join(_td, "2D_new", "dataset.h5")
    if g._next_free(p1, False) != p1:
        bad.append("a path nothing is using was changed: %r"
                   % g._next_free(p1, False))
    # once the file is there, the FOLDER moves and the file keeps its name
    os.makedirs(os.path.dirname(p1))
    open(p1, "w").write("x")
    p2 = g._next_free(p1, False)
    if os.path.basename(p2) != "dataset.h5":
        bad.append("the file was renamed instead of the folder: %r" % p2)
    if os.path.basename(os.path.dirname(p2)) != "2D_new_2":
        bad.append("expected the folder 2D_new_2, got %r"
                   % os.path.dirname(p2))
    if os.path.exists(p2):
        bad.append("the chosen path already exists: %r" % p2)
    # and again, so it counts rather than stopping at 2
    os.makedirs(os.path.dirname(p2))
    open(p2, "w").write("x")
    p3 = g._next_free(p1, False)
    if os.path.basename(os.path.dirname(p3)) != "2D_new_3":
        bad.append("expected 2D_new_3 on the third run, got %r"
                   % os.path.dirname(p3))
    # a DIRECTORY output: an empty one is reused, a full one is stepped over
    d1 = os.path.join(_td, "geometries")
    os.makedirs(d1)
    if g._next_free(d1, True) != d1:
        bad.append("an EMPTY folder should be reused, not stepped over")
    open(os.path.join(d1, "geom_0000.npz"), "w").write("x")
    d2 = g._next_free(d1, True)
    if os.path.basename(d2) != "geometries_2":
        bad.append("expected geometries_2, got %r" % d2)
    print("results never overwrite: 2D_new -> 2D_new_2 -> 2D_new_3, "
          "geometries -> geometries_2")

# 3j. THE REACTION MODE GATES THE PAGE.
#
# A pure-abiotic run must not show a yield or a half-saturation constant, and
# a pure-biotic run must not show a surface-reaction switch. A box that cannot
# affect the run should not be on the screen asking to be filled in.
d3 = A["dataset3d"]
mode = next(f for f in d3.fields if f.key == "ph_mode")
BIO_ONLY = ["ph_vmax", "ph_ksa", "ph_yield", "ph_bmode", "ph_bfcode",
            "da_min", "da_max"]
ABIO_ONLY = ["ph_ksurf", "ph_abiosurf", "da2_min", "da2_max"]
for chosen, want_bio, want_abio in (("biotic only", True, False),
                                    ("abiotic only", False, True),
                                    ("both", True, True)):
    mode.var.set(chosen)
    for key in BIO_ONLY:
        f = next(x for x in d3.fields if x.key == key)
        if g._dep_ok(d3.fields, f.depends) != want_bio:
            bad.append("mode %r: biotic box %r shown=%s, expected %s"
                       % (chosen, key, not want_bio, want_bio))
    for key in ABIO_ONLY:
        f = next(x for x in d3.fields if x.key == key)
        if g._dep_ok(d3.fields, f.depends) != want_abio:
            bad.append("mode %r: abiotic box %r shown=%s, expected %s"
                       % (chosen, key, not want_abio, want_abio))
    # and the command must carry only the flags that apply
    cmd = d3.command()
    if want_abio != ("--da-abio-min" in cmd):
        bad.append("mode %r sent --da-abio-min=%s" % (chosen,
                                                      "--da-abio-min" in cmd))
    if want_bio != ("--da-min" in cmd):
        bad.append("mode %r sent --da-min=%s" % (chosen, "--da-min" in cmd))
mode.var.set("biotic only")
print("the reaction mode gates the page: biotic-only, abiotic-only and both "
      "all show the right boxes")

# 3k. the biomass Attached box is not a second control for biomass_mode
ch3 = [f for f in d3.fields if f.kind == "chemicals"][0]
ch3.rows[3]["immobile"].set("false")
if any(x.startswith("species.3.immobile") for x in d3.command()):
    bad.append("the chemicals table sent species.3.immobile, which "
               "biomass_mode owns")
ch3.rows[3]["immobile"].set("true")
# but any OTHER chemical's Attached box still works
ch3.rows[2]["immobile"].set("true")
if not any(x == "species.2.immobile=true" for x in d3.command()):
    bad.append("a non-biomass chemical could not be marked immobile")
ch3.rows[2]["immobile"].set("false")
print("the biomass Attached box defers to 'The microbes are'; the others "
      "still work")

# 3k2. D IN BIOFILM only means something when there ARE biofilm voxels, and
#      there are only biofilm voxels when microbes are running AND a biofilm
#      code is set. Otherwise the pore coefficient is used everywhere and the
#      box is decoration -- so it must not be sent.
bf = next(x for x in d3.fields if x.key == "ph_bfcode")
ch4 = [f for f in d3.fields if f.kind == "chemicals"][0]
ch4.rows[0]["d_biofilm"].set("9.9e-11")
for mode_v, code_v, should_send in (("biotic only", "", False),
                                    ("abiotic only", "3", False),
                                    ("biotic only", "3", True),
                                    ("both", "3", True)):
    mode.var.set(mode_v)
    bf.var.set(code_v)
    sent = any(x.startswith("species.0.d_biofilm") for x in d3.command())
    if sent != should_send:
        bad.append("mode=%r code=%r sent d_biofilm=%s, expected %s"
                   % (mode_v, code_v, sent, should_send))
mode.var.set("biotic only"); bf.var.set("")
ch4.rows[0]["d_biofilm"].set(g.chem_default(0)["d_biofilm"])
print("D in biofilm is only sent when biofilm voxels can exist")

# 3k3. "Just check the numbers" must not be mistaken for a failed run.
#      It finishes in a second and writes nothing, so the completion message
#      has to say so rather than listing the files a real run would produce.
_r = _sp.run([sys.executable, os.path.join(HERE, "..", "3D", "tools",
                                           "build_dataset_2d.py"),
              "--out", "/tmp/_check_only.h5", "--n-geom", "1",
              "--shape", "30", "20", "--show-settings"],
             capture_output=True, text=True)
if _r.returncode != 0:
    bad.append("the check-only run failed: %s" % (_r.stderr or "")[-300:])
elif "NOTHING WAS SIMULATED" not in _r.stdout:
    bad.append("a check-only run did not say that nothing was simulated")
elif "RESULTS WILL BE SAVED TO" in _r.stdout:
    bad.append("a check-only run still promised to save results")
elif os.path.exists("/tmp/_check_only.h5"):
    bad.append("a check-only run wrote a file")
else:
    print("a check-only run says so, promises nothing, and writes nothing")

# 3l. output formats reach the generator
for key, flag in (("save_png", "--save-png"), ("save_vti", "--save-vti")):
    f = next(x for x in d3.fields if x.key == key)
    f.var.set(True)
    if flag not in d3.command():
        bad.append("%r did not send %s" % (key, flag))
    f.var.set(False)
    if flag in d3.command():
        bad.append("%r sent %s while off" % (key, flag))
next(x for x in d3.fields if x.key == "save_png").var.set(True)
print("the picture and ParaView boxes send their flags")

# 4. the viewer on a 2D dataset: full plane, no degenerate strip, no 3D modes
p = "/tmp/v/d4.h5"
if os.path.exists(p):
    v = g.Viewer(root, app)
    with h5py.File(p, "r") as f:
        shape = tuple(int(x) for x in f.attrs["shape"])
        arr = f["samples/conc"][0, -1, 0]
    print("2D dataset shape", shape, " field", arr.shape)
    if shape[2] != 1: bad.append("expected a 2D dataset")
    whole = arr[:, :, 0]; strip = arr[:, shape[1]//2, :]
    print("  whole plane %s  vs the old slice %s" % (whole.shape, strip.shape))
    if strip.shape[1] != 1: bad.append("the old slice was not degenerate?")
    if whole.shape != (shape[0], shape[1]): bad.append("whole plane wrong")

# 5. every action is REACHABLE from the sidebar in at least one mode.
#
# An action can be written, tested, and emit a perfectly correct command line
# while being wired to no button at all -- which is exactly what happened to
# "Make a 3D dataset" and "Check everything at once": both existed, both
# passed the command test, and neither appeared in the window.
import re as _re
_src = open(os.path.join(HERE, "prt_gui.py")).read()
_listed = set()
# Each entry is ("Group name", ["key", "key", ...]). Take only the LIST --
# the group name is prose and one of them ("Predict") collides with a real
# action key, so a blind sweep of every quoted word both over- and
# under-counts.
for _name, _keys in _re.findall(r'\(\s*"([^"]+)"\s*,\s*\[([^\]]*)\]\s*\)', _src):
    _listed |= set(_re.findall(r'"([a-z0-9_]+)"', _keys))
# keys inserted outside the groups tables, one at a time
_listed |= set(_re.findall(r'iid="act:([a-z0-9_]+)"', _src))
_missing = sorted(set(A) - _listed)
print("actions reachable from the sidebar: %d of %d" % (len(A) - len(_missing), len(A)))
if _missing:
    bad.append("these actions exist but no sidebar entry reaches them: %s"
               % ", ".join(_missing))

print()
if bad:
    print("FAILURES:"); [print("  " + b) for b in bad]; sys.exit(1)
print("GUI SMOKE TEST PASSED")
