#!/usr/bin/env python3
# =============================================================================
# prt_gui.py — PRT-DeepONet Studio
#
# One window for the whole project: 2D and 3D, geometry generation, simulation
# campaigns, dataset building, training, evaluation, prediction, and a viewer
# for the fields that come out.
#
# DESIGN RULES, because two of the three people using this are reactive
# transport scientists and not machine learning engineers:
#
#   1. Every action shows THREE boxes before it runs -- WHAT GOES IN, WHAT
#      HAPPENS, WHAT COMES OUT.  Nothing runs without the user having been told
#      what it will do and what files will appear.
#   2. Every technical term is given twice: once in transport language, once in
#      machine learning language.  "Branch CNN" is always followed by "(the part
#      that looks at the pore structure)".
#   3. Nothing is hidden behind a command line.  The exact command is shown in a
#      box and can be copied, so the GUI is a teacher rather than a black box.
#   4. Errors are never silent.  Anything that fails turns the notification
#      light red and is repeated in the Problems tab with the failing command.
#
# Standard library plus numpy, matplotlib, h5py.  No web stack, no Qt.
# Run it with:   python prt_gui.py
# =============================================================================

import os
import queue
import re
import subprocess
import sys
import threading
import time
import webbrowser

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, font as tkfont
except ImportError:
    sys.exit("This program needs Tk, which normally ships with Python.\n"
             "On Linux install it with:   sudo apt install python3-tk\n"
             "On Windows and macOS, reinstall Python from python.org and make\n"
             "sure 'tcl/tk and IDLE' is ticked in the installer.")

APP = "PRT-DeepONet Studio"
VERSION = "1.0"
SETTINGS = os.path.join(os.path.expanduser("~"), ".prt_deeponet_studio.json")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # the PRT-DeepONet mother folder
DIR_2D = os.path.join(ROOT, "2D")
DIR_3D = os.path.join(ROOT, "3D")
DIR_BRIDGE = os.path.join(ROOT, "bridge")

# Everything you PRODUCE lives under work/, one folder per experiment, and
# everything the code needs to read lives outside it. The reason is not
# tidiness: a checkpoint is meaningless without the dataset it was trained on,
# because the grid size, the chemical names and the parameter ranges all have
# to match, and evaluate.py refuses when they do not. Keeping dataset, model,
# figures and predictions in ONE folder per experiment makes that pairing
# impossible to lose -- and it stops a second run quietly overwriting the
# figures of the first, which is a mistake that has already been made here.
WORK = os.path.join(ROOT, "work")
NEW2D = os.path.join(WORK, "2D_new")
NEW3D = os.path.join(WORK, "3D_new")
TOOLS = os.path.join(DIR_3D, "tools")
MODEL = os.path.join(DIR_3D, "model")


# --------------------------------------------------------------- interpreter --
def find_python():
    """Which Python runs the analysis scripts.

    Prefers a virtual environment sitting inside 3D/, because that is where the
    project's own packages -- torch above all -- are usually installed.  The
    Python running this window is often a different one, and if we used it the
    scripts would fail on 'No module named torch' with no obvious cause.
    """
    cands = []
    for venv in (os.path.join(DIR_3D, ".venv"), os.path.join(ROOT, ".venv"),
                 os.path.join(DIR_3D, "venv"), os.path.join(ROOT, "venv")):
        cands += [os.path.join(venv, "Scripts", "python.exe"),
                  os.path.join(venv, "bin", "python")]
    for c in cands:
        if os.path.exists(c):
            return c
    return sys.executable


PYTHON = find_python()


def set_python(p):
    global PYTHON
    PYTHON = p

# ------------------------------------------------------------------ styling --
BG = "#d6d3ce"          # window chrome, as in the reference screenshot
PANEL = "#ffffff"
LINE = "#9a9894"
INK = "#1a1a1a"
MUTE = "#5a5a5a"
HEAD = "#eef1f5"
OKC = "#1c7a3e"
WARNC = "#b07000"
ERRC = "#b3261e"
INFOC = "#1c5fa8"
SEL = "#c3d5ef"


def base_font(size=9, bold=False):
    fam = "DejaVu Sans"
    if sys.platform.startswith("win"):
        fam = "Segoe UI"
    elif sys.platform == "darwin":
        fam = "Helvetica"
    return (fam, size, "bold" if bold else "normal")


def mono_font(size=9):
    fam = "DejaVu Sans Mono"
    if sys.platform.startswith("win"):
        fam = "Consolas"
    elif sys.platform == "darwin":
        fam = "Menlo"
    return (fam, size)


# =============================================================================
#  FIELD  — one configurable input on an action page
# =============================================================================
class Field:
    """One input on an action page.

    kind:  'file' 'files' 'savefile' 'dir' 'int' 'float' 'text' 'choice' 'bool'
    label: what a transport scientist would call it
    help:  one sentence, plain language, saying what it is FOR
    """

    def __init__(self, key, label, kind="text", default="", help="",
                 choices=None, flag=None, filetypes=None, optional=False,
                 depends=None, nargs=1, setting=None):
        self.key = key
        self.label = label
        self.kind = kind
        self.default = default
        self.help = help
        self.choices = choices or []
        self.flag = flag                  # command-line flag, None = positional
        self.filetypes = filetypes or [("All files", "*.*")]
        self.optional = optional
        # How many command-line tokens this field contributes after its flag.
        # 1        -> the value is ONE token, spaces and all (a path, a name)
        # int > 1  -> exactly that many, split on whitespace ("--shape 148 64")
        # "*"      -> any number, split on whitespace ("--species Ac A P Bio")
        # This must mirror the receiving script's argparse `nargs`. Guessing it
        # from the field name is what previously sent "--grid '148 64'" as a
        # single token and "--species 'Ac A P Bio'" as one species called
        # "Ac A P Bio".
        self.nargs = nargs
        # For kind == "setting": the name of the physical quantity this box
        # sets, as settings_and_units.py knows it. The box emits --set NAME=VALUE, so one
        # ordinary input box carries one physical number, with its own label,
        # its own units and its own explanation -- rather than the user having
        # to know the settings file's vocabulary before they can type anything.
        self.setting = setting
        # 'depends' gates visibility. Either the key of a bool field, or a
        # (key, expected_value) pair for choice fields.
        self.depends = depends
        self.var = None

    def make_var(self):
        if self.kind == "chemicals":
            # A list of rows rather than one value. Built here so that the
            # headless command test, which never draws anything, still gets a
            # working field.
            self.rows = [{k: tk.StringVar(value=chem_default(i)[k])
                          for k in CHEM_KEYS}
                         for i in range(int(self.default or 4))]
            self.var = tk.StringVar(value="")
            return self.var
        if self.kind == "bool":
            self.var = tk.BooleanVar(value=bool(self.default))
        elif self.kind == "int":
            self.var = tk.StringVar(value=str(self.default))
        elif self.kind == "float":
            self.var = tk.StringVar(value=str(self.default))
        else:
            self.var = tk.StringVar(value=str(self.default))
        return self.var

    def value(self):
        if self.var is None:
            # The page this box belongs to has never been drawn, so there is
            # no variable behind it yet. Its declared default is the honest
            # answer, and raising here turns "the catalogue was inspected
            # before anything was drawn" into an unexplained crash.
            return bool(self.default) if self.kind == "bool" else self.default
        v = self.var.get()
        if self.kind == "bool":
            return bool(v)
        return v

    def args(self):
        """This field's contribution to the command line.

        A field with no flag contributes NOTHING. Such fields exist only to
        gate other fields -- the 'where do the structures come from' selector,
        for instance, chooses which of two argument groups is shown and is not
        itself an argument. Passing its label through as a positional argument
        would silently corrupt the command, which is exactly what it used to do.
        """
        v = self.value()
        # A subcommand is a bare word the script expects BEFORE any flag --
        # 'complab_campaign.py build', not 'complab_campaign.py --build'. Put such a field
        # first in the action's field list and it lands in the right place.
        if self.kind == "subcommand":
            s = str(v).strip().split()[0] if str(v).strip() else ""
            return [s] if s else []
        # A section label. It is not an input and contributes nothing; it
        # exists so a long page can be read in parts.
        if self.kind == "heading":
            return []
        # The chemicals table. It decides how many chemicals there are, so it
        # also carries --n-species; there is no separate box for that, because
        # two boxes controlling one number is how they end up disagreeing.
        #
        # n_chemicals goes FIRST. The per-chemical settings after it address
        # chemicals by position, and position five does not exist until the
        # list has been grown to hold it.
        if self.kind == "chemicals":
            rows = getattr(self, "rows", [])
            out = ["--n-species", str(len(rows))]
            if len(rows) != len(CHEM_DEFAULTS):
                out += ["--set", "n_chemicals=%d" % len(rows)]
            for i, row in enumerate(rows):
                base = chem_default(i)
                for key in CHEM_KEYS:
                    # The biomass's immobile flag belongs to biomass_mode, and
                    # sending both would be two settings racing for one field.
                    if key == "immobile" and i == 3:
                        continue
                    # Nor a coefficient that nothing can apply.
                    if key == "d_biofilm" and not self._biofilm_possible():
                        continue
                    val = str(row[key].get()).strip()
                    if val and val != str(base[key]).strip():
                        out += ["--set", "species.%d.%s=%s" % (i, key, val)]
            return out
        if self.kind == "bool":
            return [self.flag] if (v and self.flag) else []
        if self.flag is None:
            return []
        s = str(v).strip()
        if not s:
            return []
        # One physical quantity, named the way the settings file names it.
        # Keyed off `setting` rather than off `kind`, because some of these
        # boxes are drop-downs and some are typed -- and a drop-down that fell
        # through to the generic path sent "--set um" instead of
        # "--set unit=um", which argparse accepted and settings_and_units.py then
        # rejected as a setting with no equals sign in it.
        if self.setting:
            # Only what you actually CHANGED is sent. A page of two dozen
            # physical boxes that all sent their value would bury the three you
            # touched in a command line nobody reads, and would also silently
            # override a settings file you had just loaded. Untouched boxes say
            # nothing, so the file wins; touched boxes win over the file. The
            # run prints every number it ended up using before it starts.
            if s == str(self.default).strip():
                return []
            return [self.flag, "%s=%s" % (self.setting, s)]
        # A flag the script accepts SEVERAL TIMES, once per value, rather than
        # once with several values. '--set tau=0.9 --set nx=64', not
        # '--set tau=0.9 nx=64'. The difference is argparse's action="append"
        # against nargs, and getting it the wrong way round is silent: the
        # first override is applied and the rest are swallowed.
        if self.kind == "repeated":
            out = []
            for tok in s.split():
                out += [self.flag, tok]
            return out
        if self.kind in ("files",):
            return [self.flag] + s.split()
        # A multi-value argument. The receiving script asked for several
        # tokens, so hand it several tokens.
        if self.nargs == "*" or (isinstance(self.nargs, int) and self.nargs > 1):
            return [self.flag] + s.split()
        return [self.flag, s]

    def _sibling(self, key, default=""):
        """What another box on the same page says, or the default.

        A field whose variable has not been created yet -- because the page
        has never been drawn -- answers with the default rather than raising.
        The command test builds the catalogue without ever drawing a page.
        """
        for x in getattr(self, "siblings", []) or []:
            if x.key == key:
                if x.var is None:
                    return str(x.default).strip()
                return str(x.value()).strip()
        return default

    def _biofilm_possible(self):
        """Can any voxel in this run be biofilm? If not, D in biofilm is
        decoration and must not be sent."""
        return (self._sibling("ph_mode", "biotic only") != "abiotic only"
                and bool(self._sibling("ph_bfcode")))

    def check(self):
        """Complain, in the user's language, if the value cannot be right.

        Returns a message, or None when the field is fine. Catching a wrong
        token count here is far kinder than letting argparse reject it three
        seconds into a run.
        """
        if self.kind == "chemicals":
            for i, row in enumerate(getattr(self, "rows", [])):
                for key in ("left_value", "initial", "d_pore", "d_biofilm",
                            "right_value"):
                    v = str(row[key].get()).strip()
                    try:
                        float(v)
                    except ValueError:
                        return ("Chemical %d, %s: \"%s\" is not a number."
                                % (i + 1, CHEM_LABELS[key], v))
                if not str(row["name"].get()).strip():
                    return "Chemical %d has no name." % (i + 1)
            names = [str(r["name"].get()).strip()
                     for r in getattr(self, "rows", [])]
            dup = [n for n in set(names) if names.count(n) > 1]
            if dup:
                return ("Two chemicals are both called \"%s\"." % dup[0])
            return None
        if self.flag is None or self.kind in ("bool", "heading"):
            return None
        s = str(self.value()).strip()
        if not s:
            return None if self.optional else None
        if isinstance(self.nargs, int) and self.nargs > 1:
            n = len(s.split())
            if n != self.nargs:
                return ("%s needs %d numbers separated by spaces, but %d %s "
                        "given (\"%s\")." % (self.label, self.nargs, n,
                                             "was" if n == 1 else "were", s))
        return None




def _next_free(path, is_dir):
    """A name near `path` that nothing is using yet.

    One experiment per folder is the convention this project runs on: a
    trained network is meaningless without the dataset it was trained on, and
    keeping the two together is what stops them being separated. So when a
    folder already holds a result, the next run goes NEXT TO it rather than
    on top of it.

        work/2D_new/dataset.h5   ->  work/2D_new_2/dataset.h5
        work/geometries          ->  work/geometries_2

    For a file, the FOLDER is what gets a new name and the file keeps its own,
    so every experiment folder holds a file called the same thing and a script
    written for one works on all of them.
    """
    if is_dir:
        base, n = path.rstrip("\\/"), 2
        while os.path.isdir(path) and os.listdir(path):
            path = "%s_%d" % (base, n)
            n += 1
        return path
    if not os.path.exists(path):
        return path
    folder, name = os.path.split(os.path.abspath(path))
    base, n = folder, 2
    while os.path.exists(os.path.join(folder, name)):
        folder = "%s_%d" % (base, n)
        n += 1
    return os.path.join(folder, name)


def _dep_ok(fields, dep):
    """Is the field guarded by `dep` currently active?"""
    if not dep:
        return True
    key, want = (dep if isinstance(dep, (tuple, list)) else (dep, None))
    g = next((x for x in fields if x.key == key), None)
    if g is None:
        return True
    if want is None:
        return bool(g.value())
    # Several acceptable values, so a box can be shown for "biotic only" AND
    # for "both" without needing two dependency rules or a duplicate box.
    if isinstance(want, (tuple, list, set, frozenset)):
        return g.value() in want
    return g.value() == want


# =============================================================================
#  ACTION  — one thing the user can do
# =============================================================================
class Action:
    def __init__(self, key, title, one_line, script, fields, inputs_text,
                 happens_text, outputs_text, group="3D", python=True,
                 total_re=None, progress_re=None, metric_re=None, cwd=None,
                 note=None, danger=None):
        self.key = key
        self.title = title
        self.one_line = one_line
        self.script = script                  # absolute path
        self.fields = fields
        self.inputs_text = inputs_text        # list of (name, explanation)
        self.happens_text = happens_text      # list of strings
        self.outputs_text = outputs_text      # list of (file, explanation)
        self.group = group
        self.python = python
        self.progress_re = progress_re        # regex with group(1) = current step
        self.total_key = total_re             # field key holding the total
        self.metric_re = metric_re            # regex -> (x, train, test)
        self.cwd = cwd
        self.note = note
        self.danger = danger

    def command(self):
        cmd = [PYTHON, "-u", self.script] if self.python else [self.script]
        for f in self.fields:
            # A field sometimes has to know what ANOTHER box says: the
            # chemicals table must not send a biofilm diffusivity when no
            # biofilm can exist, and only the mode box and the biofilm-code
            # box know whether it can.
            f.siblings = self.fields
            if not _dep_ok(self.fields, f.depends):
                continue
            cmd += f.args()
        return cmd


# =============================================================================
#  THE ACTION CATALOGUE

# =============================================================================
#  ABOUT
# =============================================================================
# Attribution and licensing. The two-dimensional figures below are taken from
# that release's own README and LICENSE, not from memory: Yehoon Kim and Heewon
# Jung, Chungnam National University; GPL v3; Copyright (C) 2025 Jung Lab.
ABOUT_TEXT = """\
%s   version %s
A workbench for the geometry-aware PRT-DeepONet project.
Released 9 August 2026.


WHAT IT IS
    Pore-scale reactive transport is accurate and slow: one three-dimensional
    simulation takes hours. This program builds, trains and checks a neural
    operator that predicts the same fields in milliseconds, and wraps the whole
    workflow -- geometry, flow, transport, training, evaluation, prediction --
    behind buttons that state what goes in, what happens and what comes out
    before anything runs.

    No command is ever hidden. Every page shows the exact command line it will
    execute, and you can copy it and run it yourself.


THE THREE-DIMENSIONAL WORK, THE SIMULATION GENERATORS AND THIS APPLICATION

    Shahram Asgari          shahram.asgari@uga.edu
    Christof Meile          cmeile@uga.edu

    Meile Lab
    University of Georgia, Athens, Georgia, USA

    Covering the three-dimensional formulation, the geometry generator, the
    D2Q9 and D3Q19 lattice-Boltzmann flow solvers, the shared
    advection-diffusion-reaction solver, the CompLaB3D campaign and collection
    tools, the training, evaluation and prediction pipeline, the three
    switches, and this application.


THE TWO-DIMENSIONAL WORK

    PRT-DeepONet, on which the two-dimensional side of this project builds:

        Yehoon Kim          Chungnam National University
                            wnsla7323 <at> naver.com
        Heewon Jung         Chungnam National University
                            hjung <at> cnu.ac.kr

    Copyright (C) 2025 Jung Lab.
    Licensed under the GNU General Public License, version 3 or later.

    Their release is included here UNMODIFIED and is treated as read-only. It
    supplies 3,000 pore domains, trained weights for three reaction types
    (irreversible sorption, reversible sorption and Monod kinetics), and the
    notebooks that run them. No file in the 2D folder is written to by any part
    of this program: every reference to it in our code is a read, a file-open
    dialog, or a "show me this folder" menu item.

    Their architecture -- a convolutional branch for the geometry, a
    fully-connected branch for the dimensionless numbers, and a trunk taking
    position, time and the geodesic distance -- is the starting point for the
    three-dimensional extension. Where this project reuses their trained
    weights it does so by reading the published parameter files; it does not
    copy or modify their source.

    Full licence text: 2D/LICENSE.


LICENCE

    The two-dimensional release keeps its own licence. GPL v3 applies to
    everything in the 2D folder, unchanged, and is distributed with it.

    The three-dimensional code and this application are offered under a DUAL
    LICENCE:

        Academic, research and teaching use
            GNU General Public License, version 3 or later. Free to use,
            study, modify and redistribute, provided derivative work carries
            the same licence and stays open.

        Commercial and industrial use
            Not covered by the above. A separate licence is required; write to
            the addresses above.

    Two things are worth knowing now rather than discovering later. Dual
    licensing requires the agreement of every copyright holder, and work
    produced at a university is usually owned in part by the institution, so a
    commercial licence would go through the University of Georgia's research
    and innovation office rather than being granted informally. Separately,
    whether a model warm-started from GPL-licensed trained weights is itself a
    derivative work is an unsettled question. Neither point has been put to
    anyone qualified to answer it. Both should be, before this is distributed
    outside the group.


HOW TO CITE

    Cite the two-dimensional work as PRT-DeepONet, Kim and Jung, Chungnam
    National University, 2025. Cite the three-dimensional extension as this
    project, Asgari and Meile, Meile Lab, University of Georgia, 2026. If you
    use both, cite both: the geometry-aware three-dimensional method extends
    their architecture, it does not replace it.


VERIFYING THIS INSTALLATION

    Set up -> Check everything at once runs ten independent groups of checks,
    including both simulators against analytic answers: plug flow must leave
    exp(-Da) at the outlet, and a front must spread as wide as the exact erfc
    solution says.
    Tools -> Where is everything? lists every path this program uses.

    Project root:
      %s

    Python running the analysis scripts:
      %s
"""


# =============================================================================
#  ONE SHORT PHRASE PER SIDEBAR ENTRY
# =============================================================================
# The page itself explains what a button does at length. This is the four-word
# version, shown in the tree, so the list can be read top to bottom without
# opening anything.
TREE_HINTS = {
    "install":     "gets the software the scripts need",
    "capabilities": "what this can and cannot do vs CompLaB",
    "checkall":    "one minute, proves the machinery works",
    "dataset2d":   "your geometry, chemicals and rates, solved in 2D",
    "dataset3d":   "your geometry, chemicals and rates, solved in 3D",
    "geometry":    "makes pore structures, nothing solved yet",
    "complab_campaign":    "writes the input files for the cluster",
    "collect_complab_output":     "turns finished CompLaB runs into a dataset",
    "ingest2d":    "extrudes 2D structures into 3D blocks",
    "ingestsims":  "reads simulation output somebody sent you",
    "heewon":      "converts published weights into ours",
    "train":       "learns the surrogate from a dataset",
    "evaluate":    "scores it on structures it never saw",
    "sweep":       "trains every switch and compares them",
    "predict":     "answers about a structure never simulated",
}


# =============================================================================
#  THE CHEMICALS TABLE
# =============================================================================
# One block per chemical, with a plus and a minus button under them. The ORDER
# is the meaning: the first is the electron donor, the second the acceptor, the
# third the product, the fourth the biomass. Anything after that is transported
# properly -- fed, advected, diffused with its own coefficient -- but takes no
# part in any reaction, which is a conservative tracer and is a useful thing to
# add rather than a limitation to apologise for.
#
# The four defaults below have to match settings_and_units.py's DEFAULT_SPECIES exactly,
# because a box whose value equals its default sends nothing. If they drifted
# apart, the window would either silently override the settings file with its
# own idea of the defaults, or fail to send a change you had made. The widget
# test checks the two agree.
CHEM_ROLES = ["the electron donor", "the electron acceptor",
              "the product", "the biomass"]
CHEM_KEYS = ["name", "left_value", "initial", "d_pore", "d_biofilm",
             "left_type", "right_type", "right_value", "immobile"]
CHEM_LABELS = {
    "name":        "Name",
    "left_value":  "Fed at inlet (mol/L)",
    "initial":     "Present at t=0 (mol/L)",
    "d_pore":      "D in pore (m2/s)",
    "d_biofilm":   "D in biofilm (m2/s)",
    "left_type":   "Inlet face",
    "right_type":  "Outlet face",
    "right_value": "Outlet value (mol/L)",
    "immobile":    "Attached (never moves)",
}
CHEM_CHOICES = {"left_type": ["Dirichlet", "Neumann"],
                "right_type": ["Dirichlet", "Neumann"],
                "immobile": ["false", "true"]}
CHEM_DEFAULTS = [
    dict(name="Ac",  left_value="2.0e-3", initial="0.0", d_pore="1.0e-9",
         d_biofilm="5.0e-10", left_type="Dirichlet", right_type="Neumann",
         right_value="0.0", immobile="false"),
    dict(name="A",   left_value="5.0e-3", initial="0.0", d_pore="1.0e-9",
         d_biofilm="5.0e-10", left_type="Dirichlet", right_type="Neumann",
         right_value="0.0", immobile="false"),
    dict(name="P",   left_value="0.0",    initial="0.0", d_pore="1.0e-9",
         d_biofilm="5.0e-10", left_type="Dirichlet", right_type="Neumann",
         right_value="0.0", immobile="false"),
    dict(name="Bio", left_value="0.0",    initial="2.0e-4", d_pore="0.0",
         d_biofilm="0.0",    left_type="Neumann",   right_type="Neumann",
         right_value="0.0", immobile="true"),
]


def chem_default(i):
    """What chemical i looks like before you touch it.

    Past the fourth this has to agree with what settings_and_units.py invents when
    n_chemicals grows, or every new chemical would arrive carrying nine
    redundant --set arguments restating its own defaults.
    """
    if i < len(CHEM_DEFAULTS):
        return dict(CHEM_DEFAULTS[i])
    return dict(name="Tracer%d" % (i - 3), left_value="1.0e-3", initial="0.0",
                d_pore="1.0e-9", d_biofilm="5.0e-10", left_type="Dirichlet",
                right_type="Neumann", right_value="0.0", immobile="false")


# =============================================================================
#  THE PHYSICAL INPUT BOXES
# =============================================================================
# One box per physical quantity, each with its own label, its own units and its
# own sentence of explanation. Every one of them emits --set NAME=VALUE, which
# is the same route the settings file and the editable block in the generator
# use, so there is exactly one place where a number becomes physics.
#
# They are addressed by POSITION, not by name: species.0 is the electron donor,
# 1 the acceptor, 2 the product, 3 the biomass. The reaction network is fixed
# by that order, and a settings file loaded from somebody else's CompLaB.xml
# will not call them Ac and A -- so boxes that referred to them by name would
# stop working the moment you loaded one.
def physical_fields():
    S = lambda key, label, setting, default, help, **kw: Field(
        key, label, kw.pop("kind", "setting"), default, help,
        flag="--set", setting=setting, **kw)
    BIO = ("ph_mode", ("biotic only", "both"))      # shown for these modes
    ABIO = ("ph_mode", ("abiotic only", "both"))
    return [
        # ---------------------------------------------------------------
        #  THE FIRST DECISION ON THE PAGE. Everything below is shown or
        #  hidden by it, so a pure-abiotic run never shows a yield or a
        #  half-saturation constant and a pure-biotic run never shows a
        #  surface-reaction switch. A box that cannot affect the run should
        #  not be on the screen asking to be filled in.
        # ---------------------------------------------------------------
        Field("HD_mode", "1.  What chemistry are you running", "heading", "",
              "Choose this first. It decides which of the boxes below appear "
              "at all."),
        S("ph_mode", "The reactions in this run", "reaction_mode",
          "biotic only",
          "BIOTIC is microbially mediated: it needs microbes present, obeys "
          "Monod kinetics in both the donor and the acceptor, grows biomass "
          "with a yield and loses it to decay. ABIOTIC is purely chemical "
          "and no microbes take any part in it: it is first order in the "
          "product and consumes it. Each gets its own Damkohler number, and "
          "a reaction that is not running records zero rather than a copy of "
          "the other one's.",
          kind="choice",
          choices=["biotic only", "abiotic only", "both"]),

        # ---------------------------------------------------------------
        #  2. the chemicals: what they are, where they start, what happens
        #     at the faces. The rate laws come straight after, because a
        #     rate is meaningless until you know what it acts on.
        # ---------------------------------------------------------------
        Field("chemicals", "The chemicals", "chemicals", 4,
              "The ORDER is the meaning. The first is the electron donor, "
              "the second the acceptor, the third the product, the fourth "
              "the biomass. Add more with the plus button and they are "
              "carried properly -- fed at the inlet, advected, and diffused "
              "with their own coefficient -- but take no part in any "
              "reaction, which is exactly what a conservative tracer is. A "
              "fifth REACTING chemical needs CompLaB. Concentrations are in "
              "mol/L and diffusion coefficients in square metres per second.",
              flag="--set"),

        # ---------------------------------------------------------------
        #  3. the biotic rate law, right after the chemicals it acts on
        # ---------------------------------------------------------------
        Field("HD_bio", "3.  The biotic reaction: rate and microbes",
              "heading", "",
              "donor + (stoichiometry) acceptor gives product + (yield) "
              "biomass, at a rate of vmax x biomass x donor/(Ks+donor) x "
              "acceptor/(Ks+acceptor). Dual Monod: whichever of the two is "
              "scarce controls the rate.", depends=BIO),
        S("ph_vmax", "Maximum specific rate", "vmax", "1.0",
          "The biotic rate per unit biomass, when nothing is limiting.",
          depends=BIO),
        S("ph_rate_unit", "Rates are quoted", "rate_unit", "per_day",
          "Microbiology is usually quoted per day. Applies to every rate on "
          "this page.", kind="choice", choices=["per_day", "per_second"],
          depends=BIO),
        S("ph_ksd", "Half saturation, donor (mol/L)", "ks_donor", "0.0",
          "Zero means the rate is first order in the donor and the constant "
          "is folded into the Damkohler number, which is the default.",
          depends=BIO),
        S("ph_ksa", "Half saturation, acceptor (mol/L)", "ks_acceptor",
          "7.5e-4",
          "The concentration at which the acceptor limits the rate to half. "
          "Well above the feed concentration switches the reaction off.",
          depends=BIO),
        S("ph_stoi", "Acceptor consumed per donor", "stoichiometry", "1.0",
          "Moles of acceptor per mole of donor, before the feed "
          "concentrations are taken into account.", depends=BIO),
        S("ph_yield", "Yield (mol biomass per mol donor)", "yield", "0.04",
          "", depends=BIO),
        S("ph_decay", "Biomass decay", "decay", "0.05", "", depends=BIO),
        S("ph_bmode", "The microbes are", "biomass_mode", "sessile",
          "SESSILE is an attached biofilm: it reacts where it is and never "
          "moves. This is what sets the fourth chemical's Attached box, so "
          "do not set that one by hand. PLANKTONIC means dissolved cells, "
          "advected and diffused like any other chemical -- and in a "
          "through-flowing sample most of them leave. Measured on a 60 by 32 "
          "domain, mean biomass fell from 0.100 to 0.019 with transport on "
          "and to 0.071 with it off, so about three quarters of the loss was "
          "washout rather than decay.",
          kind="choice", choices=["sessile", "planktonic"], depends=BIO),
        S("ph_bfcode", "Voxel value that means biofilm", "biofilm_code", "",
          "Optional, and only meaningful for sessile microbes. CompLaB gives "
          "attached microbes their own material number, so biofilm occupies "
          "named voxels in the geometry and the biomass is seeded only "
          "there. Put a number here -- 3 is the usual one, and it must "
          "differ from the pore, solid and wall codes -- and you get the "
          "same: the generator coats the grain surfaces with it, those "
          "voxels stay part of the pore space, chemicals diffuse through "
          "them at their D in biofilm rather than their D in pore, and the "
          "starting biomass goes there and nowhere else. Leave it blank and "
          "the biomass starts uniformly through the pore space, which is "
          "what every earlier dataset used. Either way nothing grows into "
          "new voxels during a run: for that, use CompLaB.",
          depends=BIO, optional=True),
        S("ph_bflayers", "How thick a biofilm lining (voxels)",
          "biofilm_layers", "1",
          "Only used when a biofilm code is set. 1 coats every pore voxel "
          "touching a grain; 2 coats that layer and the one behind it.",
          depends=BIO),
        S("ph_bcoup", "The rate follows the biomass actually present",
          "biomass_coupled", "false",
          "False holds the biomass at its starting value inside the rate. "
          "That is not an oversight: the Damkohler number is defined as "
          "vmax x B0 x L / (Ac0 x u), so the starting biomass is already "
          "inside it, and letting it vary as well would count it twice. Set "
          "true when the biomass changes a lot over a run and you want the "
          "feedback.", kind="choice", choices=["false", "true"], depends=BIO),
        Field("da_min", "Lowest BIOTIC Damkohler", "float", 0.1,
              "The microbially mediated rate against advection. Around 1 the "
              "reaction and the flow take comparable times, which is where "
              "the problem is interesting.", flag="--da-min", depends=BIO),
        Field("da_max", "Highest BIOTIC Damkohler", "float", 10.0, "",
              flag="--da-max", depends=BIO),

        # ---------------------------------------------------------------
        #  4. the abiotic rate law
        # ---------------------------------------------------------------
        Field("HD_abio", "4.  The abiotic reaction: rate", "heading", "",
              "A purely chemical reaction, first order, with no microbes "
              "involved and none needed. It turns one dissolved chemical "
              "into another at k x concentration -- which is what a mineral "
              "dissolution or a surface-catalysed transformation does. Set "
              "the reaction mode to 'abiotic only' for an experiment with no "
              "biology in it at all.", depends=ABIO),
        S("ph_ksurf", "Abiotic rate constant", "k_surf", "1.0", "",
          depends=ABIO),
        S("ph_abioreact", "The abiotic reaction consumes", "abiotic_reactant",
          "auto",
          "'product' is the CompLaB-like case: microbes make a product and a "
          "purely chemical reaction removes it. It is the wrong choice with "
          "no biotic reaction, and silently so -- nothing makes a product, so "
          "the abiotic reaction has nothing to act on and the whole field "
          "comes out zero. 'donor' consumes the fed chemical directly, which "
          "is what an abiotic-only experiment actually is. 'auto', the "
          "default, picks the product when the biotic reaction is running and "
          "the donor when it is not.",
          kind="choice", choices=["auto", "product", "donor"], depends=ABIO),
        S("ph_abioprod", "and turns it into", "abiotic_product", "product",
          "'product' means the reaction really is a reaction: the chemical it "
          "consumes BECOMES the product, one mole for one, so the donor falls "
          "and the product rises and the two of them sum to the feed. That is "
          "what you want for an abiotic-only run -- no microbes, no biomass, "
          "and still a reaction with something to show for it.\n\n"
          "'none' consumes the chemical and makes nothing that this run "
          "follows. Right for a terminal loss: a solute that precipitates out "
          "of the water, or degrades to something you are not tracking.\n\n"
          "It has no effect when the reaction already consumes the product, "
          "since turning the product into the product is not a reaction.",
          kind="choice", choices=["product", "none"], depends=ABIO),
        S("ph_abiosurf", "The abiotic reaction happens",
          "abiotic_surface_only", "false",
          "false is everywhere in the water. true is only on pore voxels "
          "that touch a grain, which is what a mineral-surface reaction "
          "does. CompLaB gates its whole abiotic network the same way.",
          kind="choice", choices=["false", "true"], depends=ABIO),
        Field("da2_min", "Lowest ABIOTIC Damkohler", "float", 0.1,
              "Drawn independently of the biotic one, so the network can "
              "learn to tell the two reactions apart.",
              flag="--da-abio-min", depends=ABIO),
        Field("da2_max", "Highest ABIOTIC Damkohler", "float", 10.0, "",
              flag="--da-abio-max", depends=ABIO),
        S("ph_usephys", "Get Damkohler from the rates above rather than "
          "sweeping it", "use_physical_kinetics", "false",
          "False, the default, sweeps Damkohler over the ranges above and "
          "reads the rate constants without using them -- which is what a "
          "TRAINING SET wants, because it has to span the space rather than "
          "sit at one system's point in it. True computes Damkohler from the "
          "rates, which is what you want when you are modelling one named "
          "system. Check what your rates imply first: at a sample of tens of "
          "micrometres with ordinary diffusion coefficients the answer is "
          "usually far below 1, because diffusion crosses the sample in "
          "about a second.",
          kind="choice", choices=["false", "true"]),

        # ---------------------------------------------------------------
        #  5. the sample and the flow
        # ---------------------------------------------------------------
        Field("HD_sample", "5.  The sample", "heading", "",
              "Everything below is in the units named on the box, not in "
              "lattice units. A box you have not changed sends nothing, so a "
              "settings file or the block in the script still decides it; a "
              "box you HAVE changed wins over both."),
        S("ph_dx", "Voxel edge length", "dx", "1.0",
          "The size of one voxel, in the unit below. This is what turns a "
          "grid into a physical sample: 32 voxels at 1 micrometre is a 32 "
          "micrometre sample."),
        S("ph_unit", "Length unit", "unit", "um",
          "um is micrometres, which is what a pore-scale image is measured "
          "in.", kind="choice", choices=["um", "mm", "m"]),
        S("ph_char", "Open pore width", "characteristic_length", "16.0",
          "The typical width of an open pore, in the same unit as the voxel "
          "edge. CompLaB divides by this when it forms Peclet and Damkohler."),
        S("ph_refl", "Take ratios against", "reference_length", "sample",
          "Which length Peclet and Damkohler are ratios of. 'sample' is the "
          "whole inlet-to-outlet distance, which is what this solver uses. "
          "'characteristic' is the pore width, which is what CompLaB uses. "
          "Both are printed, so a Peclet quoted either way can be compared.",
          kind="choice", choices=["sample", "characteristic"]),
        S("ph_pore", "Voxel value that means open pore", "pore_code", "2",
          "The material codes in your geometry file. 2 pore, 0 solid, 1 wall "
          "is CompLaB's own default. The published 2D release swaps 1 and 2, "
          "so check rather than assume: a swapped code gives a perfectly "
          "runnable simulation of the wrong rock."),
        S("ph_solid", "Voxel value that means solid grain", "solid_code",
          "0", ""),
        S("ph_wall", "Voxel value that means wall", "wall_code", "1", ""),

        Field("HD_flow", "6.  The flow", "heading", "",
              "You say what Peclet you want, in the two boxes further up, and "
              "the pressure gradient that produces it is worked out for you. "
              "The flow really is driven by a pressure gradient -- this is "
              "not a shortcut around one -- but how hard to push is an "
              "answer, not a question. Stokes flow is linear, so the "
              "gradient is exact rather than fitted, and it is computed per "
              "ROCK: a tighter pore structure needs a stronger push for the "
              "same Peclet, which is what permeability means. Each rock "
              "prints the range it needs as it is built, and every run "
              "stores the value it used."),
        S("ph_tau", "Relaxation time, tau", "tau", "0.8",
          "Sets the viscosity of the model fluid, as (tau minus 0.5) divided "
          "by 3. It must be above 0.5 or the viscosity is negative. Use 0.6 "
          "to 1.0. This is the one flow setting that is still yours to "
          "choose: it decides the accuracy and stability of the "
          "lattice-Boltzmann solve rather than how fast the water goes."),

        Field("HD_num", "7.  When to stop", "heading", "", ""),
        S("ph_nscv", "Flow is settled when it changes by less than",
          "ns_converge", "1e-6",
          "How still the flow field has to be before the transport starts. "
          "The flow is solved once per rock, so this is cheap to tighten."),
        S("ph_adecv", "Transport is settled when it changes by less than",
          "ade_converge", "2e-3",
          "The transport run stops when the mean donor concentration changes "
          "by less than this over one advective time. It is not a fixed step "
          "count on purpose: a count that is right at Peclet 50 leaves the "
          "field almost empty at Peclet 1."),
        S("ph_admax", "Never take more transport steps than",
          "ade_max_iT", "100000",
          "A backstop, not a target. A run that hits it is recorded as not "
          "settled rather than being passed off as a steady state. At 148 by "
          "64 the low-Peclet runs need about 130000, so raise it if the log "
          "reports runs that RAN OUT."),
        S("ph_declat", "Biomass decay in solver units", "decay_lattice",
          "0.006",
          "Used when Damkohler is swept rather than derived. This is what "
          "the generators used before any of these boxes existed, so "
          "changing it changes every dataset made from here on.",
          depends=BIO),
    ]


# =============================================================================
def build_actions():
    A = {}

    # ---------------------------------------------------------- 0. INSTALL
    A["install"] = Action(
        "install", "Install the Python packages",
        "Downloads the libraries the project needs, into the Python shown below.",
        os.path.join(HERE, "install_requirements.py"),
        [Field("torch", "Which build of torch", "choice", "cpu",
               "'cpu' is right unless this computer has an NVIDIA graphics card. "
               "The CPU build is much smaller and is fine for practice datasets "
               "and for prediction; only large training runs really want a GPU. "
               "'skip' installs everything except torch, which gives you the "
               "viewer and dataset tools but not training.",
               choices=["cpu", "cuda", "skip"], flag="--torch"),
         Field("upgrade", "Upgrade packages that are already there", "bool", False,
               "Leave this off the first time.", flag="--upgrade"),
         Field("check", "Only check, install nothing", "bool", False,
               "Reports what is present and stops.", flag="--check")],
        [("an internet connection", "the packages are downloaded from the Python "
          "package index"),
         ("nothing else", "it installs into the Python this window is using, which "
          "is shown at the bottom left")],
        ["Reports which packages are already present, and stops there if nothing "
         "is needed.",
         "Upgrades pip, then installs numpy, scipy, h5py, matplotlib and "
         "scikit-image.",
         "Installs torch last, because it is by far the largest, roughly 200 MB "
         "for the CPU build.",
         "Checks every package again afterwards and reports the result, so a "
         "partial failure cannot pass unnoticed."],
        [("nothing on disk that you need to find",
          "the packages go inside the Python environment. The result is that the "
          "other buttons in this window start working.")],
        group="Setup",
        note="Run this first if the window warned you that packages are missing. "
             "It installs into the Python shown in the bottom left corner, which "
             "is the one the analysis scripts use.")

    # ------------------------------------------------- 0a. YOUR OWN NUMBERS
    A["capabilities"] = Action(
        "capabilities", "What this simulator can and cannot do",
        "Where the Python simulator stands next to CompLaB, line by line. "
        "Read it once before you decide which one a question needs.",
        os.path.join(TOOLS, "settings_and_units.py"),
        [Field("capabilities", "Print the comparison", "bool", True,
               "", flag="--capabilities")],
        [("nothing", "it prints a reference list and stops")],
        ["Names what both simulators do: flow through the pore space, "
         "advection, diffusion and reaction on the same grid, Monod kinetics "
         "with dual limitation, a diffusion coefficient per chemical, and the "
         "no-slip wall in the same place -- halfway between the last fluid "
         "voxel and the first solid one, which is where Palabos puts it.",
         "Names what only CompLaB does: chemical equilibrium, precipitation "
         "and pore voxels turning into solid, surface complexation, many "
         "microbial pools, a separate diffusivity inside the biofilm, and a "
         "flow field re-solved as the biofilm grows.",
         "Names what this does that is easy to miss: chemicals past the "
         "fourth are transported properly but take no part in any reaction, "
         "which is a conservative tracer.",
         "Gives the measured cost of both, so the choice can be made on "
         "numbers."],
        [("nothing on disk", "the same text is in PYTHON_VS_COMPLAB.md at the "
          "top of the project")],
        group="Help",
        note="The short answer: build the training set here, because it takes "
             "an afternoon rather than a cluster week. Run the reference cases "
             "in CompLaB, because that is what the paper rests on.")

    # ---------------------------------------------------------- 0b. 2D DATA
    A["dataset2d"] = Action(
        "dataset2d", "Run 2D simulations to train on",
        "The same thing in two dimensions, and about fifty times cheaper. "
        "Every physical input is on this page.",
        os.path.join(TOOLS, "build_dataset_2d.py"),
        [Field("out", "Save the dataset as", "savefile",
               os.path.join(NEW2D, "dataset.h5"), "", flag="--out",
               filetypes=[("HDF5 dataset", "*.h5")]),
         Field("source", "Where the pore structures come from", "choice",
               "generate new ones",
               "Either invent them with the same statistics as the published "
               "set, or use the 3000 real domains that ship in the 2D folder.",
               choices=["generate new ones", "use the published 2D domains"]),
         Field("n_geom", "How many different rocks to generate", "int", 40,
               "Each one is a separate pore structure, generated once and "
               "then reused for every condition. More rocks teaches the "
               "network more about shape, and scoring holds out whole rocks.",
               flag="--n-geom", depends=("source", "generate new ones")),
         Field("phi_min", "Lowest porosity", "float", 0.55,
               "In 2D the pore space stops spanning the sample near 0.59, "
               "nearly three times higher than the 0.20 threshold in 3D. That "
               "is the topology gap between the two dimensions, and it is why a "
               "geometry encoder trained on 2D cannot simply be reused in 3D.",
               flag="--phi-min", depends=("source", "generate new ones")),
         Field("phi_max", "Highest porosity", "float", 0.85, "",
               flag="--phi-max", depends=("source", "generate new ones")),
         Field("jung_dir", "The 2D folder", "dir", DIR_2D,
               "The published release, with its Domains subfolder.",
               flag="--jung-dir", depends=("source", "use the published 2D domains")),
         Field("limit", "How many of them to use", "int", 200,
               "There are 3000. Two hundred is a sensible first run.",
               flag="--limit", depends=("source", "use the published 2D domains")),
         Field("shape", "Grid size (nx ny)", "text", "148 64",
               "Two numbers separated by a space. 148 by 64 is the size the "
               "published domains use.", flag="--shape", nargs=2),
         Field("n_sets", "How many conditions to run on each rock", "int", 6,
               "One simulation per condition. A condition is one combination "
               "of Peclet and Damkohler, drawn from the ranges below.",
               flag="--n-sets"),
         Field("SIMCOUNT", "", "computed", "", ""),
         Field("n_times", "Snapshots in time", "int", 11, "", flag="--n-times"),
         Field("pe_min", "Lowest Peclet", "float", 0.3, "", flag="--pe-min"),
         Field("pe_max", "Highest Peclet", "float", 30.0, "", flag="--pe-max"),
         Field("stokes", "Flow solver iterations", "int", 1500,
               "More iterations means a better converged flow field.",
               flag="--stokes-iters"),
         Field("adr", "Transport steps", "int", "",
               "LEAVE THIS BLANK. Blank means the transport is run until the "
               "field stops changing, which is the only way the solute actually "
               "crosses the sample. A fixed number here overrides that: 200 "
               "steps at Peclet 10 carries the front about five voxels into a "
               "148-voxel domain, so almost every value in the dataset is zero "
               "and a network trained on it learns to predict zero. Set a number "
               "only to cut a run short deliberately, knowing that.",
               flag="--adr-steps", optional=True),
         ] + physical_fields() + [
         Field("HD_file", "Settings files, and checking before you run",
               "heading", "",
               "The boxes above already hold every physical input. These "
               "three are for moving a whole set of numbers around, and for "
               "reading them back before you spend hours on them."),
         Field("settings", "Or load it all from a settings file", "file", "",
               "Optional, and usually you do not need it: the boxes above "
               "already hold every physical input, and they win over anything "
               "in a file. Use this to load a real CompLaB.xml, or a settings "
               "file a colleague sent you. The grid box further up is also "
               "always sent, so it wins over the grid in the file.",
               flag="--settings", optional=True,
               filetypes=[("Settings file", "*.xml *.json"),
                          ("All files", "*.*")]),
         Field("overrides", "Change one setting, without editing the file",
               "repeated", "",
               "Optional, and you can put several here separated by spaces. "
               "Write them as name=value: tau=0.9 for a top-level setting, "
               "species.1.d_pore=5e-10 for one chemical's. A name that does "
               "not exist stops the run and lists the ones that do, rather "
               "than being ignored.",
               flag="--set", optional=True),
         Field("save_settings", "Also save these numbers as a settings file",
               "savefile", "",
               "Optional. Writes everything this run used to an XML file you "
               "can read, edit, e-mail to somebody, or hand back to this page "
               "later. Leave it blank and nothing is written -- the numbers "
               "are stored inside the dataset either way.",
               flag="--save-settings", optional=True,
               filetypes=[("Settings file", "*.xml"),
                          ("Settings file", "*.json")]),
         Field("HD_out", "8.  What to write besides the dataset", "heading",
               "",
               "The dataset itself is always one .h5 file. These add files "
               "for LOOKING at, and nothing in this project reads them back, "
               "so deleting them loses nothing but the pictures."),
         Field("save_png", "Also save PNG pictures of the results", "bool",
               True,
               "One picture per saved simulation, every chemical side by "
               "side on a shared colour scale with the grains outlined. This "
               "is the fastest way to see whether the front crossed the "
               "sample or the field came out empty. CompLaB does not do "
               "this at all.", flag="--save-png"),
         Field("save_vti", "Also save ParaView .vti files", "bool", False,
               "One file per chemical per snapshot, plus the rock and the "
               "flow speed, in the same VTK ImageData format CompLaB "
               "writes -- so a Python run drops into the same ParaView "
               "session as a CompLaB run and the two can be put side by "
               "side. It is a lot of files.", flag="--save-vti"),
         Field("save_runs", "How many simulations to write those for", "int",
               4,
               "Writing pictures for all eighty runs of a big sweep is a lot "
               "of files and not much more information than the first few.",
               flag="--save-runs"),
         Field("show_settings", "Just check the numbers, do not simulate",
               "bool", False,
               "Prints every number above, every number derived from them, "
               "anything it cannot honour, and anything that is legal but "
               "almost certainly not what you meant -- then stops without "
               "building anything. Do this first, every time. It costs a "
               "second and the alternative costs hours.",
               flag="--show-settings"),
         ],
        [("either nothing", "structures are generated from a random seed"),
         ("or the 2D folder", "3000 published domains, used with OUR chemistry "
          "rather than the published chemistry, so that 2D and 3D results are "
          "directly comparable")],
        ["Builds or reads the pore structures and checks each one percolates "
         "from inlet to outlet, discarding those that do not.",
         "Computes the geodesic distance field: for every pore voxel, the "
         "shortest path back to the inlet THROUGH pore space. This is what the "
         "trunk receives, and the premise of the whole method is that it beats "
         "the straight-line distance.",
         "Solves the flow with a D2Q9 lattice-Boltzmann Stokes solver, the "
         "two-dimensional twin of the D3Q19 solver used for the 3D runs.",
         "Solves advection, diffusion and a first-order reaction to produce the "
         "concentration fields through time.",
         "Writes the same file format the 3D pipeline uses, with nz = 1. Every "
         "other button in this window then works unchanged: training notices "
         "nz = 1, switches its encoder to 2D convolutions, and drops z from the "
         "trunk, giving the published (x, y, t, geodesic) architecture."],
        [("the .h5 file you named",
          "a complete 2D dataset. Train on it with exactly the same Train button "
          "you would use in 3D.")],
        group="Setup",
        progress_re=re.compile(r"^\s*\[\s*(\d+)/(\d+)\]"),
        note="A 2D run is roughly fifty times cheaper than the 3D one. Use 2D to "
             "settle questions about the METHOD, then confirm in 3D.")

    # ------------------------------------------------------- 0b2. 3D DATA
    A["dataset3d"] = Action(
        "dataset3d", "Run 3D simulations to train on",
        "Builds pore structures, solves the flow and the reactive transport "
        "through them, and writes the file the network trains on. Every "
        "physical input is on this page.",
        os.path.join(TOOLS, "build_dataset_3d.py"),
        [Field("out", "Save the dataset as", "savefile",
               os.path.join(NEW3D, "dataset.h5"), "", flag="--out",
               filetypes=[("HDF5 dataset", "*.h5")]),
         Field("shape", "Grid size (nx ny nz)", "text", "32 32 32",
               "Three numbers separated by spaces. THIS IS THE EXPENSIVE "
               "CHOICE: the cost goes as the FIFTH power of it, because the "
               "number of voxels goes as the cube and the number of time steps "
               "goes as the square. Measured per run on one core: 32 cubed is "
               "7 seconds, 48 cubed is 58 seconds, 64 cubed is about 4 minutes. "
               "Start at 32 to prove the pipeline in an hour; 64 is the grid "
               "the CompLaB campaign uses, so a model trained at 64 can be "
               "compared with CompLaB output directly.",
               flag="--shape", nargs=3),
         Field("n_geom", "How many different rocks to generate", "int", 16,
               "Each one is a separate pore structure, generated once and "
               "then reused for every condition. The flow is solved once per "
               "rock, because the pore space does not change and the flow "
               "does not depend on the chemistry. Scoring holds out whole "
               "ROCKS, so this number, not the number of simulations, is what "
               "decides how trustworthy the held-out error is.",
               flag="--n-geom"),
         Field("n_sets", "How many conditions to run on each rock", "int", 5,
               "One simulation per condition. A condition is one combination "
               "of Peclet and Damkohler, drawn from the ranges below.",
               flag="--n-sets"),
         Field("SIMCOUNT", "", "computed", "", ""),
         Field("n_times", "Snapshots in time", "int", 8, "", flag="--n-times"),
         Field("phi_min", "Lowest porosity", "float", 0.30,
               "In 3D the pore space stops spanning the sample near 0.20; in 2D "
               "the threshold is near 0.59. Same generator, same morphology, "
               "threshold three times lower. That gap is why a 2D range of "
               "0.55 to 0.85 would be absurd here, and it is the same gap that "
               "stops a geometry encoder trained in 2D from being reused in 3D.",
               flag="--phi-min"),
         Field("phi_max", "Highest porosity", "float", 0.50, "", flag="--phi-max"),
         Field("pe_min", "Lowest Peclet", "float", 1.0,
               "Low Peclet is diffusion dominated and costs several times more "
               "per run than the advective end.", flag="--pe-min"),
         Field("pe_max", "Highest Peclet", "float", 50.0, "", flag="--pe-max"),
         Field("stokes", "Flow solver iterations", "int", 400,
               "Once per structure, not once per run.", flag="--stokes-iters"),
         Field("seed", "Random seed", "int", 0, "Fixing this makes it "
               "reproducible.", flag="--seed"),
         ] + physical_fields() + [
         Field("HD_file", "Settings files, and checking before you run",
               "heading", "",
               "The boxes above already hold every physical input. These "
               "three are for moving a whole set of numbers around, and for "
               "reading them back before you spend hours on them."),
         Field("settings", "Or load it all from a settings file", "file", "",
               "Optional, and usually you do not need it: the boxes above "
               "already hold every physical input, and they win over anything "
               "in a file. Use this to load a real CompLaB.xml, or a settings "
               "file a colleague sent you. The grid box further up is also "
               "always sent, so it wins over the grid in the file.",
               flag="--settings", optional=True,
               filetypes=[("Settings file", "*.xml *.json"),
                          ("All files", "*.*")]),
         Field("overrides", "Change one setting, without editing the file",
               "repeated", "",
               "Optional, and you can put several here separated by spaces. "
               "Write them as name=value: tau=0.9 for a top-level setting, "
               "species.1.d_pore=5e-10 for one chemical's. A name that does "
               "not exist stops the run and lists the ones that do, rather "
               "than being ignored.",
               flag="--set", optional=True),
         Field("save_settings", "Also save these numbers as a settings file",
               "savefile", "",
               "Optional. Writes everything this run used to an XML file you "
               "can read, edit, e-mail to somebody, or hand back to this page "
               "later. Leave it blank and nothing is written -- the numbers "
               "are stored inside the dataset either way.",
               flag="--save-settings", optional=True,
               filetypes=[("Settings file", "*.xml"),
                          ("Settings file", "*.json")]),
         Field("HD_out", "8.  What to write besides the dataset", "heading",
               "",
               "The dataset itself is always one .h5 file. These add files "
               "for LOOKING at, and nothing in this project reads them back, "
               "so deleting them loses nothing but the pictures."),
         Field("save_png", "Also save PNG pictures of the results", "bool",
               True,
               "One picture per saved simulation, every chemical side by "
               "side on a shared colour scale with the grains outlined. This "
               "is the fastest way to see whether the front crossed the "
               "sample or the field came out empty. CompLaB does not do "
               "this at all.", flag="--save-png"),
         Field("save_vti", "Also save ParaView .vti files", "bool", False,
               "One file per chemical per snapshot, plus the rock and the "
               "flow speed, in the same VTK ImageData format CompLaB "
               "writes -- so a Python run drops into the same ParaView "
               "session as a CompLaB run and the two can be put side by "
               "side. It is a lot of files.", flag="--save-vti"),
         Field("save_runs", "How many simulations to write those for", "int",
               4,
               "Writing pictures for all eighty runs of a big sweep is a lot "
               "of files and not much more information than the first few.",
               flag="--save-runs"),
         Field("show_settings", "Just check the numbers, do not simulate",
               "bool", False,
               "Prints every number above, every number derived from them, "
               "anything it cannot honour, and anything that is legal but "
               "almost certainly not what you meant -- then stops without "
               "building anything. Do this first, every time. It costs a "
               "second and the alternative costs hours.",
               flag="--show-settings")],
        [("nothing, or a settings file", "without one it needs no input files "
          "and creates everything itself; with one, every physical number in "
          "it is used and stored inside the dataset")],
        ["Generates pore structures by thresholding a smoothed random field, "
         "checks each one spans inlet to outlet, and turns any pore space the "
         "inlet cannot reach into solid -- because an unreachable pocket has no "
         "geodesic distance, and the infinity that stands in for one would go "
         "into the network as a coordinate.",
         "Computes the geodesic distance from the inlet THROUGH pore space, "
         "which is what the trunk receives.",
         "Solves the flow with the D3Q19 lattice-Boltzmann Stokes solver, once "
         "per structure.",
         "Solves 3D advection, diffusion and the coupled reaction with the SAME "
         "solver the 2D generator uses, so a 2D result and a 3D result from "
         "this project differ only in dimension and can be compared directly.",
         "Checks the physics of every run before writing: nothing above the "
         "inlet concentration, no checkerboard, nothing at the outlet before it "
         "has crossed the sample, and no chemical a copy of another. If any run "
         "fails, NOTHING is written -- an unusable dataset that looks fine is "
         "worse than no dataset."],
        [("the .h5 file you named",
          "a complete 3D dataset in the same format a CompLaB campaign "
          "produces, so every other button works on it unchanged")],
        group="Setup",
        progress_re=re.compile(r"^\s*\[\s*(\d+)/(\d+)\]"),
        note="This is NOT CompLaB. CompLaB is the reference simulator and runs "
             "on the cluster, hours per run -- use Build the CompLaB campaign "
             "and then Collect for that. This exists so the whole method can be "
             "exercised end to end in 3D on one machine before the cluster time "
             "is spent. It is also the BASELINE: no flow proxy, no 2D transfer, "
             "no dimension-free coordinates. Geometry in, concentrations out. "
             "The switches have to be measured against something, and this is "
             "the something.")

    # ---------------------------------------------------------- 0c. EXTERNAL
    A["ingestsims"] = Action(
        "ingestsims", "Load somebody else's 2D simulations",
        "Turns simulation output sent by a collaborator into a dataset we can "
        "train on.",
        os.path.join(TOOLS, "import_2d_simulations.py"),
        [Field("src", "Their files", "dir", "",
               "A folder of .npz files, one per run. Point it at a SINGLE file "
               "the first time and tick 'only report' below.", flag="--src"),
         Field("dry_run", "Only report what is there, write nothing", "bool", True,
               "Do this first, on one example run. It says exactly what it found "
               "and what is missing. One round trip now is much cheaper than "
               "receiving a hundred runs in a layout we cannot read.",
               flag="--dry-run"),
         Field("out", "Save the dataset as", "savefile",
               os.path.join(WORK, "2D_from_collaborator", "dataset.h5"),
               "Only used when 'only report' is off.", flag="--out",
               filetypes=[("HDF5 dataset", "*.h5")]),
         Field("n_species", "How many chemicals", "int", 1, "", flag="--n-species"),
         Field("species", "Their names", "text", "C",
               "Space-separated, in the order the arrays are stacked. Several "
               "names go in as several names: \"Ac A P Bio\" is four chemicals, "
               "not one chemical with a long name.",
               flag="--species", nargs="*"),
         Field("pe", "Peclet, if the files do not carry it", "float", "",
               "Leave blank if each file already holds its own Pe.",
               flag="--pe", optional=True),
         Field("da", "Damkohler, if the files do not carry it", "float", "",
               "", flag="--da", optional=True),
         Field("params_json", "Or a lookup table of conditions", "file", "",
               "A json mapping filename to its Pe and Da, for when the "
               "conditions live outside the arrays.", flag="--params-json",
               optional=True, filetypes=[("JSON", "*.json")])],
        [("their .npz files", "one per run: the pore geometry, the concentration "
          "field, the Peclet and Damkohler numbers, and ideally the flow field "
          "and more than one snapshot in time")],
        ["Reads each file and matches the array names against a list of aliases, "
         "so 'C', 'conc', 'concentration' and 'sol' are all understood, and "
         "reports exactly which array it took for what.",
         "Works out which voxel value means pore by testing which phase actually "
         "spans the sample, rather than assuming a convention.",
         "Normalises the concentration array to (time, chemical, x, y) whatever "
         "shape it arrived in.",
         "Counts how many DISTINCT structures there are, and warns if every run "
         "used the same one, because then a split by structure is impossible and "
         "any score would be optimistic.",
         "Warns if no run carries a flow field, because the flow-proxy and "
         "dimension-free switches then cannot be used at all."],
        [("the .h5 file you named",
          "a dataset in our standard format, which Train, Evaluate and Predict "
          "all accept unchanged")],
        group="Setup",
        note="The published 2D release contains geometries, trained weights and "
             "notebooks, but NO simulation fields. This action is for data a "
             "collaborator sends separately. See HEEWON_DATA.md for what to ask "
             "for.")

    # ---------------------------------------------------------- 0d. WARM START
    A["heewon"] = Action(
        "heewon", "Warm-start from the published 2D weights",
        "Loads the trained network from the published release into ours, so "
        "training starts from a geometry encoder that already works.",
        os.path.join(TOOLS, "load_pretrained_2d_weights.py"),
        [Field("checkpoint", "Their trained model", "file",
               os.path.join(DIR_2D, "parameters", "Monod.pt"),
               "One of the .pt files in the 2D folder.", flag="--checkpoint",
               filetypes=[("Checkpoint", "*.pt")]),
         Field("grid", "The grid it was trained on (nx ny)", "text", "148 64",
               "Two numbers separated by a space. Their fc layer expects "
               "2048 = 256 x (148/32) x (64/32) inputs, so it is tied to this "
               "grid. A different one and that layer will not load.",
               flag="--grid", nargs=2),
         Field("n_species", "Chemicals in YOUR dataset", "int", 1, "",
               flag="--n-species"),
         Field("n_params", "Parameters in YOUR dataset", "int", 6,
               "Theirs takes 2, Pe and Da. Ours take 6. With 6 the first layer "
               "of their parameter branch is skipped, and it says so rather than "
               "silently reshaping anything.", flag="--n-params"),
         Field("save", "Save the warm start as", "savefile",
               os.path.join(NEW2D, "model", "warmstart.pt"), "", flag="--save",
               filetypes=[("Checkpoint", "*.pt")])],
        [("one of 2D/parameters/*.pt", "the published trained weights")],
        ["Maps their layer names onto ours and reports what transferred, what "
         "did not, and why.",
         "Their trunk takes four inputs, which is (x, y, time, geodesic "
         "distance) -- exactly what our 2D mode builds. So the part of the "
         "network that reads the pore structure transfers with no surgery.",
         "Their output head is skipped: they predict one chemical and have no "
         "head layer, because with one chemical the branch code is already the "
         "coefficient vector.",
         "Runs a forward pass, because 'the shapes match' is not the same as "
         "'it runs'."],
        [("warmstart.pt",
          "a checkpoint to pass to Train with 'Start from an existing model', "
          "usually together with 'freeze the reaction part'")],
        group="Setup",
        note="Whether this helps is an open question, not a claim: their trunk "
             "was fitted to THEIR chemistry. Train both ways and compare.")

    # ---------------------------------------------------------- 1. GEOMETRY
    A["geometry"] = Action(
        "geometry", "Generate pore geometries",
        "Creates the porous samples the simulations will run on.",
        os.path.join(TOOLS, "build_geometry_3d.py"),
        [Field("n", "How many structures", "int", 20,
               "One simulation campaign runs on all of them.", flag="--n"),
         Field("out", "Folder to write them to", "dir",
               os.path.join(WORK, "geometries"),
               "One subfolder per geometry, plus a manifest.csv summary.", flag="--out"),
         Field("nx", "Voxels along the flow (nx)", "int", 128,
               "The new plan uses 128, elongated so the front has room to develop.",
               flag="--nx"),
         Field("ny", "Voxels across (ny)", "int", 64, "", flag="--ny"),
         Field("nz", "Voxels across (nz)", "int", 64, "", flag="--nz"),
         Field("phi_min", "Lowest porosity", "float", 0.20,
               "0.20 is the percolation threshold: below it the pore space stops "
               "spanning the sample and the run is meaningless.", flag="--phi-min"),
         Field("phi_max", "Highest porosity", "float", 0.95,
               "Near 1 the sample is almost an open box and teaches the network little.",
               flag="--phi-max"),
         Field("render", "Also draw pictures of each one", "bool", True,
               "2D slices and the geodesic distance field.", flag="--render"),
         Field("render3d", "Also draw 3D renders", "bool", False,
               "Slower, about 35 seconds each.", flag="--render3d"),
         Field("verify", "Verify the written files round-trip", "bool", True,
               "Reads each geometry.dat back and checks it matches.", flag="--verify"),
         Field("seed", "Random seed", "int", 20260728,
               "Fixing this makes the set reproducible. Change it when you want "
               "structures that CANNOT be any of the ones a model was trained "
               "on -- which is the only kind worth predicting on.",
               flag="--seed")],
        [("nothing", "Structures are generated from a random seed, not read from disk.")],
        ["Smooths white noise with a Gaussian filter to get the organic, "
         "bicontinuous pore shapes seen in real rock, then thresholds it at the "
         "porosity you asked for.",
         "Adds a bounce-back wall shell, because CompLaB has no periodic boundary "
         "option anywhere in its source.",
         "Flood-fills from the inlet face and rejects any structure whose pore "
         "space does not reach the outlet.",
         "Computes the geodesic distance field: for every pore voxel, the shortest "
         "path back to the inlet THROUGH pore space. This is what the network uses "
         "to understand the geometry."],
        [("geom_NNNN/geometry.dat", "the geometry as CompLaB reads it, one integer per line"),
         ("geom_NNNN/geom_NNNN.npz", "the same thing plus the geodesic and Euclidean distance fields"),
         ("geom_NNNN/*_slices.png", "a picture of the structure and its distance field"),
         ("manifest.csv", "porosity, correlation length and tortuosity for every structure")],
        group="Setup")

    # ---------------------------------------------------------- 2. CAMPAIGN
    A["complab_campaign"] = Action(
        "complab_campaign", "Build the CompLaB simulation campaign",
        "Writes one input file per simulation. Nothing is recompiled between runs.",
        os.path.join(TOOLS, "complab_campaign.py"),
        [Field("cmd", "What to do", "subcommand", "build",
               "'build' writes the campaign. 'status' counts how many runs have "
               "finished, failed or not started. 'retry' writes a second "
               "submission script holding only the runs that failed.",
               choices=["build", "status", "retry"]),
         Field("out", "Campaign folder", "dir", os.path.join(WORK, "complab_campaign"),
               "Where the per-run folders and batch scripts are written. For "
               "'status' and 'retry' this is the folder to read.", flag="--out"),
         Field("geometries", "Geometry folder", "dir",
               os.path.join(WORK, "geometries"),
               "The folder produced by the previous step.", flag="--geometries",
               depends=("cmd", "build")),
         Field("complab", "The CompLaB program", "file", "",
               "The compiled CompLaB executable that will run each simulation. "
               "This is the simulator itself, not part of this project; it is "
               "whatever your build produced, usually a file with no extension "
               "sitting in CompLaB's own build folder. Nothing is run now -- the "
               "path is written into the submission scripts.",
               flag="--complab", depends=("cmd", "build")),
         Field("per_geom", "Simulations per structure", "int", 20,
               "How many Peclet and Damkohler combinations to run on each "
               "structure.", flag="--per-geom", depends=("cmd", "build")),
         Field("batches", "Split into how many batches", "int", 5,
               "Batches are split by whole geometries, never mid-geometry, so the "
               "training and test split stays clean.", flag="--batches",
               depends=("cmd", "build")),
         Field("cores", "Processors per simulation", "int", 8,
               "CompLaB runs each simulation across this many processors.",
               flag="--cores"),
         Field("time", "Time limit per simulation", "text", "04:00:00",
               "Hours:minutes:seconds. A run that exceeds it is killed and "
               "reported, not silently truncated.", flag="--time"),
         Field("partition", "Cluster queue", "text", "batch",
               "The Slurm partition to submit to. Leave as it is unless your "
               "cluster uses different names.", flag="--partition")],
        [("geometry folder", "every structure the campaign will run on"),
         ("the CompLaB program", "the compiled simulator. It is not run now; its "
          "path is written into each submission script so the cluster can find "
          "it later. Without it there is nothing to submit, which is why 'build' "
          "insists on having it.")],
        ["Chooses the parameter grid: 4 Peclet values times 3 Damkohler values, "
         "so 12 runs per geometry.",
         "Converts each dimensionless pair into the physical rate constants "
         "CompLaB actually accepts, because CompLaB has no field for a Damkohler "
         "number: it only takes rate constants.",
         "Substitutes them into CompLaB.xml through the PRT_ placeholders, one XML "
         "per run.",
         "Writes submission scripts so the whole campaign runs unattended, and a "
         "failed run is skipped and reported rather than bringing the batch down."],
        [("run_NNNN/CompLaB.xml", "one ready-to-run input file per simulation"),
         ("batch_N.sh", "the submission script for each batch"),
         ("campaign.csv", "the full list of runs with their parameters")],
        group="Simulate")

    # ---------------------------------------------------------- 3. COLLECT
    A["collect_complab_output"] = Action(
        "collect_complab_output", "Collect CompLaB output into a dataset",
        "Gathers thousands of .vti files into the single file training reads. "
        "Works for CompLaB2D and CompLaB3D alike.",
        os.path.join(TOOLS, "collect_complab_output.py"),
        [Field("complab_campaign", "Campaign folder", "dir", os.path.join(WORK, "complab_campaign"),
               "The folder holding the finished runs. It may hold runs/ directly, "
               "or several batch_* subfolders, which are merged into one dataset.",
               flag="--campaign"),
         Field("geometries", "Geometry folder", "dir",
               os.path.join(WORK, "geometries"), "", flag="--geometries"),
         Field("out", "Folder to write the dataset into", "dir",
               os.path.join(NEW3D),
               "The dataset itself is written as dataset.h5 inside this "
               "folder, alongside the failure report.", flag="--out"),
         Field("mode", "Which snapshots to keep", "choice", "transient",
               "'transient' keeps the whole time series, 'steady' keeps only the "
               "final state.", choices=["transient", "steady"], flag="--mode"),
         Field("n_times", "Resample to this many snapshots", "int", 21,
               "Runs stop at different iterations, so they hold different numbers "
               "of snapshots. This puts every run on a common normalised time "
               "axis, which is what makes the trunk's time input mean the same "
               "thing in every run. Leave blank to keep them all.",
               flag="--n-times", optional=True),
         Field("no_velocity", "Skip the flow field", "bool", False,
               "Leave this OFF. Without the flow field the flow-proxy and "
               "dimension-free switches cannot be used at all.",
               flag="--no-velocity")],
        [("campaign folder", "the finished CompLaB runs, 2D or 3D. A 2D campaign "
          "is simply one whose grid has nz = 1, and everything downstream adapts "
          "to it on its own"),
         ("geometry folder", "so the geometry and its distance fields travel with "
          "the data")],
        ["Reads every .vti output file and stacks the concentration fields into "
         "one array per run. Runs that crashed, or that produced NaN or all-zero "
         "fields, are SKIPPED AND REPORTED rather than bringing the batch down.",
         "Reads the flow field from each run as well. This matters beyond "
         "curiosity: without it the flow-proxy and dimension-free switches "
         "cannot be used at all, so a loud warning is printed if none is found.",
         "Puts every run on a common normalised time axis, so runs of different "
         "length can be trained on together.",
         "Stores the flow field once per run and the geometry once per structure, "
         "rather than repeating them.",
         "Records which structure each sample came from, which is what lets the "
         "train and test split be made BY GEOMETRY. Splitting by sample would leak "
         "pore structure between the two and inflate the score."],
        [("dataset.h5",
          "everything training needs: geometry, geodesic distance, Euclidean "
          "distance, flow field, concentrations over time, and the dimensionless "
          "parameters of every run"),
         ("failures.csv", "one row per rejected run, with the specific reason"),
         ("campaign_report.md",
          "a readable summary: how many runs were usable, and why the rest were "
          "not")],
        group="Simulate")

    # ---------------------------------------------------------- 3b. PRACTICE
    # ---------------------------------------------------------- 4. 2D BRIDGE
    A["ingest2d"] = Action(
        "ingest2d", "Build the 2D transfer set",
        "Turns 2D domains into extra 3D training data, for free.",
        os.path.join(DIR_BRIDGE, "build_transfer_set.py"),
        [Field("out", "Save the transfer set as", "savefile",
               os.path.join(WORK, "transfer_2d_to_3d.h5"), "", flag="--out",
               filetypes=[("HDF5 dataset", "*.h5")]),
         Field("limit", "How many 2D domains to use", "int", 200,
               "The 2D folder holds 3000 of them.", flag="--limit"),
         Field("shape", "Grid size (must match the 3D dataset)", "text", "128 64 64",
               "If this does not match exactly, training will refuse to mix them.",
               flag="--target-shape", nargs=3),
         Field("n_sets", "Parameter sets per domain", "int", 3, "", flag="--n-sets"),
         Field("n_times", "Snapshots in time", "int", 21, "", flag="--n-times"),
         Field("n_species", "How many chemicals", "int", 4,
               "Must match the 3D dataset.", flag="--n-species"),
         Field("nz_solve", "Solve on this many z layers", "int", 8,
               "The extruded sample does not vary in z, so solving a thin slab and "
               "tiling it is EXACT and about eight times cheaper.", flag="--nz-solve"),
         Field("synthetic", "Or: invent this many domains instead", "int", "",
               "Leave blank to use the real 2D folder. Fill in a number to "
               "synthesise domains of the same morphology, which is useful for "
               "testing.", flag="--synthetic", optional=True)],
        [("2D/Domains/*.dat", "3000 two-dimensional pore domains from the published release")],
        ["Reads each 2D domain, detecting the array ordering rather than assuming "
         "it. The .dat and .npz files in the release use OPPOSITE orderings, and "
         "getting it wrong shreds the pore network into disconnected pieces "
         "silently.",
         "Translates the voxel codes: the 2D release uses 0 solid, 1 pore, "
         "2 interface, while CompLaB uses 0 solid, 1 wall, 2 pore.",
         "Resamples to your target width and checks the domain still percolates.",
         "EXTRUDES it along z. The extruded sample is prismatic, so nothing varies "
         "in z and the exact 3D solution IS the 2D solution repeated. This is not "
         "an approximation.",
         "Solves flow and transport on a thin slab and tiles the result, then "
         "MEASURES the z-variation and reports it. On the real domains it comes "
         "out at exactly zero."],
        [("train2d.h5",
          "a dataset in the same format as the 3D one, which training can mix in "
          "with the 'Use 2D data' switch")],
        group="Simulate",
        note="About ten seconds per domain, so 200 domains is roughly half an hour.")

    # ---------------------------------------------------------- 5. TRAIN
    A["train"] = Action(
        "train", "Train the network",
        "Fits the surrogate so it can predict concentrations without simulating.",
        os.path.join(MODEL, "train.py"),
        [Field("data", "Dataset", "file", "",
               "The .h5 file from 'Collect' or from 'Make a practice dataset'.",
               flag="--data", filetypes=[("HDF5 dataset", "*.h5")]),
         Field("out", "Folder for the result", "dir", os.path.join(NEW3D, "model"),
               "The trained weights and the training history go here.", flag="--out"),
         Field("epochs", "How many passes over the data", "int", 300,
               "Training stops early if the held-out error stops improving.",
               flag="--epochs"),
         Field("batch_size", "Samples per step", "int", 8,
               "Larger uses more memory and trains slightly faster.", flag="--batch-size"),
         Field("n_points", "Voxels sampled per step", "int", 8192,
               "The network is asked about this many pore voxels at a time. "
               "If you ask for more than the structure has, it simply uses all "
               "of them -- nothing is duplicated and nothing fails. A 148 x 64 "
               "2D domain holds only about 7,000 pore voxels, so anything above "
               "that means 'the whole domain every step'. A 64 x 64 x 64 3D "
               "domain holds of order 100,000, where the number really is a "
               "sample and 8192 is a reasonable one.",
               flag="--n-points"),
         Field("lr", "Learning rate", "float", 1e-3,
               "How big a correction each step makes. 0.001 is the usual choice.",
               flag="--lr"),
         Field("patience", "Stop after this many bad epochs", "int", 20, "",
               flag="--patience"),
         Field("distance", "What the network uses to sense geometry", "choice", "gdf",
               "'gdf' is the geodesic distance, the shortest path through pore "
               "space from the inlet. 'edt' is the straight-line distance, kept as "
               "the control that shows the geodesic one is what matters. 'none' "
               "removes it entirely.",
               choices=["gdf", "edt", "none"], flag="--distance"),
         Field("with_velocity", "Also show it the flow field", "bool", False,
               "Adds the three velocity components as extra input channels, "
               "keeping the geometry as well.", flag="--with-velocity"),
         Field("SW_A", "SWITCH A — use flow instead of geometry", "bool", False,
               "Replaces the geodesic distance with the advective travel time, and "
               "the pore mask with the velocity field. The geometry problem is then "
               "somebody else's and we only own the chemistry.", flag="--flow-proxy"),
         Field("flow_mode", "   what the flow switch feeds in", "choice", "tau",
               "'tau' is the travel time. 'speed' is the local speed. 'both' gives "
               "the network the geodesic distance AND the travel time, which is the "
               "safe option at low Peclet.",
               choices=["tau", "speed", "both"], flag="--flow-mode", depends="SW_A"),
         Field("keep_geom", "   keep the pore mask as well", "bool", False, "",
               flag="--keep-geometry-channel", depends="SW_A"),
         Field("SW_B", "SWITCH B — mix in 2D data", "bool", False,
               "Adds extruded 2D domains to training. 3D runs cost hours each; 2D "
               "domains are already sitting on disk.", flag=None),
         Field("transfer_2d", "   2D transfer set", "file",
               os.path.join(WORK, "transfer_2d_to_3d.h5"),
               "The file built by 'Build the 2D transfer set'.",
               flag="--transfer-2d", depends="SW_B",
               filetypes=[("HDF5 dataset", "*.h5")]),
         Field("transfer_frac", "   fraction of training from 2D", "float", 0.3,
               "Keep this at or below 0.3. Higher and the network learns the "
               "shortcut that nothing ever happens in the third direction.",
               flag="--transfer-2d-frac", depends="SW_B"),
         Field("SW_C", "SWITCH C — dimension-free coordinates", "bool", False,
               "Replaces the x, y, z, time, geodesic coordinates with travel time, "
               "wall distance and time. That is three numbers in BOTH 2D and 3D, "
               "so the same network fits both. Turns on switch A automatically.",
               flag="--dim-free"),
         Field("init_from", "Start from an existing model", "file", "",
               "Warm-start from a checkpoint, for example one trained on 2D data.",
               flag="--init-from", optional=True,
               filetypes=[("Checkpoint", "*.pt")]),
         Field("freeze_trunk", "   and freeze the reaction part", "bool", False,
               "The trunk holds the chemistry, which is the same in 2D and 3D. The "
               "branch holds the pore structure, which is not. Freezing the trunk "
               "and retraining only the branch is the right way round.",
               flag="--freeze-trunk"),
         Field("workers", "Loader processes", "int", 4, "", flag="--workers"),
         Field("seed", "Random seed", "int", 42,
               "Fixing this makes the run reproducible.", flag="--seed")],
        [("the dataset .h5", "simulation output: geometry, flow, concentrations, parameters")],
        ["Splits the data by GEOMETRY, holding out whole structures. Splitting by "
         "sample would let the network see the same pore structure in training and "
         "in testing, and the score would be meaningless.",
         "Shows the network three things at once: the pore structure through a 3D "
         "convolutional encoder (the 'branch'), the dimensionless numbers Pe and Da "
         "through a small dense network (the 'parameter branch'), and a list of "
         "voxel coordinates through the 'trunk'.",
         "Multiplies the two branch outputs together and takes their dot product "
         "with the trunk. That product is the predicted concentration at each voxel.",
         "Compares against the simulation, adjusts the weights, and repeats. The "
         "held-out error is checked every epoch and the best weights are kept."],
        [("best.pt",
          "the trained network. This is the thing that replaces the simulation."),
         ("summary.json",
          "the final held-out error per chemical, plus the full training history")],
        group="Learn",
        progress_re=re.compile(r"^epoch\s+(\d+)"),
        total_re="epochs",
        metric_re=re.compile(r"^epoch\s+(\d+)\s+train\s+([\d.eE+-]+)\s+test\s+([\d.eE+-]+)"))

    # ---------------------------------------------------------- 6. EVALUATE
    A["evaluate"] = Action(
        "evaluate", "Evaluate a trained network",
        "Scores it on structures it has never seen, and draws the comparison figures.",
        os.path.join(MODEL, "evaluate.py"),
        [Field("checkpoint", "Trained model", "file", "",
               "The best.pt written by training.", flag="--checkpoint",
               filetypes=[("Checkpoint", "*.pt")]),
         Field("data", "Dataset", "file", "",
               "The same dataset it was trained on. The held-out structures are "
               "re-derived from the same seed.", flag="--data",
               filetypes=[("HDF5 dataset", "*.h5")]),
         Field("out", "Folder for the figures", "dir", os.path.join(NEW3D, "figures"),
               "", flag="--out"),
         Field("compare", "Also compare these models", "text", "",
               "Space-separated LABEL=path/to/best.pt entries, to put several "
               "models on the same axes. Each entry is its own word, so a path "
               "with a space in it will not work here.",
               flag="--compare", optional=True, nargs="*"),
         Field("save_fields", "How many field figures to draw", "int", 4, "",
               flag="--save-fields"),
         Field("no_3d", "Skip the 3D renders", "bool", False,
               "The 3D renders are the slow part.", flag="--no-3d")],
        [("best.pt", "the trained network"),
         ("the dataset .h5", "so the held-out structures and the truth are available")],
        ["Runs the network over EVERY pore voxel of every held-out structure, not "
         "a sample of them.",
         "Computes the root-mean-square error per chemical, and the R-squared, "
         "against the simulation.",
         "Draws truth against prediction as slices and as 3D renders, plus the "
         "reaction rate fields derived from both.",
         "Plots the error against Peclet and Damkohler, so you can see WHERE the "
         "network is weak rather than only how weak it is on average."],
        [("rmse_table.csv", "one row per held-out sample, with its parameters"),
         ("metrics.json", "the summary numbers"),
         ("fields_*.png", "truth, prediction and error side by side"),
         ("physics_*.png", "the same for the reaction rate fields"),
         ("rmse_vs_params.png", "where in the parameter space the model is weak")],
        group="Learn")

    # ---------------------------------------------------------- 7. PREDICT
    A["predict"] = Action(
        "predict", "Predict on a new geometry",
        "Uses the trained network on a structure that was never simulated.",
        os.path.join(MODEL, "predict.py"),
        [Field("checkpoint", "Trained model", "file", "", "", flag="--checkpoint",
               filetypes=[("Checkpoint", "*.pt")]),
         Field("geometry", "New geometry", "file", "",
               "A geom_NNNN.npz, or a raw geometry.dat with the grid size given "
               "below. A .npz already carries its own grid size; a .dat is a "
               "bare list of numbers and does not, so for a .dat the three "
               "boxes below have to be filled in.",
               flag="--geometry",
               filetypes=[("Geometry", "*.npz *.dat"), ("All files", "*.*")]),
         Field("nx", "Voxels along x", "int", "",
               "Only for a raw .dat. Leave blank for a .npz.",
               flag="--nx", optional=True),
         Field("ny", "Voxels along y", "int", "", "", flag="--ny", optional=True),
         Field("nz", "Voxels along z", "int", "",
               "Set this to 1 for a two-dimensional structure.",
               flag="--nz", optional=True),
         Field("velocity", "Flow field", "file", "",
               "Required only if the model was trained with switch A or C, which "
               "replace the geometry with the flow. Get it from a Stokes solve, or "
               "from a flow surrogate.", flag="--velocity", optional=True,
               filetypes=[("NumPy array", "*.npy")]),
         Field("pe", "Peclet number", "float", 10.0,
               "How much faster the flow carries a chemical than diffusion spreads it.",
               flag="--pe"),
         Field("da_bio", "Biotic Damkohler", "float", 1.0,
               "How fast the microbes react compared with how fast the fluid crosses "
               "the sample.", flag="--da-bio"),
         Field("da_abio", "Abiotic Damkohler", "float", 1.0,
               "The same comparison for the chemical reaction.", flag="--da-abio"),
         Field("t_norm", "Moment in time (0 to 1)", "float", 1.0,
               "1.0 is the end of the simulation.", flag="--t-norm"),
         Field("t_series", "Or predict this many moments", "int", "",
               "Leave blank for a single snapshot.", flag="--t-series", optional=True),
         Field("out", "Folder for the output", "dir", os.path.join(NEW3D, "predictions"),
               "", flag="--out"),
         Field("no_vti", "Skip writing .vti files", "bool", False,
               "The .vti files are what you would open in ParaView.", flag="--no-vti")],
        [("best.pt", "the trained network"),
         ("a geometry file", "the new pore structure"),
         ("Pe and Da", "the flow and reaction conditions you want to ask about")],
        ["Loads the network and checks the geometry is consistent with what it was "
         "trained on.",
         "Builds the same inputs training used: the pore structure, the "
         "dimensionless numbers, and every pore voxel's coordinates.",
         "Runs the network once. This takes under a second, against hours for the "
         "simulation it replaces.",
         "Reconstructs full 3D volumes from the per-voxel predictions."],
        [("pred_*.png", "slices and 3D renders of every predicted chemical"),
         ("*.vti", "volumes you can open in ParaView"),
         ("prediction.npz", "the raw arrays")],
        group="Learn")

    # ---------------------------------------------------------- 8. SWEEP
    A["sweep"] = Action(
        "sweep", "Run the switch comparison",
        "Trains one model per configuration and tells you which wins, and where.",
        os.path.join(MODEL, "run_ablation_sweep.py"),
        [Field("data", "3D dataset", "file", "", "", flag="--data",
               filetypes=[("HDF5 dataset", "*.h5")]),
         Field("data_2d", "2D transfer set", "file",
               os.path.join(WORK, "transfer_2d_to_3d.h5"),
               "Leave blank to skip the 2D rows.", flag="--data-2d", optional=True,
               filetypes=[("HDF5 dataset", "*.h5")]),
         Field("out", "Folder for the sweep", "dir", os.path.join(NEW3D, "switch_comparison"),
               "", flag="--out"),
         Field("epochs", "Epochs per configuration", "int", 300, "", flag="--epochs"),
         Field("quick", "Quick plumbing check only", "bool", False,
               "Three epochs, tiny batches. Proves it runs; proves nothing else.",
               flag="--quick"),
         Field("skip_train", "Reuse models already in that folder", "bool", False,
               "", flag="--skip-train")],
        [("3D dataset", "the real data"),
         ("2D transfer set", "optional, enables the two 2D rows")],
        ["Trains a separate model for each configuration: the baseline, the "
         "Euclidean control, flow-with-travel-time, flow-with-both, "
         "dimension-free, and the two 2D-transfer variants.",
         "Evaluates every one of them on the SAME held-out structures, so the "
         "comparison is fair.",
         "Splits the error by Peclet band, which is the number that actually "
         "settles the question. The flow field should win at high Peclet and lose "
         "at Pe below 1, where the chemical arrives by diffusion and a stagnant "
         "velocity field cannot tell a dead-end pore from solid rock."],
        [("sweep_summary.txt", "the comparison table, split by Peclet band"),
         ("compare/rmse_table.csv", "every sample of every model"),
         ("<config>/best.pt", "one trained model per configuration")],
        group="Learn",
        note="This trains several models in sequence. Budget accordingly.")

    # ------------------------------------------------------- 9b. CHECK ALL
    A["checkall"] = Action(
        "checkall", "Check everything at once",
        "Runs every self-check in the project and prints one summary.",
        os.path.join(ROOT, "check_everything.py"),
        [],
        [("nothing", "it needs no input files and writes none")],
        ["Checks the transport solver against answers that can be written down "
         "without it: plug flow through an open channel must leave exp(-Da) at "
         "the outlet, nothing anywhere may exceed the inlet concentration, and "
         "a sealed half of a domain must stay exactly empty. That family of "
         "checks is what catches a boundary that wraps around, a time step "
         "sitting on its stability limit, or a Peclet number that is silently "
         "the grid width times the one you asked for.",
         "Checks that every switch turned OFF reproduces the original code bit "
         "for bit, by re-implementing the original inside the test.",
         "Checks the numbers quoted in the documentation against measurement.",
         "Checks that every button in this window emits a command line its "
         "script actually accepts, by asking the real argument parsers rather "
         "than a copy of them.",
         "Checks that every page of this window can be built and every view "
         "drawn."],
        [("nothing on disk", "the result is the summary table in the log. "
          "Anything that fails prints its full output underneath.")],
        group="Setup",
        note="Run this after installing, after moving the folder, and any time "
             "something behaves oddly. It takes under a minute and it is the "
             "fastest way to tell a broken installation from a broken idea.")

    return A


# =============================================================================
#  RUNNER — subprocess with live output
# =============================================================================
class Runner:
    def __init__(self, on_line, on_done):
        self.on_line = on_line
        self.on_done = on_done
        self.proc = None
        self.q = queue.Queue()
        self.t0 = None
        self._active = False        # set SYNCHRONOUSLY, see below

    def busy(self):
        # Not "self.proc is not None": the worker thread has not assigned proc
        # yet in the first milliseconds, so a fast second click would start a
        # second process on top of the first.  The flag is set before the thread
        # is even created.
        return self._active

    def start(self, cmd, cwd=None):
        if self.busy():
            return False
        self._active = True
        self.t0 = time.time()

        def work():
            try:
                env = dict(os.environ)
                env["PYTHONUNBUFFERED"] = "1"
                env["MPLBACKEND"] = "Agg"
                self.proc = subprocess.Popen(
                    cmd, cwd=cwd, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
                for line in self.proc.stdout:
                    self.q.put(("line", line.rstrip("\n")))
                self.proc.wait()
                self.q.put(("done", self.proc.returncode))
            except Exception as e:                       # noqa: BLE001
                self.q.put(("line", "LAUNCH FAILED: %s" % e))
                self.q.put(("done", -1))
            finally:
                self._active = False

        threading.Thread(target=work, daemon=True).start()
        return True

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:                            # noqa: BLE001
                pass

    def pump(self):
        n = 0
        while n < 400:
            try:
                kind, payload = self.q.get_nowait()
            except queue.Empty:
                break
            n += 1
            if kind == "line":
                self.on_line(payload)
            else:
                self.on_done(payload, time.time() - (self.t0 or time.time()))


# =============================================================================
#  SMALL WIDGET HELPERS
# =============================================================================
def hline(parent):
    f = tk.Frame(parent, height=1, bg=LINE)
    f.pack(fill="x", pady=6)
    return f


class InfoBox(tk.Frame):
    """One of the three coloured explanation boxes on every action page."""

    def __init__(self, parent, title, colour, items, numbered=False):
        super().__init__(parent, bg=PANEL, highlightbackground=LINE,
                         highlightthickness=1)
        self._msgs = []
        head = tk.Frame(self, bg=colour, height=22)
        head.pack(fill="x")
        tk.Label(head, text=title, bg=colour, fg="#ffffff", anchor="w",
                 font=base_font(9, True), padx=8, pady=3).pack(fill="x")
        body = tk.Frame(self, bg=PANEL)
        body.pack(fill="both", expand=True, padx=8, pady=6)
        for i, it in enumerate(items):
            row = tk.Frame(body, bg=PANEL)
            row.pack(fill="x", anchor="w", pady=1)
            bullet = "%d." % (i + 1) if numbered else "•"
            tk.Label(row, text=bullet, bg=PANEL, fg=MUTE, font=base_font(9),
                     width=2, anchor="nw").pack(side="left", anchor="n")
            if isinstance(it, tuple):
                name, expl = it
                txt = tk.Frame(row, bg=PANEL)
                txt.pack(side="left", fill="x", expand=True)
                tk.Label(txt, text=name, bg=PANEL, fg=INK, font=mono_font(9),
                         anchor="w", justify="left").pack(fill="x")
                w = tk.Message(txt, text=expl, bg=PANEL, fg=MUTE,
                               font=base_font(8), anchor="w", justify="left",
                               width=520)
                w.pack(fill="x")
                self._msgs.append((w, 46))
            else:
                w = tk.Message(row, text=it, bg=PANEL, fg=INK, font=base_font(9),
                               anchor="w", justify="left", width=560)
                w.pack(side="left", fill="x", expand=True)
                self._msgs.append((w, 30))

    def set_width(self, px):
        for w, pad in self._msgs:
            try:
                w.configure(width=max(200, px - pad))
            except Exception:                                  # noqa: BLE001
                pass


class ActionPage(tk.Frame):
    """The page for one Action: settings on the left, the three boxes on the
    right, the exact command at the bottom, and a Run button."""

    def __init__(self, parent, app, action):
        super().__init__(parent, bg=BG)
        self.app = app
        self.action = action

        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(top, text=action.title, bg=BG, fg=INK,
                 font=base_font(13, True), anchor="w").pack(fill="x")
        tk.Label(top, text=action.one_line, bg=BG, fg=MUTE,
                 font=base_font(9), anchor="w").pack(fill="x")
        if action.note:
            tk.Label(top, text="Note:  " + action.note, bg=BG, fg=INFOC,
                     font=base_font(8), anchor="w").pack(fill="x", pady=(3, 0))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)
        body.columnconfigure(0, weight=3, minsize=380)
        body.columnconfigure(1, weight=4, minsize=380)
        body.rowconfigure(0, weight=1)

        # ---------------- left: the settings ---------------------------------
        left = tk.Frame(body, bg=PANEL, highlightbackground=LINE,
                        highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(left, text="SETTINGS", bg=HEAD, fg=INK, anchor="w",
                 font=base_font(9, True), padx=8, pady=4).pack(fill="x")
        canvas = tk.Canvas(left, bg=PANEL, highlightthickness=0, width=380)
        sb = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=PANEL)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        self.rows = {}
        self.computed = []
        for f in action.fields:
            self.rows[f.key] = self._field_row(inner, f)
        self._sync_depends()

        # ---------------- right: what goes in / happens / comes out ----------
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        rc = tk.Canvas(right, bg=BG, highlightthickness=0)
        rsb = ttk.Scrollbar(right, orient="vertical", command=rc.yview)
        rin = tk.Frame(rc, bg=BG)
        rin.bind("<Configure>",
                 lambda e: rc.configure(scrollregion=rc.bbox("all")))
        _win = rc.create_window((0, 0), window=rin, anchor="nw", width=560)
        rc.configure(yscrollcommand=rsb.set)
        rc.pack(side="left", fill="both", expand=True)
        rsb.pack(side="right", fill="y")

        self.boxes = [
            InfoBox(rin, "WHAT GOES IN", INFOC, action.inputs_text),
            InfoBox(rin, "WHAT HAPPENS", OKC, action.happens_text, numbered=True),
            InfoBox(rin, "WHAT COMES OUT", WARNC, action.outputs_text)]
        for b in self.boxes:
            b.pack(fill="x", pady=(0, 6))

        def _fit(e):
            rc.itemconfigure(_win, width=e.width)
            for b in self.boxes:
                b.set_width(e.width)
        rc.bind("<Configure>", _fit)

        # ---------------- bottom: the exact command --------------------------
        bot = tk.Frame(self, bg=BG)
        bot.pack(fill="x", padx=10, pady=(4, 10))
        cmdbar = tk.Frame(bot, bg=BG)
        cmdbar.pack(fill="x")
        tk.Label(cmdbar, text="COMMAND THIS WILL RUN", bg=BG, fg=MUTE,
                 font=base_font(8, True), anchor="w").pack(side="left")
        tk.Button(cmdbar, text="Copy", font=base_font(8), padx=8,
                  command=self.copy_cmd).pack(side="right")
        tk.Button(cmdbar, text="Refresh", font=base_font(8), padx=8,
                  command=self.refresh_cmd).pack(side="right", padx=4)
        self.cmdbox = tk.Text(bot, height=3, font=mono_font(8), bg="#f4f4f2",
                              fg=INK, wrap="word", relief="solid", bd=1)
        self.cmdbox.pack(fill="x", pady=(2, 6))

        btns = tk.Frame(bot, bg=BG)
        btns.pack(fill="x")
        self.run_btn = tk.Button(btns, text="  ▶  Run  ", font=base_font(10, True),
                                 bg="#e8ecf1", command=self.run, padx=12, pady=4)
        self.run_btn.pack(side="left")
        tk.Button(btns, text="Stop", font=base_font(9), padx=10,
                  command=self.app.stop_run).pack(side="left", padx=6)
        tk.Button(btns, text="Open output folder", font=base_font(9), padx=10,
                  command=self.open_out).pack(side="left", padx=6)
        tk.Button(btns, text="Reset settings", font=base_font(9), padx=10,
                  command=self.reset).pack(side="left")
        self.refresh_cmd()

    # ---------------------------------------------------------------- fields
    def _field_row(self, parent, f):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=8, pady=(6, 0), anchor="w")
        f.make_var()
        is_switch = f.key.startswith("SW_")
        lab_font = base_font(9, True) if is_switch else base_font(9)
        lab_fg = INFOC if is_switch else INK

        if f.kind == "computed":
            # Not an input. It answers the question the two boxes above it
            # raise and neither of them answers: how many simulations is that,
            # in total? It updates as you type.
            lab = tk.Label(row, text="", bg=HEAD, fg=INK, anchor="w",
                           font=base_font(9, True), padx=6, pady=4,
                           justify="left")
            lab.pack(fill="x", pady=(6, 0))
            self.computed.append(lab)
            self._update_computed()
            return row
        if f.kind == "chemicals":
            self._chem_widget(row, f)
            return row
        if f.kind == "heading":
            tk.Label(row, text=f.label.upper(), bg=HEAD, fg=INK, anchor="w",
                     font=base_font(9, True), padx=6, pady=3).pack(
                         fill="x", pady=(8, 0))
            if f.help:
                tk.Message(row, text=f.help, bg=PANEL, fg=MUTE,
                           font=base_font(8), width=350, anchor="w",
                           justify="left").pack(fill="x")
            return row
        if f.kind == "bool":
            cb = tk.Checkbutton(row, text=f.label, variable=f.var, bg=PANEL,
                                fg=lab_fg, font=lab_font, anchor="w",
                                activebackground=PANEL, highlightthickness=0,
                                command=lambda: (self._sync_depends(), self.refresh_cmd()))
            cb.pack(fill="x", anchor="w")
        else:
            tk.Label(row, text=f.label, bg=PANEL, fg=lab_fg, font=lab_font,
                     anchor="w").pack(fill="x")
            ent = tk.Frame(row, bg=PANEL)
            ent.pack(fill="x")
            if f.kind in ("choice", "subcommand"):
                w = ttk.Combobox(ent, textvariable=f.var, values=f.choices,
                                 state="readonly", font=base_font(9), width=22)
                w.pack(side="left", fill="x", expand=True)
                w.bind("<<ComboboxSelected>>",
                       lambda e: (self._sync_depends(), self.refresh_cmd()))
            else:
                w = tk.Entry(ent, textvariable=f.var, font=mono_font(9),
                             relief="solid", bd=1)
                w.pack(side="left", fill="x", expand=True)
                if f.key == "ph_bfcode":
                    w.bind("<KeyRelease>",
                           lambda e: (self._sync_depends(), self.refresh_cmd()))
                else:
                    w.bind("<KeyRelease>", lambda e: self.refresh_cmd())
                if f.kind in ("file", "files"):
                    tk.Button(ent, text="...", font=base_font(8), padx=4,
                              command=lambda ff=f: self._pick_open(ff)).pack(side="left", padx=(3, 0))
                elif f.kind == "savefile":
                    tk.Button(ent, text="...", font=base_font(8), padx=4,
                              command=lambda ff=f: self._pick_save(ff)).pack(side="left", padx=(3, 0))
                elif f.kind == "dir":
                    tk.Button(ent, text="...", font=base_font(8), padx=4,
                              command=lambda ff=f: self._pick_dir(ff)).pack(side="left", padx=(3, 0))
        if f.help:
            tk.Message(row, text=f.help, bg=PANEL, fg=MUTE, font=base_font(8),
                       width=350, anchor="w", justify="left").pack(fill="x")
        return row

    def _update_computed(self):
        """The number the page is really about: rocks times conditions."""
        if not getattr(self, "computed", None):
            return
        def num(key, default):
            fl = next((x for x in self.action.fields if x.key == key), None)
            try:
                return max(int(float(str(fl.value()).strip())), 0)
            except Exception:                                  # noqa: BLE001
                return default
        ng, ns = num("n_geom", 0), num("n_sets", 0)
        txt = ("THAT IS %d SIMULATIONS:  %d rocks  x  %d conditions each.\n"
               "%d flow solves, one per rock, and %d transport solves, one "
               "per simulation." % (ng * ns, ng, ns, ng, ng * ns))
        for lab in self.computed:
            lab.configure(text=txt)

    # ---------------------------------------------------------- chemicals
    def _chem_widget(self, row, f):
        """One block per chemical, with a plus and a minus button under them.

        Rebuilt from scratch whenever the count changes. That is wasteful and
        it is the right trade: the alternative is keeping widgets and row
        indices in step by hand, and the role labels depend on position, so a
        stale index would mislabel the biomass as the product.
        """
        tk.Label(row, text="THE CHEMICALS", bg=HEAD, fg=INK, anchor="w",
                 font=base_font(9, True), padx=6, pady=3).pack(
                     fill="x", pady=(8, 0))
        tk.Message(row, text=f.help, bg=PANEL, fg=MUTE, font=base_font(8),
                   width=350, anchor="w", justify="left").pack(fill="x")
        holder = tk.Frame(row, bg=PANEL)
        holder.pack(fill="x")

        def setting_now(key, default=""):
            """What another box on this page currently says."""
            g = next((x for x in self.action.fields if x.key == key), None)
            return str(g.value()).strip() if g is not None else default

        def role(i):
            microbes = setting_now("ph_mode", "biotic only") != "abiotic only"
            if i == 3 and not microbes:
                return ("the biomass -- but there are no microbes in this "
                        "run, so it only decays")
            if i < len(CHEM_ROLES):
                return CHEM_ROLES[i]
            return "a tracer: carried, but takes no part in any reaction"

        def draw():
            for w in holder.winfo_children():
                w.destroy()
            for i, r in enumerate(f.rows):
                blk = tk.Frame(holder, bg=PANEL, highlightbackground=LINE,
                               highlightthickness=1)
                blk.pack(fill="x", pady=(6, 0))
                tk.Label(blk, text="  %d.  %s" % (i + 1, role(i)), bg=PANEL,
                         fg=INFOC, font=base_font(8, True), anchor="w").pack(
                             fill="x", pady=(3, 0))
                grid = tk.Frame(blk, bg=PANEL)
                grid.pack(fill="x", padx=4, pady=3)
                for col in range(3):
                    grid.columnconfigure(col, weight=1, uniform="chem")
                for n, key in enumerate(CHEM_KEYS):
                    # Two controls for one thing is how they come to
                    # disagree. Whether the biomass moves is set by "The
                    # microbes are" in the reaction section, because it is a
                    # property of the system rather than a line in a table,
                    # and that setting overrides this box -- so showing an
                    # editable box here would be showing a lie.
                    locked = (key == "immobile" and i == 3)
                    # D IN BIOFILM only does something when there ARE biofilm
                    # voxels, and there are only biofilm voxels when microbes
                    # are running AND a biofilm code has been given. Without
                    # both, the pore coefficient is used everywhere and this
                    # box is decoration. It was decoration for a long time
                    # before anything could act on it, which is exactly why it
                    # should say so rather than sit there looking editable.
                    microbes = setting_now("ph_mode",
                                           "biotic only") != "abiotic only"
                    has_bf = bool(setting_now("ph_bfcode"))
                    no_biofilm = (key == "d_biofilm"
                                  and not (microbes and has_bf))
                    # An outlet VALUE means nothing unless the outlet is
                    # Dirichlet. CompLaB reads the tag and never uses it when
                    # the type is Neumann; every substrate in a real
                    # CompLaB.xml carries a right_boundary_condition that is
                    # dead data.
                    dead = (key == "right_value"
                            and str(r["right_type"].get()) != "Dirichlet")
                    gr, gc = divmod(n, 3)
                    cell = tk.Frame(grid, bg=PANEL)
                    cell.grid(row=gr, column=gc, sticky="ew", padx=2, pady=2)
                    lbl = CHEM_LABELS[key]
                    if locked:
                        lbl = "Attached  (set by 'The microbes are')"
                    elif dead:
                        lbl = "Outlet value  (unused: outlet is Neumann)"
                    elif no_biofilm:
                        lbl = ("D in biofilm  (unused: no microbes)"
                               if not microbes else
                               "D in biofilm  (unused: no biofilm code set)")
                    tk.Label(cell, text=lbl, bg=PANEL,
                             fg=(LINE if (locked or dead or no_biofilm)
                                 else MUTE),
                             font=base_font(7), anchor="w").pack(fill="x")
                    if key in CHEM_CHOICES:
                        w = ttk.Combobox(cell, textvariable=r[key],
                                         values=CHEM_CHOICES[key],
                                         state="disabled" if locked
                                         else "readonly", font=base_font(8),
                                         width=9)
                        w.pack(fill="x")
                        w.bind("<<ComboboxSelected>>",
                               lambda e: (draw(), self.refresh_cmd())
                               if key in ("right_type",)
                               else self.refresh_cmd())
                    else:
                        w = tk.Entry(cell, textvariable=r[key],
                                     font=mono_font(8), relief="solid", bd=1,
                                     width=9,
                                     state="disabled" if (dead or no_biofilm)
                                     else "normal")
                        w.pack(fill="x")
                        w.bind("<KeyRelease>", lambda e: self.refresh_cmd())
            bar = tk.Frame(holder, bg=PANEL)
            bar.pack(fill="x", pady=(6, 2))
            tk.Button(bar, text="  +  add a chemical  ", font=base_font(8),
                      command=add).pack(side="left")
            tk.Button(bar, text="  \u2212  remove the last  ",
                      font=base_font(8), command=drop).pack(side="left",
                                                            padx=(6, 0))
            tk.Label(bar, text="  %d chemicals" % len(f.rows), bg=PANEL,
                     fg=MUTE, font=base_font(8)).pack(side="left", padx=(8, 0))

        def add():
            # Twelve is not a limit of the solver. It is a limit on how long a
            # page can get before it stops being usable, and the settings file
            # has no limit at all if you really want more.
            if len(f.rows) >= 12:
                self.app.set_notif("Twelve chemicals is as many as this page "
                                "holds. Use a settings file for more.", WARNC)
                return
            i = len(f.rows)
            f.rows.append({k: tk.StringVar(value=chem_default(i)[k])
                           for k in CHEM_KEYS})
            draw(); self.refresh_cmd()

        def drop():
            if len(f.rows) <= 1:
                self.app.set_notif("There has to be at least one chemical.", WARNC)
                return
            f.rows.pop()
            draw(); self.refresh_cmd()

        draw()
        # Redrawn whenever another box changes what the table should show.
        # Without this the greying is decided once, when the page is built,
        # and switching to abiotic-only leaves a live "D in biofilm" box on
        # screen for a run that has no biofilm in it.
        self._chem_redraw = draw
        return row

    def _sync_depends(self):
        r = getattr(self, "_chem_redraw", None)
        if r is not None:
            try:
                r()
            except Exception:                                  # noqa: BLE001
                pass
        for f in self.action.fields:
            if not f.depends:
                continue
            row = self.rows.get(f.key)
            if row is None:
                continue
            if _dep_ok(self.action.fields, f.depends):
                row.pack(fill="x", padx=8, pady=(6, 0), anchor="w")
            else:
                row.pack_forget()

    def _pick_open(self, f):
        p = filedialog.askopenfilename(title=f.label, filetypes=f.filetypes,
                                       initialdir=self._start_dir(f))
        if p:
            f.var.set(p); self.refresh_cmd()

    def _pick_save(self, f):
        p = filedialog.asksaveasfilename(title=f.label, filetypes=f.filetypes,
                                         initialdir=self._start_dir(f))
        if p:
            f.var.set(p); self.refresh_cmd()

    def _pick_dir(self, f):
        p = filedialog.askdirectory(title=f.label, initialdir=self._start_dir(f))
        if p:
            f.var.set(p); self.refresh_cmd()

    def _start_dir(self, f):
        cur = f.var.get()
        d = os.path.dirname(cur) if cur else ROOT
        return d if os.path.isdir(d) else ROOT

    # ---------------------------------------------------------------- actions
    def refresh_cmd(self):
        self._update_computed()
        cmd = self.action.command()
        self.cmdbox.configure(state="normal")
        self.cmdbox.delete("1.0", "end")
        self.cmdbox.insert("1.0", " ".join(_q(c) for c in cmd))
        self.cmdbox.configure(state="disabled")
        return cmd

    def copy_cmd(self):
        self.clipboard_clear()
        self.clipboard_append(" ".join(_q(c) for c in self.action.command()))
        self.app.status("Command copied to the clipboard.")

    def reset(self):
        for f in self.action.fields:
            f.var.set(f.default if f.kind != "bool" else bool(f.default))
        self._sync_depends()
        self.refresh_cmd()

    def out_dir(self):
        for key in ("out", "data", "checkpoint"):
            f = next((g for g in self.action.fields if g.key == key), None)
            if f and f.value():
                v = f.value()
                return v if os.path.isdir(v) else os.path.dirname(v)
        return ROOT

    def open_out(self):
        d = self.out_dir()
        if not os.path.isdir(d):
            # It does not exist because nothing has been run yet. Making it is
            # harmless and is obviously what was wanted; telling somebody that
            # a folder they asked to see is missing, and stopping there, is
            # not an answer to anything.
            try:
                os.makedirs(d, exist_ok=True)
                self.app.status("Made %s" % d)
            except Exception as e:                             # noqa: BLE001
                messagebox.showinfo(
                    APP, "That folder does not exist and could not be "
                         "made:\n\n%s\n\n%s" % (d, e))
                return
        try:
            if sys.platform.startswith("win"):
                os.startfile(d)                                # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception:                                      # noqa: BLE001
            messagebox.showinfo(APP, d)

    def _section_of(self, field):
        """Which labelled section a box sits in, for saying WHERE it is.

        A complaint that names a box is only half an answer on a page long
        enough to scroll. Walking back to the nearest section heading gives
        the other half.
        """
        # The chemicals table draws its own heading, so it is not preceded by
        # a heading field and walking back would name the section ABOVE it.
        if getattr(field, "kind", None) == "chemicals":
            return "THE CHEMICALS, the table with the plus and minus buttons"
        section = None
        for f in self.action.fields:
            if f.kind == "heading":
                section = f.label
            if f is field:
                return section
        return section

    def run(self):
        cmd = self.refresh_cmd()
        missing = []
        for f in self.action.fields:
            if not _dep_ok(self.action.fields, f.depends):
                continue
            # Not inputs. A section heading, the line that counts up the
            # simulations, and the chemicals table all carry no single value,
            # so the emptiness test below would flag every one of them --
            # which is exactly what it did, and the dialog then listed five
            # headings and one blank line and told the user to attend to them.
            if f.kind in ("heading", "computed"):
                continue
            # A wrong token count, a chemical with no name, two chemicals with
            # the same name: caught here, before the script is launched, and
            # reported in the field's own words.
            bad = f.check()
            if bad:
                where = self._section_of(f)
                missing.append(bad + ("" if not where else
                                      "\n      It is under %s." % where))
                continue
            where = self._section_of(f)
            under = "" if not where else "\n      It is under %s." % where
            # An OPTIONAL file box is optional to fill in, not optional to get
            # right. Left blank it is ignored; pointed at a file that is not
            # there it is a mistake, and one that used to be discovered three
            # seconds into a run as a Python traceback.
            if (f.kind in ("file", "files") and str(f.value()).strip()
                    and not os.path.exists(str(f.value()).strip())):
                missing.append('"%s" points at a file that is not there:\n'
                               "      %s\n      Use the ... button next to the "
                               "box to pick one, or clear the box to leave it "
                               "out.%s" % (f.label, f.value(), under))
                continue
            if f.optional or f.kind in ("bool", "choice", "subcommand",
                                        "chemicals", "setting"):
                continue
            if not str(f.value()).strip():
                missing.append('"%s" is empty. Type a value into it, in the '
                               "SETTINGS panel on the left.%s" % (f.label, under))
            elif f.kind == "file" and not os.path.exists(f.value()):
                missing.append('"%s" points at a file that is not there:\n'
                               "      %s\n      Use the ... button next to the "
                               "box to pick one.%s" % (f.label, f.value(), under))
        if missing:
            messagebox.showwarning(
                APP,
                "%d setting%s need%s attention before this can run.\n\n"
                "  •  %s\n\nNothing has been started, and nothing has been "
                "changed on disk."
                % (len(missing), "" if len(missing) == 1 else "s",
                   "s" if len(missing) == 1 else "",
                   "\n\n  •  ".join(missing)))
            return
        # WHERE THIS RUN'S RESULTS GO. Decided here, just before launching,
        # so that the path shown on the page is the path that is used.
        #
        # Nothing is ever written on top of an earlier result. If the place
        # you named already holds one, this run goes to the next free name
        # beside it and says so. That is not tidiness: a trained network is
        # meaningless without the dataset it was trained on, and overwriting
        # the dataset while the checkpoint survives leaves a model nobody can
        # score or reproduce.
        moved = []
        for f in self.action.fields:
            if f.key not in ("out",) or not _dep_ok(self.action.fields,
                                                    f.depends):
                continue
            want = str(f.value()).strip()
            if not want:
                continue
            free = _next_free(want, f.kind == "dir")
            if os.path.abspath(free) != os.path.abspath(want):
                f.var.set(free)
                moved.append((want, free))
            try:
                os.makedirs(free if f.kind == "dir"
                            else os.path.dirname(os.path.abspath(free)),
                            exist_ok=True)
            except Exception as e:                             # noqa: BLE001
                messagebox.showerror(
                    APP, "Could not make the folder for the results:\n\n"
                         "%s\n\n%s\n\nCheck that the drive is there and "
                         "that you can write to it." % (free, e))
                return
        if moved:
            cmd = self.refresh_cmd()
            for old, new in moved:
                self.app.logline(
                    "%s already holds a result, so this run goes to %s "
                    "instead. Nothing was overwritten.\n"
                    % (os.path.basename(os.path.dirname(os.path.abspath(old)))
                       if not os.path.isdir(old) else os.path.basename(old),
                       new))
            self.app.set_notif("Writing to a new folder: %s"
                               % os.path.basename(
                                   os.path.dirname(os.path.abspath(
                                       moved[0][1]))
                                   if not os.path.isdir(moved[0][1])
                                   else moved[0][1]), INFOC)

        if not os.path.exists(self.action.script):
            messagebox.showerror(
                APP, "Cannot find the script this action runs:\n\n%s\n\n"
                     "Check that the gui folder is still inside the PRT-DeepONet "
                     "folder, next to 2D, 3D and bridge." % self.action.script)
            return
        total = None
        if self.action.total_key:
            f = next((g for g in self.action.fields if g.key == self.action.total_key), None)
            try:
                total = int(f.value())
            except Exception:                                  # noqa: BLE001
                total = None
        self.app.launch(self.action, cmd, total)


def _q(s):
    s = str(s)
    return '"%s"' % s if " " in s else s


# =============================================================================
#  VIEWER — the ParaView-lite tab
# =============================================================================
class Viewer(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self.data = None          # dict describing whatever is loaded
        self.fig = None

        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=8, pady=6)
        tk.Label(bar, text="Open:", bg=BG, fg=INK, font=base_font(9)).pack(side="left")
        for label, fn in (("Geometry (.npz)", self.open_geometry),
                          ("Dataset (.h5)", self.open_dataset),
                          ("2D domain (.dat)", self.open_2d),
                          ("Prediction (.npz/.npy)", self.open_pred)):
            tk.Button(bar, text=label, font=base_font(8), padx=8,
                      command=fn).pack(side="left", padx=3)
        tk.Button(bar, text="Save picture", font=base_font(8), padx=8,
                  command=self.save_png).pack(side="right")

        self.ctl = tk.Frame(self, bg=PANEL, highlightbackground=LINE,
                            highlightthickness=1)
        self.ctl.pack(fill="x", padx=8)
        self.plotarea = tk.Frame(self, bg=PANEL, highlightbackground=LINE,
                                 highlightthickness=1)
        self.plotarea.pack(fill="both", expand=True, padx=8, pady=8)
        self.placeholder = tk.Label(
            self.plotarea, bg=PANEL, fg=MUTE, justify="left", font=base_font(10),
            text=("\n\n  Nothing loaded yet.\n\n"
                  "  Open a geometry to see the pore structure and the distance field\n"
                  "  the network uses to sense it.\n\n"
                  "  Open a dataset to step through the chemical fields in space and\n"
                  "  in time, exactly as the simulation produced them.\n\n"
                  "  Open a prediction to compare what the network produced.\n"))
        self.placeholder.pack(fill="both", expand=True)

        self.var_field = tk.StringVar()
        self.var_sample = tk.IntVar(value=0)
        self.var_time = tk.IntVar(value=0)
        self.var_mode = tk.StringVar(value="Three slices")
        self.var_cmap = tk.StringVar(value="viridis")
        self.var_x = tk.IntVar(value=0)
        self.var_y = tk.IntVar(value=0)
        self.var_z = tk.IntVar(value=0)

    # ------------------------------------------------------------- controls
    def _clear_ctl(self):
        for w in self.ctl.winfo_children():
            w.destroy()

    def _spin(self, parent, text, var, lo, hi, width=5):
        tk.Label(parent, text=text, bg=PANEL, fg=INK,
                 font=base_font(8)).pack(side="left", padx=(8, 2))
        s = tk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=width,
                       font=mono_font(8), command=self.redraw)
        s.pack(side="left")
        s.bind("<Return>", lambda e: self.redraw())
        return s

    def _shown_shape(self):
        """The grid of the array actually on screen, not of the first one.

        A dataset has one shape and this is the same thing; a folder of
        prediction arrays need not, and the views that make sense follow the
        array being shown.
        """
        d = self.data
        fallback = tuple(int(v) for v in d["shape"])
        arrs = d.get("arrays")
        if not arrs:
            return fallback
        a = arrs.get(self.var_field.get())
        if a is None:
            return fallback
        sh = tuple(int(v) for v in getattr(a, "shape", ()))
        return sh if len(sh) == 3 else fallback

    def _build_controls(self):
        self._clear_ctl()
        d = self.data
        if not d:
            return
        r1 = tk.Frame(self.ctl, bg=PANEL)
        r1.pack(fill="x", pady=4)
        tk.Label(r1, text=d["title"], bg=PANEL, fg=INK,
                 font=base_font(9, True)).pack(side="left", padx=8)
        tk.Label(r1, text=d["subtitle"], bg=PANEL, fg=MUTE,
                 font=base_font(8)).pack(side="left", padx=6)

        r2 = tk.Frame(self.ctl, bg=PANEL)
        r2.pack(fill="x", pady=(0, 6))
        tk.Label(r2, text="Show:", bg=PANEL, fg=INK,
                 font=base_font(8)).pack(side="left", padx=(8, 2))
        cb = ttk.Combobox(r2, textvariable=self.var_field, values=d["fields"],
                          state="readonly", width=26, font=base_font(8))
        cb.pack(side="left")
        # Rebuilding rather than only redrawing: a .npz of predictions can hold
        # arrays of different shapes, and which VIEWS make sense depends on the
        # one selected. Deciding that once from the first array offered 2D
        # views for a 3D field, and then drew its first plane under a caption
        # saying "the whole 2D field".
        cb.bind("<<ComboboxSelected>>",
                lambda e: (self._build_controls(), self.redraw()))
        if self.var_field.get() not in d["fields"]:
            self.var_field.set(d["fields"][0])

        tk.Label(r2, text="View:", bg=PANEL, fg=INK,
                 font=base_font(8)).pack(side="left", padx=(10, 2))
        # A two-dimensional field has exactly one plane. Slicing it in three
        # directions gives one real picture and two one-pixel strips, and the
        # cut-away has nothing to cut away, so those views are not offered.
        if int(self._shown_shape()[2]) == 1:
            modes = ["The whole plane", "Time montage"]
        else:
            modes = ["Three slices", "Slice along flow", "3D cut-away",
                     "Time montage"]
        if d.get("nt", 1) <= 1:
            modes.remove("Time montage")
        cb2 = ttk.Combobox(r2, textvariable=self.var_mode, values=modes,
                           state="readonly", width=16, font=base_font(8))
        cb2.pack(side="left")
        cb2.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        if self.var_mode.get() not in modes:
            self.var_mode.set(modes[0])

        tk.Label(r2, text="Colours:", bg=PANEL, fg=INK,
                 font=base_font(8)).pack(side="left", padx=(10, 2))
        cb3 = ttk.Combobox(r2, textvariable=self.var_cmap, width=10,
                           state="readonly", font=base_font(8),
                           values=["viridis", "magma", "plasma", "inferno",
                                   "coolwarm", "YlGnBu", "OrRd", "Greys"])
        cb3.pack(side="left")
        cb3.bind("<<ComboboxSelected>>", lambda e: self.redraw())

        r3 = tk.Frame(self.ctl, bg=PANEL)
        r3.pack(fill="x", pady=(0, 6))
        nx, ny, nz = self._shown_shape()
        if d.get("ns", 1) > 1:
            self._spin(r3, "run", self.var_sample, 0, d["ns"] - 1)
        if d.get("nt", 1) > 1:
            self._spin(r3, "snapshot", self.var_time, 0, d["nt"] - 1)
        self.var_x.set(min(self.var_x.get() or nx // 2, nx - 1))
        self.var_y.set(min(self.var_y.get() or ny // 2, ny - 1))
        self.var_z.set(min(self.var_z.get() or nz // 2, nz - 1))
        if nz > 1:
            self._spin(r3, "slice x", self.var_x, 0, nx - 1)
            self._spin(r3, "slice y", self.var_y, 0, ny - 1)
            self._spin(r3, "slice z", self.var_z, 0, nz - 1)
        else:
            # Nothing to slice: the whole field is already on screen.
            self.var_z.set(0)
        tk.Button(r3, text="Redraw", font=base_font(8), padx=8,
                  command=self.redraw).pack(side="right", padx=8)
        if d.get("nt", 1) > 1:
            tk.Label(r3, text="(snapshot 0 is the start, before anything has "
                              "entered the sample)", bg=PANEL, fg=MUTE,
                     font=base_font(8)).pack(side="right", padx=10)

    # ------------------------------------------------------------- loaders
    def _need(self, *mods):
        """Guard the viewer against a half-installed environment, with a message
        that says what to do rather than an import traceback."""
        miss = []
        for m in mods:
            try:
                __import__(m)
            except Exception:                                  # noqa: BLE001
                miss.append(m)
        if miss:
            messagebox.showwarning(
                APP,
                "The viewer needs these packages, which are not installed yet:\n\n"
                "    " + "\n    ".join(miss) + "\n\n"
                "Install them from:\n"
                "    Set up  >  Install the Python packages\n\n"
                "or by hand:\n    \"" + PYTHON + "\" -m pip install "
                + " ".join(miss))
            return False
        return True

    def open_geometry(self):
        if not self._need("numpy", "matplotlib"):
            return
        p = filedialog.askopenfilename(
            title="Open a geometry", initialdir=os.path.join(WORK, "geometries"),
            filetypes=[("Geometry", "*.npz"), ("All files", "*.*")])
        if not p:
            return
        try:
            import numpy as np
            z = np.load(p, allow_pickle=True)
            mat = z["material"]
            arrays = {"pore space (2 = pore, 0 = solid, 1 = wall)": mat.astype("float32")}
            if "gdf" in z:
                arrays["geodesic distance from the inlet"] = np.nan_to_num(z["gdf"])
            if "edt" in z:
                arrays["distance to the nearest solid"] = z["edt"]
            self.data = dict(kind="geom", title=os.path.basename(p),
                             subtitle="%s voxels, porosity %.3f"
                                      % ("x".join(map(str, mat.shape)),
                                         float((mat == 2).mean())),
                             fields=list(arrays), arrays=arrays, mat=mat,
                             shape=mat.shape, ns=1, nt=1)
            self._build_controls(); self.redraw()
            self.app.status("Loaded geometry %s" % os.path.basename(p))
        except Exception as e:                                 # noqa: BLE001
            self.app.problem("Could not open that geometry: %s" % e)

    def open_2d(self):
        if not self._need("numpy", "scipy", "matplotlib"):
            return
        p = filedialog.askopenfilename(
            title="Open a 2D domain", initialdir=os.path.join(DIR_2D, "Domains"),
            filetypes=[("2D domain", "*.dat *.npz"), ("All files", "*.*")])
        if not p:
            return
        try:
            import numpy as np
            sys.path.insert(0, TOOLS)
            from build_transfer_set_2d_to_3d import _reshape_flat, to_complab       # noqa
            if p.endswith(".npz"):
                z = np.load(p)
                a = np.asarray(z[list(z.keys())[0]]).squeeze()
            else:
                with open(p, "rb") as fh:
                    a = np.array(fh.read().split(), dtype="int16")
            a = _reshape_flat(a)
            g = to_complab(a)
            vol = g.astype("float32")[:, :, None]
            self.data = dict(kind="2d", title=os.path.basename(p),
                             subtitle="2D domain %s, porosity %.3f  (index order "
                                      "auto-detected)"
                                      % ("x".join(map(str, g.shape)),
                                         float((g == 2).mean())),
                             fields=["pore space"], arrays={"pore space": vol},
                             mat=vol, shape=vol.shape, ns=1, nt=1)
            self.var_mode.set("The whole plane")
            self._build_controls(); self.redraw()
            self.app.status("Loaded 2D domain %s" % os.path.basename(p))
        except Exception as e:                                 # noqa: BLE001
            self.app.problem("Could not open that 2D domain: %s" % e)

    def open_dataset(self):
        if not self._need("numpy", "h5py", "matplotlib"):
            return
        p = filedialog.askopenfilename(
            title="Open a dataset", initialdir=ROOT,
            filetypes=[("HDF5 dataset", "*.h5"), ("All files", "*.*")])
        if not p:
            return
        try:
            import h5py, numpy as np                            # noqa
            h = h5py.File(p, "r")
            species = [s.decode() if isinstance(s, bytes) else str(s)
                       for s in h.attrs["species"]]
            shape = tuple(int(v) for v in h.attrs["shape"])
            ns = int(h["samples/conc"].shape[0])
            nt = int(h["samples/conc"].shape[1])
            fields = ["chemical: " + s for s in species]
            fields += ["pore space", "geodesic distance from the inlet"]
            if "edt" in h["geom"]:
                fields.append("distance to the nearest solid")
            if "velocity" in h["samples"]:
                fields += ["flow speed", "flow along x"]
            self.data = dict(kind="h5", h=h, path=p, title=os.path.basename(p),
                             subtitle="%d runs, %d snapshots, %d chemicals, %s voxels"
                                      % (ns, nt, len(species),
                                         "x".join(map(str, shape))),
                             fields=fields, species=species, shape=shape,
                             ns=ns, nt=nt)
            self.var_time.set(max(nt - 1, 0))   # the developed field, not t = 0
            self.var_sample.set(0)
            self._build_controls(); self.redraw()
            self.app.status("Loaded dataset %s  —  showing the last snapshot"
                            % os.path.basename(p))
        except Exception as e:                                 # noqa: BLE001
            self.app.problem("Could not open that dataset: %s" % e)

    def open_pred(self):
        if not self._need("numpy", "matplotlib"):
            return
        p = filedialog.askopenfilename(
            title="Open a prediction", initialdir=ROOT,
            filetypes=[("Prediction", "*.npz *.npy"), ("All files", "*.*")])
        if not p:
            return
        try:
            import numpy as np
            arrays = {}
            if p.endswith(".npy"):
                a = np.load(p)
                if a.ndim == 4 and a.shape[0] == 3:
                    arrays["flow speed"] = np.sqrt((a ** 2).sum(0))
                    for i, n in enumerate("xyz"):
                        arrays["flow along " + n] = a[i]
                elif a.ndim == 4:
                    for i in range(a.shape[0]):
                        arrays["channel %d" % i] = a[i]
                else:
                    arrays["value"] = a
            else:
                z = np.load(p, allow_pickle=True)
                for k in z.files:
                    v = z[k]
                    if getattr(v, "ndim", 0) == 3:
                        arrays[k] = v
                    elif getattr(v, "ndim", 0) == 4:
                        for i in range(v.shape[0]):
                            arrays["%s[%d]" % (k, i)] = v[i]
            if not arrays:
                raise ValueError("no 3D arrays in that file")
            first = arrays[list(arrays)[0]]
            self.data = dict(kind="pred", title=os.path.basename(p),
                             subtitle="%d arrays, %s voxels"
                                      % (len(arrays), "x".join(map(str, first.shape))),
                             fields=list(arrays), arrays=arrays, mat=None,
                             shape=first.shape, ns=1, nt=1)
            self._build_controls(); self.redraw()
            self.app.status("Loaded prediction %s" % os.path.basename(p))
        except Exception as e:                                 # noqa: BLE001
            self.app.problem("Could not open that prediction: %s" % e)

    # ------------------------------------------------------------- get array
    def _current(self):
        import numpy as np
        d = self.data
        name = self.var_field.get()
        if d["kind"] == "h5":
            h = d["h"]
            s = int(self.var_sample.get()); t = int(self.var_time.get())
            g = int(h["samples/geom_index"][s])
            mat = h["geom/material"][g]
            if name.startswith("chemical: "):
                i = d["species"].index(name.split(": ", 1)[1])
                a = h["samples/conc"][s, t, i].astype("float32")
            elif name == "pore space":
                a = mat.astype("float32")
            elif name.startswith("geodesic"):
                a = np.nan_to_num(h["geom/gdf"][g])
            elif name.startswith("distance to"):
                a = h["geom/edt"][g][...]
            elif name == "flow speed":
                v = h["samples/velocity"][s].astype("float32")
                a = np.sqrt((v ** 2).sum(0))
            elif name == "flow along x":
                a = h["samples/velocity"][s, 0].astype("float32")
            else:
                a = mat.astype("float32")
            return np.asarray(a), mat
        arr = d["arrays"][name]
        return np.asarray(arr), d.get("mat")

    def _series(self):
        import numpy as np
        d = self.data
        if d["kind"] != "h5":
            return None
        name = self.var_field.get()
        if not name.startswith("chemical: "):
            return None
        h = d["h"]; s = int(self.var_sample.get())
        i = d["species"].index(name.split(": ", 1)[1])
        return np.asarray(h["samples/conc"][s, :, i]).astype("float32")

    # ------------------------------------------------------------- drawing
    def redraw(self):
        if not self.data:
            return
        try:
            self._draw()
        except Exception as e:                                 # noqa: BLE001
            self.app.problem("Could not draw that: %s" % e)

    def _draw(self):
        import numpy as np
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)

        for w in self.plotarea.winfo_children():
            w.destroy()
        a, mat = self._current()
        mode = self.var_mode.get()
        cmap = self.var_cmap.get()
        nx, ny, nz = a.shape
        xi = min(int(self.var_x.get()), nx - 1)
        yi = min(int(self.var_y.get()), ny - 1)
        zi = min(int(self.var_z.get()), nz - 1)
        pore = None if mat is None else (np.asarray(mat) == 2)
        show = a.astype("float32").copy()
        if pore is not None and not self.var_field.get().startswith("pore"):
            show = np.where(pore, show, np.nan)
        finite = show[np.isfinite(show)]
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-9

        # In 2D there is one plane and no third axis to cut along. Take that
        # plane whole rather than a line out of it.
        two_d = (nz == 1)
        if two_d and mode not in ("The whole plane", "Time montage"):
            mode = "The whole plane"

        def cut(vol):
            """The picture to draw, from a full (nx, ny, nz) array."""
            return vol[:, :, 0] if two_d else vol[:, yi, :]

        cut_lab = ("the whole plane" if two_d else "slice at y = %d" % yi)

        if mode == "Time montage":
            ser = self._series()
            if ser is None:
                self.var_mode.set("The whole plane" if two_d else "Three slices")
                return self._draw()
            n = ser.shape[0]
            fig, ax = plt.subplots(1, n, figsize=(min(2.0 * n, 16), 2.6))
            ax = np.atleast_1d(ax)
            vmin = float(np.nanmin(ser)); vmax = float(np.nanmax(ser)) or 1.0
            for j in range(n):
                s = cut(ser[j])
                if pore is not None:
                    s = np.where(cut(pore), s, np.nan)
                im = ax[j].imshow(s.T, origin="lower", cmap=cmap, vmin=vmin,
                                  vmax=vmax, interpolation="nearest", aspect="auto")
                ax[j].set_title("t %d" % j, fontsize=8)
                ax[j].set_xticks([]); ax[j].set_yticks([])
            fig.colorbar(im, ax=ax.tolist(), fraction=0.02)
            fig.suptitle("%s  —  %s, through time"
                         % (self.var_field.get(), cut_lab), fontsize=10)
        elif mode == "The whole plane":
            fig, ax = plt.subplots(figsize=(9, 4.6))
            im = ax.imshow(show[:, :, 0].T, origin="lower", cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest",
                           aspect="equal")
            ax.set_xlabel("x   (flow enters left, leaves right)")
            ax.set_ylabel("y")
            ax.set_title("%s   —   the whole 2D field,  range %.4g to %.4g"
                         % (self.var_field.get(), vmin, vmax), fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.03)
        elif mode == "Slice along flow":
            fig, ax = plt.subplots(figsize=(9, 4.2))
            im = ax.imshow(show[:, yi, :].T, origin="lower", cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation="nearest",
                           aspect="auto")
            ax.set_xlabel("x   (flow enters left, leaves right)")
            ax.set_ylabel("z")
            ax.set_title("%s   —   slice at y = %d" % (self.var_field.get(), yi),
                         fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.03)
        elif mode == "3D cut-away":
            fig = plt.figure(figsize=(8, 5.6))
            ax = fig.add_subplot(111, projection="3d")
            self._cutaway(ax, show, pore, cmap, vmin, vmax)
            ax.set_title("%s   —   one quarter cut away" % self.var_field.get(),
                         fontsize=10)
        else:                                        # Three slices
            fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
            panes = [(show[xi, :, :], "y", "z", "x = %d  (across the flow)" % xi),
                     (show[:, yi, :], "x", "z", "y = %d  (along the flow)" % yi),
                     (show[:, :, zi], "x", "y", "z = %d  (along the flow)" % zi)]
            for axx, (p, lx, ly, ttl) in zip(axs, panes):
                im = axx.imshow(p.T, origin="lower", cmap=cmap, vmin=vmin,
                                vmax=vmax, interpolation="nearest", aspect="auto")
                axx.set_title(ttl, fontsize=9)
                axx.set_xlabel(lx); axx.set_ylabel(ly)
            fig.colorbar(im, ax=list(axs), fraction=0.025, pad=0.02)
            fig.suptitle("%s      range %.4g to %.4g"
                         % (self.var_field.get(), vmin, vmax), fontsize=10)

        try:
            fig.tight_layout()
        except Exception:                                      # noqa: BLE001
            pass
        self.fig = fig
        canvas = FigureCanvasTkAgg(fig, master=self.plotarea)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(canvas, self.plotarea)
        tb.update()

    def _cutaway(self, ax, show, pore, cmap, vmin, vmax):
        import numpy as np
        nx, ny, nz = show.shape
        cx, cy = nx // 2, ny // 2
        # matplotlib.cm.get_cmap was deprecated in 3.7 and is removed in 3.11.
        # matplotlib.colormaps[...] is the replacement and exists from 3.6, so
        # this works on both old and new installs.
        import matplotlib
        try:
            mapper = matplotlib.colormaps[cmap]
        except Exception:                                      # noqa: BLE001
            import matplotlib.cm as _cm
            mapper = _cm.get_cmap(cmap)
        norm = lambda v: np.clip((v - vmin) / (vmax - vmin + 1e-30), 0, 1)

        def face(vals, X, Y, Z, shade=1.0):
            c = mapper(norm(np.nan_to_num(vals, nan=0.0)))
            c[..., :3] *= shade
            bad = ~np.isfinite(vals)
            c[bad] = (0.85, 0.85, 0.85, 1.0)
            ax.plot_surface(X, Y, Z, facecolors=c, shade=False, rstride=1,
                            cstride=1, linewidth=0, antialiased=False)

        Y, Z = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
        X, Z2 = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        X2, Y2 = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        face(show[0], np.zeros_like(Y), Y, Z, 0.8)
        face(show[:, 0, :], X, np.zeros_like(X), Z2, 0.9)
        face(show[:cx, :, nz - 1][:, :, None][:, :, 0],
             X2[:cx], Y2[:cx], np.full_like(X2[:cx], nz - 1), 1.0)
        face(show[cx:, :cy, nz - 1], X2[cx:, :cy], Y2[cx:, :cy],
             np.full_like(X2[cx:, :cy], nz - 1), 1.0)
        face(show[cx, cy:, :], np.full_like(Y[cy:], cx), Y[cy:], Z[cy:], 0.86)
        face(show[cx:, cy, :], X[cx:], np.full_like(X[cx:], cy), Z2[cx:], 0.98)
        ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
        try:
            ax.set_box_aspect((nx, ny, nz))
        except Exception:                                      # noqa: BLE001
            pass
        ax.set_axis_off()
        ax.view_init(elev=20, azim=-62)

    def save_png(self):
        if self.fig is None:
            messagebox.showinfo(APP, "There is nothing on screen to save yet.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG image", "*.png")])
        if p:
            self.fig.savefig(p, dpi=150, bbox_inches="tight")
            self.app.status("Saved %s" % p)


# =============================================================================
#  TUTORIAL AND GLOSSARY TEXT
# =============================================================================
TUTORIAL = """\
PRT-DeepONet Studio — a walk through, start to finish

WHAT THIS PROGRAM IS FOR
------------------------
Pore-scale reactive transport simulations are accurate and slow. One run of the
current plan takes between five and sixteen hours. If you want to know what
happens at a hundred different combinations of flow rate and reaction rate, that
is weeks of cluster time.

A surrogate is a neural network that has watched enough simulations to predict
the answer without running one. It takes under a second. This program is the
workbench for building, training and checking that surrogate.

It does NOT replace CompLaB3D. It replaces the need to run CompLaB3D again for
every new question.


THE FIVE STAGES
---------------
1.  SET UP     make the pore structures the simulations will run on
2.  SIMULATE   run CompLaB3D on them and gather the output into one dataset
3.  LEARN      train the network, then check it on structures it never saw
4.  PREDICT    ask it about a new structure, in under a second
5.  LOOK       open any of it in the Viewer

The left-hand tree follows the same order. Work down it.


YOUR FIRST HOUR, WITHOUT TOUCHING THE CLUSTER
---------------------------------------------
Everything below runs on a laptop in a few minutes.

  Step 1.  Set up  ->  Make a small practice dataset.
           Press Run. In about five seconds you have a real dataset: real pore
           structures, a real lattice-Boltzmann flow solve, real transport. It
           is only small.

  Step 2.  Set up  ->  Check the installation.
           Point it at the practice dataset. This proves the three switches
           work, and in particular that turning them OFF reproduces the original
           behaviour exactly.

  Step 3.  Learn  ->  Train the network.
           Dataset: the practice file. Set "How many passes over the data" to
           about 30 so it finishes quickly. Press Run and watch the Monitor tab:
           the two curves are the error on data it is learning from and on data
           it has never seen. When the second stops falling, it has learned all
           it can from this much data.

  Step 4.  Learn  ->  Evaluate a trained network.
           Point it at best.pt and at the same dataset. It draws truth against
           prediction, and where in the parameter space the model is weak.

  Step 5.  Look  ->  Viewer.
           Open the practice dataset and step through the chemicals in space and
           in time. Then open a geometry and look at the geodesic distance
           field, which is how the network senses the pore structure.

After that, the same buttons work on the real thing. Only the numbers get bigger.


WHAT THE NETWORK ACTUALLY SEES
------------------------------
Three inputs, and it is worth knowing which is which because every switch
changes one of them.

  THE PORE STRUCTURE   goes through a 3D convolutional encoder, called the
                       "branch". Think of it as reading the rock.

  Pe AND Da            go through a small dense network, the "parameter branch".
                       Think of it as reading the experimental conditions.

  WHERE YOU ARE        goes through the "trunk": the coordinates of a pore voxel,
                       the moment in time, and how far that voxel is from the
                       inlet THROUGH the pore space. That last number is the
                       geodesic distance, and it is the heart of the method.

The two branches are multiplied together, and the result is dotted with the
trunk. The answer is the concentration at that voxel at that moment.

Why the geodesic distance and not the straight-line distance? Because a chemical
does not travel through rock. Two voxels can be a millimetre apart in a straight
line and half a sample apart through the pore space. The "edt" option in
training uses the straight-line distance instead, and exists only as the control
that proves the geodesic one is what matters.


THE THREE SWITCHES
------------------
All three are OFF by default, and with all three off the code does exactly what
it always did. "Check the installation" proves that bit for bit.

SWITCH A — use flow instead of geometry
    Replaces the geodesic distance with the ADVECTIVE TRAVEL TIME: how long a
    parcel of fluid takes to reach this voxel from the inlet. Replaces the pore
    mask with the velocity field.

    Why bother: along a streamline the transport-reaction equation becomes
    dC/dtau = -R(C), which has no geometry in it at all. In travel-time
    coordinates the reaction front sits at the same place whatever the rock
    looks like. It also means the geometry problem becomes somebody else's:
    we would take a published flow model, and spend all our effort on chemistry.

    The catch, and it is a real one: where the fluid does not move, the flow
    field says nothing about the pore space. In a dead-end pore, or anywhere at
    Peclet 0.3, the chemical arrives by diffusion and a velocity of zero cannot
    tell a deep dead-end pore from solid rock. The geodesic distance still can.
    So expect this switch to win at high Peclet and lose at low. Use
    "what the flow switch feeds in = both" to give the network both and let it
    choose.

SWITCH B — mix in 2D data
    A 3D run costs hours. The 2D folder already holds three thousand domains.
    Extrude a 2D domain along the third axis and nothing varies in that
    direction, so the exact 3D answer IS the 2D answer repeated. That makes them
    genuine 3D training data, free.

    What transfers: the chemistry. Rate laws, the Peclet and Damkohler response,
    the shape of a front. None of it contains a dimension.
    What does not: the topology. A pore network percolates near porosity 0.20 in
    3D and near 0.59 in 2D. Same rock, wildly different connectivity.

    So the right way to use it is to keep the chemistry part and retrain the
    rock part: train on 2D, then train on 3D with "freeze the reaction part"
    ticked.

SWITCH C — dimension-free coordinates
    A and B together, and the reason they belong together. What makes 2D-to-3D
    transfer hard is the geometry mismatch, and switch A removes geometry from
    the input.

    In ordinary coordinates the trunk needs four numbers in 2D and five in 3D,
    so the same network cannot fit both. In flow coordinates it needs three in
    both: travel time, distance to the nearest wall, and time. The same network
    fits 2D and 3D with no modification at all.


READING THE MONITOR
-------------------
  train    the error on data the network is learning from
  test     the error on structures it has never seen

If train keeps falling while test flattens or rises, it has started memorising
rather than learning. Training stops on its own when test stops improving.

The split is always by STRUCTURE, never by sample. If the same pore structure
appeared in both halves the score would be flattering and meaningless.


IF SOMETHING GOES WRONG
-----------------------
The Problems tab collects every error, with the command that produced it. The
command is also shown on every action page before you run it, so you can paste
it into a terminal and see the whole thing.

The most common causes, in order:
  - a path that does not exist yet
  - a dataset built with a different grid size than the model expects
  - the flow-proxy or dimension-free switch on, but no velocity field supplied
    for prediction. Those switches replace the geometry with the flow, so a
    geometry file alone is not enough.
"""

GLOSSARY = [
    ("Surrogate", "Neural operator, DeepONet",
     "A model that has watched many simulations and predicts their result "
     "directly. Seconds instead of hours."),
    ("Branch", "Convolutional encoder",
     "The part that looks at the pore structure and turns it into a short list "
     "of numbers describing it."),
    ("Parameter branch", "Dense network on the dimensionless groups",
     "The part that reads Pe and Da, i.e. the experimental conditions."),
    ("Trunk", "Coordinate network",
     "The part that is asked 'what is the concentration HERE, at this moment?' "
     "It receives the voxel position, the time, and the geodesic distance."),
    ("GDF, geodesic distance", "Trunk geometry feature",
     "For each pore voxel, the shortest path back to the inlet THROUGH pore "
     "space. Not the straight-line distance: a chemical cannot travel through "
     "rock."),
    ("EDT, Euclidean distance", "Ablation control",
     "Straight-line distance. Used only to prove the geodesic version is what "
     "buys the accuracy."),
    ("tau, travel time", "Advective coordinate",
     "How long a parcel of fluid takes to reach a voxel from the inlet. The "
     "flow-space version of the geodesic distance."),
    ("Peclet number, Pe", "Input parameter",
     "How much faster the flow carries a chemical than diffusion spreads it. "
     "Below 1 diffusion wins; above 10 the flow wins."),
    ("Damkohler number, Da", "Input parameter",
     "How much reaction happens while one parcel of fluid crosses the sample. "
     "Da much less than 1 means almost nothing reacts; much more than 1 means "
     "everything reacts in a thin front near the inlet."),
    ("Epoch", "One pass over the training data",
     "Training makes many passes. Progress is measured in epochs."),
    ("RMSE", "Root mean square error",
     "The typical size of the difference between prediction and simulation, in "
     "the same units as the field."),
    ("Held out", "Test set",
     "Structures deliberately kept out of training, so the score means "
     "something. Split by structure, never by sample."),
    ("Checkpoint, best.pt", "Saved weights",
     "The trained network on disk. This is the deliverable."),
    ("Percolation", "Connectivity check",
     "Does the pore space actually reach from inlet to outlet? If not the run "
     "is meaningless and the structure is discarded."),
]


# =============================================================================
#  MAIN WINDOW
# =============================================================================
class Studio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP)
        self.geometry("1280x820")
        self.minsize(1040, 680)
        self.configure(bg=BG)

        self._load_settings()
        self.actions = build_actions()
        self.pages = {}
        self.current = None
        self.errors = []
        self.hist = []                       # (epoch, train, test)
        self.total_steps = None
        self.run_action = None
        self.run_cmd = None

        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:                                      # noqa: BLE001
            pass
        self.style.configure("TNotebook.Tab", font=base_font(9), padding=(10, 4))
        self.style.configure("Treeview", font=base_font(9), rowheight=20)
        self.style.configure("H.Horizontal.TProgressbar", troughcolor="#c9c6c1",
                             background="#4b7fc6")

        self._menu()
        self._body()
        self._statusbar()

        self.runner = Runner(self.on_line, self.on_done)
        self.after(120, self._pump)
        self.show_welcome()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------------------------------------------- settings
    def _load_settings(self):
        try:
            import json
            with open(SETTINGS) as fh:
                d = json.load(fh)
            p = d.get("python")
            if p and os.path.exists(p):
                set_python(p)
        except Exception:                                      # noqa: BLE001
            pass

    def _save_settings(self):
        try:
            import json
            with open(SETTINGS, "w") as fh:
                json.dump({"python": PYTHON}, fh)
        except Exception:                                      # noqa: BLE001
            pass

    def choose_python(self):
        cur = PYTHON
        msg = ("The analysis scripts are run with:\n\n    %s\n\n"
               "This must be a Python that has torch, numpy, scipy and h5py "
               "installed. If your project uses a virtual environment, point "
               "this at the python inside it:\n"
               "    Windows :  3D\\.venv\\Scripts\\python.exe\n"
               "    Linux   :  3D/.venv/bin/python\n\n"
               "Change it now?" % cur)
        if not messagebox.askyesno(APP, msg):
            return
        p = filedialog.askopenfilename(title="Choose the Python interpreter",
                                       initialdir=DIR_3D)
        if not p:
            return
        set_python(p)
        self._save_settings()
        for page in self.pages.values():
            try:
                page.refresh_cmd()
            except Exception:                                  # noqa: BLE001
                pass
        self.status("Scripts will now run with %s" % p)
        self.logline("Interpreter set to %s" % p, "hdr")

    # ------------------------------------------------------------------ menu
    def _menu(self):
        m = tk.Menu(self)

        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Open dataset in the Viewer...",
                      command=lambda: self._goto_viewer("dataset"))
        f.add_command(label="Open geometry in the Viewer...",
                      command=lambda: self._goto_viewer("geometry"))
        f.add_command(label="Open 2D domain in the Viewer...",
                      command=lambda: self._goto_viewer("2d"))
        f.add_command(label="Open prediction in the Viewer...",
                      command=lambda: self._goto_viewer("pred"))
        f.add_separator()
        f.add_command(label="Open the project folder",
                      command=lambda: self._reveal(ROOT))
        f.add_command(label="Open the 2D folder", command=lambda: self._reveal(DIR_2D))
        f.add_command(label="Open the 3D folder", command=lambda: self._reveal(DIR_3D))
        f.add_separator()
        f.add_command(label="Save the log as...", command=self.save_log)
        f.add_separator()
        f.add_command(label="Exit", command=self.on_close)
        m.add_cascade(label="File", menu=f)

        e = tk.Menu(m, tearoff=0)
        e.add_command(label="Copy the command of this page",
                      command=lambda: self.current.copy_cmd()
                      if isinstance(self.current, ActionPage) else None)
        e.add_command(label="Reset this page to defaults",
                      command=lambda: self.current.reset()
                      if isinstance(self.current, ActionPage) else None)
        e.add_separator()
        e.add_command(label="Clear the log", command=self.clear_log)
        e.add_command(label="Clear the problem list", command=self.clear_problems)
        m.add_cascade(label="Edit", menu=e)

        v = tk.Menu(m, tearoff=0)
        v.add_command(label="Viewer", command=lambda: self._goto_viewer(None))
        v.add_command(label="Log", command=lambda: self.bottom.select(0))
        v.add_command(label="Problems", command=lambda: self.bottom.select(1))
        v.add_command(label="Monitor", command=lambda: self.bottom.select(2))
        v.add_separator()
        v.add_command(label="Bigger bottom panel",
                      command=lambda: self.split.sash_place(0, 0, max(240, self.split.winfo_height() - 420)))
        v.add_command(label="Smaller bottom panel",
                      command=lambda: self.split.sash_place(0, 0, self.split.winfo_height() - 180))
        m.add_cascade(label="View", menu=v)

        t = tk.Menu(m, tearoff=0)
        t.add_command(label="Check this computer can run everything",
                      command=self.env_check)
        t.add_command(label="Self-test of the travel-time solver",
                      command=self.flow_selftest)
        t.add_command(label="Where is everything?", command=self.paths_dialog)
        t.add_separator()
        t.add_command(label="Choose which Python runs the scripts...",
                      command=self.choose_python)
        m.add_cascade(label="Tools", menu=t)

        w = tk.Menu(m, tearoff=0)
        w.add_command(label="Welcome page", command=self.show_welcome)
        w.add_command(label="Close all open pages", command=self.close_pages)
        m.add_cascade(label="Window", menu=w)

        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Tutorial", command=self.show_tutorial)
        h.add_separator()
        h.add_command(label="About " + APP, command=self.show_about)
        m.add_cascade(label="Help", menu=h)

        self.config(menu=m)

    # ------------------------------------------------------------------ body
    def _body(self):
        self.split = tk.PanedWindow(self, orient="vertical", bg=BG, sashwidth=5,
                                    sashrelief="raised")
        self.split.pack(fill="both", expand=True)

        upper = tk.PanedWindow(self.split, orient="horizontal", bg=BG,
                               sashwidth=5, sashrelief="raised")
        self.split.add(upper, stretch="always")

        # ---- left: the pipeline tree
        left = tk.Frame(upper, bg=PANEL, highlightbackground=LINE,
                        highlightthickness=1, width=300)
        upper.add(left, minsize=240)
        self.upper = upper
        tk.Label(left, text="Pipeline", bg=HEAD, fg=INK, anchor="w",
                 font=base_font(9, True), padx=8, pady=4).pack(fill="x")

        modef = tk.Frame(left, bg=PANEL)
        modef.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(modef, text="Working in:", bg=PANEL, fg=MUTE,
                 font=base_font(8)).pack(side="left")
        self.mode = tk.StringVar(value="3D")
        for txt in ("3D", "2D"):
            tk.Radiobutton(modef, text=txt, value=txt, variable=self.mode,
                           bg=PANEL, font=base_font(9, True), fg=INFOC,
                           activebackground=PANEL, highlightthickness=0,
                           command=self.on_mode).pack(side="left", padx=4)
        self.modehint = tk.Message(
            left, bg=PANEL, fg=MUTE, font=base_font(8), width=240, justify="left",
            text="3D is our own code: geometry, CompLaB3D campaigns, training, "
                 "prediction.")
        self.modehint.pack(fill="x", padx=8, pady=(0, 4))

        self.tree = ttk.Treeview(left, show="tree", selectmode="browse")
        self.tree.pack(fill="both", expand=True, padx=6, pady=6)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree)
        self.build_tree()

        # ---- right: the pages
        right = tk.Frame(upper, bg=BG)
        upper.add(right, minsize=520, stretch="always")
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)
        self.tabs.bind("<<NotebookTabChanged>>", self.on_tab)

        self.viewer = Viewer(self.tabs, self)
        self.tabs.add(self.viewer, text="Viewer")

        # ---- bottom: log / problems / monitor
        lower = tk.Frame(self.split, bg=BG, height=220)
        self.split.add(lower, minsize=120)
        self.bottom = ttk.Notebook(lower)
        self.bottom.pack(fill="both", expand=True, padx=4, pady=(0, 2))

        logf = tk.Frame(self.bottom, bg=PANEL)
        self.bottom.add(logf, text="Log")
        self.log = tk.Text(logf, font=mono_font(8), bg="#ffffff", fg=INK,
                           wrap="none", relief="flat")
        lsb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=lsb.set)
        lsb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)
        self.log.tag_configure("err", foreground=ERRC)
        self.log.tag_configure("warn", foreground=WARNC)
        self.log.tag_configure("ok", foreground=OKC)
        self.log.tag_configure("hdr", foreground=INFOC, font=mono_font(8))

        probf = tk.Frame(self.bottom, bg=PANEL)
        self.bottom.add(probf, text="Problems")
        self.prob = tk.Text(probf, font=mono_font(8), bg="#fff8f7", fg=ERRC,
                            wrap="word", relief="flat")
        psb = ttk.Scrollbar(probf, command=self.prob.yview)
        self.prob.configure(yscrollcommand=psb.set)
        psb.pack(side="right", fill="y")
        self.prob.pack(fill="both", expand=True)

        monf = tk.Frame(self.bottom, bg=PANEL)
        self.bottom.add(monf, text="Monitor")
        self.mon_left = tk.Frame(monf, bg=PANEL, width=280)
        self.mon_left.pack(side="left", fill="y", padx=8, pady=6)
        self.mon_lbl = tk.Label(self.mon_left, bg=PANEL, fg=INK, justify="left",
                                anchor="nw", font=mono_font(8),
                                text="Nothing running.")
        self.mon_lbl.pack(anchor="nw")
        self.mon_plot = tk.Frame(monf, bg=PANEL)
        self.mon_plot.pack(side="left", fill="both", expand=True)
        self.mon_canvas = None

        self.after(200, lambda: self.split.sash_place(0, 0, 560))
        self.after(220, lambda: self.upper.sash_place(0, 300, 0))

    # ------------------------------------------------------------- statusbar
    def _statusbar(self):
        sb = tk.Frame(self, bg=BG, height=26)
        sb.pack(fill="x", side="bottom")
        self.notif = tk.Label(sb, text="  ●  ready", bg=BG, fg=OKC,
                              font=base_font(9), anchor="w")
        self.notif.pack(side="left", padx=6)
        self.pbar = ttk.Progressbar(sb, length=260, mode="determinate",
                                    style="H.Horizontal.TProgressbar")
        self.pbar.pack(side="right", padx=8, pady=3)
        self.stat = tk.Label(sb, text="", bg=BG, fg=MUTE, font=base_font(8),
                             anchor="e")
        self.stat.pack(side="right", padx=6)
        tk.Label(sb, text="python: " + os.path.basename(PYTHON), bg=BG, fg=MUTE,
                 font=base_font(8)).pack(side="left", padx=10)

    # ---------------------------------------------------------------- tree
    def build_tree(self):
        self.tree.delete(*self.tree.get_children())
        label = lambda k: "  %s  (%s)" % (self.actions[k].title,
                                          TREE_HINTS.get(k, ""))
        if self.mode.get() == "3D":
            groups = [("Set up", ["install", "checkall",
                                  "dataset3d", "geometry"]),
                      ("Simulate", ["complab_campaign", "collect_complab_output", "ingest2d"]),
                      ("Learn", ["train", "evaluate", "sweep"]),
                      ("Predict", ["predict"])]
            for g, keys in groups:
                node = self.tree.insert("", "end", text=" " + g, open=True)
                for k in keys:
                    self.tree.insert(node, "end", iid="act:" + k, text=label(k))
            look = self.tree.insert("", "end", text=" Look at results", open=True)
            self.tree.insert(look, "end", iid="view",
                             text="   Viewer  (open a dataset or a prediction)")
            helpn = self.tree.insert("", "end", text=" Help", open=True)
            self.tree.insert(helpn, "end", iid="tut",
                             text="   Tutorial  (five worked projects)")
            self.tree.insert(helpn, "end", iid="act:capabilities",
                             text=label("capabilities"))
        else:
            groups = [("Set up", ["install", "checkall",
                                  "dataset2d", "ingestsims", "heewon"]),
                      ("Simulate", ["collect_complab_output"]),
                      ("Learn", ["train", "evaluate", "sweep"]),
                      ("Predict", ["predict"])]
            for g, keys in groups:
                node = self.tree.insert("", "end", text=" " + g, open=True)
                for k in keys:
                    self.tree.insert(node, "end", iid="act:" + k, text=label(k))
            pub = self.tree.insert("", "end",
                                   text=" The published release", open=True)
            self.tree.insert(pub, "end", iid="2d:browse",
                             text="   Browse its 3000 domains  (real pore structures)")
            self.tree.insert(pub, "end", iid="2d:notebooks",
                             text="   Its notebooks  (opens the folder)")
            self.tree.insert(pub, "end", iid="2d:params",
                             text="   Its trained models  (weights you can start from)")
            b = self.tree.insert("", "end", text=" Feed 2D into 3D", open=True)
            self.tree.insert(b, "end", iid="act:ingest2d", text=label("ingest2d"))
            look = self.tree.insert("", "end", text=" Look at results", open=True)
            self.tree.insert(look, "end", iid="view",
                             text="   Viewer  (open a dataset or a prediction)")
            helpn = self.tree.insert("", "end", text=" Help", open=True)
            self.tree.insert(helpn, "end", iid="tut",
                             text="   Tutorial  (five worked projects)")
            self.tree.insert(helpn, "end", iid="act:capabilities",
                             text=label("capabilities"))

        # Not an action page: the one thing the old Run menu had that nothing
        # else did. Kept where every other control lives.
        self.tree.insert("", "end", iid="stop",
                         text=" Stop what is running  (halts the current job)")

    def on_mode(self):
        if self.mode.get() == "3D":
            self.modehint.configure(
                text="3D is our own code: geometry, CompLaB3D campaigns, "
                     "training, prediction.")
        else:
            self.modehint.configure(
                text="Everything works in 2D too: build a 2D dataset, train, "
                     "evaluate and predict, all natively two-dimensional. A 2D "
                     "run is about fifty times cheaper than a 3D one.")
        self.build_tree()

    def on_tree(self, _evt=None):
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid.startswith("act:"):
            self.open_action(iid[4:])
        elif iid == "view":
            self.tabs.select(self.viewer)
        elif iid == "tut":
            self.show_tutorial()
        elif iid == "stop":
            self.stop_run()
        elif iid == "2d:browse":
            self.tabs.select(self.viewer)
            self.viewer.open_2d()
        elif iid == "2d:notebooks":
            self._reveal(DIR_2D)
            self.status("The 2D work is in Jupyter notebooks. Opened the folder; "
                        "run them with:  jupyter notebook")
        elif iid == "2d:params":
            self._reveal(os.path.join(DIR_2D, "parameters"))

    # ---------------------------------------------------------------- pages
    def open_action(self, key):
        a = self.actions[key]
        if key in self.pages and self.pages[key].winfo_exists():
            self.tabs.select(self.pages[key])
            return
        page = ActionPage(self.tabs, self, a)
        self.pages[key] = page
        self.tabs.insert(0, page, text=a.title.split(" (")[0][:26])
        self.tabs.select(page)

    def close_pages(self):
        for k, p in list(self.pages.items()):
            try:
                self.tabs.forget(p)
            except Exception:                                  # noqa: BLE001
                pass
            self.pages.pop(k, None)

    def on_tab(self, _e=None):
        try:
            w = self.tabs.nametowidget(self.tabs.select())
        except Exception:                                      # noqa: BLE001
            return
        self.current = w

    def _goto_viewer(self, what):
        self.tabs.select(self.viewer)
        if what == "dataset":
            self.viewer.open_dataset()
        elif what == "geometry":
            self.viewer.open_geometry()
        elif what == "2d":
            self.viewer.open_2d()
        elif what == "pred":
            self.viewer.open_pred()

    # ---------------------------------------------------------------- run
    def launch(self, action, cmd, total):
        if self.runner.busy():
            messagebox.showwarning(APP, "Something is already running. Stop it "
                                        "first, or wait for it to finish.")
            return
        self.hist = []
        self.total_steps = total
        self.run_action = action
        self.run_cmd = cmd
        self.pbar.configure(mode="determinate" if total else "indeterminate",
                            maximum=total or 100, value=0)
        if not total:
            self.pbar.start(60)
        self.set_notif("running", INFOC)
        self.bottom.select(0)
        self.logline("", None)
        self.logline("=" * 78, "hdr")
        self.logline("RUN  %s" % action.title, "hdr")
        self.logline("     " + " ".join(_q(c) for c in cmd), "hdr")
        self.logline("=" * 78, "hdr")
        self.runner.start(cmd, cwd=action.cwd or os.path.dirname(action.script))
        self.status("Running: %s" % action.title)

    def stop_run(self):
        if self.runner.busy():
            self.runner.stop()
            self.logline("--- stopped by the user ---", "warn")
            self.status("Stopped.")
        else:
            self.status("Nothing is running.")

    def on_line(self, line):
        low = line.lower()
        tag = None
        if ("traceback" in low or "error" in low or "failed" in low
                or line.startswith("FAIL") or " FAIL" in line):
            tag = "err"
            self.errors.append(line)
        elif "warning" in low or line.lstrip().startswith("WARNING"):
            tag = "warn"
        elif "PASS" in line or line.startswith("wrote") or "ok" == low.strip():
            tag = "ok"
        self.logline(line, tag)

        a = self.run_action
        if a is None:
            return
        if a.progress_re:
            m = a.progress_re.search(line)
            if m:
                try:
                    # A pattern with TWO groups carries its own total, as in
                    # "[7/60]". That matters for the dataset builders, whose
                    # total is rocks TIMES conditions and so is not the value
                    # of any one box on the page.
                    if m.lastindex and m.lastindex >= 2:
                        self.total_steps = max(int(m.group(2)), 1)
                        self.pbar.configure(maximum=self.total_steps)
                    if self.total_steps:
                        self.pbar.configure(value=min(int(m.group(1)),
                                                      self.total_steps))
                except Exception:                              # noqa: BLE001
                    pass
        if a.metric_re:
            m = a.metric_re.search(line)
            if m:
                try:
                    self.hist.append((int(m.group(1)), float(m.group(2)),
                                      float(m.group(3))))
                    self.update_monitor()
                except Exception:                              # noqa: BLE001
                    pass

    def on_done(self, rc, secs):
        try:
            self.pbar.stop()
        except Exception:                                      # noqa: BLE001
            pass
        self.pbar.configure(mode="determinate")
        if rc == 0:
            self.pbar.configure(value=self.pbar["maximum"])
            self.logline("--- finished in %s ---" % _dur(secs), "ok")
            self.set_notif("finished", OKC)
            self.status("Finished in %s." % _dur(secs))
            checking = any(
                f.key == "show_settings" and bool(f.value())
                for f in (self.run_action.fields if self.run_action else []))
            if checking:
                # It listed what a real run WOULD produce, after a run that
                # deliberately produced nothing. That reads as a failure, and
                # the empty output folder confirms the misreading.
                self.logline("Nothing was simulated: 'Just check the numbers, "
                             "do not simulate' was ticked.", "hdr")
                self.logline("  The report above is what your settings mean. "
                             "No file was written, which is why\n  this "
                             "finished in a second and the output folder is "
                             "empty. Untick that box\n  and press Run to "
                             "build the dataset.\n")
            elif self.run_action:
                outs = "\n".join("  • %s — %s" % (n, e)
                                 for n, e in self.run_action.outputs_text)
                self.logline("Files this produced:", "hdr")
                self.logline(outs, None)
        else:
            self.pbar.configure(value=0)
            self.logline("--- FAILED, exit code %s, after %s ---"
                         % (rc, _dur(secs)), "err")
            self.set_notif("failed", ERRC)
            self.status("Failed.")
            self.problem("%s failed with exit code %s.\nCommand:\n  %s\n\n"
                         "Last lines:\n%s"
                         % (self.run_action.title if self.run_action else "the run",
                            rc, " ".join(_q(c) for c in (self.run_cmd or [])),
                            "\n".join("  " + e for e in self.errors[-8:])))
        self.errors = []
        self.run_action = None
        self.update_monitor(final=True, secs=secs)

    def _pump(self):
        if not self.winfo_exists():
            return
        self.runner.pump()
        if self.runner.busy() and self.runner.t0:
            self.mon_lbl.configure(
                text="RUNNING\n  %s\n\n  elapsed   %s\n  epochs    %d recorded"
                     % ((self.run_action.title if self.run_action else "-"),
                        _dur(time.time() - self.runner.t0), len(self.hist)))
        try:
            self.after(120, self._pump)
        except Exception:                                      # noqa: BLE001
            pass

    # ---------------------------------------------------------------- output
    def logline(self, text, tag=None):
        self.log.insert("end", text + "\n", tag or ())
        if int(self.log.index("end-1c").split(".")[0]) > 6000:
            self.log.delete("1.0", "2000.0")
        self.log.see("end")

    def problem(self, text):
        self.prob.insert("end", time.strftime("[%H:%M:%S]  ") + text + "\n\n")
        self.prob.see("end")
        self.set_notif("problem", ERRC)
        self.bottom.select(1)

    def clear_log(self):
        self.log.delete("1.0", "end")

    def clear_problems(self):
        self.prob.delete("1.0", "end")
        self.set_notif("ready", OKC)

    def save_log(self):
        p = filedialog.asksaveasfilename(defaultextension=".txt",
                                         filetypes=[("Text", "*.txt")])
        if p:
            with open(p, "w") as fh:
                fh.write(self.log.get("1.0", "end"))
            self.status("Saved %s" % p)

    def status(self, s):
        self.stat.configure(text=s)

    def set_notif(self, txt, colour):
        self.notif.configure(text="  ●  " + txt, fg=colour)

    # ---------------------------------------------------------------- monitor
    def update_monitor(self, final=False, secs=None):
        if not self.hist:
            if final:
                self.mon_lbl.configure(text="Finished in %s." % _dur(secs or 0))
            return
        ep, tr, te = self.hist[-1]
        best = min(h[2] for h in self.hist)
        self.mon_lbl.configure(
            text=("TRAINING\n\n  epoch          %d\n  train error    %.6f\n"
                  "  test error     %.6f\n  best test      %.6f\n\n"
                  "  train  = data it is learning from\n"
                  "  test   = structures it has\n           never seen\n\n"
                  "  If train keeps falling while\n  test flattens, it has begun\n"
                  "  memorising rather than\n  learning."
                  % (ep, tr, te, best)))
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            for w in self.mon_plot.winfo_children():
                w.destroy()
            fig, ax = plt.subplots(figsize=(5.2, 2.2))
            xs = [h[0] for h in self.hist]
            # markers, not just a line: with a single epoch recorded a bare line
            # has nothing to join and the plot looks empty, which reads as "it is
            # not working" when in fact it is
            mk = "o" if len(xs) < 12 else None
            ax.plot(xs, [h[1] for h in self.hist], lw=1.4, marker=mk, ms=4,
                    label="train")
            ax.plot(xs, [h[2] for h in self.hist], lw=1.4, marker=mk, ms=4,
                    label="test (unseen)")
            if len(xs) == 1:
                ax.set_title("one epoch so far — the curves appear from the second",
                             fontsize=8, color=MUTE)
            ax.set_xlabel("epoch", fontsize=8)
            ax.set_ylabel("error", fontsize=8)
            ax.set_yscale("log")
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, frameon=False)
            ax.grid(alpha=.3, lw=.5)
            fig.tight_layout()
            c = FigureCanvasTkAgg(fig, master=self.mon_plot)
            c.draw()
            c.get_tk_widget().pack(fill="both", expand=True)
            self.mon_canvas = c
        except Exception:                                      # noqa: BLE001
            pass

    # ---------------------------------------------------------------- dialogs
    def show_welcome(self):
        if getattr(self, "_welcome", None) and self._welcome.winfo_exists():
            self.tabs.select(self._welcome)
            return
        f = tk.Frame(self.tabs, bg=BG)
        self._welcome = f
        self.tabs.insert(0, f, text="Welcome")
        self.tabs.select(f)
        tk.Label(f, text=APP, bg=BG, fg=INK, font=base_font(20, True),
                 anchor="w").pack(fill="x", padx=20, pady=(20, 0))
        tk.Label(f, text="Pore-scale reactive transport, learned instead of simulated.",
                 bg=BG, fg=MUTE, font=base_font(11), anchor="w").pack(
            fill="x", padx=20, pady=(0, 14))
        grid = tk.Frame(f, bg=BG)
        grid.pack(fill="both", expand=True, padx=20)
        cards = [
            ("1.  Set up", "Build a dataset to learn from: pore structures, "
                           "flow and reactive transport, solved here on this "
                           "machine.", "dataset3d"),
            ("2.  Simulate", "Build the CompLaB3D campaign, then gather the "
                             "output into one dataset.", "complab_campaign"),
            ("3.  Learn", "Train the network, then check it on structures it "
                          "has never seen.", "train"),
            ("4.  Predict", "Ask it about a new structure. Under a second, "
                            "against hours for the simulation.", "predict"),
        ]
        for i, (t, d, key) in enumerate(cards):
            c = tk.Frame(grid, bg=PANEL, highlightbackground=LINE,
                         highlightthickness=1)
            c.grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=6)
            grid.columnconfigure(i % 2, weight=1)
            grid.rowconfigure(i // 2, weight=1)
            tk.Label(c, text=t, bg=PANEL, fg=INFOC, font=base_font(12, True),
                     anchor="w").pack(fill="x", padx=12, pady=(10, 2))
            tk.Message(c, text=d, bg=PANEL, fg=INK, font=base_font(9),
                       width=380, anchor="w", justify="left").pack(
                fill="x", padx=12)
            tk.Button(c, text="Open", font=base_font(9), padx=14,
                      command=lambda k=key: self.open_action(k)).pack(
                anchor="w", padx=12, pady=10)
        bar = tk.Frame(f, bg=BG)
        bar.pack(fill="x", padx=20, pady=14)
        tk.Button(bar, text="Read the tutorial", font=base_font(10, True),
                  padx=16, pady=4, command=self.show_tutorial).pack(side="left")
        tk.Button(bar, text="Glossary", font=base_font(10), padx=16, pady=4,
                  command=self.show_glossary).pack(side="left", padx=8)
        tk.Button(bar, text="Open the Viewer", font=base_font(10), padx=16,
                  pady=4, command=lambda: self.tabs.select(self.viewer)).pack(
            side="left")

    def _text_window(self, title, body, mono=False, width=96, height=34):
        w = tk.Toplevel(self)
        w.title(title)
        w.configure(bg=BG)
        w.geometry("860x640")
        t = tk.Text(w, wrap="word", font=mono_font(9) if mono else base_font(10),
                    bg=PANEL, fg=INK, padx=16, pady=12, relief="flat")
        sb = ttk.Scrollbar(w, command=t.yview)
        t.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        t.pack(fill="both", expand=True)
        t.insert("1.0", body)
        t.configure(state="disabled")
        tk.Button(w, text="Close", command=w.destroy, font=base_font(9),
                  padx=16).pack(pady=6)
        return w

    def show_tutorial(self):
        p = os.path.join(HERE, "TUTORIAL.txt")
        body = TUTORIAL
        if os.path.exists(p):
            try:
                body = open(p, encoding="utf-8").read()
            except Exception:                                  # noqa: BLE001
                pass
        self._text_window("Tutorial — " + APP, body, mono=True)

    def show_glossary(self):
        w = tk.Toplevel(self)
        w.title("Glossary — our words and theirs")
        w.geometry("900x560")
        w.configure(bg=BG)
        tk.Label(w, text="The same idea, said two ways", bg=BG, fg=INK,
                 font=base_font(13, True), anchor="w").pack(fill="x", padx=14,
                                                            pady=(12, 2))
        tk.Label(w, text="Left column: what a reactive transport scientist calls "
                         "it.   Middle: what a computer scientist calls it.",
                 bg=BG, fg=MUTE, font=base_font(9), anchor="w").pack(
            fill="x", padx=14, pady=(0, 8))
        cols = ("transport", "ml", "meaning")
        tv = ttk.Treeview(w, columns=cols, show="headings", height=18)
        for c, t, wdt in (("transport", "In transport", 190),
                          ("ml", "In machine learning", 190),
                          ("meaning", "What it actually means", 470)):
            tv.heading(c, text=t)
            tv.column(c, width=wdt, anchor="w")
        for a, b, c in GLOSSARY:
            tv.insert("", "end", values=(a, b, c))
        tv.pack(fill="both", expand=True, padx=14, pady=6)
        tk.Button(w, text="Close", command=w.destroy, font=base_font(9),
                  padx=16).pack(pady=8)

    def show_about(self):
        self._text_window("About " + APP,
                          ABOUT_TEXT % (APP, VERSION, ROOT, sys.executable),
                          mono=False)

    def paths_dialog(self):
        rows = [("Project root", ROOT),
                ("", ""),
                ("CODE, read only", ""),
                ("  2D release", DIR_2D),
                ("  3D code", DIR_3D),
                ("  3D tools", TOOLS),
                ("  3D model", MODEL),
                ("  bridge", DIR_BRIDGE),
                ("  the window", HERE),
                ("", ""),
                ("WHAT YOU PRODUCE", ""),
                ("  work", WORK),
                ("  a new 2D run", NEW2D),
                ("  a new 3D run", NEW3D),
                ("  geometries", os.path.join(WORK, "geometries")),
                ("  CompLaB campaign", os.path.join(WORK, "complab_campaign")),
                ("", ""),
                ("Python used to run the scripts", PYTHON),
                ("Python running this window", sys.executable)]
        body = ("Everything under work/ is output: one folder per experiment,\n"
                "each holding its own dataset, model, figures and predictions.\n"
                "Nothing else is written to.\n\n")
        body += "\n".join(
            (n if not p else "%-32s %s   %s"
             % (n, p, "" if os.path.exists(p) else "   << not created yet"))
            for n, p in rows)
        self._text_window("Where is everything?", body, mono=True)

    def env_check(self):
        lines = ["Checking this computer can run everything.", "",
                 "Scripts are run with:", "  " + PYTHON, ""]
        ok = True
        mods = [("numpy", "arrays"), ("scipy", "distance fields"),
                ("h5py", "datasets"), ("matplotlib", "figures"),
                ("torch", "training and prediction"),
                ("skimage", "3D surface rendering")]
        probe = ("import importlib,sys\n"
                 "for m in %r:\n"
                 "    try:\n"
                 "        importlib.import_module(m); print('OK '+m)\n"
                 "    except Exception:\n"
                 "        print('NO '+m)\n" % [m for m, _ in mods])
        have = set()
        try:
            r = subprocess.run([PYTHON, "-c", probe], capture_output=True,
                               text=True, timeout=90)
            for ln in r.stdout.splitlines():
                if ln.startswith("OK "):
                    have.add(ln[3:])
        except Exception as e:                                 # noqa: BLE001
            lines.append("  could not run that Python: %s" % e)
        for mod, why in mods:
            if mod in have:
                lines.append("  OK      %-12s  %s" % (mod, why))
            else:
                ok = False
                lines.append("  MISSING %-12s  %s   ->  \"%s\" -m pip install %s"
                             % (mod, why, PYTHON, mod))
        lines.append("")
        for n, p in (("2D release", DIR_2D), ("3D code", DIR_3D),
                     ("bridge", DIR_BRIDGE)):
            lines.append("  %-10s %s   %s"
                         % (n, p, "found" if os.path.isdir(p) else "MISSING"))
            ok &= os.path.isdir(p)
        try:
            r = subprocess.run(
                [PYTHON, "-c", "import torch;print(torch.cuda.is_available());"
                                "print(torch.cuda.get_device_name(0) "
                                "if torch.cuda.is_available() else '-')"],
                capture_output=True, text=True, timeout=90)
            out = r.stdout.split()
            lines.append("")
            if out and out[0] == "True":
                lines.append("  GPU available : yes, %s" % " ".join(out[1:]))
            else:
                lines.append("  GPU available : no")
                lines.append("  Training will run on the processor, which is fine "
                             "for the practice dataset and slow for the real one.")
        except Exception:                                      # noqa: BLE001
            pass
        lines.append("")
        lines.append("RESULT: " + ("everything needed is present."
                                   if ok else "something is missing, see above."))
        self._text_window("Environment check", "\n".join(lines), mono=True)

    def flow_selftest(self):
        script = os.path.join(TOOLS, "flow_coordinates.py")
        if not os.path.exists(script):
            messagebox.showerror(APP, "Cannot find %s" % script)
            return
        a = Action("flowtest", "Travel-time self-test", "", script,
                   [Field("st", "", "bool", True, flag="--self-test")],
                   [("nothing", "runs on analytic cases it generates itself")],
                   ["Checks the travel time equals x in a straight duct.",
                    "Checks it doubles when the speed halves.",
                    "Checks it is unchanged by the Peclet number once normalised.",
                    "Checks a dead-end pocket stays finite and is flagged stagnant."],
                   [("nothing on disk", "the result is the pass or fail list in the log")])
        self.launch(a, a.command(), None)

    def _open_doc(self, path):
        if not os.path.exists(path):
            messagebox.showinfo(APP, "Not found:\n%s" % path)
            return
        try:
            body = open(path, encoding="utf-8").read()
        except Exception as e:                                 # noqa: BLE001
            messagebox.showerror(APP, str(e))
            return
        self._text_window(os.path.basename(path), body, mono=True)

    def _reveal(self, d):
        if not os.path.isdir(d):
            messagebox.showinfo(APP, "That folder does not exist:\n%s" % d)
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(d)                                # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception:                                      # noqa: BLE001
            messagebox.showinfo(APP, d)

    def on_close(self):
        if self.runner.busy():
            if not messagebox.askyesno(
                    APP, "Something is still running. Stop it and quit?"):
                return
            self.runner.stop()
        self.destroy()


def _dur(s):
    s = int(s)
    if s < 60:
        return "%d s" % s
    if s < 3600:
        return "%d min %02d s" % (s // 60, s % 60)
    return "%d h %02d min" % (s // 3600, (s % 3600) // 60)


def missing_packages():
    out = []
    for m, pkg in (("numpy", "numpy"), ("scipy", "scipy"), ("h5py", "h5py"),
                   ("matplotlib", "matplotlib"), ("skimage", "scikit-image"),
                   ("torch", "torch")):
        try:
            __import__(m)
        except Exception:                                      # noqa: BLE001
            out.append(pkg)
    return out


def main():
    app = Studio()
    missing = missing_packages()
    if missing:
        def offer():
            ans = messagebox.askyesno(
                APP,
                "Some Python packages this project needs are not installed yet:\n\n"
                "    " + "\n    ".join(missing) + "\n\n"
                "They go into:\n    " + PYTHON + "\n\n"
                "This is normal on a fresh machine, and the window can install "
                "them for you. It takes a few minutes and needs an internet "
                "connection.\n\n"
                "Open the installer now?")
            if ans:
                app.open_action("install")
                app.status("Press Run to install: " + ", ".join(missing))
            else:
                app.status("Packages missing: " + ", ".join(missing)
                           + ".  Set up > Install the Python packages.")
                app.problem(
                    "These packages are missing, so most buttons will not work "
                    "yet:\n    " + "\n    ".join(missing)
                    + "\n\nFix it with:  Set up > Install the Python packages\n"
                      "or by hand:\n    \"" + PYTHON + "\" -m pip install "
                    + " ".join(missing))
        app.after(500, offer)
    app.mainloop()


if __name__ == "__main__":
    main()
