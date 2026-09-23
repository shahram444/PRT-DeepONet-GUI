#!/usr/bin/env python3
"""
prt_core/reactions.py -- which chemistry a model is for, as data rather than code.

WHAT CHANGED FROM THE 2D VERSION
    The published 2D release has three reactions and each one is a separate
    notebook with its own hard-wired widths: Monod has two dimensionless
    numbers and a four-input trunk, reversible sorption has three and four,
    irreversible sorption has two and three because it is a steady state and
    has no time column. Nothing in that code says so out loud; the widths are
    constructor arguments typed into a cell.

    Our own work added a fourth chemistry, acetate oxidation coupled to sulfate
    reduction, and hard-wired it in the same way, in a different place.

    A reaction is not code. It is a list of chemicals, a list of dimensionless
    numbers with the ranges they were trained over, and whether it is a steady
    state or a transient. This file says that once, for every reaction we have,
    so that adding a fifth is a table entry rather than a new script, and so
    that a checkpoint can record which chemistry it is for instead of leaving
    the reader to guess from the number of columns.

WHY IT MATTERS MORE THAN IT LOOKS
    The widths are what decide whether a checkpoint loads. A model trained for
    the Monod reaction and handed reversible sorption's three parameters fails
    with a shape error if you are lucky and a silent mis-scaling if you are
    not, because the parameter branch takes whatever width it is built with.
    With the reaction named in the checkpoint, the mismatch is caught by name
    and the message says which two chemistries were confused.

WHAT IS IN HERE
    REACTIONS            every reaction we can train or predict, by key
    Reaction             one entry: species, parameters, trunk, steady or not
    get(key)             one of them, with a useful error if the key is wrong
    from_file(path)      a reaction defined in a .json or a CompLaB-style .xml,
                         so a new chemistry needs no change to this file
    normalise(r, values) raw dimensionless numbers onto the model's own scale

    python -m prt_core.reactions                 list them
    python -m prt_core.reactions --self-test     check them
"""

import json
import os

try:                                   # the package, when imported normally
    from .conventions import DISTANCE_CONVENTIONS
except ImportError:                    # run as a file, for the self-test
    DISTANCE_CONVENTIONS = ("ours", "published", "published_pore")


class Reaction:
    """One chemistry, as the model sees it.

    species       the names of the fields, in the order the dataset stores them
    params        [(name, low, high, how)] where how is 'linear' or 'log10'.
                  'linear' maps the range onto 0 to 1, which is what the
                  published notebooks do. 'log10' takes the logarithm, which is
                  what our own training does with Peclet and Damkohler because
                  they are swept over decades.
    steady        True for a steady state: the trunk then has no time column
    trunk_cols    the trunk, with the distance or flow feature LAST. That
                  position is a convention the whole project relies on, so a
                  checkpoint's columns can be read off in one order whatever
                  the switches were.
    source        'published' for the Kim and Jung release, 'ours' otherwise.
                  It decides which distance convention is the default, and they
                  are opposite to each other: see prt_core/conventions.py.
    """

    def __init__(self, key, title, species, params, steady=False,
                 trunk_cols=None, source="ours", distance_convention=None,
                 weights=None, notes=""):
        self.key = key
        self.title = title
        self.species = list(species)
        self.params = [tuple(p) for p in params]
        self.steady = bool(steady)
        self.source = source
        # The published reactions were fitted with the distance column scaled
        # their way; ours with ours. Getting this wrong hands a trained trunk a
        # column that runs the other way, which is worse than a missing input
        # because it looks plausible.
        self.distance_convention = distance_convention or (
            "published" if source == "published" else "ours")
        self.weights = weights
        self.notes = notes
        if trunk_cols is None:
            cols = ["x", "y", "z"]
            if not self.steady:
                cols.append("t")
            cols.append("gdf")
            trunk_cols = cols
        self.trunk_cols = list(trunk_cols)

    # ------------------------------------------------------------- properties
    @property
    def param_names(self):
        return [p[0] for p in self.params]

    @property
    def n_params(self):
        return len(self.params)

    def trunk_for(self, ndim):
        """The trunk columns in 2 or 3 dimensions.

        The z column is dropped in 2D deliberately: it would be constant, and a
        constant input is a dead input that still costs weights.
        """
        cols = [c for c in self.trunk_cols if not (c == "z" and ndim < 3)]
        return cols

    def normalise(self, values):
        """Raw dimensionless numbers onto the scale the parameter branch wants.

        Each column is handled by its own rule, selected by NAME, never by
        position. Position is what put a half-saturation constant through a
        logarithm once, in a layout that happened to have six columns.
        """
        out = []
        for v, (name, lo, hi, how) in zip(values, self.params):
            v = float(v)
            if how == "log10":
                import math
                out.append(math.log10(max(v, 1e-12)))
            else:
                span = float(hi) - float(lo)
                out.append((v - float(lo)) / span if span else 0.0)
        return out

    def check_values(self, values):
        """Complaints about values outside the range this was trained over.

        Returns a list of sentences, empty when everything is inside. A value
        outside the range is an extrapolation, not a prediction, and the caller
        is told rather than left to find out from a strange field.
        """
        bad = []
        if len(values) != self.n_params:
            bad.append("%s takes %d parameters (%s), %d given"
                       % (self.key, self.n_params,
                          ", ".join(self.param_names), len(values)))
            return bad
        for v, (name, lo, hi, _how) in zip(values, self.params):
            if not (float(lo) <= float(v) <= float(hi)):
                bad.append("%s = %g is outside the trained range %g to %g. "
                           "This is an extrapolation, not a prediction."
                           % (name, float(v), float(lo), float(hi)))
        return bad

    def as_dict(self):
        return dict(key=self.key, title=self.title, species=self.species,
                    params=self.params, steady=self.steady,
                    trunk_cols=self.trunk_cols, source=self.source,
                    distance_convention=self.distance_convention,
                    weights=self.weights, notes=self.notes)

    def __repr__(self):
        return "<Reaction %s: %d species, %d parameters, %s>" % (
            self.key, len(self.species), self.n_params,
            "steady" if self.steady else "transient")


# =============================================================================
#  THE REGISTRY
# =============================================================================
# The three published reactions first, with the ranges read off the notebooks
# in 2D/models/. The ranges are not decoration: they are what normalise() maps
# onto 0 to 1, and a value outside them is an extrapolation.
REACTIONS = {}


def _add(r):
    REACTIONS[r.key] = r
    return r


_add(Reaction(
    "monod", "Monod kinetics (published)",
    species=["C"],
    params=[("Pe", 1.0, 10.0, "linear"), ("Da", 0.05, 0.5, "linear")],
    steady=False, source="published",
    trunk_cols=["x", "y", "z", "t", "gdf"],
    weights="2D/parameters/Monod.pt",
    notes="Kim and Jung's Monod case. Five convolution blocks, four trunk "
          "inputs in 2D. The time ladder in the notebook is [1, 4, 8 ... 80] "
          "divided by 80."))

_add(Reaction(
    "irreversible_sorption", "Irreversible sorption (published)",
    species=["C"],
    params=[("Pe", 1.0, 10.0, "linear"), ("Da_A", 74.0, 740.0, "linear")],
    steady=True, source="published",
    trunk_cols=["x", "y", "z", "gdf"],
    distance_convention="published_pore",
    weights="2D/parameters/Irreversible_Sorption.pt",
    notes="A steady state, so no time column. This one alone scales the "
          "distance over the PORE only; the other two use the whole grid. "
          "That is the notebooks' own difference, not ours."))

_add(Reaction(
    "reversible_sorption", "Reversible sorption (published)",
    species=["C"],
    params=[("Pe", 1.0, 10.0, "linear"), ("Da_A", 0.5, 2.0, "linear"),
            ("Da_D", 0.02, 0.2, "linear")],
    steady=False, source="published",
    trunk_cols=["x", "y", "z", "t", "gdf"],
    weights="2D/parameters/Reversible_Sorption.pt",
    notes="Three dimensionless numbers, so a three-input parameter branch. "
          "The class default says four convolution blocks and it is built "
          "with five; the checkpoint wants five."))

_add(Reaction(
    "acetate_sulfate", "Acetate oxidation with sulfate reduction (ours)",
    species=["Ac", "A", "P", "Bio"],
    params=[("Pe", 1e-3, 1e3, "log10"), ("Da", 1e-3, 1e3, "log10")],
    steady=False, source="ours",
    notes="What build_dataset_2d.py and build_dataset_3d.py simulate by "
          "default. Ac is the donor, A the acceptor, P the product and Bio "
          "the biomass, which is immobile. Peclet and Damkohler are swept "
          "over decades, so they enter as logarithms."))

_add(Reaction(
    "aom_sulfate", "Anaerobic methane oxidation with sulfate (ours)",
    species=["CH4", "SO4", "HS", "HCO3"],
    params=[("Pe", 1e-3, 1e3, "log10"), ("Da", 1e-3, 1e3, "log10")],
    steady=False, source="ours",
    notes="The Phase 1 chemistry: methane and sulfate meeting on a "
          "grain-surface consortium, one mole of each giving one of sulfide "
          "and one of bicarbonate. Same shape as acetate_sulfate, different "
          "names, and the names are what the dataset records."))

DEFAULT = "acetate_sulfate"


# =============================================================================
#  LOOKUP AND LOADING
# =============================================================================
def get(key):
    """One reaction, with an error that says what the alternatives are."""
    if key in REACTIONS:
        return REACTIONS[key]
    raise KeyError(
        "no reaction called %r. This project knows: %s.\n"
        "A chemistry that is not in that list goes in a .json or a "
        "CompLaB-style .xml and is loaded with from_file(), which needs no "
        "change to any code." % (key, ", ".join(sorted(REACTIONS))))


def from_file(path):
    """A reaction defined outside this file.

    Two formats, because the project already has two: a .json holding the
    fields of Reaction, and a CompLaB settings .xml, whose <chemistry> block
    already names the substrates. The xml route means an existing CompLaB.xml
    defines the chemistry without anything being retyped.
    """
    if path.lower().endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        d.setdefault("key", os.path.splitext(os.path.basename(path))[0])
        d.setdefault("title", d["key"])
        return Reaction(**d)

    # the xml route
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    names = []
    for tag in ("name_of_substrates", "name_of_microbes"):
        for el in root.iter(tag):
            names += (el.text or "").split()
    if not names:
        raise ValueError(
            "%s has no <name_of_substrates>, so it does not say what the "
            "chemicals are. A CompLaB settings file does; a geometry file "
            "does not." % path)
    key = os.path.splitext(os.path.basename(path))[0]
    return Reaction(
        key, "%s (from %s)" % (key, os.path.basename(path)),
        species=names,
        params=[("Pe", 1e-3, 1e3, "log10"), ("Da", 1e-3, 1e3, "log10")],
        source="ours",
        notes="Read from %s. The dimensionless numbers default to Peclet and "
              "Damkohler over decades; put a params list in a .json instead if "
              "this chemistry has more." % path)


def resolve(name_or_path):
    """A registry key, or a path to a file defining one."""
    if os.path.exists(name_or_path):
        return from_file(name_or_path)
    return get(name_or_path)


def add_argument(ap, flag="--reaction", default=DEFAULT):
    """The standard command-line flag, so every script spells it the same."""
    ap.add_argument(flag, default=default,
                    help="which chemistry: %s, or the path to a .json or a "
                         "CompLaB .xml defining another one"
                         % ", ".join(sorted(REACTIONS)))
    return ap


# =============================================================================
#  SELF TEST
# =============================================================================
def _self_test():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print("  %-5s %-56s %s" % ("PASS" if cond else "FAIL", name, extra))

    check("every reaction has at least one species and one parameter",
          all(r.species and r.params for r in REACTIONS.values()))
    check("every parameter range is a real range",
          all(float(lo) < float(hi)
              for r in REACTIONS.values() for _n, lo, hi, _h in r.params))
    check("every normalisation rule is one we implement",
          all(h in ("linear", "log10")
              for r in REACTIONS.values() for _n, _l, _hi, h in r.params))
    check("the distance or flow feature is last in every trunk",
          all(r.trunk_cols[-1] in ("gdf", "edt", "tau", "speed", "dwall")
              for r in REACTIONS.values()))
    check("a steady reaction has no time column",
          all(("t" in r.trunk_cols) != r.steady for r in REACTIONS.values()))

    m = get("monod")
    check("the published ranges map onto 0 and 1 at the ends",
          m.normalise([1.0, 0.05]) == [0.0, 0.0]
          and m.normalise([10.0, 0.5]) == [1.0, 1.0],
          str(m.normalise([10.0, 0.5])))
    a = get("acetate_sulfate")
    check("ours take the logarithm instead",
          abs(a.normalise([100.0, 0.01])[0] - 2.0) < 1e-12
          and abs(a.normalise([100.0, 0.01])[1] + 2.0) < 1e-12,
          str(a.normalise([100.0, 0.01])))

    check("the trunk loses z in 2D and keeps it in 3D",
          m.trunk_for(2) == ["x", "y", "t", "gdf"]
          and m.trunk_for(3) == ["x", "y", "z", "t", "gdf"],
          "%s / %s" % (m.trunk_for(2), m.trunk_for(3)))

    check("a value outside the trained range is reported",
          len(m.check_values([50.0, 0.1])) == 1
          and not m.check_values([5.0, 0.1]))
    check("the wrong number of parameters is reported",
          len(m.check_values([1.0])) == 1)

    check("an unknown key names the alternatives",
          "acetate_sulfate" in _err(lambda: get("nope")))

    # the two published sorption cases really do differ, which is the point of
    # recording the convention per reaction
    check("irreversible sorption keeps its own distance convention",
          get("irreversible_sorption").distance_convention == "published_pore"
          and get("monod").distance_convention == "published")
    check("ours keep ours",
          get("acetate_sulfate").distance_convention == "ours")

    # round trip through a .json
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "mine.json")
    with open(p, "w") as f:
        json.dump(dict(key="mine", title="a test", species=["X", "Y"],
                       params=[["Pe", 1.0, 10.0, "linear"]]), f)
    r = from_file(p)
    check("a reaction defined in a .json loads",
          r.species == ["X", "Y"] and r.n_params == 1)

    print("\n%s" % ("Everything passed." if ok else "SOMETHING FAILED."))
    return 0 if ok else 1


def _err(fn):
    try:
        fn()
    except Exception as e:                                     # noqa: BLE001
        return str(e)
    return ""


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(_self_test())
    print("%-24s %-9s %-7s %s" % ("key", "source", "steady", "species and parameters"))
    print("-" * 92)
    for k in sorted(REACTIONS):
        r = REACTIONS[k]
        print("%-24s %-9s %-7s %s | %s" % (
            k, r.source, "yes" if r.steady else "no",
            ", ".join(r.species), ", ".join(r.param_names)))
