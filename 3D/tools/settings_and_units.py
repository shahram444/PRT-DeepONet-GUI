#!/usr/bin/env python3
"""
settings_and_units.py -- where you put your own numbers in.

WHY THIS FILE EXISTS
    CompLaB has one settings file. You open CompLaB.xml, you write your grid
    size, your voxel edge in micrometres, your relaxation time, your pressure
    drop, and then for every chemical its diffusion coefficient in the pore
    and in the biofilm, its starting concentration, and what happens at the
    left and right faces. Then you run it. Everything you control is in one
    place, in the units you measured it in, and you can read the file back
    afterwards and know exactly what you ran.

    The Python generators in this folder had none of that. They took Peclet
    and Damkohler -- two pure numbers -- and everything else was written into
    the source: the inlet concentrations, the half-saturation constant, the
    stoichiometry, the yield, the decay rate, the starting biomass. If you
    wanted acetate at 3 mM instead of 2, or a sulfate diffusivity that was not
    equal to the acetate one, there was nowhere to say so. Not in the window,
    not on the command line, not even in an obvious place in the code.

    This file is that missing place. It gives the Python side the same kind of
    settings file CompLaB has, in the same layout, using the same tag names,
    so that if you already have a CompLaB.xml you can hand it straight to
    these scripts and they will read what they understand and TELL YOU, in
    writing, what they ignored.

THE THREE THINGS IT DOES

    1. Holds your settings_and_units numbers. Micrometres, mol/L, m^2/s, per second.
       Not lattice units, not pure numbers.

    2. Converts them to the six dimensionless groups the solver and the
       network actually consume, and prints the arithmetic so the conversion
       is checkable rather than trusted.

    3. Says what it cannot do. There are things CompLaB does that this solver
       does not, and a settings file that silently drops half of what you
       wrote is worse than one that refuses it. Every tag we do not support
       is reported by name.

THE UNITS, ONCE
    length          micrometres in the file (dx, characteristic_length)
    concentration   mol/L
    diffusion       m^2/s
    rates           per second, or per day if you set <rate_unit>per_day
    pressure        lattice units, exactly as CompLaB's <delta_P>
    tau             dimensionless, and it must exceed 0.5

THE ONE SUBTLETY WORTH READING
    Peclet and Damkohler are ratios, and a ratio needs a length. CompLaB
    divides by <characteristic_length>, the open pore width. The Python
    transport solver divides by the SAMPLE LENGTH, nx voxels, because that is
    the length over which its inlet-to-outlet problem is posed.

    These are not the same number and never were. A 64-voxel sample with a
    16-voxel pore width has a factor of four between them. Both conventions
    are printed side by side in the report, with the factor, so a Peclet
    quoted from one can be compared with a Peclet quoted from the other. Set
    <reference_length> to sample or to characteristic to choose which one the
    numbers in your file mean.

USAGE
    python settings_and_units.py --template my_settings.xml     write a commented file
    python settings_and_units.py --check my_settings.xml        read it back and report
    python settings_and_units.py --check CompLaB.xml            works on a real one too
    python settings_and_units.py --capabilities                 what this solver can do
    python settings_and_units.py --self-test                    the arithmetic checks

    python build_dataset_3d.py --settings my_settings.xml --out data.h5
"""

import argparse
import json
import math
import os
import sys
import re
import xml.etree.ElementTree as ET

import numpy as np

# --------------------------------------------------------------------------
# The default system: acetate oxidation coupled to sulfate reduction, which is
# the chemistry every worked example in the tutorial uses. These numbers are
# the ones that were previously written into prtlb_2d.py / prtlb_3d.py and complab_campaign.py, so a
# settings file that changes nothing reproduces exactly what the generators
# produced before this file existed. That is deliberate and it is tested.
# --------------------------------------------------------------------------
DEFAULT_SPECIES = [
    # Keyword arguments, not a tuple. A tuple here was wrong for exactly one
    # revision: it had eight entries against nine parameters, so the last one
    # landed in right_value and the biomass came out MOBILE. Nothing raised.
    # The self test below caught it because it asks whether the biomass is
    # attached rather than whether the file loads.
    dict(name="Ac", initial=0.0, d_pore=1.0e-9, d_biofilm=5.0e-10,
         left_type="Dirichlet", left_value=2.0e-3, right_type="Neumann"),
    dict(name="A", initial=0.0, d_pore=1.0e-9, d_biofilm=5.0e-10,
         left_type="Dirichlet", left_value=5.0e-3, right_type="Neumann"),
    # The product is fed at ZERO, not left free. That is a real boundary
    # condition and not a formality: the feed entering the sample is fresh and
    # carries no product, so the product concentration at the inlet face is
    # zero by construction. Leaving it free instead lets product diffuse back
    # upstream and pile up against the inlet with nowhere to go, and the
    # product then exceeds the donor feed concentration -- 1.17 times it, on a
    # 40 by 8 channel, which is impossible for something made one-for-one from
    # a donor that never exceeds 1.
    dict(name="P", initial=0.0, d_pore=1.0e-9, d_biofilm=5.0e-10,
         left_type="Dirichlet", left_value=0.0, right_type="Neumann"),
    dict(name="Bio", initial=2.0e-4, d_pore=0.0, d_biofilm=0.0,
         left_type="Neumann", left_value=0.0, right_type="Neumann",
         immobile=True),
]
# Every mobile chemical is given the SAME diffusion coefficient by default,
# because that is what the generators did before this file existed and a
# default that quietly changed the physics would make every dataset made
# before today incomparable with every dataset made after. Change them
# independently the moment you have real numbers: complab_campaign.py, which builds
# the CompLaB runs, uses 1.5e-9 for the product and 2.0e-10 for planktonic
# cells, and those are the better values if you have nothing else.

# The biomass is the fourth chemical, everywhere in this project.
ATTACHED_INDEX = 3
RATE_UNITS = {"per_second": 1.0, "per_day": 1.0 / 86400.0}
BC_TYPES = ("Dirichlet", "Neumann")


def _coerce(text, like):
    """Turn a command-line string into the type the setting already has.

    A value that is not a string arrived from an editable block inside a
    generator script rather than from the command line, and is already the
    type it should be.
    """
    if not isinstance(text, str):
        return text
    if isinstance(like, bool):
        return str(text).lower() in ("true", "1", "yes", "on")
    if isinstance(like, int) and not isinstance(like, bool):
        return int(float(text))
    if isinstance(like, float) or like is None:
        try:
            return float(text)
        except ValueError:
            return text
    return text


def _wrap(text, width=70, indent="    "):
    """Fold one long sentence onto the page. A report you have to scroll
    sideways to read is a report nobody reads."""
    words, line, out = text.split(), "", []
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    out.append(line)
    return ("\n" + indent).join(out)


# --------------------------------------------------------------------------
def _f(node, path, default=None, cast=float):
    """Read one tag, returning default when it is absent or empty.

    CompLaB files carry tags this solver has no use for and omit tags it
    would like; both have to be survivable, or a real CompLaB.xml could never
    be read at all.
    """
    if node is None:
        return default
    el = node.find(path)
    if el is None or el.text is None or not el.text.strip():
        return default
    t = el.text.strip()
    if cast is bool:
        return t.lower() in ("true", "1", "yes", "on")
    try:
        return cast(t)
    except ValueError:
        return default


def _fl(node, path, default=None):
    """Read one tag as a LIST of numbers, returning default when it is absent.

    Written for the sweep block, where a user may say either "anywhere between
    1 and 50" or "exactly 1, 10 and 50". Separators are whatever anyone would
    reasonably type: commas, spaces, semicolons or newlines.

        <peclet_values>1, 10, 50</peclet_values>
        <peclet_values>1 10 50</peclet_values>

    An entry that is not a number is an error rather than a shrug, because a
    typo here silently changes which experiment you ran.
    """
    if node is None:
        return default
    el = node.find(path)
    if el is None or el.text is None or not el.text.strip():
        return default
    parts = [t for t in re.split(r"[,;\s]+", el.text.strip()) if t]
    out = []
    for t in parts:
        try:
            out.append(float(t))
        except ValueError:
            raise SystemExit(
                "<%s> contains '%s', which is not a number. Write the values "
                "separated by commas or spaces, for example "
                "<%s>1, 10, 50</%s>" % (path, t, path, path))
    return out or default


class Species(object):
    """One transported chemical, described the way you measured it."""

    def __init__(self, name, initial=0.0, d_pore=1.0e-9, d_biofilm=None,
                 left_type="Neumann", left_value=0.0,
                 right_type="Neumann", right_value=0.0, immobile=False):
        self.name = str(name)
        self.initial = float(initial)
        self.d_pore = float(d_pore)
        self.d_biofilm = float(d_pore / 2.0 if d_biofilm is None else d_biofilm)
        self.left_type = str(left_type)
        self.left_value = float(left_value)
        self.right_type = str(right_type)
        self.right_value = float(right_value)
        self.immobile = bool(immobile)

    # The concentration this species is measured against when everything is
    # made dimensionless. A species fed at the inlet is scaled by its own feed
    # concentration, so it runs from 0 to 1. A species that is not fed --
    # a product, the biomass -- has no feed value to scale by, and is measured
    # in units of the DONOR feed, because that is what it is made from. That
    # is what makes a yield of 0.04 mean "four hundredths of a mole of cells
    # per mole of acetate" rather than an unreadable lattice number.
    def scale(self, donor_scale):
        if self.left_type.lower().startswith("dir") and self.left_value > 0:
            return self.left_value
        return donor_scale

    def copy(self, **over):
        """A copy with some fields replaced, leaving the original alone.

        Exists so that a system-level choice -- sessile or planktonic biomass
        -- can be applied on the way into the solver without editing the
        chemical table the user typed. See Settings.chemistry().
        """
        s = Species(self.name, self.initial, self.d_pore, self.d_biofilm,
                    self.left_type, self.left_value, self.right_type,
                    self.right_value, self.immobile)
        for k, v in over.items():
            setattr(s, k, v)
        return s

    def complaints(self):
        out = []
        if self.left_type not in BC_TYPES:
            out.append("%s: left_boundary_type is '%s'; it must be Dirichlet "
                       "or Neumann" % (self.name, self.left_type))
        if self.right_type not in BC_TYPES:
            out.append("%s: right_boundary_type is '%s'; it must be Dirichlet "
                       "or Neumann" % (self.name, self.right_type))
        if self.d_pore < 0:
            out.append("%s: a negative diffusion coefficient" % self.name)
        if self.initial < 0 or self.left_value < 0:
            out.append("%s: a negative concentration" % self.name)
        if self.immobile and self.d_pore > 0:
            out.append("%s: marked immobile but given a diffusion "
                       "coefficient of %g; one of the two is wrong"
                       % (self.name, self.d_pore))
        return out


class Settings(object):
    """Everything you can set, in the units you set it in."""

    def __init__(self):
        # ---- domain -------------------------------------------------------
        self.nx, self.ny, self.nz = 32, 32, 32
        self.dx = 1.0                 # voxel edge, micrometres
        self.unit = "um"
        self.characteristic_length = 16.0   # open pore width, micrometres
        self.reference_length = "sample"    # sample | characteristic
        self.pore_code, self.solid_code, self.wall_code = 2, 0, 1

        # ---- flow ---------------------------------------------------------
        self.tau = 0.8                # LBM relaxation time
        self.delta_P = 1.5e-4         # pressure drop driving the flow
        self.fluid_viscosity = 1.0e-6  # m^2/s; water at 20 C

        # WHAT SETS THE FLOW SPEED. This is the answer to "do I write a
        # pressure gradient, or does it work from Peclet?", and until now the
        # honest answer was "Peclet, and the pressure drop did nothing".
        #
        # "peclet"   the flow field is solved, then RESCALED so that the
        #            Peclet number is the one you asked for. The pressure drop
        #            still drives the lattice-Boltzmann solve, but its
        #            magnitude divides out. This is what a TRAINING SET wants,
        #            because Peclet is the thing being swept.
        #
        # "pressure" the flow field is solved with the pressure drop you gave
        #            and NOT rescaled. The Peclet number is then whatever that
        #            flow makes it, computed from the lattice speed, the
        #            relaxation time, the voxel size and the fluid viscosity,
        #            and reported. This is what you want when you are
        #            reproducing one named experiment at one measured flow
        #            rate.
        self.flow_driver = "peclet"
        self.peclet = 10.0            # used when flow_driver is "peclet"
        self.peclet_min = None        # set both to sweep a range instead
        self.peclet_max = None
        #  OR give the exact numbers you want, which wins over the range above.
        #  A range says "anywhere between 1 and 50"; a list says "these, and
        #  nothing else". Use the list when you are reproducing a published
        #  set of conditions, or when a reviewer asks for Pe = 1, 10 and 100.
        self.peclet_values = None
        #  POROSITY. Same choice: a range the structures are drawn from, or the
        #  exact porosities you want structures built at.
        self.porosity_min = None
        self.porosity_max = None
        self.porosity_values = None

        # ---- chemistry ----------------------------------------------------
        self.species = [Species(**s) for s in DEFAULT_SPECIES]
        self.rate_unit = "per_day"    # microbiology is quoted per day

        # ---- kinetics -----------------------------------------------------
        # R_bio = vmax * B * (Ac/(Ks_Ac+Ac)) * (A/(Ks_A+A))
        #
        # These are ordinary sulfate-reducer numbers, per day, and they are
        # here to be READ as well as used: the report prints the Damkohler
        # they imply, which for a 32 micrometre sample is very small indeed.
        # That is not a bug in the defaults, it is what the physics says --
        # see use_physical_kinetics below.
        self.vmax = 1.0               # per unit biomass, in rate_unit
        self.ks_donor = 0.0           # mol/L; 0 means first order in donor
        self.ks_acceptor = 7.5e-4     # mol/L
        self.yield_ = 0.04            # mol biomass per mol donor
        self.decay = 0.05             # first-order biomass decay, in rate_unit
        self.stoichiometry = 1.0      # mol acceptor per mol donor
        self.k_surf = 1.0             # the abiotic reaction, in rate_unit
        self.damkohler = 1.0          # used when the kinetics are not settings_and_units
        self.damkohler_min = None     # set both to sweep a range instead
        self.damkohler_max = None
        self.damkohler_values = None          # or the exact numbers you want
        self.decay_lattice = 0.006    # the pre-existing default, see below

        # ---- WHICH REACTIONS RUN, kept apart because they are different
        #      things and confusing them is a modelling error, not a typo.
        #
        # BIOTIC: microbially mediated. Needs microbes present, obeys Monod
        # kinetics in both the donor and the acceptor, grows biomass with a
        # yield and loses it to decay.
        #
        #     R_bio = vmax * B * Ac/(Ks_Ac + Ac) * A/(Ks_A + A)
        #     Ac + (stoichiometry) A  ->  P + (yield) B
        #
        # ABIOTIC: purely chemical. No microbes take any part in it. First
        # order in the product, which is what its Damkohler number is defined
        # against.
        #
        #     R_abio = k_surf * P     ->  P is consumed
        #
        # The abiotic reaction is OFF by default because it did not exist at
        # all until now, and turning it on by default would change every
        # dataset ever built here while the files still claimed to be
        # comparable.
        # WHICH REACTIONS RUN. One choice, made once, at the top of the page:
        #     "biotic only"    microbes and nothing else
        #     "abiotic only"   pure chemistry, no microbes anywhere
        #     "both"           the two together
        # biotic_enabled and abiotic_enabled follow from it and are NOT set
        # independently. Two controls for one thing is how they come to
        # disagree, and a run that says "abiotic OFF" in one place and carries
        # an abiotic Damkohler in another is worse than either.
        self.reaction_mode = "biotic only"
        self.abiotic_surface_only = False   # True: only on grain surfaces
        # WHICH CHEMICAL THE ABIOTIC REACTION CONSUMES.
        #
        # "product" is the CompLaB-like case: microbes make a product and a
        # purely chemical reaction then removes it. It is the right choice
        # whenever the biotic reaction is running.
        #
        # It is the WRONG choice with no biotic reaction, and silently so:
        # nothing makes the product, so the abiotic reaction has nothing to
        # act on and an abiotic-only run produces an entirely zero field. That
        # is what "P max 0" means on a picture from such a run.
        #
        # "donor" makes the abiotic reaction consume the fed chemical
        # directly, which is what an abiotic-only experiment actually is: a
        # solute injected at the inlet and destroyed as it travels.
        #
        # "auto", the default, picks product when the biotic reaction runs and
        # donor when it does not.
        self.abiotic_reactant = "auto"
        # WHAT THE ABIOTIC REACTION TURNS ITS REACTANT INTO.
        #
        #   "product"  Ac -> P. The reactant is consumed and the product
        #              appears, one mole for one. This is what a mineral
        #              dissolution or a surface-catalysed transformation
        #              does, and it is what makes an ABIOTIC-ONLY run a
        #              reaction rather than a disappearance: the donor falls,
        #              the product rises, and the two sum to the feed.
        #   "none"     the reactant is consumed and nothing tracked appears.
        #              Right for a terminal loss -- a solute that precipitates
        #              out, or degrades to something this run does not follow.
        #
        # It used to be "none" with no way to change it, which is why an
        # abiotic-only run came out with an empty product field: there was no
        # abiotic reaction that PRODUCED anything, anywhere in the code.
        self.abiotic_product = "product"
        self.damkohler_abiotic = 1.0
        self.damkohler_abiotic_values = None   # or the exact numbers
        self.damkohler_abiotic_min = None
        self.damkohler_abiotic_max = None

        # SESSILE OR PLANKTONIC. Sessile (attached, biofilm) biomass reacts
        # where it is and never moves. Planktonic biomass is a dissolved
        # chemical: it is advected and diffused like any other, and in a
        # through-flowing sample most of it leaves. Measured on a 60 by 32
        # domain, mean biomass fell from 0.100 to 0.019 with transport on and
        # to 0.071 with it off, so about three quarters of the loss was
        # washout rather than decay.
        self.biomass_mode = "sessile"

        # WHERE THE ATTACHED BIOMASS LIVES.
        #
        # CompLaB gives each attached microbe pool its own MATERIAL NUMBER, so
        # biofilm occupies named voxels in the geometry file, distinct from
        # pore and from solid, and the biomass is seeded only there. It then
        # spreads by a cellular automaton and can turn pore into biofilm as it
        # grows.
        #
        # Here you get the first half of that and not the second. Give a
        # biofilm code and any voxel carrying it is treated as pore that
        # happens to hold biofilm: fluid flows through it, chemicals diffuse
        # through it at their in_biofilm coefficient rather than their pore
        # one, and the starting biomass is placed there and nowhere else. What
        # you do NOT get is growth changing the geometry -- no voxel ever
        # becomes biofilm or solid during a run, and there is no cellular
        # automaton spreading it. The pore space is fixed for the whole
        # simulation.
        #
        # None means what it always meant: no biofilm voxels, and the biomass
        # starts uniformly through the pore space.
        self.biofilm_code = None
        # How thick a biofilm lining to grow on the grain surfaces when the
        # generator builds a rock. 0 leaves the geometry bare. 1 coats every
        # pore voxel that touches solid; 2 coats that layer and the one behind
        # it. Ignored unless biofilm_code is set, because without a code there
        # is nothing to mark the coated voxels with.
        self.biofilm_layers = 1

        # Does the biotic rate follow the biomass that is actually there? By
        # default no, because B0 is already inside the Damkohler number and
        # letting B vary as well counts it twice. Turn it on when the biomass
        # changes a lot over a run and you want the feedback.
        self.biomass_coupled = False

        # WHICH ROUTE THE DAMKOHLER NUMBERS COME FROM.
        #
        # False, the default: the generator sweeps Damkohler directly, over
        # the range in <sweep>, exactly as it did before this file existed.
        # The concentrations, the diffusivity ratios, the stoichiometry, the
        # half-saturation constants and the yield below are still read and
        # still used -- they are ratios, and ratios behave.
        #
        # True: the Damkohler numbers are computed from vmax, k_surf and decay
        # instead. Turn it on when you are modelling ONE named system rather
        # than building a training set that has to span the space.
        #
        # Worth knowing before you turn it on. A training set wants Damkohler
        # from about 0.1 to 10, because that is where transport and reaction
        # are comparable and the problem is interesting. A real pore-scale
        # sample of 32 micrometres with a diffusion coefficient of 1e-9 m^2/s
        # is crossed by diffusion in about one second, and no ordinary
        # microbial rate competes with that -- so the honest Damkohler of the
        # defaults below is of order 1e-5, and the reaction is invisible. Real
        # columns reach Damkohler 1 by being centimetres long, not by reacting
        # faster. The report prints your number and says so.
        self.use_physical_kinetics = False

        # ---- iteration ----------------------------------------------------
        # The CAP on flow-solver iterations, not the number run: the solve
        # stops when the field stops changing, which is usually far sooner.
        # 400 was the old fixed count and was 9.8% short in 3D.
        self.ns_max_iT = 30000        # flow-solver iteration cap
        self.ns_converge = 1e-6
        self.ade_max_iT = 100000      # transport steps, the hard cap
        self.ade_converge = 2e-3
        self.n_snapshots = 8

        # ---- bookkeeping --------------------------------------------------
        # Which builder is going to use this. Set by settings_from_args, and
        # only so that a two-dimensional run is not warned that it is two
        # dimensional.
        self.dimension = None
        self.source = "defaults"
        self.ignored = []             # tags read but not supported
        self.notes = []

    # ------------------------------------------------------------- resizing
    # How many chemicals there are, as a setting you can just assign to. It
    # exists so the window's plus and minus buttons have something to send:
    # "n_chemicals=5" resizes the list, and the per-chemical settings that
    # follow it then have somewhere to land. Set it before them, which is what
    # the window does.
    @property
    def n_chemicals(self):
        return len(self.species)

    @n_chemicals.setter
    def n_chemicals(self, n):
        n = max(1, int(n))
        while len(self.species) > n:
            self.species.pop()
        while len(self.species) < n:
            i = len(self.species)
            self.species.append(Species(
                name="Tracer%d" % (i - 3) if i > 3 else "C%d" % i,
                d_pore=1.0e-9, d_biofilm=5.0e-10,
                left_type="Dirichlet", left_value=1.0e-3,
                right_type="Neumann"))

    # ---------------------------------------------------------------- names
    @property
    def names(self):
        return [s.name for s in self.species]

    def index(self, name):
        try:
            return self.names.index(name)
        except ValueError:
            return -1

    @property
    def donor(self):
        return self.species[0]

    @property
    def acceptor(self):
        return self.species[1] if len(self.species) > 1 else None

    # EXACT comparison, not a substring test. "abiotic only" CONTAINS
    # "biotic only", so asking whether the mode contains "biotic only" says
    # yes for the abiotic-only case and switches the wrong reaction off. The
    # first version of this did exactly that and reported both reactions off.
    @property
    def abiotic_index(self):
        """Which chemical the abiotic reaction consumes: 0 donor, 2 product."""
        want = str(self.abiotic_reactant).lower()
        if want.startswith("don"):
            return 0
        if want.startswith("prod"):
            return 2
        return 2 if self.biotic_enabled else 0

    @property
    def abiotic_product_index(self):
        """Which chemical the abiotic reaction MAKES, or None for a pure loss.

        None whenever there is nothing sensible to make: the setting says so,
        the run is too short to have a third chemical, or the reactant IS the
        product, in which case producing it would be a reaction that consumes
        a thing and creates the same thing.
        """
        if str(self.abiotic_product).lower() != "product":
            return None
        if len(self.species) < 3 or self.abiotic_index == 2:
            return None
        return 2

    @property
    def _mode(self):
        return " ".join(str(self.reaction_mode).lower().split())

    @property
    def biotic_enabled(self):
        return self._mode in ("biotic only", "both")

    @property
    def abiotic_enabled(self):
        return self._mode in ("abiotic only", "both")

    @property
    def immobile_flags(self):
        return [s.immobile for s in self.species]

    # ------------------------------------------------------------- geometry
    @property
    def dx_m(self):
        """Voxel edge in metres. The file is in micrometres because that is
        what a pore-scale image is measured in; every formula below is in SI."""
        f = {"um": 1e-6, "µm": 1e-6, "micron": 1e-6, "mm": 1e-3, "m": 1.0}
        return self.dx * f.get(self.unit, 1e-6)

    @property
    def L_sample_m(self):
        """Inlet to outlet, in metres. What the Python solver divides by."""
        return self.nx * self.dx_m

    @property
    def unit_m(self):
        """One unit of the file's length unit, in metres."""
        f = {"um": 1e-6, "µm": 1e-6, "micron": 1e-6, "mm": 1e-3, "m": 1.0}
        return f.get(self.unit, 1e-6)

    @property
    def L_char_m(self):
        """The open pore width, in metres. What CompLaB divides by.

        MULTIPLIED BY THE UNIT, NOT BY dx_m. characteristic_length is written
        in the file's own length unit, exactly as dx is -- the template says
        "lengths: micrometres (dx, characteristic_length)" -- so converting it
        needs the unit, and this multiplied a length by a length instead. With
        the defaults, 16 um and dx = 1 um, it returned 16 * 1e-6 * 1e-6 =
        1.6e-11 m: a pore sixteen picometres across, a tenth the width of a
        hydrogen atom. Every group taken against the characteristic length was
        then wrong by dx in metres, a factor of a million at the default, and
        the diffusive Damkohler number, which carries L squared, by 1e12.
        """
        return self.characteristic_length * self.unit_m

    @property
    def L_ref_m(self):
        return (self.L_char_m if self.reference_length.startswith("char")
                else self.L_sample_m)

    @property
    def l_ref_voxels(self):
        """The reference length in VOXELS, which is the unit the solver works
        in. Peclet is u*L/D and both Damkohler numbers carry an L, so this is
        the single number that decides what all three of them mean.

        NONE FOR THE SAMPLE CONVENTION, and that is deliberate rather than
        lazy. "The sample length" means the length of the domain the solver
        was actually handed, which is not always the nx in this file: the
        generators build domains of several shapes from one settings file, and
        a published structure comes with its own. Sending this file's nx would
        pin every one of them to the number that happened to be typed here --
        32 by default, against domains that are usually 64 or 148. None tells
        the solver to use the domain in front of it, which is what the words
        mean. Only the characteristic length is a property of the ROCK rather
        than of the box drawn around it, so only it is a fixed number.
        """
        if self.reference_length.startswith("char"):
            return max(float(self.characteristic_length) / max(self.dx, 1e-30),
                       1.0)
        return None

    @property
    def rate_factor(self):
        return RATE_UNITS.get(self.rate_unit, 1.0)

    @property
    def viscosity(self):
        """Lattice kinematic viscosity. nu = (tau - 1/2)/3, and tau below 0.5
        makes it negative, which is why tau is checked rather than trusted."""
        return (self.tau - 0.5) / 3.0

    # =====================================================================
    # reading
    # =====================================================================
    @classmethod
    def from_file(cls, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".json", ".jsn"):
            return cls.from_json(path)
        return cls.from_xml(path)

    @classmethod
    def from_xml(cls, path):
        """Read a settings file in CompLaB's layout.

        Written to survive a REAL CompLaB.xml, not only one of our templates.
        A real one has 95 substrates, twelve microbe pools, an equilibrium
        block and a precipitation block. Everything this solver cannot honour
        is recorded in self.ignored and printed, so nothing is dropped in
        silence.
        """
        s = cls()
        s.source = os.path.abspath(path)
        root = ET.parse(path).getroot()

        lb = root.find("LB_numerics")
        dom = lb.find("domain") if lb is not None else root.find(".//domain")
        if dom is not None:
            s.nx = int(_f(dom, "nx", s.nx, int))
            s.ny = int(_f(dom, "ny", s.ny, int))
            s.nz = int(_f(dom, "nz", s.nz, int))
            s.dx = _f(dom, "dx", s.dx)
            s.unit = _f(dom, "unit", s.unit, str)
            s.characteristic_length = _f(dom, "characteristic_length",
                                         s.characteristic_length)
            mn = dom.find("material_numbers")
            if mn is not None:
                s.pore_code = int(_f(mn, "pore", s.pore_code, int))
                s.solid_code = int(_f(mn, "solid", s.solid_code, int))
                s.wall_code = int(_f(mn, "bounce_back", s.wall_code, int))
                mic = [c.tag for c in mn if c.tag.startswith("microbe")]
                if len(mic) > 1:
                    s.ignored.append(
                        "material_numbers holds %d microbe codes (%s). This "
                        "solver carries ONE biomass pool, so only the first is "
                        "used and the rest are dropped."
                        % (len(mic), ", ".join(sorted(mic)[:4]) + " ..."))

        host = lb if lb is not None else root
        s.delta_P = _f(host, "delta_P", s.delta_P)
        s.peclet = _f(host, "Peclet", s.peclet)
        s.tau = _f(host, "tau", s.tau)
        # READ FROM WHEREVER IT IS. The template writes this inside <domain>,
        # nested under <LB_numerics>, because it is a property of the geometry,
        # and this looked for it as a direct child of the lattice-Boltzmann
        # block. It was therefore never found: a file could say
        # "characteristic" and load back as "sample", which is a factor of two
        # on Peclet and on both Damkohler numbers. Searched tree-wide now, so
        # it does not matter which block a hand-written file puts it in.
        _rl = root.find(".//reference_length")
        s.reference_length = _f(host, "reference_length",
                                s.reference_length, str)
        if _rl is not None and (_rl.text or "").strip():
            s.reference_length = (_rl.text or "").strip()
        s.flow_driver = _f(host, "flow_driver", s.flow_driver, str)
        s.fluid_viscosity = _f(host, "fluid_viscosity", s.fluid_viscosity)

        it = host.find("iteration") if host is not None else None
        if it is not None:
            s.ns_max_iT = int(_f(it, "ns_max_iT1",
                                 _f(it, "ns_max_iT", s.ns_max_iT, int), int))
            s.ns_converge = _f(it, "ns_converge_iT1",
                               _f(it, "ns_converge_iT", s.ns_converge))
            s.ade_max_iT = int(_f(it, "ade_max_iT", s.ade_max_iT, int))
            s.ade_converge = _f(it, "ade_converge_iT", s.ade_converge)
            s.n_snapshots = int(_f(it, "n_snapshots", s.n_snapshots, int))

        # ---- the chemicals ------------------------------------------------
        chem = root.find("chemistry")
        if chem is not None:
            s.rate_unit = _f(chem, "rate_unit", s.rate_unit, str)
            n = int(_f(chem, "number_of_substrates", 0, int))
            found = []
            i = 0
            while True:
                sub = chem.find("substrate%d" % i)
                if sub is None:
                    break
                dif = sub.find("substrate_diffusion_coefficients")
                dp = _f(dif, "in_pore", 1.0e-9)
                found.append(Species(
                    name=_f(sub, "name_of_substrates", "C%d" % i, str),
                    initial=_f(sub, "initial_concentration", 0.0),
                    d_pore=dp,
                    d_biofilm=_f(dif, "in_biofilm", dp / 2.0),
                    left_type=_f(sub, "left_boundary_type", "Neumann", str),
                    left_value=_f(sub, "left_boundary_condition", 0.0),
                    right_type=_f(sub, "right_boundary_type", "Neumann", str),
                    right_value=_f(sub, "right_boundary_condition", 0.0),
                    immobile=_f(sub, "immobile", False, bool)))
                i += 1
            if found:
                s.species = found
            if n and n != len(found):
                s.ignored.append(
                    "number_of_substrates says %d but %d substrate blocks were "
                    "found; the blocks win." % (n, len(found)))
            if len(found) > 4:
                s.notes.append(
                    "%d chemicals are described. The reaction network is donor "
                    "+ acceptor -> product + biomass, so the first four react; "
                    "the other %d are transported as CONSERVATIVE TRACERS -- "
                    "fed, advected and diffused with their own coefficient, "
                    "but taking no part in any reaction. For a fifth REACTING "
                    "chemical, use CompLaB."
                    % (len(found), len(found) - 4))

        # ---- kinetics -----------------------------------------------------
        kin = root.find("kinetics")
        if kin is not None:
            s.vmax = _f(kin, "vmax", s.vmax)
            s.ks_donor = _f(kin, "ks_donor", s.ks_donor)
            s.ks_acceptor = _f(kin, "ks_acceptor", s.ks_acceptor)
            s.yield_ = _f(kin, "yield", s.yield_)
            s.decay = _f(kin, "decay", s.decay)
            s.stoichiometry = _f(kin, "stoichiometry", s.stoichiometry)
            s.k_surf = _f(kin, "k_surf", s.k_surf)
            s.rate_unit = _f(kin, "rate_unit", s.rate_unit, str)
            s.damkohler = _f(kin, "damkohler", s.damkohler)
            s.decay_lattice = _f(kin, "decay_lattice", s.decay_lattice)
            s.use_physical_kinetics = _f(kin, "use_physical_kinetics",
                                         s.use_physical_kinetics, bool)
            s.reaction_mode = _f(kin, "reaction_mode", s.reaction_mode, str)
            s.abiotic_surface_only = _f(kin, "abiotic_surface_only",
                                        s.abiotic_surface_only, bool)
            s.abiotic_reactant = _f(kin, "abiotic_reactant",
                                    s.abiotic_reactant, str)
            s.abiotic_product = _f(kin, "abiotic_product",
                                   s.abiotic_product, str)
            s.damkohler_abiotic = _f(kin, "damkohler_abiotic",
                                     s.damkohler_abiotic)
            s.biomass_mode = _f(kin, "biomass_mode", s.biomass_mode, str)
            s.biofilm_code = _f(kin, "biofilm_code", s.biofilm_code, int)
            s.biofilm_layers = _f(kin, "biofilm_layers", s.biofilm_layers, int)
            s.biomass_coupled = _f(kin, "biomass_coupled",
                                   s.biomass_coupled, bool)

        sw = root.find("sweep")
        if sw is not None:
            s.peclet_min = _f(sw, "peclet_min", None)
            s.peclet_max = _f(sw, "peclet_max", None)
            s.damkohler_min = _f(sw, "damkohler_min", None)
            s.damkohler_max = _f(sw, "damkohler_max", None)
            s.damkohler_abiotic_min = _f(sw, "damkohler_abiotic_min", None)
            s.damkohler_abiotic_max = _f(sw, "damkohler_abiotic_max", None)
            s.porosity_min = _f(sw, "porosity_min", None)
            s.porosity_max = _f(sw, "porosity_max", None)
            # The explicit lists. Each one wins over its own min/max pair, and
            # over the matching command-line range, but a command-line LIST
            # still wins over the file, so a single run can be redirected
            # without editing the settings that describe it.
            s.peclet_values = _fl(sw, "peclet_values", None)
            s.damkohler_values = _fl(sw, "damkohler_values", None)
            s.damkohler_abiotic_values = _fl(sw, "damkohler_abiotic_values", None)
            s.porosity_values = _fl(sw, "porosity_values", None)

        # ---- what we read but cannot honour --------------------------------
        for tag, why in (
            ("equilibrium", "aqueous speciation is not solved here; every "
                            "chemical is transported as written"),
            ("precipitation", "no voxel is ever converted from pore to solid; "
                              "the pore structure is fixed for the whole run"),
            ("microbiology", "one biomass pool, attached, no growth-driven "
                             "change of the pore space"),
            ("surface_complexation", "no surface complexation model"),
            ("solid_substrate", "no solid-phase substrate bookkeeping"),
        ):
            if root.find(".//" + tag) is not None:
                s.ignored.append("<%s> was present and is not supported: %s"
                                 % (tag, why))

        sim = root.find("simulation_mode")
        if sim is not None:
            bio = _f(sim, "biotic_mode", None, bool)
            ab = _f(sim, "enable_abiotic_kinetics", None, bool)
            # CompLaB's MASTER switch. <enable_kinetics>false</enable_kinetics>
            # turns off all reaction and leaves a conservative tracer, and it
            # was not read at all -- so a CompLaB file that says "no chemistry"
            # produced a run full of chemistry. It overrides the two below,
            # because that is what a master switch is.
            master = _f(sim, "enable_kinetics", None, bool)
            if master is False:
                s.reaction_mode = "biotic only"
                s.damkohler = 0.0
                s.damkohler_abiotic = 0.0
                s.vmax = 0.0
                s.k_surf = 0.0
                s.notes.append("<enable_kinetics> is false, so every reaction "
                               "rate is set to zero and the chemicals are "
                               "transported as conservative tracers.")
            elif bio is not None or ab is not None:
                bio = True if bio is None else bio
                ab = True if ab is None else ab
                s.reaction_mode = ("both" if bio and ab else
                                   "biotic only" if bio else
                                   "abiotic only" if ab else "biotic only")
                s.notes.append("the reaction mode was taken from "
                               "<simulation_mode>: %s" % s.reaction_mode)
        return s

    @classmethod
    def from_block(cls, scalars=None, chemicals=None, where="an editable block"):
        """Build Settings from a plain dictionary written in a script.

        This is the other way in, and for most people it is the easier one.
        Instead of writing a settings file, you open the generator you are
        about to run, edit the block of numbers at the top of it, and run it.
        The block is a partial description: whatever you leave out keeps its
        default, so it stays short and stays readable.
        """
        s = cls()
        s.source = where
        if chemicals:
            s.species = [Species(**d) for d in chemicals]
        for k, v in (scalars or {}).items():
            s.apply_override(k, v)
        s.notes = [n for n in s.notes if "on the command line" not in n]
        return s

    @classmethod
    def from_json(cls, path):
        s = cls()
        s.source = os.path.abspath(path)
        with open(path) as f:
            d = json.load(f)
        sp = d.pop("species", None)
        for k, v in d.items():
            key = "yield_" if k == "yield" else k
            if hasattr(s, key):
                setattr(s, key, v)
        if sp:
            s.species = [Species(**one) for one in sp]
        return s

    # =====================================================================
    # overriding one value without editing the file
    # =====================================================================
    def apply_override(self, key, value):
        """Set one setting from a KEY=VALUE string.

        'tau' for a top-level setting, 'species.A.d_pore' for one chemical's.
        An unknown key is an error rather than a shrug: a typo that is quietly
        ignored looks exactly like a setting that had no effect, and you find
        out six hours later from a figure.
        """
        if key.startswith("species."):
            parts = key.split(".")
            if len(parts) != 3:
                raise SystemExit(
                    "'%s' is not a chemical setting. Write it as "
                    "species.NAME.FIELD, for example species.A.d_pore" % key)
            _, name, field = parts
            # By name, or by position. Position matters because the reaction
            # network is fixed by ORDER -- 0 is the donor, 1 the acceptor, 2
            # the product, 3 the biomass -- and a settings file read from
            # somebody else's CompLaB.xml will not call them Ac and A. A window
            # that referred to them by name would stop working the moment you
            # loaded such a file.
            if name.isdigit():
                i = int(name)
                sp = [self.species[i]] if i < len(self.species) else []
                if not sp:
                    raise SystemExit(
                        "this file describes %d chemicals, so there is no "
                        "chemical %d." % (len(self.species), i))
            else:
                sp = [x for x in self.species if x.name == name]
            if not sp:
                raise SystemExit("no chemical is called '%s'. This file has: %s"
                                 % (name, ", ".join(self.names)))
            if not hasattr(sp[0], field):
                raise SystemExit(
                    "a chemical has no setting called '%s'. It has: name, "
                    "initial, d_pore, d_biofilm, left_type, left_value, "
                    "right_type, right_value, immobile" % field)
            cur = getattr(sp[0], field)
            setattr(sp[0], field, _coerce(value, cur))
            self.notes.append("%s was set to %s on the command line"
                              % (key, value))
            return

        # The settings file writes <pore>, the attribute is pore_code, and a
        # user typing either should be right. Same for the tag names CompLaB
        # uses that differ from ours by a suffix.
        ALIASES = {"yield": "yield_", "pore": "pore_code",
                   "solid": "solid_code", "bounce_back": "wall_code",
                   "wall": "wall_code", "Peclet": "peclet",
                   "n_species": "n_chemicals", "number_of_substrates":
                   "n_chemicals"}
        attr = ALIASES.get(key, key)
        if attr == "n_chemicals":
            self.n_chemicals = _coerce(value, 4)
            self.notes.append("the number of chemicals was set to %s on the "
                              "command line" % value)
            return
        if not hasattr(self, attr) or attr in ("species", "ignored", "notes",
                                               "dimension", "source"):
            raise SystemExit(
                "there is no setting called '%s'. The ones you can set are:\n  "
                "%s\nand any chemical's, as species.NAME.FIELD."
                % (key, "\n  ".join(sorted(
                    ["n_chemicals"] + [
                        k.rstrip("_") for k, v in vars(self).items()
                        if k not in ("species", "ignored", "notes", "source",
                                     "dimension")]))))
        setattr(self, attr, _coerce(value, getattr(self, attr)))
        self.notes.append("%s was set to %s on the command line" % (key, value))

    # =====================================================================
    # checking
    # =====================================================================
    def complaints(self):
        """Everything wrong with this file, in the file's own vocabulary.

        Returned rather than raised so the caller can print them all at once.
        A settings file with four mistakes in it should show four, not the
        first one, four times over.
        """
        bad = []
        if self.tau <= 0.5:
            bad.append("tau is %g. Below 0.5 the viscosity (tau-0.5)/3 is "
                       "negative and the flow solver cannot run. Use 0.6 to "
                       "1.0." % self.tau)
        if self.tau > 2.0:
            bad.append("tau is %g. Above about 2 the lattice Boltzmann "
                       "accuracy degrades badly. Use 0.6 to 1.0." % self.tau)
        if min(self.nx, self.ny, self.nz) < 1:
            bad.append("the grid has a zero or negative side: %d x %d x %d"
                       % (self.nx, self.ny, self.nz))
        if self.dx <= 0:
            bad.append("dx is %g; a voxel has to have a size." % self.dx)
        if self.characteristic_length <= 0:
            bad.append("characteristic_length is %g; it is the open pore "
                       "width and must be positive." % self.characteristic_length)
        if self.characteristic_length > self.nx * self.dx:
            bad.append("characteristic_length (%g %s) is larger than the whole "
                       "sample (%g %s). One of the two is in the wrong units."
                       % (self.characteristic_length, self.unit,
                          self.nx * self.dx, self.unit))
        if not self.species:
            bad.append("no chemicals are described.")
        if self.species and self.donor.d_pore <= 0:
            bad.append("the first chemical, %s, has a diffusion coefficient of "
                       "%g. It is the reference for every dimensionless group, "
                       "so it cannot be zero or immobile."
                       % (self.donor.name, self.donor.d_pore))
        if self.species and not (
                self.donor.left_type.lower().startswith("dir")
                and self.donor.left_value > 0):
            bad.append("the first chemical, %s, is not fed at the inlet. It is "
                       "the electron donor and the concentration scale for "
                       "everything else; give it a Dirichlet left boundary "
                       "with a positive value." % self.donor.name)
        if str(self.flow_driver).lower() not in ("peclet", "pressure"):
            bad.append("flow_driver is '%s'; it must be peclet or pressure."
                       % self.flow_driver)
        if self._mode not in ("biotic only", "abiotic only", "both"):
            bad.append("reaction_mode is '%s'; it must be 'biotic only', "
                       "'abiotic only' or 'both'." % self.reaction_mode)
        if self.biofilm_code is not None and self.biofilm_code in (
                self.pore_code, self.solid_code, self.wall_code):
            bad.append("the biofilm code %d is the same as the pore, solid or "
                       "wall code. Every material needs its own number."
                       % self.biofilm_code)
        # A REACTION THAT CANNOT HAPPEN IS NOT A SETTING, IT IS A MISTAKE.
        #
        # The abiotic reaction consumes ONE named chemical. Ask it to consume
        # the product while the biotic reaction is switched off and there is
        # nothing to consume: the product is MADE by the biotic reaction, so
        # with no microbes it is zero in every voxel for the whole run and the
        # abiotic rate is a rate multiplied by zero.
        #
        # This is not a small error. Measured on a 40 by 16 channel, an
        # abiotic-only run with the reactant set to the product: the product
        # was zero everywhere, the donor and the acceptor came out as the same
        # inert tracer to six figures, and sweeping the abiotic Damkohler from
        # 0.1 to 10 changed the field only through the time step -- so a whole
        # sweep of "different" runs was one run recorded under fifty labels. A
        # network trained on it learns that Da_abio does nothing, which is
        # true of the data and false of the world.
        #
        # Refused rather than warned, because there is no reading of it that
        # gives you what you asked for. Set abiotic_reactant to donor: the
        # donor IS fed, so the reaction then has something to consume and the
        # Damkohler number means what it says.
        if (self.abiotic_enabled and not self.biotic_enabled
                and self.abiotic_index == 2):
            bad.append(
                "the abiotic reaction is set to consume the PRODUCT (%s), but "
                "the biotic reaction is off and the product is what the biotic "
                "reaction makes -- so nothing ever produces it, the abiotic "
                "rate acts on a field that is zero everywhere, and every run "
                "in the sweep comes out identical whatever the abiotic "
                "Damkohler number is. Set abiotic_reactant to 'donor', which "
                "is fed at the inlet and so can actually be consumed, or turn "
                "the biotic reaction back on."
                % (self.names[2] if len(self.names) > 2 else "P"))
        if str(self.abiotic_product).lower() not in ("product", "none"):
            bad.append("abiotic_product is '%s'; it must be 'product' (the "
                       "reactant becomes the product) or 'none' (the reactant "
                       "is simply lost)." % self.abiotic_product)
        if str(self.biomass_mode).lower() not in ("sessile", "planktonic"):
            bad.append("biomass_mode is '%s'; it must be sessile or "
                       "planktonic." % self.biomass_mode)
        if self.fluid_viscosity <= 0:
            bad.append("fluid_viscosity is %g; water is 1e-6 m2/s."
                       % self.fluid_viscosity)
        if self.rate_unit not in RATE_UNITS:
            bad.append("rate_unit is '%s'; it must be per_second or per_day."
                       % self.rate_unit)
        if self.reference_length not in ("sample", "characteristic"):
            bad.append("reference_length is '%s'; it must be sample or "
                       "characteristic." % self.reference_length)
        if self.yield_ < 0 or self.decay < 0 or self.vmax < 0:
            bad.append("a negative yield, decay or vmax.")
        if self.n_snapshots < 2:
            bad.append("n_snapshots is %d; two is the minimum, one at the "
                       "start and one at the end." % self.n_snapshots)
        for sp in self.species:
            bad.extend(sp.complaints())
        seen = set()
        for nm in self.names:
            if nm in seen:
                bad.append("two chemicals are both called '%s'." % nm)
            seen.add(nm)
        return bad

    def warnings(self):
        """Things that are legal, run, and are almost certainly not what you
        meant. Kept apart from complaints() because these do not stop a run:
        a file can be arithmetically fine and still describe a system in which
        nothing happens, and that is worth being told before you spend six
        hours on it rather than after."""
        out = []
        if len(self.species) < 4:
            out.append("only %d chemicals are described. The reaction network "
                       "is donor + acceptor -> product + biomass; with fewer "
                       "than four, the missing ones simply do not exist and "
                       "the reaction terms that need them are dropped."
                       % len(self.species))
            return out
        g = self.groups()
        nm = self.names
        if not self.biotic_enabled and len(self.species) > 3:
            out.append("this run has no microbes, so the fourth chemical, %s, "
                       "is a biomass with nothing to do. It is held at zero "
                       "rather than left to decay, so its field and its "
                       "picture will be blank. Take it off with the minus "
                       "button under the chemicals, or switch the reaction "
                       "mode back to biotic." % self.names[3])
        if self.biotic_enabled and g["b0_norm"] <= 0:
            out.append("the fourth chemical, %s, is the biomass and starts at "
                       "zero everywhere. The biotic rate is proportional to "
                       "biomass, so with none present nothing reacts, ever, "
                       "anywhere. Give it a non-zero initial_concentration."
                       % nm[3])
        # WHETHER THE BIOMASS MOVES IS DECIDED BY biomass_mode, not by the
        # immobile box on the fourth chemical -- one system-level setting
        # rather than a flag somebody has to remember. Two things are worth
        # saying about it, and neither was reachable before: chemistry() used
        # to WRITE its decision back into the chemical table before this ran,
        # so the table always already agreed with it and the warning could
        # never fire.
        sessile = str(self.biomass_mode).lower().startswith("sess")
        if not sessile:
            out.append("biomass_mode is planktonic, so the biomass is "
                       "advected and will largely wash out of the sample. "
                       "Measured on a 60 by 32 domain, the mean fell from "
                       "0.100 to 0.019, of which about three quarters was "
                       "washout rather than decay. Set biomass_mode to "
                       "sessile for attached cells, which is what a biofilm "
                       "is.")
        if self.species[3].immobile != sessile:
            out.append("the immobile box on %s says %s and biomass_mode says "
                       "%s. biomass_mode wins -- it is the system-level "
                       "setting -- so the box is being ignored. Set them to "
                       "agree so the file says what the run does."
                       % (nm[3], "immobile" if self.species[3].immobile
                          else "mobile",
                          "sessile" if sessile else "planktonic"))
        # A CHEMICAL NOTHING FEEDS AND NOTHING MAKES IS A BLANK FIELD.
        #
        # Only the biotic reaction PRODUCES anything: the product and the
        # biomass. The abiotic reaction only consumes. So with the biotic
        # reaction switched off, a chemical that is not fed at the inlet and
        # is not present at the start is zero in every voxel at every time --
        # it is stored, it is trained on, it is plotted, and it is nothing.
        #
        # This is the question that came out of a real run: "why in only
        # abiotic do I still have a Bio field?". The answer is that you asked
        # for four chemicals, and two of them cannot exist in that run. Said
        # here, before three minutes are spent producing them.
        for j_b in range(2, len(self.species)):
            sp_b = self.species[j_b]
            fed_b = (sp_b.left_type.lower().startswith("dir")
                     and sp_b.left_value > 0)
            made_abio = (self.abiotic_enabled
                         and self.abiotic_product_index == j_b)
            if self.biotic_enabled or fed_b or sp_b.initial > 0 or made_abio:
                continue
            out.append(
                "%s is not fed at the inlet, is not present at the start, and "
                "is not made by any reaction that is switched on -- only the "
                "biotic reaction produces anything, and it is off. Its field "
                "will be exactly zero everywhere, at every time, and its "
                "picture will be blank. Set n_chemicals to %d, or turn the "
                "biotic reaction on." % (sp_b.name, j_b))

        # TWO CHEMICALS THAT OBEY THE SAME EQUATION ARE ONE CHEMICAL.
        #
        # A chemical takes part in the chemistry only if a reaction touches
        # it. With the biotic reaction off, the ONLY one that does is whatever
        # the abiotic reaction consumes; everything else is a conservative
        # tracer. Two tracers with the same feed, the same starting value and
        # the same diffusivity then obey the identical equation and hold the
        # identical field -- so the dataset stores one answer twice and the
        # error the network reports on the second is the error on the first.
        # check_physics catches it after the fact; this says so beforehand,
        # when it is still cheap to fix.
        reacting = set()
        if self.biotic_enabled:
            reacting = set(range(4))
        if self.abiotic_enabled:
            reacting.add(self.abiotic_index)
            if self.abiotic_product_index is not None:
                reacting.add(self.abiotic_product_index)
        inert = [i for i in range(len(self.species)) if i not in reacting]
        # COMPARED IN THE UNITS THE SOLVER WORKS IN, not the ones you typed.
        # Every chemical is scaled by its OWN feed concentration, so a donor
        # fed at 0.002 mol/L and an acceptor fed at 0.005 both arrive at the
        # solver as a feed of exactly 1.0. Comparing the physical numbers says
        # they are different chemicals; comparing what is actually solved says
        # they are the same one. Your own run is the example: Ac and A came
        # out with correlation 1.00000, and a check on the raw values would
        # have passed it.
        _c = self.chemistry()
        for a_i in range(len(inert)):
            for b_i in range(a_i + 1, len(inert)):
                ia, ib = inert[a_i], inert[b_i]
                if ia >= len(_c.names) or ib >= len(_c.names):
                    continue
                x, y = self.species[ia], self.species[ib]
                fa, fb = _c.inlet[ia], _c.inlet[ib]
                same_feed = ((fa != fa and fb != fb)          # both unfed
                             or (fa == fa and fb == fb
                                 and abs(float(fa) - float(fb)) < 1e-12))
                if (same_feed
                        and abs(float(_c.c_init[ia] - _c.c_init[ib])) < 1e-12
                        and abs(float(_c.d_rel[ia] - _c.d_rel[ib])) < 1e-12
                        and bool(_c.immobile[ia]) == bool(_c.immobile[ib])):
                    out.append(
                        "%s and %s take no part in any reaction that is "
                        "switched on, and have the same feed, the same "
                        "starting concentration and the same diffusion "
                        "coefficient. They will therefore hold the IDENTICAL "
                        "field, and the dataset will store one chemical "
                        "twice. Give one of them a different diffusion "
                        "coefficient, or remove it with the minus button "
                        "under the chemicals." % (x.name, y.name))
        if g["ks_a_norm"] > 5:
            out.append("the acceptor half-saturation is %.3g times the "
                       "acceptor feed concentration. The Monod factor is then "
                       "below %.3g everywhere, so the reaction is effectively "
                       "switched off. Check that %s really is the electron "
                       "acceptor and that ks_acceptor is in mol/L."
                       % (g["ks_a_norm"], 1.0 / (1.0 + g["ks_a_norm"]), nm[1]))
        if g["alpha_norm"] > 20:
            out.append("%.3g moles of %s are consumed per mole of %s once the "
                       "feed concentrations are taken into account, so the "
                       "acceptor is exhausted almost immediately. Check the "
                       "substrate order and the two feed concentrations."
                       % (g["alpha_norm"], nm[1], nm[0]))
        if self.species[2].left_type.lower().startswith("dir") and \
                self.species[2].left_value > 0:
            out.append("the third chemical, %s, is the reaction PRODUCT and is "
                       "also being fed at the inlet. That is allowed, but it "
                       "means the product concentration no longer measures how "
                       "far the reaction has gone." % nm[2])
        if self.nz == 1 and self.dimension != 2:
            out.append("nz is 1, so this is a two-dimensional run. Use "
                       "build_dataset_2d.py rather than build_dataset_3d.py.")
        return out

    # =====================================================================
    # the conversion
    # =====================================================================
    def groups(self):
        """Physical numbers to the six dimensionless ones, with the arithmetic.

        Returned as a dict so the report can print each line with the formula
        that produced it. Every quantity here also appears in complab_campaign.py,
        which builds CompLaB runs from the same groups; self_test() checks the
        two agree, because a mismatch there would mean the Python dataset and
        the CompLaB dataset were labelled with different numbers for the same
        experiment.
        """
        L = self.L_ref_m
        D = self.donor.d_pore
        ac0 = self.donor.scale(1.0)
        b0 = self.species[3].initial if len(self.species) > 3 else 1.0
        rf = self.rate_factor

        # DIFFUSIVE Damkohler: rate against diffusion. Velocity does not
        # appear, so it is the one that stays fixed while Peclet is swept.
        da_bio_diff = (self.vmax * rf) * b0 / max(ac0, 1e-30) * L * L / D
        # The abiotic reaction consumes the product, so its Damkohler is taken
        # against the PRODUCT's diffusivity. When the product is immobile --
        # a precipitate, say -- there is no diffusivity to divide by, and the
        # donor's is used instead. Dividing by zero here once produced a
        # Damkohler of 5e16 and the report printed it without comment.
        d_p = self.species[2].d_pore if len(self.species) > 2 else D
        if d_p <= 0:
            d_p = D
        da_abio_diff = (self.k_surf * rf) * L * L / max(d_p, 1e-30)
        da_decay_diff = (self.decay * rf) * L * L / D

        pe = float(self.peclet)
        # ADVECTIVE Damkohler: rate against ADVECTION, which is a rate times a
        # time, Da_adv = k * L / u.
        #
        # THIS IS COMPUTED FROM THE VELOCITY, NOT BY DIVIDING THE DIFFUSIVE
        # ONE BY PECLET. The shortcut Da_adv = Da_diff / Pe is only correct
        # when the two are taken against the SAME diffusivity, and they are
        # not: the Peclet number is defined on the donor, while the abiotic
        # reaction's diffusive Damkohler is taken against the PRODUCT, which
        # is the chemical it consumes. Dividing one by the other therefore
        # leaves a stray factor of D_product / D_donor. With the template's
        # own numbers -- donor 1.0e-9, product 8.0e-10 m2/s -- the abiotic
        # Damkohler written into every dataset was 25% too large, and it moved
        # whenever somebody edited a diffusion coefficient that has nothing to
        # do with the abiotic rate.
        #
        # u_mean is the velocity the stated Peclet number implies, so working
        # from it gives each reaction its own honest ratio and the shortcut
        # still holds exactly for the biotic one, where the diffusivities do
        # match.
        u_mean = pe * D / max(L, 1e-30)
        def _adv(rate_per_s):
            return rate_per_s * L / max(u_mean, 1e-30)
        da_bio_adv = _adv((self.vmax * rf) * b0 / max(ac0, 1e-30))
        da_abio_adv = _adv(self.k_surf * rf)
        da_decay_adv = _adv(self.decay * rf)
        return dict(
            L_m=L, D_ref=D, ac0=ac0, b0=b0,
            pe=pe,
            da_bio_diff=da_bio_diff, da_abio_diff=da_abio_diff,
            da_decay_diff=da_decay_diff,
            # ZERO when the reaction is switched off, rather than a copy of
            # the other one. The generators used to write the same number into
            # both columns, so every dataset told the network there was an
            # abiotic reaction going at the biotic rate, when in fact no
            # abiotic reaction was implemented at all.
            da_bio=((da_bio_adv if self.use_physical_kinetics
                     else float(self.damkohler)) if self.biotic_enabled
                    else 0.0),
            da_abio=((da_abio_adv if self.use_physical_kinetics
                      else float(self.damkohler_abiotic))
                     if self.abiotic_enabled else 0.0),
            da_decay=(da_decay_adv if self.use_physical_kinetics else None),
            da_bio_physical=da_bio_adv,
            da_abio_physical=da_abio_adv,
            da_decay_physical=da_decay_adv,
            u_mean_implied=u_mean,
            ks_ac_norm=self.ks_donor / max(ac0, 1e-30),
            ks_a_norm=(self.ks_acceptor / max(self.acceptor.scale(ac0), 1e-30)
                       if self.acceptor is not None else 0.0),
            y_norm=self.yield_,
            alpha_norm=(self.stoichiometry * ac0 /
                        max(self.acceptor.scale(ac0), 1e-30)
                        if self.acceptor is not None else 0.0),
            b0_norm=b0 / max(ac0, 1e-30),
            u_mean=pe * D / L,                    # m/s implied by that Peclet
            t_advective=L / max(pe * D / L, 1e-30),   # seconds to cross L
            t_diffusive=L * L / D,
        )

    def peclet_from_flow(self, u_lattice, nx=None):
        """The Peclet number a measured lattice speed corresponds to.

        The lattice-Boltzmann solve gives a speed in lattice units: voxels per
        lattice time step. Turning that into a Peclet number needs the bridge
        between lattice time and real time, and the bridge is the viscosity.

            nu_lattice = (tau - 1/2) / 3          the model fluid's viscosity
            dt         = nu_lattice * dx^2 / nu   seconds per lattice step
            D_lattice  = D * dt / dx^2 = D * nu_lattice / nu

            Pe = u_lattice * L_lattice / D_lattice

        where L_lattice is the reference length measured in voxels. Everything
        cancels except the ratio of the model fluid's viscosity to the real
        one, which is the only thing tau is really choosing.

        nx  the length of the domain actually being solved, in voxels. Only
            consulted when the reference length is the SAMPLE, where it is the
            whole answer; ignored when ratios are taken against the pore
            width, which does not depend on how long a piece of rock was cut.
        """
        nu_lat = self.viscosity
        if nu_lat <= 0:
            return 0.0
        d_lat = self.donor.d_pore * nu_lat / max(self.fluid_viscosity, 1e-30)
        # None means "the domain the solver was handed"; here there is no
        # domain in hand, so this file's nx is the only sample length there
        # is. Callers that know the real one pass nx to override it.
        l_lat = self.l_ref_voxels
        if l_lat is None:
            l_lat = float(nx if nx else self.nx)
        return float(u_lattice) * l_lat / max(d_lat, 1e-30)

    def n_simulations(self, n_geom, n_sets):
        """How many separate simulations a run is about to do, spelled out.

        There is one flow solve per STRUCTURE, because the pore space does not
        change, and one transport solve per structure per parameter set. So
        "16 structures, 5 sets" is 16 flow solves and 80 transport solves, and
        80 is the number that decides how long you wait.
        """
        return dict(structures=int(n_geom), sets=int(n_sets),
                    flow_solves=int(n_geom),
                    transport_solves=int(n_geom) * int(n_sets),
                    simulations=int(n_geom) * int(n_sets))

    def pressure_for_peclet(self, u_lattice, peclet, nx=None):
        """The pressure gradient that produces a wanted Peclet on this rock.

        This is the answer to "I want Peclet 10, how hard do I have to push?".
        Stokes flow is LINEAR, so speed and therefore Peclet scale exactly
        with the driving pressure. Solve once with any seed pressure, measure
        what Peclet that gave, and the pressure you want is the seed times the
        ratio.

        It is per ROCK, not per run: a tighter pore structure needs a stronger
        push for the same Peclet, which is what permeability means. Two 148 by
        64 structures at the same seed came out at Peclet 999 and 2070, so the
        pressure they need for Peclet 10 differs by a factor of two.
        """
        seed = self.peclet_from_flow(u_lattice, nx)
        if seed <= 0:
            return 0.0
        return float(self.delta_P) * float(peclet) / seed

    def reaction_text(self):
        """The reaction network, written out, in the reader's own vocabulary.

        Printed on every run. The question "which reactions am I actually
        running, and with what rate law" should never require reading the
        solver, and until this existed it did.
        """
        nm = self.names + ["?"] * 4
        out = []
        if self.biotic_enabled:
            ac = ("%s/(Ks_%s + %s)" % (nm[0], nm[0], nm[0])
                  if self.ks_donor > 0 else nm[0])
            bio = (" * %s/%s0" % (nm[3], nm[3])) if self.biomass_coupled else ""
            out.append(("BIOTIC   (microbially mediated)",
                        ["%s + %g %s  ->  %s + %g %s"
                         % (nm[0], self.stoichiometry, nm[1], nm[2],
                            self.yield_, nm[3]),
                         "rate = vmax * %s * %s * %s/(Ks_%s + %s)%s"
                         % (nm[3], ac, nm[1], nm[1], nm[1], bio),
                         "and the biomass decays: -decay * %s" % nm[3],
                         ("the rate uses the biomass actually present"
                          if self.biomass_coupled else
                          "the rate holds the biomass at its starting value, "
                          "because B0 is already inside the Damkohler number"),
                         "the biomass is %s" % (
                             "SESSILE: attached, reacts where it is, never "
                             "moves" if str(self.biomass_mode).lower()
                             .startswith("sess") else
                             "PLANKTONIC: dissolved, advected and diffused "
                             "like any other chemical, and most of it leaves "
                             "the sample")]))
        else:
            out.append(("BIOTIC   (microbially mediated)", ["OFF"]))
        if self.abiotic_enabled:
            j = self.abiotic_index
            j_mk = self.abiotic_product_index
            out.append(("ABIOTIC  (purely chemical, no microbes)",
                        ["%s  ->  %s" % (nm[j], nm[j_mk] if j_mk is not None
                                         else "consumed, nothing tracked "
                                              "is made"),
                         "rate = k_surf * %s      (first order)" % nm[j],
                         ("one mole of %s per mole of %s, so the two of them "
                          "sum to the feed" % (nm[j_mk], nm[j])
                          if j_mk is not None else
                          "a terminal loss: right for a solute that "
                          "precipitates out or degrades to something this "
                          "run does not follow"),
                         "no microbes take any part in it, and none need to "
                         "be present",
                         "it runs %s" % ("ONLY on pore voxels touching a grain "
                                         "surface" if self.abiotic_surface_only
                                         else "everywhere in the water")]))
        else:
            out.append(("ABIOTIC  (purely chemical, no microbes)",
                        ["OFF. Nothing consumes the product, so the product "
                         "field measures how far the biotic reaction has "
                         "gone and nothing else."]))
        return out

    def chemistry(self, n_species=None):
        """The object the transport solver takes instead of its own constants.

        With a default Settings this returns exactly the numbers prtlb_2d.py / prtlb_3d.py
        used to hold as module constants. That equality is asserted in
        self_test(), because the whole value of this file rests on it: if the
        defaults drifted, every dataset made before today and every dataset
        made after would be labelled the same and be different.
        """
        g = self.groups()
        ns = int(n_species or len(self.species))
        ns = max(1, min(ns, len(self.species)))
        ac0 = g["ac0"]

        # Sessile or planktonic is a property of the SYSTEM, not of a line in
        # a chemical table, so it is decided here rather than being left to
        # whoever remembers to tick the immobile box on the fourth chemical.
        #
        # WORKED OUT, NOT WRITTEN BACK. This used to assign to self.species[3],
        # and the consequence was that calling chemistry() CHANGED the
        # settings. report() calls chemistry() and then calls complaints(),
        # and one of the complaints is "the biomass is not marked immobile, so
        # it will be advected out of the sample" -- which by then could never
        # fire, because chemistry() had just marked it immobile. The warning
        # existed, was correct, and was unreachable from the one place that
        # prints it. A method that answers a question must not also change the
        # answer.
        bio_over = {}
        if len(self.species) > 3:
            sessile = str(self.biomass_mode).lower().startswith("sess")
            if sessile:
                bio_over = dict(immobile=True, d_pore=0.0, d_biofilm=0.0)
            else:
                bio_over = dict(immobile=False)
                if self.species[3].d_pore <= 0:
                    # A planktonic cell that cannot diffuse is not planktonic.
                    # The value is the one complab_campaign.py uses: cells are about
                    # five times slower than a small solute.
                    bio_over.update(d_pore=2.0e-10, d_biofilm=1.0e-10)

        inlet, c_init, d_rel, d_bf, immob = [], [], [], [], []
        for i_sp, sp in enumerate(self.species[:ns]):
            if i_sp == ATTACHED_INDEX and bio_over:
                sp = sp.copy(**bio_over)
            sc = sp.scale(ac0)
            if sp.left_type.lower().startswith("dir"):
                inlet.append(sp.left_value / sc if sc > 0 else 0.0)
            else:
                inlet.append(np.nan)       # not a boundary species
            # NO MICROBES MEANS NO BIOMASS. Not a biomass that sits there
            # decaying: with the biotic reaction off, a starting biomass
            # produces a field that is uniform everywhere, carries no
            # information, and tells anyone who opens it that microbes were
            # present. Measured on one such run: Bio came out at 0.0609
            # across the whole pore space, having decayed from 0.1, in a
            # simulation with no microbial reaction in it at all.
            init = sp.initial
            # Index 3 is the biomass. Counted explicitly rather than from the
            # length of a list that has already been appended to this pass --
            # which is off by one, and quietly left the biomass at 0.1.
            if len(c_init) == ATTACHED_INDEX and not self.biotic_enabled:
                init = 0.0
            c_init.append(init / max(sc, 1e-30))
            d_rel.append(sp.d_pore / max(g["D_ref"], 1e-30))
            d_bf.append(sp.d_biofilm / max(g["D_ref"], 1e-30))
            immob.append(bool(sp.immobile) or sp.d_pore <= 0)

        return Chemistry(
            names=self.names[:ns],
            inlet=np.array(inlet, np.float32),
            c_init=np.array(c_init, np.float32),
            d_rel=np.array(d_rel, np.float32),
            immobile=np.array(immob, bool),
            ks_donor=float(g["ks_ac_norm"]),
            ks_acceptor=float(g["ks_a_norm"]),
            alpha=float(g["alpha_norm"]),
            yield_=float(g["y_norm"]),
            da_decay=(None if g["da_decay"] is None else float(g["da_decay"])),
            decay_lattice=(None if self.use_physical_kinetics
                           else float(self.decay_lattice)),
            source=self.source,
            biomass_coupled=bool(self.biomass_coupled),
            abiotic_surface_only=bool(self.abiotic_surface_only),
            abiotic_index=int(self.abiotic_index),
            abiotic_product_index=self.abiotic_product_index,
            biotic_on=bool(self.biotic_enabled),
            d_biofilm_rel=np.array(d_bf, np.float32),
            biofilm_code=self.biofilm_code,
            seed_in_biofilm=self.biofilm_code is not None,
            # THE LENGTH EVERY RATIO IS TAKEN AGAINST, in voxels, carried all
            # the way to the solver instead of stopping here.
            #
            # reference_length was read, validated, reported and printed with
            # a helpful note about how the two conventions differ -- and then
            # the solver used the sample length regardless, because it had no
            # way to be told otherwise. Choosing "characteristic" changed the
            # numbers in the report and nothing in the simulation, so the
            # dataset was labelled in pore widths and computed in sample
            # lengths. At the defaults, 32 voxels against a 16 um pore, that
            # is a factor of two on the Peclet number and on both Damkohler
            # numbers, in opposite directions.
            l_ref_voxels=self.l_ref_voxels)

    # =====================================================================
    # reporting
    # =====================================================================
    def report(self, out=None):
        """Print what was read, what it implies, and what was dropped."""
        w = (out or sys.stdout).write
        g = self.groups()
        line = "-" * 74 + "\n"

        w(line)
        w("SIMULATION SETTINGS   read from %s\n" % self.source)
        w(line)
        w("\nDOMAIN\n")
        w("  grid                       %d x %d x %d voxels\n"
          % (self.nx, self.ny, self.nz))
        w("  voxel edge  dx             %g %s\n" % (self.dx, self.unit))
        w("  sample length  nx*dx       %g %s   = %.4g m\n"
          % (self.nx * self.dx, self.unit, self.L_sample_m))
        w("  pore width  characteristic %g %s   = %.4g m\n"
          % (self.characteristic_length, self.unit, self.L_char_m))
        w("  ratios are taken against   the %s length, %.4g m\n"
          % (self.reference_length, self.L_ref_m))
        w("     the other convention differs by a factor of %.3g; a Peclet of "
          "%.3g\n     here is a Peclet of %.3g there.\n"
          % (self.L_sample_m / max(self.L_char_m, 1e-30), g["pe"],
             g["pe"] * (self.L_char_m / max(self.L_sample_m, 1e-30)
                        if self.reference_length == "sample"
                        else self.L_sample_m / max(self.L_char_m, 1e-30))))
        w("  material codes             pore %d, solid %d, wall %d\n"
          % (self.pore_code, self.solid_code, self.wall_code))

        w("\nFLOW\n")
        by_pressure = str(self.flow_driver).lower().startswith("pres")
        w("  what sets the flow speed   %s\n"
          % ("THE PRESSURE DROP. The flow is solved with the pressure drop "
             "below\n                             and is NOT rescaled; the "
             "Peclet number is whatever\n                             that "
             "flow makes it, and is computed per run and reported."
             if by_pressure else
             "THE PECLET NUMBER. The flow is solved, then rescaled so\n"
             "                             that Peclet is the value below. "
             "The pressure drop still\n                             drives "
             "the lattice-Boltzmann solve, but its size divides out."))
        w("  relaxation time  tau       %g   ->  viscosity %.4f lattice units\n"
          % (self.tau, self.viscosity))
        w("  pressure drop  delta_P     %g%s\n"
          % (self.delta_P, "" if by_pressure else "   (a seed; see above)"))
        w("  fluid viscosity            %g m2/s%s\n"
          % (self.fluid_viscosity,
             "   (turns lattice speed into Peclet)"))
        if by_pressure:
            w("  Peclet                     computed per run from the flow\n")
        elif self.peclet_min is not None and self.peclet_max is not None:
            w("  Peclet                     swept from %g to %g, log spaced\n"
              % (self.peclet_min, self.peclet_max))
        else:
            w("  Peclet                     %g\n" % self.peclet)
        if not by_pressure:
            w("  implied mean pore speed    %.4g m/s  (Pe * D / L)\n"
              % g["u_mean"])
            w("  time to cross the sample   %.4g s advective, %.4g s "
              "diffusive\n" % (g["t_advective"], g["t_diffusive"]))
        else:
            w("  crossing time by diffusion %.4g s   (the advective one "
              "depends on\n                             the flow and is "
              "reported per run)\n" % g["t_diffusive"])

        w("\nCHEMICALS\n")
        w("  %-6s %10s %10s %11s %11s %9s\n"
          % ("name", "initial", "inlet", "D pore", "D biofilm", "moves"))
        for sp in self.species[:8]:
            fed = (sp.left_value if sp.left_type.lower().startswith("dir")
                   else float("nan"))
            w("  %-6s %10.4g %10s %11.3g %11.3g %9s\n"
              % (sp.name, sp.initial,
                 ("%.4g" % fed) if fed == fed else "no feed",
                 sp.d_pore, sp.d_biofilm, "no" if sp.immobile else "yes"))
        w("  concentrations in mol/L, diffusion in m^2/s\n")
        if len(self.species) > 8:
            w("  ... and %d more\n" % (len(self.species) - 8))
        if len(self.species) > 4:
            w("  the first four react; the rest are conservative tracers\n")

        w("\nREACTIONS -- there are two, and they are separate\n")
        for title, lines in self.reaction_text():
            w("\n  %s\n" % title)
            for ln in lines:
                w("      %s\n" % _wrap(ln, 66, "          "))
        w("\n  the constants\n")
        nm = self.names + ["?"] * 4
        w("  vmax                       %g %s\n" % (self.vmax, self.rate_unit))
        w("  half saturation, donor     %g mol/L%s\n"
          % (self.ks_donor, "   (zero: first order in the donor)"
             if self.ks_donor == 0 else ""))
        w("  half saturation, acceptor  %g mol/L\n" % self.ks_acceptor)
        w("  yield                      %g mol biomass per mol donor\n"
          % self.yield_)
        w("  decay                      %g %s\n" % (self.decay, self.rate_unit))
        w("  abiotic rate constant      %g %s\n" % (self.k_surf, self.rate_unit))

        w("\nWHAT THE SOLVER AND THE NETWORK ACTUALLY RECEIVE\n")
        w("  these are the six numbers stored per run in the dataset\n\n")
        if by_pressure:
            # Showing a Peclet here in pressure mode is a lie with a number
            # in it. The whole point of pressure mode is that Peclet is an
            # OUTPUT, and one run of this reported "Peclet 10" in this block
            # while the simulations it then did came out at 999 and 2070.
            w("  Peclet             computed per run from the flow the "
              "pressure drop\n                     produces, and different "
              "rocks have different\n                     permeability, so "
              "they come out at different Peclet.\n"
              "                     Each one is printed as its run starts "
              "and stored\n                     with it.\n")
        else:
            w("  Peclet        %12.5g   = u L / D          , u=%.3g m/s "
              "L=%.3g m D=%.3g\n"
              % (g["pe"], g["u_mean"], g["L_m"], g["D_ref"]))
        w("  Da biotic     %12.5g   %s\n"
          % (g["da_bio"] if self.biotic_enabled else 0.0,
             ("= vmax B0 L / (Ac0 u), from the kinetics above"
              if self.use_physical_kinetics
              else "set directly; the kinetics above are read but not used "
                   "for this") if self.biotic_enabled
             else "the biotic reaction is switched OFF"))
        w("  Da abiotic    %12.5g   %s\n"
          % (g["da_abio"] if self.abiotic_enabled else 0.0,
             ("= k_surf L / u" if self.use_physical_kinetics
              else "set directly") if self.abiotic_enabled
             else "the abiotic reaction is switched OFF, so this is zero "
                  "rather\n                               than a copy of the "
                  "biotic one"))
        if not self.use_physical_kinetics:
            w("\n  YOUR KINETICS IMPLY, if you switched use_physical_kinetics "
              "on:\n")
            w("      Da biotic  %.4g      Da abiotic  %.4g\n"
              % (g["da_bio_physical"], g["da_abio_physical"]))
            if g["da_bio_physical"] < 1e-2:
                w("      That is far below 1, and it is not a mistake in your\n"
                  "      numbers. Diffusion crosses %g %s in %.3g s, and no\n"
                  "      ordinary microbial rate competes with that. Real\n"
                  "      columns reach Damkohler 1 by being centimetres long.\n"
                  "      A training set still wants 0.1 to 10, which is why\n"
                  "      the sweep sets it directly.\n"
                  % (self.characteristic_length if
                     self.reference_length.startswith("char")
                     else self.nx * self.dx, self.unit, g["t_diffusive"]))
        w("  Ks donor      %12.5g   = Ks_Ac / Ac0\n" % g["ks_ac_norm"])
        w("  Ks acceptor   %12.5g   = Ks_A / A0\n" % g["ks_a_norm"])
        w("  yield         %12.5g   = mol biomass per mol donor\n" % g["y_norm"])
        w("\n  and three more the solver needs that the network is not given\n")
        w("  stoichiometry %12.5g   = %g mol acceptor per mol donor, times "
          "Ac0/A0\n" % (g["alpha_norm"], self.stoichiometry))
        w("  initial biomass%11.5g   = B0 / Ac0\n" % g["b0_norm"])
        w("  biomass decay %12.5g   %s\n"
          % (self.chemistry().decay_for(self.nx),
             "= decay L / u, divided by nx" if self.use_physical_kinetics
             else "in lattice units, the pre-existing default"))
        w("\n  relative diffusion coefficients, against %s\n"
          % self.species[0].name)
        w("%s\n" % self.chemistry(min(4, len(self.species))).describe())

        if self.notes:
            w("\nNOTES\n")
            for n in self.notes:
                w("  * %s\n" % _wrap(n))
        if self.ignored:
            w("\nREAD BUT NOT SUPPORTED -- these settings had no effect\n")
            for n in self.ignored:
                w("  * %s\n" % _wrap(n))
        warn = self.warnings()
        if warn:
            w("\nLEGAL, BUT PROBABLY NOT WHAT YOU MEANT\n")
            for n in warn:
                w("  * %s\n" % _wrap(n))
        bad = self.complaints()
        if bad:
            w("\nPROBLEMS -- the run will not start until these are fixed\n")
            for n in bad:
                w("  * %s\n" % _wrap(n))
        elif not warn:
            w("\nno problems found.\n")
        w(line)
        return not bad

    # =====================================================================
    # writing
    # =====================================================================
    def to_xml(self):
        """A settings file with every supported tag, its units, and a note on
        what it changes. Written by hand rather than by a serialiser, because
        the comments are the point: a template you cannot read is a template
        you paste numbers into blindly."""
        sp = []
        for i, s in enumerate(self.species):
            sp.append("""
        <substrate%d>
            <name_of_substrates>%s</name_of_substrates>
            <initial_concentration>%g</initial_concentration>   <!-- mol/L everywhere in the pore at t=0 -->
            <substrate_diffusion_coefficients>
                <in_pore>%g</in_pore>        <!-- m^2/s in open water -->
                <in_biofilm>%g</in_biofilm>     <!-- m^2/s inside the biofilm; read but not yet used by the Python solver -->
            </substrate_diffusion_coefficients>
            <left_boundary_type>%s</left_boundary_type>     <!-- Dirichlet = fed at this face; Neumann = free -->
            <left_boundary_condition>%g</left_boundary_condition>  <!-- mol/L, used only when Dirichlet -->
            <right_boundary_type>%s</right_boundary_type>
            <right_boundary_condition>%g</right_boundary_condition>%s
        </substrate%d>""" % (i, s.name, s.initial, s.d_pore, s.d_biofilm,
                             s.left_type, s.left_value, s.right_type,
                             s.right_value,
                             "\n            <immobile>true</immobile>          "
                             "<!-- attached: reacts in place, never moves -->"
                             if s.immobile else "", i))

        return TEMPLATE_XML % dict(
            nx=self.nx, ny=self.ny, nz=self.nz, dx=self.dx, unit=self.unit,
            char=self.characteristic_length, refl=self.reference_length,
            pore=self.pore_code, solid=self.solid_code, wall=self.wall_code,
            tau=self.tau, dP=self.delta_P, pe=self.peclet,
            ns_max=self.ns_max_iT, ns_cv=self.ns_converge,
            ade_max=self.ade_max_iT, ade_cv=self.ade_converge,
            nsnap=self.n_snapshots, nsub=len(self.species),
            substrates="".join(sp), runit=self.rate_unit,
            vmax=self.vmax, ksd=self.ks_donor, ksa=self.ks_acceptor,
            yld=self.yield_, dec=self.decay, stoi=self.stoichiometry,
            ksurf=self.k_surf, da=self.damkohler,
            fdrv=self.flow_driver, nu=self.fluid_viscosity,
            bio="true" if self.biotic_enabled else "false",
            abio="true" if self.abiotic_enabled else "false",
            abiosurf="true" if self.abiotic_surface_only else "false",
            rmode=self.reaction_mode, areact=self.abiotic_reactant,
            aprod=self.abiotic_product,
            bfcode=("none" if self.biofilm_code is None
                    else str(self.biofilm_code)),
            bflayers=int(self.biofilm_layers),
            da2=self.damkohler_abiotic, bmode=self.biomass_mode,
            bcoup="true" if self.biomass_coupled else "false",
            declat=self.decay_lattice,
            usephys="true" if self.use_physical_kinetics else "false",
            pemin=self.peclet_min if self.peclet_min is not None else 1.0,
            pemax=self.peclet_max if self.peclet_max is not None else 50.0,
            damin=self.damkohler_min if self.damkohler_min is not None else 0.1,
            damax=self.damkohler_max if self.damkohler_max is not None else 10.0)

    def to_json(self):
        d = {k: v for k, v in vars(self).items()
             if k not in ("species", "ignored", "notes", "source",
                          "dimension")}
        d["yield"] = d.pop("yield_")
        d["species"] = [dict(name=s.name, initial=s.initial, d_pore=s.d_pore,
                             d_biofilm=s.d_biofilm, left_type=s.left_type,
                             left_value=s.left_value, right_type=s.right_type,
                             right_value=s.right_value, immobile=s.immobile)
                        for s in self.species]
        return json.dumps(d, indent=2)


# --------------------------------------------------------------------------
class Chemistry(object):
    """What the transport solver needs, all of it dimensionless.

    Separate from Settings on purpose. Settings is what YOU write, in mol/L
    and m^2/s. Chemistry is what the solver reads, in the units it integrates
    in. Keeping them apart means the conversion happens once, in one function,
    where it can be printed and checked -- rather than in scattered places
    where a missing factor of Peclet looks like a modelling choice.
    """

    def __init__(self, names, inlet, c_init, d_rel, immobile,
                 ks_donor=0.0, ks_acceptor=0.15, alpha=0.4, yield_=0.04,
                 da_decay=None, decay_lattice=None, source="built in",
                 biomass_coupled=False, abiotic_surface_only=False,
                 d_biofilm_rel=None, biofilm_code=None, seed_in_biofilm=False,
                 abiotic_index=2, biotic_on=True, l_ref_voxels=None,
                 abiotic_product_index=None):
        self.names = list(names)
        n = len(self.names)
        self.inlet = np.asarray(inlet, np.float32)
        self.c_init = np.asarray(c_init, np.float32)
        self.d_rel = np.asarray(d_rel, np.float32)
        self.immobile = np.asarray(immobile, bool)
        for a, nm in ((self.inlet, "inlet"), (self.c_init, "c_init"),
                      (self.d_rel, "d_rel"), (self.immobile, "immobile")):
            if a.shape != (n,):
                raise ValueError("chemistry: %s has %d entries but there are "
                                 "%d chemicals" % (nm, a.size, n))
        self.ks_donor = float(ks_donor)
        self.ks_acceptor = float(ks_acceptor)
        self.alpha = float(alpha)
        self.yield_ = float(yield_)
        # Decay is given EITHER as a Damkohler, which is grid independent, OR
        # already in lattice units. The solver divides the first by nx.
        self.da_decay = None if da_decay is None else float(da_decay)
        self.decay_lattice = (None if decay_lattice is None
                              else float(decay_lattice))
        # Does the biotic rate follow the biomass that is actually present, or
        # is the biomass held at its starting value inside the rate? See
        # rate_biotic() in prtlb_2d.py / prtlb_3d.py; the default is held, because B0 is
        # already inside the Damkohler number.
        self.biomass_coupled = bool(biomass_coupled)
        # Does the abiotic reaction happen only on the grain surfaces, or
        # everywhere in the water?
        self.abiotic_surface_only = bool(abiotic_surface_only)
        # Diffusion inside a biofilm voxel, relative to the first chemical's
        # pore value. Used only where the geometry actually carries biofilm
        # voxels; without a biofilm code there is nowhere for it to apply and
        # the pore value is used everywhere.
        self.d_biofilm_rel = (self.d_rel.copy() if d_biofilm_rel is None
                              else np.asarray(d_biofilm_rel, np.float32))
        self.biofilm_code = (None if biofilm_code is None else int(biofilm_code))
        self.seed_in_biofilm = bool(seed_in_biofilm)
        self.abiotic_index = int(abiotic_index)
        # Which chemical the abiotic reaction MAKES, or None for a pure
        # loss. None is what the solver did unconditionally before this
        # existed, so an abiotic reaction could only ever destroy.
        self.abiotic_product_index = (None if abiotic_product_index is None
                                      else int(abiotic_product_index))
        # Are there microbes in this run at all? With none, the biomass is not
        # a field that decays -- it is a field that does not exist.
        self.biotic_on = bool(biotic_on)
        # The length, in voxels, that Peclet and both Damkohler numbers are
        # taken against. None means the sample length, which is what the
        # solver used unconditionally before this existed.
        self.l_ref_voxels = (None if l_ref_voxels is None
                             else float(l_ref_voxels))
        self.source = source

    def decay_for(self, nx):
        if self.decay_lattice is not None:
            return float(self.decay_lattice)
        return float((self.da_decay or 0.0) / max(nx, 1))

    def truncate(self, ns):
        ns = max(1, min(int(ns), len(self.names)))
        return Chemistry(self.names[:ns], self.inlet[:ns], self.c_init[:ns],
                         self.d_rel[:ns], self.immobile[:ns],
                         self.ks_donor, self.ks_acceptor, self.alpha,
                         self.yield_, self.da_decay, self.decay_lattice,
                         self.source, self.biomass_coupled,
                         self.abiotic_surface_only, self.d_biofilm_rel[:ns],
                         self.biofilm_code, self.seed_in_biofilm,
                         self.abiotic_index, self.biotic_on,
                         self.l_ref_voxels, self.abiotic_product_index)

    def describe(self):
        rows = ["  %-6s %9s %9s %8s %9s" % ("name", "feed", "at t=0",
                                            "D/D_ref", "moves")]
        for i, nm in enumerate(self.names):
            fed = self.inlet[i]
            rows.append("  %-6s %9s %9.4g %8.3g %9s"
                        % (nm, "no feed" if fed != fed else "%.4g" % fed,
                           self.c_init[i], self.d_rel[i],
                           "no" if self.immobile[i] else "yes"))
        return "\n".join(rows)


# The numbers prtlb_2d.py / prtlb_3d.py held as module constants before this file existed.
# Written out here so that "the default" is one thing in one place.
def default_chemistry(n_species=2):
    ns = max(1, min(int(n_species), 4))
    return Chemistry(
        names=["Ac", "A", "P", "Bio"][:ns],
        # 1, 1, 0, no feed. The product IS held at zero at the inlet; the
        # attached biomass is not a boundary chemical at all. Those two are
        # different things and writing both as 0 hid the difference.
        inlet=np.array([1.0, 1.0, 0.0, np.nan], np.float32)[:ns],
        c_init=np.array([0.0, 0.0, 0.0, 0.1], np.float32)[:ns],
        d_rel=np.array([1.0, 1.0, 1.0, 0.0], np.float32)[:ns],
        d_biofilm_rel=np.array([1.0, 1.0, 1.0, 0.0], np.float32)[:ns],
        immobile=np.array([False, False, False, True])[:ns],
        ks_donor=0.0, ks_acceptor=0.15, alpha=0.4, yield_=0.04,
        decay_lattice=0.006, source="built in")


# --------------------------------------------------------------------------
# The plumbing both dataset generators share, so that --settings behaves
# identically in 2D and in 3D. It lived in neither of them on purpose: two
# copies of an argument parser is how the 2D and 3D builders came to disagree
# about what --n-times meant.
# --------------------------------------------------------------------------
def add_settings_arguments(ap):
    g = ap.add_argument_group(
        "settings_and_units settings",
        "Your own numbers, in your own units. Without any of these the "
        "generator uses the built-in defaults, which are what it always used.")
    g.add_argument("--settings", metavar="FILE", default=None,
                   help="an XML or JSON settings file. Write one with "
                        "'python settings_and_units.py --template FILE'. A real "
                        "CompLaB.xml also works; anything in it this solver "
                        "cannot honour is listed by name before the run.")
    g.add_argument("--set", metavar="KEY=VALUE", action="append", default=[],
                   help="override one setting without editing the file. "
                        "Repeatable. 'tau=0.9' for a top-level setting, "
                        "'species.A.d_pore=5e-10' for one chemical's.")
    g.add_argument("--show-settings", action="store_true",
                   help="print the full settings report and stop, without "
                        "simulating anything. This is how you check your "
                        "numbers before spending hours on them.")
    g.add_argument("--save-settings", metavar="FILE", default=None,
                   help="also write everything this run used to a settings "
                        "file, so it can be re-read, shared or edited")
    return ap


def settings_from_args(a, dimension, block=None, chemicals=None,
                       block_name="the block at the top of this script",
                       default_shape=None):
    """Build the Settings for a generator run and print the report.

    Returns (settings, chemistry). Raises SystemExit on a file with mistakes
    in it, BEFORE any simulation starts -- the whole value of checking a
    settings file is that it happens in the first second rather than the
    fourth hour.
    """
    # WHERE THE NUMBERS COME FROM, in order, and it is stated out loud in the
    # report every time:
    #   1. the editable block at the top of the generator you ran
    #   2. a settings file, if you gave one -- it replaces the block entirely
    #   3. anything set individually, which wins over both
    if getattr(a, "settings", None):
        s = Settings.from_file(a.settings)
        if block or chemicals:
            s.notes.append(
                "a settings file was given, so %s was NOT used. Delete the "
                "--settings argument to go back to editing the script."
                % block_name)
    elif block or chemicals:
        s = Settings.from_block(block, chemicals, where=block_name)
    else:
        s = Settings()
        s.source = "built-in defaults"
    for item in getattr(a, "set", []) or []:
        if "=" not in item:
            raise SystemExit("--set wants KEY=VALUE, got '%s'" % item)
        k, v = item.split("=", 1)
        s.apply_override(k.strip(), v.strip())
    s.dimension = dimension

    # THE GRID IS RESOLVED BEFORE THE REPORT IS PRINTED, and it has to be.
    # It used to be resolved afterwards, so a run on a 148 by 64 domain
    # printed a settings report describing a 32 by 32 one -- along with the
    # sample length, the implied pore speed and both crossing times, every one
    # of which is computed FROM the grid and so was wrong by the same factor.
    # The report is meant to be the thing you check before committing hours;
    # it cannot be describing a different run.
    if default_shape is not None:
        cli = getattr(a, "shape", None)
        if cli:
            got, where = tuple(cli), "the command line"
        elif not s.source.startswith(("built-in", "the MY_SETTINGS",
                                      "an editable")):
            got = ((s.nx, s.ny) if len(default_shape) == 2
                   else (s.nx, s.ny, s.nz))
            where = "the settings file"
        else:
            got, where = tuple(default_shape), "the built-in default"
        s.nx = int(got[0])
        s.ny = int(got[1])
        s.nz = int(got[2]) if len(got) > 2 else 1
        s.notes.append("the grid %s came from %s" % (got, where))
    if dimension == 2:
        s.nz = 1
    s.report()
    bad = s.complaints()
    warn = s.warnings()
    if bad:
        raise SystemExit("\nthe settings file has %d problem(s) listed above. "
                         "Nothing was simulated." % len(bad))
    if warn:
        print("the settings above will run, but read the warnings first.\n")
    dest = getattr(a, "out", None)
    checking = bool(getattr(a, "show_settings", False))
    if dest and not checking:
        print("RESULTS WILL BE SAVED TO")
        print("    %s" % os.path.abspath(dest))
        print()
    out = getattr(a, "save_settings", None)
    if out:
        d = os.path.dirname(os.path.abspath(out))
        if d:
            os.makedirs(d, exist_ok=True)
        text = s.to_json() if out.lower().endswith(".json") else s.to_xml()
        with open(out, "w") as f:
            f.write(text)
        print("settings written to %s\n" % out)
    if checking:
        # Saying "results will be saved to" and then saving nothing is worse
        # than saying nothing at all: the folder gets made, it stays empty,
        # and the run finishes in a second, which reads as a crash.
        print("=" * 70)
        print("NOTHING WAS SIMULATED. This was a check only.")
        print()
        print("  Everything above is what your settings MEAN, worked out")
        print("  without running anything. No pore structure was generated,")
        print("  no flow was solved, no transport was solved, and no file was")
        print("  written -- which is why this took a second and why the output")
        print("  folder is empty.")
        print()
        print("  When the numbers above are the ones you meant, untick")
        print("  'Just check the numbers, do not simulate' and press Run.")
        print("  On the command line, drop --show-settings.")
        print("=" * 70)
    return s, s.chemistry()


CAPABILITIES = """
WHAT THE PYTHON SIMULATOR DOES, AND WHAT IT DOES NOT

These scripts are not a reimplementation of CompLaB and are not trying to be.
CompLaB is the reference: it is the one to publish a mechanism from. The
Python side exists so that a whole training dataset can be built on one
machine in an afternoon, which is what a neural operator needs and what a
cluster queue cannot give you. Knowing exactly where the two differ is the
difference between a fair comparison and a wasted month.

FLOW
  both          incompressible flow through the pore space, lattice Boltzmann,
                no-slip at the grain surfaces
  both          the no-slip wall sits halfway between the last fluid voxel and
                the first solid one, which is where Palabos puts it; this was
                checked against the analytic parabola and is in test_flow_solvers.py
  both          TWO relaxation times, with the magic parameter 3/16, so the
                bounce-back wall sits at the mid-link at every tau rather than
                only at tau = 0.9330. Without it a four-voxel throat carried
                73% too much flow at tau = 2 and the permeability depended on
                a numerical parameter.
  CompLaB only  full MRT, all nine or nineteen relaxation rates
  CompLaB only  a flow field that is re-solved as the biofilm grows
  Python        D2Q9 in two dimensions, D3Q19 in three, double precision,
                solved once per structure because the pore space is fixed, and
                run until the field stops changing rather than for a fixed
                number of iterations

TRANSPORT
  both          advection, diffusion and reaction on the same grid as the flow
  both          Dirichlet feed at the inlet face, free outflow at the outlet,
                no flux at every grain surface
  both          a per-chemical diffusion coefficient
  both          a separate diffusivity inside the biofilm. Give the biofilm a
                material number with <biofilm_code> and diffusion_in_biofilm
                is used on those voxels, with a harmonic mean at each face so
                the flux across the pore-to-biofilm boundary is continuous.
                Without a biofilm code there are no such voxels and the pore
                value is used everywhere, which is the same answer.
  CompLaB only  Dirichlet conditions on the transverse faces
  Python        explicit time stepping with one combined stability limit; the
                run length comes from EVERY chemical settling, not from the
                donor alone and not from a fixed step count
  Python        second-order advection with a minmod limiter, so the Peclet
                number in the file is the one the sample experienced. Plain
                upwinding adds |u|dx/2 of numerical diffusion, which at
                Pe = 300 on 64 voxels is four times the settings_and_units amount.

CHEMISTRY
  both          Monod kinetics, dual limitation on donor and acceptor
  both          biomass growth from the donor with a yield, and first-order
                decay
  both          an abiotic rate constant
  CompLaB only  aqueous speciation and chemical equilibrium, the R1-R73 tableau
  CompLaB only  precipitation, and pore voxels turning into solid
  CompLaB only  surface complexation, and mineral surface sites
  CompLaB only  many microbial pools, attached and planktonic, each with its
                own kinetics file
  Python        one reaction network: donor + acceptor -> product + biomass,
                with one attached biomass pool. You may add as many further
                chemicals as you like and they are transported properly -- fed
                at the inlet, advected, and diffused with their own
                coefficient -- but they take no part in any reaction. That is
                a conservative tracer, which is useful in its own right. A
                fifth REACTING chemical needs CompLaB.

COST, MEASURED ON ONE CORE
  Python        32^3 about 7 s per run, 48^3 about 58 s, 64^3 about 4 min,
                and the flow solve is once per structure
  CompLaB       hours per run, on a cluster, in a queue

WHICH TO USE
  Use the Python generator to build the training set, to sweep Peclet and
  Damkohler across decades, and to prove a pipeline end to end before spending
  cluster time. Use CompLaB for the reference runs the paper rests on, for any
  chemistry beyond one donor and one acceptor, and for anything where the pore
  space changes while the run proceeds.

  A network trained on Python data and then evaluated against CompLaB output
  is a fair test of the method. A network trained on Python data and reported
  as though it had seen CompLaB physics is not, and that is why every dataset
  file records which generator wrote it.
"""


TEMPLATE_XML = """<?xml version="1.0" ?>
<!-- =========================================================================
     PRT-DeepONet-3D  :  SIMULATION SETTINGS for the Python generators
     =========================================================================

     HOW TO READ THIS FILE
       This is a settings file, not a program. A setting is a labelled box:
       <name>value</name> means "name = value". Boxes inside boxes group
       related settings. Anything between an angle bracket, an exclamation
       mark and two dashes, and the closing two dashes and angle bracket, is a
       note for you and is ignored by the code. This whole guide is one note.

     THE LAYOUT IS CompLaB'S ON PURPOSE
       The tag names are the ones CompLaB uses, so you can hand a real
       CompLaB.xml to these scripts and they will read what they understand.
       They will also print a list of everything in it they cannot honour, by
       name, rather than dropping it quietly. See the <usage> block below
       for the full list of what this solver does and does not do. The exact
       commands are in the <usage> block below, because an XML note is not
       allowed to contain two dashes in a row and every one of them has a flag
       in it.

     UNITS, ONCE
       lengths          micrometres (dx, characteristic_length)
       concentrations   mol/L      so 2e-3 is 2 mM
       diffusion        m^2/s      so 1e-9 is a small ion in water
       rates            per second, unless you set rate_unit to per_day
       tau, delta_P     lattice units, exactly as in CompLaB

     CHECK IT BEFORE YOU RUN IT
       The check command in <usage> prints every number in this file, every
       number derived from them, and every mistake, before anything runs.
     ========================================================================= -->
<parameters>

  <usage>
    check this file       python settings_and_units.py --check FILE
    what it cannot do     python settings_and_units.py --capabilities
    build a 3D dataset    python build_dataset_3d.py --settings FILE --out data3d.h5
    build a 2D dataset    python build_dataset_2d.py --settings FILE --out data2d.h5
  </usage>

  <LB_numerics>

    <domain>
      <nx>%(nx)d</nx>                     <!-- voxels along x; flow is +x, inlet at x=0 -->
      <ny>%(ny)d</ny>                     <!-- voxels along y -->
      <nz>%(nz)d</nz>                     <!-- voxels along z; set to 1 for a 2D run -->
      <dx>%(dx)g</dx>                     <!-- voxel edge length -->
      <unit>%(unit)s</unit>                  <!-- um, mm or m -->
      <characteristic_length>%(char)g</characteristic_length>
                                    <!-- the open pore width, in the same unit as dx -->
      <reference_length>%(refl)s</reference_length>
          <!-- WHICH LENGTH the Peclet and Damkohler numbers below are taken against.
               sample          = the whole inlet-to-outlet distance, nx*dx.
               characteristic  = the pore width, which is what CompLaB uses.
               Both now reach the solver. Whichever you pick is the length the
               simulation itself divides by, not merely the one the report
               prints.
               They differ by nx*dx / characteristic_length. Both are printed by
               the check command so you can convert a number quoted the other
               way. -->
      <material_numbers>
        <pore>%(pore)d</pore>                <!-- open pore, fluid -->
        <solid>%(solid)d</solid>               <!-- solid grain -->
        <bounce_back>%(wall)d</bounce_back>         <!-- no-slip wall voxel -->
      </material_numbers>
    </domain>

    <tau>%(tau)g</tau>
        <!-- LBM relaxation time. The lattice viscosity is (tau - 0.5)/3, so tau
             must be above 0.5 or the viscosity is negative. Use 0.6 to 1.0. -->
    <delta_P>%(dP)g</delta_P>
        <!-- Seed pressure drop driving the flow. Kept small on purpose: the flow
             is rescaled afterwards to hit the Peclet below, so this only has to
             be small enough to stay stable while the permeability is measured. -->
    <flow_driver>%(fdrv)s</flow_driver>
        <!-- WHAT SETS THE FLOW SPEED. This is the answer to "do I write a
             pressure gradient, or does it work from Peclet?".
             peclet    the flow is solved, then RESCALED so the Peclet number is
                       the one below. The pressure drop above still drives the
                       lattice-Boltzmann solve, but its size divides out. This is
                       what a training set wants, because Peclet is the thing
                       being swept.
             pressure  the flow is solved with the pressure drop above and is NOT
                       rescaled. Peclet is whatever that flow makes it, computed
                       from the lattice speed, tau, dx and the fluid viscosity,
                       and recorded per run. This is what you want when you are
                       reproducing one measured flow rate. -->
    <fluid_viscosity>%(nu)g</fluid_viscosity>
        <!-- m^2/s. Water at 20 C is 1e-6. Only used in pressure mode, where it
             is what turns a lattice speed into a Peclet number. -->
    <Peclet>%(pe)g</Peclet>
        <!-- advection against diffusion, u * L / D of the first chemical.
             1 means they balance. Above about 10 the front channels through the
             fast paths; below 1 diffusion fills the dead ends. -->

    <iteration>
      <ns_max_iT>%(ns_max)d</ns_max_iT>          <!-- flow-solver steps per structure -->
      <ns_converge_iT>%(ns_cv)g</ns_converge_iT>
      <ade_max_iT>%(ade_max)d</ade_max_iT>       <!-- hard cap on transport steps -->
      <ade_converge_iT>%(ade_cv)g</ade_converge_iT>
          <!-- the transport run stops when the mean donor concentration changes
               by less than this over one advective time. It is NOT a fixed step
               count on purpose: a count that is right at Peclet 50 leaves the
               field almost empty at Peclet 1. -->
      <n_snapshots>%(nsnap)d</n_snapshots>
          <!-- how many times are stored per run. They are spaced logarithmically,
               because the interesting part is the beginning. -->
    </iteration>

  </LB_numerics>

  <!-- ===================== THE CHEMICALS =====================
       The reaction network is fixed: substrate0 is the electron donor,
       substrate1 the acceptor, substrate2 the product, substrate3 the biomass.
       Chemicals beyond the fourth are read and reported but not simulated.
       ========================================================= -->
  <chemistry>
    <number_of_substrates>%(nsub)d</number_of_substrates>
    <rate_unit>%(runit)s</rate_unit>       <!-- per_second or per_day; applies to every rate below -->
%(substrates)s
  </chemistry>

  <!-- ===================== THE REACTIONS =====================
       biotic:   donor + (stoichiometry) acceptor  ->  product + (yield) biomass
                 rate = vmax * biomass * donor/(ks_donor+donor)
                                       * acceptor/(ks_acceptor+acceptor)
       abiotic:  a first-order loss of the product at k_surf
       ========================================================= -->
  <kinetics>
    <vmax>%(vmax)g</vmax>              <!-- maximum specific rate, per unit biomass -->
    <ks_donor>%(ksd)g</ks_donor>          <!-- mol/L. Zero makes the rate first order in the donor. -->
    <ks_acceptor>%(ksa)g</ks_acceptor>    <!-- mol/L -->
    <yield>%(yld)g</yield>             <!-- mol biomass per mol donor -->
    <decay>%(dec)g</decay>             <!-- first-order biomass decay -->
    <stoichiometry>%(stoi)g</stoichiometry>  <!-- mol acceptor consumed per mol donor -->
    <k_surf>%(ksurf)g</k_surf>            <!-- the abiotic rate constant -->

    <!-- ===== WHICH REACTIONS RUN =====
         Two different things, kept apart on purpose.
         BIOTIC is microbially mediated: it needs microbes present, obeys Monod
         kinetics in both the donor and the acceptor, grows biomass with a yield
         and loses it to decay.
             donor + (stoichiometry) acceptor -> product + (yield) biomass
             rate = vmax * biomass * donor/(ks_donor+donor)
                                   * acceptor/(ks_acceptor+acceptor)
         ABIOTIC is purely chemical and no microbes take any part in it. It is
         first order in the product, which is what its Damkohler number is
         defined against, and it consumes the product.
             product -> consumed        rate = k_surf * product
         ============================== -->
    <reaction_mode>%(rmode)s</reaction_mode>
        <!-- biotic only | abiotic only | both.
             THIS is the tag that decides which reactions run, and it is the one
             written back out. An earlier template wrote biotic_enabled and
             abiotic_enabled instead: two booleans derived from this one, which
             nothing reads, so saving a file and loading it again silently
             reset the reaction mode to the default. -->
    <abiotic_reactant>%(areact)s</abiotic_reactant>
        <!-- auto | donor | product. auto means the product when there is a
             biotic reaction to make one, and the donor when there is not. -->
    <abiotic_product>%(aprod)s</abiotic_product>
        <!-- product | none.
             product = the reactant BECOMES the product, one mole for one, so
                       an abiotic-only run is a real reaction: the donor falls,
                       the product rises, and the two sum to the feed. This is
                       what a mineral dissolution or a surface-catalysed
                       transformation does.
             none    = the reactant is consumed and nothing tracked appears.
                       Right for a terminal loss: a solute that precipitates
                       out, or degrades to something this run does not follow. -->
    <abiotic_surface_only>%(abiosurf)s</abiotic_surface_only>
        <!-- true: the abiotic reaction happens only on pore voxels that touch a
             grain surface, which is what a mineral-surface reaction does.
             false: everywhere in the water. -->
    <damkohler_abiotic>%(da2)g</damkohler_abiotic>

    <biofilm_code>%(bfcode)s</biofilm_code>
        <!-- the material number for attached biofilm, or "none". A voxel with
             this number is not solid: chemicals diffuse through it, at
             diffusion_in_biofilm rather than the pore value. -->
    <biofilm_layers>%(bflayers)d</biofilm_layers>
        <!-- how many voxels thick to grow the coating on every grain surface. -->

    <biomass_mode>%(bmode)s</biomass_mode>
        <!-- sessile     attached biofilm. It reacts where it is and never moves.
             planktonic  dissolved cells. They are advected and diffused like any
                         other chemical, and in a through-flowing sample most of
                         them leave: measured on a 60 by 32 domain, mean biomass
                         fell from 0.100 to 0.019 with transport on and to 0.071
                         with it off, so about three quarters of the loss was
                         washout rather than decay. -->
    <biomass_coupled>%(bcoup)s</biomass_coupled>
        <!-- false: the biotic rate holds the biomass at its starting value.
             That is not an oversight. The Damkohler number is defined as
             vmax * B0 * L / (Ac0 * u), so B0 is already inside it, and letting
             B vary as well would count it twice.
             true: the rate follows the biomass actually present, which is the
             honest choice when the biomass changes a lot over a run. -->

    <use_physical_kinetics>%(usephys)s</use_physical_kinetics>
        <!-- WHERE THE DAMKOHLER NUMBERS COME FROM. This is the one switch in
             this file that changes what is simulated rather than how.

             false  the Damkohler numbers are set directly, below or by the
                    <sweep> block, and vmax / k_surf / decay are read and
                    reported but not used. This is what you want for a
                    TRAINING SET, which has to span the dimensionless space
                    rather than sit at one system's point in it.

             true   the Damkohler numbers are computed from vmax, k_surf and
                    decay. This is what you want when you are modelling one
                    named system and the answer has to be that system's.

             Before switching it on, run the check command and read the
             Damkohler your kinetics imply. At a sample of tens of micrometres with ordinary
             diffusion coefficients it is usually far below 1, because
             diffusion crosses the sample in about a second and no microbial
             rate competes with that. A column reaches Damkohler 1 by being
             centimetres long, not by reacting faster. -->
    <damkohler>%(da)g</damkohler>
        <!-- used when use_physical_kinetics is false and no sweep is given -->
    <decay_lattice>%(declat)g</decay_lattice>
        <!-- biomass decay when use_physical_kinetics is false, in the solver's
             own time units. 0.006 is what the generators used before this file
             existed, and changing it changes every dataset made from here on. -->
  </kinetics>

  <!-- ===================== SWEEPING =====================
       A training set needs MANY conditions, not one. Uncomment this block and
       the generator varies the conditions across the runs instead of using the
       single values above.

       For each quantity you have TWO ways to say what you want, and you pick
       one. Give a RANGE and the generator draws from it at random, spaced
       logarithmically -- log-uniform rather than uniform because these
       quantities span decades and a uniform draw would put nine tenths of the
       runs in the top decade. Give a LIST and it uses exactly those numbers
       and nothing else, which is what you want when you are reproducing
       published conditions or answering a reviewer who asked for Pe = 1, 10
       and 100.

       If you give both, the list wins and the range is ignored.

       POROSITY works the same way. A range means every structure is built at
       its own random porosity inside it; a list means structures are built at
       exactly those porosities, cycling through the list in order, so ten
       structures over a list of two porosities gives five of each.

       When you list values, the generator makes every combination of them.
       Three Peclet numbers and two Damkohler numbers is six conditions per
       structure, and 'runs per structure' is set to six for you unless you
       ask for a different number.
       ==================================================== -->
  <!--
  <sweep>
    <peclet_min>%(pemin)g</peclet_min>
    <peclet_max>%(pemax)g</peclet_max>
    <damkohler_min>%(damin)g</damkohler_min>
    <damkohler_max>%(damax)g</damkohler_max>
    <damkohler_abiotic_min>%(damin)g</damkohler_abiotic_min>
    <damkohler_abiotic_max>%(damax)g</damkohler_abiotic_max>
    <porosity_min>0.35</porosity_min>
    <porosity_max>0.50</porosity_max>

    ... or, instead of any of the pairs above, the exact numbers:

    <peclet_values>1, 10, 50</peclet_values>
    <damkohler_values>0.1, 1, 10</damkohler_values>
    <damkohler_abiotic_values>0.5, 5</damkohler_abiotic_values>
    <porosity_values>0.35, 0.40, 0.45</porosity_values>
  </sweep>
  -->

</parameters>
"""


# --------------------------------------------------------------------------
def self_test():
    """Checks the conversion, not the implementation of it."""
    ok = True

    def check(name, cond, note=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %-58s %s %s" % (name, "PASS" if cond else "FAIL", note))

    s = Settings()
    check("the defaults have nothing wrong with them", s.complaints() == [],
          str(s.complaints()[:1]))

    # 1. THE ONE THAT MATTERS. A default Settings must reproduce, exactly, the
    #    constants prtlb_2d.py / prtlb_3d.py used to hold. If it does not, every dataset made
    #    before today and every dataset made after carry the same labels and
    #    different physics.
    c = s.chemistry(4)
    d = default_chemistry(4)
    same = (np.allclose(np.nan_to_num(c.inlet, nan=-1),
                        np.nan_to_num(d.inlet, nan=-1), atol=1e-6) and
            np.allclose(c.c_init, d.c_init, atol=1e-6) and
            np.allclose(c.d_rel, d.d_rel, atol=1e-6) and
            list(c.immobile) == list(d.immobile) and
            abs(c.ks_donor - d.ks_donor) < 1e-9 and
            abs(c.ks_acceptor - d.ks_acceptor) < 1e-6 and
            abs(c.alpha - d.alpha) < 1e-6 and
            abs(c.yield_ - d.yield_) < 1e-9 and
            abs(c.decay_for(32) - d.decay_for(32)) < 1e-12)
    check("the default settings reproduce the old hard-coded numbers", same,
          "inlet %s c0 %s ks %.4f alpha %.4f"
          % (np.round(c.inlet, 3), np.round(c.c_init, 4), c.ks_acceptor,
             c.alpha))

    # 2. the stoichiometry is a ratio of feeds, not 1
    check("acceptor per donor is 1 * Ac0/A0 = 0.4",
          abs(s.chemistry(4).alpha - 0.4) < 1e-6)

    # 3. the two Damkohler conventions differ by exactly Peclet
    g = s.groups()
    check("Da diffusive = Da advective * Peclet",
          abs(g["da_bio_diff"] - g["da_bio_physical"] * g["pe"]) < 1e-9 * max(
              1.0, g["da_bio_diff"]),
          "%.5g vs %.5g" % (g["da_bio_diff"], g["da_bio_physical"] * g["pe"]))

    # 3a2. EVERY SETTING THAT CLAIMS TO CONTROL THE SOLVER MUST REACH IT.
    #      This has now been the same bug three times: tau and delta_P were
    #      read, printed in the report, and never passed to the flow solver;
    #      then the two stopping rules were read, printed, and never passed to
    #      the transport solver, so raising the step cap because runs were
    #      being truncated did nothing at all. A setting that is displayed and
    #      ignored is worse than one that does not exist, because you spend
    #      the afternoon believing you changed something.
    import inspect
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import build_dataset_2d as m2
        import build_dataset_3d as m3
        # The FLOW force is no longer passed as cfg.delta_P directly: in
        # Peclet mode the seed is lowered when it would run the rock above
        # Mach 0.05, which is outside the range lattice Boltzmann is valid in.
        # So the chain to check is that delta_P is where the force STARTS and
        # that the lowered value is what the solver is handed.
        want_both = ["_force = float(cfg.delta_P)", "force=_force",
                     "tau_lb=cfg.tau", "tol=float(cfg.ns_converge)",
                     "nit=_nit", "int(cfg.ns_max_iT)",
                     "tol=cfg.ade_converge", "max_steps=int(cfg.ade_max_iT)"]
        for mod, name, wanted in ((m2, "build_dataset_2d", want_both),
                                  (m3, "build_dataset_3d", want_both)):
            src = inspect.getsource(mod)
            missing = [w for w in wanted if w not in src]
            check("%s hands every stopping rule to the solvers" % name,
                  not missing, "missing: %s" % ", ".join(missing))
    except Exception as e:                                    # pragma: no cover
        check("the generators could be inspected", False, repr(e))

    # 3a3. THE PRESSURE GRADIENT IS DERIVED FROM THE PECLET YOU ASK FOR.
    #      Stokes flow is linear, so this has an exact answer and a testable
    #      one: doubling the wanted Peclet must exactly double the pressure,
    #      and feeding the derived pressure back must return the Peclet you
    #      started from.
    t = Settings()
    u_lat = 6.24e-3
    p10 = t.pressure_for_peclet(u_lat, 10.0)
    p20 = t.pressure_for_peclet(u_lat, 20.0)
    check("twice the Peclet needs exactly twice the pressure",
          abs(p20 - 2 * p10) < 1e-12 * max(1.0, p20),
          "%.4g and %.4g" % (p10, p20))
    t2 = Settings()
    t2.delta_P = p10
    check("and driving at that pressure gives back that Peclet",
          abs(t2.peclet_from_flow(u_lat * p10 / Settings().delta_P) - 10.0)
          < 1e-6,
          "%.6f" % t2.peclet_from_flow(u_lat * p10 / Settings().delta_P))
    tight = Settings()
    check("a slower rock needs a stronger push for the same Peclet",
          tight.pressure_for_peclet(u_lat / 2, 10.0) >
          tight.pressure_for_peclet(u_lat, 10.0))

    # 3b. and the switch does what it says
    t = Settings()
    t.damkohler = 3.0
    check("with settings_and_units kinetics off, Damkohler is the one you set",
          abs(t.groups()["da_bio"] - 3.0) < 1e-12)
    t.use_physical_kinetics = True
    check("with it on, Damkohler comes from vmax",
          abs(t.groups()["da_bio"] - t.groups()["da_bio_physical"]) < 1e-30)
    check("and the decay then comes from the decay rate, not the default",
          t.chemistry().decay_for(32) != Settings().chemistry().decay_for(32))

    # 4. agreement with complab_campaign.py, which builds the CompLaB runs from the
    #    same groups. A disagreement here means a Python run and a CompLaB run
    #    labelled Pe=10 are not the same experiment.
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import complab_campaign
        t = Settings()
        t.reference_length = "characteristic"
        t.species[0].left_value = complab_campaign.AC0
        t.species[1].left_value = complab_campaign.A0
        t.species[3].initial = complab_campaign.B0
        t.species[0].d_pore = complab_campaign.D_AC
        t.species[2].d_pore = complab_campaign.D_P
        t.stoichiometry = complab_campaign.ALPHA
        gg = t.groups()
        want = complab_campaign.derive(gg["pe"], gg["da_bio_diff"], gg["da_abio_diff"],
                               gg["ks_ac_norm"], gg["ks_a_norm"], gg["y_norm"],
                               t.characteristic_length, t.dx, t.tau)
        check("the characteristic length agrees with complab_campaign.py",
              abs(want["L_m"] - gg["L_m"]) < 1e-15,
              "%.4g vs %.4g" % (want["L_m"], gg["L_m"]))
        # complab_campaign.py works in per-second, the settings file in per-day by
        # default, so the comparison has to carry the unit with it. Forgetting
        # that is a factor of 86400, which is exactly the kind of error this
        # cross-check exists to catch.
        vmax_si = t.vmax * t.rate_factor
        ksurf_si = t.k_surf * t.rate_factor
        check("vmax round-trips through complab_campaign.py's derivation",
              abs(want["vmax"] - vmax_si) < 1e-9 * max(1.0, vmax_si),
              "%.6g vs %.6g per second" % (want["vmax"], vmax_si))
        check("k_surf round-trips through complab_campaign.py's derivation",
              abs(want["ksurf"] - ksurf_si) < 1e-9 * max(1.0, ksurf_si),
              "%.6g vs %.6g per second" % (want["ksurf"], ksurf_si))
    except Exception as e:                                # pragma: no cover
        check("complab_campaign.py cross-check ran", False, repr(e))

    # 5. a template written and read back is the same settings
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.xml")
        with open(p, "w") as f:
            f.write(s.to_xml())
        back = Settings.from_xml(p)
        check("a written template reads back unchanged",
              back.names == s.names and abs(back.tau - s.tau) < 1e-12
              and abs(back.vmax - s.vmax) < 1e-18
              and abs(back.species[1].left_value -
                      s.species[1].left_value) < 1e-18
              and back.species[3].immobile,
              "%s tau=%g vmax=%g" % (back.names, back.tau, back.vmax))
        check("a written template has nothing wrong with it",
              back.complaints() == [], str(back.complaints()[:1]))

        # and JSON round-trips too
        q = os.path.join(td, "t.json")
        with open(q, "w") as f:
            f.write(s.to_json())
        bj = Settings.from_json(q)
        check("a written JSON file reads back unchanged",
              bj.names == s.names and abs(bj.yield_ - s.yield_) < 1e-15)

    # 6. mistakes are caught, one by one
    for note, mutate in (
        ("tau below 0.5", lambda x: setattr(x, "tau", 0.4)),
        ("an unfed donor", lambda x: setattr(x.species[0], "left_type",
                                             "Neumann")),
        ("a misspelled boundary type",
         lambda x: setattr(x.species[1], "left_type", "dirchlet")),
        ("a pore wider than the sample",
         lambda x: setattr(x, "characteristic_length", 1e6)),
        ("two chemicals with one name",
         lambda x: setattr(x.species[1], "name", "Ac")),
        ("immobile with a diffusion coefficient",
         lambda x: setattr(x.species[3], "d_pore", 1e-9)),
    ):
        t = Settings()
        mutate(t)
        check("caught: " + note, t.complaints() != [])

    # 7. a real CompLaB.xml, if one is lying about, is readable rather than
    #    fatal -- that is the whole promise of the reader
    t = Settings()
    t.species = [Species("C%d" % i) for i in range(95)]
    check("95 chemicals can be cut down to 4 rather than crashing",
          len(t.chemistry(4).names) == 4)
    check("and can equally be carried in full",
          len(t.chemistry(95).names) == 95)

    # 7b. a fifth chemical is a tracer: transported, but not reacting
    t = Settings()
    t.species.append(Species("Tracer", d_pore=1.0e-9,
                             left_type="Dirichlet", left_value=1.0e-3))
    c5 = t.chemistry()
    check("a fifth chemical is carried and fed", len(c5.names) == 5
          and abs(c5.inlet[4] - 1.0) < 1e-6,
          "%s inlet %s" % (c5.names, c5.inlet))

    # 7c. the plus and minus buttons: n_chemicals resizes the list, and the
    #     per-chemical settings that follow have somewhere to land
    t = Settings()
    t.apply_override("n_chemicals", "6")
    check("n_chemicals grows the list", len(t.species) == 6,
          str(t.names))
    t.apply_override("species.5.name", "Bromide")
    t.apply_override("species.5.d_pore", "2.0e-9")
    check("and the new chemical can then be configured",
          t.species[5].name == "Bromide"
          and abs(t.species[5].d_pore - 2.0e-9) < 1e-18)
    t.apply_override("n_chemicals", "2")
    check("n_chemicals shrinks it again", len(t.species) == 2, str(t.names))
    t.apply_override("n_chemicals", "0")
    check("and never below one", len(t.species) == 1)

    # 8. per-species diffusivity actually reaches the solver
    t = Settings()
    t.species[1].d_pore = 5.0e-10
    check("a halved acceptor diffusivity arrives as D/D_ref = 0.5",
          abs(t.chemistry(4).d_rel[1] - 0.5) < 1e-6,
          "%.4f" % t.chemistry(4).d_rel[1])

    # 8b. the warnings fire on the things that run but mean nothing
    for note, mutate in (
        ("a biomass that starts at zero",
         lambda x: setattr(x.species[3], "initial", 0.0)),
        ("a mobile biomass",
         lambda x: setattr(x.species[3], "immobile", False)),
        ("a half-saturation far above the feed",
         lambda x: setattr(x, "ks_acceptor", 1.0)),
        ("a two-dimensional grid handed to the 3D builder",
         lambda x: setattr(x, "nz", 1)),
    ):
        t = Settings()
        mutate(t)
        check("warned: " + note, t.warnings() != [] and t.complaints() == [])
    check("a good file warns about nothing", Settings().warnings() == [],
          str(Settings().warnings()[:1]))

    # 8c. an immobile product does not produce a Damkohler of 1e16
    t = Settings()
    t.species[2].d_pore = 0.0
    t.species[2].immobile = True
    check("an immobile product falls back to the donor diffusivity",
          t.groups()["da_abio_physical"] < 1e3,
          "%.4g" % t.groups()["da_abio_physical"])

    # 9. changing a feed concentration changes the stoichiometric ratio, which
    #    is the whole reason ALPHA was not 1
    t = Settings()
    t.species[1].left_value = 2.0e-3     # same as the donor now
    check("equal feeds give an acceptor-per-donor ratio of 1",
          abs(t.chemistry(4).alpha - 1.0) < 1e-6,
          "%.4f" % t.chemistry(4).alpha)

    # ---- units, and the things that used to be read and then dropped -----
    #
    # Every one of these is a setting the file accepted, validated, printed a
    # helpful note about, and then failed to act on. A setting that does
    # nothing is worse than a missing one: the file says the run was
    # configured a way it was not.

    # a. the characteristic length is a length, converted once
    u = Settings(); u.dx = 2.0; u.unit = "um"; u.characteristic_length = 16.0
    check("the characteristic length converts by the UNIT, not by dx",
          abs(u.L_char_m - 16e-6) < 1e-18, "%.4g m" % u.L_char_m)
    u.reference_length = "characteristic"
    check("the reference length in voxels is pore width / dx",
          abs(u.l_ref_voxels - 8.0) < 1e-12, "%.4g voxels" % u.l_ref_voxels)
    u.reference_length = "sample"
    check("and is None for the sample convention, so the solver uses the "
          "domain it was handed", u.l_ref_voxels is None)

    # b. the reference length REACHES THE SOLVER
    import numpy as _np
    from prtlb_2d import solve_adr as _solve, PORE as _P
    _g = _np.full((40, 8), _P, _np.uint8)
    _v = _np.zeros((3, 40, 8), _np.float32); _v[0] = 1.0
    c_s = Settings(); c_s.reference_length = "sample"
    c_c = Settings(); c_c.reference_length = "characteristic"
    c_c.characteristic_length = 20.0; c_c.dx = 1.0
    i_s, i_c = {}, {}
    _solve(_g, _v, pe=30.0, da=1.0, n_t=2, n_species=4,
           chem=c_s.chemistry(), info=i_s)
    _solve(_g, _v, pe=30.0, da=1.0, n_t=2, n_species=4,
           chem=c_c.chemistry(), info=i_c)
    check("reference_length reaches the solver and changes what it does",
          i_s["l_ref_voxels"] == 40.0 and i_c["l_ref_voxels"] == 20.0,
          "sample %g voxels, characteristic %g voxels"
          % (i_s["l_ref_voxels"], i_c["l_ref_voxels"]))

    # c. the abiotic Damkohler carries the PRODUCT's diffusivity, so it cannot
    #    be the diffusive one divided by a Peclet defined on the donor
    k = Settings()
    k.use_physical_kinetics = True
    k.reaction_mode = "both"
    k.k_surf = 1.0
    gk = k.groups()
    exp = k.k_surf * k.rate_factor * gk["L_m"] / gk["u_mean_implied"]
    check("the abiotic Damkohler is k_surf L / u, with no stray "
          "diffusivity ratio", abs(gk["da_abio"] - exp) < 1e-9 * max(exp, 1),
          "%.6g against %.6g" % (gk["da_abio"], exp))
    k.species[2].d_pore = k.donor.d_pore * 0.5
    check("and does not move when the PRODUCT's diffusivity is edited",
          abs(k.groups()["da_abio"] - gk["da_abio"]) < 1e-12 * max(exp, 1),
          "%.6g" % k.groups()["da_abio"])

    # d. chemistry() answers a question without changing the answer
    m = Settings(); m.biomass_mode = "sessile"
    m.species[3].immobile = False
    before = m.species[3].immobile
    m.chemistry()
    check("chemistry() does not write back into the chemical table",
          m.species[3].immobile == before,
          "immobile was %s, is %s" % (before, m.species[3].immobile))
    check("so the warning about a mobile biomass can still be reached",
          any("immobile box" in t for t in m.warnings()))
    check("and the sessile choice still reaches the solver",
          bool(m.chemistry().immobile[3]))

    # e. every setting survives a round trip through the file it writes
    import tempfile
    import os as _os
    r = Settings()
    r.reaction_mode = "both"; r.abiotic_reactant = "donor"
    r.biofilm_code = 3; r.biofilm_layers = 2
    r.biomass_mode = "planktonic"; r.reference_length = "characteristic"
    fd, pth = tempfile.mkstemp(suffix=".xml"); _os.close(fd)
    with open(pth, "w") as fh:
        fh.write(r.to_xml())
    back = Settings.from_xml(pth)
    _os.unlink(pth)
    for fld in ("reaction_mode", "abiotic_reactant", "biofilm_code",
                "biofilm_layers", "biomass_mode", "reference_length",
                "ns_max_iT"):
        check("a saved file reloads with the same %s" % fld,
              getattr(r, fld) == getattr(back, fld),
              "%r against %r" % (getattr(r, fld), getattr(back, fld)))

    # f. CompLaB's master switch
    xml = r.to_xml().replace("</parameters>",
                             "  <simulation_mode>\n"
                             "    <enable_kinetics>false</enable_kinetics>\n"
                             "  </simulation_mode>\n</parameters>")
    fd, pth = tempfile.mkstemp(suffix=".xml"); _os.close(fd)
    with open(pth, "w") as fh:
        fh.write(xml)
    off = Settings.from_xml(pth)
    _os.unlink(pth)
    check("<enable_kinetics>false switches every rate off",
          off.damkohler == 0 and off.damkohler_abiotic == 0
          and off.vmax == 0 and off.k_surf == 0,
          "Da %g, Da_abio %g" % (off.damkohler, off.damkohler_abiotic))

    # g. the configuration that produced a whole sweep of identical runs
    ab = Settings(); ab.reaction_mode = "abiotic only"
    ab.abiotic_reactant = "product"
    check("an abiotic reaction on a product nothing makes is REFUSED",
          any("nothing ever produces it" in c for c in ab.complaints()))
    check("and the donor, which IS fed, is accepted",
          not [c for c in
               (lambda x: (setattr(x, "abiotic_reactant", "donor"), x.complaints())[1])(ab)
               if "nothing ever produces it" in c])
    ab.abiotic_reactant = "product"
    check("and the two chemicals that come out identical are named",
          any("IDENTICAL field" in w for w in ab.warnings()))
    ab2 = Settings(); ab2.reaction_mode = "abiotic only"
    ab2.abiotic_reactant = "donor"; ab2.abiotic_product = "none"
    # the abiotic reaction is now a REACTION, so the product it makes must no
    # longer be reported as a field that will come out blank
    ap = Settings(); ap.reaction_mode = "abiotic only"
    check("with the abiotic reaction making it, P is not called blank",
          not any("P is not fed" in w for w in ap.warnings()))
    ap.abiotic_product = "none"
    check("and with abiotic_product = none it is called blank again",
          any("P is not fed" in w for w in ap.warnings()))
    ap.abiotic_product = "product"
    check("the abiotic reaction is written out as donor -> product",
          any("Ac  ->  P" in ln
              for _t, lns in ap.reaction_text() for ln in lns))
    check("abiotic_product only takes the two values it documents",
          any("abiotic_product is" in c for c in
              (lambda x: (setattr(x, "abiotic_product", "banana"),
                          x.complaints())[1])(Settings())))
    r_ap = Settings(); r_ap.abiotic_product = "none"
    import tempfile as _tf, os as _o
    _fd, _p = _tf.mkstemp(suffix=".xml"); _o.close(_fd)
    open(_p, "w").write(r_ap.to_xml())
    check("and survives a round trip through the file",
          Settings.from_xml(_p).abiotic_product == "none")
    _o.unlink(_p)

    check("a chemical nothing feeds and nothing makes is called out",
          any("blank" in w and "P is not fed" in w for w in ab2.warnings()))
    check("and a biotic run, where the product IS made, is left alone",
          not any("blank" in w for w in Settings().warnings()))
    for mode, react in (("biotic only", "auto"), ("both", "auto"),
                        ("abiotic only", "donor")):
        q = Settings(); q.reaction_mode = mode; q.abiotic_reactant = react
        check("%s / %s is left alone" % (mode, react),
              not any("IDENTICAL field" in w for w in q.warnings())
              and not any("nothing ever produces it" in c
                          for c in q.complaints()))

    # g. the configuration that produced a whole sweep of identical runs
    ab = Settings(); ab.reaction_mode = "abiotic only"
    ab.abiotic_reactant = "product"
    check("an abiotic reaction on a product nothing makes is REFUSED",
          any("nothing ever produces it" in c for c in ab.complaints()))
    check("and the two chemicals that come out identical are named",
          any("IDENTICAL field" in w for w in ab.warnings()))
    ab2 = Settings(); ab2.reaction_mode = "abiotic only"
    ab2.abiotic_reactant = "donor"; ab2.abiotic_product = "none"
    # the abiotic reaction is now a REACTION, so the product it makes must no
    # longer be reported as a field that will come out blank
    ap = Settings(); ap.reaction_mode = "abiotic only"
    check("with the abiotic reaction making it, P is not called blank",
          not any("P is not fed" in w for w in ap.warnings()))
    ap.abiotic_product = "none"
    check("and with abiotic_product = none it is called blank again",
          any("P is not fed" in w for w in ap.warnings()))
    ap.abiotic_product = "product"
    check("the abiotic reaction is written out as donor -> product",
          any("Ac  ->  P" in ln
              for _t, lns in ap.reaction_text() for ln in lns))
    check("abiotic_product only takes the two values it documents",
          any("abiotic_product is" in c for c in
              (lambda x: (setattr(x, "abiotic_product", "banana"),
                          x.complaints())[1])(Settings())))
    r_ap = Settings(); r_ap.abiotic_product = "none"
    import tempfile as _tf, os as _o
    _fd, _p = _tf.mkstemp(suffix=".xml"); _o.close(_fd)
    open(_p, "w").write(r_ap.to_xml())
    check("and survives a round trip through the file",
          Settings.from_xml(_p).abiotic_product == "none")
    _o.unlink(_p)

    check("a chemical nothing feeds and nothing makes is called out",
          any("blank" in w and "P is not fed" in w for w in ab2.warnings()))
    check("and a biotic run, where the product IS made, is left alone",
          not any("blank" in w for w in Settings().warnings()))
    for mode, react in (("biotic only", "auto"), ("both", "auto"),
                        ("abiotic only", "donor")):
        q = Settings(); q.reaction_mode = mode; q.abiotic_reactant = react
        check("%s / %s is left alone" % (mode, react),
              not any("IDENTICAL field" in w for w in q.warnings())
              and not any("nothing ever produces it" in c
                          for c in q.complaints()))

    # h. THE TWO GENERATORS MUST TREAT A BAD RUN THE SAME WAY. They did not:
    #    faced with the identical complaint, 3D stopped before writing and 2D
    #    wrote the file and printed a warning under it, so the same mistake
    #    left a dataset on disk in one dimension and none in the other.
    try:
        import inspect
        import build_dataset_2d as _m2
        import build_dataset_3d as _m3
        for _mod, _nm in ((_m2, "build_dataset_2d"), (_m3, "build_dataset_3d")):
            _src = inspect.getsource(_mod)
            _i_stop = _src.find("physics complaint(s)")
            _i_write = _src.find("h5py.File(a.out")
            check("%s refuses to write a file with physics complaints" % _nm,
                  0 < _i_stop < _i_write,
                  "stop at %d, write at %d" % (_i_stop, _i_write))
    except Exception as _e:                                   # pragma: no cover
        check("both generators could be inspected", False, repr(_e))

    print()
    print("SELF TEST PASSED" if ok else "SELF TEST FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Write, read and check the settings_and_units settings the Python "
                    "generators run from.")
    ap.add_argument("--template", metavar="FILE",
                    help="write a fully commented settings file here")
    ap.add_argument("--json", action="store_true",
                    help="write the template as JSON instead of XML")
    ap.add_argument("--check", metavar="FILE",
                    help="read a settings file and report everything it means")
    ap.add_argument("--capabilities", action="store_true",
                    help="what this solver does and does not do, against CompLaB")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if a.capabilities:
        print(CAPABILITIES)
        return 0
    if a.template:
        s = Settings()
        text = s.to_json() if a.json else s.to_xml()
        d = os.path.dirname(os.path.abspath(a.template))
        os.makedirs(d, exist_ok=True)
        with open(a.template, "w") as f:
            f.write(text)
        print("wrote %s" % a.template)
        print()
        print("Open it, change the numbers you care about, then check it with")
        print("  python settings_and_units.py --check %s" % a.template)
        print("and run it with")
        print("  python build_dataset_3d.py --settings %s --out data3d.h5"
              % a.template)
        return 0
    if a.check:
        s = Settings.from_file(a.check)
        good = s.report()
        return 0 if good else 1

    ap.print_help()
    print()
    print(CAPABILITIES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
