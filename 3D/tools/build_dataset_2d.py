#!/usr/bin/env python3
"""
build_dataset_2d.py — a NATIVE 2D dataset, the way Jung's PRT-DeepONet works.

This is not the 2D-to-3D transfer path.  That is build_transfer_set_2d_to_3d.py, which EXTRUDES a
2D domain into a 3D volume so the 3D network can train on it.  This file is the
other thing: a genuinely two-dimensional problem, solved in two dimensions,
trained by a two-dimensional network.  Use it when you want to reproduce or
extend the published 2D work, or when you want a cheap sandbox before spending
sixteen hours on a 3D run.

WHY IT IS WORTH HAVING BOTH
    A 2D run is roughly fifty times cheaper than the 3D one.  Every question
    about the METHOD rather than about the rock -- does this rate law train at
    all, is this Damkohler range sensible, does the geodesic field beat the
    Euclidean one -- can be answered in 2D in minutes and then confirmed in 3D.

WHAT IT PRODUCES
    Exactly the same HDF5 layout as the 3D pipeline, with nz = 1.  That is
    deliberate: one dataset format, one Dataset class, one training script, one
    evaluation script and one viewer serve both dimensions.  The network notices
    nz = 1 and switches its encoder from 3D convolutions to 2D ones, and the
    trunk drops the z column, giving Jung's original (x, y, t, GDF).

THE PHYSICS
    Flow       D2Q9 BGK lattice Boltzmann, body-force driven along +x,
               halfway bounce-back on the solid, so the no-slip wall sits at
               the mid-link, half a voxel outside the last fluid node. That is
               where Palabos puts it, and therefore where CompLaB puts it.
               The 2D counterpart of the D3Q19 solver used for the 3D runs.
    Transport  explicit advection with upwind differencing, plus diffusion,
               plus a first-order reaction.
    Geometry   thresholded correlated Gaussian field, or Jung's own domains.

    Note the porosity range.  In 3D a thresholded Gaussian field stops
    percolating near porosity 0.20; in 2D the threshold is near 0.59.  The
    default range here starts at 0.55 for that reason, and every domain is
    still checked inlet-to-outlet individually.

USAGE
    # our own domains
    python build_dataset_2d.py --out data2d.h5 --n-geom 40 --n-sets 6

    # Jung's actual domains, our chemistry
    python build_dataset_2d.py --out data2d.h5 --jung-dir ../../2D --limit 200

Then train exactly as in 3D:
    python ../model/train.py --data data2d.h5 --out runs/2d
"""

import argparse
import os
import sys
import time
import numpy as np
import h5py
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prtlb_2d import stokes_d2q9                            # noqa: E402
from prtlb_2d import (SOLID, WALL, PORE, INLET, KS_A, ALPHA, Y, KD, B0,
                      ATTACHED, solve_adr, check_physics,      # noqa: F401
                      keep_inlet_connected, assert_finite_distance,
                      coat_with_biofilm)
# write_vti_and_png.py was removed from the project. It was only ever used by
# --save-vti and --save-png, so a hard import at load time turned a missing
# optional writer into a crash on startup for every run. Import it if it is
# there; if it is not, the two flags say so and the dataset is still built.
try:
    import write_vti_and_png                                              # noqa: E402
except ImportError:
    write_vti_and_png = None
from settings_and_units import (add_settings_arguments,                  # noqa: E402
                      settings_from_args)


# ===========================================================================
#  YOUR NUMBERS.  EDIT THIS BLOCK, SAVE, AND RUN THE FILE.
# ===========================================================================
#  This is the place to put your own settings_and_units inputs if you would rather work
#  in the code than in a settings file or in the window. It is the same set of
#  quantities CompLaB's XML holds, under the same names, and everything you
#  leave out keeps its default -- so you can delete any line you have no
#  opinion about.
#
#  UNITS, once:
#      lengths          micrometres  (dx, characteristic_length)
#      concentrations   mol/L        (so 2e-3 is 2 mM)
#      diffusion        m^2/s        (so 1e-9 is a small ion in water)
#      rates            per day, unless you change rate_unit to per_second
#      tau, delta_P     lattice units, exactly as in CompLaB
#
#  To see what these numbers imply before you spend hours on them:
#      python settings_and_units.py --check settings_example.xml
#  and to check YOUR edited block, run this script with --show-settings.
#
#  If you pass --settings FILE on the command line, that file replaces this
#  block entirely, and the run says so. Individual --set NAME=VALUE arguments,
#  and the boxes on the window's page, win over both.
# ---------------------------------------------------------------------------

MY_SETTINGS = dict(
    # ---- the sample --------------------------------------------------------
    dx                    = 1.0,        # voxel edge length, micrometres
    unit                  = "um",       # um, mm or m
    characteristic_length = 16.0,       # the open pore width, same unit as dx
    reference_length      = "sample",   # "sample" (nx*dx, what this solver
                                        # uses) or "characteristic" (the pore
                                        # width, what CompLaB uses). They
                                        # differ by nx*dx/characteristic_length
                                        # and BOTH are printed, so a Peclet
                                        # quoted either way can be compared.

    # ---- the flow ----------------------------------------------------------
    tau      = 0.8,        # LBM relaxation time. Viscosity is (tau-0.5)/3, so
                           # it must exceed 0.5. Use 0.6 to 1.0.
    delta_P  = 1.5e-4,     # the pressure drop driving the flow

    #  DO WE WRITE A PRESSURE GRADIENT, OR DOES IT WORK FROM PECLET?
    #  Both, and this chooses which.
    #
    #      "peclet"    the flow is solved with delta_P above, then RESCALED so
    #                  that the Peclet number is the one being swept. The size
    #                  of delta_P divides out, so it only has to be small
    #                  enough to keep the lattice-Boltzmann solve stable. This
    #                  is what a TRAINING SET wants, because Peclet is the
    #                  thing being varied.
    #
    #      "pressure"  the flow is solved with delta_P and NOT rescaled. The
    #                  Peclet number is whatever that flow makes it, computed
    #                  per structure from the lattice speed, tau, dx and the
    #                  fluid viscosity, and recorded. Different rocks have
    #                  different permeability, so they come out at different
    #                  Peclet -- which is exactly what happens in a real core
    #                  holder at one pump setting.
    flow_driver     = "peclet",
    fluid_viscosity = 1.0e-6,   # m^2/s, water at 20 C. Only used in pressure
                                # mode, where it turns a lattice speed into a
                                # Peclet number.

    # ---- WHICH REACTIONS RUN ----------------------------------------------
    #  Two different things, kept apart on purpose.
    #
    #  BIOTIC is microbially mediated. It needs microbes present, obeys Monod
    #  kinetics in BOTH the donor and the acceptor, grows biomass with a yield
    #  and loses it to decay:
    #
    #      donor + (stoichiometry) acceptor  ->  product + (yield) biomass
    #      rate = vmax * biomass * donor/(ks_donor + donor)
    #                            * acceptor/(ks_acceptor + acceptor)
    #
    #  ABIOTIC is purely chemical. No microbes take any part in it. It is
    #  first order in the product, which is what its Damkohler number is
    #  defined against, and it consumes the product:
    #
    #      product  ->  consumed          rate = k_surf * product
    #
    #  ONE choice, not two switches:
    #      "biotic only"    microbes and nothing else. The default.
    #      "abiotic only"   pure chemistry, no microbes anywhere.
    #      "both"           the two together, each with its own Damkohler.
    #  The abiotic reaction is off by default because it did not exist in this
    #  code until recently, and turning it on by default would change every
    #  dataset ever built here while the files still claimed to be comparable.
    reaction_mode        = "biotic only",
    abiotic_surface_only = False,   # True: only on pore voxels touching a
                                    # grain, which is what a mineral-surface
                                    # reaction does. False: everywhere.
    damkohler_abiotic    = 1.0,

    #  SESSILE OR PLANKTONIC.
    #      "sessile"     attached biofilm. Reacts where it is, never moves.
    #      "planktonic"  dissolved cells, advected and diffused like any other
    #                    chemical. In a through-flowing sample most of them
    #                    leave: measured on a 60 by 32 domain, mean biomass
    #                    fell 0.100 -> 0.019 with transport on and 0.100 ->
    #                    0.071 with it off, so about three quarters of the
    #                    loss was washout rather than decay.
    biomass_mode = "sessile",

    #  WHERE THE ATTACHED BIOMASS LIVES. CompLaB gives each attached pool its
    #  own MATERIAL NUMBER, so biofilm occupies named voxels in the geometry
    #  and the biomass is seeded only there. Set a code here and you get the
    #  same: the generator coats the grain surfaces with that code, those
    #  voxels stay part of the pore space, chemicals diffuse through them at
    #  their in_biofilm coefficient rather than their pore one, and the
    #  starting biomass goes there and nowhere else.
    #
    #  What you do NOT get is the second half of what CompLaB does: no voxel
    #  ever becomes biofilm or solid during a run, there is no cellular
    #  automaton spreading it, and the flow is never re-solved. The pore space
    #  is fixed for the whole simulation. For clogging, flow rerouting or
    #  precipitation, use CompLaB.
    #
    #  None means no biofilm voxels at all, and the biomass starts uniformly
    #  through the pore space. That is what every dataset built before this
    #  existed used.
    biofilm_code   = None,
    biofilm_layers = 1,

    #  Does the biotic rate follow the biomass that is actually there? By
    #  default no, and that is not an oversight. The Damkohler number is
    #  defined as vmax * B0 * L / (Ac0 * u), so B0 is already inside it, and
    #  letting B vary as well would count it twice. Turn it on when the
    #  biomass changes a lot over a run and you want the feedback.
    biomass_coupled = False,

    # ---- the rate constants ------------------------------------------------
    rate_unit     = "per_day",
    vmax          = 1.0,      # maximum specific rate, per unit biomass
    ks_donor      = 0.0,      # mol/L. Zero means first order in the donor.
    ks_acceptor   = 7.5e-4,   # mol/L
    yield_        = 0.04,     # mol biomass per mol donor
    decay         = 0.05,     # first-order biomass decay
    stoichiometry = 1.0,      # mol acceptor consumed per mol donor
    k_surf        = 1.0,      # the abiotic rate constant

    # Where the Damkohler numbers come from. False, the default, sweeps them
    # directly over the range below, which is what a TRAINING SET wants. True
    # computes them from vmax and k_surf above, which is what you want when you
    # are modelling one named system. Read --show-settings before turning it
    # on: at a sample of tens of micrometres with ordinary diffusion
    # coefficients the honest Damkohler is usually far below 1, because
    # diffusion crosses the sample in about a second.
    use_physical_kinetics = False,
    decay_lattice         = 0.006,   # biomass decay when the above is False

    # ---- what to sweep -----------------------------------------------------
    # Leave these alone to use the ranges on the command line.
    # peclet_min = 1.0,   peclet_max = 50.0,
    # damkohler_min = 0.1, damkohler_max = 10.0,
)

#  THE CHEMICALS, in this order and only this order:
#      0 the electron donor      1 the electron acceptor
#      2 the product             3 the biomass
#  The reaction is  donor + (stoichiometry) acceptor -> product + (yield)
#  biomass, so the position decides the role and the name is only a label.
#
#  ADD AS MANY MORE AS YOU LIKE. Anything past the fourth is transported
#  properly -- fed at the inlet, advected, and diffused with its own
#  coefficient -- but takes no part in any reaction. That is a conservative
#  tracer, and it is the standard way to separate where the water went from
#  where the chemistry got to. A fifth REACTING chemical needs CompLaB.
#  Remember to raise --n-species to match, or only the first few are solved.
#
#      left_type   "Dirichlet" holds this chemical at left_value on the inlet
#                  face -- that is how you inject something. "Neumann" leaves
#                  the face free.
#      immobile    True means it reacts where it is and never moves. The
#                  biomass is attached by default.
MY_CHEMICALS = [
    dict(name="Ac",  initial=0.0,    d_pore=1.0e-9, d_biofilm=5.0e-10,
         left_type="Dirichlet", left_value=2.0e-3, right_type="Neumann"),
    dict(name="A",   initial=0.0,    d_pore=1.0e-9, d_biofilm=5.0e-10,
         left_type="Dirichlet", left_value=5.0e-3, right_type="Neumann"),
    dict(name="P",   initial=0.0,    d_pore=1.0e-9, d_biofilm=5.0e-10,
         left_type="Dirichlet", left_value=0.0,    right_type="Neumann"),
    dict(name="Bio", initial=2.0e-4, d_pore=0.0,   d_biofilm=0.0,
         left_type="Neumann",   left_value=0.0,    right_type="Neumann",
         immobile=True),
]
# ===========================================================================
#  END OF THE BLOCK YOU EDIT.
# ===========================================================================


# --------------------------------------------------------------- geometry ---
def blob_2d(shape, phi, rng, sigma=3.0):
    """Thresholded correlated Gaussian field: Jung's morphology, in 2D."""
    f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    g = np.where(f > np.quantile(f, 1.0 - phi), PORE, SOLID).astype(np.uint8)
    g[:4] = g[-4:] = PORE                       # inlet and outlet buffers
    return g


def keep_spanning_cluster(g):
    lab, n = ndimage.label(g == PORE)
    if n <= 1:
        return g, n
    keep = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    g = g.copy()
    g[(g == PORE) & (lab != keep)] = SOLID
    return g, n


def percolates(g):
    lab, _ = ndimage.label(g == PORE)
    a = set(np.unique(lab[0][g[0] == PORE]).tolist())
    b = set(np.unique(lab[-1][g[-1] == PORE]).tolist())
    return bool((a & b) - {0})


def geodesic_2d(g):
    """Shortest path from the inlet to every pore voxel, THROUGH pore space.

    Not the straight-line distance: a chemical cannot travel through rock. This
    is the field the trunk receives, and the whole premise of the method is that
    it beats the Euclidean one.
    """
    INF = np.float32(1e9)
    d = np.where(g == PORE, INF, np.nan).astype(np.float32)
    d[0][g[0] == PORE] = 0.0
    for _ in range(4 * g.shape[0]):
        prev = d.copy()
        for ax in (0, 1):
            for sh in (1, -1):
                nb = np.roll(d, sh, axis=ax)
                sl = [slice(None)] * 2
                sl[ax] = 0 if sh > 0 else -1
                nb[tuple(sl)] = INF
                d = np.where(g == PORE,
                             np.fmin(d, np.nan_to_num(nb, nan=INF) + 1.0), np.nan)
        if np.nanmax(np.abs(np.nan_to_num(d - prev, nan=0.0))) < 1e-6:
            break
    return np.nan_to_num(d, nan=0.0).astype(np.float32)


# ------------------------------------------------------------------ flow ---
# ------------------------------------------------------------- transport ---
def adr_2d(g, vel, pe, da, n_t, n_species=2, steps=None, safety=1.0,
           tol=2e-3, max_steps=100000, info=None, chem=None, da_abio=0.0,
           on_progress=None):
    """The 2D transport solve. A thin wrapper on the one shared solver.

    There used to be a second, independent implementation of this in
    build_practice_dataset.py. Six bugs were found and fixed here and every one of them
    survived over there, silently, because fixing one copy makes the other
    look fixed. There is now one solver, and prtlb_2d.py and prtlb_3d.py hold it word
    for word -- test_flow_solvers.py compares them byte for byte on every
    run, so the two copies cannot drift apart the way the old two did.
    """
    return solve_adr(g, vel, pe, da, n_t, n_species=n_species, steps=steps,
                     tol=tol, max_steps=max_steps, info=info, chem=chem,
                     da_abio=da_abio, on_progress=on_progress)


# ------------------------------------------------------- settings helpers ---
def _resolve(from_cli, cfg, built_in):
    """The grid: the command line first, then the settings file, then the
    built-in default. In that order and stated out loud, because a grid that
    silently came from somewhere other than where you were looking is the
    single easiest way to spend an afternoon on the wrong dataset."""
    if from_cli:
        print("grid %s, from the command line" % (tuple(from_cli),))
        return tuple(from_cli)
    if cfg.source.startswith("built-in"):
        print("grid %s, the built-in default" % (built_in,))
        return built_in
    got = (cfg.nx, cfg.ny) if len(built_in) == 2 else (cfg.nx, cfg.ny, cfg.nz)
    print("grid %s, from the settings file" % (got,))
    return got


def _range(cli_lo, cli_hi, cfg_lo, cfg_hi, def_lo, def_hi):
    """Same order of precedence, for a swept range."""
    if cli_lo is not None or cli_hi is not None:
        return (cli_lo if cli_lo is not None else def_lo,
                cli_hi if cli_hi is not None else def_hi)
    if cfg_lo is not None and cfg_hi is not None:
        return cfg_lo, cfg_hi
    return def_lo, def_hi


# ----------------------------------------------------------------- main ----
class Progress(object):
    """One line per simulation, with how long it took and how long is left.

    The transport loop used to print one line per STRUCTURE, after all of its
    conditions had finished. At 148 by 64 with a Peclet of 0.3 that is several
    minutes of complete silence, and silence during a long run is
    indistinguishable from a hang. So: a line as each simulation finishes,
    carrying the conditions it ran at, what it cost, and an estimate of what
    is left based on the simulations that have actually finished rather than
    on a guess made beforehand.
    """

    def __init__(self, total):
        self.total = int(total)
        self.done = 0
        self.t0 = time.monotonic()

    @staticmethod
    def _hms(sec):
        sec = int(max(sec, 0))
        if sec < 90:
            return "%d s" % sec
        if sec < 5400:
            return "%d min" % round(sec / 60.0)
        return "%.1f h" % (sec / 3600.0)

    def start(self, structure, pe, da, da_abio):
        """Begin one simulation. Returns the callback the solver reports to."""
        self._t = time.monotonic()
        self._last = 0.0
        self._head = ("  [%*d/%d] rock %-3d Pe=%-7.3g Da=%-7.3g%s"
                      % (len(str(self.total)), self.done + 1, self.total,
                         structure, pe, da,
                         (" Da_ab=%-7.3g" % da_abio) if da_abio else ""))

        def beat(step, limit, mean, change):
            # Throttled by TIME, not by step count. A step at 148 by 64 takes
            # about a millisecond and a step at 16 cubed takes a tenth of that,
            # so any fixed step interval is either a flood or a silence
            # depending on the grid. Five seconds is neither.
            now = time.monotonic()
            if now - self._t < 5.0 or now - self._last < 5.0:
                return
            self._last = now
            frac = step / float(max(limit, 1))
            note = ""
            if frac > 0.9:
                note = ("  -- close to the step cap, so this one will be "
                        "recorded as NOT settled")
            print("%s   working: %s, %d steps, still changing by %.1e%s"
                  % (self._head, self._hms(now - self._t), step, change, note),
                  flush=True)

        return beat

    def step(self, structure, pe, da, da_abio, info, seconds):
        self.done += 1
        elapsed = time.monotonic() - self.t0
        left = (elapsed / self.done) * (self.total - self.done)
        settled = info.get("converged", True)
        print("  [%*d/%d] rock %-3d Pe=%-7.3g Da=%-7.3g%s  %6s  %6d steps  "
              "%s   about %s left"
              % (len(str(self.total)), self.done, self.total, structure,
                 pe, da,
                 (" Da_ab=%-7.3g" % da_abio) if da_abio else "",
                 self._hms(seconds), int(info.get("steps", 0)),
                 "settled" if settled else "RAN OUT",
                 self._hms(left)), flush=True)

    def finish(self):
        print("  all %d simulations done in %s"
              % (self.total, self._hms(time.monotonic() - self.t0)), flush=True)


def announce(cfg, n_geom, n_sets, shape, pe_lo, pe_hi, da_lo, da_hi,
             da2_lo, da2_hi):
    """Say, before anything runs, exactly what is about to be run.

    "How many pore structures" and "parameter sets per structure" are two
    numbers whose PRODUCT is the thing you actually wait for, and neither of
    them is that number. So it is printed here, once, in words.
    """
    n = cfg.n_simulations(n_geom, n_sets)
    by_pressure = str(cfg.flow_driver).lower().startswith("pres")
    print("=" * 70)
    print("THIS RUN WILL DO %d SIMULATIONS." % n["simulations"])
    print()
    print("  %d pore structures      -- %d different rocks, each generated"
          % (n["structures"], n["structures"]))
    print("                            once and then used for every condition")
    print("  x %d conditions each    -- one flow-and-transport simulation per"
          % n["sets"])
    print("                            combination of Peclet and Damkohler")
    print("  = %d simulations" % n["simulations"])
    print()
    print("  %d flow solves          -- one per STRUCTURE, not per simulation."
          % n["flow_solves"])
    print("                            The pore space does not change, and")
    print("                            the flow does not depend on the")
    print("                            chemistry, so it is solved once and")
    print("                            reused.")
    print("  %d transport solves     -- one per simulation. This is the"
          % n["transport_solves"])
    print("                            expensive part.")
    print()
    print("  grid                     %s" % (tuple(shape),))
    if by_pressure:
        print("  flow driven by           the PRESSURE DROP, %g. Peclet is"
              % cfg.delta_P)
        print("                           whatever that flow makes it, and is")
        print("                           computed and recorded per run.")
    else:
        print("  flow driven by           the PECLET NUMBER, swept %g to %g."
              % (pe_lo, pe_hi))
        print("                           The pressure drop only seeds the")
        print("                           lattice-Boltzmann solve; its size")
        print("                           divides out when the field is")
        print("                           rescaled to hit that Peclet.")
    print("  biotic reaction          %s"
          % ("ON, Damkohler swept %g to %g" % (da_lo, da_hi)
             if cfg.biotic_enabled else "OFF"))
    print("  abiotic reaction         %s"
          % ("ON, Damkohler swept %g to %g%s"
             % (da2_lo, da2_hi,
                ", on grain surfaces only" if cfg.abiotic_surface_only
                else ", everywhere in the water")
             if cfg.abiotic_enabled else "OFF, so its Damkohler is recorded "
                                         "as zero"))
    print("  biomass                  %s"
          % ("SESSILE: attached, reacts in place, never moves"
             if str(cfg.biomass_mode).lower().startswith("sess")
             else "PLANKTONIC: advected and diffused like a solute"))

    # WILL THE STEP CAP BITE? Worth saying BEFORE the run, not after.
    #
    # The transport step is limited by diffusion, and the diffusion
    # coefficient is nx/Peclet -- so the lower the Peclet the smaller the step
    # and the more steps it takes to settle. The count goes roughly as the
    # grid area, and barely depends on Peclet at the low end because the
    # shorter settling time and the smaller step cancel. Measured at 148 by
    # 64: about 91,000 steps at Peclet 0.76 and 100,000 was not enough at
    # Peclet 1.2.
    nd = 3 if len(shape) > 2 and shape[2] > 1 else 2
    need = int(3 * nd * max(shape[0], 1) ** 2)
    cap = int(cfg.ade_max_iT)
    print("  transport step cap       %d" % cap)
    if cap < need:
        print("      A run at this grid typically needs about %d steps to"
              % need)
        print("      settle at the low-Peclet end, so some runs will stop")
        print("      early and be recorded as NOT settled. They are still")
        print("      valid transients and fine to train on, but their last")
        print("      snapshot is 'as far as we got', not a steady state.")
        print("      To avoid it, raise the box 'Never take more transport")
        print("      steps than' to about %d. That costs roughly %.1f times"
              % (int(need * 1.5), need * 1.5 / cap))
        print("      the time.")
    print("=" * 70)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--jung-dir", default=None,
                     help="a PRT-DeepONet checkout; uses HIS domains with OUR "
                          "chemistry, natively in 2D with no extrusion")
    src.add_argument("--n-geom", type=int, default=None,
                     help="generate this many domains instead")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on how many of Jung's domains to read")
    ap.add_argument("--shape", type=int, nargs=2, default=None,
                    help="Jung's release is 148 x 64, which is the default. A "
                         "settings file's grid is used when this is not given.")
    ap.add_argument("--n-sets", type=int, default=6,
                    help="parameter sets, i.e. Pe and Da pairs, per domain")
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--n-species", type=int, default=2)
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--phi-min", type=float, default=0.55,
                    help="2D percolates near 0.59, not 0.20 as in 3D")
    ap.add_argument("--phi-max", type=float, default=0.85)
    ap.add_argument("--pe-min", type=float, default=None)
    ap.add_argument("--pe-max", type=float, default=None)
    ap.add_argument("--da-min", type=float, default=None,
                    help="lowest BIOTIC Damkohler")
    ap.add_argument("--da-max", type=float, default=None,
                    help="highest BIOTIC Damkohler")
    ap.add_argument("--da-abio-min", type=float, default=None,
                    help="lowest ABIOTIC Damkohler; only used when the "
                         "abiotic reaction is switched on")
    ap.add_argument("--da-abio-max", type=float, default=None)
    ap.add_argument("--stokes-iters", type=int, default=30000,
                    help="the CAP on flow iterations, not the number "
                         "run: the solve stops when the field stops "
                         "changing, which is usually far sooner")
    ap.add_argument("--adr-steps", type=int, default=None,
                    help="transport steps per run. LEAVE THIS ALONE: by default "
                         "it is derived from Peclet so the front actually "
                         "crosses the sample. A fixed value that is too small "
                         "produces training fields that are almost entirely "
                         "zero, and a network trained on those learns to "
                         "predict zero.")
    ap.add_argument("--sigma", type=float, default=3.0,
                    help="Gaussian filter width; sets the grain size")
    ap.add_argument("--interface", choices=["pore", "solid"], default="pore")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-vti", action="store_true",
                    help="also write ParaView .vti files, one per chemical "
                         "per snapshot, in the same format CompLaB writes")
    ap.add_argument("--save-png", action="store_true",
                    help="also write PNG pictures of each simulation, which "
                         "is the fastest way to see whether the front "
                         "crossed the sample")
    ap.add_argument("--save-runs", type=int, default=4,
                    help="how many simulations to write pictures and VTI "
                         "for. Writing all of them is a lot of files.")
    add_settings_arguments(ap)
    a = ap.parse_args()

    # THE PHYSICAL SETTINGS COME FIRST, before a single voxel is generated.
    # Reading them here means a mistake in the file costs a second, not four
    # hours followed by a figure nobody can interpret.
    cfg, chem = settings_from_args(a, dimension=2,
                               block=MY_SETTINGS,
                               chemicals=MY_CHEMICALS,
                               block_name="the MY_SETTINGS block at the top of build_dataset_2d.py",
                               default_shape=(148, 64))
    if a.show_settings:
        return
    # Settled inside settings_from_args, before the report was printed.
    nx, ny = cfg.nx, cfg.ny
    pe_lo, pe_hi = _range(a.pe_min, a.pe_max, cfg.peclet_min, cfg.peclet_max,
                          0.3, 30.0)
    da_lo, da_hi = _range(a.da_min, a.da_max, cfg.damkohler_min,
                          cfg.damkohler_max, 0.1, 10.0)
    da2_lo, da2_hi = _range(a.da_abio_min, a.da_abio_max,
                            cfg.damkohler_abiotic_min,
                            cfg.damkohler_abiotic_max, 0.1, 10.0)
    by_pressure = str(cfg.flow_driver).lower().startswith("pres")
    # SAME ORDER as collect_complab_output.py writes, so a file from here and a file from a
    # CompLaB complab_campaign can be mixed or compared without channel 1 meaning "P"
    # in one and "A" in the other.
    species = a.species or chem.names[:a.n_species]
    a.n_species = len(species)
    chem = chem.truncate(a.n_species)
    pnames = ["pe", "da_bio", "da_abio", "ks_ac_norm", "ks_a_norm", "y_norm"]
    rng = np.random.default_rng(a.seed)

    # How many simulations this is, said before any of them start. The two
    # boxes on the page are the number of ROCKS and the number of CONDITIONS
    # per rock; the thing you actually wait for is their product.
    announce(cfg, a.n_geom or (a.limit or 20), a.n_sets, (nx, ny),
             pe_lo, pe_hi, da_lo, da_hi, da2_lo, da2_hi)

    # ---------------- 1. the domains -------------------------------------
    doms = []
    if a.jung_dir:
        from build_transfer_set_2d_to_3d import read_jung, to_complab, resize2d
        raw = read_jung(a.jung_dir, limit=a.limit)
        if not raw:
            raise SystemExit("found no 2D domains under %s" % a.jung_dir)
        print("read %d domains from %s" % (len(raw), a.jung_dir))
        for r in raw:
            g = resize2d(to_complab(r, interface=a.interface), (nx, ny))
            g, _ = keep_spanning_cluster(g)
            if percolates(g):
                doms.append(g)
    else:
        n = a.n_geom or 20
        tries = 0
        while len(doms) < n and tries < n * 20:
            tries += 1
            phi = a.phi_min + (a.phi_max - a.phi_min) * rng.random()
            g = blob_2d((nx, ny), phi, rng, sigma=a.sigma)
            g, _ = keep_spanning_cluster(g)
            if percolates(g):
                doms.append(g)
        print("generated %d domains in %d attempts" % (len(doms), tries))
    if not doms:
        raise SystemExit("no percolating domains")
    G = len(doms)
    print("kept %d percolating domains" % G)

    # ---------------- 2. geometry fields and flow ------------------------
    shape = (nx, ny, 1)
    mat = np.zeros((G,) + shape, np.uint8)
    gdf = np.zeros((G,) + shape, np.float32)
    edt = np.zeros((G,) + shape, np.float32)
    vels = []
    umean_lat = []          # mean pore speed in lattice units, per structure
    n_bf = 0
    bfilm = np.zeros((G,) + shape, bool)
    solve_mat = []
    for i, g in enumerate(doms):
        # The COATED copy goes to the transport solver; the stored geometry
        # keeps biofilm voxels as plain pore and records where they were in
        # geom/biofilm. A biofilm voxel is part of the pore space -- the flow
        # runs through it and the geodesic distance runs through it -- so
        # giving it its own code in the stored array would make the flow
        # solver, the distance field and the network's pore list all treat it
        # as rock.
        if cfg.biofilm_code is not None:
            gc, nb = coat_with_biofilm(g, cfg.biofilm_code, cfg.biofilm_layers)
            n_bf += nb
            bfilm[i] = (gc == cfg.biofilm_code)[:, :, None]
            solve_mat.append(gc)
        else:
            solve_mat.append(g)
        mat[i] = g[:, :, None]
        gdf[i] = geodesic_2d(g)[:, :, None]
        bad_d = assert_finite_distance(gdf[i][:, :, 0], g)
        if bad_d:
            raise SystemExit("domain %d: %s" % (i, bad_d))
        edt[i] = ndimage.distance_transform_edt(g == PORE)[:, :, None]
        # tau and delta_P from your settings now reach the flow solver. They
        # did not before: it used its own defaults, so both were read,
        # reported, and ignored.
        # ns_max_iT and ns_converge from the settings file now decide how
        # long the flow solve runs and when it stops. They were read,
        # validated and reported, and reached nothing.
        _nit = (int(cfg.ns_max_iT) if not cfg.source.startswith("built-in")
                else int(a.stokes_iters))
        # THE DRIVING FORCE IS CHOSEN SO THE SOLVE SITS INSIDE THE RANGE
        # LATTICE BOLTZMANN IS VALID IN.
        #
        # The equilibrium is a low-Mach expansion and is only reliable below
        # about Mach 0.05. A real run reported 0.078 on one rock and 0.093 on
        # another, from the default seed of 1.5e-4 -- so the flow field every
        # transport solve was handed carried a compressibility error the model
        # does not claim to represent, and the mass flux through successive
        # planes varied by 2.6%, which means it was not even divergence free.
        #
        # In PECLET MODE that was pure waste, because Stokes flow is linear
        # and the field is rescaled afterwards: the size of the seed divides
        # out completely, so there was never a reason for it to be a number
        # that breaks the solver. The seed is measured once at a small force
        # and rescaled to land at Mach 0.02, comfortably inside the valid
        # range, before the field that is actually used is solved.
        #
        # In PRESSURE MODE the pressure drop IS the experiment and is left
        # exactly as given; the warning still fires if it is too large,
        # because there the answer really does depend on it.
        _force = float(cfg.delta_P)
        if not by_pressure:
            _probe = stokes_d2q9(g, nit=_nit, force=1e-6, tau_lb=cfg.tau,
                           tol=float(cfg.ns_converge))
            _peak = float(np.sqrt((_probe ** 2).sum(0)).max())
            if _peak > 0:
                # Mach = u / c_s, with c_s = 1/sqrt(3) in lattice units.
                _want = 1e-6 * (0.02 / np.sqrt(3.0)) / _peak
                # ONLY EVER LOWER IT. Scaling UP to hit a Mach target is a
                # different and much worse mistake: on a tight rock the force
                # needed to push the peak speed to Mach 0.02 is enormous, and
                # the body force has to stay a small perturbation on the
                # equilibrium or the expansion the whole method rests on stops
                # being valid. Asking for it on a rock of porosity 0.55 gave a
                # force of 0.023, and the solve overflowed to not-a-number
                # within its first few hundred iterations. A LOW Mach number
                # is not a problem to be corrected -- it is the regime the
                # method is built for. The seed is a ceiling; this only brings
                # it down when the rock is permeable enough to need it.
                if _want < _force:
                    print("     driving force lowered from %.3g to %.3g: this "
                          "rock is permeable enough that the seed would have "
                          "run it at Mach %.3f,\n     outside the range "
                          "lattice Boltzmann is valid in. Peclet mode "
                          "rescales the field afterwards, so nothing is lost."
                          % (_force, _want,
                             _peak * np.sqrt(3.0) * _force / 1e-6), flush=True)
                    _force = _want
        v = stokes_d2q9(g, nit=_nit, force=_force, tau_lb=cfg.tau,
                        tol=float(cfg.ns_converge))
        vels.append(v)
        # THE MEAN AXIAL VELOCITY, not the mean speed. The transport solver
        # normalises by the axial mean, so a Peclet number computed here from
        # the speed describes a different flow from the one being solved.
        # They differ by the tortuosity -- measured 1.140 on a real Stokes
        # field, and geometry dependent, so it is not even a constant offset.
        sp = np.sqrt((v[:2] ** 2).sum(0))
        _ax = float(v[0][g == PORE].mean())
        _spd = float(sp[g == PORE].mean())
        umean_lat.append(_ax if _ax > 0.1 * _spd else _spd)
        _pe_lo = cfg.pressure_for_peclet(umean_lat[-1], pe_lo, g.shape[0])
        _pe_hi = cfg.pressure_for_peclet(umean_lat[-1], pe_hi, g.shape[0])
        if not by_pressure:
            print("     to reach Peclet %g to %g on this one the pressure "
                  "gradient runs %.3g to %.3g" % (pe_lo, pe_hi, _pe_lo, _pe_hi),
                  flush=True)
        print("  domain %3d  phi=%.3f  gdf_max=%5.1f  umean=%.2e"
              % (i, float((g == PORE).mean()), gdf[i].max(),
                 float(sp[g == PORE].mean())), flush=True)

    # ---------------- 3. transport ---------------------------------------
    S = G * a.n_sets
    T, C = a.n_times, a.n_species
    conc = np.zeros((S, T, C) + shape, np.float32)
    vel = np.zeros((S, 3) + shape, np.float32)
    gi = np.zeros(S, np.int32)
    par = np.zeros((S, len(pnames)), np.float32)
    tn = np.zeros((S, T), np.float32)
    k = 0
    n_bad_physics = 0
    settled = np.ones(S, bool)
    prog = Progress(S)
    seen_pressure_pe = set()
    dP = np.zeros(S, np.float32)
    for i in range(G):
        for _ in range(a.n_sets):
            # PECLET. Either swept directly, or -- in pressure mode -- taken
            # from the flow the pressure drop actually produced.
            if by_pressure:
                pe = cfg.peclet_from_flow(umean_lat[i], shape[0])
                if i not in seen_pressure_pe:
                    seen_pressure_pe.add(i)
                    want = 10.0
                    print("     rock %d came out at Peclet %.4g. For Peclet "
                          "%.4g set the pressure drop to %.4g."
                          % (i, pe, want,
                             cfg.pressure_for_peclet(umean_lat[i], want, shape[0])),
                          flush=True)
            else:
                pe = float(10 ** rng.uniform(np.log10(pe_lo), np.log10(pe_hi)))
            # THE PRESSURE GRADIENT THAT PRODUCES THIS PECLET on this rock.
            # The flow really is driven by a pressure gradient; you simply say
            # what Peclet you want and this is the push it takes. Stokes flow
            # is linear, so it is the seed pressure times the ratio of wanted
            # Peclet to the Peclet the seed produced -- exact, not fitted. It
            # differs from rock to rock, because a tighter pore structure
            # needs a stronger push for the same Peclet, which is what
            # permeability means.
            dP[k] = cfg.pressure_for_peclet(umean_lat[i], pe, shape[0])
            # THE TWO DAMKOHLER NUMBERS, drawn separately, because they are two
            # different reactions. A reaction that is switched off gets zero,
            # not a copy of the other one's number.
            da = (float(10 ** rng.uniform(np.log10(da_lo), np.log10(da_hi)))
                  if cfg.biotic_enabled else 0.0)
            da2 = (float(10 ** rng.uniform(np.log10(da2_lo), np.log10(da2_hi)))
                   if cfg.abiotic_enabled else 0.0)
            gi[k] = i
            # The recorded parameters are the ones the run WAS GIVEN. They used
            # to be the literals 0.1, 0.1, 0.05 while the solver used 0, 0.15
            # and 0.04, so the network was told three numbers that were not the
            # ones its training data had been made with. Constant columns, so
            # nothing learned from them -- but a dataset that misreports its own
            # conditions is not one to hand anybody else.
            par[k] = [pe, da, da2, chem.ks_donor, chem.ks_acceptor,
                      chem.yield_]
            vel[k] = vels[i][:, :, :, None]
            nfo = {}
            beat = prog.start(i, pe, da, da2)
            t_run = time.monotonic()
            # THE STOPPING RULES FROM YOUR SETTINGS NOW REACH THE SOLVER.
            # They did not before: the two boxes were read, printed in the
            # report, and then the solver used its own built-in defaults --
            # so raising the step cap because runs were being truncated had
            # no effect whatsoever, which is the worst kind of setting to
            # have.
            cc, tt = adr_2d(solve_mat[i], vels[i], pe, da, T, C, steps=a.adr_steps,
                            info=nfo, chem=chem, da_abio=da2,
                            on_progress=beat, tol=cfg.ade_converge,
                            max_steps=int(cfg.ade_max_iT))
            prog.step(i, pe, da, da2, nfo, time.monotonic() - t_run)
            if (a.save_vti or a.save_png) and k < a.save_runs:
                if write_vti_and_png is None:
                    if k == 0:
                        print("     NOTE: --save-vti / --save-png asked for, "
                              "but write_vti_and_png.py is not in this "
                              "project. No pictures are written; the dataset "
                              "is unaffected.", flush=True)
                else:
                    write_vti_and_png.save_run(
                    os.path.splitext(a.out)[0] + "_fields", k, cc, tt,
                    species, mat=doms[i], vel=vels[i][:2],
                    want_vti=a.save_vti, want_png=a.save_png,
                    spacing=cfg.dx,
                    # Label it with the Damkohler that is actually running.
                    # "Da=0" on an abiotic-only run reads as no reaction at
                    # all, when in fact the abiotic one is going.
                    note="Pe=%.3g  Da_bio=%.3g  Da_abio=%.3g"
                         % (pe, da, da2))
            settled[k] = bool(nfo.get("converged", True))
            conc[k] = cc[:, :, :, :, None]
            tn[k] = tt
            for msg in check_physics(cc, doms[i], chem=chem, da=da, da_abio=da2):
                print("     PHYSICS run %d (Pe=%.2f Da_bio=%.2f "
                      "Da_abio=%.2f): %s" % (k, pe, da, da2, msg), flush=True)
                n_bad_physics += 1
            reach = int((conc[k][-1, 0, :, :, 0].max(axis=1) > 1e-3).sum())
            if reach < 0.5 * nx:
                print("     WARNING run %d: the front only reached %d of %d "
                      "columns" % (k, reach, nx))
            k += 1
    prog.finish()
    if cfg.biofilm_code is not None:
        print("  %d voxels were marked as biofilm (code %d), %d layer(s) "
              "thick on the grain surfaces" % (n_bf, cfg.biofilm_code,
                                               cfg.biofilm_layers))

    # REFUSE TO WRITE A FILE YOU HAVE JUST BEEN TOLD NOT TO TRAIN ON.
    #
    # This used to write the dataset and then print a warning underneath it.
    # The 3D generator, faced with the identical complaint, stopped before
    # writing -- so the same mistake produced a file in 2D and no file in 3D,
    # and the 2D file sat on disk looking exactly like a good one. A warning
    # printed after the artefact exists is a warning that gets scrolled past;
    # the .h5 is what survives the session, and it carried no sign of the
    # problem in its name, its size or its contents.
    #
    # Both now stop here, with the same message.
    if n_bad_physics:
        raise SystemExit(
            "%d physics complaint(s), listed above. This file is not usable "
            "as training data, so it has NOT been written -- writing it would "
            "leave a dataset on disk that looks exactly like a good one. Fix "
            "what the complaints describe and run again." % n_bad_physics)

    # ---------------- 4. write -------------------------------------------
    scale = np.maximum(
        conc.reshape(S * T * C, -1).max(1).reshape(S, T, C).max((0, 1)), 1e-6)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with h5py.File(a.out, "w") as h:
        h.attrs["n_samples"] = S
        h.attrs["shape"] = np.array(shape, np.int32)
        h.attrs["species"] = np.array([s.encode() for s in species])
        h.attrs["param_names"] = np.array([s.encode() for s in pnames])
        h.attrs["dimension"] = 2
        h.attrs["source"] = (b"jung_domains" if a.jung_dir else b"generated")
        # The whole settings file, stored inside the dataset. A dataset that
        # cannot say what conditions made it is a dataset you have to trust.
        h.attrs["settings_source"] = str(cfg.source).encode()
        h.attrs["settings"] = cfg.to_json().encode()
        gg = h.create_group("geom")
        gg.create_dataset("gid", data=np.arange(G, dtype=np.int32))
        gg.create_dataset("material", data=mat, compression="gzip")
        gg.create_dataset("gdf", data=gdf, compression="gzip")
        gg.create_dataset("edt", data=edt, compression="gzip")
        if bfilm.any():
            gg.create_dataset("biofilm", data=bfilm.astype(np.uint8),
                              compression="gzip")
        sg = h.create_group("samples")
        sg.create_dataset("geom_index", data=gi)
        sg.create_dataset("run_id", data=np.arange(S, dtype=np.int32))
        sg.create_dataset("params", data=par)
        sg.create_dataset("t_norm", data=tn)
        # The pressure gradient each run needed to reach its Peclet, in
        # lattice units. Derived, not chosen: it is what the flow solver would
        # have to be driven with to produce that Peclet on that rock.
        sg.create_dataset("delta_P", data=dP)
        # True where the run reached a steady state; False where it ran out of
        # steps first, so that its t = 1.0 means "as far as we got".
        sg.create_dataset("settled", data=settled)
        d = sg.create_dataset("conc", data=conc.astype(np.float16),
                              compression="gzip")
        d.attrs["conc_scale"] = scale.astype(np.float32)
        sg.attrs["conc_scale"] = scale.astype(np.float32)
        sg.create_dataset("velocity", data=vel.astype(np.float16),
                          compression="gzip")

    print()
    print("=" * 70)
    print("SAVED. Everything this run produced is in ONE file:")
    print()
    print("    %s" % os.path.abspath(a.out))
    print()
    print("    folder:  %s" % os.path.dirname(os.path.abspath(a.out)))
    print("    size:    %.1f MB" % (os.path.getsize(a.out) / 1e6))
    print()
    print("That one file holds the pore structures, both distance fields, the")
    print("flow field, the concentrations at every stored time, the conditions")
    print("each simulation was given, and a copy of every settings_and_units setting used")
    print("to make it. Nothing else was written anywhere. Open it with the")
    print("Viewer, or train on it -- training writes its own output to its own")
    print("--out folder, not here.")
    print("=" * 70)
    print("  %d domains, %d samples, %d snapshots, %d chemicals, grid %d x %d"
          % (G, S, T, C, nx, ny))
    print("  this is a NATIVE 2D dataset: nz = 1, so training switches its")
    print("  encoder to 2D convolutions and the trunk becomes (x, y, t, gdf),")
    print("  which is Jung's original architecture.")
    n_unsettled = int((~settled).sum())
    if n_unsettled:
        print()
        print("  " + "-" * 66)
        print("  %d of %d runs stopped at the step cap of %d before settling."
              % (n_unsettled, S, int(cfg.ade_max_iT)))
        print("  They are valid transients and fine to train on, but their last")
        print("  snapshot is as far as the integration got, not a steady state,")
        print("  and t = 1 should not be read as one. Which runs is recorded in")
        print("  samples/settled inside the file.")
        print("  To settle them, raise 'Never take more transport steps than'")
        print("  and run again. It is the low-Peclet runs that need it.")
        print("  " + "-" * 66)
    # Reaching here at all means every run passed: a complaint stops the
    # program above, before the file exists.
    print("  physics checked on every run: concentrations stay at or below")
    print("  the inlet value, no checkerboard, nothing reaches the outlet")
    print("  before it has crossed the sample, no chemical is a copy of")
    print("  another, and every reaction that was given a rate had something")
    print("  to react with.")
    print("\nnext:")
    print("  python ../model/train.py --data %s --out runs/2d" % a.out)


if __name__ == "__main__":
    main()
