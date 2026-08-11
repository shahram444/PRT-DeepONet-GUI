#!/usr/bin/env python3
"""
build_dataset_3d.py — a NATIVE 3D dataset. The exact twin of build_dataset_2d.py.

WHAT THIS IS, AND WHAT IT IS NOT
    It is the baseline. Real 3D pore structures, a real D3Q19 Stokes solve, a
    real 3D advection-diffusion-reaction solve, and the same chemistry the 2D
    generator uses -- so a 2D result and a 3D result from this project are
    directly comparable, because only the dimension differs.

    It is NOT CompLaB. CompLaB is the reference simulator, it runs on the
    cluster, and one run there takes hours. Use complab_campaign.py + collect_complab_output.py for
    that. This file exists so the whole method can be exercised end to end in
    3D on one machine, in hours rather than in cluster-weeks, before spending
    the cluster time.

    It is also NOT any of the cost-reducing switches. No flow proxy, no 2D
    transfer, no dimension-free coordinates. Geometry in, concentrations out.
    That is the point: this is the thing the switches have to be measured
    against, so it must be built without them.

COST -- MEASURED, and it is the fifth power of the grid
    Transport cost goes as (voxels) x (steps), voxels go as n^3 and the
    explicit step count goes as n^2, so doubling the grid costs 32 times more.
    Measured here, per run, one core:

        32^3    7 s        48^3   58 s        64^3   ~4 min

    Stokes is once per STRUCTURE, not per run: 3 s, 10 s and 25 s respectively.

    So for 16 structures x 5 parameter sets = 80 runs:

        32^3    about 11 minutes
        48^3    about 80 minutes
        64^3    about 5.5 hours

    and for a dataset you would actually publish from, 60 structures x 6 sets:

        48^3    about 6 hours
        64^3    about a day

    Start at 32^3 to prove the pipeline in an hour, then commit to 48^3 or
    64^3 overnight. 64^3 is the grid complab_campaign.py uses, so a model trained here
    at 64^3 can be compared with CompLaB output directly.

POROSITY
    Defaults to 0.30-0.50. In 3D a thresholded Gaussian field stops
    percolating near porosity 0.20; in 2D the threshold is near 0.59. That gap
    is why a 2D range of 0.55-0.85 would be absurd here, and it is the same
    gap that stops a geometry encoder trained in 2D from being reused in 3D.

USAGE
    python build_dataset_3d.py --out data3d.h5 --n-geom 16 --n-sets 5 --shape 32 32 32
    python build_dataset_3d.py --out data3d.h5 --n-geom 60 --n-sets 6 --shape 64 64 64

Then exactly as in 2D -- the same scripts, unchanged:
    python ../model/train.py --data data3d.h5 --out runs/3d
    python ../model/evaluate.py --checkpoint runs/3d/best.pt --data data3d.h5 --out figs/3d
"""

import argparse
import os
import sys
import time
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prtlb_3d import (SOLID, WALL, PORE, solve_adr, check_physics,     # noqa: F401
                      keep_inlet_connected, assert_finite_distance,
                      coat_with_biofilm)
from build_practice_dataset import geometry, geodesic, stokes                    # noqa: E402
from build_dataset_2d import _resolve, _range, Progress               # noqa: E402
import write_vti_and_png                                                  # noqa: E402
from settings_and_units import (add_settings_arguments,                         # noqa: E402
                      settings_from_args)
from scipy import ndimage                                             # noqa: E402


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


def percolates_3d(g):
    """Does pore space span inlet to outlet along x?"""
    lab, _ = ndimage.label(g == PORE)
    a = set(np.unique(lab[0][g[0] == PORE]).tolist()) - {0}
    b = set(np.unique(lab[-1][g[-1] == PORE]).tolist()) - {0}
    return bool(a & b)


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
    print("=" * 70)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-geom", type=int, default=16,
                    help="how many pore structures to generate")
    ap.add_argument("--n-sets", type=int, default=5,
                    help="parameter sets, i.e. Pe and Da pairs, per structure")
    ap.add_argument("--n-times", type=int, default=8)
    ap.add_argument("--n-species", type=int, default=2)
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--shape", type=int, nargs=3, default=None,
                    help="cost goes as the FIFTH power of this. 32 to prove "
                         "the pipeline, 64 to match complab_campaign.py. A settings "
                         "file's grid is used when this is not given.")
    ap.add_argument("--phi-min", type=float, default=0.30,
                    help="3D percolates near 0.20, not 0.59 as in 2D")
    ap.add_argument("--phi-max", type=float, default=0.50)
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
                    help="LEAVE THIS UNSET. The transport runs until the field "
                         "stops changing. A fixed value that is too small "
                         "produces training fields that are almost entirely "
                         "zero, and a network trained on those learns to "
                         "predict zero.")
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

    # The settings are read and checked BEFORE any structure is generated, so
    # a mistake in the file costs a second rather than the whole run.
    cfg, chem = settings_from_args(a, dimension=3,
                               block=MY_SETTINGS,
                               chemicals=MY_CHEMICALS,
                               block_name="the MY_SETTINGS block at the top of build_dataset_3d.py",
                               default_shape=(32, 32, 32))
    if a.show_settings:
        return
    # The grid was settled inside settings_from_args, before the report was
    # printed, so the report and the run describe the same thing.
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    shape0 = (nx, ny, nz)
    pe_lo, pe_hi = _range(a.pe_min, a.pe_max, cfg.peclet_min, cfg.peclet_max,
                          1.0, 50.0)
    da_lo, da_hi = _range(a.da_min, a.da_max, cfg.damkohler_min,
                          cfg.damkohler_max, 0.1, 10.0)
    da2_lo, da2_hi = _range(a.da_abio_min, a.da_abio_max,
                            cfg.damkohler_abiotic_min,
                            cfg.damkohler_abiotic_max, 0.1, 10.0)
    by_pressure = str(cfg.flow_driver).lower().startswith("pres")
    # SAME ORDER as collect_complab_output.py and build_dataset_2d.py, so a file from here, a
    # file from the 2D generator and a file from a CompLaB complab_campaign can be
    # compared without channel 1 meaning "P" in one and "A" in another.
    species = a.species or chem.names[:a.n_species]
    a.n_species = len(species)
    chem = chem.truncate(a.n_species)
    pnames = ["pe", "da_bio", "da_abio", "ks_ac_norm", "ks_a_norm", "y_norm"]
    rng = np.random.default_rng(a.seed)

    announce(cfg, a.n_geom, a.n_sets, shape0, pe_lo, pe_hi, da_lo, da_hi,
             da2_lo, da2_hi)
    print("no switches, no transfer, no flow proxy -- this is the baseline the")
    print("cost-reducing options have to be measured against.")
    print()

    # ---------------- 1. the structures ----------------------------------
    G = a.n_geom
    shape = (nx, ny, nz)
    mat = np.zeros((G,) + shape, np.uint8)
    gdf = np.zeros((G,) + shape, np.float32)
    edt = np.zeros((G,) + shape, np.float32)
    vels = []
    umean_lat = []          # mean pore speed in lattice units, per structure
    bfilm = np.zeros((G,) + shape, bool)
    solve_mat = []
    made, tries, n_bf = 0, 0, 0
    while made < G:
        tries += 1
        if tries > 20 * G:
            raise SystemExit("could not find %d percolating structures; try a "
                             "higher --phi-min" % G)
        phi = float(rng.uniform(a.phi_min, a.phi_max))
        g = geometry(shape, phi, seed=int(rng.integers(1, 2 ** 31)))
        g, _ = keep_inlet_connected(g)
        if not percolates_3d(g):
            continue
        # A biofilm lining, if asked for. The COATED copy goes to the
        # transport solver, which needs to know which voxels are biofilm. The
        # stored geometry keeps them as plain pore, and where the biofilm was
        # is recorded separately in geom/biofilm.
        #
        # That split matters. A biofilm voxel is part of the pore space: the
        # flow solver must not wall it off, the geodesic distance must run
        # through it, and the network's pore-voxel list must include it.
        # Marking it with its own code in the stored array made all three
        # treat it as rock, and the first symptom was five pore voxels with an
        # infinite distance from the inlet.
        if cfg.biofilm_code is not None:
            gc, nb = coat_with_biofilm(g, cfg.biofilm_code, cfg.biofilm_layers)
            n_bf += nb
            bfilm[made] = (gc == cfg.biofilm_code)
            solve_mat.append(gc)
        else:
            solve_mat.append(g)
        mat[made] = g
        gdf[made] = geodesic(g)
        bad_d = assert_finite_distance(gdf[made], g)
        if bad_d:
            raise SystemExit("structure %d: %s" % (made, bad_d))
        edt[made] = ndimage.distance_transform_edt(g == PORE)
        # The relaxation time and the pressure drop from your settings now
        # actually reach the flow solver. They did not before: it used its own
        # defaults, so tau and delta_P were read, reported, and ignored.
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
            _probe = stokes(g, nit=_nit, force=1e-6, tau_lb=cfg.tau,
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
        v = stokes(g, nit=_nit, force=_force, tau_lb=cfg.tau,
                   tol=float(cfg.ns_converge))
        vels.append(v)
        # THE MEAN AXIAL VELOCITY, not the mean speed. The transport solver
        # normalises by the axial mean, so a Peclet number computed here from
        # the speed describes a different flow from the one being solved.
        # They differ by the tortuosity -- measured 1.140 on a real Stokes
        # field, and geometry dependent, so it is not even a constant offset.
        sp = np.sqrt((v[:3] ** 2).sum(0))
        _ax = float(v[0][g == PORE].mean())
        _spd = float(sp[g == PORE].mean())
        umean_lat.append(_ax if _ax > 0.1 * _spd else _spd)
        _pe_lo = cfg.pressure_for_peclet(umean_lat[-1], pe_lo, g.shape[0])
        _pe_hi = cfg.pressure_for_peclet(umean_lat[-1], pe_hi, g.shape[0])
        if not by_pressure:
            print("     to reach Peclet %g to %g on this one the pressure "
                  "gradient runs %.3g to %.3g" % (pe_lo, pe_hi, _pe_lo, _pe_hi),
                  flush=True)
        print("  structure %3d  phi=%.3f  gdf_max=%5.1f  umean=%.2e"
              % (made, float((g == PORE).mean()), gdf[made].max(),
                 float(sp[g == PORE].mean())), flush=True)
        made += 1

    # ---------------- 2. transport ---------------------------------------
    S = G * a.n_sets
    T, C = a.n_times, a.n_species
    conc = np.zeros((S, T, C) + shape, np.float32)
    vel = np.zeros((S, 3) + shape, np.float32)
    gi = np.zeros(S, np.int32)
    par = np.zeros((S, len(pnames)), np.float32)
    tn = np.zeros((S, T), np.float32)
    settled = np.ones(S, bool)
    n_bad = 0
    k = 0
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
            # The conditions the run was actually given, not three literals
            # that happened to be near them. See the same note in the 2D
            # builder: these were 0.1, 0.1, 0.05 while the solver used 0, 0.15
            # and 0.04.
            par[k] = [pe, da, da2, chem.ks_donor, chem.ks_acceptor,
                      chem.yield_]
            vel[k] = vels[i]
            nfo = {}
            beat = prog.start(i, pe, da, da2)
            t_run = time.monotonic()
            # The stopping rules from your settings reach the solver. They
            # did not before: both boxes were read, reported, and ignored.
            cc, tt = solve_adr(solve_mat[i], vels[i], pe, da, T, C,
                               steps=a.adr_steps, info=nfo, chem=chem,
                               da_abio=da2, on_progress=beat,
                               tol=cfg.ade_converge,
                               max_steps=int(cfg.ade_max_iT))
            prog.step(i, pe, da, da2, nfo, time.monotonic() - t_run)
            if (a.save_vti or a.save_png) and k < a.save_runs:
                write_vti_and_png.save_run(
                    os.path.splitext(a.out)[0] + "_fields", k, cc, tt,
                    species, mat=mat[i], vel=vels[i],
                    want_vti=a.save_vti, want_png=a.save_png,
                    spacing=cfg.dx,
                    # Label it with the Damkohler that is actually running.
                    # "Da=0" on an abiotic-only run reads as no reaction at
                    # all, when in fact the abiotic one is going.
                    note="Pe=%.3g  Da_bio=%.3g  Da_abio=%.3g"
                         % (pe, da, da2))
            conc[k] = cc
            tn[k] = tt
            settled[k] = bool(nfo.get("converged", True))
            for msg in check_physics(cc, mat[i], chem=chem, da=da, da_abio=da2):
                print("     PHYSICS run %d (Pe=%.2f Da_bio=%.2f "
                      "Da_abio=%.2f): %s" % (k, pe, da, da2, msg), flush=True)
                n_bad += 1
            reach = int((conc[k][-1, 0].reshape(nx, -1).max(1) > 1e-3).sum())
            if reach < 0.5 * nx:
                print("     WARNING run %d: the front only reached %d of %d "
                      "slices" % (k, reach, nx), flush=True)
            k += 1
    prog.finish()
    if cfg.biofilm_code is not None:
        print("  %d voxels were marked as biofilm (code %d), %d layer(s) "
              "thick on the grain surfaces" % (n_bf, cfg.biofilm_code,
                                               cfg.biofilm_layers))

    if n_bad:
        raise SystemExit("%d physics complaint(s): this file is not usable as "
                         "training data, and writing it would hide that." % n_bad)

    # ---------------- 3. write -------------------------------------------
    scale = np.maximum(
        conc.reshape(S * T * C, -1).max(1).reshape(S, T, C).max((0, 1)), 1e-6)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with h5py.File(a.out, "w") as h:
        h.attrs["n_samples"] = S
        h.attrs["shape"] = np.array(shape, np.int32)
        h.attrs["species"] = np.array([s.encode() for s in species])
        h.attrs["param_names"] = np.array([s.encode() for s in pnames])
        h.attrs["dimension"] = 3
        h.attrs["source"] = b"generated_3d"
        # The settings that made it, stored inside it.
        h.attrs["settings_source"] = str(cfg.source).encode()
        h.attrs["settings"] = cfg.to_json().encode()
        gg = h.create_group("geom")
        gg.create_dataset("gid", data=np.arange(G, dtype=np.int32))
        gg.create_dataset("material", data=mat, compression="gzip")
        gg.create_dataset("gdf", data=gdf, compression="gzip")
        gg.create_dataset("edt", data=edt, compression="gzip")
        if bfilm.any():
            # Where the biofilm was. Kept apart from the material array so
            # that everything downstream keeps treating those voxels as the
            # pore space they are.
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
        sg.create_dataset("settled", data=settled)
        d = sg.create_dataset("conc", data=conc.astype(np.float16),
                              compression="gzip")
        d.attrs["conc_scale"] = scale.astype(np.float32)
        sg.create_dataset("velocity", data=vel.astype(np.float16),
                          compression="gzip")

    mb = os.path.getsize(a.out) / 1e6
    print()
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
    print("  %d structures, %d runs, %d snapshots, %d chemicals, grid %d x %d x %d"
          % (G, S, T, C, nx, ny, nz))
    n_uns = int((~settled).sum())
    if n_uns:
        print()
        print("  " + "-" * 66)
        print("  %d of %d runs stopped at the step cap of %d before settling."
              % (n_uns, S, int(cfg.ade_max_iT)))
        print("  They are valid transients and fine to train on, but their last")
        print("  snapshot is as far as the integration got, not a steady state,")
        print("  and t = 1 should not be read as one. Which runs is recorded in")
        print("  samples/settled inside the file.")
        print("  To settle them, raise 'Never take more transport steps than'")
        print("  and run again. It is the low-Peclet runs that need it.")
        print("  " + "-" * 66)
    print("  physics checked on every run: concentrations stay at or below the")
    print("  inlet value, no checkerboard, nothing reaches the outlet before it")
    print("  has crossed the sample, and no chemical is a copy of another.")
    print()
    print("next, exactly as in 2D:")
    print("  python ../model/train.py --data %s --out runs/3d" % a.out)


if __name__ == "__main__":
    main()
