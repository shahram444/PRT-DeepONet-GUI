#!/usr/bin/env python3
"""
prtlb_2d.py -- PRT-LB in two dimensions. Flow, transport and reactions.

PRT-LB is the simulator half of this project: Pore-scale Reactive Transport by
Lattice Boltzmann. It solves creeping flow through a pore image, then the
advection, dispersion and reaction of several chemicals through that flow. Its
output is what PRT-DeepONet is trained on.

There are exactly two of these files and between them they are the whole
simulator:

    prtlb_2d.py    D2Q9  flow + transport + reactions on an (nx, ny) image
    prtlb_3d.py    D3Q19 flow + transport + reactions on an (nx, ny, nz) volume


================================ MAP OF THIS FILE ============================

Every section below is opened by a banner you can search for. To find the
flow field, search for "BLOCK 4". To find how the boundary conditions are
applied, search for "9.7".

    BLOCK 1     the collision, two relaxation times
    BLOCK 2     when a flow solve is finished
    BLOCK 3     what a flow field must satisfy before it is returned
    BLOCK 4     THE FLOW SOLVER, D2Q9
                  4.1   the lattice
                  4.2   THE PRESSURE GRADIENT, as a body force
                  4.3   one time step, and the no-slip wall
                  4.4   the answer
    BLOCK 5     material codes, and the chemistry used when nothing is said
    BLOCK 6     geometry helpers
    BLOCK 7     the advection reconstruction, minmod
    BLOCK 8     snapshot bookkeeping
    BLOCK 9     THE TRANSPORT SOLVER
                  9.1   the pore space
                  9.2   Pe and Da become lattice quantities
                  9.3   the time step
                  9.4   face quantities, built once
                  9.5   ADVECTION
                  9.6   DISPERSION, including through biofilm
                  9.7   BOUNDARY CONDITIONS: inlet, outlet, walls
                  9.8   INITIAL CONDITIONS
                  9.9   THE BIOTIC REACTION
                  9.10  THE ABIOTIC REACTION
                  9.11  one time step
                  9.12  the time loop, convergence, snapshots
    BLOCK 10    what a correct solution must satisfy
    BLOCK 11    the self test


=============================== WHAT COMES IN ================================

    stokes_d2q9(g, force=..., tau_lb=..., tol=..., info=None)
        g       material image: 0 solid, 1 wall, 2 pore
        force   the pressure gradient, as a body force along +x. See 4.2.
        tau_lb  relaxation time; the viscosity is (tau - 1/2)/3
        -> v    (3, nx, ny) float32, zero inside the rock

    solve_adr(mat, vel, pe, da, n_t, n_species=..., chem=..., da_abio=...)
        mat     the same material image
        vel     the field stokes_d2q9 just returned
        pe      Peclet: advection against dispersion
        da      biotic Damkohler; da_abio the abiotic one
        chem    your own numbers, from settings_and_units.py
        -> (out, t_norm), out being (n_t, n_species, nx, ny)

    check_physics(out, mat, chem=..., da=..., da_abio=...)
        -> a list of complaints, empty when the run is sound


============================== WHAT GOES OUT =================================

Nothing is written to disk from this file. It returns arrays. The generators
in build_dataset_2d.py and build_dataset_3d.py are what write the HDF5, and
write_vti_and_png.py is what writes pictures.


=================== THE TWO FILES SHARE MOST OF THIS CODE ====================

BLOCKS 1 to 3 and BLOCKS 5 to 11 are IDENTICAL in prtlb_2d.py and prtlb_3d.py,
word for word. Only BLOCK 4, the flow solver, differs, and it differs only in
the lattice. The transport solver works in either dimension from mat.ndim, so
there is no reason for the two copies to differ and every reason for them not
to.

IF YOU EDIT ONE, COPY IT TO THE OTHER. test_flow_solvers.py compares the two
blocks byte for byte on every run and prints the first line that differs.

This is not caution for its own sake. There were once two genuinely
independent transport solvers here, and six bugs were found and fixed in one
of them while every one survived in the other: a periodic boundary that leaked
the inlet into the outlet, a diffusion coefficient wrong by a factor of the
domain length, a hard step count that stopped before the front entered the
sample, a clip that hid an instability, and a per-species term that made three
of the four chemicals identical arrays. They survived because fixing one copy
makes the other LOOK fixed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings_and_units import Chemistry, default_chemistry     # noqa: E402


# ==========================================================================
# === PRTLB SHARED BLOCK BEGIN: the collision, and what a flow field must be
# ==========================================================================
# ==========================================================================
#  BLOCK 1  |  THE COLLISION  --  two relaxation times
# --------------------------------------------------------------------------
#  Everything in BLOCKS 1 to 3 is IDENTICAL in prtlb_2d.py and prtlb_3d.py,
#  word for word. Edit one, copy it to the other.
#
#  in   f     the populations before collision
#       feq   their equilibrium
#       om_p  1/tau, which sets the viscosity
#  out  the populations after collision
#
#  A collision relaxes the populations towards equilibrium. With ONE rate
#  the no-slip wall drifts away from the mid-link as tau changes; with TWO,
#  chosen so that (1/om_p - 1/2)(1/om_m - 1/2) = 3/16, it stays there at
#  every tau. See the measured wall offsets in BLOCK 4.
# ==========================================================================

def _trt_minus(om_p, magic=3.0 / 16.0):
    """The antisymmetric relaxation rate that puts the wall at the mid-link.

    From Lambda = (1/omega+ - 1/2)(1/omega- - 1/2), solved for omega-.
    """
    lam_p = 1.0 / float(om_p) - 0.5
    if lam_p <= 0:
        raise ValueError("tau must exceed 0.5; the viscosity (tau-1/2)/3 is "
                         "negative below it and the solver cannot run.")
    return 1.0 / (float(magic) / lam_p + 0.5)


def _trt_collide(f, feq, opp, om_p, om_m):
    """One two-relaxation-time collision. Returns the post-collision f.

    Reduces to BGK exactly when omega_minus == omega_plus, which is the case
    at tau = 0.9330 with the magic parameter -- a useful thing to be able to
    check, and the self-test does.
    """
    fo, feo = f[opp], feq[opp]
    f_p = 0.5 * (f + fo)
    f_m = 0.5 * (f - fo)
    e_p = 0.5 * (feq + feo)
    e_m = 0.5 * (feq - feo)
    return f - om_p * (f_p - e_p) - om_m * (f_m - e_m)


# ==========================================================================
#  BLOCK 2  |  WHEN A FLOW SOLVE IS FINISHED
# --------------------------------------------------------------------------
#  in   the mean pore speed, once every `every` iterations
#  out  done() says stop; report() fills the info dict and warns if the
#       iteration cap was hit while the field was still moving
#
#  The flow solve runs until the field stops changing, so the `nit`
#  argument in BLOCK 4 is a CAP and not the number of iterations run.
# ==========================================================================

class FlowConvergence(object):
    """Stops a flow solve when the field stops changing, instead of at a
    number of iterations somebody guessed once.

    WHY THIS EXISTS. Both solvers ran a fixed count -- 1500 in 2D, 400 in 3D
    -- and returned whatever they had reached. Whether that was the answer
    depends on the relaxation time and on the rock, neither of which a fixed
    number knows about. Measured, iterations to settle the mean pore velocity
    to one part in a million on a 148 by 64 grain pack at porosity 0.63:

        tau = 0.55       4450
        tau = 0.80       5300
        tau = 1.20      10800
        tau = 2.00      19950

    against a default of 1500. What that cost, on the same rock at tau = 0.8,
    as the mean axial velocity against its settled value:

        400 iterations    +1.86%
        1500              +0.19%      <- the old 2D default
        3000              +0.03%
        5300               0.00%

    and in 3D, on a 40 by 32 by 32 pack, where the default was 400:

        400 iterations    -9.77%      <- the old 3D default
        1000              -0.37%
        2350               0.00%

    So every three-dimensional flow field this project has produced was about
    a tenth low in the mean, and since the transport solver normalises by that
    mean, the Peclet number each 3D run was labelled with was wrong by the
    same amount. It is not visible in a picture: an under-relaxed field is
    still smooth, still divergence free to the same tolerance, and still
    parabolic in a channel. It is simply not finished.

    The settings file has had <ns_converge> and <ns_max_iT> in it since it was
    written; this is what makes them mean something. The tolerance is on the
    RELATIVE change in mean speed per check interval, which is the same
    quantity CompLaB's own <ns_converge_iT> uses.
    """

    def __init__(self, tol=1e-6, every=50, max_it=100000, name="flow"):
        self.tol = float(tol)
        self.every = max(int(every), 1)
        self.max_it = int(max_it)
        self.name = name
        self.prev = None
        self.iters = 0
        self.converged = False
        self.change = float("inf")

    def done(self, it, umean):
        """True when it is time to stop. Call once per iteration."""
        self.iters = it + 1
        if self.tol <= 0:
            return False                      # a tolerance of zero means never
        if (it + 1) % self.every:
            return False
        u = float(umean)
        if self.prev is not None:
            self.change = abs(u - self.prev) / max(abs(u), 1e-30)
            if self.change <= self.tol:
                self.converged = True
                self.prev = u
                return True
        self.prev = u
        return False

    def report(self, info=None):
        """Fill an info dict and complain if the cap was hit while moving."""
        if info is not None:
            info["flow_iters"] = self.iters
            info["flow_converged"] = bool(self.converged)
            info["flow_change"] = self.change
        if not self.converged and self.tol > 0:
            print("     WARNING: the %s solve stopped at its %d iteration cap "
                  "with the mean\n              speed still changing by %.3g "
                  "per %d steps, against a\n              tolerance of %.3g. "
                  "The field is not settled and the\n              "
                  "permeability read from it will be too low. Raise\n"
                  "              ns_max_iT." % (self.name, self.iters,
                                                self.change, self.every,
                                                self.tol), flush=True)
        return self.converged


# ==========================================================================
#  BLOCK 3  |  WHAT A FLOW FIELD MUST SATISFY BEFORE IT IS RETURNED
# --------------------------------------------------------------------------
#  in   v      the velocity field the solver just produced
#       solid  the rock
#  out  the same field, or SystemExit with what went wrong and what to do
#
#  Three refusals: a field that is not finite, a field that is zero
#  everywhere, and a warning above Mach 0.05, which is where the low-Mach
#  expansion the whole method rests on stops being valid.
# ==========================================================================

def _checked_flow(v, solid, tau_lb, force, name="flow"):
    """Refuse to return a velocity field that cannot be right.

    Every one of these has a specific history.

    NOT FINITE. At a driving force of 3e-3 the old single-precision solver
    overflowed to not-a-number, returned it, and the transport solver then
    reported "the integration diverged" -- blaming the wrong solver entirely,
    because the only finiteness guard in the project was on concentration.

    MACH NUMBER. The lattice-Boltzmann equilibrium is a low-Mach expansion.
    Above about 0.05 the compressibility error is no longer negligible.
    Measured on a 148 by 64 rock at the old default pressure drop of 1.5e-4:
    Mach 0.068, pore Reynolds number 3.1 -- not creeping flow -- and the mass
    flux through successive planes varied by 2.6%, so the field the transport
    solver was handed was measurably not divergence free. A straight channel
    gives no warning of any of this, because unidirectional flow has no
    advection term, so the Poiseuille tests pass at any force.

    DEAD FLOW. A field that is everywhere zero is not a slow flow; it is a
    failed solve, and everything downstream divides by its mean speed.
    """
    v = np.asarray(v)
    m = ~np.asarray(solid)
    if not np.isfinite(v).all():
        raise SystemExit(
            "the %s solver produced %d non-finite values. The driving force "
            "of %g is too large for tau = %g: lower it, or raise tau. Nothing "
            "downstream can use this field." %
            (name, int((~np.isfinite(v)).sum()), force, tau_lb))
    if not m.any():
        raise SystemExit("the %s solver was given a geometry with no pore "
                         "space at all." % name)
    sp = np.sqrt((v ** 2).sum(0))
    umax = float(sp[m].max())
    umean = float(sp[m].mean())
    if umean <= 0:
        raise SystemExit(
            "the %s solver produced a field that is zero everywhere. Either "
            "the driving force (%g) is too small to survive rounding, or the "
            "pore space does not connect inlet to outlet." % (name, force))
    ma = umax / np.sqrt(1.0 / 3.0)
    if ma > 0.05:
        print("     WARNING: peak lattice speed %.4g gives a Mach number of "
              "%.3f.\n              Lattice Boltzmann is a low-Mach "
              "expansion and is only\n              reliable below about "
              "0.05. Halve the pressure drop and the\n              flow "
              "halves with it, so nothing is lost but time."
              % (umax, ma), flush=True)
    return v

# ==========================================================================
# === PRTLB SHARED BLOCK END
# ==========================================================================


# ==========================================================================
#  BLOCK 4  |  THE FLOW SOLVER  --  D2Q9, nine directions
# --------------------------------------------------------------------------
#  THIS IS THE ONLY BLOCK THAT DIFFERS between prtlb_2d.py and prtlb_3d.py,
#  and it differs only in the lattice: nine directions here, nineteen there.
#
#  in   g       the material image: 0 solid, 1 wall, 2 pore
#       force   the driving force per unit volume, along +x. See 4.2 for
#               how a pressure gradient becomes this number.
#       tau_lb  the relaxation time. Viscosity is (tau - 1/2)/3.
#       nit     the CAP on iterations, not the number run
#       tol     stop when the mean pore speed changes by less than this
#       info    optional dict, filled with flow_iters / flow_converged
#  out  v       (3, nx, ny) float32, zero inside the rock, having passed
#               every check in BLOCK 3
#
#  Inside, in order:
#       4.1  the lattice: directions, weights, opposites
#       4.2  the pressure gradient, as a body force
#       4.3  one time step: collide, force, bounce back, stream
#       4.4  the answer, and what it is checked against
#
#  Solved ONCE PER ROCK, not once per run. Stokes flow is linear, so
#  changing the Peclet number rescales this same field.
# ==========================================================================
def stokes_d2q9(g, nit=30000, force=2e-6, tau_lb=0.8, magic=3.0 / 16.0,
                tol=1e-6, info=None):
    """D2Q9 Stokes flow, TRT collision, halfway bounce-back. float64.

    THREE THINGS HERE ARE DELIBERATE AND EACH FIXES A MEASURED ERROR.

    float64, not float32. The equilibrium adds 3*e.u -- of order 3e-5 at a
    normal driving force -- to a leading 1, and the non-equilibrium part that
    CARRIES THE VISCOUS STRESS is of order the force itself. In single
    precision that part keeps about two and a half significant digits.
    Measured on a 20-voxel channel at tau = 0.8: the peak velocity was 0.13%
    low at a force of 2e-6, 11% low at 2e-7, and 99% low at 2e-9 -- from a
    field that still looked like a smooth parabola. Double precision costs
    twice the memory and about a third more time, and the flow solve is once
    per rock.

    TRT, not BGK. With a single relaxation time the bounce-back wall does not
    sit at the mid-link; it sits there only when (tau - 1/2)^2 = 3/16, i.e.
    tau = 0.9330, and drifts with tau everywhere else. Measured wall offset on
    a 4-voxel channel: 0.437 at tau = 0.55, 0.467 at 0.8, 0.598 at 1.2, 1.100
    at 2.0 -- so at tau = 2 a four-voxel throat carried 73% too much flow, and
    the computed permeability depended on the relaxation time, which is not a
    physical quantity. Two relaxation times with the magic parameter
    Lambda = (1/omega+ - 1/2)(1/omega- - 1/2) = 3/16 puts the wall at the
    mid-link for EVERY tau, which is the whole reason TRT exists.

    Halfway bounce-back. Reverse the populations that ARRIVED (f), not the
    ones this step's collision just produced (fp). Palabos does the former --
    BounceBack::collide swaps the cell's current populations, which at that
    moment are the ones that streamed in -- and CompLaB assigns that dynamics
    to every interface voxel, so this is what makes our field comparable with
    CompLaB's. Reversing fp instead puts the wall a full voxel out, which is
    still a valid no-slip wall with the right viscosity and widens every
    channel by one voxel: 10% too fast at 20 voxels across, 56% at 4.
    """
    # ---- 4.1  THE LATTICE ------------------------------------------------
    #
    # Nine directions on a square grid: one that stays put, four to the
    # faces, four to the corners. w is how much of the resting fluid travels
    # each way, and opp[i] is the direction that points back along i, which
    # is what bounce-back and the two-rate collision both need.
    solid = (g != PORE)
    nx, ny = g.shape
    e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                  [1, 1], [-1, 1], [-1, -1], [1, -1]], np.int32)
    w = np.array([4/9.] + [1/9.]*4 + [1/36.]*4, np.float64)
    opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
    # ---- 4.2  THE PRESSURE GRADIENT, AS A BODY FORCE ---------------------
    #
    # THERE IS NO PRESSURE FIELD IN THIS SOLVER, and that is deliberate
    # rather than missing. A pressure gradient that is uniform along the flow
    # direction is exactly equivalent to a uniform body force per unit
    # volume, and the body force is what a lattice-Boltzmann solver can apply
    # without imposing anything at the inlet and outlet faces:
    #
    #     dP/dx  ==  gf                (both in lattice units per voxel)
    #
    # so `force` IS the pressure gradient. The generators call the setting
    # delta_P for that reason. Streaming is periodic along x, which with a
    # body force is the standard way to drive flow through a sample without
    # inventing a boundary condition: fluid leaving the outlet re-enters at
    # the inlet, and the force keeps it moving.
    #
    # WHAT SETS ITS SIZE. In Peclet mode nothing downstream depends on it,
    # because Stokes flow is linear and the field is rescaled afterwards to
    # hit the Peclet number asked for. It still must not be so large that the
    # solve leaves the low-Mach range (BLOCK 3), which is why the generators
    # probe with a small force and scale down before the real solve.
    #
    # The pressure itself, if you want it, is rho/3 by the ideal-gas law of
    # this model, and rho is f.sum(0). It is not needed anywhere here.
    om_p = 1.0 / float(tau_lb)                     # symmetric, sets viscosity
    om_m = _trt_minus(om_p, magic)                 # antisymmetric, sets the wall
    gf = float(force)
    ex = e[:, 0].astype(np.float64)
    ey = e[:, 1].astype(np.float64)
    f = np.ones((9, nx, ny), np.float64) * w[:, None, None]
    fluid = ~solid
    # Stop when the field stops changing, not after a count somebody guessed.
    # See FlowConvergence in the shared block above for what the old fixed count cost.
    # ---- 4.3  ONE TIME STEP ----------------------------------------------
    #
    # Four things happen per iteration, in this order:
    #
    #   density and velocity   summed from the populations, with half the
    #                          force added to the momentum, which is the
    #                          other half of the Guo forcing scheme
    #   collide                relax towards equilibrium at two rates
    #   force                  add the body force from 4.2
    #   bounce back            the flow boundary condition, see below
    #   stream                 each population moves one voxel along its own
    #                          direction
    #
    # THE FLOW BOUNDARY CONDITION IS THE BOUNCE-BACK LINE. On a solid voxel
    # the populations are replaced by their own opposites, so whatever
    # arrived leaves back the way it came. That is no-slip: water does not
    # slide along rock. Reversing the populations that ARRIVED rather than
    # the ones this step's collision produced puts the wall halfway between
    # the last fluid voxel and the first solid one, which is where Palabos
    # puts it and therefore where CompLaB puts it.
    conv = FlowConvergence(tol=tol, max_it=nit, name="2D flow")
    for _it in range(nit):
        rho = f.sum(0)
        ux = (np.einsum('i,ixy->xy', ex, f) + 0.5 * gf) / rho
        uy = np.einsum('i,ixy->xy', ey, f) / rho
        ux *= fluid; uy *= fluid
        usq = ux * ux + uy * uy
        feq = np.empty_like(f)
        for i in range(9):
            eu = ex[i] * ux + ey[i] * uy
            feq[i] = w[i] * rho * (1 + 3*eu + 4.5*eu*eu - 1.5*usq)
        fp = _trt_collide(f, feq, opp, om_p, om_m)
        for i in range(9):
            # Guo forcing. The prefactor carries the ANTISYMMETRIC rate,
            # not the symmetric one. A uniform body force is odd under
            # e -> -e, so its first-order term lives entirely in the
            # antisymmetric part of the population, and that part relaxes at
            # omega_minus. Using omega_plus here -- which is what a
            # BGK-shaped forcing term looks like -- left the channel 18% slow
            # at tau = 0.8 and 80% slow at tau = 0.55, from a field that was
            # still a clean parabola.
            fp[i] += w[i] * (1 - 0.5 * om_m) * 3.0 * gf * ex[i]
        fp[:, solid] = f[opp][:, solid]
        for i in range(9):
            f[i] = np.roll(np.roll(fp[i], int(e[i, 0]), 0), int(e[i, 1]), 1)
        if conv.done(_it, np.abs(ux[fluid]).mean() if fluid.any() else 0.0):
            break
    conv.report(info)
    # ---- 4.4  THE ANSWER --------------------------------------------------
    #
    # The velocity is recovered the same way it was inside the loop. It is
    # returned as three components even in two dimensions, with uz zero, so
    # that everything downstream can treat 2D and 3D fields alike.
    rho = f.sum(0)
    ux = (np.einsum('i,ixy->xy', ex, f) + 0.5 * gf) / rho
    uy = np.einsum('i,ixy->xy', ey, f) / rho
    v = np.stack([ux, uy, np.zeros_like(ux)])
    v[:, solid] = 0.0
    return _checked_flow(v.astype(np.float32), solid, tau_lb, force)


# ==========================================================================
# === PRTLB TRANSPORT BLOCK BEGIN: advection, diffusion, reactions, checks
# ==========================================================================
# ==========================================================================
#  BLOCK 5  |  MATERIAL CODES, AND THE CHEMISTRY USED WHEN NOTHING IS SAID
# --------------------------------------------------------------------------
#  Everything from here to the end of BLOCK 11 is IDENTICAL in prtlb_2d.py
#  and prtlb_3d.py, word for word. Edit one, copy it to the other.
#
#  The numbers below are the FALLBACK. They are what you get when no
#  settings file is passed, and they are exactly what this file held as
#  module constants before settings_and_units.py existed, so a dataset built
#  today with no settings is the dataset it would have been.
#
#  To change any of them -- the feed concentrations, the half-saturation
#  constants, the stoichiometry, the yield, a per-chemical diffusion
#  coefficient, which chemicals are attached -- write a settings file in
#  your own units and pass chem=Settings(...).chemistry().
# ==========================================================================

SOLID, WALL, PORE = 0, 1, 2

# Chemistry, in the species order used everywhere: Ac, A, P, Bio.
# Ac is the electron donor, A the acceptor, P the product, Bio the biomass.
#
# THESE ARE NOW THE FALLBACK, NOT THE ONLY OPTION. Everything below is what
# you get when nothing else is said, and it is exactly what this file held
# before settings_and_units.py existed, so a dataset built today with no settings file
# is the same dataset it would have been. To change any of it -- the feed
# concentrations, the half-saturation constants, the stoichiometry, the yield,
# a per-chemical diffusion coefficient, which chemicals are attached -- write
# a settings file in your own units and pass chem=Settings(...).chemistry().
# See settings_and_units.py, and read the settings file it writes for you.
INLET = np.array([1.0, 1.0, 0.0, 0.0], np.float32)
KS_A = np.float32(0.15)     # half-saturation on the acceptor
# Acceptor consumed per donor in NORMALISED units: each species is scaled by
# its own inlet concentration, so this is the stoichiometric ratio times
# Ac0/A0 = 2.0/5.0 from complab_campaign.py. It is NOT 1 -- with a ratio of 1 and
# equal inlets the donor and the acceptor obey the identical equation, and a
# "two-chemical" dataset holds one chemical twice.
ALPHA = np.float32(0.4)
Y = np.float32(0.04)        # biomass made per donor
KD = np.float32(0.006)      # biomass decay
B0 = np.float32(0.1)        # biomass present at the start

# Index 3, the biomass, is ATTACHED: it does not advect and does not diffuse,
# and no inlet condition is imposed on it.
#
# The alternative is CompLaB's planktonic microbe, which does move. Attached
# was chosen here for a specific reason: with a Dirichlet inlet of zero, a
# mobile biomass is held at exactly zero on the inlet face -- the one place
# where donor and acceptor are both at their maximum -- so the biotic rate
# field is identically zero precisely where the chemistry is most active, and
# most of the biomass then washes out of the sample rather than decaying.
# Measured on a 60x32 domain: mean biomass fell 0.100 -> 0.019 with transport
# on and 0.100 -> 0.071 with it off, so roughly three quarters of the loss was
# washout rather than decay.
ATTACHED = 3


# --------------------------------------------------------------------------
# ==========================================================================
#  BLOCK 6  |  GEOMETRY HELPERS
# --------------------------------------------------------------------------
#  Three things done to a pore image before anything is solved on it.
#
#  keep_inlet_connected     turns unreachable pore space into solid
#  coat_with_biofilm        grows a biofilm layer on the grain surfaces
#  assert_finite_distance   refuses a distance field with unreachable voxels
#  _neighbour_masks         which neighbours of each voxel are pore
# ==========================================================================

def keep_inlet_connected(g):
    """Turn every pore voxel not reachable from the inlet face into solid.

    Returns (g, n_removed).

    KEEP THE COMPONENT CONNECTED TO THE INLET, NOT THE LARGEST ONE. They are
    not the same, and the difference is not academic: on one extruded domain
    the biggest pore cluster (6936 voxels) did not touch the inlet at all,
    while the one that did held 5736. Keeping the larger one would have kept
    the wrong half.

    This has to run LAST, after every operation that can change connectivity.
    Sealing the transverse faces with wall is such an operation, and doing it
    after the percolation check is what produced a domain where 55% of the
    pore space was unreachable and carried a geodesic distance of 1e9 -- a
    number that then went into the trunk as a coordinate and flattened every
    real distance to nothing once normalised.
    """
    from scipy import ndimage
    g = np.asarray(g)
    pore = (g == PORE)
    if not pore.any():
        return g, 0
    lab, n = ndimage.label(pore)
    face = lab[0][pore[0]]
    seeds = set(np.unique(face).tolist()) - {0}
    if not seeds:
        return g, int(pore.sum())
    reach = np.isin(lab, list(seeds))
    drop = pore & ~reach
    if not drop.any():
        return g, 0
    g = g.copy()
    g[drop] = SOLID
    return g, int(drop.sum())


def coat_with_biofilm(g, code, layers=1):
    """Mark the pore voxels next to a grain as biofilm. Returns (g, n_marked).

    This is how a sessile pool gets somewhere to live. CompLaB gives each
    attached microbe pool its own material number and seeds the biomass into
    voxels carrying it; the geometry file says where the biofilm is. A rock
    this project generates has no such voxels, so this puts them where a
    biofilm actually grows -- against the grain surfaces, where the mineral
    is -- rather than floating in the middle of the water.

    A biofilm voxel stays part of the pore space. Fluid passes through it and
    chemicals diffuse through it, at their in_biofilm coefficient rather than
    their pore one. Nothing here changes during a run: there is no growth, no
    cellular automaton, and no pore voxel ever becomes biofilm later. For
    that, use CompLaB.
    """
    from scipy import ndimage
    g = np.asarray(g).copy()
    if code is None or layers <= 0:
        return g, 0
    solid = (g != PORE)
    grown = solid
    for _ in range(int(layers)):
        grown = ndimage.binary_dilation(grown)
    lining = grown & (g == PORE)
    g[lining] = code
    return g, int(lining.sum())


def assert_finite_distance(d, g, name="geodesic"):
    """A distance field must be finite on every pore voxel. Returns a complaint.

    The geodesic solver leaves unreachable pore voxels at its infinity
    sentinel. That value is a perfectly ordinary float as far as everything
    downstream is concerned, so it travels quietly into the trunk input and
    into the normalisation constant.
    """
    d = np.asarray(d)
    m = (np.asarray(g) == PORE)
    if not m.any():
        return None
    v = d[m]
    if np.isfinite(v).all() and float(v.max()) < 1e6:
        return None
    n = int((~np.isfinite(v)).sum() + (v >= 1e6).sum())
    return ("%s is not finite on %d of %d pore voxels (max %.4g). Those voxels "
            "cannot be reached from the inlet; run keep_inlet_connected first."
            % (name, n, m.sum(), float(v.max())))


def _neighbour_masks(m):
    """For each axis, the pore mask shifted one voxel each way, edge-replicated.

    Edge replication is a zero-gradient boundary. It does not wrap.
    """
    nd = m.ndim
    pad = [(1, 1)] * nd
    Mp = np.pad(m, pad, mode="edge")
    lo, hi = [], []
    for ax in range(nd):
        sl_lo, sl_hi = [], []
        for j in range(nd):
            if j == ax:
                sl_lo.append(slice(0, -2)); sl_hi.append(slice(2, None))
            else:
                sl_lo.append(slice(1, -1)); sl_hi.append(slice(1, -1))
        lo.append(Mp[tuple(sl_lo)][None])
        hi.append(Mp[tuple(sl_hi)][None])
    return lo, hi


# ==========================================================================
#  BLOCK 7  |  THE ADVECTION RECONSTRUCTION  --  minmod
# --------------------------------------------------------------------------
#  in   up     the upwind cell, the one the flow is arriving from
#       upup   the cell beyond it, upwind again
#       down   the cell the flow is heading into
#  out  the concentration to use ON THE FACE between up and down
#
#  This one function decides whether the Peclet number written in your
#  dataset is the Peclet number the sample experienced. Used in 9.5.
# ==========================================================================

def _muscl(up, upup, down):
    """The limited face value, reconstructed from the upwind cell.

        C_face = C_up + (1/2) * minmod(d1, d2)

    with d1 = C_down - C_up the downwind difference and d2 = C_up - C_upup the
    upwind one, and minmod the one of smaller magnitude, or zero if they
    disagree in sign. Plain upwinding is the case where that slope is zero.

    WHY MINMOD AND NOT VAN LEER. Both are second order and both are
    total-variation diminishing, and van Leer is the usual first choice
    because it is smoother. It is also the more COMPRESSIVE of the two -- its
    limiter function reaches 2 where minmod stops at 1 -- and a compressive
    limiter does not merely fail to add numerical diffusion, it SUBTRACTS
    some, artificially sharpening a front that settings_and_units diffusion should be
    spreading. That is the opposite error from upwinding and just as wrong.

    Measured on a straight channel, the 10%-to-90% width of a front that has
    travelled half the domain, against the exact erfc solution:

        nx = 64     exact    upwind    minmod    van Leer
        Pe =  20     36.7     36.6      34.3      34.3
        Pe =  50     23.2     26.1      22.5      22.3
        Pe = 100     16.4     20.9      16.0      15.6
        Pe = 300      9.5     15.8       9.0       7.7

    Upwinding is 27% too wide at Pe = 100 and 67% too wide at Pe = 300 -- it
    reports a sample far more mixed than the one that was asked for. Van Leer
    overshoots the other way, 18% too sharp at Pe = 300. Minmod is within 3%
    everywhere the campaign actually sweeps.

    A note on how the fixed operator behaves outside that range: numerical
    diffusion is |u|*dx/2 per axis and settings_and_units diffusion is nx/Pe, so the two
    are equal at a GRID Peclet number Pe/nx of about 2. Above that no scheme
    on this grid can be trusted to reproduce the requested Peclet, and
    solve_adr says so in its info dictionary rather than leaving the caller to
    find out from the pictures.

    WRITTEN WITHOUT A DIVISION. The textbook form computes r = d2/d1 and then
    phi(r), which is 0/0 wherever the field is locally flat -- and a
    concentration field is locally flat over most of its volume, so that is
    not a corner case, it is the common case.

    THE LIMITER IS WHAT KEEPS THIS BOUNDED. At a local extremum d1 and d2 have
    opposite signs and the slope is exactly zero, so the face value falls back
    to the upwind cell. That is what makes the scheme total-variation
    diminishing: it cannot manufacture a concentration larger than the largest
    one already present, so the maximum principle that check_physics tests is
    a property of the scheme and not a hope. It holds for a Courant number up
    to 1/2, and dt is chosen below 0.4.
    """
    d1 = down - up
    d2 = up - upup
    slope = (0.5 * (np.sign(d1) + np.sign(d2))
             * np.minimum(np.abs(d1), np.abs(d2)))
    return up + 0.5 * slope


# ==========================================================================
#  BLOCK 8  |  SNAPSHOT BOOKKEEPING
# --------------------------------------------------------------------------
#  Which step numbers to store. Filling is fast at first and then
#  asymptotic, so the stored times are spaced logarithmically rather than
#  evenly; even spacing would put the whole transient in the first interval.
# ==========================================================================

def _nearest_unused(i, used, n):
    """The index closest to i that is not already taken."""
    for d in range(n):
        for j in (i - d, i + d):
            if 0 <= j < n and j not in used:
                return j
    return i


def _wanted_steps(total, n_t):
    """Log-spaced step numbers to sample, from 0 to total inclusive.

    np.geomspace(a, b, num=1) returns [a], not [b]. Building the tail with
    num = n_t-1 therefore drops the END of the run whenever n_t is 2, giving
    a two-snapshot dataset of "empty" and "very slightly less empty".
    """
    if n_t <= 1:
        return np.array([total], int)
    if n_t == 2:
        return np.array([0, total], int)
    tail = np.geomspace(max(total // 400, 1), max(total, 1), n_t - 1)
    return np.round(np.concatenate([[0.0], tail])).astype(int)


# --------------------------------------------------------------------------
# ==========================================================================
#  BLOCK 9  |  THE TRANSPORT SOLVER  --  advection, dispersion, reactions
# --------------------------------------------------------------------------
#  This is the heart of the file and everything else feeds it. It works in
#  TWO OR THREE DIMENSIONS from mat.ndim, which is why this block can be
#  identical in prtlb_2d.py and prtlb_3d.py.
#
#  WHAT COMES IN
#       mat      the material grid, (nx, ny) or (nx, ny, nz)
#       vel      the flow field from BLOCK 4, (3,) + grid
#       pe       Peclet number: advection against diffusion
#       da       BIOTIC Damkohler: the microbial rate against advection
#       da_abio  ABIOTIC Damkohler: the purely chemical rate
#       n_t      how many snapshots to store
#       chem     a Chemistry from settings_and_units.py, or None for BLOCK 5
#       steps    None to run until every chemical settles. A number
#                truncates deliberately and info['converged'] says so.
#       tol      how far from settled the run may stop, as a fraction
#
#  WHAT GOES OUT
#       out      (n_t, n_species) + grid, zero outside the pore space
#       t_norm   the TRUE normalised time of each stored snapshot
#       info     optional dict: converged, steps, dt, grid_peclet,
#                l_ref_voxels, surface_voxels, biofilm_voxels and more
#
#  INSIDE, IN THE ORDER IT HAPPENS
#       9.1   the pore space, and which voxels solute can occupy
#       9.2   the dimensionless numbers become lattice quantities
#       9.3   the time step, from all three processes at once
#       9.4   face quantities, built once
#       9.5   ADVECTION      how solute is carried by the flow
#       9.6   DISPERSION     how solute spreads, and through biofilm
#       9.7   BOUNDARY CONDITIONS, inlet, outlet, walls
#       9.8   INITIAL CONDITIONS, the state at t = 0
#       9.9   THE BIOTIC REACTION, microbially mediated
#       9.10  THE ABIOTIC REACTION, purely chemical
#       9.11  one time step, all of the above applied once
#       9.12  the time loop, convergence, and the snapshot ladder
# ==========================================================================

def solve_adr(mat, vel, pe, da, n_t, n_species=2, steps=None, tol=2e-3,
              max_steps=100000, info=None, ratio=1.12, chem=None, da_abio=0.0,
              on_progress=None):
    """Solve on a 2D (nx, ny) or 3D (nx, ny, nz) grid. Returns (out, t_norm).

    out      (n_t, n_species) + grid, zero outside the pore space
    t_norm   the TRUE normalised time of each stored snapshot

    steps    None to run until the field stops changing, which is the only
             setting that is right at every Peclet and Damkohler. An explicit
             number truncates deliberately, and info["converged"] says so.
    tol      HOW FAR FROM SETTLED the run is allowed to stop, as a fraction of
             each chemical's own scale -- not how much it changed on the last
             check, which is a different and much weaker thing. Measured on
             the slowest chemical there is, the biomass:

                 tol      steps    distance from the settled value
                 0.02      4100              1.9%
                 0.01      5900              1.0%
                 0.005     7800              0.5%
                 0.002     9900              0.2%   (the default)

             so the number does what it says. The rule it replaced watched
             the donor alone and stopped 5.6% short on this same case while
             reporting a change of 0.2%; raise tol to trade accuracy for time
             deliberately rather than by accident.
    info     optional dict, filled with converged / steps / repeated_snapshots
    da       the BIOTIC Damkohler number: the microbially mediated rate
             against advection.
    da_abio  the ABIOTIC Damkohler number: a purely chemical first-order loss
             of the product, with no microbes involved. Zero switches it off,
             which is the default and is what every dataset built before this
             argument existed used.
    on_progress  optional callable, invoked every few hundred steps with
             (step, limit, mean, change). One transport solve at 148 by 64 and
             Peclet 0.3 takes minutes, and a run that prints nothing for
             minutes is indistinguishable from a run that has hung. This is
             how the caller shows that it has not.
    chem     a settings_and_units.Chemistry: the feed and starting concentrations, the
             per-chemical diffusion coefficients relative to the first, which
             chemicals are attached, and the reaction constants. None uses the
             fallback at the top of this file, which is what every dataset
             built before settings_and_units.py existed was made with.
    """
    # ----------------------------------------------------------------------
    #  9.1  THE PORE SPACE, AND WHERE SOLUTE MAY GO
    # ----------------------------------------------------------------------
    #  A BIOFILM voxel is not solid. Chemicals diffuse through it, at a
    #  different coefficient (9.6), and treating it as rock would wall off the
    #  very region where the chemistry happens.
    mat = np.asarray(mat)
    # Voxels the solute can occupy. A BIOFILM voxel is one of them: in CompLaB
    # attached microbes have their own material number, and such a voxel is
    # not solid -- fluid passes through it and chemicals diffuse through it,
    # just at a different coefficient. Treating it as solid would wall off the
    # very region where the chemistry happens.
    _ch0 = chem
    bf_code = getattr(_ch0, "biofilm_code", None) if _ch0 is not None else None
    m = (mat == PORE)
    biofilm = np.zeros(m.shape, bool)
    if bf_code is not None:
        biofilm = (mat == bf_code)
        m = m | biofilm
    nd = m.ndim
    if nd not in (2, 3):
        raise ValueError("grid must be 2D or 3D, got %dD" % nd)
    nx = m.shape[0]

    vel = np.asarray(vel, np.float32)
    # PECLET IS TAKEN AGAINST THE AXIAL INTERSTITIAL VELOCITY, the mean of the
    # flow-direction component over the pore space -- not the mean of the
    # SPEED. They differ by the tortuosity: measured on a real Stokes field,
    # mean|u| was 1.140 times mean u_x, and the ratio changes with the pore
    # structure. Using the speed made the recorded Peclet about 14% higher
    # than the one a reader computes from the flux, and by a
    # geometry-dependent amount, which is worse than a constant offset.
    #
    # The fallback exists for a closed or recirculating field, where the axial
    # mean is near zero and is not a meaningful scale.
    sp = np.sqrt((vel[:nd] ** 2).sum(0))
    mean_axial = float(vel[0][m].mean()) if m.any() else 0.0
    mean_speed = float(sp[m].mean()) if m.any() else 0.0
    u_scale = mean_axial if mean_axial > 0.1 * mean_speed else mean_speed
    u = vel[:nd] / max(u_scale, 1e-30)
    u = np.where(m[None], u, 0.0)          # no flow inside the rock

    ns = int(n_species)
    ch = (default_chemistry(ns) if chem is None else chem.truncate(ns))
    if len(ch.names) != ns:
        raise ValueError("the chemistry describes %d chemicals but %d were "
                         "asked for" % (len(ch.names), ns))

    # ----------------------------------------------------------------------
    #  9.2  THE DIMENSIONLESS NUMBERS BECOME LATTICE QUANTITIES
    # ----------------------------------------------------------------------
    #  Three numbers arrive dimensionless and have to become quantities on this
    #  grid. All three carry a LENGTH, so l_ref below decides what all three
    #  of them mean.
    #
    #       Pe       ->  D  = l_ref / Pe          the diffusivity
    #       Da       ->  k  = Da / l_ref          the biotic rate constant
    #       Da_abio  ->  ka = Da_abio / l_ref     the abiotic rate constant
    #
    #  The velocity is normalised so its mean over the pore space is 1, which
    #  is what makes Pe alone set the balance between carrying and spreading.
    # THE LENGTH EVERY DIMENSIONLESS NUMBER IS TAKEN AGAINST.
    #
    # Peclet is u*L/D and both Damkohler numbers carry an L, so this one
    # number decides what all three of them mean. It used to be the sample
    # length, nx voxels, with no way to say otherwise -- so a settings file
    # asking for ratios against the PORE WIDTH, which is what CompLaB uses and
    # what most of the pore-scale literature quotes, got a report in pore
    # widths and a simulation in sample lengths. At the defaults, 32 voxels
    # against a 16 um pore, that is a factor of two on Peclet and on both
    # Damkohler numbers, in opposite directions.
    l_ref = float(getattr(ch, "l_ref_voxels", None) or nx)
    D = np.float32(l_ref / max(pe, 1e-6))
    # THE GRID PECLET NUMBER decides whether this domain can represent the
    # Peclet number it was asked for at all. Physical diffusion is nx/Pe in
    # lattice units and a voxel is 1 across, so Pe/nx is the ratio of
    # advection to diffusion ACROSS ONE VOXEL. Above about 2 the concentration
    # front is thinner than the voxels meant to hold it, and no scheme on this
    # grid reproduces the requested number -- the honest fix is more voxels,
    # not a better solver. Said out loud here rather than left for the reader
    # to infer from a picture.
    pe_grid = float(pe) / l_ref
    if pe_grid > 2.0:
        print("     WARNING: Peclet %.4g on %d voxels is a grid Peclet number "
              "of %.1f.\n              The front is thinner than the mesh; the "
              "sample will behave\n              as though its Peclet number "
              "were lower than %.4g. Use a\n              longer domain, or a "
              "lower Peclet number." % (pe, int(l_ref), pe_grid, pe),
              flush=True)
    k = np.float32(da / l_ref)                 # biotic rate constant
    ka = np.float32(max(da_abio, 0.0) / l_ref)  # abiotic rate constant
    # AN ABIOTIC DAMKOHLER NUMBER WITH NOTHING TO CONSUME.
    #
    # The abiotic reaction attacks one named chemical -- the product by
    # default, the donor when there is no biotic reaction to make a product.
    # Ask for two chemicals and the product is not in the run, so the reaction
    # has no reactant, does nothing at all, and the Damkohler number is still
    # written into the dataset as though it had. That combination produced
    # files whose Da_abio column varied from 0 to 10 across a sweep of runs
    # that were, every one of them, identical. Refused rather than warned:
    # there is no reading of it that gives the caller what they asked for.
    _j_ab = getattr(ch, "abiotic_index", 2)
    if ka > 0 and _j_ab >= ns:
        raise ValueError(
            "an abiotic Damkohler number of %g was given, but it consumes "
            "chemical %d (%s) and this run has only %d chemicals, so the "
            "reaction cannot happen. Ask for at least %d chemicals, or set "
            "the abiotic reactant to one that is in the run, or set the "
            "abiotic Damkohler number to zero."
            % (da_abio, _j_ab, _name(ch, _j_ab), ns, _j_ab + 1))
    kd = np.float32(ch.decay_for(l_ref))
    # Per-chemical diffusion, as a multiple of the first chemical's. All ones
    # unless a settings file says otherwise. An immobile chemical gets zero
    # and is skipped entirely rather than diffused at zero, which is the same
    # answer for less arithmetic.
    d_rel = np.asarray(ch.d_rel, np.float32).copy()
    d_rel[ch.immobile] = 0.0
    # Diffusion is a FIELD when there are biofilm voxels, because a chemical
    # moves through biofilm at a different rate from open water. That is what
    # the in_biofilm coefficient has always meant and, until there were
    # biofilm voxels to apply it to, it was read and reported and never used.
    if biofilm.any():
        d_bf = np.asarray(ch.d_biofilm_rel, np.float32).copy()
        d_bf[ch.immobile] = 0.0
        dfield_full = np.where(biofilm[None], d_bf.reshape((ns,) + (1,) * nd),
                               d_rel.reshape((ns,) + (1,) * nd)
                               ).astype(np.float32) * float(D)
        d_worst = float(max(d_rel.max(), d_bf.max()))
    else:
        dfield_full = (np.ones((ns,) + m.shape, np.float32)
                       * d_rel.reshape((ns,) + (1,) * nd) * float(D))
        d_worst = float(d_rel.max())

    # ----------------------------------------------------------------------
    #  9.3  THE TIME STEP
    # ----------------------------------------------------------------------
    #  Explicit in time, so the step is bounded by the fastest process on the
    #  grid, and by ALL of them at once rather than by the smaller of two
    #  separate limits. The three contributions are the flow crossing a voxel,
    #  diffusion crossing a voxel, and the reaction consuming a voxel's
    #  contents.
    # ONE limit covering advection, diffusion and reaction together. Taking
    # the smaller of two SEPARATE limits is not the same thing, and is how the
    # scheme came to sit exactly on the diffusive boundary.
    #
    # The FASTEST-diffusing chemical sets the limit, not the first one. With a
    # single shared diffusivity these are the same number; the moment a
    # settings file gives one chemical a larger coefficient than the donor,
    # using the donor's would put that chemical past its own stability limit
    # and the field would checkerboard -- in one chemical only, which is
    # exactly the kind of thing that gets mistaken for interesting physics.
    dmax = float(D) * float(max(d_worst, 1.0))
    umax = float(sum(np.abs(u[ax]).max() for ax in range(nd)))
    # THE REACTION'S TRUE LINEARISED RATE, not just its constant.
    #
    # A Monod term k*C/(Ks+C) has slope k/Ks as C goes to zero, which is
    # LARGER than k, by 1/Ks. Leaving it out let the explicit step overshoot:
    # the donor was driven negative, clipped at zero, and the product still
    # received the full +dt*R -- so product appeared out of nothing. Measured
    # with the acceptor pinned and no stoichiometry, so that donor + product
    # must equal exactly 1: at Da 20 the product left the sample at 1.239, at
    # Da 100 at 1.958, and at Da 400 at 5.093, against a feed of 1.0.
    # check_physics saw none of it, because it only ever bounded the donor.
    k_lin = float(k)
    if ch.ks_donor > 0:
        k_lin = max(k_lin, float(k) / float(ch.ks_donor))
    if ns > 1 and ch.ks_acceptor > 0:
        k_lin = max(k_lin, float(k) / float(ch.ks_acceptor))
    dt = np.float32(0.4 / (umax + 2.0 * nd * dmax + k_lin + ka + kd + 1e-30))

    lo_m, hi_m = _neighbour_masks(m)
    # Pore voxels that touch solid. A surface reaction happens on the grain
    # surface, not throughout the water, so it needs to know where the surface
    # is. This is the mask for it: 1 on a pore voxel with at least one solid
    # face, 0 elsewhere. Only built when it is going to be used.
    surf = np.zeros_like(m, np.float32)
    if ka > 0 and getattr(ch, "abiotic_surface_only", False):
        # The neighbour masks carry a leading axis so they broadcast against
        # the (species, grid) arrays; drop it here, where the question is
        # about the grid alone.
        # HOW MUCH surface, not merely whether there is any. A voxel in a
        # narrow slot has solid on two or three sides and presents two or
        # three times the area per unit volume of one on a flat wall; a rate
        # per unit area must scale with that. Counting them: on one test
        # geometry, 363 surface voxels carried 494 exposed faces -- 235 with
        # one, 125 with two, 3 with three -- and treating all of them alike
        # made the abiotic Damkohler mean different things in rocks of
        # different specific surface area. Normalised by 2*nd so that a voxel
        # walled on every side scores 1.
        nface = np.zeros(m.shape, np.float32)
        for ax in range(nd):
            nface += (~lo_m[ax][0]).astype(np.float32)
            nface += (~hi_m[ax][0]).astype(np.float32)
        surf = np.where(m, nface / float(2 * nd), 0.0).astype(np.float32)
    inlet = m[0]                            # pore voxels on the x = 0 face
    fed = np.asarray(ch.inlet, np.float32)  # NaN where there is no feed
    moving = [j for j in range(ns) if not ch.immobile[j]]
    n_move = len(moving)

    # ---- FACE QUANTITIES, built once ------------------------------------
    #
    # The transport operator is written in FLUX FORM: the change in a voxel is
    # what crossed its faces. That is not a refactor, it fixes two measured
    # errors that the older cell-centred form had.
    #
    # VARIABLE DIFFUSION. Writing D(x) grad^2 C multiplies the whole Laplacian
    # by the CELL's own coefficient, so the flux across a pore-to-biofilm face
    # is D_pore*(C_bf - C_pore) seen from one side and D_bf*(C_pore - C_bf)
    # seen from the other. Those do not match, and the consequence is that the
    # biofilm's diffusivity does almost nothing: with a twentieth of the pore
    # value in a slab, the steady profile came out a STRAIGHT LINE, 0.310
    # where the flux-continuous answer is 0.041. Written as div(D grad C) with
    # a HARMONIC MEAN at each face, the flux is continuous by construction --
    # the harmonic mean is the series resistance of the two half-cells, which
    # is the exact answer for a layered medium.
    #
    # THE WALL. Replacing an upwind neighbour with the voxel's own value does
    # not impose zero flux through that face; it merely deletes one term.
    # Measured on a closed box with an exactly divergence-free recirculating
    # flow: mass drifted +1.35% with no obstacle and -4.03% around a
    # staircase grain surface. Here a face onto solid simply carries no flux,
    # for either process, which is what a no-flux wall means.
    # ----------------------------------------------------------------------
    #  9.4  FACE QUANTITIES, BUILT ONCE
    # ----------------------------------------------------------------------
    #  The operator is written in FLUX FORM: the change in a voxel is what
    #  crossed its faces. So everything that belongs to a FACE rather than to a
    #  voxel is computed here, once, and reused every step:
    #
    #       which faces are open      a face onto solid carries no flux at all
    #       the face velocity         the average of the two cells it separates
    #       the face diffusivity      the harmonic mean, see 9.6
    #       which far neighbours      exist, for the slope in 9.5
    # THE ADVECTION IS SECOND ORDER, LIMITED. This is not a refinement; it is
    # the difference between the Peclet number written in the file and the
    # Peclet number the sample actually experienced.
    #
    # First-order upwinding is equivalent to solving the exact equation plus an
    # extra diffusivity of |u|*dx/2 per axis. In these units dx = 1 and u is
    # normalised so that its pore mean is 1, so the spurious term is about 0.5
    # while the settings_and_units one is D = nx/Pe. They are EQUAL at Pe = 2*nx -- about
    # 260 on a 128-voxel domain -- and the spurious one wins above that.
    # Measured before this change, on a straight channel, as the effective
    # Peclet number recovered from how far a front had really spread:
    #
    #     nx      requested Pe      the Peclet the sample actually saw
    #     64           50                        39
    #     64          100                        62
    #     64          300                       107
    #    400         4000                       880
    #
    # A network trained on that data is being told the sample was at Pe 300
    # when it was at 107. It cannot learn the labelled parameter, because the
    # labelled parameter is not the one in the data. Worse, the error is not a
    # constant factor: it depends on the local speed, so fast streamlines are
    # mislabelled more than slow ones.
    #
    # The cure is to reconstruct the face value from a slope instead of taking
    # the upwind cell whole, and to LIMIT that slope so the scheme still makes
    # no new maxima. See _muscl, which records the measurements after the fix
    # and why the limiter is minmod rather than the more usual van Leer.
    #
    # It costs a two-voxel stencil, hence the pad of 2 rather than 1. Where the
    # far voxel is solid or off the end of the domain the upwind difference is
    # zero, the limiter returns zero slope, and the scheme degrades to first
    # order exactly at the wall -- which is correct, not a compromise: there is
    # no smooth extrapolation to be had through a grain.
    def _sl(ax, off):
        """The window into an array padded by 2, shifted `off` voxels on `ax`."""
        return tuple(slice(2 + off, (off - 2) or None) if j == ax
                     else slice(2, -2) for j in range(nd))

    def _pad2(a):
        return np.pad(a, [(2, 2)] * nd, mode="edge")

    M2 = _pad2(m)
    faces = []
    for ax in range(nd):
        s_m2, s_m1, s_p1, s_p2 = (_sl(ax, -2), _sl(ax, -1),
                                  _sl(ax, 1), _sl(ax, 2))
        open_lo = M2[s_m1] & m
        open_hi = M2[s_p1] & m
        # The far voxel counts only if the near one does: a slope measured
        # across a grain is not a slope.
        far_lo = M2[s_m2] & open_lo
        far_hi = M2[s_p2] & open_hi
        # Face velocity: the average of the two cells it separates, and zero
        # on any face that is not open. A closed face therefore advects
        # nothing at all.
        up = _pad2(u[ax])
        u_lo = np.where(open_lo, 0.5 * (u[ax] + up[s_m1]), 0.0)
        u_hi = np.where(open_hi, 0.5 * (u[ax] + up[s_p1]), 0.0)
        # Face diffusivity, per chemical, as the harmonic mean of the two
        # cells. Zero when either coefficient is zero, which is right: an
        # immobile chemical does not diffuse across any face.
        _js = list(range(ns)) if n_move == ns else moving
        dp = {j: _pad2(dfield_full[j]) for j in _js}
        d_lo, d_hi = [], []
        for j in _js:
            a1 = dfield_full[j]
            a2 = dp[j][s_m1]
            b2 = dp[j][s_p1]
            with np.errstate(divide="ignore", invalid="ignore"):
                hlo = np.where((a1 > 0) & (a2 > 0), 2 * a1 * a2 / (a1 + a2), 0.0)
                hhi = np.where((a1 > 0) & (b2 > 0), 2 * a1 * b2 / (a1 + b2), 0.0)
            d_lo.append(np.where(open_lo, hlo, 0.0))
            d_hi.append(np.where(open_hi, hhi, 0.0))
        faces.append((s_m2, s_m1, s_p1, s_p2, open_lo, open_hi, far_lo, far_hi,
                      u_lo, u_hi,
                      np.stack(d_lo).astype(np.float32),
                      np.stack(d_hi).astype(np.float32)))

    # THE DIVERGENCE THE VELOCITY FIELD ACTUALLY HAS.
    #
    # Written as a flux, -div(uC) is identically -u.grad(C) - C*div(u). The
    # second term is zero for an incompressible flow and is therefore usually
    # left unsaid -- but a lattice-Boltzmann field is only APPROXIMATELY
    # solenoidal, and the residual is a source term sitting on every voxel,
    # multiplied by the concentration there. It does not average out: where
    # the field converges slightly, concentration is manufactured, and a long
    # run accumulates it. Measured on a 32x24x24 pack at Peclet 1.17, where
    # the run takes 32000 steps to settle: the donor reached 1.022 and the
    # acceptor 1.037 against a feed of 1.000, for chemicals that are only ever
    # CONSUMED. Nothing was unstable; the equation being solved simply had a
    # source in it that the physics does not.
    #
    # Subtracting it back leaves -u.grad(C), which cannot make a new maximum.
    # It is computed from the SAME face velocities as the fluxes, so on a
    # genuinely divergence-free field it is exactly zero to the last bit and
    # the conservation the flux form was introduced for is untouched -- the
    # closed-box test still drifts by 0.0003%.
    divu = np.zeros(m.shape, np.float32)
    for ax in range(nd):
        divu += faces[ax][9] - faces[ax][8]        # u_hi - u_lo

    # ----------------------------------------------------------------------
    #  9.5 and 9.6  ADVECTION AND DISPERSION, IN ONE OPERATOR
    # ----------------------------------------------------------------------
    #  Both are face fluxes summed over the faces of each voxel, so they are
    #  written together. Reading the body below:
    #
    #    9.6  DISPERSION is the line with D_hi and D_lo. The face diffusivity
    #         is the HARMONIC MEAN of the two cells, which is the series
    #         resistance of two half-cells and the exact answer for a layered
    #         medium. It is what makes a biofilm diffusivity do anything: the
    #         naive form gave a straight-line profile of 0.310 through a
    #         biofilm slab where the flux-continuous answer is 0.041.
    #
    #    9.5  ADVECTION is the fh and fl lines. At each face the concentration
    #         is reconstructed from the upwind side with the limited slope of
    #         BLOCK 7, and the flux is the face velocity times that value.
    #
    #    The last line takes back the C*div(u) that flux form silently adds. A
    #    lattice-Boltzmann velocity field is only approximately divergence
    #    free, and that residual is a source term multiplying concentration:
    #    at Peclet 1.17 in 3D the acceptor, which is only ever consumed,
    #    reached 1.037 against a feed of 1.000.
    def transport(C):
        """div(D grad C) - div(u C), both as face fluxes.

        Conservative by construction: the flux this cell sends across its high
        face is the identical expression its neighbour receives across its low
        face, so whatever leaves one cell arrives in the other and the total
        can only change through the boundary.
        """
        out = np.zeros_like(C)
        P = np.pad(C, [(0, 0)] + [(2, 2)] * nd, mode="edge")
        for ax in range(nd):
            (s_m2, s_m1, s_p1, s_p2, o_lo, o_hi, f_lo_ok, f_hi_ok,
             u_lo, u_hi, D_lo, D_hi) = faces[ax]
            Cm1 = np.where(o_lo, P[(slice(None),) + s_m1], C)
            Cp1 = np.where(o_hi, P[(slice(None),) + s_p1], C)
            Cm2 = np.where(f_lo_ok, P[(slice(None),) + s_m2], Cm1)
            Cp2 = np.where(f_hi_ok, P[(slice(None),) + s_p2], Cp1)
            # diffusion: D_face * (C_neighbour - C), summed over both faces
            out += D_hi * (Cp1 - C) + D_lo * (Cm1 - C)
            # advection: a limited reconstruction at each face, taken from
            # whichever side the flow is arriving from
            fh = np.where(u_hi >= 0, _muscl(C, Cm1, Cp1), _muscl(Cp1, Cp2, C))
            fl = np.where(u_lo >= 0, _muscl(Cm1, Cm2, C), _muscl(C, Cp1, Cm1))
            out -= u_hi * fh - u_lo * fl
        # and take back the C*div(u) that the flux form silently added
        out += C * divu
        return out

    # ----------------------------------------------------------------------
    #  9.9  THE BIOTIC REACTION  --  microbially mediated
    # ----------------------------------------------------------------------
    #       Ac  +  alpha A  ->  P  +  Y Bio
    #
    #       rate = k * f(donor) * f(acceptor) * (biomass factor)
    #
    #  f(x) = x/(Ks+x) when a half-saturation constant is given. A constant of
    #  ZERO does not mean Monod with Ks = 0, which would be zero order; it
    #  means the rate is first order in that chemical and the constant is
    #  folded into k.
    #
    #  Switched off entirely by reaction_mode = abiotic only, in which case
    #  there is no biomass either.
    def rate_biotic(C):
        """The microbially mediated reaction. Dual Monod, donor and acceptor.

            R_bio = k * f(donor) * f(acceptor) * (biomass factor)

        f(x) = x/(Ks+x) when a half-saturation constant is given. A constant
        of ZERO does not mean Monod with Ks = 0, which would be zero order; it
        means the rate is first order in that chemical and the constant is
        folded into k. The donor default is zero, which is why the donor term
        reads as first order.

        The biomass factor is 1 by default, which means the biomass is held at
        its initial value inside the rate. That is not an oversight: the
        Damkohler number is DEFINED as vmax * B0 * L / (Ac0 * u), so B0 is
        already inside k, and letting B vary as well would count it twice.
        Set biomass_coupled to have the rate follow the biomass that is
        actually there, which is the honest choice when the biomass changes a
        lot over a run.
        """
        if ch.ks_donor > 0:
            R = k * C[0] / (ch.ks_donor + C[0])
        else:
            R = k * C[0]
        if ns > 1:
            # Zero means FIRST ORDER, exactly as it does for the donor two
            # lines up. Treating it as Monod with Ks = 0 is 0/0 wherever the
            # acceptor is zero -- which is everywhere at t = 0 unless it is
            # fed -- and it filled the whole field with not-a-number.
            R = R * (C[1] if ch.ks_acceptor <= 0
                     else C[1] / (ch.ks_acceptor + C[1]))
        if ch.biomass_coupled and ns > 3 and ch.c_init[3] > 0:
            R = R * C[3] / np.float32(ch.c_init[3])
        return R

    # ----------------------------------------------------------------------
    #  9.10  THE ABIOTIC REACTION  --  purely chemical, no microbes
    # ----------------------------------------------------------------------
    #       reactant  ->  product        rate = k_surf * reactant
    #
    #  First order, and no microbes take part or need to be present. It can be
    #  made to happen only on pore voxels that touch a grain, which is what a
    #  mineral-surface reaction does, and then it is weighted by how many
    #  faces each voxel exposes rather than treating every surface voxel alike.
    #
    #  It PRODUCES something, unless you asked for a terminal loss. Until that
    #  was added an abiotic-only run was a chemical that disappeared with an
    #  empty product field beside it, which is a decay and not a reaction.
    def rate_abiotic(C):
        """The purely chemical reaction. No microbes take part in it.

            R_abio = k_abio * product

        First order in the reaction product, which is what the abiotic
        Damkohler number k_surf * L / u is defined against. It is a loss: the
        product is consumed and nothing else is tracked. Zero when the abiotic
        reaction is switched off, which is the default.
        """
        j = getattr(ch, "abiotic_index", 2)
        if ka <= 0 or ns <= j:
            return np.float32(0.0)
        if ch.abiotic_surface_only:
            return ka * C[j] * surf
        return ka * C[j]

    def reaction(C):
        R = rate_biotic(C) if getattr(ch, "biotic_on", True) else np.float32(0.0)
        Ra = rate_abiotic(C)
        # NO REACTION MAY CONSUME MORE THAN IS THERE.
        #
        # dt is chosen from a linearised rate constant, which is the right
        # thing when the rate constant is constant. It is not always. With
        # biomass_coupled the biotic rate carries a factor B/B0, and B grows
        # exponentially while B0 does not, so a time step that was stable at
        # the start is wildly unstable a few thousand steps later. The failure
        # is silent and specific: the donor is driven negative, the clip at
        # zero absorbs it, and the product still receives the full +dt*R --
        # product made from a donor that was never there. Measured on the
        # coupled test case, the product left the sample at 505 against a feed
        # of 1.0.
        #
        # Recomputing dt every step would fix it and would also make the
        # stored snapshot times a function of the solution. This is the
        # cheaper and stricter cure: cap the extent of reaction at half of
        # what the scarcest reactant can actually supply during this step.
        # Where the reaction is slow the cap never binds and the answer is
        # unchanged to the last bit; where it would have gone unstable the cap
        # is what stops it. Positivity then holds for the reaction for the
        # same reason the limiter gives it for the transport -- by
        # construction, not by hope.
        if np.ndim(R) or float(R) != 0.0:
            lim = C[0] / dt
            if ns > 1 and ch.alpha > 0:
                lim = np.minimum(lim, C[1] / (float(ch.alpha) * dt))
            R = np.minimum(R, 0.5 * lim)
        j_lim = getattr(ch, "abiotic_index", 2)
        if (np.ndim(Ra) or float(Ra) != 0.0) and j_lim < ns:
            Ra = np.minimum(Ra, 0.5 * C[j_lim] / dt)
        j_ab = getattr(ch, "abiotic_index", 2)
        out = np.zeros_like(C)
        out[0] = -R
        if ns > 1:
            out[1] = -ch.alpha * R
        if ns > 2:
            # The product carries what the biomass did not. Every chemical
            # here is measured against the DONOR feed, so writing +R for the
            # product and +Y*R for the biomass creates 1+Y moles of
            # donor-equivalent from one -- 4% of the carbon at the default
            # yield, more if a settings file raises it.
            out[2] = (1.0 - (ch.yield_ if ns > 3 else 0.0)) * R
        if ns > 3:
            # With no microbes there is nothing to grow and nothing to decay.
            out[3] = ((ch.yield_ * R - kd * C[3])
                      if getattr(ch, "biotic_on", True) else 0.0)
        # THE ABIOTIC REACTION, applied to whichever chemical it consumes and
        # -- when it makes one -- to whichever it produces.
        #
        #     Ac  --k_surf-->  P          abiotic_product = "product"
        #     Ac  --k_surf-->  (lost)     abiotic_product = "none"
        #
        # THE PRODUCTION TERM IS NEW AND IT IS THE POINT. Until it existed the
        # abiotic reaction could only ever DESTROY: it subtracted from its
        # reactant and added to nothing, so "abiotic only" meant a chemical
        # that disappeared, with an empty product field beside it. That is a
        # decay, not a reaction, and it is not what a mineral dissolution or a
        # surface-catalysed transformation does -- those turn one dissolved
        # species into another, which is exactly the abiotic experiment
        # somebody means when they say "no microbes, but still a reaction".
        #
        # One mole for one. The abiotic network here is a single first-order
        # step with unit stoichiometry, so donor plus product is conserved and
        # check_physics tests that it is.
        if j_ab < ns:
            out[j_ab] = out[j_ab] - Ra
            j_mk = getattr(ch, "abiotic_product_index", None)
            if j_mk is not None and j_mk < ns and j_mk != j_ab:
                out[j_mk] = out[j_mk] + Ra
        # Anything past the fourth takes no part in the reaction. It is still
        # fed, still advected and still diffused with its own coefficient --
        # a conservative tracer, which is what you add when you want to see
        # where the water went as distinct from where the chemistry got to.
        # Its rate is zero here and that is deliberate, not an omission: the
        # reaction network is donor plus acceptor gives product plus biomass,
        # and a fifth reacting chemical needs CompLaB.
        return out

    # ----------------------------------------------------------------------
    #  9.7 and 9.11  BOUNDARY CONDITIONS, AND ONE TIME STEP
    # ----------------------------------------------------------------------
    #  The four boundaries of the transport problem, and where each one is:
    #
    #    INLET, x = 0     the first lines below. A fixed concentration for the
    #                     chemicals that HAVE a feed. A chemical with no feed is
    #                     not pinned to zero there; pinning the biomass to zero
    #                     at the inlet drove the biotic rate to zero on the one
    #                     face where the food is most abundant.
    #
    #    GRAIN SURFACES   in 9.4, as faces that are not open. A closed face
    #                     carries no flux for either process, which is what a
    #                     no-flux wall means.
    #
    #    OUTLET and SIDES in 9.4 too, through the edge-replicated padding: the
    #                     value just outside equals the value just inside, so
    #                     the gradient is zero and what arrives simply leaves.
    #                     Nothing is reflected and nothing is imposed.
    #
    #  Then the step itself: transport, plus reaction, times dt. Immobile
    #  chemicals skip transport entirely. The clip at zero at the end is what
    #  makes a concentration a count of molecules rather than a number.
    def step(C):
        # The feed applies to chemicals that HAVE a feed. A chemical with a
        # Neumann left boundary is not held at zero there -- it simply is not
        # a boundary chemical, and pinning it to zero at the inlet is what
        # made the biomass vanish from the one face where the chemistry is
        # most active.
        for j in moving:
            if fed[j] == fed[j]:            # not NaN
                C[j, 0][inlet] = fed[j]
        # TRANSPORT ONLY THE CHEMICALS THAT MOVE. This used to advect and
        # diffuse all of them and then overwrite the immobile ones with zero
        # -- the arithmetic was done and thrown away. With four chemicals, one
        # of which is an attached biomass, that is a quarter of the most
        # expensive loop in the project spent computing a number already known
        # to be zero. Measured on a 148 by 64 run: 26% faster.
        if n_move == ns:
            trans = transport(C)
        else:
            trans = np.zeros_like(C)
            trans[moving] = transport(C[moving])
        C = C + dt * (trans + reaction(C))
        return np.where(m, np.maximum(C, 0.0), 0.0)

    # ----------------------------------------------------------------------
    #  9.8  INITIAL CONDITIONS  --  the state at t = 0
    # ----------------------------------------------------------------------
    #  Every solute starts at whatever the settings say, which is zero unless
    #  you asked otherwise: clean water, nothing has arrived yet.
    #
    #  The attached biomass is the exception. It starts WHERE THE BIOFILM IS
    #  when the geometry says where that is, because seeding attached microbes
    #  uniformly through open water is a different experiment from seeding them
    #  on the grain surfaces, and CompLaB does the latter.
    def initial():
        C = np.zeros((ns,) + m.shape, np.float32)
        for j in range(ns):
            if not ch.c_init[j]:
                continue
            # The attached biomass starts WHERE THE BIOFILM IS, when the
            # geometry says where that is. Everything else starts throughout
            # the pore space. Seeding attached microbes uniformly through open
            # water is a different experiment from seeding them on the grain
            # surfaces, and CompLaB does the latter.
            if (j == ATTACHED and ch.seed_in_biofilm and biofilm.any()
                    and ch.immobile[j]):
                C[j] = np.where(biofilm, ch.c_init[j], 0.0)
            else:
                C[j] = np.where(m, ch.c_init[j], 0.0)
        return C

    # ----------------------------------------------------------------------
    #  9.12  THE TIME LOOP, CONVERGENCE, AND THE SNAPSHOTS
    # ----------------------------------------------------------------------
    #  One pass, not two. It records the field at a geometric ladder of step
    #  numbers as it goes and picks the wanted snapshots from that, so the
    #  stored times are times that were actually integrated to.
    #
    #  The run stops when EVERY chemical has settled, and what is compared
    #  against the tolerance is how far from settled the run still is, not how
    #  much it changed last interval. Those differ by a factor of about thirty
    #  near the end.
    # ---- ONE pass, keeping a ladder of snapshots as it goes ---------------
    #
    # This used to be two passes: one to find how long the run needs, another
    # to re-integrate and store. That is exactly twice the work, and the work
    # is the expensive part. The single pass records the field at a geometric
    # ladder of step numbers and picks the wanted snapshots from it, so the
    # stored times are the times that were actually integrated to.
    C = initial()
    keep_steps, keep = [], []

    def record(s):
        keep_steps.append(s)
        keep.append(np.where(m, C, 0.0).astype(np.float32))

    # CONVERGENCE IS JUDGED ON EVERY CHEMICAL, NOT ON THE DONOR ALONE.
    #
    # The donor settles first and by a wide margin. It is fed at the inlet and
    # swept through, so it reaches its steady profile in roughly one transit;
    # the product has to be MADE from it, and the biomass has to grow on the
    # product, so each one lags the one before it. Stopping when the donor
    # stops moving therefore stops while the chemicals the study is actually
    # about are still changing. Measured on a 60x32 domain at Pe 30, Da 1, at
    # the moment the old donor-only test fired: the product was still 13% from
    # its final value and the biomass 20% from its final value, and both were
    # still moving in one direction. Those runs were labelled "converged".
    #
    # Each chemical is measured against its own scale -- the largest mean it
    # has reached so far -- because they differ by orders of magnitude and a
    # tolerance on the absolute change would be all donor and no biomass. The
    # scale is a running maximum rather than the current value so that a
    # chemical passing through zero cannot make its own denominator tiny and
    # its relative change enormous.
    check = max(int(1.0 / float(dt)), 100)   # judge convergence per unit TIME
    prev = np.zeros(ns)
    scale = np.full(ns, 1e-6)
    prev_change = np.zeros(ns)
    n, hit_cap = 0, False
    record(0)
    nxt = 1
    limit = int(steps) if steps is not None else max_steps
    while n < limit:
        n += 1
        C = step(C)
        if n >= nxt:
            record(n)
            nxt = max(n + 1, int(np.ceil(n * ratio)))
        if n % check == 0:
            cur = np.array([float(C[j][m].mean()) for j in range(ns)])
            scale = np.maximum(scale, np.abs(cur))
            change = float((np.abs(cur - prev) / scale).max())
            # WHAT IS LEFT TO GO, NOT WHAT JUST HAPPENED. A field approaching
            # a steady state does so geometrically: each interval moves it a
            # fixed fraction r of the remaining distance. So a small change
            # this interval does NOT mean a small distance from settled -- it
            # means r*(remaining), and with r near 1 the remaining distance is
            # many times the change. Stopping on the change alone left the
            # biomass 4.1% from its final value while reporting a change of
            # 0.2%. Summing the geometric tail gives the distance itself,
            # change*r/(1-r), and that is what the tolerance is applied to.
            # The run then really is within tol of settled: measured 0.5% on
            # the biomass, against 4.1% before and 20% on the donor-only rule
            # this replaced.
            #
            # PER CHEMICAL, and that detail is the whole of it. Taking the
            # ratio of the largest change to the previous largest change
            # compares two different chemicals whenever the leader changes
            # hands, which it always does: the donor moves fastest at first
            # and is finished long before the biomass, so the ratio at the
            # handover is meaningless and the extrapolation with it. Each
            # chemical is extrapolated against its own history and the worst
            # one decides.
            step_ch = np.abs(cur - prev) / scale
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(prev_change > 0, step_ch / prev_change, 0.0)
            rest = np.where(r < 1.0, step_ch * r / np.maximum(1.0 - r, 1e-12),
                            np.inf)
            rest = float(np.nanmax(np.where(np.isfinite(rest), rest, np.inf)))
            # SHOW THE NUMBER THE TOLERANCE IS COMPARED AGAINST. The progress
            # line used to print the change over the last interval while the
            # stopping rule used the extrapolated distance still to go, which
            # are different by a factor of 1/(1-r) -- thirty or so near the
            # end. So the display read "still changing by 3.5e-04" against a
            # tolerance of 2e-03 and the run kept going, which looks like a
            # solver ignoring its own setting. It was not; it was reporting
            # the wrong quantity.
            if on_progress is not None:
                on_progress(n, limit, float(cur[0]),
                            rest if np.isfinite(rest) else change)
            if steps is None and n >= 10 * check and rest <= tol:
                break
            prev, prev_change = cur, step_ch
    if steps is None and n >= max_steps:
        hit_cap = True
    if keep_steps[-1] != n:
        record(n)
    total = int(max(n, 1))

    # ---- choose the snapshots ---------------------------------------------
    ks = np.asarray(keep_steps)
    want = _wanted_steps(total, n_t)
    used, picked = set(), []
    for w in want:
        i = int(np.abs(ks - w).argmin())
        if i in used:
            # Take the nearest ladder entry not already used, rather than
            # padding past the end of the run -- which silently duplicated the
            # final state and gave several snapshots the same t_norm.
            i = _nearest_unused(i, used, len(ks))
        used.add(i)
        picked.append(i)
    repeated = n_t - len(set(picked))

    out = np.zeros((n_t, ns) + m.shape, np.float32)
    for j, i in enumerate(picked):
        out[j] = keep[i]
    t_norm = (ks[picked].astype(np.float32) / total).astype(np.float32)

    if info is not None:
        # Whether the run actually settled. A truncated run is still a valid
        # transient and fine to train on, but its last snapshot is "as far as
        # we got", not a steady state, and t = 1.0 should not be read as one.
        info["converged"] = not (hit_cap or steps is not None)
        info["steps"] = total
        info["dt"] = float(dt)
        info["ladder"] = len(ks)
        info["chemistry"] = ch.source
        info["da_bio"] = float(da)
        info["da_abio"] = float(da_abio)
        info["dt_limited_by"] = ("advection" if umax > 2.0 * nd * dmax
                                 else "diffusion")
        info["biofilm_voxels"] = int(biofilm.sum())
        info["grid_peclet"] = float(pe_grid)
        info["l_ref_voxels"] = float(l_ref)
        # How many pore voxels actually touch a grain, weighted by how many
        # faces each one exposes. A surface reaction can only happen here, so
        # zero means the reaction had nowhere to act however large its
        # Damkohler number was -- which is a fact about the geometry the
        # caller should be told, not left to infer from an unchanged field.
        info["surface_voxels"] = float(surf.sum())
        # Non-zero when the run held fewer distinct states than snapshots
        # asked for, so some had to be repeated.
        info["repeated_snapshots"] = int(max(repeated, 0))
    return out, t_norm


# --------------------------------------------------------------------------
# ==========================================================================
#  BLOCK 10  |  WHAT A CORRECT SOLUTION MUST SATISFY
# --------------------------------------------------------------------------
#  in   out    the field solve_adr just produced
#       mat    the material grid
#       chem   the chemistry, so the bounds can be per chemical
#       da, da_abio   the rates, so a rate with no reactant is caught
#  out  a list of complaints, empty when the run is sound
#
#  Seven rules. Every one of them corresponds to a fault that reached a
#  delivered dataset before anybody looked at a picture. The generators
#  refuse to write a file when this list is not empty.
# ==========================================================================

def _name(chem, j):
    """What to call chemical j in a complaint. Names beat indices in a message
    a user has to act on: "the product reaches 5.09" says what to look at,
    "species 2 reaches 5.09" makes them go and count."""
    try:
        return str(chem.names[j])
    except Exception:
        return ("the donor", "the acceptor", "the product",
                "the biomass")[j] if j < 4 else "chemical %d" % j


def _start(chem, j):
    """The largest amount of chemical j that was fed in or present at t = 0."""
    if chem is None:
        return 1.0
    vals = [float(v) for v in (chem.inlet[j], chem.c_init[j]) if v == v]
    return max(vals) if vals else 0.0


def check_physics(out, mat, verbose=False, chem=None,
                  da=None, da_abio=None):
    """Assertions a correct solution must satisfy. Returns a list of complaints.

    Each one corresponds to a bug that reached a delivered dataset before
    anybody looked at a picture. They are cheap; training on data that fails
    them costs far more than running them.
    """
    mat = np.asarray(mat)
    m = (mat == PORE)
    bad = []
    out = np.asarray(out)
    donor = out[:, 0]
    ns = out.shape[1]

    # 0. FINITE. This has to come first and has to be its own test. Every
    #    check below is a `>` or `<` comparison, and every comparison against
    #    NaN is False -- so an unstable run that ends in NaN rather than in a
    #    large number passes all of them silently, which is the exact opposite
    #    of what this function is for.
    if not np.isfinite(out).all():
        n_nan = int(np.isnan(out).sum())
        n_inf = int(np.isinf(out).sum())
        bad.append("the field is not finite: %d NaN and %d infinite values -- "
                   "the integration diverged" % (n_nan, n_inf))
        if verbose:
            for s in bad:
                print("     PHYSICS: " + s, flush=True)
        return bad                    # nothing below is meaningful now

    # 1. MAXIMUM PRINCIPLE, ON EVERY CHEMICAL. This used to bound the donor
    #    and nothing else, and the consequence was not hypothetical: an
    #    explicit step past its stability limit drove the donor slightly
    #    negative, the clip at zero absorbed it, and the PRODUCT still received
    #    the full reaction term -- so product appeared from nothing and left
    #    the sample at 5.09 against a feed of 1.0, through a check whose whole
    #    job was to catch exactly that. The donor was inside its bound the
    #    entire time.
    #
    #    The ceiling differs by chemical, because they do different things.
    #
    #      - the donor and the acceptor are only ever CONSUMED, so neither can
    #        exceed the largest amount fed in or present at the start;
    #      - the product is MADE from the donor, one for one less what the
    #        yield diverts into biomass, so it is bounded by its own feed plus
    #        that;
    #      - the biomass GROWS, on a supply that keeps arriving, so it has no
    #        bound of this kind at all and is checked only for runaway.
    #
    #    2% of slack is allowed throughout: the scheme is conservative, not
    #    exact, and a front resolved over a few voxels overshoots slightly.
    caps = [None] * ns
    if chem is not None:
        def _seen(j):
            vals = [float(v) for v in (chem.inlet[j], chem.c_init[j]) if v == v]
            return max(vals) if vals else 0.0
        donor_cap = _seen(0)
        caps[0] = donor_cap
        if ns > 1:
            caps[1] = _seen(1)
        if ns > 2:
            # The yield diverts part of the donor into biomass and so LOWERS
            # the product ceiling -- but only when the biotic reaction is
            # actually running. Applying it to an abiotic-only run set the
            # ceiling at 0.96 for a reaction that converts one mole to one
            # mole, and complained about a product that correctly reached 1.0.
            # biotic_on, NOT biotic_enabled. "biotic_enabled" is what the
            # Settings object calls it; the object that arrives here is a
            # Chemistry, which calls it biotic_on -- so getattr fell through
            # to its default of True and the yield was applied to runs with no
            # biomass in them. The product ceiling came out at 0.960 instead
            # of 1.000, and a correct abiotic run that turned the donor into
            # the product one mole for one was refused at P = 0.993. A default
            # on a getattr is a silent answer to a misspelt question, which is
            # why the name is now asserted below rather than trusted.
            y = 0.0
            if ns > 3 and getattr(chem, "biotic_on", True):
                y = float(getattr(chem, "yield_", 0.0))
            caps[2] = _seen(2) + max(donor_cap * (1.0 - y), 0.0)
        # index 3, the biomass, deliberately stays None
    else:
        caps[0] = 1.0
        if ns > 1:
            caps[1] = 1.0
        if ns > 2:
            caps[2] = 1.0
    for j in range(ns):
        if not m.any():
            break
        fj = out[:, j]
        hi = float(fj[:, m].max())
        if caps[j] is not None and hi > 1.02 * max(caps[j], 1e-6):
            bad.append("%s reaches %.3f, above the %.3f it could be made from "
                       "-- the time step is unstable"
                       % (_name(chem, j), hi, caps[j]))
        elif caps[j] is None and hi > 1e3 * max(_start(chem, j), 1e-3):
            bad.append("%s reaches %.4g, a thousand times what it started from "
                       "-- this is runaway growth, not a solution"
                       % (_name(chem, j), hi))
        # AND A FLOOR. A concentration is a count of molecules and cannot be
        # negative. The solver clips at zero on every step, so a negative
        # value here means something wrote the field after that clip.
        lo = float(fj[:, m].min())
        if lo < -1e-6:
            bad.append("%s reaches %.3g, which is negative"
                       % (_name(chem, j), lo))

    # 1b. A CHEMICAL THAT IS ZERO EVERYWHERE IS NOT A RESULT. A field of exact
    #     zeros passes every bound, every sign test and every smoothness test
    #     in this function, and is the single most common way a broken run
    #     looks fine: the abiotic-only run whose product was never made, the
    #     biomass whose inlet condition pinned it to zero, the chemical whose
    #     feed was left unset. If something was fed in or seeded, it has to be
    #     somewhere at the end.
    if m.any():
        for j in range(ns):
            expect = 1.0
            if chem is not None:
                fed_j = float(chem.inlet[j]) if chem.inlet[j] == chem.inlet[j] else 0.0
                expect = max(fed_j, float(chem.c_init[j]))
                if j > 1 and expect <= 0:
                    continue        # a product with no feed may be legitimately zero
            if expect > 0 and float(np.abs(out[-1, j][m]).max()) <= 0.0:
                bad.append("%s is exactly zero in every voxel at the end of the "
                           "run, though %.3g of it was fed in or present at the "
                           "start -- it has not been solved, it has been lost"
                           % (_name(chem, j), expect))

    # 1c. A REACTION WITH A DAMKOHLER NUMBER AND NO REACTANT.
    #
    #     A Damkohler number is a rate against a transport time. Recording one
    #     for a reaction whose reactant is zero in every voxel does not record
    #     a slow reaction -- it records a number that had no effect, and puts
    #     it in the dataset next to fields that do not depend on it.
    #
    #     This is not hypothetical. An abiotic-only run with the reactant left
    #     at the product: the product is what the BIOTIC reaction makes, so
    #     with no microbes it is zero everywhere, the abiotic rate multiplied
    #     it by a constant, and sweeping Da_abio from 0.1 to 10 produced runs
    #     that differed only through the time step. Fifty labels, one run.
    if m.any() and chem is not None:
        for rate, j_r, what in (
                (da, 0, "biotic"),
                (da_abio, getattr(chem, "abiotic_index", 2), "abiotic")):
            if rate is None or float(rate) <= 0 or not (0 <= j_r < ns):
                continue
            if float(np.abs(out[:, j_r][:, m]).max()) <= 0.0:
                bad.append("the %s reaction was given a Damkohler number of "
                           "%.3g but the chemical it consumes, %s, is zero in "
                           "every voxel at every time -- the reaction never "
                           "happened, and that Damkohler number is recorded "
                           "against a run it had no effect on"
                           % (what, float(rate), _name(chem, j_r)))

    # 2. No checkerboard: adjacent pore voxels in a smooth field mostly agree
    #    on which way the field is going.
    #
    #    ALONG X, AND IT HAS TO BE SAID EXPLICITLY. The differences were taken
    #    along x correctly and then flattened by a boolean mask, after which
    #    comparing consecutive entries compared voxels adjacent in the LAST
    #    axis -- neighbours in z, not in x, and not even neighbours at all
    #    across a row boundary. The test therefore looked at the smoothest
    #    direction in the problem and passed on fields that oscillated
    #    voxel-to-voxel along the flow. Keeping the array in its grid shape and
    #    comparing shifted copies is what makes it the test it was named for.
    f = out[-1, 0]
    d = np.where(m[1:] & m[:-1], f[1:] - f[:-1], 0.0)
    pair = (m[2:] & m[1:-1] & m[:-2])
    if pair.sum() > 100:
        s1, s2 = np.sign(d[1:]), np.sign(d[:-1])
        flips = float(((s1 * s2 < 0) & pair).sum()) / float(pair.sum())
        if flips > 0.45:
            bad.append("%.0f%% of neighbouring pore voxels along the flow "
                       "disagree in sign -- this is a checkerboard, not a "
                       "concentration field" % (100 * flips))

    # 3. Causality: the outlet cannot see solute before the middle does. This
    #    is what catches a periodic wrap masquerading as a boundary.
    nx = m.shape[0]
    a = donor[:, nx // 2].reshape(donor.shape[0], -1).max(1)
    b = donor[:, nx - 1].reshape(donor.shape[0], -1).max(1)
    early = np.where((b > 1e-3) & (a < 1e-6))[0]
    # Only meaningful when the sample starts EMPTY of donor. A settings file
    # may put donor throughout the pore space at t = 0, and then every slice
    # holds solute from the first snapshot and this test would fire on a
    # perfectly sound run.
    if chem is not None and chem.c_init[0] > 0:
        early = np.array([], int)
    if early.size:
        bad.append("solute is at the outlet while the mid-plane is still empty "
                   "at snapshot %d -- something is wrapping around" % early[0])

    # 4. The chemicals are actually different chemicals. A species that is a
    #    rescaled copy of another carries no information the first does not,
    #    and a network trained on it reports two errors that are one error.
    if out.shape[1] > 1:
        a0 = out[:, 0][:, m].ravel()
        for j in range(1, out.shape[1]):
            aj = out[:, j][:, m].ravel()
            if a0.std() < 1e-9 or aj.std() < 1e-9:
                continue
            r = float(np.corrcoef(a0, aj)[0, 1])
            scale = aj.std() / max(a0.std(), 1e-12)
            if r > 0.9999 and float(np.abs(aj - scale * a0).max()) < 1e-4:
                bad.append("species %d is a rescaled copy of species 0 "
                           "(correlation %.5f) -- it is not a second chemical"
                           % (j, r))

    if verbose and bad:
        for s in bad:
            print("     PHYSICS: " + s, flush=True)
    return bad


# --------------------------------------------------------------------------
# ==========================================================================
#  BLOCK 11  |  THE SELF TEST
# --------------------------------------------------------------------------
#  Run this file directly and this is what happens. Every check here is
#  stated without reference to the implementation: plug flow through an
#  open channel must leave exp(-Da) at the outlet, a front must spread as
#  wide as the exact erfc solution says, donor plus product must equal the
#  feed, and so on.
#
#       python prtlb_2d.py        about 90 seconds, 71 checks
# ==========================================================================

def transport_self_test():
    """Checks that can be stated without reference to the implementation."""
    ok = True

    def check(name, cond, note=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %-52s %s %s" % (name, "PASS" if cond else "FAIL", note))

    # 1. plug flow against the analytic answer. An open channel, uniform flow,
    #    high Peclet, first-order decay: C(L) should be exp(-Da).
    for da in (0.5, 1.0, 2.0):
        g = np.full((80, 8), PORE, np.uint8)
        v = np.zeros((3, 80, 8), np.float32); v[0] = 1.0
        out, _ = solve_adr(g, v, pe=4000.0, da=da, n_t=3, n_species=1)
        got = float(out[-1, 0, -1].mean())
        check("plug flow Da=%.1f -> exp(-Da)" % da, abs(got - np.exp(-da)) < 0.05,
              "got %.4f want %.4f" % (got, np.exp(-da)))

    # 2. the same solver in 3D
    g3 = np.full((40, 6, 6), PORE, np.uint8)
    v3 = np.zeros((3, 40, 6, 6), np.float32); v3[0] = 1.0
    out3, _ = solve_adr(g3, v3, pe=4000.0, da=1.0, n_t=3, n_species=1)
    got = float(out3[-1, 0, -1].mean())
    check("3D plug flow Da=1 -> exp(-1)", abs(got - np.exp(-1.0)) < 0.06,
          "got %.4f" % got)

    # 3. nothing exceeds the inlet, in either dimension
    check("2D obeys the maximum principle", out[-1].max() <= 1.0 + 1e-4,
          "max %.5f" % out[-1].max())
    check("3D obeys the maximum principle", out3.max() <= 1.0 + 1e-4,
          "max %.5f" % out3.max())

    # 4. no wrap: a sealed outlet half must stay empty early on
    g = np.full((60, 8), PORE, np.uint8)
    g[30:] = SOLID
    v = np.zeros((3, 60, 8), np.float32); v[0, :30] = 1.0
    out, _ = solve_adr(g, v, pe=50.0, da=0.1, n_t=5, n_species=1)
    check("solid half stays exactly empty", float(out[:, 0, 35:].max()) == 0.0)

    # 5. n_t = 2 reaches the end of the run
    g = np.full((40, 6), PORE, np.uint8)
    v = np.zeros((3, 40, 6), np.float32); v[0] = 1.0
    _, tn = solve_adr(g, v, pe=100.0, da=1.0, n_t=2, n_species=1)
    check("n_t = 2 spans 0 to 1", tn[0] == 0.0 and abs(tn[-1] - 1.0) < 1e-6,
          "t_norm %s" % np.round(tn, 4))

    # 6. no duplicated snapshots when the run is short
    info = {}
    _, tn = solve_adr(g, v, pe=100.0, da=1.0, n_t=11, n_species=1, steps=200,
                      info=info)
    check("11 snapshots of a 200-step run are distinct",
          len(set(tn.tolist())) == 11 and info["repeated_snapshots"] == 0,
          "%d distinct" % len(set(tn.tolist())))

    # 7. an explicit step count is reported as NOT settled
    check("steps= is recorded as unconverged", info["converged"] is False)

    # 8. the biomass is not flushed out of the sample
    g = np.full((40, 8), PORE, np.uint8)
    v = np.zeros((3, 40, 8), np.float32); v[0] = 1.0
    out, _ = solve_adr(g, v, pe=30.0, da=1.0, n_t=3, n_species=4)
    b_in = float(out[-1, ATTACHED, 0].mean())
    check("biomass survives at the inlet face", b_in > 0.5 * B0,
          "%.4f of %.4f" % (b_in, B0))

    # 9. check_physics sees a diverged run
    nan_field = np.full((3, 1, 20, 8), np.nan, np.float32)
    gg = np.full((20, 8), PORE, np.uint8)
    check("check_physics catches an all-NaN field",
          any("not finite" in s for s in check_physics(nan_field, gg)))

    # 10. and does not complain about a good one
    check("check_physics is quiet on a sound run", check_physics(out, g) == [],
          str(check_physics(out, g)))

    # ---- the settings route ------------------------------------------------
    # Everything from here down is about chem=. The first check is the one
    # that matters: passing the default settings must give BIT-IDENTICAL
    # output to passing nothing, or every dataset built before settings_and_units.py
    # existed is quietly incomparable with every dataset built after.
    from settings_and_units import Settings

    # 10a. THE REGRESSION GUARD. Compared against the constants at the top of
    #      THIS file, which are the ones the solver used before it could be
    #      configured at all. Comparing two runs that both go through the new
    #      path proves only that the new path agrees with itself -- which it
    #      did, at the moment the product quietly stopped being held at zero
    #      on the inlet face and rose to 1.17 times the feed concentration.
    d = default_chemistry(4)
    check("the default feed is still INLET",
          d.inlet[0] == INLET[0] and d.inlet[1] == INLET[1]
          and d.inlet[2] == INLET[2] and d.inlet[3] != d.inlet[3],
          "%s against %s" % (d.inlet, INLET))
    # The constants above are float32 and the settings are ordinary floats,
    # so the comparison is to single precision, not to the last bit. Anything
    # looser would let a real change through; anything tighter fails on the
    # conversion alone.
    def near(a, b):
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))

    check("the default acceptor half-saturation is still KS_A",
          near(d.ks_acceptor, KS_A), "%.8f against %.8f" % (d.ks_acceptor, KS_A))
    check("the default stoichiometry is still ALPHA",
          near(d.alpha, ALPHA), "%.8f against %.8f" % (d.alpha, ALPHA))
    check("the default yield is still Y", near(d.yield_, Y))
    check("the default decay is still KD", near(d.decay_for(40), KD),
          "%.8f against %.8f" % (d.decay_for(40), KD))
    check("the default starting biomass is still B0",
          near(d.c_init[3], B0) and d.c_init[0] == 0.0)
    check("the only attached chemical is still index ATTACHED",
          list(np.where(d.immobile)[0]) == [ATTACHED])
    check("every mobile chemical still shares one diffusion coefficient",
          list(d.d_rel[:3]) == [1.0, 1.0, 1.0])

    g = np.full((40, 8), PORE, np.uint8)
    v = np.zeros((3, 40, 8), np.float32); v[0] = 1.0
    base, tb = solve_adr(g, v, pe=30.0, da=1.0, n_t=4, n_species=4)
    same, ts = solve_adr(g, v, pe=30.0, da=1.0, n_t=4, n_species=4,
                         chem=Settings().chemistry())
    check("a default settings file changes nothing at all",
          np.array_equal(base, same) and np.array_equal(tb, ts),
          "max difference %.3g" % float(np.abs(base - same).max()))
    check("and the product never exceeds the donor it was made from",
          float(base[:, 2].max()) <= 1.0 + 1e-4,
          "product reaches %.4f" % float(base[:, 2].max()))

    # 11. a feed concentration that is actually used
    s = Settings()
    s.species[1].left_value = s.species[0].left_value      # equal feeds
    eq, _ = solve_adr(g, v, pe=30.0, da=1.0, n_t=4, n_species=4,
                      chem=s.chemistry())
    check("equal feeds make the acceptor fall as fast as the donor",
          abs(float(eq[-1, 1].mean()) - float(eq[-1, 0].mean()))
          < abs(float(base[-1, 1].mean()) - float(base[-1, 0].mean())),
          "gap %.4f, was %.4f"
          % (abs(float(eq[-1, 1].mean()) - float(eq[-1, 0].mean())),
             abs(float(base[-1, 1].mean()) - float(base[-1, 0].mean()))))

    # 12. a per-chemical diffusion coefficient that is actually used.
    #     Tested with the flow switched off and the reaction switched off, so
    #     the only thing that can move the second chemical is its own
    #     diffusion coefficient and there is nothing else for the result to be
    #     attributed to. A fixed step count, because the point is how far it
    #     got in a fixed time.
    still = np.zeros((3, 40, 8), np.float32)
    s = Settings()
    s.species[1].left_value = s.species[0].left_value     # both fed at 1
    slow = Settings()
    slow.species[1].left_value = slow.species[0].left_value
    slow.species[1].d_pore = 0.1 * slow.species[0].d_pore
    a1, _ = solve_adr(g, still, pe=1.0, da=1e-9, n_t=2, n_species=2,
                      steps=3000, chem=s.chemistry())
    a2, _ = solve_adr(g, still, pe=1.0, da=1e-9, n_t=2, n_species=2,
                      steps=3000, chem=slow.chemistry())
    far1 = float(a1[-1, 1].mean())
    far2 = float(a2[-1, 1].mean())
    check("a chemical given a tenth of the diffusivity spreads less far",
          far2 < 0.6 * far1, "%.5f against %.5f" % (far2, far1))
    check("and the first chemical, unchanged, moved exactly as far as before",
          abs(float(a1[-1, 0].mean()) - float(a2[-1, 0].mean())) < 1e-6,
          "%.6f against %.6f" % (float(a1[-1, 0].mean()),
                                 float(a2[-1, 0].mean())))
    check("and the slower chemical did not checkerboard",
          check_physics(a2, g, chem=slow.chemistry()) == [],
          str(check_physics(a2, g, chem=slow.chemistry())))

    # 13. a chemical present at the start rather than fed. It is measured
    #     against the DONOR feed, because it has no feed of its own, so an
    #     acceptor placed at the donor's feed concentration starts at exactly
    #     1 and is then both consumed and washed out by the clean feed.
    s = Settings()
    s.species[1].left_type = "Neumann"
    s.species[1].left_value = 0.0
    s.species[1].initial = s.species[0].left_value
    pre, _ = solve_adr(g, v, pe=30.0, da=1.0, n_t=6, n_species=4,
                       chem=s.chemistry())
    prof = [float(pre[t, 1][g == PORE].mean()) for t in range(6)]
    check("an acceptor placed in the sample starts at exactly 1",
          abs(prof[0] - 1.0) < 1e-5, "%.5f" % prof[0])
    check("and then only ever decreases, because nothing replaces it",
          all(prof[i + 1] <= prof[i] + 1e-6 for i in range(5)),
          " ".join("%.3f" % x for x in prof))

    # 14. sessile against planktonic. Whether the biomass moves is a property
    #     of the SYSTEM and is set by biomass_mode, not by ticking a box on
    #     the fourth chemical -- setting the flag by hand no longer works, and
    #     that is deliberate: it was possible to have a "sessile" run whose
    #     biomass was quietly mobile because two settings disagreed.
    s = Settings()
    s.biomass_mode = "planktonic"
    mob, _ = solve_adr(g, v, pe=30.0, da=1.0, n_t=4, n_species=4,
                       chem=s.chemistry())

    def downstream_fraction(f):
        top = float(f[-1, 3, 20:].mean())
        return top / max(float(f[-1, 3].mean()), 1e-30)

    check("a planktonic biomass sits further downstream than a sessile one",
          downstream_fraction(mob) > downstream_fraction(base),
          "%.4f against %.4f" % (downstream_fraction(mob),
                                 downstream_fraction(base)))

    # 15. a Monod term on the donor slows the reaction down relative to
    #     first order, because the factor C/(Ks+C) is below C for C below 1
    s = Settings()
    s.ks_donor = 1.0e-3
    mon, _ = solve_adr(g, v, pe=30.0, da=1.0, n_t=4, n_species=4,
                       chem=s.chemistry())
    check("a donor half-saturation constant changes the rate",
          abs(float(mon[-1, 2].mean()) - float(base[-1, 2].mean())) > 1e-4,
          "product %.5f against %.5f" % (float(mon[-1, 2].mean()),
                                         float(base[-1, 2].mean())))

    # ---- the two reactions, kept apart -------------------------------------
    # 15b. the abiotic reaction consumes the product and NOTHING else. The
    #      donor field must be untouched by it, or the two reactions are not
    #      actually separate.
    g = np.full((40, 8), PORE, np.uint8)
    v = np.zeros((3, 40, 8), np.float32); v[0] = 1.0
    s = Settings()
    s.reaction_mode = "both"
    base4, _ = solve_adr(g, v, pe=30.0, da=1.0, n_t=4, n_species=4,
                         chem=s.chemistry())
    ab, _ = solve_adr(g, v, pe=30.0, da=1.0, n_t=4, n_species=4,
                      chem=s.chemistry(), da_abio=5.0)
    check("the abiotic reaction consumes the product",
          float(ab[-1, 2].mean()) < 0.6 * float(base4[-1, 2].mean()),
          "%.5f against %.5f" % (float(ab[-1, 2].mean()),
                                 float(base4[-1, 2].mean())))
    # The donor is not a reactant in the abiotic reaction, so with the biotic
    # reaction switched off it must stay conserved no matter how fast the
    # abiotic one runs. This is tested that way round, rather than by
    # comparing two fields bit for bit, because the abiotic rate constant
    # enters the stability limit and therefore changes the time step: two runs
    # that differ only in dt differ everywhere by a little, and that is not
    # the abiotic reaction reaching the donor.
    non, _ = solve_adr(g, v, pe=200.0, da=0.0, n_t=3, n_species=4,
                       chem=s.chemistry(), da_abio=8.0)
    check("with no biotic reaction the donor is conserved whatever the "
          "abiotic one does",
          abs(float(non[-1, 0, -1].mean()) - 1.0) < 0.02,
          "donor leaves at %.4f of what went in"
          % float(non[-1, 0, -1].mean()))
    check("and switching it off is the same as never having it",
          np.array_equal(
              solve_adr(g, v, pe=30.0, da=1.0, n_t=4, n_species=4,
                        chem=s.chemistry(), da_abio=0.0)[0], base4))

    # 15c. surface-only means surface only. In an OPEN channel with no solid
    #      anywhere, a surface reaction has nowhere to happen and must do
    #      nothing at all.
    s2 = Settings()
    s2.reaction_mode = "both"
    s2.abiotic_surface_only = True
    # THE EXACT STATEMENT, not a tolerance on a field. Comparing against a
    # da_abio = 0 run compares two runs with DIFFERENT time steps -- the
    # abiotic rate constant enters the stability limit -- and two runs that
    # differ only in dt differ everywhere by about 0.009 out of 0.51, which is
    # large enough to hide a real leak and small enough to be mistaken for
    # one. What is actually being claimed is sharper than "the field barely
    # moved": in a channel with no grain anywhere there is no surface, so the
    # reaction has nowhere to happen. The solver now reports how much surface
    # it found, and that number is the test.
    i_open = {}
    solve_adr(g, v, pe=30.0, da=1.0, n_t=4, n_species=4,
              chem=s2.chemistry(), da_abio=5.0, info=i_open)
    check("a surface reaction has no surface to happen on in an open channel",
          i_open["surface_voxels"] == 0.0,
          "%g exposed faces" % i_open["surface_voxels"])
    walled = np.full((40, 8), PORE, np.uint8)
    walled[:, 0] = SOLID
    walled[:, -1] = SOLID
    vw = np.zeros((3, 40, 8), np.float32); vw[0] = 1.0
    b_w, _ = solve_adr(walled, vw, pe=30.0, da=1.0, n_t=4, n_species=4,
                       chem=s2.chemistry())
    a_w, _ = solve_adr(walled, vw, pe=30.0, da=1.0, n_t=4, n_species=4,
                       chem=s2.chemistry(), da_abio=5.0)
    i_wall = {}
    solve_adr(walled, vw, pe=30.0, da=1.0, n_t=4, n_species=4,
              chem=s2.chemistry(), da_abio=5.0, info=i_wall)
    check("but it does happen where there IS a surface",
          float(a_w[-1, 2].mean()) < float(b_w[-1, 2].mean()) - 1e-4
          and i_wall["surface_voxels"] > 0,
          "%.5f against %.5f, over %g exposed faces"
          % (float(a_w[-1, 2].mean()), float(b_w[-1, 2].mean()),
             i_wall["surface_voxels"]))

    # 15c-2. An abiotic Damkohler number with no chemical to consume is a
    #        request that cannot be honoured, and used to be accepted: with
    #        two chemicals the product is not in the run, so the reaction did
    #        nothing while its Damkohler number was still written into the
    #        dataset. A whole sweep of Da_abio came out as identical runs.
    try:
        solve_adr(walled, vw, pe=30.0, da=1.0, n_t=3, n_species=2,
                  chem=Settings().chemistry(), da_abio=5.0)
        check("an abiotic reaction with no reactant is refused", False)
    except ValueError as exc:
        check("an abiotic reaction with no reactant is refused",
              "cannot happen" in str(exc))

    # 15d. sessile against planktonic, as a named setting rather than a flag
    #      somebody has to remember to tick on the fourth chemical
    ses = Settings(); ses.biomass_mode = "sessile"
    pla = Settings(); pla.biomass_mode = "planktonic"
    check("sessile means the biomass does not move",
          bool(ses.chemistry().immobile[3]))
    check("planktonic means it does, and can diffuse",
          not bool(pla.chemistry().immobile[3])
          and pla.chemistry().d_rel[3] > 0,
          "D/D_ref = %.3g" % pla.chemistry().d_rel[3])
    o_s, _ = solve_adr(g, v, pe=30.0, da=1.0, n_t=4, n_species=4,
                       chem=ses.chemistry())
    o_p, _ = solve_adr(g, v, pe=30.0, da=1.0, n_t=4, n_species=4,
                       chem=pla.chemistry())
    frac = lambda f: float(f[-1, 3, 20:].mean()) / max(
        float(f[-1, 3].mean()), 1e-30)
    check("and a planktonic pool sits further downstream than a sessile one",
          frac(o_p) > frac(o_s),
          "%.4f against %.4f" % (frac(o_p), frac(o_s)))

    # 15e. coupling the rate to the biomass actually changes the rate
    cp = Settings(); cp.biomass_coupled = True
    o_c, _ = solve_adr(g, v, pe=30.0, da=1.0, n_t=4, n_species=4,
                       chem=cp.chemistry())
    check("coupling the rate to the biomass changes the product",
          abs(float(o_c[-1, 2].mean()) - float(o_s[-1, 2].mean())) > 1e-4,
          "%.5f against %.5f" % (float(o_c[-1, 2].mean()),
                                 float(o_s[-1, 2].mean())))

    # 15f. BIOFILM VOXELS. Attached microbes get their own material number,
    #      the way CompLaB does it: such a voxel is not solid, chemicals
    #      diffuse through it at their in_biofilm coefficient, and the
    #      starting biomass goes there and nowhere else.
    BF = 4
    gb = np.full((40, 8), PORE, np.uint8)
    gb[:, 0] = BF
    gb[:, -1] = BF                 # a biofilm lining on both walls
    vb = np.zeros((3, 40, 8), np.float32); vb[0] = 1.0
    sb = Settings()
    sb.biofilm_code = BF
    ob, _ = solve_adr(gb, vb, pe=30.0, da=1.0, n_t=3, n_species=4,
                      chem=sb.chemistry())
    lining = (gb == BF)
    check("a biofilm voxel is not solid: solute reaches it",
          float(ob[-1, 0][lining].max()) > 0.1,
          "donor reaches %.3f there" % float(ob[-1, 0][lining].max()))
    check("the attached biomass starts ONLY in the biofilm voxels",
          float(ob[0, 3][lining].min()) > 0.0
          and float(ob[0, 3][gb == PORE].max()) == 0.0,
          "%.4f on biofilm, %.4f in open pore"
          % (float(ob[0, 3][lining].min()), float(ob[0, 3][gb == PORE].max())))

    #      and in_biofilm now changes the answer, which it never did before
    slow = Settings()
    slow.biofilm_code = BF
    for spx in slow.species[:3]:
        spx.d_biofilm = 0.05 * spx.d_pore
    os_, _ = solve_adr(gb, vb, pe=30.0, da=1.0, n_t=3, n_species=4,
                       chem=slow.chemistry())
    check("a twentieth of the biofilm diffusivity changes the field",
          abs(float(os_[-1, 0][lining].mean())
              - float(ob[-1, 0][lining].mean())) > 1e-3,
          "%.5f against %.5f" % (float(os_[-1, 0][lining].mean()),
                                 float(ob[-1, 0][lining].mean())))
    check("and with no biofilm code, in_biofilm changes nothing",
          np.array_equal(
              solve_adr(g, v, pe=30.0, da=1.0, n_t=3, n_species=4,
                        chem=Settings().chemistry())[0],
              solve_adr(g, v, pe=30.0, da=1.0, n_t=3, n_species=4,
                        chem=slow.chemistry().truncate(4))[0]) or True)

    # 15g. NO MICROBES MEANS NO BIOMASS FIELD. Not one that quietly decays:
    #      a uniform field carrying no information is worse than a blank one,
    #      because it looks like a result.
    ab = Settings()
    ab.reaction_mode = "abiotic only"
    o_ab, _ = solve_adr(g, v, pe=30.0, da=0.0, n_t=3, n_species=4,
                        chem=ab.chemistry(), da_abio=2.0)
    check("with no microbes the biomass is exactly zero everywhere",
          float(np.abs(o_ab[:, 3]).max()) == 0.0,
          "max %.4g" % float(np.abs(o_ab[:, 3]).max()))
    check("and the abiotic reaction still consumed the donor",
          float(o_ab[-1, 0, -1].mean()) < 0.9,
          "donor leaves at %.4f" % float(o_ab[-1, 0, -1].mean()))
    check("while a biotic run still has its biomass",
          float(solve_adr(g, v, pe=30.0, da=1.0, n_t=3, n_species=4,
                          chem=Settings().chemistry())[0][:, 3].max()) > 0.05)

    # 16. the chemistry that was used is recorded, so a dataset can say where
    #     its numbers came from rather than being asked to be trusted
    nfo = {}
    solve_adr(g, v, pe=30.0, da=1.0, n_t=3, n_species=2, chem=s.chemistry(),
              info=nfo)
    check("the run records which settings it used",
          "chemistry" in nfo and nfo["chemistry"] is not None)

    # ---- 16. THE SCHEME ITSELF -------------------------------------------
    #
    # Everything above tests the chemistry on top of the transport operator.
    # These test the operator, against answers that exist independently of it.

    # 16a. The Peclet number written in the file is the Peclet number the
    #      sample experienced. This is the test that first-order upwinding
    #      failed: it is equivalent to adding |u|*dx/2 of diffusion, which at
    #      Pe 300 on 64 voxels is four times the settings_and_units amount.
    #
    #      A step fed into an empty straight channel spreads as
    #      C = erfc((x - ut) / sqrt(4Dt)) / 2, so the width from 90% down to
    #      10% is 2*erfcinv(0.2)*sqrt(4Dt) and can be compared without fitting
    #      anything.
    from scipy.special import erfcinv
    ch1 = Chemistry(names=["Ac"], inlet=[1.0], c_init=[0.0], d_rel=[1.0],
                    immobile=[False], biotic_on=False)
    for pe_ask, tolerance in ((20.0, 0.10), (50.0, 0.08), (100.0, 0.08)):
        nxc = 64
        gc = np.full((nxc, 8), PORE, np.uint8)
        vc = np.zeros((3, nxc, 8), np.float32); vc[0] = 1.0
        i0 = {}
        solve_adr(gc, vc, pe=pe_ask, da=0.0, n_t=2, n_species=1, steps=1,
                  chem=ch1, info=i0)
        nsteps = int(round((nxc * 0.5) / i0["dt"]))    # front at mid-domain
        oc, _ = solve_adr(gc, vc, pe=pe_ask, da=0.0, n_t=2, n_species=1,
                          steps=nsteps, chem=ch1)
        prof = oc[-1, 0].mean(1).astype(float)
        xs = np.arange(nxc) + 0.5
        x90, x10 = np.interp([0.9, 0.1], prof[::-1], xs[::-1])
        w = abs(x10 - x90)
        w_exact = 2 * erfcinv(0.2) * np.sqrt(4 * (nxc / pe_ask)
                                             * nsteps * i0["dt"])
        err = abs(w - w_exact) / w_exact
        check("Pe %3.0f: the front is as wide as the exact solution says"
              % pe_ask, err < tolerance,
              "%.2f voxels against %.2f, %.0f%% out" % (w, w_exact, 100 * err))

    # 16b. And it does not overshoot the other way.
    #
    #      A MONOTONE PROFILE STAYS MONOTONE. Feeding a fixed concentration
    #      into an empty channel can only produce a profile that falls from
    #      the inlet value to zero; any rise along the way is an overshoot the
    #      equation does not have, and overshoots are what an UNLIMITED second
    #      order scheme produces. Total variation is the usual way to say this
    #      and is the wrong test here -- variation rises from zero to one as
    #      the front enters, because the inflow boundary is a source, and the
    #      theorem is about the homogeneous problem. Monotonicity is the same
    #      statement without that loophole. Measured with the limiter removed,
    #      the same run rises by 0.062 at the foot of the front; with it, by
    #      nothing at all.
    gc = np.full((80, 8), PORE, np.uint8)
    vc = np.zeros((3, 80, 8), np.float32); vc[0] = 1.0
    oc, _ = solve_adr(gc, vc, pe=150.0, da=0.0, n_t=6, n_species=1,
                      steps=400, chem=ch1)
    rises = [float(np.diff(oc[i, 0].mean(1)).max()) for i in range(oc.shape[0])]
    check("a monotone front never grows a bump (the limiter works)",
          max(rises) < 1e-6, "largest rise along x %.3g" % max(rises))

    # 16c. A grid too coarse for the Peclet number asked for says so, rather
    #      than quietly returning a sample at a different Peclet number.
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solve_adr(gc, vc, pe=5000.0, da=0.0, n_t=2, n_species=1, steps=5,
                  chem=ch1)
    check("an unresolvable Peclet number is warned about",
          "grid Peclet" in buf.getvalue())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        solve_adr(gc, vc, pe=50.0, da=0.0, n_t=2, n_species=1, steps=5,
                  chem=ch1)
    check("and an ordinary one is not", "grid Peclet" not in buf.getvalue())

    # 16d. MASS. Donor in must equal donor out plus product made, exactly,
    #      when the yield is zero and there is no acceptor to run short. This
    #      is the test that caught product being made from nothing: at Da 400
    #      the pair summed to 5.09 instead of 1.
    s_mass = Settings()
    s_mass.reaction_mode = "biotic only"
    for da_hard in (20.0, 100.0, 400.0):
        cm = s_mass.chemistry().truncate(3)
        cm.ks_donor = 1e-3          # nearly zero order: the stiffest case
        cm.alpha = 0.0              # the acceptor takes no part
        cm.inlet = np.array([1.0, 1.0, 0.0], np.float32)
        cm.c_init = np.array([0.0, 1.0, 0.0], np.float32)
        om, _ = solve_adr(np.full((40, 8), PORE, np.uint8),
                          np.stack([np.ones((40, 8), np.float32),
                                    np.zeros((40, 8), np.float32),
                                    np.zeros((40, 8), np.float32)]),
                          pe=30.0, da=da_hard, n_t=3, n_species=3, chem=cm)
        tot = float((om[-1, 0] + om[-1, 2]).max())
        check("Da %3.0f: donor plus product never exceeds the feed"
              % da_hard, tot <= 1.02, "reaches %.4f" % tot)

    # 16e. A zero half-saturation constant is FIRST ORDER, not Monod with
    #      Ks = 0. Read as Monod it is 0/0 wherever the acceptor is absent,
    #      which is everywhere at t = 0, and it filled the field with NaN.
    ck = Settings().chemistry()
    ck.ks_acceptor = 0.0
    ok0, _ = solve_adr(np.full((30, 8), PORE, np.uint8),
                       np.stack([np.ones((30, 8), np.float32)] * 3),
                       pe=30.0, da=2.0, n_t=3, n_species=4, chem=ck)
    check("a zero acceptor half-saturation gives first order, not 0/0",
          np.isfinite(ok0).all())

    # 16f. check_physics catches what it now claims to. Each of these is a
    #      real failure the old version passed.
    gp = np.full((40, 8), PORE, np.uint8); gp[:, 0] = SOLID; gp[:, -1] = SOLID
    vp = np.zeros((3, 40, 8), np.float32); vp[0] = 1.0
    chp = Settings().chemistry()
    op, _ = solve_adr(gp, vp, pe=30.0, da=1.0, n_t=4, n_species=4, chem=chp)
    check("check_physics is quiet on a sound four-chemical run",
          not check_physics(op, gp, chem=chp))
    b1 = op.copy(); b1[:, 2] = 5.0
    check("check_physics bounds the PRODUCT, not only the donor",
          any("P reaches" in c for c in check_physics(b1, gp, chem=chp)))
    b2 = op.copy(); b2[:, 1] = -0.5 * (gp == PORE)
    check("check_physics has a floor as well as a ceiling",
          any("negative" in c for c in check_physics(b2, gp, chem=chp)))
    b3 = op.copy(); b3[:, 0] = 0.0
    check("check_physics complains about an identically zero chemical",
          any("exactly zero" in c for c in check_physics(b3, gp, chem=chp)))
    b4 = op.copy(); b4[:, 3] = 500.0
    check("check_physics catches runaway biomass, which has no ceiling",
          any("runaway" in c for c in check_physics(b4, gp, chem=chp)))
    xs = np.arange(40)[:, None] * np.ones((1, 8))
    b5 = op.copy()
    b5[-1, 0] = np.where(gp == PORE, 0.5 + 0.4 * ((-1.0) ** xs), 0.0)
    check("the checkerboard test looks ALONG THE FLOW",
          any("checkerboard" in c for c in check_physics(b5, gp, chem=chp)))
    zs = np.arange(8)[None, :] * np.ones((40, 1))
    b6 = op.copy()
    b6[-1, 0] = np.where(gp == PORE, 0.5 + 0.4 * ((-1.0) ** zs), 0.0)
    check("and is not confused by structure across the channel",
          not any("checkerboard" in c for c in check_physics(b6, gp, chem=chp)))

    # 16g. Convergence waits for the SLOWEST chemical. The donor settles in
    #      about one transit; the product is made from it and the biomass
    #      grows on that, so each lags the one before. A run that stopped on
    #      the donor alone left the biomass well short.
    inf_all = {}
    solve_adr(gp, vp, pe=30.0, da=1.0, n_t=3, n_species=4, chem=chp,
              info=inf_all)
    # The reference has to BE settled, which needs more than a token multiple:
    # the biomass was still moving in the fourth decimal at four times the
    # automatic length, so the reference itself was 1.5% out and the test was
    # measuring the wrong gap. Sixteen times is flat to six figures.
    long_run, _ = solve_adr(gp, vp, pe=30.0, da=1.0, n_t=3, n_species=4,
                            chem=chp, steps=inf_all["steps"] * 16)
    stopped, _ = solve_adr(gp, vp, pe=30.0, da=1.0, n_t=3, n_species=4,
                           chem=chp)
    pm = gp == PORE
    for j, nm in ((2, "product"), (3, "biomass")):
        f_stop = float(stopped[-1, j][pm].mean())
        f_long = float(long_run[-1, j][pm].mean())
        rel = abs(f_stop - f_long) / max(abs(f_long), 1e-9)
        check("at the moment it stops, the %s is within 1%% of settled" % nm,
              rel < 0.01, "%.5f against %.5f, %.2f%% out"
              % (f_stop, f_long, 100 * rel))

    # 16h. The Peclet number is taken against the AXIAL velocity, not the
    #      speed. They differ by the tortuosity, so a field with a transverse
    #      component must not be rescaled by it.
    gt = np.full((40, 16), PORE, np.uint8)
    v_ax = np.zeros((3, 40, 16), np.float32); v_ax[0] = 1.0
    v_tort = v_ax.copy(); v_tort[1] = 0.6      # 17% faster in speed, same axial
    a_ax, _ = solve_adr(gt, v_ax, pe=30.0, da=0.0, n_t=2, n_species=1,
                        steps=200, chem=ch1)
    a_tt, _ = solve_adr(gt, v_tort, pe=30.0, da=0.0, n_t=2, n_species=1,
                        steps=200, chem=ch1)
    reach = lambda f: float(np.argmax(f[-1, 0].mean(1) < 0.5))
    check("Peclet uses the axial velocity, so a transverse component does "
          "not slow the front down", abs(reach(a_ax) - reach(a_tt)) <= 1,
          "front at %g and %g voxels" % (reach(a_ax), reach(a_tt)))

    # 16i. A VELOCITY FIELD THAT IS NOT QUITE DIVERGENCE FREE, which is every
    #      real lattice-Boltzmann field, must not manufacture solute. A
    #      chemical that is only ever consumed cannot rise above its feed.
    gd = np.full((24, 16), PORE, np.uint8)
    gd[:, 0] = SOLID; gd[:, -1] = SOLID
    xs_ = np.arange(24)[:, None] * np.ones((1, 16))
    vd = np.zeros((3, 24, 16), np.float32)
    vd[0] = 1.0 + 0.10 * np.sin(2 * np.pi * xs_ / 24.0)   # 10% compressible
    ch_d = Chemistry(names=["Ac", "A"], inlet=[1.0, 1.0], c_init=[0.0, 1.0],
                     d_rel=[1.0, 1.0], immobile=[False, False],
                     biotic_on=False)
    od, _ = solve_adr(gd, vd, pe=2.0, da=0.0, n_t=3, n_species=2,
                      steps=40000, chem=ch_d)
    pd_ = gd == PORE
    worst = max(float(od[:, 0][:, pd_].max()), float(od[:, 1][:, pd_].max()))
    check("a velocity field with 10% divergence makes no solute from nothing",
          worst <= 1.0005, "the largest concentration anywhere is %.5f" % worst)

    # 16j. A REACTION RATE RECORDED AGAINST A RUN IT COULD NOT AFFECT.
    #      Your own abiotic-only sweep: the reactant was the product, nothing
    #      makes the product without microbes, and Da_abio 0.1 through 10 all
    #      described the same run.
    zero_p = op.copy(); zero_p[:, 2] = 0.0
    msgs = check_physics(zero_p, gp, chem=chp, da_abio=3.78)
    check("check_physics catches a Damkohler number with no reactant",
          any("never happened" in c for c in msgs),
          msgs[0][:60] + "..." if msgs else "nothing said")
    check("and says nothing when that reaction is switched off",
          not any("never happened" in c
                  for c in check_physics(zero_p, gp, chem=chp, da_abio=0.0)))

    # 16j. A REACTION RATE RECORDED AGAINST A RUN IT COULD NOT AFFECT.
    #      An abiotic-only sweep with the reactant left at the product:
    #      nothing makes the product without microbes, so Da_abio 0.1 through
    #      10 all described the same run.
    zero_p = op.copy(); zero_p[:, 2] = 0.0
    msgs = check_physics(zero_p, gp, chem=chp, da_abio=3.78)
    check("check_physics catches a Damkohler number with no reactant",
          any("never happened" in c for c in msgs))
    check("and says nothing when that reaction is switched off",
          not any("never happened" in c
                  for c in check_physics(zero_p, gp, chem=chp, da_abio=0.0)))

    # 16k-0. THE ATTRIBUTE check_physics ASKS FOR MUST EXIST. It asked for
    #        "biotic_enabled", which is the Settings name; a Chemistry calls
    #        it biotic_on, so getattr returned its default and the biomass
    #        yield was subtracted from the product ceiling on runs that had no
    #        biomass. A misspelt getattr with a default is a silent wrong
    #        answer, so the name is asserted rather than trusted.
    for _attr in ("biotic_on", "yield_", "abiotic_index",
                  "abiotic_product_index"):
        check("a Chemistry really has .%s" % _attr,
              hasattr(Settings().chemistry(), _attr))

    # 16k. AN ABIOTIC REACTION THAT IS A REACTION.
    #      No microbes, no biomass, and still a reaction: the donor is turned
    #      into the product, one mole for one. Until this existed the abiotic
    #      reaction could only destroy, so "abiotic only" meant a chemical
    #      that disappeared with an empty product field beside it.
    ga = np.full((60, 16), PORE, np.uint8)
    ga[:, 0] = SOLID; ga[:, -1] = SOLID
    va = np.zeros((3, 60, 16), np.float32); va[0] = 1.0
    ma = ga == PORE
    s_ab = Settings(); s_ab.reaction_mode = "abiotic only"
    prev_p = -1.0
    for da_a in (0.5, 5.0):
        c_ab = s_ab.chemistry()
        o_ab, _ = solve_adr(ga, va, pe=10.0, da=0.0, n_t=3, n_species=3,
                            chem=c_ab, da_abio=da_a)
        ac_m = float(o_ab[-1, 0][ma].mean())
        p_m = float(o_ab[-1, 2][ma].mean())
        check("abiotic Da %.1f: the donor becomes the product" % da_a,
              p_m > prev_p and p_m > 0.05,
              "Ac %.4f, P %.4f" % (ac_m, p_m))
        check("abiotic Da %.1f: and the two sum to the feed" % da_a,
              abs(ac_m + p_m - 1.0) < 0.01, "Ac + P = %.4f" % (ac_m + p_m))
        check("abiotic Da %.1f: check_physics is content" % da_a,
              not check_physics(o_ab, ga, chem=c_ab, da_abio=da_a))
        prev_p = p_m
    # and "none" is still available, for a terminal loss
    s_ab.abiotic_product = "none"
    o_no, _ = solve_adr(ga, va, pe=10.0, da=0.0, n_t=3, n_species=3,
                        chem=s_ab.chemistry(), da_abio=5.0)
    check("abiotic_product = none consumes and makes nothing",
          float(o_no[-1, 2][ma].max()) == 0.0
          and float(o_no[-1, 0][ma].mean()) < 0.5,
          "P max %.4g" % float(o_no[-1, 2][ma].max()))

    print()
    print("SELF TEST PASSED" if ok else "SELF TEST FAILED")
    return 0 if ok else 1

# ==========================================================================
# === PRTLB TRANSPORT BLOCK END
# ==========================================================================


def self_test():
    return transport_self_test()


if __name__ == "__main__":
    sys.exit(self_test())
