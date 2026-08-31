#!/usr/bin/env python3
"""
make_demo_complab.py
====================

Builds a DEMO campaign that looks exactly like finished CompLaB output, so the
whole downstream chain can be exercised without a cluster: 25 two dimensional
runs and 25 three dimensional runs, each with a geometry file, a flow file,
concentration files for every chemical at every snapshot, a mask file, and a
reaction rate file. Each run also gets the CompLaB.xml it was "launched with",
and each campaign gets the two kinetics headers.

    THE NUMBERS ARE NOT SIMULATED. Nothing here solves Navier-Stokes or a
    transport equation. The geometry is a real random packing with its
    unconnected pockets removed, the speed field is built from the local pore
    size, and the fronts are placed by travel time along the flow, so every
    field has the right shape, the right units and the right file layout. Use
    it to test the collector, the viewer and the training code. Do not use it
    to draw a conclusion about physics.

What it writes
--------------

    <out>/
        demo_2D/
            defineKinetics.hh
            defineAbioticKinetics.hh
            README_2D.txt
            R01_Pe0.5/
                CompLaB.xml            the input file this run was launched with
                inputGeom.vti          the rock,  array "tag"
                nsLattice_0000000.vti  the flow,  arrays "velocityNorm", "velocity"
                maskLattice_0000000.vti
                subsLattice0_0000000.vti ... one per chemical per snapshot
                subsLattice1_*.vti
                subsLattice2_*.vti
                bioLattice0_*.vti      the microbe, same footing as a chemical
                rateLattice_0000000.vti  how fast the reaction ran
            R02_Pe1.0/
            ...  25 folders
        demo_3D/
            ...  the same, with nz > 1

Five rocks are used, each at five Peclet numbers, so 25 runs share 5 distinct
geometries. That is deliberate: it is the case the dataset format exists for,
and it is what lets whole rocks be held out later.

How to collect it
-----------------

In the GUI, action "Collect CompLaB runs":

    the run folders            <out>/demo_2D
    where dataset.h5 goes      <out>/demo_2D_dataset
    everything else            leave blank

and again with demo_3D. Two campaigns cannot go in one file, because one file
holds one grid.

A note on the rate files
------------------------

rateLattice_*.vti is written because CompLaB now writes it, and because it is
worth looking at in ParaView. The collector does not read it yet, so it is
ignored during collection rather than mistaken for a chemical. Nothing breaks.

Requires numpy and scipy, both of which the collector needs anyway.
"""

import argparse
import os
import struct
import sys

import numpy as np

try:
    from scipy import ndimage
except ImportError:                                            # pragma: no cover
    sys.exit("scipy is needed. Install it with:  pip install scipy")


# ============================================================== VTI writing ==
# Plain VTK ImageData with the arrays appended as raw little endian binary,
# each preceded by a UInt32 byte count. That is the format both readers in
# collect_complab_output.py handle, and the one ParaView opens without fuss.
#
# VTK stores a field with x varying fastest. An array held here as (nx, ny, nz)
# therefore has to be written as f.transpose(2, 1, 0).ravel(), and a vector
# held as (3, nx, ny, nz) as v.transpose(3, 2, 1, 0).ravel(), which puts the
# three components together at each point.

def write_vti(path, arrays, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """arrays: list of (name, ndarray). Scalars are (nx,ny,nz), vectors (3,nx,ny,nz)."""
    first = arrays[0][1]
    shape = first.shape if first.ndim == 3 else first.shape[1:]
    nx, ny, nz = shape

    blobs, decls, offset = [], [], 0
    for name, a in arrays:
        if a.ndim == 3:
            flat = np.ascontiguousarray(a.transpose(2, 1, 0), dtype=np.float32).ravel()
            ncomp = 1
        elif a.ndim == 4 and a.shape[0] == 3:
            flat = np.ascontiguousarray(a.transpose(3, 2, 1, 0), dtype=np.float32).ravel()
            ncomp = 3
        else:
            raise ValueError("array %s has shape %s, expected (nx,ny,nz) or "
                             "(3,nx,ny,nz)" % (name, a.shape))
        payload = flat.tobytes()
        blobs.append(struct.pack("<I", len(payload)) + payload)
        decls.append('        <DataArray type="Float32" Name="%s" '
                     'NumberOfComponents="%d" format="appended" offset="%d"/>\n'
                     % (name, ncomp, offset))
        offset += 4 + len(payload)

    ext = "0 %d 0 %d 0 %d" % (nx - 1, ny - 1, nz - 1)
    head = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" '
        'header_type="UInt32">\n'
        '  <ImageData WholeExtent="%s" Origin="%g %g %g" Spacing="%g %g %g">\n'
        '    <Piece Extent="%s">\n'
        '      <PointData Scalars="%s">\n'
        % (ext, origin[0], origin[1], origin[2],
           spacing[0], spacing[1], spacing[2], ext, arrays[0][0])
        + "".join(decls) +
        '      </PointData>\n'
        '    </Piece>\n'
        '  </ImageData>\n'
        '  <AppendedData encoding="raw">\n_'
    )
    tail = '\n  </AppendedData>\n</VTKFile>\n'

    with open(path, "wb") as f:
        f.write(head.encode("ascii"))
        for b in blobs:
            f.write(b)
        f.write(tail.encode("ascii"))


# ================================================================ geometry ===
# Overlapping spheres, or discs in two dimensions, with everything that has no
# face connected path from inlet to outlet removed. That last step matters: an
# isolated pocket carries no flow, and leaving it in gives the distance fields
# something meaningless to describe.
#
# Material codes are written to match <material_numbers> in the input file:
#     pore 2, wall (the shell of solid touching pore) 1, grain interior 0

PORE, WALL, GRAIN = 2, 1, 0


def make_geometry(shape, target_phi, r_lo, r_hi, rng, max_grains=4000):
    """Overlapping grains added until the porosity target is reached.

    A fixed grain count does not travel between grid sizes: the same ninety
    discs that leave a 128 by 64 domain open seal a 64 by 32 one completely.
    Adding grains until a target porosity is hit works at any size.
    """
    nx, ny, nz = shape
    is2d = (nz == 1)
    gx, gy, gz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing="ij")
    solid = np.zeros(shape, bool)
    n = 0
    while n < max_grains:
        for _ in range(4):                       # add a few, then re-measure
            n += 1
            cx = rng.uniform(r_hi, nx - r_hi)
            cy = rng.uniform(0, ny)
            cz = rng.uniform(0, nz) if not is2d else 0.0
            r = rng.uniform(r_lo, r_hi)
            # periodic across y and z so grains wrap rather than leaving a
            # smooth artificial wall down the side of the domain
            dy = np.minimum(np.abs(gy - cy), ny - np.abs(gy - cy))
            dz = (np.minimum(np.abs(gz - cz), nz - np.abs(gz - cz))
                  if not is2d else 0.0)
            solid |= ((gx - cx) ** 2 + dy ** 2 + dz ** 2) < r ** 2
        if 1.0 - solid.mean() <= target_phi:
            break

    pad = max(3, int(round(0.04 * nx)))
    solid[:pad] = False              # inlet plenum
    solid[-pad:] = False             # outlet plenum

    st = ndimage.generate_binary_structure(3, 1)     # face neighbours only
    lab, _ = ndimage.label(~solid, structure=st)
    inlet = set(np.unique(lab[0][~solid[0]]))
    outlet = set(np.unique(lab[-1][~solid[-1]]))
    keep = sorted((inlet & outlet) - {0})
    if not keep:
        return None                                   # nothing percolates
    pore = np.isin(lab, keep)

    mat = np.full(shape, GRAIN, np.int32)
    mat[pore] = PORE
    shell = ndimage.binary_dilation(pore, structure=st) & ~pore
    mat[shell] = WALL
    return mat


def flow_and_traveltime(mat, u_mean, dx):
    """A speed field and a travel time, built from the pore size.

    Wide channels carry most of the flow, so the distance to the nearest grain
    is a serviceable stand in for speed. The travel time is that speed
    integrated along x and then smoothed across the flow, which is what makes a
    front finger instead of arriving as a flat plane.
    """
    pore = (mat == PORE)
    edt = ndimage.distance_transform_edt(pore).astype(np.float32)
    s = ndimage.gaussian_filter(edt, 1.5)
    s = np.where(pore, s, 0.0)
    s = s / max(s[pore].mean(), 1e-30)
    speed = np.where(pore, u_mean * s ** 1.6, 0.0).astype(np.float32)

    inv = np.where(pore, 1.0 / np.maximum(speed, u_mean * 0.05), 0.0)
    tau = np.cumsum(inv, axis=0) * dx
    sig = (0.0, 2.0, 0.0) if mat.shape[2] == 1 else (0.0, 1.5, 1.5)
    tau = ndimage.gaussian_filter(tau.astype(np.float32), sig)
    out_mean = tau[-1][pore[-1]].mean() if pore[-1].any() else tau[pore].max()
    tau = tau / max(out_mean, 1e-30)                  # 1.0 at the outlet
    return speed, tau.astype(np.float32), pore


def velocity_vector(speed, mat):
    """A divergence free field is not needed for a demo, but a plausible one is.

    The component along the flow carries the speed. The transverse components
    are a small deflection towards the middle of the local channel, which is
    what makes streamlines in ParaView look like they go round the grains
    rather than through them.
    """
    pore = (mat == PORE)
    edt = ndimage.distance_transform_edt(pore).astype(np.float32)
    sm = ndimage.gaussian_filter(edt, 2.0)
    # np.gradient needs at least three cells along an axis, and a 2D case has
    # exactly one along z, so each axis is differenced only where it can be
    gy = np.gradient(sm, axis=1) if mat.shape[1] > 2 else np.zeros_like(sm)
    gz = np.gradient(sm, axis=2) if mat.shape[2] > 2 else np.zeros_like(sm)
    scale = 0.25 * speed / max(speed.max(), 1e-30)
    v = np.zeros((3,) + mat.shape, np.float32)
    v[0] = speed
    v[1] = np.where(pore, scale * gy, 0.0)
    v[2] = np.where(pore, scale * gz, 0.0)
    return v


# ================================================================ chemistry ==
# Ac + A -> P, driven by a microbe Bio that grows on the pair.
#
#     R = Vmax * Bio * Ac/(Ks_ac+Ac) * A/(Ks_a+A)
#
# Ac and A are both injected at the inlet. Bio starts as a thin film on the
# grain surfaces, which is where attached biomass lives, and grows where both
# substrates reach it. P is the product and washes downstream.
#
# Every field is kept inside 0 to 1 so that the collector's default bound of
# --max-conc 1.0 is not tripped.

def fields_at(tau, pore, wallness, t, cond):
    front = 0.5 * (1.0 - np.tanh((tau - t) / cond["width"]))     # 1 behind, 0 ahead
    age = np.clip(t - tau, 0.0, None)

    Ac = cond["ac0"] * front * np.exp(-cond["burn_ac"] * age)
    A = cond["a0"] * front * np.exp(-cond["burn_a"] * age)
    Bio = cond["b0"] + cond["grow"] * wallness * front * (1.0 - np.exp(-3.0 * age))
    P = cond["ymax"] * (1.0 - np.exp(-cond["burn_ac"] * age)) * front \
        * np.exp(-0.25 * age)

    R = (cond["vmax"] * Bio
         * Ac / (cond["ks_ac"] + Ac)
         * A / (cond["ks_a"] + A))

    out = []
    for f in (Ac, A, P, Bio, R):
        g = np.where(pore, f, 0.0).astype(np.float32)
        np.clip(g, 0.0, 1.0, out=g)
        out.append(g)
    return out                                     # Ac, A, P, Bio, R


# ================================================================== the XML ==
# The dialect the collector reads: <LB_numerics><domain><material_numbers>,
# <LB_numerics><Peclet>, <chemistry><substrateN><name_of_substrates>,
# <microbiology><microbeN><name_of_microbes>, <IO><subs_filename>. Those are
# the paths collect_foreign_complab.py matches on, so a demo that uses other
# spellings would collect with the conditions missing.

XML = """<?xml version="1.0" ?>
<!-- ==========================================================================
     DEMO input file, written by make_demo_complab.py.

     It is the real CompLaB campaign dialect and it is what the collector
     reads, but the results beside it were CONSTRUCTED, not simulated.

         Ac + A -> P     driven by the microbe Bio
         R = Vmax * Bio * Ac/(Ks_ac+Ac) * A/(Ks_a+A)

     ========================================================================== -->
<parameters>

    <path>
        <src_path>src</src_path>
        <input_path>input</input_path>
        <output_path>output</output_path>
    </path>

    <simulation_mode>
        <biotic_mode>true</biotic_mode>
        <enable_kinetics>true</enable_kinetics>
        <enable_abiotic_kinetics>false</enable_abiotic_kinetics>
        <enable_validation_diagnostics>false</enable_validation_diagnostics>
    </simulation_mode>

    <LB_numerics>
        <domain>
            <nx>{nx}</nx>                       <!-- flow runs along +x -->
            <ny>{ny}</ny>
            <nz>{nz}</nz>
            <dx>{dx_um}</dx>
            <unit>um</unit>
            <characteristic_length>{char_len}</characteristic_length>
            <filename>geometry.dat</filename>
            <material_numbers>
                <pore>2</pore>                  <!-- read by the collector -->
                <solid>0</solid>
                <bounce_back>1</bounce_back>
            </material_numbers>
        </domain>
        <delta_P>{delta_p:.6g}</delta_P>
        <Peclet>{pe:.6g}</Peclet>               <!-- the condition of this run -->
        <tau>0.8</tau>
        <track_performance>false</track_performance>
        <iteration>
            <ns_max_iT1>200000</ns_max_iT1>
            <ns_max_iT2>50000</ns_max_iT2>
            <ns_converge_iT1>1e-8</ns_converge_iT1>
            <ns_converge_iT2>1e-6</ns_converge_iT2>
            <ade_max_iT>{ade_max}</ade_max_iT>
            <ade_converge_iT>1e-9</ade_converge_iT>
            <ns_update_interval>1</ns_update_interval>
            <ade_update_interval>1</ade_update_interval>
        </iteration>
    </LB_numerics>

    <chemistry>
        <number_of_substrates>3</number_of_substrates>

        <substrate0>
            <name_of_substrates>Ac</name_of_substrates>          <!-- electron donor -->
            <initial_concentration>0.0</initial_concentration>
            <substrate_diffusion_coefficients><in_pore>1.0e-9</in_pore><in_biofilm>5.0e-10</in_biofilm></substrate_diffusion_coefficients>
            <left_boundary_type>Dirichlet</left_boundary_type>
            <right_boundary_type>Neumann</right_boundary_type>
            <left_boundary_condition>{ac0:.6g}</left_boundary_condition>
            <right_boundary_condition>0.0</right_boundary_condition>
        </substrate0>

        <substrate1>
            <name_of_substrates>A</name_of_substrates>            <!-- electron acceptor -->
            <initial_concentration>0.0</initial_concentration>
            <substrate_diffusion_coefficients><in_pore>1.2e-9</in_pore><in_biofilm>6.0e-10</in_biofilm></substrate_diffusion_coefficients>
            <left_boundary_type>Dirichlet</left_boundary_type>
            <right_boundary_type>Neumann</right_boundary_type>
            <left_boundary_condition>{a0:.6g}</left_boundary_condition>
            <right_boundary_condition>0.0</right_boundary_condition>
        </substrate1>

        <substrate2>
            <name_of_substrates>P</name_of_substrates>            <!-- product -->
            <initial_concentration>0.0</initial_concentration>
            <substrate_diffusion_coefficients><in_pore>1.0e-9</in_pore><in_biofilm>5.0e-10</in_biofilm></substrate_diffusion_coefficients>
            <left_boundary_type>Neumann</left_boundary_type>
            <right_boundary_type>Neumann</right_boundary_type>
            <left_boundary_condition>0.0</left_boundary_condition>
            <right_boundary_condition>0.0</right_boundary_condition>
        </substrate2>
    </chemistry>

    <microbiology>
        <number_of_microbes>1</number_of_microbes>
        <maximum_biomass_density>1.0</maximum_biomass_density>
        <thrd_biofilm_fraction>0.1</thrd_biofilm_fraction>
        <CA_method>half</CA_method>

        <microbe0>
            <name_of_microbes>Bio</name_of_microbes>
            <solver_type>CA</solver_type>
            <reaction_type>kinetics</reaction_type>
            <initial_densities>{b0:.6g}</initial_densities>
            <decay_coefficient>1.0e-7</decay_coefficient>
            <biomass_diffusion_coefficients>
                <in_pore>0.0</in_pore>
                <in_biofilm>0.0</in_biofilm>
            </biomass_diffusion_coefficients>
            <half_saturation_constants>{ks_ac:.6g} {ks_a:.6g} 0.0</half_saturation_constants>
            <maximum_uptake_flux>{vmax:.6g} {vmax:.6g} 0.0</maximum_uptake_flux>
            <left_boundary_type>Neumann</left_boundary_type>
            <right_boundary_type>Neumann</right_boundary_type>
            <left_boundary_condition>0.0</left_boundary_condition>
            <right_boundary_condition>0.0</right_boundary_condition>
        </microbe0>
    </microbiology>

    <equilibrium>
        <enabled>false</enabled>
        <components>none</components>
        <stoichiometry><species0>0.0</species0></stoichiometry>
        <logK><species0>0.0</species0></logK>
    </equilibrium>

    <IO>
        <read_NS_file>false</read_NS_file>
        <ns_rerun_iT0>0</ns_rerun_iT0>
        <read_ADE_file>false</read_ADE_file>
        <ns_filename>nsLattice</ns_filename>
        <mask_filename>maskLattice</mask_filename>
        <subs_filename>subsLattice</subs_filename>       <!-- gives subsLattice0_*.vti -->
        <bio_filename>bioLattice</bio_filename>
        <save_VTK_interval>{vtk_every}</save_VTK_interval>
        <save_CHK_interval>0</save_CHK_interval>

        <save_reaction_rates>true</save_reaction_rates>
        <rate_filename>rateLattice</rate_filename>
        <rate_interval>0</rate_interval>
        <rate_unit>mol/L/s</rate_unit>
    </IO>

</parameters>
"""


KINETICS_HH = '''/* defineKinetics.hh  --  DEMO, written by make_demo_complab.py
 *
 * One microbial reaction:
 *
 *     Ac + A -> P     R = Vmax * Bio * Ac/(Ks_ac+Ac) * A/(Ks_a+A)
 *
 * Chemical order, and it must match CompLaB.xml:
 *     C[0] = Ac    C[1] = A    C[2] = P
 * Microbe order:
 *     B[0] = Bio
 *
 * The constants below are what the collector reads into inputs/kinetics. They
 * are the constants a run of this demo would have been built with.
 */
#ifndef DEFINE_KINETICS_HH
#define DEFINE_KINETICS_HH

#include <vector>
#include <string>
#include <cstddef>
#include <algorithm>

namespace KineticParams {
    constexpr int AC = 0, A = 1, P = 2;          // chemicals
    constexpr int BIO = 0;                       // microbes

    const double Vmax        = 5.0e-5;   // mol/gDW/s   maximum uptake flux
    const double Ks_ac       = 1.0e-4;   // mol/L       half saturation, donor
    const double Ks_a        = 5.0e-5;   // mol/L       half saturation, acceptor
    const double Y           = 0.35;     // gDW/mol     yield on the donor
    const double k_decay     = 1.0e-7;   // 1/s         first order decay
    const double dt_kinetics = 1.25e-2;  // s           one solver step
    const double MAX_RATE_FRACTION = 0.5;// -           per step stability limit
}

namespace KineticsStats {
    static const double MIN_BIOMASS = 1e-12;
    static double iter_sum_dB = 0.0, iter_max_biomass = 0.0, iter_max_dB = 0.0,
                  iter_min_DOC = 1e30;
    static long iter_cells_with_biomass = 0, iter_cells_with_growth = 0;
    inline void resetIteration() {
        iter_sum_dB = 0; iter_max_biomass = 0; iter_max_dB = 0; iter_min_DOC = 1e30;
        iter_cells_with_biomass = 0; iter_cells_with_growth = 0;
    }
    inline void accumulate(double biomass, double donor, double dB) {
        if (biomass > MIN_BIOMASS) {
            iter_cells_with_biomass++; iter_sum_dB += dB;
            if (biomass > iter_max_biomass) iter_max_biomass = biomass;
            if (dB > iter_max_dB) iter_max_dB = dB;
            if (donor < iter_min_DOC && donor > 0) iter_min_DOC = donor;
            if (dB > 0) iter_cells_with_growth++;
        }
    }
    inline void getStats(long& cb, long& cg, double& s, double& mB,
                         double& mdB, double& mD) {
        cb = iter_cells_with_biomass; cg = iter_cells_with_growth;
        s = iter_sum_dB; mB = iter_max_biomass; mdB = iter_max_dB;
        mD = (iter_min_DOC < 1e20) ? iter_min_DOC : 0.0;
    }
}

/* Per reaction rate output, the opt in read by complab3d_rates.hh. */
#define COMPLAB_HAS_RXN_RATES 1
namespace RxnRates {
    inline const std::vector<std::string>& names() {
        static const std::vector<std::string> n = { "Bio_growth_on_Ac_and_A" };
        return n;
    }
}

void defineRxnKinetics(std::vector<double> B, std::vector<double> C,
                       std::vector<double>& subsR, std::vector<double>& bioR,
                       plb::plint mask, std::vector<double>* rxnR = 0)
{
    using namespace KineticParams;
    for (std::size_t i = 0; i < subsR.size(); ++i) subsR[i] = 0.0;
    for (std::size_t i = 0; i < bioR.size();  ++i) bioR[i]  = 0.0;
    if (rxnR) for (std::size_t i = 0; i < rxnR->size(); ++i) (*rxnR)[i] = 0.0;

    if (B.size() < 1 || C.size() < 3 || subsR.size() < 3) return;
    if (mask < 2) return;

    const double bio = std::max(B[BIO], 0.0);
    const double ac  = std::max(C[AC],  0.0);
    const double a   = std::max(C[A],   0.0);

    double R = Vmax * bio * (ac / (Ks_ac + ac)) * (a / (Ks_a + a));

    const double cap = std::min(ac, a) * MAX_RATE_FRACTION / dt_kinetics;
    if (R > cap) R = cap;

    subsR[AC] = -R;
    subsR[A]  = -R;
    subsR[P]  = +R;
    bioR[BIO] = Y * R - k_decay * bio;

    if (!rxnR) KineticsStats::accumulate(bio, ac, bioR[BIO]);
    if (rxnR && rxnR->size() >= 1) (*rxnR)[0] = R;
}

#endif
'''

ABIOTIC_HH = '''/* defineAbioticKinetics.hh  --  DEMO, written by make_demo_complab.py
 *
 * This demo has no chemical reaction without a microbe, so every rate here is
 * zero and <enable_abiotic_kinetics> is false in CompLaB.xml. The file exists
 * because the solver includes it unconditionally, and because the collector
 * records that it was found rather than leaving a hole in inputs/files.
 */
#ifndef DEFINE_ABIOTIC_KINETICS_HH
#define DEFINE_ABIOTIC_KINETICS_HH

#include <vector>
#include <cstddef>

void defineAbioticRxnKinetics(std::vector<double> C, std::vector<double>& subsR,
                              plb::plint mask, std::vector<double>* rxnR = 0)
{
    (void) C; (void) mask;
    for (std::size_t i = 0; i < subsR.size(); ++i) subsR[i] = 0.0;
    if (rxnR) for (std::size_t i = 0; i < rxnR->size(); ++i) (*rxnR)[i] = 0.0;
}

#endif
'''


# =================================================================== driver ==

def build_campaign(root, tag, shape, n_rocks, pe_values, n_snap, dx_um,
                   target_phi, r_lo, r_hi, seed, quiet=False):
    os.makedirs(root, exist_ok=True)
    nx, ny, nz = shape
    dx = dx_um * 1e-6                       # m

    open(os.path.join(root, "defineKinetics.hh"), "w").write(KINETICS_HH)
    open(os.path.join(root, "defineAbioticKinetics.hh"), "w").write(ABIOTIC_HH)

    # the rocks, made once and shared by every Peclet number
    rocks = []
    rng = np.random.default_rng(seed)
    tries = 0
    while len(rocks) < n_rocks and tries < 40 * n_rocks:
        tries += 1
        m = make_geometry(shape, target_phi, r_lo, r_hi, rng)
        if m is not None and (m == PORE).mean() > 0.15:
            rocks.append(m)
    if len(rocks) < n_rocks:
        sys.exit("could not build %d percolating rocks at %s. Raise the\n"
                 "porosity target with --porosity." % (n_rocks, shape))

    run_no = 0
    for ri, mat in enumerate(rocks):
        pore = (mat == PORE)
        # how much of a voxel's neighbourhood is grain: where attached biomass sits
        wallness = ndimage.gaussian_filter((mat != PORE).astype(np.float32), 1.5)
        wallness = np.where(pore, np.clip(wallness * 2.0, 0.0, 1.0), 0.0)

        for pe in pe_values:
            run_no += 1
            name = "R%02d_rock%d_Pe%g" % (run_no, ri + 1, pe)
            rd = os.path.join(root, name)
            os.makedirs(rd, exist_ok=True)

            # a velocity that scales with the Peclet number of this run
            D0 = 1.0e-9
            char_len = ny * dx
            u_mean = pe * D0 / char_len
            speed, tau, _ = flow_and_traveltime(mat, u_mean, dx)
            vel = velocity_vector(speed, mat)
            t_adv = (nx * dx) / max(u_mean, 1e-30)

            cond = dict(
                width=0.10 + 0.05 / max(pe, 0.2),     # sharper front at higher Pe
                ac0=1.0, a0=0.85, b0=0.05,
                grow=0.55, ymax=0.60,
                burn_ac=0.9, burn_a=0.7,
                vmax=0.9, ks_ac=0.15, ks_a=0.10,
            )

            # Every run keeps the same number of snapshots, at the same
            # iterations. Real runs stop at their own convergence iteration and
            # the collector trims to the shortest, but a demo that lost
            # snapshots for no reason would only be confusing.
            every = 2000
            iters = [every * k for k in range(n_snap)]
            last = iters[-1]

            write_vti(os.path.join(rd, "inputGeom.vti"),
                      [("tag", mat.astype(np.float32))],
                      spacing=(dx_um, dx_um, dx_um))
            write_vti(os.path.join(rd, "maskLattice_%07d.vti" % iters[-1]),
                      [("Density", mat.astype(np.float32))],
                      spacing=(dx_um, dx_um, dx_um))
            write_vti(os.path.join(rd, "nsLattice_%07d.vti" % iters[-1]),
                      [("velocityNorm", speed), ("velocity", vel)],
                      spacing=(dx_um, dx_um, dx_um))

            for k, it in enumerate(iters):
                t = (k + 1) / float(n_snap) * 1.3       # up to 1.3 pore volumes
                Ac, A, P, Bio, R = fields_at(tau, pore, wallness, t, cond)
                for i, f in enumerate((Ac, A, P)):
                    write_vti(os.path.join(rd, "subsLattice%d_%07d.vti" % (i, it)),
                              [("Density", f)], spacing=(dx_um, dx_um, dx_um))
                write_vti(os.path.join(rd, "bioLattice0_%07d.vti" % it),
                          [("Density", Bio)], spacing=(dx_um, dx_um, dx_um))
                write_vti(os.path.join(rd, "rateLattice_%07d.vti" % it),
                          [("R_Bio_growth_on_Ac_and_A", R)],
                          spacing=(dx_um, dx_um, dx_um))

            open(os.path.join(rd, "CompLaB.xml"), "w").write(XML.format(
                nx=nx, ny=ny, nz=nz, dx_um=dx_um, char_len="%.6g" % (ny * dx_um),
                delta_p=1.0e-4 * pe, pe=pe, ade_max=last,
                ac0=cond["ac0"], a0=cond["a0"], b0=cond["b0"],
                ks_ac=cond["ks_ac"], ks_a=cond["ks_a"], vmax=cond["vmax"],
                vtk_every=every))

            if not quiet:
                print("  %-24s rock %d  Pe %-5g  porosity %.2f  %d snapshots"
                      % (name, ri + 1, pe, pore.mean(), len(iters)))

    readme = os.path.join(root, "README_%s.txt" % tag)
    open(readme, "w").write(
        "DEMO CompLaB output, %s, written by make_demo_complab.py\n"
        "%s\n\n"
        "%d runs on %d rocks, %d Peclet numbers each. Grid %d x %d x %d, "
        "voxel %g um.\n\n"
        "THE NUMBERS ARE NOT SIMULATED. The geometry is a real percolating\n"
        "packing and the file layout is exactly what CompLaB writes, but the\n"
        "fields were constructed from travel time, not solved.\n\n"
        "To collect, in the GUI, action \"Collect CompLaB runs\":\n"
        "    the run folders         this folder\n"
        "    where dataset.h5 goes   a new folder\n"
        "    everything else         leave blank\n\n"
        "Each run folder carries its own CompLaB.xml, and the two kinetics\n"
        "headers sit here beside the runs, so the collector finds all three\n"
        "without being told where they are.\n\n"
        "rateLattice_*.vti holds the reaction rate. The collector ignores it\n"
        "for now; open it in ParaView.\n\n"
        "About the conditions: the collector reads the Peclet number of each\n"
        "run out of that run's own CompLaB.xml, which is why it differs run to\n"
        "run here. The other five (da_bio, da_abio, ks_ac_norm, ks_a_norm,\n"
        "y_norm) are not in the input file anywhere the collector looks, so it\n"
        "reports them as NOT FOUND and stores zero. Type them on the GUI page\n"
        "if you want them filled; they then apply to the whole campaign.\n"
        % (tag, "=" * 60, run_no, n_rocks, len(pe_values), nx, ny, nz, dx_um))
    return run_no


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="./demo_complab",
                   help="folder to write demo_2D and demo_3D into")
    p.add_argument("--rocks", type=int, default=5,
                   help="distinct geometries per campaign (default 5)")
    p.add_argument("--pe", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0, 5.0, 10.0],
                   help="Peclet numbers, one run per rock per value")
    p.add_argument("--snapshots", type=int, default=8)
    p.add_argument("--nx2d", type=int, default=128)
    p.add_argument("--ny2d", type=int, default=64)
    p.add_argument("--n3d", type=int, default=32,
                   help="the 3D grid is this cubed")
    p.add_argument("--dx-um", type=float, default=10.0, dest="dx_um")
    p.add_argument("--porosity", type=float, default=0.45,
                   help="grains are added until the porosity drops to this")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--only", choices=["2d", "3d"], default=None,
                   help="build just one of the two campaigns")
    a = p.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    print("writing into %s" % out)
    print("%d rocks x %d Peclet numbers = %d runs per campaign, %d snapshots each\n"
          % (a.rocks, len(a.pe), a.rocks * len(a.pe), a.snapshots))

    if a.only != "3d":
        print("2D campaign, grid %d x %d x 1" % (a.nx2d, a.ny2d))
        build_campaign(os.path.join(out, "demo_2D"), "2D",
                       (a.nx2d, a.ny2d, 1), a.rocks, a.pe, a.snapshots,
                       a.dx_um, target_phi=a.porosity,
                       r_lo=max(2.0, 0.05 * a.ny2d), r_hi=max(4.0, 0.12 * a.ny2d),
                       seed=a.seed)
        print()

    if a.only != "2d":
        n = a.n3d
        print("3D campaign, grid %d x %d x %d" % (n, n, n))
        build_campaign(os.path.join(out, "demo_3D"), "3D",
                       (n, n, n), a.rocks, a.pe, a.snapshots,
                       a.dx_um, target_phi=a.porosity,
                       r_lo=max(2.0, 0.06 * n), r_hi=max(3.5, 0.14 * n),
                       seed=a.seed + 1000)
        print()

    print("done.")
    print("Collect each campaign separately: one dataset file holds one grid.")


if __name__ == "__main__":
    main()
