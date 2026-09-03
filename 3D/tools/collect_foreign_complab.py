#!/usr/bin/env python3
# =============================================================================
# CHANGED FROM THE 2D VERSION
#
#   WHAT CHANGED HERE, IN ONE LINE
#     While collecting a campaign this file now also writes the three flow
#     descriptors the flow pipeline wants: geom/mis, geom/uprm and geom/dw2.
#
#   WHERE THE DESCRIPTORS CAME FROM
#     github.com/hjunglab/PRT-DeepONet   branch/folder: velocity-informed
#     ported into 3D/tools/flow_features.py, which is what is called below.
#     MIS is how wide the pore is at each voxel. UPRM is how wide the NARROWEST
#     THROAT between the inlet and that voxel is. dw2 is the squared wall
#     distance, scaled to [0, 1].
#
#   WHY THEY ARE COMPUTED HERE RATHER THAN AT TRAINING TIME
#     They depend on the GEOMETRY ALONE, not on the run conditions. A campaign
#     of 500 runs over 20 rocks has 20 answers, not 500, and computing them
#     once at collection costs about a second a rock. Doing it in the training
#     loop would redo the same work on every epoch of every run.
#
#   THE TWO NEW FLAGS
#     --no-flow-features   skip them entirely. The dataset stays perfectly
#                          usable; add_flow_features.py can put them in later
#                          without recollecting anything.
#     --flow-buffer N      how many OPEN voxels pad each end of the flow axis.
#                          This must match your campaign: 10 for the published
#                          2D set, 5 for the geometries this project generates,
#                          0 for none. Get it wrong and the MIS treatment
#                          measures your padding instead of your rock.
#
#   IF THE DESCRIPTOR STEP FAILS
#     It is caught, reported by name, and collection continues. A missing
#     descriptor is RECORDED, never invented, which is the same rule the rest
#     of this collector already follows for every other absent field.
# =============================================================================
"""
collect_foreign_complab.py — CompLaB output that was NOT set up by
complab_campaign.py, collected into the one dataset.h5 everything else reads.

WHY THIS EXISTS, AND WHEN NOT TO USE IT
---------------------------------------
collect_complab_output.py is the normal route. It expects the layout our own
campaign builder writes: runs/run_000000/{params.json, status.json, output/}.
Those two small json files carry the conditions, and without them a .vti file
is anonymous: it holds an array of numbers on a grid and nothing else. No
Peclet, no Damkohler, no note of which rock it was computed on.

If you ran CompLaB by hand, those files do not exist. This script is that case.
It reads the conditions from somewhere you point it at, works out the pore
structure and its two distance fields from the geometry the run carries, and
writes dataset.h5 directly. It never fabricates a condition it cannot find.

THE TRAP THIS SCRIPT EXISTS TO CATCH
------------------------------------
CompLaB's material codes are set per run in CompLaB.xml, so a geometry file
written by one setup can mean the OPPOSITE of one written by another. Our
pipeline uses 0 solid, 1 interface, 2 pore. A file written with 0 = pore, fed
in unswapped, trains the network on the rock instead of the pore space, and
nothing anywhere complains. So the pore code is either given explicitly or
worked out by comparing each code's voxel fraction against the porosity you
supply, and the decision is always printed.

    python collect_foreign_complab.py --inspect --runs ./my_output
    python collect_foreign_complab.py --runs ./my_output --out ./dataset
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collect_complab_output import read_vti          # noqa: E402

SOLID, WALL, PORE = 0, 1, 2
PNAMES = ["pe", "da_bio", "da_abio", "ks_ac_norm", "ks_a_norm", "y_norm"]


# ----------------------------------------------------------------- geometry --
def geodesic(g):
    """Distance from the inlet THROUGH pore space, in voxels.

    Neighbour relaxation, the same scheme the rest of the project uses. Solid
    voxels stay NaN throughout, so the distance can never take a short cut
    through rock, which is the whole point of this field.
    """
    inf = np.float32(1e9)
    d = np.where(g == PORE, inf, np.nan).astype(np.float32)
    d[0][g[0] == PORE] = 0.0
    for _ in range(4 * g.shape[0]):
        prev = d.copy()
        for ax in range(3):
            for sh in (1, -1):
                nb = np.roll(d, sh, axis=ax)
                sl = [slice(None)] * 3
                sl[ax] = 0 if sh > 0 else -1
                nb[tuple(sl)] = inf
                d = np.where(g == PORE,
                             np.fmin(d, np.nan_to_num(nb, nan=inf) + 1.0),
                             np.nan)
        if np.nanmax(np.abs(np.nan_to_num(d - prev, nan=0.0))) < 1e-6:
            break
    return np.nan_to_num(d, nan=0.0).astype(np.float32)


def decide_pore_code(tag, porosity, forced):
    """Which value in the geometry file means pore.

    Returns (code, how_it_was_decided). If the caller forced one, that wins and
    is still checked against the porosity so a wrong answer is visible.
    """
    vals = sorted(int(v) for v in np.unique(tag))
    frac = {v: float((tag == v).mean()) for v in vals}
    if forced is not None:
        if forced not in frac:
            raise ValueError("--pore-code %d is not in the file, which holds %s"
                             % (forced, vals))
        return forced, "given on the command line"
    if porosity is None:
        raise ValueError(
            "cannot tell which code means pore. The geometry holds %s, "
            "covering %s of the box. Fix it in either of two ways: give the "
            "run a CompLaB.xml with <material_numbers><pore>, which is where "
            "the solver itself read it from, or pass --pore-code. Getting this "
            "wrong trains the network on the rock instead of the pore space "
            "and nothing downstream complains, which is why it is not guessed."
            % (vals, {k: round(v, 4) for k, v in frac.items()}))
    best = min(vals, key=lambda v: abs(frac[v] - porosity))
    if abs(frac[best] - porosity) > 0.05:
        raise ValueError(
            "no code matches the porosity %.4f. Fractions are %s. Give "
            "--pore-code explicitly." % (porosity,
                                         {k: round(v, 4) for k, v in frac.items()}))
    return best, "matched to porosity %.4f (this code covers %.4f)" % (
        porosity, frac[best])


def to_our_codes(tag, pore_code):
    """Their codes to ours: pore becomes 2, whatever touches pore becomes 1,
    everything else becomes 0. The shell is found by dilation rather than
    assumed, because which of their remaining codes is the bounce-back layer
    differs between setups."""
    from scipy import ndimage
    pore = (tag == pore_code)
    touching = ndimage.binary_dilation(pore) & (~pore)
    return np.where(pore, PORE, np.where(touching, WALL, SOLID)).astype(np.uint8)


# --------------------------------------------------------------- conditions --
def read_csv_row(run_dir, pattern):
    hits = sorted(glob.glob(os.path.join(run_dir, pattern)))
    if not hits:
        return None
    with open(hits[0]) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def from_xml_pairs(pairs, suffix):
    """One setting out of a flattened input file, as a number, or None."""
    for p, v in pairs:
        if p.endswith(suffix):
            try:
                return float(str(v).split()[0])
            except (ValueError, IndexError):
                return None
    return None


def conditions_for(run_dir, args, xml_pairs=()):
    """The conditions of one run, and where each came from.

    The INPUT FILE is the source. It is the thing the solver actually read, so
    a value taken from it is a record rather than an inference. Anything not in
    it is left as None and nothing is guessed.

    A run summary csv is read ONLY when one is asked for by name. It is off by
    default: those files are written by whatever script happened to post-process
    the run, they differ between campaigns, and a number lifted out of one and
    stored beside the data is indistinguishable afterwards from a number the
    solver was actually given.
    """
    got = {k: None for k in PNAMES}
    src = {k: "not found" for k in PNAMES}
    row = None

    pe = from_xml_pairs(xml_pairs, "/LB_numerics/Peclet")
    if pe is not None:
        got["pe"], src["pe"] = pe, "CompLaB.xml, <Peclet>"

    if args.conditions_csv:
        row = read_csv_row(run_dir, args.conditions_csv)
        if row is not None and args.pe_column in row and got["pe"] is None:
            got["pe"] = float(row[args.pe_column])
            src["pe"] = "%s, column %s" % (args.conditions_csv, args.pe_column)

    # Typed in on the page. These override, because you typed them knowing
    # what the run was.
    typed = dict(pe=args.pe, da_bio=args.da_bio, da_abio=args.da_abio,
                 ks_ac_norm=args.ks_ac, ks_a_norm=args.ks_a, y_norm=args.y)
    for k, v in typed.items():
        if v is not None:
            got[k], src[k] = float(v), "given on the page"

    porosity = None
    if row is not None and args.porosity_column in row:
        porosity = float(row[args.porosity_column])
    return got, src, porosity, row


def recorded_dt(row, column):
    """The reaction time step the run itself wrote down, if it wrote one.

    This is worth more than any formula. The relation between tau, dx and the
    diffusivity depends on the lattice the build was compiled for: a D3Q7
    scheme divides by four where a D3Q19 one divides by three, and a build that
    tunes its own relaxation time to hit a target Peclet does not use the one
    written in the input file at all. Computing the step from the input file
    and calling the answer "what the run used" is wrong on any build but the
    one the formula was written for.
    """
    if not row or not column:
        return None
    if column not in row:
        return None
    try:
        v = float(row[column])
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


# -------------------------------------------------------------------- files --
def species_files(run_dir, spec):
    """[(iteration, path)] for one chemical, in iteration order.

    The number at the end of the filename is the ITERATION, and it is kept.
    Runs stop at their own convergence iteration, so ordering alone would put
    different physical times in the same slot.
    """
    hits = []
    for p in glob.glob(os.path.join(run_dir, spec)):
        m = re.search(r"_(\d+)\.vti$", os.path.basename(p))
        if m:
            hits.append((int(m.group(1)), p))
    return sorted(hits)


def rate_channels(path, shape):
    """[(name, array)] for every channel in one rate snapshot, in file order.

    The names are whatever the solver wrote, and the solver got them from the
    user's own kinetics header. Nothing here knows what a reaction is called,
    which is the point: change the chemistry and the channels follow without a
    line changing anywhere but that header.

    Only arrays that match the grid are kept, so a vector quantity that happens
    to be in the same file is passed over rather than reshaped into nonsense.
    """
    arrs, _ = read_vti(path)
    return [(k, v) for k, v in arrs.items() if v.shape == shape]


def one_array(path):
    arrs, _ = read_vti(path)
    if not arrs:
        raise ValueError("no readable array in %s" % os.path.basename(path))
    for key in ("Density", "density", "concentration", "C"):
        if key in arrs:
            return key, arrs[key]
    k = list(arrs.keys())[0]
    return k, arrs[k]


# ------------------------------------------------------- what each setting is --
# A path and a value is a record. A path, a value, a plain-language name, what it
# is for and its unit is something a person can read a year later without the
# manual open beside them. The names below come from the 2D and 3D sources, and
# cover every setting either solver reads.
#
# The key is the tail of the path, so one entry serves both versions and both
# spellings of a nested block.
SETTING_INFO = {
    # ---- where things live
    "path/src_path":        ("source folder", "", "where the solver looks for its own code"),
    "path/input_path":      ("input folder", "", "where the geometry and any metabolic models are read from"),
    "path/output_path":     ("output folder", "", "where results are written"),
    # ---- master switches, 3D only
    "simulation_mode/biotic_mode":        ("microbes at all", "", "false skips the whole microbiology section"),
    "simulation_mode/enable_kinetics":    ("hand written rate laws on", "", "the biological reactions in defineKinetics.hh"),
    "simulation_mode/enable_abiotic_kinetics": ("chemical rate laws on", "", "reactions with no microbe, in defineAbioticKinetics.hh. Also what gives precipitation something to accumulate"),
    "simulation_mode/enable_fba_glpk":    ("flux balance by GLPK", "", "a metabolic model decides growth instead of a rate law"),
    "simulation_mode/enable_fba_cobrapy": ("flux balance by COBRApy", "", "the same, solved in Python. Cannot be mixed with GLPK"),
    "simulation_mode/enable_surrogate":   ("surrogate metabolism on", "", "a pre-trained stand-in for flux balance"),
    "simulation_mode/fba_concentration_basis": ("uptake sees", "", "free ion, or the total including every complex carrying it"),
    "simulation_mode/glpk_lpsolver":      ("GLPK algorithm", "", "1 simplex, 2 interior point, 3 exact"),
    "simulation_mode/glpk_save_problem":  ("dump the linear program", "", "debugging only"),
    "simulation_mode/enable_validation_diagnostics": ("per step mass balance printing", "", "verbose, worth having on for new chemistry"),
    # ---- the grid
    "domain/nx":   ("voxels along x", "voxels", "flow runs along +x"),
    "domain/ny":   ("voxels along y", "voxels", ""),
    "domain/nz":   ("voxels along z", "voxels", "3D only"),
    "domain/dx":   ("voxel edge along x", "unit below", "the physical size of one voxel"),
    "domain/dy":   ("voxel edge along y", "unit below", "defaults to dx"),
    "domain/dz":   ("voxel edge along z", "unit below", "defaults to dx"),
    "domain/unit": ("length unit", "", "m, mm or um"),
    "domain/characteristic_length": ("pore size that sets the Peclet number", "unit above",
                                     "the open width of a duct, or a typical pore throat"),
    "domain/filename": ("geometry file", "", "the material map, one integer per voxel"),
    "material_numbers/pore":        ("which code means pore", "code", "open fluid. Reading this wrong trains the network on the rock, and nothing complains"),
    "material_numbers/solid":       ("which code means grain interior", "code", "no dynamics at all"),
    "material_numbers/bounce_back": ("which code means the wall", "code", "the no-slip layer between pore and grain"),
    # ---- the flow
    "LB_numerics/Peclet":  ("Peclet number", "", "advection over diffusion, on the pore size above. 0 means diffusion only"),
    "LB_numerics/tau":     ("relaxation time for the flow", "", "sets the model viscosity. Stay between 0.5 and 2"),
    "LB_numerics/delta_P": ("pressure drop seed", "lattice", "a starting guess. The solver re-solves to hit the target Peclet"),
    "LB_numerics/track_performance": ("time the solver instead of writing output", "", "suppresses all output when on"),
    # ---- how long
    "iteration/ns_max_iT1":     ("flow step cap, first solve", "steps", ""),
    "iteration/ns_max_iT2":     ("flow step cap, later solves", "steps", ""),
    "iteration/ns_converge_iT1": ("steady flow tolerance, first solve", "", ""),
    "iteration/ns_converge_iT2": ("steady flow tolerance, later solves", "", ""),
    "iteration/ns_rerun_iT0":   ("flow restart iteration", "steps", "when reading a checkpoint"),
    "iteration/ns_update_interval": ("re-solve the flow at most every", "steps", ""),
    "iteration/ade_max_iT":     ("transport and reaction steps to run", "steps", ""),
    "iteration/ade_converge_iT": ("transport tolerance", "", "read, but there is no transport convergence test"),
    "iteration/ade_rerun_iT0":  ("chemistry restart iteration", "steps", "when reading a checkpoint"),
    "iteration/ade_update_interval": ("refresh the diffusivities at most every", "steps", ""),
    # ---- the chemicals
    "chemistry/number_of_substrates": ("how many chemicals", "count", ""),
    "chemistry/fix_concentration":    ("hold these chemicals fixed", "", "the reaction change is computed but not applied"),
    "chemistry/fix_lower_bounds":     ("clamp the uptake bound to the local amount", "", ""),
    "name_of_substrates":  ("name", "", "fixes which output file is which chemical"),
    "initial_concentration": ("starting amount everywhere", "mol/L", ""),
    "substrate_diffusion_coefficients/in_pore":    ("diffusivity in open water", "m2/s", ""),
    "substrate_diffusion_coefficients/in_biofilm": ("diffusivity inside biofilm", "m2/s", ""),
    "immobile": ("solid phase", "", "3D only. Never moves, only accumulates its reaction where it sits"),
    # ---- the microbes
    "microbiology/number_of_microbes":      ("how many microbe pools", "count", ""),
    "microbiology/maximum_biomass_density": ("biomass that fills a voxel", "mol/L", "above this, attached biomass spills into neighbours"),
    "microbiology/thrd_biofilm_fraction":   ("fraction of that at which a voxel counts as biofilm", "", "and starts resisting flow"),
    "microbiology/CA_method":               ("how spilling works", "", "move only the excess, or half the total"),
    "name_of_microbes":   ("name", "", ""),
    "solver_type":        ("how this pool moves", "", "CA attached by cellular automaton, FD attached by diffusion, LBM planktonic and carried with the flow"),
    "reaction_type":      ("what decides its growth", "", "none, kinetics, glpk, cobrapy, surrogate, or a pair of them"),
    "initial_densities":  ("starting biomass", "mol/L", "one value per material number this microbe is seeded on"),
    "decay_coefficient":  ("death rate", "1/s", "first order"),
    "viscosity_ratio_in_biofilm": ("how much more the biofilm resists flow", "", "0 means it is a solid wall"),
    "biomass_diffusion_coefficients/in_pore":    ("how fast the cells spread in open water", "m2/s", ""),
    "biomass_diffusion_coefficients/in_biofilm": ("how fast the cells spread inside biofilm", "m2/s", ""),
    "half_saturation_constants": ("half saturation, one per chemical", "mol/L", "read by the flux balance path. The hand written path carries its own"),
    "maximum_uptake_flux": ("top uptake rate, one per chemical", "mol/mol/s in 2D, mmol/gDW/h for flux balance",
                            "in 2D this one tag is shared by both paths, in two different units. 3D splits it"),
    "fba_maximum_uptake_flux": ("top uptake rate for flux balance", "mmol/gDW/h", "3D only, so it need not share with the kinetics path"),
    "biomass_molar_mass": ("grams dry weight per mole of cells", "g/mol", "3D only. The one place biomass is converted for a metabolic model"),
    "model_filename":     ("metabolic model file", "", ""),
    "exchange_reaction_indices": ("which reaction exchanges each chemical", "index",
                                 "POSITIONAL: a model revision that inserts one reaction shifts every index after it"),
    "substrate_lower_bounds": ("floor on the exchange fluxes", "mmol/gDW/h", "eating is negative"),
    "substrate_upper_bounds": ("ceiling on the exchange fluxes", "mmol/gDW/h", ""),
    "objective_direction":    ("maximize or minimize the model objective", "", ""),
    "equate_bounds":          ("pin the flux to exactly the Monod rate", "", ""),
    "constraint_indices":     ("reactions to override at load time", "index", "for knocking one out or forcing one on"),
    "constraint_lower_bounds": ("their floor", "mmol/gDW/h", ""),
    "constraint_upper_bounds": ("their ceiling", "mmol/gDW/h", ""),
    # ---- boundaries
    "left_boundary_type":       ("inlet condition", "", "Dirichlet holds the value, Neumann lets it flow out"),
    "left_boundary_condition":  ("inlet value", "mol/L", ""),
    "right_boundary_type":      ("outlet condition", "", ""),
    "right_boundary_condition": ("outlet value", "mol/L", ""),
    # ---- speciation, clogging, reopening. 3D only
    "equilibrium/enabled":     ("aqueous speciation on", "", "3D only. Solved in every pore voxel every step"),
    "equilibrium/components":  ("the master species", "", "their names must match the chemical names"),
    "precipitation/enabled":   ("mineral growth and pore clogging on", "", "3D only. Needs the chemical rate laws on as well"),
    "precipitation/solid_substrate": ("which chemical holds the mineral", "index", "must be marked immobile"),
    "precipitation/max_precipRho":   ("amount at which a voxel seals", "mol/L", "1000 divided by the molar volume in cm3/mol"),
    "precipitation/surface_only":    ("grow only on wall voxels", "", "1 is heterogeneous nucleation, the physical case"),
    "precipitation/perm_ratio":      ("how permeable a sealed voxel is", "", "0 means an impermeable wall"),
    "precipitation/update_interval": ("steps between re-checking the fill", "steps", "every re-check re-solves the flow"),
    "dissolution/enabled":         ("mineral loss and pore reopening on", "", "3D only. The rate itself is in defineAbioticKinetics.hh"),
    "dissolution/reopen_fraction": ("reopen below this fraction of full", "", "hysteresis, so a voxel does not flip every check"),
    "dissolution/surface_only":    ("only wetted voxels dissolve", "", "the physical case"),
    "dissolution/update_interval": ("steps between geometry re-checks", "steps", ""),
    "material_number": ("which code this phase occupies", "code", ""),
    "full_density":    ("amount in a completely full voxel", "mol/L", ""),
    "initial_fill":    ("amount a voxel starts with", "mol/L", "0 for a precipitate, full for an original grain"),
    "is_precipitate":  ("this is what precipitation converts voxels into", "", "so they can later reopen"),
    # ---- output
    "IO/read_NS_file":       ("restart the flow from a checkpoint", "", ""),
    "IO/read_ADE_file":      ("restart the chemistry from a checkpoint", "", ""),
    "IO/ns_filename":        ("flow output prefix", "", ""),
    "IO/subs_filename":      ("chemical output prefix", "", "in 2D the files are numbered, so this plus the order is what names them"),
    "IO/bio_filename":       ("biomass output prefix", "", ""),
    "IO/mask_filename":      ("material map output prefix", "", "where a changing pore structure is recorded"),
    "IO/save_VTK_interval":  ("steps between snapshots", "steps", ""),
    "IO/save_CHK_interval":  ("steps between restart points", "steps", ""),
    "IO/debug_updRxn":       ("print reaction debug lines", "", "3D only"),
}


def describe_setting(path):
    """A readable name, a unit and a sentence, for one setting.

    Longest match wins, so a per-chemical setting is described by the tail that
    is common to all of them rather than needing one entry per chemical.
    """
    best = None
    for key in SETTING_INFO:
        if path.endswith(key) and (best is None or len(key) > len(best)):
            best = key
    if best is None:
        return path.rsplit("/", 1)[-1].replace("_", " "), "", ""
    return SETTING_INFO[best]


def xml_pairs_with_comments(text):
    """Every setting in the file, with the comment that was written beside it.

    ElementTree drops comments, and the comments in these files are where the
    person who wrote the run said what they meant by a number. Keeping them is
    most of the value of keeping the file at all.
    """
    import xml.etree.ElementTree as ET
    out = []
    try:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        parser.feed(text)
        root = parser.close()
    except Exception:                                          # noqa: BLE001
        return out

    def walk(el, path):
        # A comment sits on either side of the setting it belongs to, and which
        # side decides which setting it is about:
        #
        #     <tau> 0.8 </tau>    <!-- sets the viscosity -->     belongs BACK
        #     <!-- the grid -->
        #     <nx> 100 </nx>                                      belongs FORWARD
        #
        # ElementTree hands both over as plain siblings with no line numbers, so
        # the only thing separating them is whether the previous element's tail
        # got as far as a newline before the comment started. Reading every
        # comment forward, which is the obvious way, shifts each one onto the
        # following setting and quietly mislabels the whole table.
        pending = ""
        prev = None                 # index in `out` of the last leaf written
        prev_tail = ""
        for k in list(el):
            tag = k.tag
            if not isinstance(tag, str):                 # a comment node
                text = " ".join((k.text or "").split())
                if prev is not None and "\n" not in prev_tail:
                    p, v, c = out[prev]
                    out[prev] = (p, v, (c + " " + text).strip())
                else:
                    pending = text
                prev_tail = k.tail or ""
                continue
            sub = path + "/" + tag
            if len(k):
                walk(k, sub)
                prev = None
            else:
                out.append((sub, (k.text or "").strip(), pending))
                prev = len(out) - 1
            pending = ""
            prev_tail = k.tail or ""
    walk(root, root.tag)
    return out


# ------------------------------------------------------------------ inputs --
# CompLaB opens CompLaB.xml from the folder it was launched in, by that exact
# name, and never copies it next to the results. Output written a year ago
# therefore arrives with nothing describing it, which is the normal case rather
# than the exception. So nothing in this section is allowed to stop a run being
# collected. Every reader below returns a record saying what it found or,
# failing that, where it looked and did not find it, and the caller writes that
# record into the file either way.

XML_NAMES = ["CompLaB.xml"]
KIN_NAMES = ["defineKinetics.hh"]
ABIO_NAMES = ["defineAbioticKinetics.hh"]


def _looks_like_complab_xml(path):
    """A CompLaB input has <parameters> as its root. Anything else that happens
    to be sitting in the folder is somebody else's file."""
    try:
        import xml.etree.ElementTree as ET
        return ET.parse(path).getroot().tag == "parameters"
    except Exception:                                          # noqa: BLE001
        return False


def find_beside_run(run_dir, names, given=None, sniff=None):
    """Where an input file might be, in the order worth looking.

    Returns (path, note). path is None when nothing matched, and the note then
    says where the search went, so the answer in the dataset is "not found, and
    here is where I looked" rather than a bare false.
    """
    tried = []
    run_name = os.path.basename(run_dir.rstrip(os.sep))
    if given:
        if os.path.isfile(given):
            # One file named explicitly means THE SAME file for every run. Say
            # so in the note, because a campaign where each run had its own is
            # then being recorded with somebody else's settings.
            return given, "given on the page, the same file for every run"
        if os.path.isdir(given):
            # A folder is allowed, and it is looked in twice. First for a
            # subfolder named after this run, which is how a set of per run
            # inputs kept away from the results is laid out. Then for one file
            # sitting directly in it, which is the shared campaign case.
            for nm in names:
                cand = os.path.join(given, run_name, nm)
                if os.path.isfile(cand):
                    return cand, "given on the page, this run's own copy"
            for nm in names:
                cand = os.path.join(given, nm)
                if os.path.isfile(cand):
                    return cand, "given on the page, shared by the campaign"
        tried.append(given)

    parent = os.path.dirname(run_dir.rstrip(os.sep))
    for base in (run_dir, os.path.join(run_dir, "input"), parent):
        for nm in names:
            cand = os.path.join(base, nm)
            if os.path.isfile(cand):
                return cand, "found beside the run"
        tried.append(base)

    if sniff:
        for base in (run_dir, parent):
            for cand in sorted(glob.glob(os.path.join(base, sniff))):
                if _looks_like_complab_xml(cand):
                    return cand, "the only file here with <parameters> in it"

    return None, "not found. Looked in " + ", ".join(
        os.path.basename(t.rstrip(os.sep)) or t for t in tried)


def flatten_xml(text):
    """Every setting in the file as a (path, value) pair.

    A viewer then shows the input file as a two column table with no parser
    involved, which is the difference between a record somebody can read and a
    blob nobody opens.
    """
    import xml.etree.ElementTree as ET
    out = []
    try:
        root = ET.fromstring(text)
    except Exception:                                          # noqa: BLE001
        return out

    def walk(el, path):
        kids = list(el)
        if not kids:
            v = (el.text or "").strip()
            out.append((path, v))
            return
        for k in kids:
            walk(k, path + "/" + k.tag)
    walk(root, root.tag)
    return out


def read_xml_input(run_dir, given=None):
    """The run's CompLaB.xml, whole and flattened, or a plain note saying there
    was none. Never raises."""
    rec = dict(present=False, note="", source="", text="", pairs=[], rows=[])
    path, note = find_beside_run(run_dir, XML_NAMES, given, sniff="*.xml")
    if path is None:
        rec["note"] = note
        return rec
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:                                     # noqa: BLE001
        rec["note"] = "found %s but could not read it: %s" % (path, e)
        return rec
    rec.update(present=True, note=note, source=path, text=text,
               pairs=flatten_xml(text), rows=xml_pairs_with_comments(text))
    if not rec["pairs"]:
        rec["note"] = note + ", but it did not parse as XML, so only the text " \
                             "is kept"
    return rec


# A rate constant in these files is a named number with its meaning written
# beside it. Both spellings appear: the 3D files declare them inside a named
# block, the 2D file declares them loose inside the routine.
_CONST_RE = re.compile(
    r"^\s*(?:constexpr\s+|static\s+|const\s+)*(?:double|float)\s+"
    r"([A-Za-z_]\w*)\s*=\s*([-+0-9.eE]+)\s*;\s*(?://\s*(.*))?$")
# The index block, which is that file's own statement of what order it expects
# the chemicals to be in: constexpr int AC=0, U6=1, U4=2, ...
# Not anchored at the end of the line: the declaration usually carries a note
# after its semicolon, and requiring the line to STOP at the semicolon quietly
# skipped every block that had one.
_INDEX_RE = re.compile(r"^\s*constexpr\s+int\s+([^;]+);")
_PAIR_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*(-?\d+)")
_ROUTINE_RE = re.compile(
    r"^\s*(?:inline\s+)?(?:void|double)\s+([A-Za-z_]\w*)\s*\(")


def parse_kinetics(text):
    """Pull the numbers and the expected chemical order out of a kinetics file.

    Nothing here is executed or compiled. The file itself is kept whole beside
    this, and that copy is the record; this is the convenience view, so a
    partial parse is a partial view and never an error.
    """
    consts, index, routines = [], [], []
    # A declaration does not have to fit on one line, and the one that matters
    # most does not: the chemical order block is written across two, which a
    # line-at-a-time reader misses entirely and then reports the file as
    # having no expected order at all.
    #
    #     constexpr int AC=0, U6=1, U4=2, FE2=3, FE3E=4, FE3M=5, FE3H=6,
    #                   SO4=7, HS=8, FES=9, HCO3=10, HP=11, CH4=12, S0=13;
    #
    # So a statement that has opened and not yet reached its semicolon takes
    # the following lines with it.
    lines, pending = [], None
    for raw in text.splitlines():
        if pending is not None:
            pending += " " + raw.strip()
            if ";" in raw:
                lines.append(pending)
                pending = None
            continue
        if re.match(r"^\s*constexpr\s+(?:int|double|float)\s", raw) \
                and ";" not in raw:
            pending = raw.rstrip()
            continue
        lines.append(raw)
    if pending is not None:
        lines.append(pending)

    for line in lines:
        m = _CONST_RE.match(line)
        if m:
            try:
                consts.append((m.group(1), float(m.group(2)),
                               (m.group(3) or "").strip()))
            except ValueError:
                pass
            continue
        m = _INDEX_RE.match(line)
        if m:
            index += [(a, int(b)) for a, b in _PAIR_RE.findall(m.group(1))]
            continue
        m = _ROUTINE_RE.match(line)
        if m and m.group(1) not in ("if", "for", "while", "switch", "return"):
            routines.append(m.group(1))
    dt = next((v for n, v, _ in consts if n == "dt_kinetics"), None)
    return dict(constants=consts, index=index, routines=routines,
                dt_kinetics=dt)


def read_kinetics(run_dir, names, given=None):
    """One kinetics header, whole and parsed, or a note saying there was none."""
    rec = dict(present=False, note="", source="", text="", constants=[],
               index=[], routines=[], dt_kinetics=None)
    path, note = find_beside_run(run_dir, names, given)
    if path is None:
        rec["note"] = note
        return rec
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:                                     # noqa: BLE001
        rec["note"] = "found %s but could not read it: %s" % (path, e)
        return rec
    rec.update(present=True, note=note, source=path, text=text)
    rec.update(parse_kinetics(text))
    return rec


def species_from_xml(pairs):
    """The chemicals a run wrote, named, from its own input file.

    This is the whole reason the input file is worth keeping. A build that
    numbers its output writes subsLattice0, subsLattice1 and so on, and those
    names say nothing: which chemical subsLattice0 holds is decided only by the
    ORDER of the <substrateN> blocks. Pair the two and the files stop being
    anonymous.

    Returns [(name, glob)], or an empty list if the input file did not say.
    """
    d = {}
    for p, v in pairs:
        d[p] = (v or "").strip()

    def get(suffix):
        for p, v in d.items():
            if p.endswith(suffix):
                return v
        return None

    try:
        n = int(float(get("/chemistry/number_of_substrates")))
    except (TypeError, ValueError):
        return []
    subs = get("/IO/subs_filename") or "subsLattice"
    out = []
    for i in range(n):
        nm = get("/substrate%d/name_of_substrates" % i) or ("chemical_%d" % i)
        out.append((nm, "%s%d_*.vti" % (subs, i)))

    # Microbes are stored on the same footing as the chemicals, in their own
    # numbered files, so they belong in the same list.
    try:
        m = int(float(get("/microbiology/number_of_microbes")))
    except (TypeError, ValueError):
        m = 0
    bio = get("/IO/bio_filename") or "bioLattice"
    for i in range(m):
        nm = get("/microbe%d/name_of_microbes" % i) or ("microbe_%d" % i)
        out.append((nm, "%s%d_*.vti" % (bio, i)))
    return out


def ade_dt_from(pairs):
    """The reaction time step the run really used, from its own input file.

        ade_dt = ((tau - 0.5) / 3) * dx^2 / D0

    Both kinetics files carry a hardcoded dt_kinetics that has to equal this,
    because it is what their per-step stability limit divides by. When the two
    disagree the limiter clipped the rates differently from what was intended,
    and that is worth being able to see rather than wonder about later.

    Returns None rather than a guess if the input file did not say.
    """
    d = {p: v for p, v in pairs}

    def num(key):
        for p, v in d.items():
            if p.endswith(key):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
        return None

    tau = num("/LB_numerics/tau")
    dx = num("/domain/dx")
    unit = None
    for p, v in d.items():
        if p.endswith("/domain/unit"):
            unit = (v or "").strip()
    d0 = num("/substrate0/substrate_diffusion_coefficients/in_pore")
    if None in (tau, dx) or d0 in (None, 0):
        return None
    scale = {"m": 1.0, "mm": 1e-3, "um": 1e-6}.get(unit or "um", 1e-6)
    dxm = dx * scale
    return ((tau - 0.5) / 3.0) * dxm * dxm / d0



def _trim(note, width=60):
    """One short line. A rate constant's note in the source is often a whole
    citation, and a citation repeated down every row of a table is what turns
    a table into a wall."""
    note = " ".join(str(note).split())
    return note if len(note) <= width else note[:width - 3].rstrip(" ,.;") + "..."


def _unit_from_note(note):
    """The unit, when the source wrote one in the note beside the constant.

    These files put the unit in the comment rather than anywhere structured,
    so this looks for the shapes they actually use: (L/mol/s; ...), 1/s, mol/L,
    [seconds]. Anything it cannot find stays blank rather than being invented.
    """
    import re as _re
    m = _re.search(r"\(([^();]*?/[^();]*?)[;)]", note)
    if m:
        return m.group(1).strip()
    # Only the explicit written-out forms. Matching prose like "per second"
    # inside "mol donor per mol cells per second" gives the wrong unit, and a
    # wrong unit in a table is worse than a blank one.
    m = _re.search(r"\b(mmol/gDW/h|mol/mol/s|m\^?2/s|mol/L|1/s|L/mol/s|"
                   r"g/mol|s\^-1)\b", note)
    return m.group(1) if m else ""


def _num_or_nan(text):
    """A setting's value as a number, or not-a-number when it is not one.

    A list ("1.0e-9 5.0e-10") takes its first entry, because a column has to
    hold one number and the first is the one people mean when they sort on it.
    """
    try:
        return float(str(text).split()[0])
    except (ValueError, IndexError, TypeError):
        return float("nan")


def write_inputs(h, recs, notes):
    """The inputs/ group: what each run was actually told to do.

    FOUR TABLES, and each is one node. Click it and you are looking at the
    thing itself, not at an index pointing somewhere else.

        xml        every setting in the input file
        kinetics   every rate constant in the .hh files
        order      the chemical order those files expect
        files      which file each run got, or why it got none

    A table here is an HDF5 compound dataset, which is to say a real table with
    named columns rather than a pile of parallel arrays. The columns that
    matter:

        name      a readable name, in words rather than in tag spelling
        meaning   what it is for, taken from the comment written beside it in
                  the run's own file, and otherwise from what the source says
                  the setting does
        unit      blank where it has none
        value     exactly as written, so nothing is lost
        number    THE SAME AS A NUMBER, or not-a-number where it is not one

    That last column is the point of the whole design. A Peclet stored only as
    the text "3.3" has to be parsed by every script that ever wants to sort on
    it. Stored as a float beside its own name and unit, it is usable as it
    stands.

    The first axis of every table is the run, so a run that had no input file
    is a row of blanks and not-a-numbers rather than a gap in the layout.
    """
    import h5py
    vs = h5py.string_dtype("utf-8")
    S = len(recs)
    g = h.create_group("inputs")

    # Five columns, not seven. An earlier version also carried the raw tag
    # path and a full sentence of explanation on every row, which meant the
    # same paragraph repeated down sixty-five rows and again for every run,
    # and the two columns that actually get used, value and number, pushed off
    # the right of the screen. The name is the readable form of the path, so
    # the path was saying the same thing twice; and a note that is identical
    # on every row is not telling anyone anything.
    SETTING = np.dtype([("name", vs), ("value", vs), ("number", "f8"),
                        ("unit", vs), ("note", vs)])
    CONSTANT = np.dtype([("name", vs), ("value", "f8"), ("unit", vs),
                         ("note", vs), ("file", vs)])
    ORDER = np.dtype([("name", vs), ("position", "i4"), ("file", vs)])
    FILES = np.dtype([("what", vs), ("found", vs), ("where", vs)])

    # ---- 1. every setting in the input file -------------------------------
    # The columns are the union over the runs, in the order they first appear,
    # so a run whose file is missing a setting shows a blank in that column
    # instead of shifting everything after it.
    order, seen = [], {}
    for r in recs:
        for p, _, _ in r["xml"]["rows"]:
            if p not in seen:
                seen[p] = len(order)
                order.append(p)
    P = len(order)
    xml = np.empty((S, max(P, 1)), SETTING)
    xml[...] = ("", "", np.nan, "", "")
    for i, p in enumerate(order):
        nm, unit, _ = describe_setting(p)
        for k in range(S):
            xml[k, i] = (nm, "", np.nan, unit, "")
    for k, r in enumerate(recs):
        for p, v, comment in r["xml"]["rows"]:
            j = seen[p]
            nm, unit, mean = describe_setting(p)
            # The note carries the comment the person wrote beside THIS
            # setting in THIS run, because that is the only place a run says
            # why its own number is that number. The general description of
            # what a setting is for is the same on every row of every run, so
            # it is used only where a setting would otherwise have no note at
            # all, and it is cut to one short line.
            note = comment or mean
            if len(note) > 60:
                note = note[:57].rstrip(" ,.;") + "..."
            xml[k, j] = (nm, v, _num_or_nan(v), unit, note)
    d = g.create_dataset("xml", data=xml)
    d.attrs["how_to_read_this"] = (
        b"One row per setting, one table per run: move the run slider. The "
        b"number column is a real number wherever the setting is one, so it "
        b"can be sorted, differenced or fed straight into anything else "
        b"without parsing the value column. Not-a-number means the setting is "
        b"not a number, or that run had no input file.")

    # ---- 2. every rate constant -------------------------------------------
    keys, kseen = [], {}
    for r in recs:
        for src, fam in (("kin", "defineKinetics.hh"),
                         ("abio", "defineAbioticKinetics.hh")):
            for nm, _, note in r[src]["constants"]:
                if (fam, nm) not in kseen:
                    kseen[(fam, nm)] = len(keys)
                    keys.append((nm, fam, note))
    K = len(keys)
    kin = np.empty((S, max(K, 1)), CONSTANT)
    kin[...] = ("", np.nan, "", "", "")
    for i, (nm, fam, note) in enumerate(keys):
        for k in range(S):
            kin[k, i] = (nm, np.nan, "", _trim(note), fam)
    for k, r in enumerate(recs):
        for src, fam in (("kin", "defineKinetics.hh"),
                         ("abio", "defineAbioticKinetics.hh")):
            for nm, v, note in r[src]["constants"]:
                i = kseen[(fam, nm)]
                kin[k, i] = (nm, v, _unit_from_note(note), _trim(note), fam)
    d = g.create_dataset("kinetics", data=kin)
    d.attrs["how_to_read_this"] = (
        b"The rate constants, with the note written beside each one in the "
        b"source. These live in the .hh files and never in the input file, so "
        b"a result cannot be reproduced without them.")

    # ---- 3. the chemical order those files expect -------------------------
    # Each kinetics file opens with its own statement of what order it needs
    # the chemicals to be in. Comparing it against the species list is the one
    # check the solver itself will not do for you.
    rows = []
    seeno = set()
    for r in recs:
        for src, fam in (("kin", "defineKinetics.hh"),
                         ("abio", "defineAbioticKinetics.hh")):
            for nm, pos in r[src]["index"]:
                if (fam, nm) not in seeno:
                    seeno.add((fam, nm))
                    rows.append((nm, pos, fam))
    orr = np.empty(max(len(rows), 1), ORDER)
    orr[...] = ("", -1, "")
    for i, (nm, pos, fam) in enumerate(rows):
        orr[i] = (nm, pos, fam)
    g.create_dataset("order", data=orr)

    # ---- 4. which file each run got ---------------------------------------
    fam = [("the input file", "xml"), ("the kinetics file", "kin"),
           ("the abiotic kinetics file", "abio")]
    files = np.empty((S, len(fam)), FILES)
    for k, r in enumerate(recs):
        for i, (label, key) in enumerate(fam):
            rec = r[key]
            files[k, i] = (label, "yes" if rec["present"] else "no",
                           rec["source"] if rec["present"] else rec["note"])
    d = g.create_dataset("files", data=files)
    d.attrs["how_to_read_this"] = (
        b"CompLaB reads its input file from the folder it is launched in and "
        b"never copies it next to the results, so older output arrives with "
        b"nothing describing it. A no here is that, not a fault, and the "
        b"where column says exactly which places were searched.")

    # ---- 5. the reaction time step ----------------------------------------
    # The one number that has to agree between the input file and the kinetics
    # file: the stability limit inside the kinetics file divides by the step it
    # assumes, so a mismatch means the rates were clipped somewhere other than
    # intended. Both are kept, and neither is corrected to match the other.
    real, assumed = [], []
    for r in recs:
        v = r.get("dt_recorded")
        if v is None:
            v = ade_dt_from([(p, val) for p, val, _ in r["xml"]["rows"]])
        real.append(v if v is not None else np.nan)
        a = r["kin"]["dt_kinetics"] if r["kin"]["present"] else None
        assumed.append(a if a is not None else np.nan)
    g.create_dataset("time_step", data=np.array(
        list(zip(real, assumed)), np.float64))
    g["time_step"].attrs["columns"] = (
        b"the step the run really used, and the step its kinetics file "
        b"assumes. They have to agree.")

    g.attrs["runs_without_input_file"] = int(
        sum(1 for r in recs if not r["xml"]["present"]))
    return S


def write_grid(h, recs, names):
    """What was imposed at the edges of the box.

    A concentration field means one thing if the inlet was held at a fixed
    value and quite another if it was left to flow freely, and the fields
    themselves do not say which. This is read out of each run's input file
    where there is one.
    """
    import h5py
    vs = h5py.string_dtype("utf-8")
    BC = np.dtype([("species", vs), ("face", vs), ("type", vs), ("value", "f8")])
    S, C = len(recs), len(names)
    bc = np.empty((S, max(C * 2, 1)), BC)
    bc[...] = ("", "", "", np.nan)
    for k, r in enumerate(recs):
        d = {p: v for p, v, _ in r["xml"]["rows"]}

        def get(i, side, what):
            for p, v in d.items():
                if p.endswith("/substrate%d/%s_boundary_%s" % (i, side, what)):
                    return v
            return None
        for i, nm in enumerate(names):
            for j, side in enumerate(("left", "right")):
                t = get(i, side, "type")
                v = get(i, side, "condition")
                bc[k, i * 2 + j] = (nm, "inlet" if side == "left" else "outlet",
                                    t or "", _num_or_nan(v) if v is not None
                                    else np.nan)
    gg = h.create_group("grid")
    gg.create_dataset("boundary_conditions", data=bc)
    gg["boundary_conditions"].attrs["how_to_read_this"] = (
        b"Dirichlet means the value was held there, so it is an injection. "
        b"Neumann means nothing was imposed and whatever arrived flowed out. "
        b"Blank means the run kept no input file to read it from.")


def write_ancillary(h, recs, notes):
    """Free text. Anything worth writing down that is not a number.

    Deliberately not a place for computed properties of the rock. Those get
    recomputed with a better method eventually, and a stale copy sitting in an
    old file is worse than no copy at all.
    """
    import h5py
    vs = h5py.string_dtype("utf-8")
    a = h.create_group("ancillary")
    a.create_dataset("run_note", data=np.array(
        ["; ".join(notes.get(r["name"], [])) for r in recs], dtype=vs))
    a.create_dataset("run_name", data=np.array(
        [r["name"] for r in recs], dtype=vs))
    a.attrs["dataset_note"] = b""
    a.attrs["how_to_read_this"] = (
        b"run_note is whatever the collector had to say about a particular "
        b"run. dataset_note is empty and yours to write in.")


# --------------------------------------------------------------------- main --


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True,
                   help="folder holding one subfolder per run")
    p.add_argument("--out", default="./dataset",
                   help="dataset.h5 is written inside this folder")
    p.add_argument("--inspect", action="store_true",
                   help="report what was found and write nothing")

    g = p.add_argument_group("where things are inside each run folder")
    g.add_argument("--geometry-file", default="inputGeom.vti")
    g.add_argument("--species", nargs="+", default=None, metavar="NAME=GLOB",
                   help="one entry per chemical, e.g. "
                        "--species Ac=Ac_*.vti A=A_*.vti. "
                        "Default: C=subsLattice0_*.vti")
    g.add_argument("--flow-glob", default="nsLattice*.vti")
    g.add_argument("--no-velocity", action="store_true")
    g.add_argument("--rate-glob", default="rateLattice*.vti", dest="rate_glob",
                   help="the reaction rate snapshots, written when "
                        "<save_reaction_rates> is on. One channel per array "
                        "inside, named by the kinetics header. A run without "
                        "them is collected exactly as before.")
    g.add_argument("--no-rates", action="store_true",
                   help="do not look for the rate files at all")

    g = p.add_argument_group("which code means pore")
    g.add_argument("--pore-code", type=int, default=None,
                   help="Normally leave this alone. The run's CompLaB.xml is "
                        "read for <material_numbers><pore> first, and that is "
                        "the same place the solver read it from. This is the "
                        "override for a run that has no input file.")

    g = p.add_argument_group("where the conditions come from")
    g.add_argument("--conditions-csv", default="", dest="conditions_csv",
                   help="OFF by default. The conditions are taken from the "
                        "run's CompLaB.xml, which is what the solver actually "
                        "read. Name a csv here only if you want one read as "
                        "well, and it will fill in gaps rather than override "
                        "the input file.")
    g.add_argument("--pe-column", default="Pe_achieved")
    g.add_argument("--porosity-column", default="Porosity")
    g.add_argument("--dt-column", default="dt_s", dest="dt_column",
                   help="the column of that csv holding the reaction time "
                        "step in seconds. Used to check the kinetics file "
                        "against the run. Blank to skip.")
    g.add_argument("--pe", type=float, default=None)
    g.add_argument("--da-bio", type=float, default=None)
    g.add_argument("--da-abio", type=float, default=None)
    g.add_argument("--ks-ac", type=float, default=None)
    g.add_argument("--ks-a", type=float, default=None)
    g.add_argument("--y", type=float, default=None)

    g = p.add_argument_group("the input files each run was given")
    g.add_argument("--xml", default=None, metavar="PATH",
                   help="the CompLaB.xml these runs were launched with, or a "
                        "folder holding it. Without this each run folder and "
                        "its parent are searched. A run with no input file is "
                        "still collected; the dataset simply records that "
                        "there was none.")
    g.add_argument("--kinetics", default=None, metavar="PATH",
                   help="defineKinetics.hh, same search if not given")
    g.add_argument("--abiotic-kinetics", default=None, metavar="PATH",
                   dest="abiotic_kinetics",
                   help="defineAbioticKinetics.hh, 3D only, same search")
    g.add_argument("--no-inputs", action="store_true",
                   help="do not look for the input files at all")

    g = p.add_argument_group("the flow descriptors")
    g.add_argument("--no-flow-features", action="store_true",
                   help="skip MIS and UPRM. They depend only on the geometry and cost "
                        "about a second a rock; add_flow_features.py can add them later.")
    g.add_argument("--flow-buffer", type=int, default=10,
                   help="open buffer voxels at each end of the flow axis, for the MIS "
                        "treatment. Must match your campaign's padding: 10 for the "
                        "published 2D set, 5 for our generator, 0 for none.")

    g = p.add_argument_group("checks")
    g.add_argument("--max-conc", type=float, default=1.0,
                   help="reject a run whose concentration exceeds this")
    g.add_argument("--clip", action="store_true",
                   help="clip to --max-conc instead of rejecting")
    p.add_argument("--no-provenance", action="store_true",
                   help="do not copy the conditions file into the dataset. The "
                        "csv is still READ for Peclet and for the pore code; "
                        "this only stops its other columns being kept "
                        "alongside the data.")
    args = p.parse_args()

    # A subfolder with no VTK output in it is not a run. Campaign folders
    # collect other things beside the runs: the input files, a notes folder,
    # somewhere to put rejects. Listing those as runs and then rejecting them
    # one by one buries the real rejections in noise.
    runs, skipped = [], []
    for d in sorted(os.listdir(args.runs)):
        p = os.path.join(args.runs, d)
        if not os.path.isdir(p):
            continue
        if glob.glob(os.path.join(p, "*.vti")):
            runs.append(d)
        else:
            skipped.append(d)

    pairs, pairs_src = [], ""
    if args.species:
        for s in args.species:
            if "=" not in s:
                sys.exit("--species must be NAME=GLOB, got %r" % s)
            n, gl = s.split("=", 1)
            pairs.append((n, gl))
        pairs_src = "given on the page"
    elif not args.no_inputs:
        # Ask the input file what the chemicals are called before falling back
        # to a numbered guess. One file is enough: a campaign that used two
        # different chemical lists is not one dataset anyway, and each run's
        # own file is still read and stored separately below.
        for name in runs:
            x = read_xml_input(os.path.join(args.runs, name), args.xml)
            if x["present"]:
                pairs = species_from_xml(x["pairs"])
                if pairs:
                    pairs_src = "read from %s" % os.path.basename(x["source"])
                break
    if not pairs:
        pairs = [("C", "subsLattice0_*.vti")]
        pairs_src = ("no input file said, so falling back to one unnamed "
                     "chemical")
    if not runs:
        sys.exit("no run folders under %s" % args.runs)
    print("%d run folder(s) under %s" % (len(runs), args.runs))
    if skipped:
        print("   not runs, no .vti inside, left alone: %s"
              % ", ".join(skipped))
    print("chemicals: %s" % ", ".join("%s from %s" % (n, g) for n, g in pairs))
    print("           (%s)" % pairs_src)
    print()

    mats, gdfs, edts, mat_hash = [], [], [], {}
    recs, failures = [], []
    notes = {}          # run name -> things worth saying about it in the file
    shape = None

    for name in runs:
        rd = os.path.join(args.runs, name)
        try:
            # The input files, if this run kept any. None of this can stop a
            # run being collected: a missing file is recorded as missing and
            # the fields it would have filled stay empty rather than guessed.
            if args.no_inputs:
                xin = dict(present=False, note="not looked for", source="",
                           text="", pairs=[], rows=[])
                kin = dict(present=False, note="not looked for", source="",
                           text="", constants=[], index=[], routines=[],
                           dt_kinetics=None)
                abio = dict(kin)
            else:
                xin = read_xml_input(rd, args.xml)
                kin = read_kinetics(rd, KIN_NAMES, args.kinetics)
                abio = read_kinetics(rd, ABIO_NAMES, args.abiotic_kinetics)

            got, src, porosity, row = conditions_for(rd, args, xin["pairs"])

            gpath = os.path.join(rd, args.geometry_file)
            if not os.path.isfile(gpath):
                # The geometry is the one thing a run cannot be collected
                # without: no rock means no material, no distance fields and
                # nothing for the network to look at. So try the names CompLaB
                # actually writes before giving up on the run.
                alt = [p2 for nm in ("inputGeom.vti", "geometry.vti",
                                     "maskLattice*.vti", "*Geom*.vti")
                       for p2 in sorted(glob.glob(os.path.join(rd, nm)))]
                if not alt:
                    raise ValueError(
                        "no geometry file here. Looked for %s and for "
                        "inputGeom.vti, geometry.vti, maskLattice*.vti. "
                        "A run without one cannot be collected."
                        % args.geometry_file)
                gpath = alt[0]
                print("     geometry taken from %s" % os.path.basename(gpath))
            _, raw = read_vti(gpath)
            arrs, _ = read_vti(gpath)
            tag = np.rint(list(arrs.values())[0]).astype(np.int32)
            # Which number means pore is set per run in the input file, and a
            # geometry written by one setup can mean the opposite of one
            # written by another. When the input file says, that is the answer
            # and there is nothing to work out.
            forced = args.pore_code
            how_pore = None
            if forced is None:
                v = from_xml_pairs(xin["pairs"], "/material_numbers/pore")
                if v is not None:
                    forced = int(v)
                    how_pore = "read from CompLaB.xml, <material_numbers><pore>"
            code, how = decide_pore_code(tag, porosity, forced)
            if how_pore:
                how = how_pore
            mat = to_our_codes(tag, code)

            if shape is None:
                shape = mat.shape
            elif mat.shape != shape:
                raise ValueError("grid %s does not match the first run's %s"
                                 % (mat.shape, shape))

            # one entry per DISTINCT rock, so runs that share a structure
            # share it here too
            h = hashlib.sha1(mat.tobytes()).hexdigest()
            if h not in mat_hash:
                from scipy import ndimage
                mat_hash[h] = len(mats)
                mats.append(mat)
                edts.append(ndimage.distance_transform_edt(mat == PORE
                                                           ).astype(np.float32))
                gdfs.append(geodesic(mat))
            gid = mat_hash[h]

            # Which chemicals this run actually wrote. A run that is missing
            # one of them is not a broken run, it is a run of a smaller
            # system, and it is collected with the ones it has.
            found = []
            for nm, gl in pairs:
                fs = species_files(rd, gl)
                if fs:
                    found.append((nm, fs))
                else:
                    notes.setdefault(name, []).append(
                        "no file matching %r for chemical %s" % (gl, nm))
            if not found:
                raise ValueError(
                    "no concentration files at all. Looked for %s. Give the "
                    "right pattern with --species NAME=GLOB."
                    % ", ".join(repr(g2) for _, g2 in pairs))

            # Snapshots the chemicals have IN COMMON. Different chemicals
            # stopping at different iterations is ordinary; lining them up by
            # position instead would put different physical times in the same
            # slot, which is the one thing that must not happen.
            common = set.intersection(*[{i for i, _ in fs} for _, fs in found])
            if not common:
                raise ValueError(
                    "the chemicals here share no snapshot iteration: %s"
                    % "; ".join("%s at %s" % (nm, sorted(i for i, _ in fs))
                                for nm, fs in found))
            iters = sorted(common)
            for nm, fs in found:
                dropped = sorted({i for i, _ in fs} - common)
                if dropped:
                    notes.setdefault(name, []).append(
                        "%s also had snapshots at %s, which the other "
                        "chemicals did not, so they were left out"
                        % (nm, dropped))

            per, kept_names = [], []
            for nm, fs in found:
                by_it = dict(fs)
                per.append([one_array(by_it[i])[1].astype(np.float32)
                            for i in iters])
                kept_names.append(nm)

            conc = np.stack([np.stack(c) for c in per], 1)      # (T, C, ...)
            pore = (mat == PORE)
            hi = float(conc[:, :, pore].max())
            if hi > args.max_conc:
                if args.clip:
                    conc = np.minimum(conc, args.max_conc)
                else:
                    raise ValueError(
                        "concentration %.4g exceeds the bound %.4g. Raise "
                        "--max-conc, or pass --clip, once you know why."
                        % (hi, args.max_conc))
            bad = ~np.isfinite(conc)
            if bad.any():
                # A handful of these at a sharp front is a numerical artefact,
                # not a reason to throw a whole run away. They are set to zero
                # and counted, so the count is in the file rather than in
                # somebody's memory of the console.
                notes.setdefault(name, []).append(
                    "%d of %d concentration values were not a number and were "
                    "set to zero" % (int(bad.sum()), int(bad.size)))
                conc = np.nan_to_num(conc, nan=0.0, posinf=0.0, neginf=0.0)
            if float(np.abs(conc).max()) == 0.0:
                raise ValueError("every concentration is identically zero")

            vel, vmag_only = None, False
            if not args.no_velocity:
                fs = sorted(glob.glob(os.path.join(rd, args.flow_glob)))
                if fs:
                    a, _ = read_vti(fs[-1])
                    vec = [v for v in a.values() if v.shape == (3,) + shape]
                    vel = np.zeros((3,) + shape, np.float32)
                    if vec:
                        vel = vec[0].astype(np.float32)
                    else:
                        mag = [v for v in a.values() if v.shape == shape]
                        if mag:
                            vel[0] = mag[0]
                            vmag_only = True

            # The reaction rates, when the run wrote them. They are read on the
            # SAME snapshot iterations as the concentrations, so a rate and a
            # concentration in the same slot are the same instant. A snapshot
            # the rate files do not have is not filled in from a neighbour:
            # the run is collected without rates rather than with invented ones.
            rate, rate_names = None, []
            if not args.no_rates:
                rf = species_files(rd, args.rate_glob)
                if rf:
                    by_it = dict(rf)
                    missing_it = [i for i in iters if i not in by_it]
                    if missing_it:
                        notes.setdefault(name, []).append(
                            "reaction rate files exist but not at %s, so no "
                            "rates were kept for this run" % missing_it)
                    else:
                        first = rate_channels(by_it[iters[0]], shape)
                        rate_names = [nm for nm, _ in first]
                        if not rate_names:
                            notes.setdefault(name, []).append(
                                "the reaction rate file holds no array that "
                                "matches the grid, so no rates were kept")
                        else:
                            stack = []
                            for i in iters:
                                got_ch = dict(rate_channels(by_it[i], shape))
                                if any(nm not in got_ch for nm in rate_names):
                                    stack = []
                                    notes.setdefault(name, []).append(
                                        "the rate channels change between "
                                        "snapshots here, so no rates were kept")
                                    break
                                stack.append(np.stack(
                                    [got_ch[nm] for nm in rate_names]))
                            if stack:
                                rate = np.stack(stack).astype(np.float32)
                                rate = np.nan_to_num(rate, nan=0.0,
                                                     posinf=0.0, neginf=0.0)
                else:
                    notes.setdefault(name, []).append(
                        "no reaction rate files matching %r" % args.rate_glob)

            span = max(iters[-1], 1)
            tnorm = np.array(iters, np.float32) / span
            recs.append(dict(name=name, gid=gid, conc=conc, vel=vel,
                             rate=rate, rate_names=rate_names,
                             tnorm=tnorm, got=got, src=src, iters=iters,
                             vmag_only=vmag_only, maxc=hi, row=row,
                             species=kept_names, xml=xin, kin=kin, abio=abio,
                             dt_recorded=recorded_dt(row, args.dt_column)))

            print("%-26s rock %d  grid %s" % (name, gid, shape))
            print("     pore code %d, %s" % (code, how))
            for lab, rec in (("input file", xin), ("kinetics", kin),
                             ("abiotic kinetics", abio)):
                if rec["present"]:
                    print("     %-17s %s" % (lab, os.path.basename(rec["source"])))
                elif rec["note"] != "not looked for":
                    print("     %-17s none. %s" % (lab, rec["note"]))
            print("     snapshots at %s" % iters)
            print("     highest concentration in pore space %.4g" % hi)
            if rate is not None:
                print("     reaction rates: %d channel(s), %s"
                      % (len(rate_names), ", ".join(rate_names)))
            if vmag_only:
                print("     FLOW IS A MAGNITUDE, NOT A VECTOR. Travel time "
                      "cannot be computed from it.")
            miss = [k for k in PNAMES if got[k] is None]
            print("     conditions: " + ", ".join(
                "%s=%.4g (%s)" % (k, got[k], src[k])
                for k in PNAMES if got[k] is not None))
            if miss:
                print("     NOT FOUND: %s. Give them on the page, or they go "
                      "in as zero." % ", ".join(miss))
            for line in notes.get(name, []):
                print("     NOTE: %s" % line)
            print()
        except Exception as e:                            # noqa: BLE001
            failures.append((name, str(e)))
            print("%-26s REJECTED: %s\n" % (name, e))

    print("=" * 70)
    print("usable: %d of %d" % (len(recs), len(runs)))
    for n, why in failures:
        print("   rejected  %-24s %s" % (n, why))
    if args.inspect:
        print("\n--inspect was given, so nothing was written.")
        return
    if not recs:
        sys.exit("nothing to write")

    # One file holds one set of chemicals, so if the runs kept different sets
    # the file gets the ones they ALL have. Dropping a chemical that only some
    # runs wrote is the honest choice: padding the others with zeros would put
    # "this chemical was absent" and "this chemical was measured as zero" in
    # the same numbers, and nothing downstream could tell them apart.
    common = set.intersection(*[set(r["species"]) for r in recs])
    names = [n for n, _ in pairs if n in common]
    if not names:
        sys.exit("the collected runs share no chemical, so there is nothing "
                 "one file can hold. Collect them separately.")
    for r in recs:
        if r["species"] != names:
            keep = [r["species"].index(n) for n in names]
            r["conc"] = r["conc"][:, keep]
            notes.setdefault(r["name"], []).append(
                "kept %s; %s were dropped because other runs did not have them"
                % (", ".join(names),
                   ", ".join(n for n in r["species"] if n not in common)))
            r["species"] = list(names)

    # Snapshot counts can differ between runs too, for the same reason: a run
    # that converged sooner simply has fewer. Trim to the shortest rather than
    # rejecting the short ones, and say by how much.
    T = min(r["conc"].shape[0] for r in recs)
    for r in recs:
        if r["conc"].shape[0] > T:
            notes.setdefault(r["name"], []).append(
                "had %d snapshots, trimmed to the %d every run has"
                % (r["conc"].shape[0], T))
            r["conc"] = r["conc"][:T]
            r["tnorm"] = r["tnorm"][:T]
        if r.get("rate") is not None and r["rate"].shape[0] > T:
            r["rate"] = r["rate"][:T]

    # The reaction rate channels, on the same rule as the chemicals: the file
    # gets the ones every run that has rates at all agrees on. A run with no
    # rate files is not a problem and does not empty the field; its rows are
    # written as not-a-number, which is distinguishable from a measured zero.
    with_rate = [r for r in recs if r.get("rate") is not None]
    rate_names = []
    if with_rate:
        shared = set.intersection(*[set(r["rate_names"]) for r in with_rate])
        rate_names = [n for n in with_rate[0]["rate_names"] if n in shared]
        if not rate_names:
            print("   the runs that have reaction rates share no channel, so "
                  "no rate field is written")
        else:
            for r in with_rate:
                if r["rate_names"] != rate_names:
                    keep = [r["rate_names"].index(n) for n in rate_names]
                    r["rate"] = r["rate"][:, keep]
                    notes.setdefault(r["name"], []).append(
                        "kept the rate channels %s; %s were dropped because "
                        "other runs did not have them"
                        % (", ".join(rate_names),
                           ", ".join(n for n in r["rate_names"]
                                     if n not in shared)))
                    r["rate_names"] = list(rate_names)
        without = [r["name"] for r in recs if r.get("rate") is None]
        if without and rate_names:
            print("   %d run(s) have no reaction rates and are stored as "
                  "blank in that field: %s"
                  % (len(without), ", ".join(without[:6])
                     + (" ..." if len(without) > 6 else "")))

    import h5py
    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, "dataset.h5")
    S, G = len(recs), len(mats)
    C = recs[0]["conc"].shape[1]

    with h5py.File(outp, "w") as h:
        gg = h.create_group("geom")
        gg.create_dataset("gid", data=np.arange(G, dtype=np.int32))
        gg.create_dataset("material", data=np.stack(mats), compression="gzip")
        gg.create_dataset("gdf", data=np.stack(gdfs), compression="gzip")
        gg.create_dataset("edt", data=np.stack(edts), compression="gzip")

        # ---- the flow descriptors ---------------------------------------------
        # MIS and UPRM depend only on the geometry, so they are computed here, once
        # per rock, rather than on every training run. --no-flow-features skips them
        # if the extra second per rock matters; add_flow_features.py can put them in
        # afterwards without recollecting anything.
        if not getattr(args, "no_flow_features", False):
            try:
                import flow_features as _ff
                _buf = int(getattr(args, "flow_buffer", 10))
                _mis, _up, _e2, _pores = [], [], [], []
                print("  computing MIS and UPRM for %d rocks (buffer %d)" % (G, _buf))
                for _m in mats:
                    _p = _ff.pore_mask_from_material(_m, pore_code)
                    _pores.append(_p)
                    _f = _ff.all_features(_p, buf=_buf)
                    _mis.append(_f["mis"]); _up.append(_f["uprm"])
                    _e = _ff.distance_transform_edt(_p).astype(np.float32)
                    _e2.append(_e * _e)
                _mis = np.stack(_mis); _up = np.stack(_up); _e2 = np.stack(_e2)
                _mmu, _msd = _ff.zscore_stats(list(_mis))
                _umu, _usd = _ff.zscore_stats(list(_up))
                _all = np.concatenate([_e2[i][_pores[i]] for i in range(G)])
                _lo, _hi = float(_all.min()), float(_all.max())
                if _hi <= _lo:
                    _hi = _lo + 1.0
                _dw2 = np.zeros_like(_e2)
                for i in range(G):
                    _pp = _pores[i]
                    _d = np.zeros(_pp.shape, np.float32)
                    _d[_pp] = np.clip((_e2[i][_pp] - _lo) / (_hi - _lo), 0, 1)
                    _dw2[i] = _d
                for _n, _a, _at in (("mis", _mis, {"mis_mu": _mmu, "mis_sd": _msd}),
                                    ("uprm", _up, {"uprm_mu": _umu, "uprm_sd": _usd}),
                                    ("dw2", _dw2, {"dw2_min": _lo, "dw2_max": _hi})):
                    _d = gg.create_dataset(_n, data=_a.astype(np.float32),
                                           compression="gzip")
                    for _k, _v in _at.items():
                        _d.attrs[_k] = float(_v)
                    _d.attrs["buffer"] = _buf
                h.attrs["flow_features_buffer"] = _buf
                print("  MIS mean %.3f sd %.3f | UPRM mean %.3f sd %.3f (voxels)"
                      % (_mmu, _msd, _umu, _usd))
            except Exception as _e:
                # A missing descriptor is recorded, never invented. The dataset is
                # still perfectly usable without the flow pipeline.
                print("  NOTE: flow descriptors not written (%s). Add them later with "
                      "add_flow_features.py." % _e)

        sg = h.create_group("samples")
        sg.create_dataset("geom_index",
                          data=np.array([r["gid"] for r in recs], np.int32))
        sg.create_dataset("run_id", data=np.arange(S, dtype=np.int32))
        sg.create_dataset("params", data=np.array(
            [[0.0 if r["got"][k] is None else r["got"][k] for k in PNAMES]
             for r in recs], np.float32))
        sg.create_dataset("t_norm",
                          data=np.array([r["tnorm"] for r in recs], np.float32))

        allc = np.stack([r["conc"] for r in recs])
        scale = np.maximum(np.abs(allc).reshape(S * T, C, -1).max((0, 2)),
                           1e-12).astype(np.float32)
        d = sg.create_dataset("conc", shape=(S, T, C) + shape, dtype=np.float16,
                              compression="gzip", chunks=(1, 1, 1) + shape)
        for i, r in enumerate(recs):
            d[i] = (r["conc"] / scale[None, :, None, None, None]).astype(np.float16)
        sg.attrs["conc_scale"] = scale

        if rate_names:
            nr = len(rate_names)
            allr = np.stack([r["rate"] for r in with_rate])
            rscale = np.maximum(
                np.abs(allr).reshape(len(with_rate) * T, nr, -1).max((0, 2)),
                1e-12).astype(np.float32)
            dr = sg.create_dataset("rate", shape=(S, T, nr) + shape,
                                   dtype=np.float16, compression="gzip",
                                   chunks=(1, 1, 1) + shape)
            for i, r in enumerate(recs):
                if r.get("rate") is None:
                    # not measured, and not the same thing as measured zero
                    dr[i] = np.float16("nan")
                else:
                    dr[i] = (r["rate"]
                             / rscale[None, :, None, None, None]).astype(np.float16)
            sg.attrs["rate_scale"] = rscale
            h.attrs["reactions"] = np.array([n.encode() for n in rate_names])
            h.attrs["rate_channel_source"] = (
                "the array names inside the rate .vti, which the solver took "
                "from the kinetics header").encode()

        if any(r["vel"] is not None for r in recs):
            dv = sg.create_dataset("velocity", shape=(S, 3) + shape,
                                   dtype=np.float16, compression="gzip",
                                   chunks=(1, 3) + shape)
            for i, r in enumerate(recs):
                if r["vel"] is not None:
                    dv[i] = r["vel"].astype(np.float16)

        write_grid(h, recs, names)
        write_ancillary(h, recs, notes)

        # A csv is only read when one is asked for by name, which is off by
        # default. Without one there is nothing for this group to hold, and an
        # empty group with a run name in it is just a second copy of something
        # ancillary/ already has.
        rows = [r["row"] for r in recs]
        prov = None
        if any(x is not None for x in rows) and not args.no_provenance:
            prov = h.create_group("provenance")
        if prov is not None and all(x is not None for x in rows):
            for k in rows[0]:
                try:
                    prov.create_dataset(k, data=np.array(
                        [float(x[k]) for x in rows], np.float64))
                except ValueError:
                    prov.create_dataset(k, data=np.array(
                        [str(x[k]).encode() for x in rows]))
        if prov is not None:
            prov.attrs["condition_sources"] = json.dumps(
                {r["name"]: r["src"] for r in recs}).encode()

        write_inputs(h, recs, notes)

        h.attrs["species"] = np.array([n.encode() for n in names])
        h.attrs["species_role"] = np.array([b"dissolved"] * len(names))
        h.attrs["param_names"] = np.array([k.encode() for k in PNAMES])
        h.attrs["shape"] = np.array(shape, np.int32)
        h.attrs["n_samples"] = S
        h.attrs["n_geometries"] = G
        h.attrs["mode"] = "steady" if T == 1 else "transient"
        h.attrs["n_times"] = T
        h.attrs["dimension"] = 2 if int(shape[2]) == 1 else 3
        h.attrs["structure_evolves"] = False
        # The voxel size, and the unit it is in, read out of the input files.
        # A grid of a hundred voxels means nothing without it.
        sp, unit = [np.nan, np.nan, np.nan], ""
        for r in recs:
            d = {p: v for p, v, _ in r["xml"]["rows"]}
            for i, ax in enumerate(("dx", "dy", "dz")):
                for p, v in d.items():
                    if p.endswith("/domain/" + ax):
                        sp[i] = _num_or_nan(v)
            for p, v in d.items():
                if p.endswith("/domain/unit"):
                    unit = (v or "").strip()
            if np.isfinite(sp[0]):
                break
        if np.isfinite(sp[0]):
            if not np.isfinite(sp[1]):
                sp[1] = sp[0]
            if not np.isfinite(sp[2]):
                sp[2] = sp[0]
        h.attrs["spacing"] = np.array(sp, np.float64)
        h.attrs["spacing_unit"] = (unit or "not stated").encode()
        h.attrs["param_units"] = np.array(
            [b"none", b"none", b"none", b"none", b"none", b"none"])
        h.attrs["source"] = b"collected by collect_foreign_complab.py"
        h.attrs["velocity_magnitude_only_runs"] = int(
            sum(1 for r in recs if r["vmag_only"]))

    print("\nwrote %s" % outp)
    print("   %d run(s) over %d distinct rock(s), %d chemical(s), %d snapshot(s)"
          % (S, G, C, T))
    if rate_names:
        print("   %d reaction rate channel(s): %s"
              % (len(rate_names), ", ".join(rate_names)))
    print("   %.1f MB" % (os.path.getsize(outp) / 1e6))
    if G < 2:
        print("\n   NOTE: every run shares one rock, so whole rocks cannot be "
              "held out and this file cannot be trained on honestly. It is "
              "fine for looking at.")


if __name__ == "__main__":
    main()
