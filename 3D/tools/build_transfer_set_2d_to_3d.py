#!/usr/bin/env python3
"""
build_transfer_set_2d_to_3d.py — SWITCH B.  Turn 2D pore domains into 3D training data.

THE IDEA
--------
3D runs cost 5.5 to 16 hours each, so 3D geometries are the scarce resource.
2D domains are not: Jung's PRT-DeepONet release ships 2000 of them, and a 2D
solve is roughly fifty times cheaper than the 3D one.

Take a 2D domain and EXTRUDE it along z.  The resulting 3D medium is prismatic:
nothing varies in z, so the exact 3D solution IS the 2D solution repeated.  This
is not an approximation and not a domain-adaptation trick.  An extruded 2D
domain is a genuine, if unusually simple, member of the 3D problem class, and
its solution is exact.

WHAT TRANSFERS AND WHAT DOES NOT
--------------------------------
The reaction physics transfers completely: rate laws, the Peclet and Damkohler
response, the shape of a front, Monod saturation.  None of it contains a
dimension.

The TOPOLOGY does not transfer at all.  A thresholded Gaussian field percolates
near porosity 0.20 in 3D and near 0.59 in 2D, and a 3D pore network offers far
more routes around an obstacle, so tortuosity and dead-end fraction at equal
porosity are entirely different.  An encoder trained only on 2D has learned the
wrong connectivity statistics.

That split maps onto the DeepONet factorisation: transfer the TRUNK and the
parameter branch, retrain the GEOMETRY branch.  Hence train.py's --init-from
plus --freeze-trunk.  It also caps how much 2D you should mix in: an extruded
domain has zero z-tortuosity, so at a high mixing fraction the network learns
the shortcut "nothing ever happens in z".  Keep --transfer-2d-frac at or below
about 0.3.

NO Z-WALLS
----------
The extruded volume is written with the z faces OPEN (pore), not walled.  With
walls at z = 0 and z = nz-1 the true 3D flow develops a Poiseuille profile
across z and the extruded field would no longer be the exact solution -- the
error would be a whole profile, not a boundary layer.  Open z faces keep the
sample exactly prismatic and the physics exact.  --z-walls forces the other
choice if you would rather match the 3D geometry convention than stay exact.

WHERE THE FIELDS COME FROM
--------------------------
By default we take only the GEOMETRIES from the 2D source and generate the flow
and concentration fields ourselves, with the same D3Q19 Stokes solver and the
same advection-diffusion-reaction integrator used elsewhere in this repository.
That is deliberate: Jung's solution fields are for HIS chemistry and HIS
parameter ranges, not ours, and mixing two different reaction systems into one
training set would teach the network nothing useful.  Extruded 2D solves are
cheap, so regenerating is affordable.

USAGE
-----
    # Jung's domains, our chemistry
    python build_transfer_set_2d_to_3d.py --jung-dir /path/to/PRT-DeepONet-main \\
                        --out /data/train2d.h5 --target-shape 128 64 64

    # no 2D source yet: synthesise 2D domains of the same morphology
    python build_transfer_set_2d_to_3d.py --synthetic 40 --out /data/train2d.h5 \\
                        --target-shape 128 64 64

Then:
    python train.py --data 3d.h5 --transfer-2d /data/train2d.h5 --dim-free
"""

import argparse, glob, os, sys
import numpy as np
import h5py
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_practice_dataset import geodesic, stokes, adr                      # noqa: E402
from prtlb_3d import (check_physics, keep_inlet_connected,          # noqa: E402
                      assert_finite_distance)

SOLID, WALL, PORE = 0, 1, 2

# Jung's PRT-DeepONet convention, measured from the release:
#   0 = solid, 1 = pore, 2 = interface        (ours: 0 = solid, 1 = wall, 2 = pore)
#
# Interface voxels are a one-voxel skin and only 6.7 percent of the domain.
# Counting them as pore gives porosity 0.493 on Domain_Monod.npz, which sits
# inside the 0.486-0.565 range measured on the .dat domains; counting them as
# solid gives 0.426, which does not.  So pore is the default, and --interface
# solid flips it.
JUNG_TO_COMPLAB = {0: SOLID, 1: PORE, 2: PORE}


# ---------------------------------------------------------------- readers ---
def _reshape_flat(a, nx_hint=148, ny_hint=64, quiet=False):
    """Reshape a flat domain array, DETECTING the index order rather than
    assuming it.

    This matters, and it is not hypothetical.  Measured on the actual release:

        Domains/domain_*.dat   ->  reshape(148, 64) C-order   1 pore cluster
        geometries/Domain_*.npz->  reshape(64, 148) then .T   1 pore cluster

    The two file formats use OPPOSITE orderings.  Applying the .dat convention
    to a .npz produces a domain with 81 disconnected pore clusters that fails
    the percolation test, so every npz domain would have been silently dropped
    with no error and no warning.

    We try both and keep whichever gives fewer connected pore components: a
    correctly ordered pore network is connected, a wrongly ordered one is
    shredded into stripes.
    """
    if a.ndim != 1:
        return a
    n = a.size
    cands = []
    if n % ny_hint == 0:
        cands.append(("C(nx,ny)", a.reshape(-1, ny_hint)))
    if n % nx_hint == 0:
        cands.append(("C(ny,nx).T", a.reshape(-1, nx_hint).T))
    if not cands:
        return None
    best, best_n, best_name = None, None, None
    for name, b in cands:
        _, k = ndimage.label(b != 0)
        if best_n is None or k < best_n:
            best, best_n, best_name = b, k, name
    if not quiet and len(cands) > 1:
        print("    index order detected: %s  (%d pore clusters)" % (best_name, best_n))
    return best


def _read_dat(p, nx_hint, ny_hint):
    """Fast integer-per-line reader.  np.loadtxt is ~20x slower and we may be
    reading three thousand of these."""
    with open(p, "rb") as f:
        raw = f.read()
    try:
        a = np.array(raw.split(), dtype=np.int16)
    except ValueError:
        return None
    return _reshape_flat(a, nx_hint, ny_hint, quiet=True)


def read_jung(directory, limit=None, nx_hint=148, ny_hint=64):
    """Load 2D domains from a PRT-DeepONet checkout.

    Reads Domains/domain_*.dat first, because the release ships three thousand
    of those and only three .npz.  Both formats are handled, each with its own
    index order detected per file.
    """
    out = []
    dats = sorted(glob.glob(os.path.join(directory, "**", "*.dat"), recursive=True))
    if dats:
        print("  scanning %d .dat domains" % len(dats))
    for p in dats:
        a = _read_dat(p, nx_hint, ny_hint)
        if a is not None and a.size > 1000:
            out.append(a.astype(np.int16))
        if limit and len(out) >= limit:
            return out
    for p in sorted(glob.glob(os.path.join(directory, "**", "*.npz"), recursive=True)):
        try:
            z = np.load(p)
        except Exception:
            continue
        for key in ("m", "material", "domain", "arr_0"):
            if key in z:
                a = np.asarray(z[key]).squeeze()
                print("  %s" % os.path.basename(p))
                a = _reshape_flat(a, nx_hint, ny_hint)
                if a is not None and a.ndim == 2 and a.size > 1000:
                    out.append(a.astype(np.int16))
                break
        if limit and len(out) >= limit:
            break
    return out[:limit] if limit else out


def synthetic_2d(n, shape2d, seed=0, phi_lo=0.55, phi_hi=0.85):
    """2D thresholded Gaussian fields, the same morphology family as our 3D set.

    Note the porosity range.  In 3D we sample from about 0.20 upward because
    that is where a thresholded Gaussian field stops percolating.  In 2D the
    threshold is near 0.59, so sampling the 3D range here would throw most
    domains away.  This is the topology gap that switch B has to work around,
    made concrete: the same morphology, the same generator, and a percolation
    threshold nearly three times higher.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        phi = phi_lo + (phi_hi - phi_lo) * rng.random()
        f = ndimage.gaussian_filter(rng.standard_normal(shape2d), 3.0, mode="wrap")
        g = np.where(f > np.quantile(f, 1.0 - phi), PORE, SOLID).astype(np.int16)
        g[:4] = g[-4:] = PORE
        out.append(g)
    return out


# ---------------------------------------------------------------- shaping ---
def to_complab(a, interface="pore"):
    """Map whatever code convention the source uses onto CompLaB's.

    Jung: 0 solid, 1 pore, 2 interface.   CompLaB: 0 solid, 1 wall, 2 pore.
    The tell is which code is in the majority among the non-solid voxels: in
    Jung's files that is 1, in ours it is 2.
    """
    u = set(np.unique(a).tolist())
    if u <= {0, 1, 2} and 1 in u and (a == 1).mean() > 0.25:
        m = dict(JUNG_TO_COMPLAB)
        m[2] = PORE if interface == "pore" else SOLID
        out = np.zeros_like(a)
        for k, v in m.items():
            out[a == k] = v
        return out.astype(np.uint8)
    return np.clip(a, 0, 2).astype(np.uint8)      # already CompLaB-coded


def resize2d(a, shape2d):
    """Nearest-neighbour resample of a label image."""
    if a.shape == tuple(shape2d):
        return a
    zoom = (shape2d[0] / a.shape[0], shape2d[1] / a.shape[1])
    return ndimage.zoom(a, zoom, order=0).astype(np.uint8)


def largest_cluster(g):
    lab, n = ndimage.label(g == PORE)
    if n <= 1:
        return g, n
    keep = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    g = g.copy()
    g[(g == PORE) & (lab != keep)] = SOLID
    return g, n


def percolates(g2):
    """Inlet-to-outlet connectivity of the 2D domain, before extrusion."""
    lab, _ = ndimage.label(g2 == PORE)
    a = set(np.unique(lab[0][g2[0] == PORE]).tolist())
    b = set(np.unique(lab[-1][g2[-1] == PORE]).tolist())
    return len(a & b - {0}) > 0


def extrude(g2, nz, z_walls=False):
    """(nx,ny) -> (nx,ny,nz).  z faces stay OPEN unless z_walls is set."""
    g = np.repeat(g2[:, :, None], nz, axis=2)
    g[:, 0, :] = g[:, -1, :] = WALL                # transverse walls in y, as in 3D
    if z_walls:
        g[:, :, 0] = g[:, :, -1] = WALL
    return g


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--jung-dir", help="a PRT-DeepONet-main checkout")
    src.add_argument("--synthetic", type=int, metavar="N",
                     help="synthesise N 2D domains instead of reading a source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-shape", type=int, nargs=3, default=[128, 64, 64],
                    help="must match the 3D dataset exactly, or train.py refuses")
    ap.add_argument("--n-sets", type=int, default=3, help="parameter sets per domain")
    ap.add_argument("--n-times", type=int, default=21)
    ap.add_argument("--n-species", type=int, default=4)
    ap.add_argument("--species", nargs="*", default=None,
                    help="species names; must match the 3D file")
    ap.add_argument("--stokes-iters", type=int, default=30000)
    ap.add_argument("--adr-steps", type=int, default=None,
                    help="LEAVE UNSET. The transport runs until the field stops "
                         "changing. The old default of 200 stopped before the "
                         "front had entered the sample, so the transfer set "
                         "was almost entirely zeros -- and a network trained "
                         "on it learns to predict zero.")
    ap.add_argument("--nz-solve", type=int, default=8,
                    help="solve the flow and transport on a THIN slab of this "
                         "many z layers, then tile the result up to the full nz. "
                         "Because the extruded medium is prismatic, nothing "
                         "varies in z and the tiled field is EXACT, not "
                         "interpolated. At the default 8 against nz=64 this makes "
                         "the whole ingest 8x cheaper, which is the difference "
                         "between a laptop job and a cluster job. Set equal to nz "
                         "to disable. Ignored when --z-walls is given, since walls "
                         "break z-invariance.")
    ap.add_argument("--z-walls", action="store_true",
                    help="close the z faces. Matches the 3D geometry convention "
                         "but makes the extruded solution only approximate.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--interface", choices=["pore", "solid"], default="pore",
                    help="what to do with Jung's code-2 interface voxels. 'pore' "
                         "(default) reproduces the porosity range of his .dat "
                         "domains; 'solid' treats the skin as grain surface.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    nx, ny, nz = a.target_shape
    species = a.species or (["Ac", "A", "P", "Bio"][:a.n_species])
    a.n_species = len(species)
    # collect_complab_output.py's names, so every writer in this project agrees and the rate
    # figures do not silently fall back to their defaults.
    pnames = ["pe", "da_bio", "da_abio", "ks_ac_norm", "ks_a_norm", "y_norm"]

    # ---- 1. get the 2D domains ------------------------------------------
    if a.jung_dir:
        raw = read_jung(a.jung_dir, limit=a.limit)
        if not raw:
            raise SystemExit(
                "found no 2D domains under %s.\n"
                "Expected Domain_*.npz with a flat uint8 array, or "
                "Domains/domain_*.dat with one integer per line.\n"
                "Use --synthetic N to proceed without a 2D source."
                % a.jung_dir)
        print("read %d 2D domains from %s" % (len(raw), a.jung_dir))
    else:
        raw = synthetic_2d(a.synthetic, (nx, ny), seed=a.seed)
        print("synthesised %d 2D domains" % len(raw))

    # ---- 2. clean, resample, percolation-check --------------------------
    doms, dropped = [], 0
    for g in raw:
        g = resize2d(to_complab(g, interface=a.interface), (nx, ny))
        g, _ = largest_cluster(g)
        if not percolates(g):
            dropped += 1
            continue
        doms.append(g)
    print("kept %d, dropped %d that do not percolate" % (len(doms), dropped))
    if not doms:
        raise SystemExit("no percolating domains survived")

    # ---- 3. extrude and generate the physics ----------------------------
    G = len(doms)
    S = G * a.n_sets
    T, C = a.n_times, a.n_species
    shape = (nx, ny, nz)

    nzs = nz if a.z_walls else max(2, min(int(a.nz_solve), nz))
    if nzs != nz:
        print("solving on a %d-layer slab and tiling to nz=%d (exact: the medium "
              "is prismatic)" % (nzs, nz))

    def tile_z(arr):
        """(..., nzs) -> (..., nz) by tiling. Exact for a z-invariant field."""
        if arr.shape[-1] == nz:
            return arr
        reps = int(np.ceil(nz / arr.shape[-1]))
        return np.concatenate([arr] * reps, axis=-1)[..., :nz]

    mat = np.zeros((G,) + shape, np.uint8)
    gdf = np.zeros((G,) + shape, np.float32)
    edt = np.zeros((G,) + shape, np.float32)
    mats_s, vels_s = [], []           # slab versions, used for the physics
    zvar = []
    for i, g2 in enumerate(doms):
        gs = extrude(g2, nzs, z_walls=a.z_walls)          # slab
        g = extrude(g2, nz, z_walls=a.z_walls)            # full, for the branch
        # extrude() seals the transverse faces with wall, which can DISCONNECT
        # pore space that was connected in 2D. The percolation check ran before
        # that, so it cannot see it. Re-check here, after the last operation
        # that changes connectivity, and against the inlet rather than against
        # size.
        g, cut = keep_inlet_connected(g)
        gs, cut_s = keep_inlet_connected(gs)
        if cut:
            print("     domain %d: %d pore voxels were cut off from the inlet "
                  "when the side walls went on; they are now solid."
                  % (i, cut), flush=True)
        mat[i] = g
        gdf[i] = geodesic(g)
        bad_d = assert_finite_distance(gdf[i], g)
        if bad_d:
            raise SystemExit("domain %d: %s" % (i, bad_d))
        edt[i] = ndimage.distance_transform_edt(g == PORE)
        v = stokes(gs, nit=a.stokes_iters)
        mats_s.append(gs); vels_s.append(v)
        # the extrusion is only exact if the solved field really is z-invariant;
        # measure it rather than assume it
        m = gs == PORE
        sp = np.sqrt((v ** 2).sum(0))
        zv = float(np.nanmax(np.abs(sp - sp.mean(2, keepdims=True))[m])
                   / max(sp[m].mean(), 1e-30)) if m.any() else 0.0
        zvar.append(zv)
        print("  domain %3d  phi=%.3f  gdf_max=%5.1f  z-variation %.2e"
              % (i, (g2 == PORE).mean(), gdf[i].max(), zv), flush=True)

    print("max z-variation across all domains: %.2e  (0 means the extrusion is exact)"
          % max(zvar))
    if max(zvar) > 1e-3 and not a.z_walls:
        print("WARNING: the flow field is not z-invariant. The extrusion is then "
              "not exact and these samples are approximate 3D data.")

    rng = np.random.default_rng(a.seed)
    conc = np.zeros((S, T, C) + shape, np.float32)
    vel = np.zeros((S, 3) + shape, np.float32)
    gi = np.zeros(S, np.int32); par = np.zeros((S, len(pnames)), np.float32)
    settled = np.ones(S, bool)
    n_bad = 0
    tn = np.zeros((S, T), np.float32)
    k = 0
    for i in range(G):
        for _ in range(a.n_sets):
            pe = float(10 ** rng.uniform(np.log10(0.3), np.log10(30)))
            da = float(10 ** rng.uniform(-1, 1))
            gi[k] = i
            par[k] = [pe, da, da, 0.1, 0.1, 0.05]
            vel[k] = tile_z(vels_s[i])
            nfo = {}
            cc, tt = adr(mats_s[i], vels_s[i], pe, da, T, C,
                         steps=a.adr_steps, info=nfo)
            conc[k] = tile_z(cc)
            # The times the solver actually integrated to, not a 0..1 ramp.
            tn[k] = tt
            settled[k] = bool(nfo.get("converged", True))
            for msg in check_physics(cc, mats_s[i]):
                print("     PHYSICS run %d (Pe=%.2f Da=%.2f): %s"
                      % (k, pe, da, msg), flush=True)
                n_bad += 1
            k += 1
        print("  physics for domain %d done (%d/%d samples)" % (i, k, S), flush=True)

    if n_bad:
        raise SystemExit("%d physics complaint(s): this transfer set is not "
                         "usable, and writing it would hide that." % n_bad)
    scale = np.maximum(conc.reshape(S * T * C, -1).max(1).reshape(S, T, C).max((0, 1)), 1e-6)
    with h5py.File(a.out, "w") as h:
        h.attrs["n_samples"] = S
        h.attrs["shape"] = np.array(shape, np.int32)
        h.attrs["species"] = np.array([s.encode() for s in species])
        h.attrs["param_names"] = np.array([s.encode() for s in pnames])
        h.attrs["source"] = b"extruded_2d"
        h.attrs["z_walls"] = bool(a.z_walls)
        h.attrs["max_z_variation"] = float(max(zvar))
        h.attrs["nz_solve"] = int(nzs)
        gg = h.create_group("geom")
        gg.create_dataset("gid", data=np.arange(G, dtype=np.int32))
        gg.create_dataset("material", data=mat, compression="gzip")
        gg.create_dataset("gdf", data=gdf, compression="gzip")
        gg.create_dataset("edt", data=edt, compression="gzip")
        sg = h.create_group("samples")
        sg.create_dataset("geom_index", data=gi)
        sg.create_dataset("run_id", data=np.arange(S, dtype=np.int32))
        sg.create_dataset("params", data=par)
        sg.create_dataset("t_norm", data=tn)
        sg.create_dataset("settled", data=settled)
        d = sg.create_dataset("conc", data=conc.astype(np.float16), compression="gzip")
        d.attrs["conc_scale"] = scale.astype(np.float32)
        sg.attrs["conc_scale"] = scale.astype(np.float32)
        sg.create_dataset("velocity", data=vel.astype(np.float16), compression="gzip")

    print("\nwrote %s" % a.out)
    print("  %d extruded 2D geometries, %d samples, grid %s, species %s"
          % (G, S, shape, species))
    print("  use with:  train.py --data <3d.h5> --transfer-2d %s --transfer-2d-frac 0.3"
          % a.out)


if __name__ == "__main__":
    main()
