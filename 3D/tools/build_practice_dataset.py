#!/usr/bin/env python3
"""
build_practice_dataset.py — build a tiny but REAL dataset_reader.h5 for testing the three
feature switches without touching the cluster.

Everything in it is genuine: the geometries are thresholded Gaussian fields with
a percolation check, the velocity fields come from the same D3Q19 Stokes solver
as tools/demo/stokes_lbm.py, and the concentration fields are an actual
advection-diffusion-reaction solve.  It is small (24 x 16 x 16), not fake.

    python build_practice_dataset.py --out /tmp/test3d.h5 --n-geom 4 --n-sets 3
"""

import argparse, os, sys
import numpy as np
import h5py
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prtlb_3d import stokes_d3q19 as stokes                                  # noqa: E402
from prtlb_3d import (SOLID, WALL, PORE, solve_adr, check_physics,   # noqa: F401
                      keep_inlet_connected, assert_finite_distance)


# =============================================================================
#  BLOCK 1.  A REAL PORE STRUCTURE, SMALL
#
#  Gaussian random media thresholded at the porosity, which is the same
#  morphology build_geometry_3d.py makes, just tiny. That is what makes this a
#  practice dataset rather than a fake one: the numbers below are solved, not
#  invented, and the only thing shrunk is the grid.
# =============================================================================
def geometry(shape, phi, seed):
    rng = np.random.default_rng(seed)
    # mode="wrap" so no artificial correlation is introduced at the edges.
    f = ndimage.gaussian_filter(rng.standard_normal(shape), 2.0, mode="wrap")
    g = np.where(f > np.quantile(f, 1.0 - phi), PORE, SOLID).astype(np.uint8)
    g[:3] = g[-3:] = PORE       # three open layers at each end: inlet and outlet
    g[:, 0, :] = g[:, -1, :] = g[:, :, 0] = g[:, :, -1] = WALL
    # Keep what the inlet can reach, not what is biggest -- and do it AFTER
    # the side walls go on, because sealing a face can disconnect pore space
    # that was connected before it.
    g, _ = keep_inlet_connected(g)
    return g


# =============================================================================
#  BLOCK 2.  THE GEODESIC DISTANCE, BY NEIGHBOUR RELAXATION
#
#  Deliberately not scikit-fmm. Every dataset in this project is built this way,
#  so the practice file has to be built this way too, or a model trained on it
#  meets a slightly different distance field on the real data.
# =============================================================================
def geodesic(g):
    """Geodesic distance from the inlet through pore space, in voxels."""
    inf = np.float32(1e9)
    d = np.where(g == PORE, inf, np.nan).astype(np.float32)
    d[0][g[0] == PORE] = 0.0
    # An upper bound on how many sweeps a path can need, not a tuned number: a
    # geodesic path cannot be longer than a few times the domain length. The
    # loop breaks as soon as nothing changes, so the bound only ever stops a
    # pathological case from running forever.
    for _ in range(4 * g.shape[0]):
        prev = d.copy()
        for ax in range(3):
            for sh in (1, -1):
                nb = np.roll(d, sh, axis=ax)
                sl = [slice(None)] * 3
                sl[ax] = 0 if sh > 0 else -1
                nb[tuple(sl)] = inf
                d = np.where(g == PORE, np.fmin(d, np.nan_to_num(nb, nan=inf) + 1.0), np.nan)
        if np.nanmax(np.abs(np.nan_to_num(d - prev, nan=0.0))) < 1e-6:
            break
    return np.nan_to_num(d, nan=0.0).astype(np.float32)


def adr(g, vel, pe, da, n_t, n_species=2, steps=None, info=None):
    """The 3D transport solve. A thin wrapper on the one shared solver.

    THIS USED TO BE A SECOND, INDEPENDENT SOLVER, and it carried every bug
    that had already been found and fixed in the 2D one -- a periodic np.roll
    boundary that leaked inlet concentration into the outlet, D = 1/Pe so the
    simulated Peclet was nx times the one asked for, a hard step count of 60
    that stopped before the front entered the sample, a clip at 2.0 papering
    over the instability, and `C[1] *= 0.6` as the only per-species term, so
    species 0, 2 and 3 were byte-identical arrays.

    That last one mattered well beyond this practice file: build_transfer_set_2d_to_3d.py calls
    this function to build the 2D-to-3D transfer set behind switch B, so the
    transfer training data held one chemical replicated four times. The name
    check in train.py could not see it, because the NAMES were all different.
    """
    return solve_adr(g, vel, pe, da, n_t, n_species=n_species, steps=steps,
                     info=info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/test3d.h5")
    ap.add_argument("--n-geom", type=int, default=4)
    ap.add_argument("--n-sets", type=int, default=3)
    ap.add_argument("--n-times", type=int, default=3)
    ap.add_argument("--shape", type=int, nargs=3, default=[24, 16, 16])
    ap.add_argument("--stokes-iters", type=int, default=30000)
    a = ap.parse_args()

    shape = tuple(a.shape)
    # The same names every other writer uses. They were capitalised and spelled
    # differently here, and because the reader accepts both spellings nothing
    # ever complained -- but a dataset made here and one made by collect_complab_output.py
    # then disagreed on what their own columns were called, which is precisely
    # the sort of silent difference that turns into a wrong rate field later.
    # Channel 1 is the ACCEPTOR here, as it is in every other 2-species file
    # this project writes. It used to be "P", which meant channel 1 named a
    # different chemical depending on which script wrote the file -- and since
    # evaluate.py labels its columns from the DATASET while the network's
    # output heads come from the CHECKPOINT, a model trained on one and scored
    # on the other would report rmse_P for the head that learned the acceptor.
    species = ["Ac", "A"]
    pnames = ["pe", "da_bio", "da_abio", "ks_ac_norm", "ks_a_norm", "y_norm"]
    G, S = a.n_geom, a.n_geom * a.n_sets
    C, T = len(species), a.n_times

    mat = np.zeros((G,) + shape, np.uint8)
    gdf = np.zeros((G,) + shape, np.float32)
    edt = np.zeros((G,) + shape, np.float32)
    vels = []
    for i in range(G):
        phi = 0.35 + 0.15 * i / max(G - 1, 1)
        g = geometry(shape, phi, seed=100 + i)
        mat[i] = g
        gdf[i] = geodesic(g)
        bad_d = assert_finite_distance(gdf[i], g)
        if bad_d:
            raise SystemExit("geometry %d: %s" % (i, bad_d))
        edt[i] = ndimage.distance_transform_edt(g == PORE)
        vels.append(stokes(g, nit=a.stokes_iters))
        print("  geom %d  phi=%.3f  gdf_max=%.1f  umean=%.2e"
              % (i, (g == PORE).mean(), gdf[i].max(),
                 np.sqrt((vels[i] ** 2).sum(0))[g == PORE].mean()), flush=True)

    rng = np.random.default_rng(0)
    conc = np.zeros((S, T, C) + shape, np.float32)
    vel = np.zeros((S, 3) + shape, np.float32)
    gi = np.zeros(S, np.int32); par = np.zeros((S, len(pnames)), np.float32)
    tn = np.zeros((S, T), np.float32)
    settled = np.ones(S, bool)
    n_bad = 0
    k = 0
    for i in range(G):
        for j in range(a.n_sets):
            pe = float(10 ** rng.uniform(-0.5, 1.5))
            da = float(10 ** rng.uniform(-1, 1))
            gi[k] = i
            par[k] = [pe, da, da * 0.5, 0.1, 0.1, 0.05]
            vel[k] = vels[i]
            nfo = {}
            # The solver returns the times it ACTUALLY integrated to. This
            # used to overwrite them with a straight 0..1 ramp, which told the
            # trunk that the snapshots were evenly spaced in time when they
            # were not.
            conc[k], tn[k] = adr(mat[i], vels[i], pe, da, T, C, info=nfo)
            settled[k] = bool(nfo.get("converged", True))
            for msg in check_physics(conc[k], mat[i]):
                print("     PHYSICS run %d (Pe=%.2f Da=%.2f): %s"
                      % (k, pe, da, msg), flush=True)
                n_bad += 1
            k += 1
    print("  %d samples, conc range %.3f .. %.3f" % (S, conc.min(), conc.max()))
    if n_bad:
        raise SystemExit("%d physics complaint(s): this file is not usable as "
                         "training data, and writing it would hide that." % n_bad)
    if not settled.all():
        print("  %d of %d runs stopped before settling; recorded in "
              "samples/settled" % (int((~settled).sum()), S))

    scale = np.maximum(conc.reshape(S * T * C, -1).max(1).reshape(S, T, C).max((0, 1)), 1e-6)
    with h5py.File(a.out, "w") as h:
        h.attrs["n_samples"] = S
        h.attrs["shape"] = np.array(shape, np.int32)
        h.attrs["species"] = np.array([s.encode() for s in species])
        h.attrs["param_names"] = np.array([s.encode() for s in pnames])
        gg = h.create_group("geom")
        gg.create_dataset("gid", data=np.arange(G, dtype=np.int32))
        gg.create_dataset("material", data=mat)
        gg.create_dataset("gdf", data=gdf)
        gg.create_dataset("edt", data=edt)
        sg = h.create_group("samples")
        sg.create_dataset("geom_index", data=gi)
        sg.create_dataset("run_id", data=np.arange(S, dtype=np.int32))
        sg.create_dataset("params", data=par)
        sg.create_dataset("t_norm", data=tn)
        sg.create_dataset("settled", data=settled)
        d = sg.create_dataset("conc", data=conc.astype(np.float16))
        d.attrs["conc_scale"] = scale.astype(np.float32)
        sg.attrs["conc_scale"] = scale.astype(np.float32)
        sg.create_dataset("velocity", data=vel.astype(np.float16))
    print("wrote %s  (%d geometries, %d samples, %d snapshots, %d species)"
          % (a.out, G, S, T, C))


if __name__ == "__main__":
    main()
