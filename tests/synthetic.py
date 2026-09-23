#!/usr/bin/env python3
"""
synthetic.py -- build a tiny dataset and a tiny checkpoint, with no simulation.

Every test in this folder needs a file that looks like a real dataset and a
file that looks like a real checkpoint.  Producing them by running the
simulator would take minutes and would make the tests depend on the physics
being right, which is a separate question.  These two builders write the same
layout `tools/dataset_reader.py` expects, filled with smooth analytic fields,
in well under a second.

    make_dataset(path, ...)   -> the HDF5 layout, 2 geometries, 2 species
    make_checkpoint(path, ..) -> an untrained model saved the way train.py saves

Nothing here asserts anything.  It exists so the tests can.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "3D", "tools"), os.path.join(ROOT, "3D", "model")):
    if p not in sys.path:
        sys.path.insert(0, p)

SOLID, WALL, PORE = 0, 1, 2

SPECIES = ["Ac", "A"]
PNAMES_PE_DA = ["pe", "da"]
PNAMES_FULL = ["pe", "da_bio", "da_abio", "ks_ac_norm", "ks_a_norm", "y_norm"]


# =============================================================================
#  A ROCK THAT PERCOLATES, EVERY TIME
#  A test fixture that sometimes seals its inlet turns a real failure into an
#  intermittent one, which is the worst kind to chase. The geometry below is
#  drawn from a fixed seed and the inlet and outlet faces are forced open, so
#  the same call always gives the same connected rock.
# =============================================================================
def _geometry(n, seed):
    """A percolating pore space: a slab with a few spheres of solid in it."""
    rng = np.random.default_rng(seed)
    mat = np.full((n, n, n), PORE, np.uint8)
    for _ in range(4):
        c = rng.integers(2, n - 2, 3)
        r = rng.integers(2, max(3, n // 4))
        z, y, x = np.ogrid[:n, :n, :n]
        d2 = (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2
        mat[d2 <= r * r] = SOLID
    mat[:, :, 0] = PORE          # keep the inlet and outlet open
    mat[:, :, -1] = PORE
    # one voxel of wall on every solid face, which is what the real builder
    # records so the branch can see the reacting surface
    solid = mat == SOLID
    if solid.any():
        from scipy import ndimage
        shell = ndimage.binary_dilation(solid) & ~solid
        mat[shell] = WALL
    return mat


def _fields(mat, n, n_t, n_c, seed):
    """Smooth, bounded, monotone-in-x concentrations, and a plug velocity."""
    rng = np.random.default_rng(seed)
    pore = mat != SOLID
    x = np.linspace(0.0, 1.0, n)[None, None, :]
    conc = np.zeros((n_t, n_c, n, n, n), np.float32)
    for t in range(n_t):
        front = (t + 1) / float(n_t)
        for c in range(n_c):
            f = np.exp(-3.0 * c * x) * np.clip(front - x + 0.5, 0.0, 1.0)
            conc[t, c] = (f * pore).astype(np.float32)
    vel = np.zeros((3, n, n, n), np.float32)
    vel[2] = (1e-3 * pore).astype(np.float32)          # flow along the last axis
    vel += (1e-5 * rng.standard_normal(vel.shape)).astype(np.float32) * pore
    return conc, vel


# =============================================================================
#  A FILE IN THE PROJECT'S OWN LAYOUT
#  Written by hand rather than by the real builders on purpose: a fixture built
#  by the code under test cannot catch that code writing the wrong layout. Every
#  group, name and attribute here is what dataset_reader.py expects to find.
# =============================================================================
def make_dataset(path, n=12, n_geom=2, n_sets=2, n_t=2, params="pe_da",
                 species=None, with_velocity=True, seed=0):
    """Write a dataset in the project's HDF5 layout.  Returns its path."""
    import h5py
    from scipy import ndimage                                   # noqa: F401

    species = list(species or SPECIES)
    n_c = len(species)
    pnames = PNAMES_FULL if params == "full" else PNAMES_PE_DA

    mats = [_geometry(n, seed + g) for g in range(n_geom)]
    S = n_geom * n_sets
    conc = np.zeros((S, n_t, n_c, n, n, n), np.float32)
    vel = np.zeros((S, 3, n, n, n), np.float32)
    gi = np.zeros(S, np.int32)
    par = np.zeros((S, len(pnames)), np.float32)
    tn = np.zeros((S, n_t), np.float32)
    rid = np.arange(S, dtype=np.int32)

    k = 0
    for g in range(n_geom):
        for s in range(n_sets):
            c, v = _fields(mats[g], n, n_t, n_c, seed + 100 * g + s)
            conc[k], vel[k], gi[k] = c, v, g
            par[k, 0] = 1.0 + 9.0 * s                     # pe
            par[k, 1:] = 1.0 + s                          # da (or the rest)
            tn[k] = np.linspace(1.0 / n_t, 1.0, n_t)
            k += 1

    gdf = np.zeros((n_geom, n, n, n), np.float32)
    edt = np.zeros((n_geom, n, n, n), np.float32)
    for g, m in enumerate(mats):
        pore = (m != SOLID).astype(np.float32)
        # distance from the inlet face along the flow axis, masked to the pore
        ramp = np.arange(n, dtype=np.float32)[None, None, :]
        gdf[g] = ramp * pore
        edt[g] = ndimage.distance_transform_edt(pore).astype(np.float32)

    with h5py.File(path, "w") as h:
        gg = h.create_group("geom")
        gg.create_dataset("material", data=np.stack(mats))
        gg.create_dataset("gdf", data=gdf)
        gg.create_dataset("edt", data=edt)
        gg.create_dataset("gid", data=np.arange(n_geom, dtype=np.int32))
        sg = h.create_group("samples")
        sg.create_dataset("conc", data=conc)
        sg.create_dataset("geom_index", data=gi)
        sg.create_dataset("params", data=par)
        sg.create_dataset("t_norm", data=tn)
        sg.create_dataset("run_id", data=rid)
        sg.create_dataset("settled", data=np.ones(S, np.uint8))
        if with_velocity:
            sg.create_dataset("velocity", data=vel)
        sg.attrs["conc_scale"] = np.ones(n_c, np.float32)
        h.attrs["species"] = np.array([s.encode() for s in species])
        h.attrs["param_names"] = np.array([s.encode() for s in pnames])
        h.attrs["param_layout"] = params
        h.attrs["da_column"] = "da_bio" if params == "pe_da" else ""
        h.attrs["n_samples"] = S
        h.attrs["shape"] = np.array([n, n, n], np.int32)
    return path


# =============================================================================
#  A CHECKPOINT THAT LOADS
#  epochs defaults to 0, so this trains nothing. The tests below are about
#  shapes, keys and recorded fields; a fixture that trained would make them slow
#  and would not make them stricter.
# =============================================================================
def make_checkpoint(path, data, species=None, epochs=0, seed=0):
    """Save an UNTRAINED model the way train.py saves a trained one.

    The point of the tests that use this is the wiring, not the accuracy: a
    checkpoint with random weights exercises exactly the same load, build and
    write path that a trained one does.
    """
    import h5py
    import torch
    from deeponet_model import PRT_DeepONet3D

    torch.manual_seed(seed)
    with h5py.File(data) as h:
        pnames = [s.decode() for s in h.attrs["param_names"]]
        all_sp = [s.decode() for s in h.attrs["species"]]
        n = int(h.attrs["shape"][0])
        n_t = int(h["samples"]["t_norm"].shape[1])
    sp = species or all_sp[0]
    trunk_dim = 5 if n_t > 1 else 4

    model = PRT_DeepONet3D(in_channels=1, n_params=len(pnames),
                           trunk_in_dim=trunk_dim, grid=(n, n, n))
    ck = {"model": model.state_dict(),
          "species": sp,
          "param_names": pnames,
          "in_channels": 1,
          "trunk_in_dim": trunk_dim,
          "grid": [n, n, n],
          "epochs": epochs,
          "args": {"distance": "gdf", "with_velocity": False,
                   "flow_proxy": False, "dim_free": False,
                   "geom_features": False, "flow_mode": "tau",
                   "keep_geometry_channel": False, "u_floor": 0.01,
                   "velocity_informed": "off"},
          "with_time": trunk_dim == 5,
          "n_times": n_t,
          "trunk_cols": (["x", "y", "z"] + (["t"] if trunk_dim == 5 else [])
                         + ["gdf"]),
          "branch_ch": ["material"],
          "param_layout": "full" if len(pnames) == 6 else "pe_da",
          "da_column": "" if len(pnames) == 6 else "da_bio",
          "conc_scale": [1.0]}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(ck, path)
    return path


if __name__ == "__main__":
    import tempfile
    d = tempfile.mkdtemp()
    p = make_dataset(os.path.join(d, "s.h5"))
    c = make_checkpoint(os.path.join(d, "c.pt"), p)
    print("wrote %s and %s" % (p, c))
