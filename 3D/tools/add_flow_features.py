#!/usr/bin/env python3
# =============================================================================
# CHANGED FROM THE 2D VERSION
#
#   WHERE IT CAME FROM
#     Nothing. This file has no counterpart in their release.
#     Their pipeline reads MIS from a pre-built cache (mis_map.npz) and UPRM
#     from split.pt, both produced by code that was not published.
#     github.com/hjunglab/PRT-DeepONet   branch/folder: velocity-informed
#
#   WHY IT EXISTS
#     Our datasets are one HDF5 file per campaign, so the maps belong INSIDE
#     that file rather than in a cache beside it. The collectors write them
#     for new campaigns; this adds them to a file you already have, in place.
#     Recollecting a real campaign means reading a few hundred thousand VTI
#     files again for two arrays that depend only on the geometry.
#
#   ONE DESIGN DECISION WORTH KNOWING
#     The z-scoring constants are stored as ATTRIBUTES beside the arrays
#     rather than applied to them. The maps stay in voxels, so they can be
#     read and looked at and compared with a picture of the rock, and the
#     reader applies the scaling. Applying it here would leave a file whose
#     MIS map is in units that only make sense next to the file it came from.
# =============================================================================
"""Add the flow descriptors to a dataset that was collected without them.

The collectors write MIS and UPRM for new campaigns. This adds them to a file that
already exists, in place, without recollecting anything, because recollecting a real
campaign means reading a few hundred thousand VTI files again for two arrays that
depend only on the geometry.

    geom/mis    (G, nx, ny, nz) float32, with mis_mu and mis_sd on the dataset
    geom/uprm   (G, nx, ny, nz) float32, with uprm_mu and uprm_sd
    geom/dw2    (G, nx, ny, nz) float32, scaled to [0, 1], with the range used

The z scoring constants are written as ATTRIBUTES beside the arrays rather than being
applied to them. The arrays stay in voxel units, so they can be read and looked at, and
the reader applies the scaling. Applying it here instead would leave a file whose MIS
map is in units that only make sense next to the file it came from.

Run it like this.

    python add_flow_features.py --data work/demo/dataset.h5
    python add_flow_features.py --data work/demo/dataset.h5 --buffer 5 --force

What you get back. Three new datasets inside the file you named, and a line per rock
while it works. Nothing else on disk. With --dry-run it reports what it would do and
writes nothing.

Worth knowing. The buffer matters. MIS is computed on the interior and painted outward
into the inlet and outlet padding, because that padding is fully open and a sphere
placed in it would report a radius the rock never has. --buffer must match the padding
your campaign actually used: the published 2D release pads 10 voxels at each end, our
own generator defaults to 5, and a hand built campaign may have none. Pass 0 when there
is no padding. The value is stored on the file so a later reader can see it.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import h5py
except ImportError:                                                    # pragma: no cover
    sys.exit("add_flow_features.py needs h5py. Run: python gui/install_requirements.py")

import flow_features as ff


def add_features(path, buffer=10, force=False, dry_run=False, verbose=True):
    mode = "r" if dry_run else "r+"
    with h5py.File(path, mode) as h:
        if "geom/material" not in h:
            raise SystemExit("%s has no geom/material, so there is no geometry to "
                             "compute anything from." % path)
        # Refusing to overwrite is not tidiness. The scaling constants written
        # beside these arrays are what the reader subtracts, so silently
        # recomputing them with a different buffer would leave every checkpoint
        # trained on this file scaled against numbers that no longer exist.
        present = [k for k in ("mis", "uprm", "dw2") if k in h["geom"]]
        if present and not force and not dry_run:
            raise SystemExit(
                "This file already has geom/%s. Pass --force to recompute, which is "
                "the right thing to do only if the buffer was wrong the first time."
                % ", geom/".join(present))

        mat = np.asarray(h["geom/material"])
        # READ, never guessed. CompLaB and this project number their materials
        # differently, and a wrong pore code inverts the rock silently.
        pore_code = int(h.attrs["pore_code"]) if "pore_code" in h.attrs else None
        dim = int(h.attrs.get("dimension", 3 if mat.shape[-1] > 1 else 2))
        G = len(mat)
        # A 2D campaign is stored as a 3D array one voxel deep, so that one
        # reader serves both. The descriptors are computed on the real 2D image
        # and put back into that flat axis, rather than on a 64 x 148 x 1 volume
        # where every voxel touches a wall in z and every MIS radius is 0.5.
        squeeze = (dim == 2 and mat.ndim == 4 and mat.shape[-1] == 1)
        if verbose:
            print("%s" % path)
            print("  %d rocks, grid %s, %dD, buffer %d"
                  % (G, tuple(mat.shape[1:]), dim, buffer))
            if present:
                print("  already present: %s (will be replaced)" % ", ".join(present))
        if dry_run:
            print("  --dry-run, nothing written")
            return None

        out_shape = mat.shape
        # float32 throughout. These are radii in voxels, so the largest value in
        # a 64-cube is about 32: nothing here needs double precision, and the
        # arrays are the biggest thing this script writes.
        mis = np.zeros(out_shape, np.float32)
        uprm = np.zeros(out_shape, np.float32)
        e2 = np.zeros(out_shape, np.float32)
        pores = []
        t0 = time.time()
        for i in range(G):
            m = mat[i][..., 0] if squeeze else mat[i]
            pore = ff.pore_mask_from_material(m, pore_code)
            pores.append(pore)
            f = ff.all_features(pore, buf=buffer)
            e = ff.distance_transform_edt(pore).astype(np.float32)
            if squeeze:
                mis[i, ..., 0] = f["mis"]
                uprm[i, ..., 0] = f["uprm"]
                e2[i, ..., 0] = e * e
            else:
                mis[i] = f["mis"]
                uprm[i] = f["uprm"]
                e2[i] = e * e
            # Every 25 rocks, and always the last. A 3000-rock campaign is 40
            # minutes of silence otherwise, which reads as a hang.
            if verbose and ((i + 1) % 25 == 0 or i == G - 1):
                print("  %d/%d  %.0fs" % (i + 1, G, time.time() - t0), flush=True)

        # One scaling for the whole campaign, from every rock in the file. The reader
        # applies it; a split that fitted its own constants would make a training rock
        # and a held-out rock of the same width look different.
        mis_mu, mis_sd = ff.zscore_stats([mis[i] for i in range(G)])
        up_mu, up_sd = ff.zscore_stats([uprm[i] for i in range(G)])
        # dw2 is scaled by its absolute range, not z scored, because zero has to
        # keep meaning "on the wall". Over PORE voxels only: including the solid
        # would put a huge spike at zero into the range and squash the rest.
        allp = np.concatenate([e2[i][..., 0][pores[i]] if squeeze else e2[i][pores[i]]
                               for i in range(G)])
        lo, hi = float(allp.min()), float(allp.max())
        if hi <= lo:
            hi = lo + 1.0                    # a one-voxel-wide rock: avoid 0/0
        dw2 = np.zeros_like(e2)
        for i in range(G):
            p = pores[i]
            if squeeze:
                d = np.zeros(p.shape, np.float32)
                d[p] = np.clip((e2[i][..., 0][p] - lo) / (hi - lo), 0, 1)
                dw2[i, ..., 0] = d
            else:
                d = np.zeros(p.shape, np.float32)
                d[p] = np.clip((e2[i][p] - lo) / (hi - lo), 0, 1)
                dw2[i] = d

        g = h["geom"]
        for name, arr, attrs in (
                ("mis", mis, {"mis_mu": mis_mu, "mis_sd": mis_sd}),
                ("uprm", uprm, {"uprm_mu": up_mu, "uprm_sd": up_sd}),
                ("dw2", dw2, {"dw2_min": lo, "dw2_max": hi})):
            if name in g:
                del g[name]     # --force only; the guard above refuses otherwise
            # One chunk per rock. Training reads one rock at a time and never a
            # slice across rocks, so this is the access pattern; chunking the
            # other way makes every read pull the whole array through gzip.
            d = g.create_dataset(name, data=arr, compression="gzip",
                                 chunks=(1,) + arr.shape[1:])
            for k, v in attrs.items():
                d.attrs[k] = float(v)
            # The buffer rides along with the data. Six months from now the only
            # way to know whether these maps measured the rock or the padding is
            # if the file says so itself.
            d.attrs["buffer"] = int(buffer)
        g["mis"].attrs["how_to_read_this"] = (
            b"MIS, the maximum inscribed sphere radius, in VOXELS. The radius of the "
            b"largest sphere that fits in the pore space and contains this voxel. "
            b"Local: how wide the pore is here. Multiply by nothing; subtract mis_mu "
            b"and divide by mis_sd to get what the network is given.")
        g["uprm"].attrs["how_to_read_this"] = (
            b"UPRM, the upstream constrained pore radius, in VOXELS. The radius of the "
            b"largest sphere the inlet could deliver here, set by the narrowest throat "
            b"on the best path. NOT local: two identically shaped pockets differ if one "
            b"sits behind a bottleneck.")
        g["dw2"].attrs["how_to_read_this"] = (
            b"The squared distance to the nearest wall, already scaled to [0,1] over "
            b"dw2_min..dw2_max. A no-slip profile is parabolic in the wall distance, so "
            b"the square is what the trunk wants.")
        h.attrs["flow_features_buffer"] = int(buffer)
        h.attrs["flow_features_source"] = b"3D/tools/add_flow_features.py"

        if verbose:
            print("\n  wrote geom/mis, geom/uprm, geom/dw2")
            print("  MIS   mean %.3f  sd %.3f  (voxels)" % (mis_mu, mis_sd))
            print("  UPRM  mean %.3f  sd %.3f  (voxels)" % (up_mu, up_sd))
            print("  dw2   scaled over [%.1f, %.1f] (voxels squared)" % (lo, hi))
            frac = float(np.mean([uprm[i][pores[i][..., None] if squeeze else pores[i]].mean()
                                  / max(mis[i][pores[i][..., None] if squeeze else pores[i]].mean(), 1e-9)
                                  for i in range(G)]))
            print("  UPRM is on average %.2f of MIS across the pore space, which is the "
                  "part\n  a convolution cannot work out for itself." % frac)
        return dict(mis_mu=mis_mu, mis_sd=mis_sd, uprm_mu=up_mu, uprm_sd=up_sd,
                    dw2_min=lo, dw2_max=hi)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Add MIS, UPRM and the squared wall distance to an existing dataset.")
    ap.add_argument("--data", required=True, help="dataset.h5 to add them to, in place")
    ap.add_argument("--buffer", type=int, default=10,
                    help="open buffer voxels at each end of the flow axis. MUST match "
                         "the padding your campaign used: 10 for the published 2D set, "
                         "5 for our own generator, 0 for none. Default 10.")
    ap.add_argument("--force", action="store_true", help="recompute if already present")
    ap.add_argument("--dry-run", action="store_true", help="report and write nothing")
    a = ap.parse_args(argv)
    add_features(a.data, a.buffer, a.force, a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
