#!/usr/bin/env python3
"""Add the flow descriptors to a dataset that was collected without them.

NEW IN THE FLOW VERSION
    The 2D release has no flow descriptors, so there was nothing to add and no
    file like this. What makes it necessary here is arithmetic: a real campaign is
    a few hundred thousand .vti files, and recollecting one to gain two arrays that
    depend only on the geometry costs hours of reading to compute something that
    takes seconds. So the descriptors are added in place, to the file that already
    exists, and the original collection is left alone.

The collectors write MIS and UPRM for new campaigns. This adds them to a file that
already exists, in place.

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


# =============================================================================
#  THE ONE PASS OVER THE FILE
#  Opened "r+", so this writes into the dataset in place. The three arrays it
#  adds depend only on the geometry, never on a run's conditions, which is why
#  they can be added afterwards at all: no simulation has to be repeated.
# =============================================================================
def add_features(path, buffer=10, force=False, dry_run=False, verbose=True):
    # "r" for a dry run, so a file cannot be modified by a command whose whole
    # purpose is to report what a real run would do.
    mode = "r" if dry_run else "r+"
    with h5py.File(path, mode) as h:
        if "geom/material" not in h:
            raise SystemExit("%s has no geom/material, so there is no geometry to "
                             "compute anything from." % path)
        present = [k for k in ("mis", "uprm", "dw2") if k in h["geom"]]
        if present and not force and not dry_run:
            raise SystemExit(
                "This file already has geom/%s. Pass --force to recompute, which is "
                "the right thing to do only if the buffer was wrong the first time."
                % ", geom/".join(present))

        # Read whole: the descriptors need every rock before any can be scaled,
        # so there is nothing to gain from reading this one lazily.
        mat = np.asarray(h["geom/material"])
        pore_code = int(h.attrs["pore_code"]) if "pore_code" in h.attrs else None
        dim = int(h.attrs.get("dimension", 3 if mat.shape[-1] > 1 else 2))
        G = len(mat)
        # A 2D campaign is stored with a third axis of 1. Squeeze it for the
        # feature routines, which work in the dimension the rock actually has.
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

        # ---------------------------------------------------------------
        #  ROCK BY ROCK
        #  The whole campaign's descriptors are held in memory here, because
        #  the scaling below has to see every rock before any of it can be
        #  written. That is the one real cost of this file: three float32
        #  arrays the size of the geometry stack.
        # ---------------------------------------------------------------
        out_shape = mat.shape
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
            # SQUARED below, not here: a no-slip profile is parabolic in the
            # wall distance, so the square is the quantity the trunk wants.
            e = ff.distance_transform_edt(pore).astype(np.float32)
            if squeeze:
                mis[i, ..., 0] = f["mis"]
                uprm[i, ..., 0] = f["uprm"]
                e2[i, ..., 0] = e * e
            else:
                mis[i] = f["mis"]
                uprm[i] = f["uprm"]
                e2[i] = e * e
            if verbose and ((i + 1) % 25 == 0 or i == G - 1):
                print("  %d/%d  %.0fs" % (i + 1, G, time.time() - t0), flush=True)

        # One scaling for the whole campaign, from every rock in the file. The reader
        # applies it; a split that fitted its own constants would make a training rock
        # and a held-out rock of the same width look different.
        mis_mu, mis_sd = ff.zscore_stats([mis[i] for i in range(G)])
        up_mu, up_sd = ff.zscore_stats([uprm[i] for i in range(G)])
        allp = np.concatenate([e2[i][..., 0][pores[i]] if squeeze else e2[i][pores[i]]
                               for i in range(G)])
        lo, hi = float(allp.min()), float(allp.max())
        if hi <= lo:
            hi = lo + 1.0        # a rock one voxel wide everywhere: do not divide by 0
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

        # ---------------------------------------------------------------
        #  WRITING, WITH THE SCALING BESIDE THE ARRAY
        #  Every array carries the constants it was scaled with and a
        #  how_to_read_this string. A descriptor whose scaling lives only in
        #  the script that wrote it is unusable the moment somebody opens the
        #  file on their own, which is most of the times it gets opened.
        # ---------------------------------------------------------------
        g = h["geom"]
        for name, arr, attrs in (
                ("mis", mis, {"mis_mu": mis_mu, "mis_sd": mis_sd}),
                ("uprm", uprm, {"uprm_mu": up_mu, "uprm_sd": up_sd}),
                ("dw2", dw2, {"dw2_min": lo, "dw2_max": hi})):
            if name in g:
                del g[name]
            d = g.create_dataset(name, data=arr, compression="gzip",
                                 chunks=(1,) + arr.shape[1:])
            for k, v in attrs.items():
                d.attrs[k] = float(v)
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


# =============================================================================
#  THE COMMAND LINE
#  --buffer is the one argument that has to be right and cannot be guessed: it
#  is how many voxels of open space the campaign padded onto each end of the
#  flow axis, and UPRM is measured from that face.
# =============================================================================
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
