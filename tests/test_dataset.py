#!/usr/bin/env python3
"""
test_dataset.py -- the dataset layer, on a synthetic file, with no simulation
and no training.

What each test would catch

  target_shape       the target growing a species axis back
  species_default    the default silently changing which chemical is learned
  species_select     --species not reaching the reader
  species_unknown    a typo in a species name being accepted and quietly
                     training on the wrong field
  param_layouts      the two-column and six-column layouts stopping to coexist,
                     which is what lets a new dataset be Pe and Da while an old
                     one still trains
  trunk_dim          a steady file getting a time column, or a transient one
                     losing it
  split_by_geometry  train and test sharing a geometry, which turns the
                     held-out error into a training error
  species_of_ckpt    an older checkpoint that stored a list of species no
                     longer loading
  scatter            the volume writer refusing a flat prediction

The synthetic dataset is built by synthetic.py in this folder, in about a
second, from analytic fields.  Nothing here checks the physics; it checks that
the plumbing carries the right shapes.
"""

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "3D", "tools"), os.path.join(ROOT, "3D", "model"),
          HERE):
    sys.path.insert(0, p)

import numpy as np                                              # noqa: E402

import synthetic                                                # noqa: E402
from dataset_reader import (PRT3DDataset, dataset_kwargs_from_ckpt,   # noqa: E402
                            scatter_to_volume, species_of_ckpt,
                            split_by_geometry)


# =============================================================================
#  WHAT THIS PROTECTS
#  Three things that have each been wrong once: the target regrowing a species
#  axis, --species not reaching the reader, and a split that shares a geometry
#  between train and test. The last is the dangerous one, because it makes every
#  held-out number better and nothing anywhere says so.
# =============================================================================
class Dataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="prt_ds_")
        cls.two = synthetic.make_dataset(os.path.join(cls.dir, "pe_da.h5"))
        cls.six = synthetic.make_dataset(os.path.join(cls.dir, "full.h5"),
                                         params="full")
        cls.steady = synthetic.make_dataset(os.path.join(cls.dir, "steady.h5"),
                                            n_t=1)

    def test_target_shape(self):
        ds = PRT3DDataset(self.two, n_points=64)
        b1, b2, tk, y = ds[0]
        self.assertEqual(len(y.shape), 1, "one field, so no species axis")
        self.assertEqual(tuple(y.shape), (64,))
        self.assertEqual(tuple(b2.shape), (2,))      # pe and da, not six
        self.assertEqual(tuple(tk.shape), (64, ds.trunk_dim))
        self.assertEqual(b1.shape[0], ds.in_channels)

    def test_species_default(self):
        ds = PRT3DDataset(self.two, n_points=16)
        self.assertEqual(ds.species, ["Ac", "A"])
        self.assertEqual(ds.target_species, "Ac",
                         "the default is the first species in the file")
        self.assertEqual(ds.species_index, 0)

    def test_species_select(self):
        ds = PRT3DDataset(self.two, n_points=16, species="A")
        self.assertEqual(ds.target_species, "A")
        self.assertEqual(ds.species_index, 1)
        # Same seed on both, so the sampled voxels are identical and the only
        # thing that can differ is which field the target came from.
        a = PRT3DDataset(self.two, n_points=16, species="Ac", seed=3)[0][3]
        b = PRT3DDataset(self.two, n_points=16, species="A", seed=3)[0][3]
        self.assertFalse(np.allclose(np.asarray(a), np.asarray(b)),
                         "selecting a different species must change the target")

    def test_species_unknown(self):
        with self.assertRaises(ValueError):
            PRT3DDataset(self.two, n_points=16, species="not_a_species")

    def test_param_layouts(self):
        self.assertEqual(PRT3DDataset(self.two, n_points=16).param_names,
                         ["pe", "da"])
        # The six-column layout has to keep working beside the two-column one:
        # both exist in real files and the branch is sized from the file.
        six = PRT3DDataset(self.six, n_points=16)
        self.assertEqual(len(six.param_names), 6)
        self.assertEqual(tuple(six[0][1].shape), (6,),
                         "a six-column dataset still feeds six inputs")

    def test_trunk_dim(self):
        self.assertEqual(PRT3DDataset(self.two, n_points=16).trunk_dim, 5,
                         "transient: x, y, z, t, gdf")
        self.assertEqual(PRT3DDataset(self.steady, n_points=16).trunk_dim, 4,
                         "steady: x, y, z, gdf")

    # -------------------------------------------------------------------
    #  THE TRUNK'S COLUMNS
    #  Width first, then range. A trunk of the right width carrying a column in
    #  raw voxels instead of 0 to 1 trains without complaint and transfers to
    #  nothing, which is the failure these two catch between them.
    # -------------------------------------------------------------------
    def test_trunk_values_are_bounded(self):
        tk = np.asarray(PRT3DDataset(self.two, n_points=256)[0][2])
        self.assertTrue(np.isfinite(tk).all())
        self.assertGreaterEqual(tk.min(), -1e-6)
        self.assertLessEqual(tk.max(), 1.0 + 1e-6,
                             "every trunk column is normalised into [0, 1]")

    def test_split_by_geometry(self):
        tr, te = split_by_geometry(self.two, frac=0.5, seed=0)
        self.assertTrue(tr.size and te.size)
        a = PRT3DDataset(self.two, indices=tr, n_points=16)
        b = PRT3DDataset(self.two, indices=te, n_points=16)
        self.assertFalse(set(np.asarray(a.geom_index)[tr].tolist())
                         & set(np.asarray(b.geom_index)[te].tolist()),
                         "train and test must not share a geometry")

    # -------------------------------------------------------------------
    #  READING A CHECKPOINT BACK
    #  evaluate.py and predict.py rebuild the dataset configuration from the
    #  checkpoint rather than from flags typed again. These check that what was
    #  written is what comes back, including for older files that predate a
    #  field and have to keep working.
    # -------------------------------------------------------------------
    def test_species_of_ckpt(self):
        self.assertEqual(species_of_ckpt({"species": "Ac"}), "Ac")
        self.assertEqual(species_of_ckpt({"species": ["A", "Ac"]}), "A",
                         "an older checkpoint stored a list; take the first")
        self.assertIsNone(species_of_ckpt({}))

    def test_dataset_kwargs_from_ckpt(self):
        ck = {"species": "A", "grid": [12, 12, 12],
              "args": {"distance": "gdf", "with_velocity": False,
                       "flow_proxy": False, "dim_free": False,
                       "geom_features": False,
                       "velocity_informed": "off"}}
        kw, cfg = dataset_kwargs_from_ckpt(ck)
        self.assertEqual(kw.get("species"), "A",
                         "the species must travel with the checkpoint")
        ds = PRT3DDataset(self.two, n_points=16, **kw)
        self.assertEqual(ds.target_species, "A")

    def test_scatter_to_volume_takes_a_flat_field(self):
        ds = PRT3DDataset(self.two, n_points=64)
        _, _, tk, y = ds[0]
        n = int(ds.shape[0])
        pts = (np.asarray(tk)[:, :3] * (n - 1)).round().astype(int)
        vol = scatter_to_volume(np.asarray(y), pts, (n, n, n))
        self.assertEqual(vol.shape, (n, n, n))
        self.assertTrue(np.isfinite(vol[pts[:, 0], pts[:, 1], pts[:, 2]]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
