#!/usr/bin/env python3
"""
test_03_data.py — files written by this project can be read back correctly.

WHAT THIS IS ABOUT
    Every result this project produces lands in a file: an HDF5 dataset, an npz
    geometry, a PyTorch checkpoint. A round trip test writes one, reads it back,
    and checks that what came out is what went in.

    That sounds trivial. It is not. The three faults it catches are all silent:

      1  a value that changes on the way through, because it was stored in a
         type that cannot hold it
      2  a key written under one name and read under another, so a field is
         quietly missing and treated as absent
      3  a scaling applied on write and not undone on read, so every number is
         off by a constant nobody notices

    None of these raise. All of them ruin a result.

NOTHING IS SIMULATED. Everything is written to a temporary folder and deleted.
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PORE, SOLID, open_channel, two_channels, TOOLS, run   # noqa: E402

import h5py                                                        # noqa: E402
import flow_features as ff                                         # noqa: E402


class TempFolder(unittest.TestCase):
    """Every test here gets a fresh folder and it is removed afterwards."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="prt_tests_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)


# =============================================================================
#  BLOCK 1.  HDF5
# =============================================================================
class HDF5RoundTrip(TempFolder):
    """Write, close, reopen, compare."""

    def test_an_array_survives_a_round_trip_exactly(self):
        p = os.path.join(self.d, "t.h5")
        a = np.arange(60, dtype=np.float32).reshape(3, 4, 5) * 0.123
        with h5py.File(p, "w") as h:
            h.create_dataset("g/a", data=a, compression="gzip")
        with h5py.File(p, "r") as h:
            back = np.asarray(h["g/a"])
        # Exactly equal, not close. gzip is lossless; if this ever becomes
        # merely close, something is converting the type on the way through.
        self.assertTrue(np.array_equal(a, back))

    def test_attributes_survive_too(self):
        """The scaling constants ride along as attributes beside the arrays,
        so this is not a detail: it is how a map in voxels becomes the number
        the network is given."""
        p = os.path.join(self.d, "t.h5")
        with h5py.File(p, "w") as h:
            ds = h.create_dataset("g/a", data=np.zeros((2, 2), np.float32))
            ds.attrs["mis_mu"] = 5.161643981933594
            ds.attrs["buffer"] = 10
        with h5py.File(p, "r") as h:
            self.assertAlmostEqual(float(h["g/a"].attrs["mis_mu"]),
                                   5.161643981933594, places=12)
            self.assertEqual(int(h["g/a"].attrs["buffer"]), 10)

    def test_float16_storage_is_lossy_and_we_know_by_how_much(self):
        """Concentrations are stored as float16 to keep the file small. That IS
        lossy, and the test says how lossy rather than pretending otherwise:
        about three decimal digits. Anything needing more must not use it.
        """
        p = os.path.join(self.d, "t.h5")
        a = np.linspace(0.0, 1.0, 101).astype(np.float32)
        with h5py.File(p, "w") as h:
            h.create_dataset("a", data=a.astype(np.float16))
        with h5py.File(p, "r") as h:
            back = np.asarray(h["a"], np.float32)
        self.assertLess(float(np.abs(a - back).max()), 1e-3)


class DatasetLayout(TempFolder):
    """The layout every writer in this project must produce.

    Training, evaluation and prediction all assume this shape. A writer that
    quietly produced a different one would be found by whichever of them ran
    first, minutes later, as a missing key.
    """

    REQUIRED_GEOM = ("material", "gdf", "edt")
    REQUIRED_SAMPLES = ("conc", "params", "geom_index")

    def _tiny_dataset(self, path, with_velocity=True, with_features=False):
        """A dataset in the project's layout, built here by hand.

        Built by hand ON PURPOSE. Calling a builder would test the builder;
        this tests that the LAYOUT is what everything downstream expects.
        """
        G, S, T, C = 2, 4, 3, 2
        shape = (10, 8, 1)
        mats = np.stack([open_channel(10, 8)[:, :, None],
                         two_channels(10, 8)[:, :, None]]).astype(np.uint8)
        with h5py.File(path, "w") as h:
            h.attrs["shape"] = np.array(shape, np.int32)
            h.attrs["dimension"] = 2
            h.attrs["pore_code"] = PORE
            h.attrs["param_names"] = np.array([b"pe", b"da"])
            g = h.create_group("geom")
            g.create_dataset("material", data=mats)
            g.create_dataset("gdf", data=np.zeros((G,) + shape, np.float32))
            g.create_dataset("edt", data=np.zeros((G,) + shape, np.float32))
            if with_features:
                for n, mu, sd in (("mis", 2.0, 1.0), ("uprm", 1.5, 1.0)):
                    ds = g.create_dataset(n, data=np.ones((G,) + shape, np.float32))
                    ds.attrs[n + "_mu"] = mu
                    ds.attrs[n + "_sd"] = sd
                    ds.attrs["buffer"] = 0
                d2 = g.create_dataset("dw2", data=np.zeros((G,) + shape, np.float32))
                d2.attrs["dw2_min"] = 0.0
                d2.attrs["dw2_max"] = 1.0
            s = h.create_group("samples")
            s.create_dataset("geom_index", data=np.array([0, 0, 1, 1], np.int32))
            s.create_dataset("params", data=np.array(
                [[1.0, 0.5], [10.0, 0.5], [1.0, 5.0], [10.0, 5.0]], np.float32))
            s.create_dataset("conc", data=np.zeros((S, T, C) + shape, np.float16))
            s.create_dataset("t_norm", data=np.linspace(0, 1, T, dtype=np.float32))
            if with_velocity:
                s.create_dataset("velocity", data=np.zeros((S, 3) + shape, np.float16))
        return path

    def test_the_layout_is_readable(self):
        p = self._tiny_dataset(os.path.join(self.d, "d.h5"))
        with h5py.File(p, "r") as h:
            for k in self.REQUIRED_GEOM:
                self.assertIn(k, h["geom"], "geom/%s is missing" % k)
            for k in self.REQUIRED_SAMPLES:
                self.assertIn(k, h["samples"], "samples/%s is missing" % k)

    def test_every_run_points_at_a_rock_that_exists(self):
        """geom_index is the join between the two halves of the file. An index
        past the end of geom/material is a dangling reference, and it would be
        found at training time as an IndexError inside a data loader."""
        p = self._tiny_dataset(os.path.join(self.d, "d.h5"))
        with h5py.File(p, "r") as h:
            n_rocks = h["geom/material"].shape[0]
            idx = np.asarray(h["samples/geom_index"])
        self.assertTrue((idx >= 0).all() and (idx < n_rocks).all())

    def test_the_parameter_table_is_named(self):
        """A column of numbers with no names is a column nobody can read back
        in six months, and it is how the wrong number reached the wrong input
        once already."""
        p = self._tiny_dataset(os.path.join(self.d, "d.h5"))
        with h5py.File(p, "r") as h:
            names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in h.attrs["param_names"]]
            self.assertEqual(len(names), h["samples/params"].shape[1],
                             "one name per column, no more and no fewer")

    def test_the_reader_can_open_it(self):
        """The real Dataset class, on a file this test wrote."""
        from dataset_reader import PRT3DDataset
        p = self._tiny_dataset(os.path.join(self.d, "d.h5"))
        ds = PRT3DDataset(p, n_points=64)
        self.assertGreater(len(ds), 0)
        b1, b2, tr, y = ds[0]
        self.assertEqual(b1.shape[0], ds.in_channels,
                         "the branch got a different channel count than declared")
        self.assertEqual(tr.shape[1], ds.trunk_dim,
                         "the trunk got a different width than declared")
        self.assertEqual(tr.shape[0], y.shape[0],
                         "one target per query point")

    def test_the_reader_refuses_a_file_that_cannot_serve_the_request(self):
        """Asking for the predicted velocity from a file that has none must
        fail IMMEDIATELY with a sentence saying what to run, not minutes later
        inside a data loader."""
        from dataset_reader import PRT3DDataset
        p = self._tiny_dataset(os.path.join(self.d, "d.h5"))
        with self.assertRaises(ValueError) as cm:
            PRT3DDataset(p, n_points=64, velocity_informed="predicted")
        self.assertIn("predict_velocity", str(cm.exception),
                      "the error should name the command that fixes it")

    def test_the_pore_size_maps_are_read_when_present(self):
        from dataset_reader import PRT3DDataset
        p = self._tiny_dataset(os.path.join(self.d, "d.h5"), with_features=True)
        ds = PRT3DDataset(p, n_points=64, velocity_informed="simulated",
                          geom_features=True)
        self.assertIn("mis", ds.branch_ch)
        self.assertIn("uprm", ds.branch_ch)
        b1, _b2, _tr, _y = ds[0]
        self.assertEqual(b1.shape[0], ds.in_channels)


# =============================================================================
#  BLOCK 2.  NPZ GEOMETRIES
# =============================================================================
class GeometryFiles(TempFolder):

    def test_a_geometry_survives_a_round_trip(self):
        p = os.path.join(self.d, "g.npz")
        m = two_channels()
        np.savez_compressed(p, material=m[:, :, None])
        back = np.load(p)["material"][:, :, 0]
        self.assertTrue(np.array_equal(m, back))

    def test_the_pore_fraction_is_unchanged(self):
        """The single number a wrong material convention would change."""
        m = two_channels()
        before = float((m == PORE).mean())
        p = os.path.join(self.d, "g.npz")
        np.savez_compressed(p, material=m)
        after = float((np.load(p)["material"] == PORE).mean())
        self.assertEqual(before, after)


# =============================================================================
#  BLOCK 3.  ADDING THE DESCRIPTORS TO A FILE THAT ALREADY EXISTS
# =============================================================================
class AddFlowFeatures(TempFolder):
    """add_flow_features.py changes a dataset IN PLACE. That is the riskiest
    kind of operation in this project, so it gets its own tests."""

    def _dataset(self):
        p = os.path.join(self.d, "d.h5")
        mats = np.stack([two_channels(10, 8)[:, :, None],
                         open_channel(10, 8)[:, :, None]]).astype(np.uint8)
        with h5py.File(p, "w") as h:
            h.attrs["shape"] = np.array((10, 8, 1), np.int32)
            h.attrs["dimension"] = 2
            h.attrs["pore_code"] = PORE
            g = h.create_group("geom")
            g.create_dataset("material", data=mats)
            g.create_dataset("gdf", data=np.zeros(mats.shape, np.float32))
        return p

    def test_it_adds_three_arrays_and_touches_nothing_else(self):
        p = self._dataset()
        with h5py.File(p, "r") as h:
            before = np.asarray(h["geom/material"])
        rc, out = run([sys.executable, os.path.join(TOOLS, "add_flow_features.py"),
                       "--data", p, "--buffer", "0"], timeout=300)
        self.assertEqual(rc, 0, out[-500:])
        with h5py.File(p, "r") as h:
            for k in ("mis", "uprm", "dw2"):
                self.assertIn(k, h["geom"], "geom/%s was not written" % k)
            self.assertTrue(np.array_equal(before, np.asarray(h["geom/material"])),
                            "the geometry was modified, and must not be")
            self.assertIn("gdf", h["geom"], "an existing array was removed")

    def test_the_scaling_constants_are_stored_beside_the_arrays(self):
        p = self._dataset()
        run([sys.executable, os.path.join(TOOLS, "add_flow_features.py"),
             "--data", p, "--buffer", "0"], timeout=300)
        with h5py.File(p, "r") as h:
            self.assertIn("mis_mu", h["geom/mis"].attrs)
            self.assertIn("mis_sd", h["geom/mis"].attrs)
            self.assertIn("uprm_mu", h["geom/uprm"].attrs)
            self.assertIn("buffer", h["geom/mis"].attrs)
            self.assertEqual(int(h["geom/mis"].attrs["buffer"]), 0,
                             "the buffer that was used must be recorded")

    def test_it_refuses_to_overwrite_without_being_told_to(self):
        """Recomputing with a different buffer silently would leave every
        checkpoint trained on this file scaled against constants that no longer
        exist."""
        p = self._dataset()
        run([sys.executable, os.path.join(TOOLS, "add_flow_features.py"),
             "--data", p, "--buffer", "0"], timeout=300)
        rc, out = run([sys.executable, os.path.join(TOOLS, "add_flow_features.py"),
                       "--data", p, "--buffer", "2"], timeout=300)
        self.assertNotEqual(rc, 0, "a second run should have been refused")
        self.assertIn("--force", out, "the refusal should say how to override it")

    def test_dry_run_writes_nothing(self):
        p = self._dataset()
        rc, _ = run([sys.executable, os.path.join(TOOLS, "add_flow_features.py"),
                     "--data", p, "--buffer", "0", "--dry-run"], timeout=300)
        self.assertEqual(rc, 0)
        with h5py.File(p, "r") as h:
            self.assertNotIn("mis", h["geom"], "--dry-run wrote to the file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
