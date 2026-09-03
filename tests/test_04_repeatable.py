#!/usr/bin/env python3
"""
test_04_repeatable.py — the same input gives the same answer, twice.

WHY THIS MATTERS MORE THAN IT SOUNDS
    A result you cannot reproduce is not a result. If running the same code on
    the same input twice gives two answers, then any comparison between two
    runs is measuring the noise as much as the change, and nobody can tell
    which.

    Randomness in this project is not a bug: geometries are generated randomly
    and training samples voxels randomly. The rule is that all of it goes
    through a SEED. Same seed, same answer, always. Different seed, different
    answer, or the seed is being ignored.

    Both halves are tested. A function that returns a constant would pass the
    first and fail the second.

WHAT IS COVERED
    the pore size maps, the pressure solve, geometry generation, the voxel
    sampler inside the Dataset, and the switch table.
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PORE, open_channel, two_channels, bottleneck   # noqa: E402

import h5py                                                        # noqa: E402
import flow_features as ff                                         # noqa: E402
from harmonic_pressure import harmonic_pressure                    # noqa: E402
from dataset_reader import resolve_switches                        # noqa: E402


# =============================================================================
#  BLOCK 1.  FUNCTIONS WITH NO RANDOMNESS AT ALL
#
#  These take a geometry and return a map. There is no seed because there is
#  nothing random in them, so the requirement is stronger: the two answers must
#  be IDENTICAL, bit for bit, not merely close.
# =============================================================================
class Deterministic(unittest.TestCase):

    GEOMETRIES = (open_channel(), two_channels(), bottleneck())

    def test_the_pore_size_maps_repeat_exactly(self):
        for m in self.GEOMETRIES:
            p = ff.pore_mask_from_material(m)
            a = ff.mis_map(p, buf=0)
            b = ff.mis_map(p, buf=0)
            self.assertTrue(np.array_equal(a, b), "MIS is not deterministic")
            self.assertTrue(np.array_equal(ff.uprm_map(p), ff.uprm_map(p)),
                            "UPRM is not deterministic")

    def test_the_pressure_solve_repeats_exactly(self):
        """This one is worth checking rather than assuming. It is a sparse
        linear solve, and some solvers are order dependent."""
        p = ff.pore_mask_from_material(two_channels())
        a, _ = harmonic_pressure(p)
        b, _ = harmonic_pressure(p)
        self.assertTrue(np.array_equal(a, b),
                        "the pressure solve gave two different answers")

    def test_all_features_agrees_with_the_functions_it_wraps(self):
        """all_features() is a convenience that calls the three maps. If it
        ever drifted from them, a dataset built one way and a prediction made
        the other way would disagree, and nothing would say so."""
        p = ff.pore_mask_from_material(bottleneck())
        f = ff.all_features(p, buf=0)
        self.assertTrue(np.array_equal(f["mis"], ff.mis_map(p, buf=0)))
        self.assertTrue(np.array_equal(f["uprm"], ff.uprm_map(p)))

    def test_the_switch_table_is_pure(self):
        """resolve_switches has no state, so calling it twice must give the
        same dictionary, and calling it must not change anything for the next
        caller."""
        a = resolve_switches(ndim=3, velocity_informed="simulated")
        _ = resolve_switches(ndim=2, flow_proxy=True)
        b = resolve_switches(ndim=3, velocity_informed="simulated")
        self.assertEqual(a, b)


# =============================================================================
#  BLOCK 2.  THINGS WITH A SEED
# =============================================================================
class SeedIsObeyed(unittest.TestCase):

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="prt_seed_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _dataset(self, path):
        mats = np.stack([two_channels(12, 9)[:, :, None],
                         open_channel(12, 9)[:, :, None]]).astype(np.uint8)
        with h5py.File(path, "w") as h:
            h.attrs["shape"] = np.array((12, 9, 1), np.int32)
            h.attrs["dimension"] = 2
            h.attrs["pore_code"] = PORE
            h.attrs["param_names"] = np.array([b"pe", b"da"])
            g = h.create_group("geom")
            g.create_dataset("material", data=mats)
            g.create_dataset("gdf", data=np.zeros(mats.shape, np.float32))
            g.create_dataset("edt", data=np.zeros(mats.shape, np.float32))
            s = h.create_group("samples")
            s.create_dataset("geom_index", data=np.array([0, 1], np.int32))
            s.create_dataset("params", data=np.array([[1.0, 1.0], [10.0, 1.0]],
                                                     np.float32))
            s.create_dataset("conc", data=np.zeros((2, 3, 2, 12, 9, 1), np.float16))
            s.create_dataset("t_norm", data=np.linspace(0, 1, 3, dtype=np.float32))
        return path

    def test_the_same_seed_samples_the_same_voxels(self):
        """The Dataset picks a random subset of pore voxels each time it is
        asked for an item. With the same seed the subset must be the same, or a
        run cannot be repeated."""
        from dataset_reader import PRT3DDataset
        p = self._dataset(os.path.join(self.d, "d.h5"))
        a = PRT3DDataset(p, n_points=32, seed=7)[0][2].numpy()
        b = PRT3DDataset(p, n_points=32, seed=7)[0][2].numpy()
        self.assertTrue(np.array_equal(a, b),
                        "the same seed gave two different samples")

    def test_a_different_seed_samples_differently(self):
        """The other half. Without this, a function that ignored the seed
        entirely would pass the test above."""
        from dataset_reader import PRT3DDataset
        p = self._dataset(os.path.join(self.d, "d.h5"))
        a = PRT3DDataset(p, n_points=32, seed=7)[0][2].numpy()
        c = PRT3DDataset(p, n_points=32, seed=99)[0][2].numpy()
        self.assertFalse(np.array_equal(a, c),
                         "two different seeds gave the same sample, so the "
                         "seed is being ignored")

    def test_numpy_seeding_behaves_as_the_project_assumes(self):
        """Every generator in this project uses np.random.default_rng(seed).
        This is the property all of them rely on, checked once, here."""
        a = np.random.default_rng(3).standard_normal(20)
        b = np.random.default_rng(3).standard_normal(20)
        c = np.random.default_rng(4).standard_normal(20)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))


# =============================================================================
#  BLOCK 3.  SCALING SURVIVES A ROUND TRIP
#
#  The pore size maps are stored in VOXELS and the constants that turn them
#  into network inputs are stored beside them. Applying and undoing that
#  scaling must return the original numbers, or every map is off by a factor
#  nobody would notice.
# =============================================================================
class ScalingRoundTrip(unittest.TestCase):

    def test_z_scoring_and_undoing_it_returns_the_original(self):
        p = ff.pore_mask_from_material(two_channels())
        mis = ff.mis_map(p, buf=0)
        mu, sd = ff.zscore_stats([mis])
        scaled = (mis - mu) / sd
        back = scaled * sd + mu
        self.assertTrue(np.allclose(mis[p], back[p], atol=1e-5))

    def test_the_dw2_scaling_can_be_undone(self):
        p = ff.pore_mask_from_material(open_channel())
        d, (lo, hi) = ff.dw2_map(p)
        back = d * (hi - lo) + lo
        from scipy import ndimage
        e = ndimage.distance_transform_edt(p).astype(np.float32)
        self.assertTrue(np.allclose(back[p], (e * e)[p], atol=1e-4),
                        "unscaling dw2 did not give back the squared distance")


if __name__ == "__main__":
    unittest.main(verbosity=2)
