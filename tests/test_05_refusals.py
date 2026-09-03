#!/usr/bin/env python3
"""
test_05_refusals.py — when something is wrong, does it say so, or carry on?

WHAT THIS IS ABOUT
    The worst kind of failure is not a crash. It is a run that finishes, writes
    a file, prints a number, and is wrong. Nobody looks for a fault they were
    not told about.

    So this project is written to REFUSE rather than guess, and every refusal
    is supposed to name the thing to fix. This file does the wrong thing on
    purpose, twelve different ways, and checks two things each time:

      1  it failed, rather than carrying on
      2  the message says what to do about it

    A guard that has quietly stopped firing looks exactly like a guard that is
    working, which is why they have to be tested by breaking things.

NOTHING HERE PRODUCES A USEFUL RESULT. Everything is meant to fail.
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import PORE, TOOLS, MODEL, open_channel, two_channels, run   # noqa: E402

import h5py                                                        # noqa: E402


def tiny_dataset(path, velocity=True):
    """A dataset in the project's layout, small and valid."""
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
        # The velocity is OPTIONAL here on purpose: half these tests need a
        # file that has one and half need a file that does not.
        if velocity:
            s.create_dataset("velocity", data=np.zeros((2, 3, 12, 9, 1), np.float16))
    return path


class Temp(unittest.TestCase):
    """A fresh temporary folder per test, removed afterwards.

    Per TEST, not per class: these tests deliberately leave broken files
    behind, and one test's wreckage must not become the next one's input.
    """

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="prt_refuse_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)


# =============================================================================
#  BLOCK 1.  ASKING FOR SOMETHING THE FILE DOES NOT HAVE
# =============================================================================
class MissingData(Temp):

    def test_asking_for_a_simulated_flow_a_file_has_not_got(self):
        from dataset_reader import PRT3DDataset
        p = tiny_dataset(os.path.join(self.d, "d.h5"), velocity=False)
        with self.assertRaises(ValueError) as cm:
            PRT3DDataset(p, n_points=32, velocity_informed="simulated")
        self.assertIn("velocity", str(cm.exception).lower())

    def test_asking_for_a_predicted_flow_nobody_has_written(self):
        from dataset_reader import PRT3DDataset
        p = tiny_dataset(os.path.join(self.d, "d.h5"))
        with self.assertRaises(ValueError) as cm:
            PRT3DDataset(p, n_points=32, velocity_informed="predicted")
        self.assertIn("predict_velocity", str(cm.exception),
                      "the message should name the script that writes it")

    def test_asking_for_pore_size_maps_that_are_not_there(self):
        from dataset_reader import PRT3DDataset
        p = tiny_dataset(os.path.join(self.d, "d.h5"))
        with self.assertRaises(ValueError) as cm:
            PRT3DDataset(p, n_points=32, velocity_informed="simulated",
                         geom_features=True)
        self.assertIn("add_flow_features", str(cm.exception),
                      "the message should name the command that adds them")

    def test_a_flow_switch_with_no_flow_field(self):
        from dataset_reader import PRT3DDataset
        p = tiny_dataset(os.path.join(self.d, "d.h5"), velocity=False)
        with self.assertRaises(ValueError):
            PRT3DDataset(p, n_points=32, flow_proxy=True)


# =============================================================================
#  BLOCK 2.  ASKING FOR TWO THINGS THAT CONTRADICT EACH OTHER
# =============================================================================
class Contradictions(Temp):

    def test_switch_A_and_the_flow_pipeline_together_are_refused(self):
        """They are two different answers to one question. Switch A REPLACES
        the trunk's geometry feature; the flow pipeline leaves the trunk alone
        and adds to the branch. Whichever won would be a coin toss the log did
        not record, so the script refuses instead of choosing."""
        p = tiny_dataset(os.path.join(self.d, "d.h5"))
        rc, out = run([sys.executable, os.path.join(MODEL, "train.py"),
                       "--data", p, "--out", os.path.join(self.d, "o"),
                       "--epochs", "1", "--flow-proxy",
                       "--velocity-informed", "simulated"], timeout=300)
        self.assertNotEqual(rc, 0, "this combination should have been refused")
        self.assertIn("cannot be combined", out,
                      "the refusal should explain the clash:\n" + out[-400:])

    def test_the_pore_size_maps_without_the_flow(self):
        """--geom-features on its own does nothing, so asking for it alone is
        a mistake worth naming rather than silently ignoring."""
        p = tiny_dataset(os.path.join(self.d, "d.h5"))
        rc, out = run([sys.executable, os.path.join(MODEL, "train.py"),
                       "--data", p, "--out", os.path.join(self.d, "o2"),
                       "--epochs", "1", "--geom-features"], timeout=300)
        self.assertNotEqual(rc, 0)
        self.assertIn("velocity-informed", out)

    def test_an_unknown_velocity_source(self):
        from dataset_reader import resolve_switches
        with self.assertRaises(ValueError):
            resolve_switches(ndim=2, velocity_informed="maybe")


# =============================================================================
#  BLOCK 3.  BAD COMMAND LINES
# =============================================================================
class BadCommandLines(Temp):

    def test_a_flag_that_does_not_exist_is_rejected(self):
        p = tiny_dataset(os.path.join(self.d, "d.h5"))
        rc, out = run([sys.executable, os.path.join(MODEL, "train.py"),
                       "--data", p, "--out", os.path.join(self.d, "o"),
                       "--not-a-real-flag", "1"], timeout=180)
        self.assertNotEqual(rc, 0)
        self.assertIn("unrecognized", out.lower())

    def test_a_missing_required_argument_is_rejected(self):
        # No arguments at all. --data is required, so argparse must say so.
        rc, out = run([sys.executable, os.path.join(MODEL, "train.py")],
                      timeout=180)
        self.assertNotEqual(rc, 0)
        self.assertIn("required", out.lower())

    def test_a_file_that_is_not_there_is_reported_by_name(self):
        rc, out = run([sys.executable, os.path.join(TOOLS, "add_flow_features.py"),
                       "--data", os.path.join(self.d, "no_such_file.h5")],
                      timeout=180)
        self.assertNotEqual(rc, 0)
        self.assertIn("no_such_file", out,
                      "the message should name the file it could not find")


# =============================================================================
#  BLOCK 4.  CHECKPOINTS
# =============================================================================
class Checkpoints(Temp):

    def test_a_bare_state_dict_is_refused(self):
        """A checkpoint here carries the grid, the scaling and the held-out
        list as well as the weights. A bare state_dict has none of that, so
        nothing in it says how its inputs were scaled, and running it would
        produce a plausible field that is wrong by a constant."""
        import torch
        p = os.path.join(self.d, "bare.pt")
        # Weights and nothing else: no grid, no scaling, no held-out list.
        torch.save({"some.layer.weight": torch.zeros(2, 2)}, p)
        rc, out = run([sys.executable, os.path.join(MODEL, "predict_velocity.py"),
                       "--checkpoint", p, "--geometry",
                       os.path.join(self.d, "x.npz")], timeout=300)
        self.assertNotEqual(rc, 0)
        self.assertIn("scal", out.lower(),
                      "the refusal should explain that the scaling is missing")

    def test_a_geometry_at_the_wrong_size_is_refused(self):
        """The branch's fully connected layer is tied to the grid, so a
        checkpoint cannot be run at a different size. It must say so rather
        than reshaping something into silence."""
        import torch
        from velocity_model import build_velocity_model
        model, trunk_in = build_velocity_model((16, 12))
        ck = os.path.join(self.d, "vel.pt")
        torch.save({"model": model.state_dict(), "grid": (16, 12), "dim": 2,
                    "n_comp": 2, "trunk_in": trunk_in, "buffer": 0,
                    "pressure": "solve", "condition": "pe",
                    "stats": {"uprm_mu": 0.0, "uprm_sd": 1.0,
                              "mis_mu": 0.0, "mis_sd": 1.0,
                              "dw2_min": 0.0, "dw2_max": 1.0,
                              "cond_mu": 0.0, "cond_sd": 1.0,
                              "vel_mu": [0.0, 0.0], "vel_sd": [1.0, 1.0]}}, ck)
        g = os.path.join(self.d, "wrong_size.npz")
        np.savez_compressed(g, material=open_channel(30, 20))
        rc, out = run([sys.executable, os.path.join(MODEL, "predict_velocity.py"),
                       "--checkpoint", ck, "--geometry", g, "--pe", "10",
                       "--out", os.path.join(self.d, "p")], timeout=300)
        self.assertNotEqual(rc, 0)
        self.assertIn("grid", out.lower())


# =============================================================================
#  BLOCK 5.  GEOMETRY THAT CANNOT BE USED
# =============================================================================
class ImpossibleGeometry(unittest.TestCase):

    def test_a_domain_with_no_path_through_it_reaches_nothing(self):
        """A rock the inlet cannot reach through is not an error, but nothing
        can be delivered to the far side of it and the descriptor must say so
        rather than filling the far side in."""
        import flow_features as ff
        from _common import blocked
        p = ff.pore_mask_from_material(blocked())
        uprm = ff.uprm_map(p)
        far = uprm[6:]
        self.assertEqual(float(far.max()), 0.0,
                         "pore space behind a full wall got a nonzero UPRM")

    def test_a_solid_block_gives_zeros_rather_than_raising(self):
        import flow_features as ff
        from harmonic_pressure import harmonic_pressure
        from _common import all_solid
        p = ff.pore_mask_from_material(all_solid())
        self.assertTrue((ff.mis_map(p, buf=0) == 0).all())
        field, _ = harmonic_pressure(p)
        self.assertTrue((field == 0).all())

    def test_a_one_voxel_domain_does_not_divide_by_zero(self):
        import flow_features as ff
        p = np.ones((1, 1), bool)
        ff.mis_map(p, buf=0)
        ff.uprm_map(p)
        d, (lo, hi) = ff.dw2_map(p)
        self.assertGreater(hi, lo, "the scaling range must never be empty")


if __name__ == "__main__":
    unittest.main(verbosity=2)
