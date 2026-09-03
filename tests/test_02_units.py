#!/usr/bin/env python3
"""
test_02_units.py — one function at a time, against an answer worked out by hand.

WHAT A UNIT TEST IS
    A test of ONE function, in isolation, on an input small enough that a person
    can work out the right answer themselves. Nothing here reads a dataset,
    trains anything, or takes longer than a second.

    The rule that makes these worth having: **the expected answer must not come
    from this code**. Every number below was derived from the definition of the
    quantity, or is a property that must hold whatever the implementation. A
    test whose expected value was produced by running the thing it tests proves
    only that the code is deterministic.

    You do not need to know any reactive transport to read these. Each one says
    in plain words what it is checking and why the answer must be what it is.

WHAT IS COVERED
    the pore mask conventions, the three pore size maps, the pressure solve,
    the loss functions, the network's shape arithmetic, and the table that
    decides what the network is fed.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (PORE, SOLID, open_channel, two_channels,      # noqa: E402
                     bottleneck, all_solid)

import flow_features as ff                                         # noqa: E402
from harmonic_pressure import harmonic_pressure, harmonic_gradient  # noqa: E402
from flow_coordinates import squash                                # noqa: E402
from dataset_reader import resolve_switches, VELOCITY_SOURCES      # noqa: E402


# =============================================================================
#  BLOCK 1.  READING A PORE STRUCTURE
# =============================================================================
class PoreMask(unittest.TestCase):
    """Which numbers in the material array mean 'this voxel is water'.

    This project follows CompLaB: 0 solid, 1 the wall skin, 2 pore. The
    published 2D release uses the OPPOSITE for 1 and 2. Getting it backwards
    inverts the rock, and it has happened, so the convention is tested.
    """

    def test_project_convention(self):
        m = np.array([[SOLID, 1, PORE, 3]])
        got = ff.pore_mask_from_material(m)
        # 2 and above is pore; 0 and 1 are not.
        self.assertEqual(list(got.ravel()), [False, False, True, True])

    def test_explicit_pore_code_wins(self):
        """When you know the code, it is used and the convention is ignored."""
        m = np.array([[0, 1, 2]])
        got = ff.pore_mask_from_material(m, pore_code=1)
        self.assertEqual(list(got.ravel()), [False, True, False])

    def test_a_boolean_array_is_passed_through(self):
        m = np.array([[True, False]])
        self.assertEqual(list(ff.pore_mask_from_material(m).ravel()),
                         [True, False])


# =============================================================================
#  BLOCK 2.  THE THREE PORE SIZE MAPS
#
#  MIS  = how wide the pore is HERE
#  UPRM = how wide the narrowest throat between the inlet and here is
#  dw2  = the squared distance to the nearest wall, scaled to [0, 1]
# =============================================================================
class MIS(unittest.TestCase):
    """MIS measures local width, and nothing else."""

    def test_a_wide_channel_reads_wider_than_a_narrow_one(self):
        """Two straight channels side by side, 5 voxels wide and 1 voxel wide.

        MIS is a RADIUS, so a channel five voxels across admits a disk of
        radius 3 (it spans voxels centre-1, centre, centre+1 and touches the
        walls), and a channel one voxel across admits radius 1. Those are the
        numbers a pencil gives, and they are the numbers asserted.
        """
        p = ff.pore_mask_from_material(two_channels())
        mis = ff.mis_map(p, buf=0)
        self.assertAlmostEqual(float(mis[:, 3].max()), 3.0, places=5)
        self.assertAlmostEqual(float(mis[:, 9].max()), 1.0, places=5)

    def test_solid_voxels_are_exactly_zero(self):
        p = ff.pore_mask_from_material(two_channels())
        mis = ff.mis_map(p, buf=0)
        self.assertTrue((mis[~p] == 0).all(),
                        "a grain has no pore width and must read exactly zero")

    def test_it_is_never_negative(self):
        p = ff.pore_mask_from_material(open_channel())
        self.assertTrue((ff.mis_map(p, buf=0) >= 0).all())


class UPRM(unittest.TestCase):
    """UPRM measures what the INLET can deliver. This is the whole point."""

    def test_a_chamber_behind_a_throat_is_limited_by_the_throat(self):
        """A wide chamber reachable only through a one-voxel throat.

        The chamber is exactly as wide as the inlet chamber, so MIS reads the
        same in both: 3. But nothing wider than the throat can be delivered to
        the far one, so its UPRM must drop to the throat's own radius, 1.

        If this test ever fails, UPRM has become a local quantity and the
        descriptor no longer carries the information it exists for.
        """
        p = ff.pore_mask_from_material(bottleneck())
        mis = ff.mis_map(p, buf=0)
        uprm = ff.uprm_map(p)

        far_mis = float(mis[10:, 2:7].max())
        far_uprm = float(uprm[10:, 2:7].max())
        near_uprm = float(uprm[0:5, 2:7].max())

        self.assertAlmostEqual(far_mis, 3.0, places=5,
                               msg="MIS should see a wide chamber")
        self.assertAlmostEqual(far_uprm, 1.0, places=5,
                               msg="UPRM should be cut down to the throat")
        self.assertAlmostEqual(near_uprm, 3.0, places=5,
                               msg="before the throat, nothing limits it")
        self.assertLess(far_uprm, far_mis,
                        "UPRM must be smaller than MIS behind a bottleneck")

    def test_uprm_never_exceeds_mis(self):
        """A sphere delivered to a voxel must still fit in that voxel's pore.

        This has to hold everywhere, in every geometry, by definition.
        """
        for m in (open_channel(), two_channels(), bottleneck()):
            p = ff.pore_mask_from_material(m)
            mis = ff.mis_map(p, buf=0)
            uprm = ff.uprm_map(p)
            self.assertTrue((uprm <= mis + 1e-6).all(),
                            "UPRM exceeded MIS somewhere, which is impossible")

    def test_unreachable_pore_space_gets_nothing(self):
        """A pocket the inlet cannot reach can be delivered nothing at all."""
        m = np.full((10, 9), SOLID, np.uint8)
        m[:, 3:6] = PORE                 # a channel through
        m[4:6, 7] = PORE                 # a sealed pocket, touching nothing
        p = ff.pore_mask_from_material(m)
        uprm = ff.uprm_map(p)
        self.assertEqual(float(uprm[4:6, 7].max()), 0.0)


class DW2(unittest.TestCase):
    """The squared wall distance, scaled to [0, 1]."""

    def test_it_lands_inside_zero_and_one(self):
        p = ff.pore_mask_from_material(open_channel())
        d, (lo, hi) = ff.dw2_map(p)
        self.assertGreaterEqual(float(d[p].min()), 0.0)
        self.assertLessEqual(float(d[p].max()), 1.0)
        self.assertGreater(hi, lo)

    def test_the_scaling_can_be_reused(self):
        """Training must apply ONE pair of constants across the whole campaign.

        Rescaling each rock to its own range would tell the network that a wide
        rock and a narrow one look the same, which is the opposite of what the
        descriptor is for. So the constants come back out and can be passed in.
        """
        p1 = ff.pore_mask_from_material(open_channel(nx=12, ny=7))
        p2 = ff.pore_mask_from_material(open_channel(nx=12, ny=15))
        _d1, (lo, hi) = ff.dw2_map(p1)
        d2, (lo2, hi2) = ff.dw2_map(p2, dw2_min=lo, dw2_max=hi)
        self.assertEqual((lo2, hi2), (lo, hi))
        # The wider rock saturates at 1 under the narrow rock's scaling, rather
        # than being squeezed back into the same range.
        self.assertAlmostEqual(float(d2[p2].max()), 1.0, places=5)

    def test_solid_is_zero(self):
        p = ff.pore_mask_from_material(two_channels())
        d, _ = ff.dw2_map(p)
        self.assertTrue((d[~p] == 0).all())


class ZScoreStats(unittest.TestCase):
    """The scaling constants, over pore voxels only."""

    def test_solid_zeros_are_excluded(self):
        """A map of [0, 0, 2, 4] has mean 3 over its pore voxels, not 1.5.

        Including the solid would move the mean with the POROSITY of the rock
        rather than with its pore size, which is a different quantity.
        """
        maps = [np.array([[0.0, 0.0, 2.0, 4.0]], np.float32)]
        mu, sd = ff.zscore_stats(maps)
        self.assertAlmostEqual(mu, 3.0, places=5)
        self.assertAlmostEqual(sd, 1.0, places=5)

    def test_an_empty_rock_does_not_divide_by_zero(self):
        mu, sd = ff.zscore_stats([np.zeros((4, 4), np.float32)])
        self.assertEqual((mu, sd), (0.0, 1.0))


# =============================================================================
#  BLOCK 3.  THE PRESSURE SOLVE
# =============================================================================
class HarmonicPressure(unittest.TestCase):
    """Laplace's equation on the pore space.

    In a straight open channel the answer is known exactly: the pressure falls
    LINEARLY from inlet to outlet and does not vary across the channel at all.
    That is the textbook solution, and it is what is asserted.
    """

    def test_a_straight_channel_gives_a_straight_line(self):
        p = ff.pore_mask_from_material(open_channel(nx=12, ny=7))
        field, _info = harmonic_pressure(p)
        column = field[:, 3]
        self.assertAlmostEqual(float(column[0]), 1.0, places=4,
                               msg="the inlet should be at 1")
        self.assertAlmostEqual(float(column[-1]), 0.0, places=4,
                               msg="the outlet should be at 0")
        steps = np.diff(column)
        self.assertLess(float(steps.std()), 1e-5,
                        "every step should be the same size: that is linear")

    def test_no_sideways_gradient_in_a_straight_channel(self):
        """Nothing pushes the fluid across a straight channel, so the
        transverse component of the gradient must be exactly zero."""
        p = ff.pore_mask_from_material(open_channel(nx=12, ny=7))
        g = harmonic_gradient(p)
        self.assertEqual(g.shape[0], 2, "2D should give two components")
        self.assertTrue(np.allclose(g[1][p], 0.0, atol=1e-6),
                        "there should be no transverse gradient here")
        self.assertTrue((np.abs(g[0][p]) > 0).any(),
                        "there should be a gradient ALONG the channel")

    def test_solid_carries_no_pressure(self):
        p = ff.pore_mask_from_material(open_channel())
        field, _ = harmonic_pressure(p)
        self.assertTrue((field[~p] == 0).all())


# =============================================================================
#  BLOCK 4.  ARITHMETIC INSIDE THE NETWORK
# =============================================================================
class ShapeArithmetic(unittest.TestCase):
    """How many times a grid can be halved before an axis runs out.

    Pure arithmetic, checkable on paper. 24 halves to 12, 6, 3, and then stops,
    because 1 cannot be halved: four times. The published network uses five
    halvings, so five is the cap and a smaller grid gets fewer.
    """

    def test_counts_are_what_the_arithmetic_says(self):
        from velocity_model import max_pool_blocks
        self.assertEqual(max_pool_blocks((64, 148)), 5)     # the published grid
        self.assertEqual(max_pool_blocks((40, 24)), 4)      # 24 -> 12 -> 6 -> 3
        self.assertEqual(max_pool_blocks((8, 8)), 3)        # 8 -> 4 -> 2 -> 1
        self.assertEqual(max_pool_blocks((2, 2)), 1)
        self.assertEqual(max_pool_blocks((1, 1)), 1,        # never zero
                         "it must never return zero blocks")

    def test_a_grid_too_small_is_refused_with_a_reason(self):
        """Asking for more halvings than a grid can take used to crash deep
        inside a pooling layer with a message naming no axis and no fix."""
        from velocity_model import CustomCNN
        with self.assertRaises(ValueError) as cm:
            CustomCNN(1, 128, num_blocks=5, grid=(40, 24), dim=2)
        msg = str(cm.exception)
        self.assertIn("axis", msg, "the error should name the axis")


class Losses(unittest.TestCase):
    """The two terms the velocity operator is trained on."""

    def test_a_perfect_prediction_costs_nothing(self):
        import torch
        from velocity_model import ROIHuberLoss
        y = torch.randn(2, 8, 6, 2)
        roi = torch.ones(2, 8, 6)
        self.assertAlmostEqual(float(ROIHuberLoss()(y, y, roi)), 0.0, places=7)

    def test_only_pore_voxels_count(self):
        """The error is averaged over the PORE, not the whole grid. Averaging
        over the grid would make a low porosity rock look easy, because most of
        its voxels are solid and predicted zero by construction."""
        import torch
        from velocity_model import ROIHuberLoss
        crit = ROIHuberLoss()
        target = torch.zeros(1, 4, 4, 2)
        pred = torch.zeros(1, 4, 4, 2)
        pred[0, 0, 0, 0] = 1.0                 # one voxel wrong by 1
        roi_all = torch.ones(1, 4, 4)
        roi_one = torch.zeros(1, 4, 4)
        roi_one[0, 0, 0] = 1.0                 # only that voxel is pore
        # Same error, sixteen times fewer pore voxels: sixteen times the loss.
        self.assertAlmostEqual(float(crit(pred, target, roi_one)),
                               16 * float(crit(pred, target, roi_all)), places=5)

    def test_a_uniform_flow_has_no_divergence(self):
        """Mass is conserved in a flow that is the same everywhere."""
        import torch
        from velocity_model import divergence_penalty, interior_pore_mask
        pore = np.ones((8, 6), bool)
        pore[:, 0] = pore[:, -1] = False
        inner = torch.from_numpy(interior_pore_mask(pore))[None]
        pred = torch.ones(1, 8, 6, 2)
        self.assertAlmostEqual(
            float(divergence_penalty(pred, inner, [1.0, 1.0])), 0.0, places=7)

    def test_a_ramp_does_have_divergence(self):
        """A flow speeding up along x is NOT divergence free, and the penalty
        must see it. A test that only checks the zero case would pass on a
        function that always returns zero."""
        import torch
        from velocity_model import divergence_penalty, interior_pore_mask
        pore = np.ones((8, 6), bool)
        pore[:, 0] = pore[:, -1] = False
        inner = torch.from_numpy(interior_pore_mask(pore))[None]
        ramp = torch.zeros(1, 8, 6, 2)
        ramp[0, :, :, 0] = torch.arange(8, dtype=torch.float32)[:, None]
        # du/dx = 1 everywhere, and the penalty is the MEAN SQUARE, so 1.
        self.assertAlmostEqual(
            float(divergence_penalty(ramp, inner, [1.0, 1.0])), 1.0, places=5)

    def test_interior_excludes_the_border(self):
        """A central difference at the domain edge would reach outside it."""
        from velocity_model import interior_pore_mask
        pore = ff.pore_mask_from_material(open_channel(nx=12, ny=7))
        inner = interior_pore_mask(pore)
        self.assertFalse(inner[0].any(), "row 0 is the inlet face")
        self.assertFalse(inner[-1].any(), "the last row is the outlet face")
        self.assertTrue(inner[5].any(), "the middle should have an interior")


class Squash(unittest.TestCase):
    """t / (1 + t), the form the travel time reaches the network in."""

    def test_the_three_values_anyone_can_check(self):
        got = squash(np.array([0.0, 1.0, 3.0]))
        self.assertAlmostEqual(float(got[0]), 0.0, places=6)
        self.assertAlmostEqual(float(got[1]), 0.5, places=6)
        self.assertAlmostEqual(float(got[2]), 0.75, places=6)

    def test_it_preserves_order_and_stays_below_one(self):
        """It is a reparameterisation, not a clip: nothing is destroyed and the
        ordering is exact. That is why the tail can be compressed safely."""
        x = np.array([0.0, 0.1, 1.0, 10.0, 1000.0, 1e6])
        y = squash(x)
        self.assertTrue(np.all(np.diff(y) > 0), "order must be preserved")
        self.assertTrue(np.all(y < 1.0), "it must never reach 1")


# =============================================================================
#  BLOCK 5.  WHAT THE NETWORK IS FED
#
#  resolve_switches() is one function and one dictionary, and it decides every
#  input the model sees. It is pure: no files, no randomness, no state. That
#  makes it the single most testable thing in the project and the single most
#  expensive thing to get wrong.
# =============================================================================
class WhatTheNetworkIsFed(unittest.TestCase):

    def test_everything_off_is_the_original(self):
        off = resolve_switches(ndim=3)
        self.assertEqual(off["trunk_cols"], ["x", "y", "z", "t", "gdf"])
        self.assertEqual(off["branch_ch"], ["material"])
        self.assertEqual(off["trunk_dim"], 5)

    def test_two_dimensions_drop_the_z_column(self):
        """A 2D dataset has no third coordinate, and a constant column is a
        dead input. Dropping it reproduces the published 4-input trunk."""
        off = resolve_switches(ndim=2)
        self.assertEqual(off["trunk_cols"], ["x", "y", "t", "gdf"])
        self.assertEqual(off["trunk_dim"], 4)

    def test_switch_A_replaces_the_geometry_feature(self):
        a = resolve_switches(ndim=3, flow_proxy=True)
        self.assertIn("tau", a["trunk_cols"])
        self.assertNotIn("gdf", a["trunk_cols"])
        self.assertEqual(a["branch_ch"], ["ux", "uy", "uz"])

    def test_switch_C_turns_on_switch_A(self):
        c = resolve_switches(ndim=3, dim_free=True)
        self.assertTrue(c["flow_proxy"], "C implies A")
        self.assertEqual(c["trunk_cols"], ["t", "dwall", "tau"])

    def test_switch_C_is_the_same_width_in_2D_and_3D(self):
        """This is the entire argument for switch C: a network transfers
        between dimensions unchanged only if its trunk has the same number of
        inputs in both. The ordinary trunk does not: 4 against 5."""
        c2 = resolve_switches(ndim=2, dim_free=True)
        c3 = resolve_switches(ndim=3, dim_free=True)
        self.assertEqual(c2["trunk_dim"], c3["trunk_dim"])
        self.assertNotEqual(resolve_switches(ndim=2)["trunk_dim"],
                            resolve_switches(ndim=3)["trunk_dim"])

    def test_the_flow_pipeline_leaves_the_trunk_alone(self):
        """This is what separates it from switch A. The geometry feature stays
        in the trunk and the velocity is ADDED to the branch."""
        d = resolve_switches(ndim=3, velocity_informed="simulated")
        self.assertEqual(d["trunk_cols"], ["x", "y", "z", "t", "gdf"])
        self.assertEqual(d["branch_ch"], ["material", "ux", "uy", "uz"])

    def test_the_pore_size_maps_are_two_more_branch_channels(self):
        d = resolve_switches(ndim=3, velocity_informed="simulated",
                             geom_features=True)
        self.assertEqual(d["branch_ch"],
                         ["material", "ux", "uy", "uz", "mis", "uprm"])

    def test_an_unknown_velocity_source_is_refused(self):
        self.assertEqual(VELOCITY_SOURCES, ("off", "simulated", "predicted"))
        with self.assertRaises(ValueError):
            resolve_switches(ndim=3, velocity_informed="nonsense")

    def test_the_channel_count_matches_the_channel_list(self):
        """in_channels is what the first convolution is built with. If it ever
        disagreed with branch_ch the model would be built one size and fed
        another, and the error would arrive as a shape mismatch mid-training."""
        for kw in ({}, {"flow_proxy": True}, {"dim_free": True},
                   {"velocity_informed": "simulated"},
                   {"velocity_informed": "predicted", "geom_features": True}):
            for ndim in (2, 3):
                cfg = resolve_switches(ndim=ndim, **kw)
                self.assertEqual(cfg["in_channels"], len(cfg["branch_ch"]),
                                 "mismatch for ndim=%d %r" % (ndim, kw))
                self.assertEqual(cfg["trunk_dim"], len(cfg["trunk_cols"]),
                                 "mismatch for ndim=%d %r" % (ndim, kw))


# =============================================================================
#  BLOCK 6.  DEGENERATE INPUTS
#
#  A rock with no pore space, a rock with no path through it. These are not
#  hypothetical: a porosity sweep produces them at the low end.
# =============================================================================
class NothingToWorkWith(unittest.TestCase):

    def test_a_solid_block_does_not_crash_anything(self):
        p = ff.pore_mask_from_material(all_solid())
        self.assertTrue((ff.mis_map(p, buf=0) == 0).all())
        self.assertTrue((ff.uprm_map(p) == 0).all())
        d, _ = ff.dw2_map(p)
        self.assertTrue((d == 0).all())
        field, _ = harmonic_pressure(p)
        self.assertTrue((field == 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
