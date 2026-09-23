#!/usr/bin/env python3
"""
test_model.py -- the network itself, in under a second, with no data and no
training.

What each test would catch

  shapes            the output growing a species axis again, or the trunk and
                    the branch disagreeing about the latent width
  scalar_bias       the bias becoming a vector, which is how the multi-species
                    head crept in last time
  no_film           the geodesic re-injection coming back: it is an addition to
                    the published architecture, not a port of it
  param_width       the parameter branch stopping to honour n_params, which is
                    what lets a 2-column and a 6-column dataset both train
  two_d_fallback    a dataset with nz = 1 being sent through 3D convolutions,
                    which is both wasteful and wrong
  block_clamp       a small grid asking for more halving blocks than it has
                    voxels, which used to fail deep inside the forward pass
  ladder            the channel ladder or the flatten width drifting away from
                    the published model's 2 x 4 x 256

Run it directly, or through ../smoke_test.py with everything else.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "3D", "model"))

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402

from deeponet_model import (BranchCNN3D, BranchFNN, PRT_DeepONet3D,  # noqa: E402
                            Trunk, count_parameters)


# =============================================================================
#  THE ARCHITECTURE MUST STAY THE PUBLISHED ONE
#  One output field, one scalar bias, no FiLM, a parameter branch that honours
#  n_params, and a dataset with nz = 1 going through 2D convolutions. Every one
#  of these is a thing that would train perfectly well if it drifted, and would
#  quietly stop being comparable with the 2D release.
# =============================================================================
class Architecture(unittest.TestCase):

    def test_shapes(self):
        m = PRT_DeepONet3D(in_channels=1, n_params=2, grid=(16, 16, 16),
                           trunk_in_dim=5)
        y = m(torch.randn(3, 1, 16, 16, 16), torch.randn(3, 2),
              torch.rand(3, 128, 5))
        self.assertEqual(tuple(y.shape), (3, 128),
                         "the output must be (batch, points), with no species "
                         "axis")

    def test_scalar_bias(self):
        m = PRT_DeepONet3D(grid=(16, 16, 16))
        self.assertEqual(tuple(m.bias.shape), (1,),
                         "one scalar bias, as in the 2D model")

    def test_no_film(self):
        t = Trunk(in_dim=5, out_dim=128, num_layers=8, width=128)
        self.assertFalse(hasattr(t, "film"),
                         "FiLM re-injection is not part of the 2D model")
        self.assertFalse(hasattr(t, "inject_every"))
        # and the trunk must be a plain stack: first, hidden, last
        self.assertEqual(len(t.hidden), 6)
        for name, mod in (("first", t.first), ("last", t.last)):
            self.assertIsInstance(mod, nn.Linear, name)

    def test_trunk_widths(self):
        for in_dim in (4, 5):
            t = Trunk(in_dim=in_dim, out_dim=128, num_layers=8, width=128)
            y = t(torch.rand(2, 32, in_dim))
            self.assertEqual(tuple(y.shape), (2, 32, 128))

    def test_param_width(self):
        for n in (2, 3, 6):
            m = PRT_DeepONet3D(n_params=n, grid=(16, 16, 16), trunk_in_dim=5)
            y = m(torch.randn(2, 1, 16, 16, 16), torch.randn(2, n),
                  torch.rand(2, 16, 5))
            self.assertEqual(tuple(y.shape), (2, 16))
        with self.assertRaises(RuntimeError):
            m = PRT_DeepONet3D(n_params=2, grid=(16, 16, 16), trunk_in_dim=5)
            m(torch.randn(2, 1, 16, 16, 16), torch.randn(2, 6),
              torch.rand(2, 16, 5))

    # -------------------------------------------------------------------
    #  2D IS NOT A SEPARATE CODE PATH
    #  A grid with nz = 1 must use Conv2d, and a real 3D grid must use Conv3d,
    #  decided by the data and never by a flag. These two are what stop a 2D
    #  fork of the network appearing.
    # -------------------------------------------------------------------
    def test_two_d_fallback(self):
        b = BranchCNN3D(in_channels=1, out_dim=128, num_blocks=5,
                        grid=(64, 64, 1))
        self.assertTrue(b.two_d, "nz = 1 is a 2D problem")
        kinds = {type(m).__name__ for m in b.features}
        self.assertIn("Conv2d", kinds)
        self.assertNotIn("Conv3d", kinds)
        # it must accept the 3D-shaped tensor the loader hands it
        self.assertEqual(tuple(b(torch.randn(2, 1, 64, 64, 1)).shape), (2, 128))

    def test_three_d_uses_3d_convs(self):
        b = BranchCNN3D(grid=(64, 64, 64))
        self.assertFalse(b.two_d)
        kinds = {type(m).__name__ for m in b.features}
        self.assertIn("Conv3d", kinds)

    # -------------------------------------------------------------------
    #  THE 2048 IDENTITY
    #  148 by 64 halved five times is 4 by 2 at 256 channels. 64 cubed halved
    #  five times is 2 by 2 by 2 at 256. Both flatten to 2048, which is why the
    #  published weights load into the 3D network at all. If this ever fails,
    #  the warm start has stopped being possible and nothing else would say so.
    # -------------------------------------------------------------------
    def test_flatten_width_matches_the_paper(self):
        b = BranchCNN3D(grid=(64, 64, 64), num_blocks=5)
        self.assertEqual(b.flat, 2 * 2 * 2 * 256)
        two = BranchCNN3D(grid=(64, 148, 1), num_blocks=5)
        self.assertEqual(two.flat, 2 * 4 * 256)

    def test_block_clamp(self):
        b = BranchCNN3D(grid=(8, 8, 8), num_blocks=5)
        self.assertLessEqual(b.num_blocks, 3)
        self.assertEqual(tuple(b(torch.randn(1, 1, 8, 8, 8)).shape), (1, 128))

    def test_channel_ladder(self):
        b = BranchCNN3D(grid=(64, 64, 64), num_blocks=5)
        out = [m.out_channels for m in b.features if hasattr(m, "out_channels")]
        self.assertEqual(out, [16, 32, 64, 128, 256])

    def test_parameter_branch_depth(self):
        f = BranchFNN(in_dim=2, out_dim=128)
        lin = [m for m in f.net if isinstance(m, nn.Linear)]
        self.assertEqual(len(lin), 3, "three Linear layers, as in the paper")
        self.assertNotIsInstance(f.net[-1], nn.SiLU,
                                 "no activation on the last layer")

    def test_velocity_channels(self):
        """Geometry plus three velocity components, the flow-aware input."""
        m = PRT_DeepONet3D(in_channels=4, grid=(16, 16, 16), trunk_in_dim=5)
        y = m(torch.randn(2, 4, 16, 16, 16), torch.randn(2, 2),
              torch.rand(2, 16, 5))
        self.assertEqual(tuple(y.shape), (2, 16))

    def test_size_is_reasonable(self):
        m = PRT_DeepONet3D(grid=(64, 64, 64))
        n = count_parameters(m)
        self.assertGreater(n, 1e6)
        self.assertLess(n, 5e6, "the model should not have grown a head again")

    def test_determinism(self):
        torch.manual_seed(0)
        m = PRT_DeepONet3D(grid=(16, 16, 16), trunk_in_dim=5).eval()
        a = torch.randn(1, 1, 16, 16, 16)
        b = torch.randn(1, 2)
        t = torch.rand(1, 32, 5)
        with torch.no_grad():
            y1, y2 = m(a, b, t), m(a, b, t)
        self.assertTrue(torch.equal(y1, y2))

    def test_gradients_reach_every_part(self):
        m = PRT_DeepONet3D(grid=(16, 16, 16), trunk_in_dim=5)
        y = m(torch.randn(2, 1, 16, 16, 16), torch.randn(2, 2),
              torch.rand(2, 16, 5))
        y.sum().backward()
        dead = [n for n, p in m.named_parameters()
                if p.grad is None or not torch.isfinite(p.grad).all()]
        self.assertEqual(dead, [], "these parameters got no usable gradient")


if __name__ == "__main__":
    unittest.main(verbosity=2)
