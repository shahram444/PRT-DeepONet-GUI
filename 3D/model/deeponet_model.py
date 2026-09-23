#!/usr/bin/env python3
"""
deeponet_model.py -- the 3D network. The definition now lives in prt_core.

WHAT CHANGED, AND WHY NOTHING ELSE HAD TO
    This file used to hold its own copy of the architecture, and so did
    2D_scripts/prt2d_model.py, and so does each published notebook. Three
    copies of one network is three chances for them to drift, and they had
    drifted: the reversible sorption notebook's class default says four
    convolution blocks and it is built with five.

    The architecture is written once in prt_core/model.py. This file re-exports
    it under the names every script in 3D/ already imports, so train.py,
    evaluate.py, predict.py, run_ablation_sweep.py and the tools continue to
    say `from deeponet_model import PRT_DeepONet3D` and get the one definition.
    Nothing about the tensors, the key names or the numbers changed: the
    checkpoints written before this change load unaltered, which the core's own
    self-test checks tensor for tensor.

THE ARCHITECTURE, UNCHANGED
    A geometry branch CNN (5 blocks of Conv, SiLU, AvgPool, channels
    16/32/64/128/256, flatten, Linear to 128) multiplied by a parameter branch
    FNN (3 Linear layers taking the dimensionless numbers), dot-producted with
    a trunk (8 Linear layers, SiLU on all but the last, out 128), giving ONE
    scalar output field plus one scalar bias. One model per reaction and per
    chemical species, exactly as in the 2D release.

    Only the extra dimension forces any change, and there are three:

    1. Conv2d to Conv3d, AvgPool2d to AvgPool3d. Nothing else about the branch
       changes: a 64^3 volume through 5 halving blocks gives 2x2x2 at 256
       channels = 2048, which is EXACTLY the flatten width of the 2D model's
       4x2x256. The encoder ports over verbatim, and prt_core/model.py asserts
       that identity rather than trusting it.

    2. The trunk takes (x, y, z, gdf) for a steady dataset and (x, y, z, t, gdf)
       for a transient one, instead of the 2D (x, y, gdf) and (x, y, t, gdf).
       Which of those it is comes from the reaction registry now, not from a
       number typed into a constructor.

    3. The trunk is evaluated at a random subset of pore voxels rather than the
       full grid. This is a memory necessity, not an architecture change: the
       2D model evaluates all 9,472 grid points; the same thing at 64^3 is
       262,144, which is a 26.8 GB activation tensor at batch 25. At 8,192
       sampled points the trunk costs 944 MMACs, slightly LESS than the 2D
       model's 1,091, and memory stays near 0.3 GB. Subsampling happens in
       dataset_reader.py.

2D IS NOT A SEPARATE CODE PATH
    BranchCNN3D checks whether the grid's third dimension is 1 and switches to
    Conv2d and AvgPool2d when it is, squeezing the singleton axis away. That
    reproduces the published 2D branch exactly, so one file serves both and
    there is no 2D fork to keep in step.

Shapes
    branch1  (B, Cin, nx, ny, nz)     geometry, plus 3 velocity channels
    branch2  (B, n_params)            the dimensionless numbers
    trunk    (B, P, 4 or 5)           sampled query points
    ->       (B, P)
"""

import os
import sys

# Three dirname calls up from 3D/model/ is the repository root, which is where
# prt_core sits. Computed rather than assumed, so the tree can be moved.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Prepended only if absent, so a caller that already arranged its own path is
# left alone.
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# =============================================================================
#  THE RE-EXPORT
#  These are the names every script in 3D/ already imports from this file. They
#  now come from prt_core/model.py, so `from deeponet_model import
#  PRT_DeepONet3D` keeps working and there is one definition behind it. Nothing
#  about the tensors or the key names changed, which is what lets checkpoints
#  written before this change load unaltered.
# =============================================================================
from prt_core.model import (BranchCNN3D, BranchFNN, PRT_DeepONet3D,  # noqa: E402,F401
                            Trunk, build, checkpoint_matches, count_parameters,
                            load_state, remap_published, remap_to_published)

# Named explicitly: a star import of this module should give the network and
# the loading helpers, not os and sys as well.
__all__ = ["BranchCNN3D", "BranchFNN", "Trunk", "PRT_DeepONet3D",
           "count_parameters", "build", "load_state", "remap_published",
           "remap_to_published", "checkpoint_matches"]


# =============================================================================
#  RUN THIS FILE TO SEE THE SIZES
#  Two configurations: geometry alone with two dimensionless numbers, and the
#  velocity-informed branch with three. Both at 64 cubed, which is the grid the
#  3D work uses.
# =============================================================================
if __name__ == "__main__":
    import torch

    print("the definition is prt_core/model.py; this file re-exports it\n")
    for cin, npar in ((1, 2), (4, 3)):
        m = PRT_DeepONet3D(in_channels=cin, n_params=npar)
        b1 = torch.randn(2, cin, 64, 64, 64)
        b2 = torch.randn(2, npar)
        tk = torch.rand(2, 8192, 5)     # 8192 sampled pore voxels, not the full grid
        y = m(b1, b2, tk)
        print("in_channels=%d  n_params=%d  params=%.2fM  out=%s"
              % (cin, npar, count_parameters(m) / 1e6, tuple(y.shape)))
