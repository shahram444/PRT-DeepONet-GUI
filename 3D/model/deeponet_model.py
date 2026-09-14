#!/usr/bin/env python3
"""
deeponet_model.py — PRT-DeepONet lifted from 2D to 3D.

The architecture is the published 2D model, block for block: a geometry branch
CNN (5 blocks of Conv+SiLU+AvgPool, channels 16/32/64/128/256, flatten ->
Linear -> 128) multiplied by a parameter branch FNN (3 Linear layers, input Pe
and Da, output 128), dot-producted with a trunk (8 Linear layers, SiLU on all
but the last, output 128), giving ONE scalar output field plus one scalar bias.
One model is trained per reaction system and per chemical species, exactly as
in the 2D release.

Only the extra dimension forces any change, and there are three such changes:

1. Conv2d -> Conv3d, AvgPool2d -> AvgPool3d.  Nothing else about the branch
   changes: a 64^3 volume through 5 halving blocks gives 2x2x2 at 256 channels
   = 2048 values, which is EXACTLY the flatten dimension of the 2D model's
   2x4x256.  The encoder ports over verbatim.

2. The trunk takes (x, y, z, gdf) for a steady dataset and (x, y, z, t, gdf)
   for a transient one, instead of the 2D (x, y, gdf) and (x, y, gdf, t).

3. The trunk is evaluated at a random subset of pore voxels rather than the
   full grid.  This is a memory necessity, not an architecture change: the 2D
   model evaluates all 9,472 grid points; the same thing at 64^3 is 262,144,
   which is a 26.8 GB activation tensor at batch 25.  At 8,192 sampled points
   the trunk costs 944 MMACs -- slightly LESS than the 2D model's 1,091 -- and
   memory stays near 0.3 GB.  Subsampling happens in dataset_reader.py.

Shapes
    branch1  (B, Cin, nx, ny, nz)     geometry (+3 velocity channels, optional)
    branch2  (B, n_params)            dimensionless numbers
    trunk    (B, P, 4 or 5)           sampled query points
    ->       (B, P)
"""

import numpy as np
import torch
import torch.nn as nn


class BranchCNN3D(nn.Module):
    """Geometry branch.

    DIMENSION AWARE.  A dataset with nz = 1 is a genuinely two-dimensional
    problem -- Jung's published setting -- and running 3D convolutions over a
    single-voxel third axis is both wasteful and broken: five halving blocks
    would reduce that axis to zero.  When the grid's last dimension is 1 the
    encoder switches to Conv2d and AvgPool2d and the singleton axis is squeezed
    away, which reproduces the published 2D branch exactly.  Everything else in
    the network is unchanged.
    """

    def __init__(self, in_channels=1, out_dim=128, num_blocks=5,
                 grid=(64, 64, 64), width=(16, 32, 64, 128, 256)):
        super().__init__()
        self.two_d = (len(grid) >= 3 and int(grid[2]) == 1)
        if self.two_d:
            grid = (int(grid[0]), int(grid[1]))
        # Each block halves every dimension.  Asking for more blocks than the
        # smallest dimension can survive gives a 1x1x1 tensor fed to AvgPool3d(2)
        # and a runtime error deep inside the forward pass, which is a miserable
        # thing to debug.  Clamp here instead, and say so.
        fit = max(1, min(int(np.log2(max(g, 1))) for g in grid))
        if num_blocks > fit:
            print("BranchCNN3D: grid %s supports at most %d halving blocks, "
                  "reducing from %d" % (tuple(grid), fit, num_blocks))
            num_blocks = fit
        self.num_blocks = num_blocks
        ch = [in_channels] + list(width)[:num_blocks]
        Conv = nn.Conv2d if self.two_d else nn.Conv3d
        Pool = nn.AvgPool2d if self.two_d else nn.AvgPool3d
        layers = []
        for i in range(num_blocks):
            layers += [Conv(ch[i], ch[i + 1], 3, 1, 1), nn.SiLU(), Pool(2)]
        self.features = nn.Sequential(*layers)
        d = [max(g // (2 ** num_blocks), 1) for g in grid]
        self.flat = ch[num_blocks]
        for v in d:
            self.flat *= v
        self.fc = nn.Linear(self.flat, out_dim)

    def forward(self, x):
        if self.two_d and x.dim() == 5:
            x = x[..., 0]                 # (B, C, nx, ny, 1) -> (B, C, nx, ny)
        x = self.features(x)
        return self.fc(x.reshape(x.size(0), -1))


class BranchFNN(nn.Module):
    """Parameter branch: the dimensionless groups, Pe and Da."""

    def __init__(self, in_dim=2, out_dim=128, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Trunk(nn.Module):
    """Trunk on (x, y, z, gdf) for a steady dataset, (x, y, z, t, gdf) for a
    transient one.  A plain stack of Linear layers with SiLU on every layer
    except the last, which is the published 2D trunk unchanged apart from the
    extra coordinate."""

    def __init__(self, in_dim=5, out_dim=128, num_layers=8, width=128):
        super().__init__()
        self.in_dim = in_dim
        self.first = nn.Linear(in_dim, width)
        self.hidden = nn.ModuleList(
            nn.Linear(width, width) for _ in range(num_layers - 2))
        self.last = nn.Linear(width, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.first(x))
        for lin in self.hidden:
            h = self.act(lin(h))
        return self.last(h)


class PRT_DeepONet3D(nn.Module):
    """branch1 (geometry) * branch2 (parameters), dotted with the trunk.

    One scalar output field plus one scalar bias, which is the 2D formulation
    exactly.  A dataset holding several chemical species is handled the way the
    2D release handles it: one model per species, selected at training time.
    """

    def __init__(self, in_channels=1, n_params=2, out_dim=128,
                 trunk_in_dim=5, cnn_blocks=5, trunk_layers=8, trunk_width=128,
                 grid=(64, 64, 64)):
        super().__init__()
        self.out_dim = out_dim
        self.branch1 = BranchCNN3D(in_channels, out_dim, cnn_blocks, grid)
        self.branch2 = BranchFNN(n_params, out_dim)
        self.trunk = Trunk(trunk_in_dim, out_dim, trunk_layers, trunk_width)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, b1, b2, trunk_pts):
        code = self.branch1(b1) * self.branch2(b2)          # (B, p)
        basis = self.trunk(trunk_pts)                       # (B, P, p)
        return (basis * code.unsqueeze(1)).sum(-1) + self.bias   # (B, P)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for cin, npar in ((1, 2), (4, 3)):
        m = PRT_DeepONet3D(in_channels=cin, n_params=npar)
        b1 = torch.randn(2, cin, 64, 64, 64)
        b2 = torch.randn(2, npar)
        tk = torch.rand(2, 8192, 5)
        y = m(b1, b2, tk)
        print("in_channels=%d  n_params=%d  params=%.2fM  out=%s"
              % (cin, npar, count_parameters(m) / 1e6, tuple(y.shape)))
