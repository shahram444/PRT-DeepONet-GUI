#!/usr/bin/env python3
"""
deeponet_model.py — PRT-DeepONet lifted from 2D to 3D.

Straight port of the architecture in PRT-DeepONet_Monod.ipynb, with three
changes, each forced by the extra dimension:

1. Conv2d -> Conv3d, AvgPool2d -> AvgPool3d.  Nothing else about the branch
   changes: a 64^3 volume through 5 halving blocks gives 2x2x2 at 256 channels
   = 2048 values, which is EXACTLY the flatten dimension of the 2D model's
   2x4x256.  The encoder ports over verbatim.

2. The trunk takes 5 inputs (x, y, z, t_norm, gdf) instead of 4, and is
   evaluated at a random subset of pore voxels rather than the full grid.  The
   2D model evaluates all 9,472 grid points; the same thing at 64^3 is 262,144,
   which is a 26.8 GB activation tensor at batch 25.  At 8,192 sampled points
   the trunk costs 944 MMACs -- slightly LESS than the 2D model's 1,091 -- and
   memory stays near 0.3 GB.  Subsampling happens in dataset_reader.py.

3. Multi-species output.  The branch emits n_species x p coefficients against
   ONE shared trunk, rather than n_species independent trunks.  The trunk is the
   part exposed to the per-voxel cost, so duplicating it would multiply the
   dominant cost by the number of species for no benefit.

Shapes
    branch1  (B, Cin, nx, ny, nz)     geometry (+3 velocity channels, optional)
    branch2  (B, n_params)            dimensionless numbers
    trunk    (B, P, 5)                sampled query points
    ->       (B, P, n_species)
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


# =============================================================================
#  BLOCK 2.  THE PARAMETER BRANCH
#
#  The dimensionless groups: Peclet, the Damkohler numbers, the half saturation
#  constants and the yield. Three layers is enough because there are only a
#  handful of numbers and no structure among them to discover.
# =============================================================================
class BranchFNN(nn.Module):
    """Parameter branch: the dimensionless groups."""

    def __init__(self, in_dim=6, out_dim=128, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
#  BLOCK 3.  THE TRUNK, AND WHY IT KEEPS BEING SHOWN THE GEOMETRY
# =============================================================================
class Trunk(nn.Module):
    """Trunk on (x, y, z, t_norm, geodesic_distance).

    `inject_every` re-feeds the geometry feature (the last input column) into
    every Nth hidden layer.  Deep networks with a single early geometry input
    provably lose that information with depth -- "Do Neural Operators Forget
    Geometry?" (2026) formalises it via a data-processing-inequality argument.
    The 2D model is shallow enough not to care; a 3D one is not.  Set
    inject_every=0 to reproduce the plain 2D behaviour.
    """

    def __init__(self, in_dim=5, out_dim=128, num_layers=8, width=128,
                 inject_every=3):
        super().__init__()
        self.inject_every = int(inject_every)
        self.in_dim = in_dim
        self.first = nn.Linear(in_dim, width)
        self.hidden = nn.ModuleList()
        self.film = nn.ModuleList()
        for i in range(num_layers - 2):
            self.hidden.append(nn.Linear(width, width))
            use = self.inject_every > 0 and (i + 1) % self.inject_every == 0
            self.film.append(nn.Linear(1, 2 * width) if use else None)
        self.last = nn.Linear(width, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        # The LAST column by convention, not by name. dataset_reader.py builds
        # the trunk in this order and both ends have to agree; that agreement
        # lives in resolve_switches().
        geo = x[..., -1:]                      # the geodesic column
        h = self.act(self.first(x))
        for lin, film in zip(self.hidden, self.film):
            h = self.act(lin(h))
            if film is not None:               # FiLM re-injection
                gb = film(geo)
                g, b = gb.chunk(2, dim=-1)
                h = h * (1 + g) + b
        return self.last(h)


# =============================================================================
#  BLOCK 4.  THE WHOLE NETWORK
#
#  branch1 (geometry) times branch2 (parameters), dotted with the trunk. A
#  DeepONet's product-then-sum gives ONE number per query point, and there are
#  several chemicals, so the head widens the fused code to one coefficient
#  vector per species before the dot product.
# =============================================================================
class PRT_DeepONet3D(nn.Module):
    """branch1 (geometry) * branch2 (parameters), dotted with the shared trunk."""

    def __init__(self, in_channels=1, n_params=6, n_species=4, out_dim=128,
                 trunk_in_dim=5, cnn_blocks=5, trunk_layers=8, trunk_width=128,
                 grid=(64, 64, 64), inject_every=3):
        super().__init__()
        self.n_species = n_species
        self.out_dim = out_dim
        self.branch1 = BranchCNN3D(in_channels, out_dim, cnn_blocks, grid)
        self.branch2 = BranchFNN(n_params, out_dim)
        self.trunk = Trunk(trunk_in_dim, out_dim, trunk_layers, trunk_width,
                           inject_every)
        # one set of p coefficients per species, from the fused branch code
        self.head = nn.Linear(out_dim, n_species * out_dim)
        # One learned bias per species. The dot product has no constant term,
        # and a concentration field has a mean.
        self.bias = nn.Parameter(torch.zeros(n_species))

    def forward(self, b1, b2, trunk_pts):
        code = self.branch1(b1) * self.branch2(b2)          # (B, p)
        coef = self.head(code).view(-1, self.n_species, self.out_dim)
        basis = self.trunk(trunk_pts)                       # (B, P, p)
        # einsum rather than a reshape and a matmul: the index letters say what
        # is contracted, which is the part that is easy to get silently wrong.
        out = torch.einsum("bsp,bnp->bns", coef, basis)     # (B, P, S)
        return out + self.bias


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for cin, nsp in ((1, 4), (4, 4)):
        m = PRT_DeepONet3D(in_channels=cin, n_species=nsp)
        b1 = torch.randn(2, cin, 64, 64, 64)
        b2 = torch.randn(2, 6)
        tk = torch.rand(2, 8192, 5)
        y = m(b1, b2, tk)
        print("in_channels=%d  params=%.2fM  out=%s"
              % (cin, count_parameters(m) / 1e6, tuple(y.shape)))
