#!/usr/bin/env python3
# =============================================================================
# CHANGED FROM THE 2D VERSION
#
#   WHERE IT CAME FROM
#     github.com/hjunglab/PRT-DeepONet   branch/folder: velocity-informed
#     flow/models/PRT-DeepONet_Velocity_load.ipynb, code cells 4 and 5
#     flow/models/PRT-DeepONet_Velocity.ipynb, code cells 2 and 3
#     concentration/models/PRT-DeepONet_Monod.ipynb, code cell 3
#
#   WHAT THEIR 2D CODE DOES
#     Three networks. A U-Net from the pore mask to the pressure gradient; a
#     DeepONet from (mask, UPRM, MIS), the Reynolds number and a five-input
#     trunk to (ux, uy); and the concentration DeepONet with the velocity
#     added to its branch.
#
#   WHAT WE CHANGED, AND WHY
#     1. THE 2D CLASSES ARE UNCHANGED, DELIBERATELY. Channel ladders, block
#        counts, layer widths, SiLU placement and the latent split are theirs
#        exactly. That is what lets Velocity.pt, Pressure_component_UNet.pt
#        and Monod.pt load with no missing key, which is what makes any
#        comparison with their published numbers a comparison with THEIR
#        method rather than with our reimplementation. Do not tidy them.
#     2. WE ADDED 3D FORMS. Conv2d becomes Conv3d, two output components
#        become three, and the trunk grows from five inputs to seven: three
#        coordinates, three pressure gradient components, one wall distance.
#     3. THE 3D LATENT IS 192 WIDE, NOT 128. The latent is cut into one block
#        per output component. 128 / 2 = 64 in 2D; 128 / 3 is not an integer,
#        so 192 keeps 64 per component instead of silently narrowing each one.
#     4. THE GRID IS A PARAMETER. Theirs is fixed at 64 x 148. The branch
#        flattens after five halvings, so a smaller grid would hit an axis of
#        size zero inside AvgPool; max_pool_blocks() reduces the block count
#        and says so, instead of crashing with a message that names no axis.
#     5. THE 3D PRESSURE U-NET IS NARROWER THAN THEIRS. Six levels of 64
#        channels does not fit in three dimensions at any useful batch size.
#        Those defaults are ours, and are marked as ours in the class.
# =============================================================================
"""The velocity operator, and the pressure U-Net that feeds it.

Two networks, and the 2D forms of both are exact ports so that the weights released
with the paper load into them unchanged. test_velocity_model.py proves that by loading
Velocity.pt and Pressure_component_UNet.pt and reporting any parameter that does not
match by name and shape; if either reports a missing key, the port has drifted and the
comparison with their published numbers is void.

    PressureComponentUNet   pore mask -> the two components of a harmonic pressure
                            gradient. Harmonic means it solves Laplace's equation, so
                            it does not depend on the flow rate at all: it is computed
                            ONCE per rock and reused at every Reynolds number. That is
                            the whole reason it can be afforded.

    VelocityDeepONet        branch1 = [mask, UPRM, MIS], branch2 = [Re],
                            trunk = [x, y, dPx, dPy, dw2]  ->  (ux, uy)

    ConcentrationDeepONet   branch1 = [mask, ux, uy], branch2 = [Pe, Da],
                            trunk = [x, y, t, GDF]  ->  C

PROVENANCE. The 2D classes are ports of the reference implementation released with the
velocity informed PRT-DeepONet by Jo and Jung (github.com/hjunglab/PRT-DeepONet,
velocity-informed). Channel ladders, block counts, layer widths, the SiLU placement and
the latent split are theirs and are not ours to change: changing any of them means the
released weights no longer load and the comparison stops being a comparison.

THE THREE DIMENSIONAL FORMS ARE OURS. Conv2d becomes Conv3d, the output grows from two
components to three, and the trunk takes seven inputs rather than five: three
coordinates, three pressure gradient components and the squared wall distance. One
choice needs stating. The latent vector is split into one block per output component,
so a 128 wide latent gives 64 per component in 2D. Three components do not divide 128,
so the 3D default is 192, which keeps 64 per component rather than silently narrowing
each one. Pass out_dim yourself if you want otherwise; it must be divisible by the
number of components.

Run it like this.

    python velocity_model.py --summary
    python velocity_model.py --summary --dim 3
    python velocity_model.py --check-reference /path/to/velocity-informed

What you get back. Nothing on disk. --summary prints the parameter count of every
sub network and the shape a forward pass produces, which is the cheapest way to see
that a grid size you are about to train on actually flattens to something sensible.

Worth knowing. The CNN branch flattens after five halvings, so the flatten width
depends on the grid. At 64 by 148 that is 256 x 2 x 4 = 2048, which is exactly the 2D
paper's number. A grid whose sides are not divisible by 32 gets floor divided at every
block and still works, but the fully connected layer is then tied to that grid: a
checkpoint trained at one grid will not load at another. --summary prints the flatten
width so you can see it before spending a night on it.
"""
import argparse
import math
import os
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                                                    # pragma: no cover
    sys.exit("velocity_model.py needs torch. Run: python gui/install_requirements.py")


# The reference grid, in the reference's own axis order: 64 across, 148 along the flow.
REF_NX, REF_NY = 64, 148

# Pressure U-Net shape constants, from the released code. Not ours to tune.
N_BLOCKS, BASE_CH, DEPTH, DROPOUT = 4, 64, 6, 0.1

def max_pool_blocks(grid, cap=5):
    """How many halving blocks a grid can take before an axis reaches zero.

    Five is the released number and the one every published figure was measured with,
    so it is the cap rather than a target. A smaller grid gets fewer blocks and the
    builders say so, because the alternative is a crash inside AvgPool with a message
    about output size that names no axis and no fix.
    """
    n = cap
    for g in grid:
        k = 0
        v = int(g)
        while v >= 2 and k < cap:
            v //= 2
            k += 1
        n = min(n, k)
    return max(n, 1)


__all__ = [
    "max_pool_blocks",
    "PressureComponentUNet", "PressureComponentUNet3D",
    "VelocityDeepONet", "VelocityDeepONet3D",
    "ConcentrationDeepONet", "ConcentrationDeepONet3D",
    "build_velocity_model", "build_concentration_model",
    "divergence_penalty", "ROIHuberLoss",
]


# ============================================================ the pressure U-Net
class ConvBlock(nn.Module):
    """Conv, ReLU, dropout, batch norm. In that order, which is theirs."""

    def __init__(self, cin, cout, p=DROPOUT, dim=2):
        super().__init__()
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        Drop = nn.Dropout2d if dim == 2 else nn.Dropout3d
        Norm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
        self.conv = Conv(cin, cout, 3, padding=1)
        self.drop = Drop(p)
        self.bn = Norm(cout)             # AFTER dropout, which is unusual; theirs

    def forward(self, x):
        return self.bn(self.drop(F.relu(self.conv(x))))


class Container(nn.Module):
    """N conv blocks in a row. The U-Net is built entirely out of these."""

    def __init__(self, cin, cout, n=N_BLOCKS, p=DROPOUT, dim=2):
        super().__init__()
        self.net = nn.Sequential(*([ConvBlock(cin, cout, p, dim)]
                                   + [ConvBlock(cout, cout, p, dim) for _ in range(n - 1)]))

    def forward(self, x):
        return self.net(x)


class _PressureUNetBase(nn.Module):
    """Symmetric U-Net from the pore mask to the pressure gradient components.

    Every level holds the same channel count rather than doubling, which is unusual
    and is theirs. Six levels of halving on a 64 by 148 grid takes the bottleneck to
    1 by 2, so the receptive field covers the whole domain, which is what a harmonic
    field needs: the pressure at a point depends on the entire boundary, not on a
    neighbourhood.
    """

    def __init__(self, in_ch=1, out_ch=2, base=BASE_CH, depth=DEPTH, n=N_BLOCKS,
                 p=DROPOUT, dim=2):
        super().__init__()
        Pool = nn.MaxPool2d if dim == 2 else nn.MaxPool3d
        Up = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        self.dim = dim
        self.depth = depth
        self.enc = nn.ModuleList()
        c = in_ch
        for _ in range(depth):
            self.enc.append(Container(c, base, n, p, dim))
            c = base
        self.pool = Pool(2)
        self.bottleneck = Container(base, base, n, p, dim)
        self.up = nn.ModuleList([Up(base, base, 2, stride=2) for _ in range(depth)])
        self.dec = nn.ModuleList([Container(base * 2, base, n, p, dim) for _ in range(depth)])
        self.head_container = Container(base, base, n, p, dim)
        self.head = Conv(base, out_ch, 1)

    def forward(self, x):
        skips = []
        for d in range(self.depth):
            x = self.enc[d](x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for d in range(self.depth):
            skip = skips[self.depth - 1 - d]
            x = self.up[d](x)
            # Odd sizes do not halve evenly, so the upsampled tensor comes back one
            # voxel short. Pad rather than crop the skip: cropping would quietly shift
            # the field by half a voxel relative to the geometry.
            pad = []
            for ax in range(1, self.dim + 1):
                diff = skip.shape[-ax] - x.shape[-ax]
                pad += [diff // 2, diff - diff // 2]
            if any(pad):
                x = F.pad(x, pad)
            x = torch.cat([x, skip], 1)
            x = self.dec[d](x)
        return self.head(self.head_container(x))


class PressureComponentUNet(_PressureUNetBase):
    """2D. in = 1 (the pore mask), out = 2 (dP/dx, dP/dy). Loads the released weights."""

    def __init__(self, in_ch=1, out_ch=2, base=BASE_CH, depth=DEPTH, n=N_BLOCKS, p=DROPOUT):
        super().__init__(in_ch, out_ch, base, depth, n, p, dim=2)


class PressureComponentUNet3D(_PressureUNetBase):
    """3D. in = 1, out = 3. Ours; there is no released 3D checkpoint to load."""

    def __init__(self, in_ch=1, out_ch=3, base=32, depth=4, n=2, p=DROPOUT):
        # A 3D U-Net at their width and depth is enormous: six levels of 64 channels
        # in three dimensions does not fit on one card at any useful batch size. The
        # defaults here are narrower and shallower, and they are OURS, not theirs.
        super().__init__(in_ch, out_ch, base, depth, n, p, dim=3)


# ============================================================== the velocity operator
class CustomCNN(nn.Module):
    """The geometry branch. Five halving blocks, then one fully connected layer.

    The flatten width is tied to the grid, which is why a checkpoint does not transfer
    between grid sizes. flatten_width() reports it without building the module.
    """

    def __init__(self, in_channels, out_dim=128, num_blocks=5, grid=(REF_NX, REF_NY), dim=2):
        super().__init__()
        # The channel ladder is theirs, verbatim. At five blocks on 64 x 148 the
        # flatten width comes to 256 * 2 * 4 = 2048, which is the number printed
        # in their paper: the arithmetic is the check that this is their network.
        ladder = [in_channels, 16, 32, 64, 128, 256, 512][:num_blocks + 1]
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        Pool = nn.AvgPool2d if dim == 2 else nn.AvgPool3d
        layers = []
        for i in range(num_blocks):
            layers += [Conv(ladder[i], ladder[i + 1], 3, 1, 1), nn.SiLU(), Pool(2)]
        self.features = nn.Sequential(*layers)
        sizes = [int(g) for g in grid]
        for _ in range(num_blocks):
            sizes = [s // 2 for s in sizes]
        if min(sizes) < 1:
            raise ValueError(
                "a %s grid cannot take %d halving blocks: axis %d reaches zero. "
                "Use num_blocks=%d, or pad the grid so every axis is at least %d."
                % (tuple(grid), num_blocks, int(np.argmin(sizes)),
                   max_pool_blocks(grid), 2 ** num_blocks))
        self.blocks = num_blocks
        self.pooled = tuple(sizes)
        # The flatten width is tied to the GRID, so a checkpoint trained at one
        # grid will not load at another. That is not a limitation to work around;
        # it is why every checkpoint here records the grid it was trained on and
        # refuses a dataset of a different size rather than reshaping silently.
        self.flat = ladder[num_blocks] * math.prod(sizes)
        self.fc = nn.Linear(self.flat, out_dim)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.reshape(x.size(0), -1))


class ScalarMLP(nn.Module):
    """The condition branch. Three linear layers, SiLU on all but the last."""

    def __init__(self, in_dim=1, out_dim=128, hidden=128, layers=3):
        super().__init__()
        m = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(layers - 2):
            m += [nn.Linear(hidden, hidden), nn.SiLU()]
        m += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*m)

    def forward(self, x):
        return self.net(x)


def make_trunk(in_dim=5, out_dim=128, layers=8, width=128):
    """Eight linear layers, SiLU on the first seven. Theirs, unchanged."""
    m = [nn.Linear(in_dim, width), nn.SiLU()]
    for _ in range(layers - 2):
        m += [nn.Linear(width, width), nn.SiLU()]
    m += [nn.Linear(width, out_dim)]
    return nn.Sequential(*m)


class _VelocityBase(nn.Module):
    """branch1 (CNN) * branch2 (MLP) * trunk, split into one block per component.

    The output is a vector, and a DeepONet produces a scalar. The released code gets a
    vector out by cutting the latent into as many equal blocks as there are components
    and summing within each block, so a 128 wide latent becomes two 64 wide dot
    products. It is not a separate head per component; the branch and the trunk are
    shared, and only the reduction differs.
    """

    def __init__(self, b1_ch, out_dim, n_out, trunk_in, grid, dim, blocks=5):
        super().__init__()
        if out_dim % n_out:
            raise ValueError(
                "out_dim (%d) must divide by the number of output components (%d), "
                "otherwise the latent cannot be split evenly. Try %d."
                % (out_dim, n_out, n_out * (out_dim // n_out)))
        self.branch1 = CustomCNN(b1_ch, out_dim, num_blocks=blocks, grid=grid, dim=dim)
        self.branch2 = ScalarMLP(1, out_dim, 128, 3)     # one flow number in
        self.trunk = make_trunk(trunk_in, out_dim, 8, 128)
        # One bias per component, learned. A DeepONet's product-then-sum has no
        # constant term of its own, and a velocity field has a mean.
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.grid = tuple(grid)
        self.n_out = n_out
        self.half = out_dim // n_out

    def forward(self, b1, b2, tr):
        # (N, Lp, D) -> flatten the point axis so the trunk sees one long batch
        # of coordinates, then fold it back. Lp is every voxel of the grid here,
        # not a sample of them, which is why the trunk stays narrow.
        N, Lp, D = tr.shape
        t = self.trunk(tr.reshape(-1, D)).view(N, Lp, -1)
        b1o = self.branch1(b1).unsqueeze(1)      # (N, 1, out_dim)
        b2o = self.branch2(b2).unsqueeze(1)      # (N, 1, out_dim)
        # THE VECTOR TRICK. A DeepONet's product-then-sum gives ONE number. To
        # get a vector out, cut the latent into n_out equal blocks and sum
        # within each block: 128 wide becomes two 64-wide dot products. Not a
        # head per component; the branch and trunk are shared and only the
        # reduction differs. This is theirs, unchanged.
        prod = (b1o * b2o * t).view(N, Lp, self.n_out, self.half).sum(-1)
        return (prod + self.bias).view((N,) + self.grid + (self.n_out,))


class VelocityDeepONet(_VelocityBase):
    """2D. branch1 = [mask, UPRM, MIS] | branch2 = [Re] | trunk = [x, y, dPx, dPy, dw2].

    Attribute names match the released checkpoint, so Velocity.pt loads with no
    missing and no unexpected keys.
    """

    def __init__(self, b1_ch=3, out_dim=128, n_out=2, grid=(REF_NX, REF_NY), blocks=None):
        super().__init__(b1_ch, out_dim, n_out, trunk_in=5, grid=grid, dim=2,
                         blocks=blocks or max_pool_blocks(grid))
        self.nx, self.ny = grid


class VelocityDeepONet3D(_VelocityBase):
    """3D. branch1 = [mask, UPRM, MIS] | branch2 = [Re] |
    trunk = [x, y, z, dPx, dPy, dPz, dw2] -> (ux, uy, uz).

    out_dim defaults to 192 so each of the three components still gets 64 latent
    values, the same width each of the two gets in 2D.
    """

    def __init__(self, b1_ch=3, out_dim=192, n_out=3, grid=(64, 64, 64), blocks=None):
        super().__init__(b1_ch, out_dim, n_out, trunk_in=7, grid=grid, dim=3,
                         blocks=blocks or max_pool_blocks(grid))


# ========================================================= the concentration operator
class _ConcentrationBase(nn.Module):
    """branch1 = [mask, u...] | branch2 = [Pe, Da] | trunk = [x.., t, GDF] -> C.

    One scalar out, so no latent split. This is the original PRT-DeepONet with two
    extra branch channels, which is the whole intervention the follow up paper makes.
    """

    def __init__(self, b1_ch, b2_in, trunk_in, out_dim, grid, dim, blocks=5):
        super().__init__()
        self.branch1_net = CustomCNN(b1_ch, out_dim, num_blocks=blocks, grid=grid, dim=dim)
        self.branch2_net = ScalarMLP(b2_in, out_dim, 128, 3)
        self.trunk_net = make_trunk(trunk_in, out_dim, 8, 128)
        self.bias = nn.Parameter(torch.zeros(1))
        self.grid = tuple(grid)

    def forward(self, b1, b2, tr):
        N = tr.shape[0]
        to = self.trunk_net(tr).unsqueeze(1)
        b1o = self.branch1_net(b1).unsqueeze(1).unsqueeze(2)
        b2o = self.branch2_net(b2).unsqueeze(1).unsqueeze(2)
        out = (b1o * b2o * to).sum(-1) + self.bias
        return out.view((N,) + self.grid + (1,))


class ConcentrationDeepONet(_ConcentrationBase):
    """2D. Loads the released Irreversible_Sorption.pt and Monod.pt."""

    def __init__(self, nx=REF_NX, ny=REF_NY, trunk_in_dim=4, out_dim=128,
                 branch1_ch=3, branch2_in_dim=2, blocks=None):
        super().__init__(branch1_ch, branch2_in_dim, trunk_in_dim, out_dim, (nx, ny), 2,
                         blocks=blocks or max_pool_blocks((nx, ny)))
        self.nx, self.ny = nx, ny


class ConcentrationDeepONet3D(_ConcentrationBase):
    """3D. branch1 = [mask, ux, uy, uz], trunk = [x, y, z, t, GDF]."""

    def __init__(self, grid=(64, 64, 64), trunk_in_dim=5, out_dim=128,
                 branch1_ch=4, branch2_in_dim=2, blocks=None):
        super().__init__(branch1_ch, branch2_in_dim, trunk_in_dim, out_dim, grid, 3,
                         blocks=blocks or max_pool_blocks(grid))


# ============================================================================ losses
class ROIHuberLoss(nn.Module):
    """Huber, averaged over the pore voxels only.

    Averaging over the whole grid instead would make a low porosity rock look easy,
    because most of its voxels are solid and predicted zero by construction. The
    denominator here is the number of pore voxels times the component count, so the
    number means the same thing at every porosity.
    """

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, roi):
        # The mask is the pore space. Note the denominator below: pore voxels
        # times components, NOT the whole grid. Dividing by the grid would make
        # a low-porosity rock look easy, because most of its voxels are solid
        # and predicted zero by construction.
        mask = (roi > 0.5).unsqueeze(-1)
        diff = (pred - target) * mask
        absd = torch.abs(diff)
        quad = 0.5 * absd.pow(2)
        lin = self.delta * (absd - 0.5 * self.delta)
        loss = torch.where(absd <= self.delta, quad, lin)
        return loss.sum() / (mask.sum().clamp(min=1.0) * pred.shape[-1])


def divergence_penalty(pred, interior, ratios):
    """Mean squared divergence over the interior pore voxels, in standardized units.

    The components are z scored separately before training, so a raw sum of their
    derivatives is not a divergence: each term carries its own scale. ratios rescales
    every component back onto the first one's units, which is what makes the sum mean
    something. In 2D that is a single number, SD_v / SD_u.

    Interior means a voxel whose face neighbours along every axis are also pore, since
    a central difference against a solid neighbour is not a derivative of the field.
    """
    ndim = pred.dim() - 2
    div = torch.zeros_like(pred[..., 0])
    # Each component was z-scored SEPARATELY before training, so a raw sum of
    # their derivatives is not a divergence: every term carries its own scale.
    # ratios puts them back on a common footing. In 2D that is one number,
    # SD_v / SD_u, which is exactly what their RV does.
    for ax in range(ndim):
        comp = pred[..., ax]
        sl_c = [slice(None)] + [slice(1, -1)] * ndim
        sl_p = list(sl_c)
        sl_m = list(sl_c)
        sl_p[ax + 1] = slice(2, None)
        sl_m[ax + 1] = slice(None, -2)
        d = torch.zeros_like(comp)
        d[tuple(sl_c)] = (comp[tuple(sl_p)] - comp[tuple(sl_m)]) * 0.5
        div = div + float(ratios[ax]) * d
    m = interior.float()
    # clamp, not an if. A rock so tight that no voxel has pore neighbours on
    # every axis has no interior at all, and a zero divergence term is the right
    # answer for it; a division by zero would poison the whole batch's gradient.
    return (div.pow(2) * m).sum() / m.sum().clamp(min=1.0)


def interior_pore_mask(pore):
    """Voxels whose face neighbours along every axis are pore. Torch or numpy."""
    is_torch = torch.is_tensor(pore)
    p = pore if is_torch else torch.as_tensor(pore)
    p = p.bool()
    ndim = p.dim()
    # The border is excluded by construction: `core` never touches index 0 or -1,
    # so a voxel on the inlet face is never interior. A central difference there
    # would reach outside the domain.
    inner = torch.zeros_like(p)
    core = tuple(slice(1, -1) for _ in range(ndim))
    acc = p[core].clone()
    for ax in range(ndim):
        for shift in (1, -1):
            sl = list(core)
            sl[ax] = slice(2, None) if shift == 1 else slice(None, -2)
            acc = acc & p[tuple(sl)]
    inner[core] = acc
    return inner if is_torch else inner.numpy()


# ========================================================================== builders
def build_velocity_model(grid, b1_ch=3, out_dim=None):
    """The right velocity operator for a grid, 2D or 3D, with its trunk width."""
    grid = tuple(int(g) for g in grid)
    # A 2D campaign is stored one voxel deep so that one reader serves both.
    # Building a 3D network for it would ask a 3x3x3 kernel to see a domain that
    # is one voxel thick in z, so the flat axis is dropped here instead.
    if len(grid) == 3 and grid[2] == 1:
        grid = grid[:2]
    if len(grid) == 2:
        # 128 wide, cut into two blocks of 64. Theirs, and the released weights
        # load into it. Trunk: x, y, dPx, dPy, dw2 = 5.
        return VelocityDeepONet(b1_ch=b1_ch, out_dim=out_dim or 128, n_out=2, grid=grid), 5
    # 192, not 128, and that is a real choice rather than a round number. Three
    # components do not divide 128, and the nearest legal value below it, 126,
    # would narrow each component to 42. 192 keeps 64 per component, which is
    # what the 2D model gives each of its two. Trunk gains z and dPz = 7.
    return VelocityDeepONet3D(b1_ch=b1_ch, out_dim=out_dim or 192, n_out=3, grid=grid), 7


def build_concentration_model(grid, branch1_ch=3, trunk_in_dim=None, branch2_in_dim=2):
    """The right concentration operator for a grid, 2D or 3D."""
    grid = tuple(int(g) for g in grid)
    if len(grid) == 3 and grid[2] == 1:
        grid = grid[:2]
    if len(grid) == 2:
        return ConcentrationDeepONet(nx=grid[0], ny=grid[1],
                                     trunk_in_dim=trunk_in_dim or 4,
                                     branch1_ch=branch1_ch,
                                     branch2_in_dim=branch2_in_dim)
    return ConcentrationDeepONet3D(grid=grid, trunk_in_dim=trunk_in_dim or 5,
                                   branch1_ch=branch1_ch, branch2_in_dim=branch2_in_dim)


def load_reference_weights(model, path, verbose=True):
    """Load a released checkpoint and REPORT what did not match, rather than hiding it.

    strict=False is used deliberately, because a partial load that is announced is
    more useful than a hard failure with no detail. A non empty missing list means the
    port has drifted from the released code, and any number measured after that is not
    comparable with the paper.
    """
    try:
        st = torch.load(path, map_location="cpu")
    except Exception:
        st = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(st, dict) and "state_dict" in st:
        st = st["state_dict"]
    missing, unexpected = model.load_state_dict(st, strict=False)
    if verbose:
        print("  loaded %s" % os.path.basename(path))
        print("    missing keys    : %d %s" % (len(missing), list(missing)[:4] or ""))
        print("    unexpected keys : %d %s" % (len(unexpected), list(unexpected)[:4] or ""))
        shape_bad = []
        for k, v in st.items():
            own = dict(model.state_dict()).get(k)
            if own is not None and tuple(own.shape) != tuple(v.shape):
                shape_bad.append((k, tuple(v.shape), tuple(own.shape)))
        print("    shape mismatches: %d %s" % (len(shape_bad), shape_bad[:3] or ""))
        if missing or unexpected or shape_bad:
            print("    THE PORT HAS DRIFTED. Numbers measured from this model are not "
                  "comparable with the published ones.")
    return missing, unexpected


# ============================================================================== cli
def _count(m):
    return sum(p.numel() for p in m.parameters())


def _summary(dim, grid):
    print("velocity operator, %dD, grid %s" % (dim, grid))
    model, trunk_in = build_velocity_model(grid)
    print("  branch1 CNN     %9d params, %d halving blocks, pooled to %s, flatten width %d"
          % (_count(model.branch1), model.branch1.blocks, model.branch1.pooled,
             model.branch1.flat))
    print("  branch2 MLP     %9d params" % _count(model.branch2))
    print("  trunk           %9d params, %d inputs" % (_count(model.trunk), trunk_in))
    print("  whole model     %9d params, %d output components, %d latent each"
          % (_count(model), model.n_out, model.half))
    n = 2
    Lp = math.prod(grid)
    b1 = torch.zeros((n, 3) + tuple(grid))
    b2 = torch.zeros((n, 1))
    tr = torch.zeros((n, Lp, trunk_in))
    with torch.no_grad():
        out = model(b1, b2, tr)
    print("  forward pass    %s -> %s" % (tuple(b1.shape), tuple(out.shape)))

    print("\nconcentration operator, %dD, grid %s" % (dim, grid))
    ch = 3 if dim == 2 else 4
    tin = 4 if dim == 2 else 5
    cm = build_concentration_model(grid, branch1_ch=ch, trunk_in_dim=tin)
    print("  branch1 CNN     %9d params, flatten width %d"
          % (_count(cm.branch1_net), cm.branch1_net.flat))
    print("  whole model     %9d params, branch1 %d channels, trunk %d inputs"
          % (_count(cm), ch, tin))
    with torch.no_grad():
        out = cm(torch.zeros((n, ch) + tuple(grid)), torch.zeros((n, 2)),
                 torch.zeros((n, Lp, tin)))
    print("  forward pass    -> %s" % (tuple(out.shape),))

    print("\npressure U-Net, %dD" % dim)
    un = PressureComponentUNet(1, 2) if dim == 2 else PressureComponentUNet3D(1, 3)
    print("  whole model     %9d params" % _count(un))
    if dim == 3:
        print("  NOTE: the 3D U-Net defaults are ours, narrower and shallower than "
              "the released 2D one, which does not fit in three dimensions.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="The velocity operator and its pressure U-Net.")
    ap.add_argument("--summary", action="store_true", help="print the shapes and counts")
    ap.add_argument("--dim", type=int, default=2, choices=(2, 3))
    ap.add_argument("--grid", type=int, nargs="+", default=None,
                    help="grid size; default is the reference 64 148 in 2D, 64 64 64 in 3D")
    ap.add_argument("--check-reference", metavar="DIR",
                    help="path to the released velocity-informed folder; loads "
                         "Velocity.pt and Pressure_component_UNet.pt into the ported "
                         "classes and reports every key that does not match")
    a = ap.parse_args(argv)

    grid = tuple(a.grid) if a.grid else ((REF_NX, REF_NY) if a.dim == 2 else (64, 64, 64))
    if a.check_reference:
        root = a.check_reference
        vel = os.path.join(root, "flow", "parameters", "Velocity.pt")
        unet = os.path.join(root, "flow", "parameters", "Pressure_component_UNet.pt")
        conc = os.path.join(root, "concentration", "parameters", "Monod.pt")
        print("checking the port against the released weights")
        ok = True
        if os.path.exists(unet):
            m, u = load_reference_weights(PressureComponentUNet(1, 2), unet)
            ok = ok and not m and not u
        else:
            print("  not found:", unet)
            ok = False
        if os.path.exists(vel):
            m, u = load_reference_weights(
                VelocityDeepONet(3, 128, 2, grid=(REF_NX, REF_NY)), vel)
            ok = ok and not m and not u
        else:
            print("  not found:", vel)
            ok = False
        if os.path.exists(conc):
            m, u = load_reference_weights(
                ConcentrationDeepONet(REF_NX, REF_NY, 4, 128, 3, 2), conc)
            ok = ok and not m and not u
        else:
            print("  not found:", conc)
        print("\n" + ("The port matches the released code."
                      if ok else "THE PORT DOES NOT MATCH. Read the lines above."))
        return 0 if ok else 1

    if a.summary:
        _summary(a.dim, grid)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
