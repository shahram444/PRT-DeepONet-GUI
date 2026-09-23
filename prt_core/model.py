#!/usr/bin/env python3
"""
prt_core/model.py -- the network, defined once, for 2D and 3D and for both
key layouts.

WHAT CHANGED FROM THE 2D VERSION
    There were three copies of this architecture in the repository. The three
    published notebooks each build it inline with their own constructor
    arguments. 2D_scripts/prt2d_model.py is a port of those. 3D/model/
    deeponet_model.py is the same network lifted one dimension. Three copies of
    one architecture is three chances for them to drift, and they had already
    drifted in one place: the reversible sorption notebook's class default says
    four convolution blocks and it is built with five.

    The architecture is written once here. deeponet_model.py and prt2d_model.py
    now re-export from this file, so the names every existing script imports
    still work and there is one definition behind them.

THE TWO KEY LAYOUTS, AND WHY BOTH ARE KEPT
    The released .pt files hold branch1_net, branch2_net, trunk_net and bias,
    with the trunk as a plain Sequential. Those names are inside files we did
    not write and cannot rename.

    Our own checkpoints hold branch1, branch2, trunk and bias, with the trunk
    as first, hidden and last, which is how it was written when the 3D side
    began.

    Both are produced from the SAME modules here. remap_published() converts
    one to the other, positionally and completely, and the conversion is
    checked in the self-test rather than trusted.

THE IDENTITY THAT MAKES THE TRANSFER WORK
    148 x 64, halved five times, is 4 x 2 at 256 channels = 2048.
    64 x 64 x 64, halved five times, is 2 x 2 x 2 at 256 channels = 2048.
    The flatten width is the same number, so the published geometry encoder and
    the published trunk load into the 3D network with no surgery. That is not a
    coincidence worth relying on blindly, so the self-test asserts it.

2D IS NOT A SEPARATE CODE PATH
    A grid whose third dimension is 1 is a two-dimensional problem. The branch
    switches to Conv2d and AvgPool2d, the trunk drops its z column, and the
    velocity branch drops uz. Nothing selects this by hand; it follows the data.

    python -m prt_core.model --self-test
"""

import numpy as np
import torch
import torch.nn as nn

try:                                   # the package, when imported normally
    from . import reactions as _reactions
except ImportError:                    # run as a file, for the self-test
    import reactions as _reactions


# =============================================================================
#  THE PIECES
# =============================================================================
class BranchCNN3D(nn.Module):
    """The geometry encoder, in whichever dimension the grid has.

    A dataset with nz = 1 is a genuinely two-dimensional problem, and running
    3D convolutions over a single-voxel third axis is both wasteful and broken:
    five halving blocks would reduce that axis to zero. When the grid's last
    dimension is 1 the encoder uses Conv2d and AvgPool2d and squeezes the
    singleton axis away, which reproduces the published 2D branch exactly,
    tensor for tensor.
    """

    def __init__(self, in_channels=1, out_dim=128, num_blocks=5,
                 grid=(64, 64, 64), width=(16, 32, 64, 128, 256)):
        super().__init__()
        self.two_d = (len(grid) >= 3 and int(grid[2]) == 1)
        if self.two_d:
            grid = (int(grid[0]), int(grid[1]))
        # Each block halves every dimension. Asking for more blocks than the
        # smallest dimension can survive gives a 1x1x1 tensor fed to AvgPool(2)
        # and a runtime error deep inside the forward pass, which is a
        # miserable thing to debug. Clamp here instead, and say so.
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


def BranchCNN(in_channels, out_dim, num_blocks=5, nx=64, ny=148):
    """The published 2D signature, on the same module.

    Kept because prt2d_model.py and the notebooks spell it this way. It is one
    line and it is the whole compatibility layer for the geometry branch.
    """
    return BranchCNN3D(in_channels, out_dim, num_blocks, grid=(nx, ny, 1))


class BranchFNN(nn.Module):
    """The dimensionless numbers. (B, n) -> (B, out_dim).

    Three linear layers, which is what every published notebook builds and what
    our own training builds. The width of the FIRST layer is the number of
    dimensionless groups the reaction has, so it is decided by the reaction
    registry and not by a number typed into a constructor.
    """

    def __init__(self, in_dim=2, out_dim=128, hidden_dim=128, num_layers=3,
                 activation="silu"):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.SiLU
        layers = [nn.Linear(in_dim, hidden_dim), act()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Trunk(nn.Module):
    """Eight linear layers, SiLU on the first seven. OUR key layout.

    first, hidden.0 ... hidden.5, last. The published files hold the same eight
    layers as a Sequential; remap_published() converts between the two, and the
    mapping is positional and complete because the layers are the same layers.
    """

    def __init__(self, in_dim=5, out_dim=128, num_layers=8, width=128,
                 activation="silu"):
        super().__init__()
        self.in_dim = in_dim
        self.first = nn.Linear(in_dim, width)
        self.hidden = nn.ModuleList(
            nn.Linear(width, width) for _ in range(num_layers - 2))
        self.last = nn.Linear(width, out_dim)
        self.act = nn.ReLU() if activation == "relu" else nn.SiLU()

    def forward(self, x):
        h = self.act(self.first(x))
        for lin in self.hidden:
            h = self.act(lin(h))
        return self.last(h)


def create_trunk_net(trunk_in_dim, out_dim, num_layers=8, width=128,
                     activation="silu"):
    """The same eight layers as a Sequential. THE PUBLISHED key layout.

    Used only where the released key names matter: building a network that a
    released .pt loads into without remapping.
    """
    act = nn.ReLU if activation == "relu" else nn.SiLU
    layers = [nn.Linear(trunk_in_dim, width), act()]
    for _ in range(num_layers - 2):
        layers += [nn.Linear(width, width), act()]
    layers += [nn.Linear(width, out_dim)]
    return nn.Sequential(*layers)


# =============================================================================
#  THE OPERATOR
# =============================================================================
class PRT_DeepONet3D(nn.Module):
    """branch1 (geometry) times branch2 (parameters), dotted with the trunk.

    One scalar output field plus one scalar bias, which is the published
    formulation exactly. A dataset holding several chemical species is handled
    the way the release handles it: one model per species, chosen at training
    time and recorded in the checkpoint.

    Shapes
        b1         (B, Cin, nx, ny, nz)   geometry, plus velocity channels
        b2         (B, n_params)          the dimensionless numbers
        trunk_pts  (B, P, D)              sampled query points
        ->         (B, P)
    """

    def __init__(self, in_channels=1, n_params=2, out_dim=128,
                 trunk_in_dim=5, cnn_blocks=5, trunk_layers=8, trunk_width=128,
                 grid=(64, 64, 64)):
        super().__init__()
        self.out_dim = out_dim
        self.grid = tuple(int(v) for v in grid)
        self.branch1 = BranchCNN3D(in_channels, out_dim, cnn_blocks, grid)
        self.branch2 = BranchFNN(n_params, out_dim)
        self.trunk = Trunk(trunk_in_dim, out_dim, trunk_layers, trunk_width)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, b1, b2, trunk_pts):
        code = self.branch1(b1) * self.branch2(b2)               # (B, p)
        basis = self.trunk(trunk_pts)                            # (B, P, p)
        return (basis * code.unsqueeze(1)).sum(-1) + self.bias   # (B, P)


class PRT2D(nn.Module):
    """The same operator with the RELEASED key names and the released forward.

    This exists so a file in 2D/parameters/ loads with no remapping and no
    strict=False, and so the port can be compared with the notebooks tensor for
    tensor. The modules inside it are the ones above; only the attribute names
    and the output shape differ.

    The output is (N, nx, ny, 1) because that is what the notebooks reshape to
    before plotting. The raw field is what comes back: clipping to 0 and 1 and
    zeroing the solid is a display convention and lives in as_displayed().
    """

    def __init__(self, trunk_in_dim, branch2_in_dim, cnn_blocks=5, out_dim=128,
                 num_layers=8, width=128, nx=64, ny=148):
        super().__init__()
        self.branch1_net = BranchCNN(1, out_dim, num_blocks=cnn_blocks,
                                     nx=nx, ny=ny)
        self.branch2_net = BranchFNN(branch2_in_dim, out_dim, hidden_dim=width,
                                     num_layers=3)
        self.trunk_net = create_trunk_net(trunk_in_dim, out_dim,
                                          num_layers=num_layers, width=width)
        self.bias = nn.Parameter(torch.zeros(1))
        self.nx, self.ny = nx, ny

    def forward(self, branch1_input, branch2_input, trunk_input):
        if trunk_input.ndim == 4 and trunk_input.shape[1] == 1:
            trunk_input = trunk_input.squeeze(1)          # (N, L, D)
        n, npts, dim = trunk_input.shape

        trunk_out = self.trunk_net(trunk_input.reshape(-1, dim))
        trunk_out = trunk_out.view(n, npts, -1).unsqueeze(1)             # (N,1,L,C)
        b1 = self.branch1_net(branch1_input).unsqueeze(1).unsqueeze(2)   # (N,1,1,C)
        b2 = self.branch2_net(branch2_input).unsqueeze(1).unsqueeze(2)   # (N,1,1,C)

        out = (b1 * b2 * trunk_out).sum(-1) + self.bias                  # (N,1,L)
        return out.view(n, self.nx, self.ny, 1)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# =============================================================================
#  THE TWO KEY LAYOUTS
# =============================================================================
# trunk_net is [Linear, SiLU, Linear, SiLU, ... , Linear], so the linear layers
# sit at the even indices 0, 2, 4, 6, 8, 10, 12, 14. Written out rather than
# computed, because it is the one place a silent off-by-one would load six of
# eight layers and leave a network that runs and is wrong.
_TRUNK_POSITIONS = {0: "trunk.first", 14: "trunk.last"}
for _i, _n in enumerate((2, 4, 6, 8, 10, 12)):
    _TRUNK_POSITIONS[_n] = "trunk.hidden.%d" % _i


def remap_published(src):
    """A released state dict, in our key layout. Positional and complete."""
    out = {}
    for k, v in src.items():
        if k.startswith("branch1_net."):
            out["branch1." + k[len("branch1_net."):]] = v
        elif k.startswith("branch2_net."):
            out["branch2." + k[len("branch2_net."):]] = v
        elif k.startswith("trunk_net."):
            rest = k[len("trunk_net."):]
            idx, _, tail = rest.partition(".")
            tgt = _TRUNK_POSITIONS.get(int(idx))
            if tgt:
                out["%s.%s" % (tgt, tail)] = v
        elif k == "bias":
            out["bias"] = v
    return out


def remap_to_published(src):
    """The other direction, so one of our checkpoints can be read by the
    published notebooks. Used by the parity test, and by anyone who wants to
    hand a trained model back in the form the release expects."""
    back = {v: k for k, v in _TRUNK_POSITIONS.items()}
    out = {}
    for k, v in src.items():
        if k.startswith("branch1."):
            out["branch1_net." + k[len("branch1."):]] = v
        elif k.startswith("branch2."):
            out["branch2_net." + k[len("branch2."):]] = v
        elif k.startswith("trunk."):
            rest = k[len("trunk."):]
            if rest.startswith("hidden."):
                i, _, tail = rest[len("hidden."):].partition(".")
                idx = back["trunk.hidden.%s" % i]
            else:
                head, _, tail = rest.partition(".")
                idx = back["trunk.%s" % head]
            out["trunk_net.%d.%s" % (idx, tail)] = v
        elif k == "bias":
            out["bias"] = v
    return out


def load_state(model, state, strict=True, say=print):
    """Load a state dict of either layout, and report honestly what happened.

    Returns (taken, missing, clashes). A shape clash is DROPPED rather than
    raised on, and named, because load_state_dict raises on a size mismatch
    whatever strict is set to, and a warm start into a wider parameter branch
    is a legitimate thing to want.
    """
    if any(k.startswith(("branch1_net.", "trunk_net.")) for k in state):
        state = remap_published(state)
    own = model.state_dict()
    took, clashes, unknown = {}, [], []
    for k, v in state.items():
        if k not in own:
            unknown.append(k)
        elif tuple(own[k].shape) != tuple(v.shape):
            clashes.append((k, tuple(v.shape), tuple(own[k].shape)))
        else:
            took[k] = v
    missing = [k for k in own if k not in took
               and not k.endswith("num_batches_tracked")]

    if strict and (missing or clashes or unknown):
        strict = False
    model.load_state_dict(took, strict=strict)

    if say:
        if clashes:
            say("  shape mismatch, not loaded:")
            for k, a, b in clashes:
                say("    %-34s file %s   model %s" % (k, a, b))
        if missing:
            say("  freshly initialised: %s" % ", ".join(missing))
        if unknown:
            say("  not present in this model: %s" % ", ".join(unknown))
    return took, missing, clashes


# =============================================================================
#  BUILDING ONE FROM A REACTION
# =============================================================================
def build(reaction, grid, in_channels=1, extra_trunk_cols=(), cnn_blocks=5,
          out_dim=128, trunk_layers=8, trunk_width=128, layout="ours"):
    """The network for one reaction on one grid.

    reaction          a key, a path, or a Reaction. It decides how many
                      dimensionless numbers the parameter branch takes and
                      whether the trunk has a time column. Nothing else does.
    grid              (nx, ny, nz). nz = 1 means a 2D problem, and the branch,
                      the trunk and the channels all follow it.
    in_channels       1 for geometry alone; 4 with a velocity field
    extra_trunk_cols  columns the switches add, such as tau or dwall
    layout            'ours' for our key names, 'published' for the released
                      ones. Both are the same modules.

    Returns (model, info), where info is everything the checkpoint should
    record so that evaluate and predict rebuild this exact network without
    guessing: the reaction key, the trunk columns by name, the distance
    convention, the grid and the channel count.
    """
    r = reaction if isinstance(reaction, _reactions.Reaction) \
        else _reactions.resolve(reaction)
    grid = tuple(int(v) for v in grid)
    if len(grid) == 2:
        grid = grid + (1,)
    ndim = 2 if grid[2] == 1 else 3

    cols = r.trunk_for(ndim)
    for c in extra_trunk_cols:
        if c not in cols:
            cols.append(c)

    info = dict(reaction=r.key, species=list(r.species),
                param_names=r.param_names,
                distance_convention=r.distance_convention,
                trunk_cols=list(cols), trunk_in_dim=len(cols),
                in_channels=int(in_channels), grid=list(grid), ndim=ndim,
                steady=r.steady, source=r.source, layout=layout)

    if layout == "published":
        if ndim != 2:
            raise ValueError(
                "the published key layout is a 2D layout: its trunk_net and "
                "its fc layer were fitted on a %d by %d grid. Build a 3D model "
                "with layout='ours' and warm start it from the published "
                "weights instead." % (64, 148))
        model = PRT2D(trunk_in_dim=len(cols), branch2_in_dim=r.n_params,
                      cnn_blocks=cnn_blocks, out_dim=out_dim,
                      num_layers=trunk_layers, width=trunk_width,
                      nx=grid[0], ny=grid[1])
    else:
        model = PRT_DeepONet3D(in_channels=in_channels, n_params=r.n_params,
                               out_dim=out_dim, trunk_in_dim=len(cols),
                               cnn_blocks=cnn_blocks, trunk_layers=trunk_layers,
                               trunk_width=trunk_width, grid=grid)
    return model, info


def load_published(reaction, weights_path, grid=(64, 148, 1), layout="published",
                   device="cpu", say=print):
    """Build the network for a published reaction and load its released file.

    With layout='published' the keys match the file exactly and the load is
    strict: anything missing is a real mismatch and must not pass quietly.
    With layout='ours' the same weights go into the 3D-capable network through
    remap_published(), which is the warm start.
    """
    r = reaction if isinstance(reaction, _reactions.Reaction) \
        else _reactions.resolve(reaction)
    if r.source != "published":
        raise ValueError(
            "%s is not one of the published reactions, so there is no released "
            "weight file for it. The published three are: %s."
            % (r.key, ", ".join(sorted(k for k, v in _reactions.REACTIONS.items()
                                       if v.source == "published"))))
    model, info = build(r, grid, layout=layout)
    model = model.to(device)

    try:
        state = torch.load(weights_path, map_location=device)
    except Exception:
        state = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))

    took, missing, clashes = load_state(model, state, strict=True, say=say)
    model.eval()
    info["weights"] = weights_path
    info["tensors_loaded"] = len(took)
    info["strict"] = not (missing or clashes)
    return model, info


def checkpoint_matches(info, ck, say=print):
    """Complaints about using a checkpoint for something it was not trained for.

    Returns a list of sentences, empty when everything agrees. The two that
    matter and used to pass silently: a different chemistry, which changes what
    the parameter columns mean, and a different distance convention, which
    hands the trunk a column running the other way.
    """
    bad = []
    for key, what in (("reaction", "chemistry"),
                      ("distance_convention", "distance convention"),
                      ("trunk_cols", "trunk columns")):
        a, b = info.get(key), ck.get(key)
        if a is None or b is None:
            continue                      # an older checkpoint simply does not
            # say, and refusing to run on that basis would break every file
            # written before this field existed
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            differ = list(a) != list(b)
        else:
            differ = a != b
        if differ:
            bad.append("this run uses the %s %r and the checkpoint was trained "
                       "with %r." % (what, a, b))
    if bad and say:
        for s in bad:
            say("  " + s)
    return bad


# =============================================================================
#  SELF TEST
# =============================================================================
def _self_test():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print("  %-5s %-58s %s" % ("PASS" if cond else "FAIL", name, extra))

    # the identity the whole transfer rests on
    b2d = BranchCNN3D(1, 128, 5, grid=(148, 64, 1))
    b3d = BranchCNN3D(1, 128, 5, grid=(64, 64, 64))
    check("148 x 64 and 64 cubed flatten to the same width",
          b2d.flat == b3d.flat == 2048, "%d and %d" % (b2d.flat, b3d.flat))
    check("and the 2D branch really is a Conv2d branch",
          b2d.two_d and isinstance(b2d.features[0], nn.Conv2d))
    check("while the 3D one is Conv3d",
          not b3d.two_d and isinstance(b3d.features[0], nn.Conv3d))

    # a reaction decides the widths, and nothing else does
    m3, i3 = build("monod", (64, 64, 64))
    m2, i2 = build("monod", (148, 64, 1))
    check("monod gives a two-input parameter branch",
          m3.branch2.net[0].in_features == 2)
    check("the trunk keeps z in 3D and loses it in 2D",
          i3["trunk_cols"] == ["x", "y", "z", "t", "gdf"]
          and i2["trunk_cols"] == ["x", "y", "t", "gdf"],
          "%s / %s" % (i3["trunk_cols"], i2["trunk_cols"]))
    mr, ir = build("reversible_sorption", (148, 64, 1))
    check("reversible sorption gives a three-input parameter branch",
          mr.branch2.net[0].in_features == 3)
    ms, isx = build("irreversible_sorption", (148, 64, 1))
    check("a steady reaction has no time column",
          "t" not in isx["trunk_cols"], str(isx["trunk_cols"]))
    check("and the checkpoint records which distance convention it is for",
          isx["distance_convention"] == "published_pore"
          and i2["distance_convention"] == "published")

    # the two key layouts hold the same tensors, shape for shape
    pub, _ = build("monod", (148, 64, 1), layout="published")
    ours, _ = build("monod", (148, 64, 1), layout="ours")
    mapped = remap_published(pub.state_dict())
    own = ours.state_dict()
    same = all(k in own and tuple(own[k].shape) == tuple(v.shape)
               for k, v in mapped.items())
    check("every published tensor has a counterpart of the same shape",
          same and len(mapped) == len(own),
          "%d published, %d ours" % (len(mapped), len(own)))

    # the remap loads strictly, which is the claim that matters
    took, missing, clashes = load_state(ours, pub.state_dict(), strict=True,
                                        say=None)
    check("a published state dict loads into our layout with nothing missing",
          not missing and not clashes and len(took) == len(own),
          "%d tensors" % len(took))

    # and the two networks then give the same numbers, which is the claim the
    # whole warm start rests on. The published forward is written out here
    # rather than called, because PRT2D reshapes to the full 9472-point grid
    # and this compares a sample of points instead.
    torch.manual_seed(0)
    ours.eval(); pub.eval()
    b1 = (torch.rand(2, 1, 148, 64, 1) > 0.4).float()
    pv = torch.randn(2, 2)
    tk = torch.rand(2, 512, 4)
    with torch.no_grad():
        ya = ours(b1, pv, tk)
        code = pub.branch1_net(b1[..., 0]) * pub.branch2_net(pv)     # (2, 128)
        basis = pub.trunk_net(tk.reshape(-1, 4)).view(2, 512, -1)    # (2,512,128)
        yb = (basis * code.unsqueeze(1)).sum(-1) + pub.bias
    d = float((ya - yb).abs().max())
    check("the remapped network reproduces the published forward pass",
          d < 1e-5, "largest difference %.2e" % d)

    # the round trip through the published names returns the same keys
    there = remap_published(pub.state_dict())
    back = remap_to_published(there)
    check("the key remap round trips",
          set(back) == set(pub.state_dict()),
          "%d keys" % len(back))

    # a 3D forward pass runs and is finite
    with torch.no_grad():
        y = m3(torch.rand(2, 1, 64, 64, 64), torch.randn(2, 2),
               torch.rand(2, 1024, 5))
    check("a 3D forward pass runs and is finite",
          tuple(y.shape) == (2, 1024) and bool(torch.isfinite(y).all()),
          "%s, %.2fM parameters" % (tuple(y.shape),
                                    count_parameters(m3) / 1e6))

    # velocity channels and an extra trunk column
    mv, iv = build("acetate_sulfate", (32, 32, 32), in_channels=4,
                   extra_trunk_cols=("tau",))
    check("the velocity switch widens the branch and the trunk",
          mv.branch1.features[0].in_channels == 4
          and iv["trunk_cols"][-1] == "tau", str(iv["trunk_cols"]))

    # asking for the published layout in 3D is refused with a reason
    check("the published layout is refused in 3D by name",
          "warm start" in _err(lambda: build("monod", (64, 64, 64),
                                             layout="published")))

    # a reaction with no released weights says so
    check("asking for released weights for our own chemistry is refused",
          "not one of the published"
          in _err(lambda: load_published("acetate_sulfate", "nope.pt")))

    print("\n%s" % ("Everything passed." if ok else "SOMETHING FAILED."))
    return 0 if ok else 1


def _err(fn):
    try:
        fn()
    except Exception as e:                                     # noqa: BLE001
        return str(e)
    return ""


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--reaction", default="acetate_sulfate")
    ap.add_argument("--grid", type=int, nargs=3, default=[64, 64, 64])
    a = ap.parse_args()
    if a.self_test:
        sys.exit(_self_test())
    m, info = build(a.reaction, a.grid)
    print("%s on %s" % (info["reaction"], "x".join(str(v) for v in info["grid"])))
    for k in ("ndim", "trunk_cols", "param_names", "distance_convention",
              "in_channels", "steady"):
        print("  %-22s %s" % (k, info[k]))
    print("  %-22s %.2fM" % ("parameters", count_parameters(m) / 1e6))
