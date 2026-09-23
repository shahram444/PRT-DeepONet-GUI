#!/usr/bin/env python3
"""
load_pretrained_2d_weights.py — load the PUBLISHED 2D trained network into our model.

This answers Christof's todo item directly:

    "discuss with Heewon if we can just look at one geometry and use his GDF
     weights(?), implementation(?) directly, so that we can focus on the
     reactions only."

The answer, measured rather than assumed: his branch CNN and his trunk are
SHAPE-IDENTICAL to ours in 2D mode, and transfer with no surgery at all.

WHAT IS IN HIS CHECKPOINT
    branch1_net   Conv2d 1->16->32->64->128->256, then fc 2048 -> 128
    branch2_net   Linear 2 -> 128 -> 128 -> 128          (2 inputs: Pe and Da)
    trunk_net     Linear 4 -> 128, six 128 -> 128, 128 -> 128   (8 layers)
    bias          one number                              (ONE species)

WHAT MATCHES
    Everything.  Our model is his architecture lifted one dimension, so in 2D
    mode with two parameters every tensor in his checkpoint has a
    shape-identical counterpart in ours and the load is exact.
    branch1  matches exactly.  Note 2048 = 256 x (148/32) x (64/32), so it is
             tied to the 148 x 64 grid; a different grid changes that number and
             the fc layer will not load.
    trunk    matches exactly, INCLUDING the input width of 4, which is
             (x, y, t, GDF) -- precisely what our 2D mode builds.
    branch2  matches at --n-params 2, which is his Pe and Da.  A dataset with a
             third dimensionless group needs --n-params 3, and then his first
             parameter-branch layer is skipped and it says so.
    bias     one number in his checkpoint and one number in ours, because both
             networks predict ONE field.

WHAT THIS IS AND IS NOT
    It is a way to start from a geometry encoder that already works, and spend
    our effort on the reactions.
    It is NOT his training data.  The release ships geometries, weights and
    notebooks -- no concentration fields and no flow fields.  Training on his
    SIMULATIONS needs him to send them; see HEEWON_DATA.md for exactly what to
    ask for.

USAGE
    python load_pretrained_2d_weights.py --checkpoint ../../2D/parameters/Monod.pt
    python load_pretrained_2d_weights.py --checkpoint ../../2D/parameters/Monod.pt \\
                                  --save warmstart.pt --grid 148 64
    # then
    python ../model/train.py --data data2d.h5 --init-from warmstart.pt --freeze-trunk
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
from deeponet_model import PRT_DeepONet3D, count_parameters                # noqa: E402


# =============================================================================
#  THEIR KEY NAMES TO OURS
#  The released files hold branch1_net, branch2_net and a trunk_net Sequential.
#  Ours hold branch1, branch2 and a trunk with named layers. Both are the same
#  eight trunk layers in the same order, so the mapping is positional and
#  complete. The mapping itself now lives in prt_core/model.py, where it is
#  checked round trip; this is the copy the command line uses.
# =============================================================================
def remap(src):
    """His key names -> ours.  Both networks have the same eight trunk layers,
    so the mapping is positional and complete."""
    out = {}
    trunk_pos = {0: "trunk.first", 14: "trunk.last"}
    for i, n in enumerate((2, 4, 6, 8, 10, 12)):
        trunk_pos[n] = "trunk.hidden.%d" % i
    for k, v in src.items():
        if k.startswith("branch1_net."):
            out["branch1." + k[len("branch1_net."):]] = v
        elif k.startswith("branch2_net."):
            out["branch2." + k[len("branch2_net."):]] = v
        elif k.startswith("trunk_net."):
            rest = k[len("trunk_net."):]
            idx, _, tail = rest.partition(".")
            tgt = trunk_pos.get(int(idx))
            if tgt:
                out["%s.%s" % (tgt, tail)] = v
        elif k == "bias":
            out["bias"] = v
    return out


# =============================================================================
#  LOADING, AND SAYING WHAT HAPPENED
#  At two parameters every tensor in the released file has a counterpart of the
#  same shape, so the load is STRICT and anything missing is a real mismatch.
#  At three the first parameter-branch layer cannot fit and is skipped, out
#  loud. A warm start that quietly loaded most of a network would be worse than
#  one that refused.
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="one of 2D/parameters/*.pt")
    ap.add_argument("--grid", type=int, nargs=2, default=[148, 64],
                    help="the 2D grid his fc layer was trained on")
    ap.add_argument("--n-params", type=int, default=2,
                    help="his parameter branch takes 2 (Pe and Da), which is "
                         "also our default. Use 3 if your dataset carries a "
                         "third dimensionless group, and his branch2 first "
                         "layer will be skipped.")
    ap.add_argument("--species", default="C", metavar="NAME",
                    help="the chemical this warm start is for. Recorded in the "
                         "saved checkpoint; the model predicts one field, so "
                         "one warm start belongs to one species.")
    ap.add_argument("--save", default=None,
                    help="write a checkpoint our train.py can --init-from")
    a = ap.parse_args()

    # weights_only=False because the released files are plain pickles saved
    # before torch made the safe loader the default.
    src = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(src, dict):
        sys.exit("that file does not hold a state dict")
    # A released file is a bare state dict; one of ours wraps it under "model".
    # Both are accepted so a warm start can be chained from either.
    if "model" in src and isinstance(src["model"], dict):
        src = src["model"]

    print("published checkpoint : %s" % a.checkpoint)
    print("  tensors            : %d" % len(src))

    nx, ny = a.grid
    model = PRT_DeepONet3D(in_channels=1, n_params=a.n_params, trunk_in_dim=4,
                           grid=(nx, ny, 1))
    print("our model in 2D mode : grid %d x %d x 1, trunk 4 inputs "
          "(x, y, t, gdf), %.2fM parameters"
          % (nx, ny, count_parameters(model) / 1e6))

    mapped = remap(src)          # their names to ours, before anything is compared
    own = model.state_dict()
    took, shape_clash, unknown = {}, [], []
    for k, v in mapped.items():
        if k not in own:
            unknown.append(k)
        elif tuple(own[k].shape) != tuple(v.shape):
            shape_clash.append((k, tuple(v.shape), tuple(own[k].shape)))
        else:
            took[k] = v
    missing = [k for k in own if k not in took]     # ours with no counterpart

    print("\n%-46s %s" % ("TRANSFERRED", "%d tensors" % len(took)))
    for grp in ("branch1", "branch2", "trunk"):
        n = sum(1 for k in took if k.startswith(grp))
        tot = sum(1 for k in own if k.startswith(grp))
        mark = "all" if n == tot else "%d of %d" % (n, tot)
        print("   %-10s %s" % (grp, mark))

    if shape_clash:
        print("\nSHAPE MISMATCH, not loaded:")
        for k, a_, b_ in shape_clash:
            print("   %-34s his %s   ours %s" % (k, a_, b_))
    if unknown:
        print("\nNOT PRESENT IN OUR MODEL: %s" % ", ".join(unknown))
    fresh = [k for k in missing if not k.endswith("num_batches_tracked")]
    if fresh:
        print("\nFRESHLY INITIALISED (his network has no counterpart):")
        for k in fresh:
            print("   %-34s %s" % (k, tuple(own[k].shape)))

    # With one scalar bias and a two-input parameter branch every tensor he
    # ships has a counterpart of the same shape, so at --n-params 2 the load is
    # STRICT: anything missing is a real mismatch and must not pass quietly.
    if not missing and not shape_clash and not unknown:
        model.load_state_dict(took, strict=True)
        print("\nstrict load             : every tensor matched")
    else:
        model.load_state_dict(took, strict=False)

    # a forward pass, because "the shapes match" is not the same as "it runs"
    # A forward pass, because "the shapes match" and "it runs" are different
    # claims and only the second one is worth reporting.
    b1 = torch.zeros(1, 1, nx, ny, 1)
    b1[0, 0, :, :, 0] = torch.rand(nx, ny) > 0.4
    b2 = torch.randn(1, a.n_params)
    tk = torch.rand(1, 256, 4)
    with torch.no_grad():
        y = model(b1, b2, tk)
    ok = torch.isfinite(y).all().item()
    print("\nforward pass          : output %s, all finite: %s"
          % (tuple(y.shape), ok))

    if a.save:
        torch.save({"model": model.state_dict(),
                    "args": {"distance": "gdf", "with_velocity": False},
                    "species": a.species,
                    "param_names": ["pe", "da"] if a.n_params == 2
                                   else ["p%d" % i for i in range(a.n_params)],
                    "trunk_in_dim": 4, "with_time": True, "n_times": 1,
                    "in_channels": 1, "grid": [nx, ny, 1],
                    "source": "warm start from " + os.path.basename(a.checkpoint)},
                   a.save)
        print("\nwrote %s" % a.save)
        print("use it with:")
        print("  python ../model/train.py --data <2d dataset.h5> \\")
        print("         --init-from %s --freeze-trunk" % a.save)
        print("\nThe dataset must match: %d x %d grid, %d parameters, and be "
              "trained with --species %s." % (nx, ny, a.n_params, a.species))

    print("""
WHAT THIS DOES AND DOES NOT GIVE YOU
   It gives a geometry encoder and a trunk that already know how to read a
   pore structure and a geodesic distance field, so training starts from
   something that works rather than from noise.
   It does NOT give his simulation data. The release has no concentration
   fields and no flow fields. See HEEWON_DATA.md for what to ask him for.""")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
