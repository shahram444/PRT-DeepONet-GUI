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

WHAT MATCHES AND WHAT DOES NOT
    branch1  matches exactly.  Note 2048 = 256 x (148/32) x (64/32), so it is
             tied to the 148 x 64 grid; a different grid changes that number and
             the fc layer will not load.
    trunk    matches exactly, INCLUDING the input width of 4, which is
             (x, y, t, GDF) -- precisely what our 2D mode builds.  This is the
             single most useful fact here: the geometry-sensing part of his
             network drops straight into ours.
    branch2  his takes 2 numbers, ours takes 6.  Loadable only if we also run
             with two parameters.
    output   his network predicts ONE species and has no head layer, because
             with one species the branch code is already the coefficient
             vector.  Ours always has a head.  So the head is always freshly
             initialised, whatever else transfers.

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
#  BLOCK 1.  THEIR KEY NAMES, TO OURS
#
#  Both networks have the same eight trunk layers in the same order, so the
#  mapping is POSITIONAL: their trunk_net.0 is our trunk.first, their
#  trunk_net.14 is our trunk.last, and the even indices between are our hidden
#  list in order. The odd indices are activations and carry no weights.
#
#  A key that matches nothing is dropped silently HERE and reported by the
#  caller, which counts what arrived. Raising on the first unknown key would
#  stop the conversion on a difference that may not matter.
# =============================================================================
def remap(src):
    """His key names -> ours.  Both networks have the same eight trunk layers,
    so the mapping is positional and complete."""
    out = {}
    # 0 and 14 are the first and last Linear in their Sequential. The odd
    # indices between are activations and hold no weights at all.
    trunk_pos = {0: "trunk.first", 14: "trunk.last"}
    # Six hidden layers, at the even indices, in order.
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
            # Deliberately dropped. Theirs is one number for one chemical; ours
            # is one per species. Broadcasting it would start every chemical at
            # the same offset, which is a wrong initialisation dressed up as a
            # loaded weight.
            pass                       # his is (1,), ours is (n_species,)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="one of 2D/parameters/*.pt")
    ap.add_argument("--grid", type=int, nargs=2, default=[148, 64],
                    help="the 2D grid his fc layer was trained on")
    ap.add_argument("--n-species", type=int, default=1)
    ap.add_argument("--n-params", type=int, default=2,
                    help="his parameter branch takes 2 (Pe and Da). Use 6 to "
                         "match our datasets, and his branch2 will be skipped.")
    ap.add_argument("--save", default=None,
                    help="write a checkpoint our train.py can --init-from")
    a = ap.parse_args()

    src = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(src, dict):
        sys.exit("that file does not hold a state dict")
    if "model" in src and isinstance(src["model"], dict):
        src = src["model"]

    print("published checkpoint : %s" % a.checkpoint)
    print("  tensors            : %d" % len(src))

    nx, ny = a.grid
    model = PRT_DeepONet3D(in_channels=1, n_params=a.n_params,
                           n_species=a.n_species, trunk_in_dim=4,
                           grid=(nx, ny, 1), inject_every=0)
    print("our model in 2D mode : grid %d x %d x 1, trunk 4 inputs "
          "(x, y, t, gdf), %.2fM parameters"
          % (nx, ny, count_parameters(model) / 1e6))

    mapped = remap(src)
    own = model.state_dict()
    took, shape_clash, unknown = {}, [], []
    for k, v in mapped.items():
        if k not in own:
            unknown.append(k)
        elif tuple(own[k].shape) != tuple(v.shape):
            shape_clash.append((k, tuple(v.shape), tuple(own[k].shape)))
        else:
            took[k] = v
    missing = [k for k in own if k not in took]

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

    model.load_state_dict(took, strict=False)

    # a forward pass, because "the shapes match" is not the same as "it runs"
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
                    "species": ["C"][:a.n_species] if a.n_species == 1
                               else ["s%d" % i for i in range(a.n_species)],
                    "param_names": ["pe", "da"][:a.n_params] if a.n_params == 2
                                   else ["p%d" % i for i in range(a.n_params)],
                    "trunk_in_dim": 4, "with_time": True, "n_times": 1,
                    "in_channels": 1, "grid": [nx, ny, 1],
                    "source": "warm start from " + os.path.basename(a.checkpoint)},
                   a.save)
        print("\nwrote %s" % a.save)
        print("use it with:")
        print("  python ../model/train.py --data <2d dataset.h5> \\")
        print("         --init-from %s --freeze-trunk" % a.save)
        print("\nThe dataset must match: %d x %d grid, %d species, %d parameters."
              % (nx, ny, a.n_species, a.n_params))

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
