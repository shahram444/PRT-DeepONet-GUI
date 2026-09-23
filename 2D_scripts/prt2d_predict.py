"""
prt2d_predict.py -- run the published 2D PRT-DeepONet from the command line.

The same prediction the notebooks in 2D/models/ make, without Jupyter. It loads
a released checkpoint, builds the geometry, parameter and coordinate inputs, and
writes the predicted concentration field as a .npz and a .png.

    python prt2d_predict.py --reaction monod
    python prt2d_predict.py --reaction monod --pe 10 --da 0.5 --times 0.25 0.5 1.0
    python prt2d_predict.py --reaction irreversible_sorption --all-conditions
    python prt2d_predict.py --reaction reversible_sorption --domain my_domain.npz

With no --pe/--da the published condition grid is used, which is what the
notebooks show. With no --times the published time ladder is used, and for
irreversible_sorption there is no time at all because it is a steady state.

Nothing is invented here: every default comes from CONFIGS in prt2d_model.py,
which was read off the notebooks.
"""

import argparse
import itertools
import os
import sys

import numpy as np
import torch

# The repository root is one directory up, which is where 2D/parameters and
# 2D/geometries live.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prt2d_model as M                                       # noqa: E402

REPO = os.path.dirname(HERE)


# =============================================================================
#  THE ARGUMENTS
#  Deliberately the same names the 3D side uses: --geometry, --out, --reaction.
#  A person who has run the 3D predictor should not have to learn a second
#  spelling to run the published 2D one.
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Predict a 2D concentration field with the published model.")
    p.add_argument("--reaction", required=True, choices=M.REACTIONS)
    p.add_argument("--weights", help="a .pt file; default is the released one")
    p.add_argument("--domain", help="a .npz domain, or a .dat from Input_domains.zip")
    p.add_argument("--pe", type=float, nargs="*", help="Peclet values")
    p.add_argument("--da", type=float, nargs="*",
                   help="Damkohler values; for reversible_sorption give Da_A here")
    p.add_argument("--da-d", type=float, nargs="*",
                   help="the desorption Damkohler, reversible_sorption only")
    p.add_argument("--all-conditions", action="store_true",
                   help="every combination of the published grid, which is the default")
    p.add_argument("--times", type=float, nargs="*",
                   help="normalised times between 0 and 1; ignored for a steady state")
    p.add_argument("--out", default="prediction", help="output prefix")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-png", action="store_true")
    return p.parse_args()


# =============================================================================
#  WHICH CONDITIONS TO PREDICT AT
#  Either what was asked for, or the grid of values the notebook itself plots.
#  The second is the default because it reproduces the published figure, which
#  is the first thing anybody wants to see from this script.
# =============================================================================
def conditions_from(args, cfg):
    """The list of raw condition tuples to predict at."""
    grid = list(cfg["param_grid"])
    given = [args.pe, args.da, args.da_d][:len(grid)]
    axes = []
    for g, supplied in zip(grid, given):
        axes.append(tuple(supplied) if supplied else tuple(g))
    return [tuple(c) for c in itertools.product(*axes)]


# =============================================================================
#  ONE FORWARD PASS, AND WHAT IS WRITTEN
#  The RAW field is what is saved. Clipping to 0 and 1 and zeroing the solid is
#  the notebooks' DISPLAY convention, applied only to the picture, because a
#  saved field that has been clipped cannot be compared with a simulation.
# =============================================================================
def main():
    args = parse_args()
    cfg = M.CONFIGS[args.reaction]

    w_default, d_default = M.default_paths(REPO, args.reaction)
    weights = args.weights or w_default
    domain = args.domain or d_default
    for path, what in ((weights, "weights"), (domain, "domain")):
        if not os.path.exists(path):
            sys.exit("no %s at %s\nPass --%s, or run this from inside the "
                     "repository so the released files can be found." %
                     (what, path, what))

    m_bin = M.read_dat(domain) if domain.endswith(".dat") else M.read_domain(domain)
    gdf = M.make_inlet_distance_norm(m_bin, cfg["gdf_scope"])

    model, missing, unexpected = M.load_model(args.reaction, weights, args.device)
    print("reaction    : %s" % args.reaction)
    print("weights     : %s" % weights)
    print("              %d missing, %d unexpected tensors" %
          (len(missing), len(unexpected)))
    if missing or unexpected:
        print("              the checkpoint and the architecture do not match. "
              "Stopping would be safer than trusting this.")
    print("domain      : %s   porosity %.3f" % (domain, float((m_bin == 1).mean())))
    print("trunk       : %s" % ", ".join(cfg["trunk_cols"]))
    print("parameters  : %s" % ", ".join(cfg["param_names"]))

    conds = conditions_from(args, cfg)
    print("conditions  : %d" % len(conds))
    for c in conds[:8]:
        print("              " + ", ".join("%s=%g" % (n, v)
                                           for n, v in zip(cfg["param_names"], c)))
    if len(conds) > 8:
        print("              ... and %d more" % (len(conds) - 8))

    steady = cfg["times"] is None
    times = [None] if steady else (list(args.times) if args.times else list(cfg["times"]))
    if not steady:
        print("times       : %d, from %.3f to %.3f" % (len(times), times[0], times[-1]))

    fields = []
    for t in times:
        b1, b2, trunk = M.build_inputs(args.reaction, m_bin, conds, t_norm=t, gdf=gdf)
        fields.append(M.predict(model, b1, b2, trunk, args.device))
    pred = np.stack(fields, axis=0)                     # (T, N, 64, 148)
    if steady:
        pred = pred[0]                                  # (N, 64, 148)

    shown = M.as_displayed(pred, m_bin)

    out_npz = args.out + ".npz"
    np.savez_compressed(
        out_npz, prediction=pred.astype(np.float32),
        prediction_display=shown, material=m_bin.astype(np.uint8),
        gdf=gdf.astype(np.float32),
        conditions=np.array(conds, dtype=np.float32),
        param_names=np.array(cfg["param_names"]),
        times=np.array([] if steady else times, dtype=np.float32),
        reaction=np.array(args.reaction))
    print("\nprediction  : %s   %s" % (out_npz, tuple(pred.shape)))
    print("              %s" % ("(condition, x, y)" if steady
                                else "(time, condition, x, y)"))
    print("              'prediction' is the raw field; 'prediction_display' is\n"
          "              the same field clipped to 0 and 1 with the solid zeroed,\n"
          "              which is what the published figures show")

    if not args.no_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        last = shown if steady else shown[-1]
        n = len(conds)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.2 * nrow),
                                 squeeze=False, constrained_layout=True)
        for i, ax in enumerate(axes.ravel()):
            if i >= n:
                ax.axis("off")
                continue
            f = np.where(m_bin == 1, last[i], np.nan)   # blank the grains
            im = ax.imshow(f.T, origin="lower", cmap="viridis",
                           interpolation="nearest", aspect="auto")
            ax.set_title(", ".join("%s=%g" % (a, b)
                                   for a, b in zip(cfg["param_names"], conds[i])),
                         fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.03)
        fig.suptitle("%s%s" % (args.reaction,
                               "" if steady else "   t = %.3f" % times[-1]),
                     fontsize=11)
        fig.savefig(args.out + ".png", dpi=150)
        print("figure      : %s.png" % args.out)


if __name__ == "__main__":
    main()
