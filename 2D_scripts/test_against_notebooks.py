"""
test_against_notebooks.py -- prove the port is a port.

For each of the three reactions this runs the published notebook in
2D/models/ cell by cell, then runs prt2d_model.py on the same weights and the
same domain, and compares. It checks three things separately, because they can
fail for different reasons:

    1. the geodesic distance column, which the three notebooks scale differently
    2. the trunk, branch1 and branch2 tensors this module builds
    3. the predicted field

A notebook is the reference here, not the other way round. If a number moves,
this test is what says so.

    python test_against_notebooks.py
    python test_against_notebooks.py --reaction monod

Exit status is 0 only if everything matched.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NBDIR = os.path.join(REPO, "2D", "models")
sys.path.insert(0, HERE)
import prt2d_model as M                                       # noqa: E402

NOTEBOOK = {
    "monod": "PRT-DeepONet_Monod_load.ipynb",
    "irreversible_sorption": "PRT-DeepONet_Irreversible_Sorption_load.ipynb",
    "reversible_sorption": "PRT-DeepONet_Reversible_Sorption_load.ipynb",
}


# =============================================================================
#  RUNNING THE NOTEBOOK ITSELF
#  Cell by cell, in one namespace, exactly as Jupyter would. Not converted to a
#  script first and not summarised: the whole point is that the thing on the
#  other side of the comparison is the released notebook and not our reading of
#  it.
# =============================================================================
def run_notebook(path):
    """Execute the notebook's code cells and hand back its namespace."""
    import matplotlib
    matplotlib.use("Agg")
    cells = ["".join(c["source"])
             for c in json.load(open(path))["cells"] if c["cell_type"] == "code"]
    cwd = os.getcwd()
    os.chdir(NBDIR)                       # its paths are ../parameters, ../geometries
    ns = {"__name__": "__main__"}
    try:
        for i, src in enumerate(cells):
            exec(compile(src, "%s:cell%d" % (os.path.basename(path), i), "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns


def close(a, b, tol):
    # Absolute difference, not relative: these fields are already on 0 to 1, and
    # a relative measure would make a difference near zero look enormous.
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False, "shape %s against %s" % (a.shape, b.shape)
    d = float(np.max(np.abs(a - b)))
    return d <= tol, "max difference %.3e" % d


# =============================================================================
#  THE FOUR STAGES COMPARED
#  The distance column, the three input tensors, the raw prediction, and the
#  notebook's own plotted output. Comparing only the last would hide a
#  disagreement that two errors cancelled; comparing only the first would miss
#  one introduced afterwards.
# =============================================================================
def check(reaction, verbose=True):
    cfg = M.CONFIGS[reaction]
    print("=" * 66)
    print(reaction)
    ns = run_notebook(os.path.join(NBDIR, NOTEBOOK[reaction]))
    ok = True

    # ---------------------------------------------------------------- 1. GDF
    m_bin = np.asarray(ns["m_bin"])
    gdf_ours = M.make_inlet_distance_norm(m_bin, cfg["gdf_scope"])
    good, msg = close(ns["GDF"], gdf_ours, 0.0)
    print("  distance column     %-6s %s" % ("MATCH" if good else "DIFFERS", msg))
    ok &= good

    # ------------------------------------------------------------ 2. tensors
    if reaction == "irreversible_sorption":
        conds = [tuple(c) for c in ns["pairs"]]
        t_norm = None
        nb_b1, nb_b2, nb_t = ns["B1_t"], ns["B2_t"], ns["T_t"]
    else:
        conds = [tuple(c) for c in (ns["COND_PAIRS"] if reaction == "monod"
                                    else ns["COND_TRIPLES"])]
        # the notebooks plot a subset of the times; Y_seq[0] is the first of
        # those, not the first of the full ladder
        shown = ns.get("T_NORMS_SHOW", ns["T_NORMS"])
        t_norm = float(shown[0])
        nb_b1, nb_b2, nb_t = ns["build_inputs_for_time"](t_norm)

    b1, b2, trunk = M.build_inputs(reaction, m_bin, conds, t_norm=t_norm,
                                   gdf=gdf_ours)
    for name, mine, theirs in (("geometry input", b1, nb_b1),
                               ("parameter input", b2, nb_b2),
                               ("trunk input", trunk, nb_t)):
        good, msg = close(mine.numpy(), np.asarray(theirs), 0.0)
        print("  %-19s %-6s %s" % (name, "MATCH" if good else "DIFFERS", msg))
        ok &= good

    # ---------------------------------------------------------- 3. prediction
    weights, _ = M.default_paths(REPO, reaction)
    model, missing, unexpected = M.load_model(reaction, weights)
    if missing or unexpected:
        print("  weights             DIFFERS %d missing, %d unexpected"
              % (len(missing), len(unexpected)))
        ok = False
    ours = M.predict(model, b1, b2, trunk)

    with torch.no_grad():
        theirs = ns["model"](nb_b1, nb_b2, nb_t).cpu().numpy()[..., 0]
    good, msg = close(ours, theirs, 1e-6)
    print("  prediction          %-6s %s" % ("MATCH" if good else "DIFFERS", msg))
    ok &= good

    # and against what the notebook itself reported, end to end
    # the notebooks clip to 0 and 1 and zero the solid before plotting, and
    # irreversible_sorption clips only, so compare through the same convention
    if reaction == "irreversible_sorption":
        nb_field = np.asarray(ns["Y_pred"])
        good, msg = close(np.clip(ours, 0.0, 1.0), nb_field, 1e-6)
    else:
        nb_field = np.asarray(ns["Y_seq"])[0]
        good, msg = close(M.as_displayed(ours, m_bin), nb_field, 1e-6)
        msg += "   (t = %.3f, the first time the notebook plots)" % t_norm
    print("  notebook output     %-6s %s" % ("MATCH" if good else "DIFFERS", msg))
    ok &= good
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reaction", choices=M.REACTIONS)
    a = p.parse_args()
    which = [a.reaction] if a.reaction else list(M.REACTIONS)
    results = {r: check(r) for r in which}
    print("=" * 66)
    for r, ok in results.items():
        print("  %-24s %s" % (r, "all match" if ok else "SOMETHING DIFFERS"))
    if not all(results.values()):
        print("\nThe port and the notebooks disagree. Fix the port, not the test.")
        sys.exit(1)
    print("\nThe scripts reproduce the published notebooks exactly.")


if __name__ == "__main__":
    main()
