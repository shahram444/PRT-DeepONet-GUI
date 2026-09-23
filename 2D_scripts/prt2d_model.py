"""
prt2d_model.py -- the published 2D PRT-DeepONet, as a module instead of a notebook.

WHAT CHANGED FROM THE 2D VERSION
    Nothing about the model, deliberately. What changed is that it can now be run
    without Jupyter, which is what a cluster needs, and that the three reactions'
    differences are FIELDS in a table rather than numbers typed into three
    separate notebook cells.

    The one difference that had been invisible and changes the numbers: the
    geodesic distance is scaled over the WHOLE grid for monod and
    reversible_sorption, and over the PORE ONLY for irreversible_sorption. It is
    gdf_scope below, and prt_core/conventions.py names both.

This is a port, not a reimplementation. Every layer, every constant and every
normalisation is taken from the three notebooks in 2D/models/, and the port is
checked against them: test_against_notebooks.py runs the notebooks and this
module on the same weights and the same domain and requires the predictions to
agree exactly. Nothing here is an improvement on the published model, and that
is deliberate.

The three reactions differ in more than their names, so each one has its own
entry in CONFIGS:

    reaction                convolution   trunk            parameter branch
    ---------------------------------------------------------------------------
    monod                   5 blocks      x, y, t, gdf     Pe, Da
    irreversible_sorption   5 blocks      x, y, gdf        Pe, Da_A       steady
    reversible_sorption     4 blocks      x, y, t, gdf     Pe, Da_A, Da_D

The class and attribute names branch1_net, branch2_net, trunk_net and bias are
the names inside the released .pt files. They are not ours to rename.

WHERE THE NETWORK ITSELF NOW LIVES
    prt_core/model.py. It used to be written out below, and the same
    architecture was written out again in 3D/model/deeponet_model.py and once
    more in each notebook. This file now re-exports the core's modules under
    the names the notebooks and the port use, so nothing that imports from here
    changed and there is one definition behind it.

    test_against_notebooks.py is what holds that claim to account, and after the
    move it still reports a difference of exactly zero for all three reactions.
"""

import os
import sys
from collections import deque

import numpy as np
import torch

# prt_core sits at the repository root, one directory up from 2D_scripts/.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from prt_core.model import (BranchCNN, BranchFNN, PRT2D,       # noqa: E402,F401
                            create_trunk_net)

# =============================================================================
#  THE PUBLISHED GRID
#  64 by 148, and the encoder's fully connected layer is sized by it: halved
#  five times that is 2 by 4 at 256 channels, which is the 2048 the released
#  weights expect. A different grid changes that number and the weights stop
#  loading, so this is not a default that can be edited.
# =============================================================================
NX, NY = 64, 148          # the published grid; the encoder is sized by it
L = NX * NY               # 9472 points


# ------------------------------------------------------------------- configs
CONFIGS = {
    "monod": {
        "weights": "Monod.pt",
        "domain": "Domain_Monod.npz",
        "cnn_blocks": 5,
        "trunk_in_dim": 4,
        "trunk_cols": ("x", "y", "t", "gdf"),
        "param_names": ("Pe", "Da"),
        "param_ranges": ((1.0, 10.0), (0.05, 0.5)),
        "param_grid": ((1.0, 10.0), (0.05, 0.5)),
        "times": tuple(np.array([1] + list(range(4, 81, 4)), dtype=float) / 80.0),
        "gdf_scope": "full",
    },
    "irreversible_sorption": {
        "weights": "Irreversible_Sorption.pt",
        "domain": "Domain_Irreversible_Sorption.npz",
        "cnn_blocks": 5,
        "trunk_in_dim": 3,
        "trunk_cols": ("x", "y", "gdf"),
        "param_names": ("Pe", "Da_A"),
        "param_ranges": ((1.0, 10.0), (74.0, 740.0)),
        "param_grid": ((1.0, 5.0, 10.0), (74.0, 296.0, 740.0)),
        "times": None,                       # steady state: no time column
        "gdf_scope": "pore",
    },
    "reversible_sorption": {
        "weights": "Reversible_Sorption.pt",
        "domain": "Domain_Reversible_Sorption.npz",
        "cnn_blocks": 5,          # the class default in the notebook is 4, but it
                                  # is instantiated with 5: the checkpoint's fc
                                  # layer is 256 x 2 x 4 = 2048 wide
        "trunk_in_dim": 4,
        "trunk_cols": ("x", "y", "t", "gdf"),
        "param_names": ("Pe", "Da_A", "Da_D"),
        "param_ranges": ((1.0, 10.0), (0.5, 2.0), (0.02, 0.2)),
        "param_grid": ((1.0, 10.0), (0.5, 2.0), (0.02, 0.2)),
        "times": tuple(np.array([1] + list(range(20, 101, 20)), dtype=float) / 100.0),
        "gdf_scope": "full",
    },
}

REACTIONS = tuple(CONFIGS)


# ------------------------------------------------------- geometry and the GDF
def boundary_distance_map(m_bin, boundary="inlet"):
    """Four-neighbour geodesic distance through the pore from one face.

    The seeds are at 0.5 and every pore step adds 1.0, which is the release's
    convention. Solid is never visited.
    """
    nx, ny = m_bin.shape
    dist_map = np.zeros_like(m_bin, dtype=np.float32)

    if boundary == "inlet":
        starts = [(i, 0) for i in range(nx) if m_bin[i, 0] == 1]
    else:
        starts = [(i, ny - 1) for i in range(nx) if m_bin[i, ny - 1] == 1]

    q = deque()
    for (x, y) in starts:
        dist_map[x, y] = 0.5
        q.append((x, y))

    for_x, for_y = [-1, 1, 0, 0], [0, 0, -1, 1]
    while q:
        x, y = q.popleft()
        for dx, dy in zip(for_x, for_y):
            xx, yy = x + dx, y + dy
            if 0 <= xx < nx and 0 <= yy < ny and m_bin[xx, yy] == 1:
                new_dist = dist_map[x, y] + 1.0
                if dist_map[xx, yy] == 0 or dist_map[xx, yy] > new_dist:
                    dist_map[xx, yy] = new_dist
                    q.append((xx, yy))
    return dist_map


def make_inlet_distance_norm(m_bin, scope="full"):
    """The trunk's distance column, high at the inlet and zero in the solid.

    scope decides what the scaling is computed over, and the two published
    variants disagree. 'full' uses the whole grid, so the solid, which has just
    been set to one more than the furthest pore and then negated, becomes the
    minimum and pins the bottom of the range. 'pore' uses the pore only. monod
    and reversible_sorption are 'full'; irreversible_sorption is 'pore'.
    """
    dist_inlet = boundary_distance_map(m_bin, boundary="inlet")
    max_val = float(np.nanmax(dist_inlet[m_bin == 1])) if (m_bin == 1).any() else 0.0
    dist_inlet[m_bin == 0] = max_val + 1.0
    dist_inlet = -dist_inlet

    gdf = np.zeros_like(dist_inlet, dtype=np.float32)
    if scope == "pore":
        valid = (m_bin == 1) & np.isfinite(dist_inlet)
        if np.any(valid):
            vmin = float(dist_inlet[valid].min())
            vmax = float(dist_inlet[valid].max())
            gdf[valid] = ((dist_inlet[valid] - vmin) / (vmax - vmin)
                          if vmax > vmin else 1.0)
    else:
        valid = np.isfinite(dist_inlet)
        vmin = float(dist_inlet[valid].min())
        vmax = float(dist_inlet[valid].max())
        gdf[valid] = ((dist_inlet[valid] - vmin) / (vmax - vmin)
                      if vmax > vmin else 0.0)
    gdf[m_bin == 0] = 0.0
    return gdf


# =============================================================================
#  READING A DOMAIN
#  Two formats, both from the release: the .npz domains that ship beside the
#  weights, and the 3000 raw .dat files in Domains/Input_domains.zip. Both are
#  9472 values, 1 for pore and 0 for solid, and both come back as the same
#  64 by 148 array of ones and zeros.
# =============================================================================
def read_domain(path):
    """The released domains are a flat array of 9472 zeros and ones."""
    arr = np.load(path)
    flat = arr["m"] if "m" in arr.files else arr[list(arr.files)[0]]
    return (flat.reshape(NX, NY) > 0.5).astype(np.int32)


def read_dat(path):
    """One of the 3000 domains from Domains/Input_domains.zip."""
    raw = np.loadtxt(path).ravel()
    if raw.size != L:
        raise ValueError("expected %d values, got %d" % (L, raw.size))
    return (raw.reshape(NX, NY) > 0.5).astype(np.int32)


# ----------------------------------------------------------------- the inputs
def normalise_params(reaction, values):
    """Map the raw dimensionless numbers onto 0 to 1 with the training range."""
    cfg = CONFIGS[reaction]
    out = []
    for v, (lo, hi) in zip(values, cfg["param_ranges"]):
        out.append((float(v) - lo) / (hi - lo))
    return out


def build_inputs(reaction, m_bin, conditions, t_norm=None, gdf=None):
    """Return branch1, branch2 and trunk for one time and a list of conditions.

    conditions is a list of tuples, one value per entry in param_names.
    """
    cfg = CONFIGS[reaction]
    if gdf is None:
        gdf = make_inlet_distance_norm(m_bin, cfg["gdf_scope"])

    n = len(conditions)
    b1 = np.repeat(m_bin[None, None, ...], n, axis=0).astype(np.float32)

    b2 = np.array([normalise_params(reaction, c) for c in conditions],
                  dtype=np.float32)

    xs = np.arange(NX, dtype=np.float32) / (NX - 1)
    ys = np.arange(NY, dtype=np.float32) / (NY - 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    cols = [X.reshape(-1, 1), Y.reshape(-1, 1)]
    if "t" in cfg["trunk_cols"]:
        if t_norm is None:
            raise ValueError("%s needs a time; pass t_norm" % reaction)
        cols.append(np.full((L, 1), float(t_norm), dtype=np.float32))
    cols.append(gdf.reshape(-1, 1))
    trunk_single = np.concatenate(cols, axis=1).astype(np.float32)
    trunk = np.repeat(trunk_single[None, None, ...], n, axis=0)

    return (torch.tensor(b1), torch.tensor(b2), torch.tensor(trunk))


# ------------------------------------------------------------------- loading
def load_model(reaction, weights_path, device="cpu"):
    """Build the network for one reaction and load the released weights.

    The load is checked: a missing or unexpected key means the architecture and
    the checkpoint have drifted apart, and the caller is told rather than left
    with a network that is partly random.
    """
    cfg = CONFIGS[reaction]
    model = PRT2D(trunk_in_dim=cfg["trunk_in_dim"],
                  branch2_in_dim=len(cfg["param_names"]),
                  cnn_blocks=cfg["cnn_blocks"]).to(device)

    try:
        state = torch.load(weights_path, map_location=device)
    except Exception:
        state = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, list(missing), list(unexpected)


@torch.no_grad()
def predict(model, b1, b2, trunk, device="cpu"):
    """One forward pass. Returns (N, 64, 148)."""
    y = model(b1.to(device), b2.to(device), trunk.to(device)).cpu()
    return y[..., 0].contiguous().numpy()


def as_displayed(field, m_bin):
    """The notebooks' display convention, kept separate from the prediction.

    Every released notebook clips the field to 0 and 1 and sets the solid to
    zero before plotting. That is presentation, not model output, so predict()
    returns the raw field and this applies the convention when it is wanted.
    Comparing with a published figure means comparing with this.
    """
    out = np.clip(np.asarray(field), 0.0, 1.0)
    return np.where(m_bin.astype(bool), out, 0.0).astype(np.float32)


def default_paths(repo_root, reaction):
    """Where the released weights and domain sit, relative to the repository."""
    cfg = CONFIGS[reaction]
    return (os.path.join(repo_root, "2D", "parameters", cfg["weights"]),
            os.path.join(repo_root, "2D", "geometries", cfg["domain"]))
