#!/usr/bin/env python3
"""
dataset_reader.py — PyTorch Dataset over dataset_reader.h5, shaped exactly the way
the 3D PRT-DeepONet consumes it.  This is the "ready for the model" layer.

Each item is the 4-tuple the 2D notebook's model already expects, lifted to 3D:

    branch1  (Cin, nx, ny, nz)   geometry  (+ velocity channels if requested)
    branch2  (n_params,)         dimensionless numbers
    trunk    (n_points, 5)       (x, y, z, t_norm, gdf)  at sampled pore voxels
    target   (n_points, n_species)

Why the trunk is SAMPLED, not full-grid
---------------------------------------
The 2D model evaluates the trunk at all 9,472 grid points per sample.  The same
thing at 64^3 is 262,144 points, which is a 26.8 GB activation tensor at batch 25
and ~30x the compute.  Drawing 8,192 random PORE voxels per sample per step costs
944 MMACs — slightly LESS than the 2D model's 1,091 — and keeps memory at ~0.3 GB.
Solid voxels are excluded because there is nothing to predict there.

At evaluation time use `full_grid=True` to get every pore voxel, chunked by the
caller.  Reconstruct a dense volume with `scatter_to_volume`.

    from dataset_reader import PRT3DDataset, split_by_geometry
    tr_idx, te_idx = split_by_geometry("dataset/dataset_reader.h5", frac=0.15)
    train = PRT3DDataset("dataset/dataset_reader.h5", indices=tr_idx, n_points=8192)
    loader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True,
                                         num_workers=4)
"""

import json
import os
import sys
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:                                   # allow inspection without torch
    torch = None
    Dataset = object

import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_coordinates import travel_time, normalize_velocity, squash, stats  # noqa: E402

PORE = 2


# ===========================================================================
#                        THE THREE FEATURE SWITCHES
# ===========================================================================
# All three default to OFF.  With all three off, this file behaves EXACTLY as it
# did before they existed -- same branch channels, same trunk columns, in the
# same order.  There is a regression test for that claim in
# tools/test_three_switches.py; run it after touching anything here.
#
#   A  flow_proxy   Christof: "can we just take the flow field instead of the
#                   GDF?"  Replaces the trunk's geodesic column with the
#                   ADVECTIVE TRAVEL TIME tau, and feeds the branch the
#                   normalised velocity field instead of the binary geometry.
#                   The geometry-encoding problem is then somebody else's
#                   (Geo-ONet) and we only own the reaction part.
#
#   B  transfer_2d  Use 2D training data for the 3D model.  Lives in train.py,
#                   not here: it mixes a SECOND dataset -- Jung's 2D domains
#                   extruded into 3D by tools/build_transfer_set_2d_to_3d.py -- into the loader.
#                   An extruded 2D domain is an exact 3D problem, so this is not
#                   an approximation.  The dataset flag below only records that
#                   a sample came from the 2D source, so it can be weighted.
#
#   C  dim_free     A and B together.  Replaces the Cartesian trunk
#                   (x, y, z, t, gdf) with the flow-space trunk
#                   (t, d_wall, tau), which has the SAME NUMBER OF INPUTS in 2D
#                   and 3D.  That is the whole point: the Cartesian trunk needs
#                   4 columns in 2D and 5 in 3D and therefore cannot transfer,
#                   while the flow-space trunk transfers with no modification.
#
# Column ORDER matters.  Trunk.forward in deeponet_model.py applies FiLM re-injection
# to the LAST column, so the primary geometry/flow feature is always placed
# last.  resolve_switches() below is the single place that decides the layout;
# train.py, evaluate.py and predict.py all call it so they cannot drift apart.
# ===========================================================================

FLOW_MODES = ("tau", "speed", "both")


def resolve_switches(flow_proxy=False, dim_free=False, distance="gdf",
                     with_velocity=False, with_time=True,
                     flow_mode="tau", keep_geometry_channel=False, ndim=3):
    """The one place the switch semantics live.

    Returns a dict describing the concrete configuration:
        trunk_cols   list of column names, in order, feature LAST
        branch_ch    list of branch1 channel names, in order
        trunk_dim    len(trunk_cols)
        in_channels  len(branch_ch)
        film         True if the last trunk column is a geometry/flow feature
        needs_flow   True if tau / normalised velocity must be computed
        label        a short human-readable description for logs
    """
    if dim_free:
        flow_proxy = True                       # C implies A

    # A 2D dataset (nz = 1) has no third coordinate to give the trunk, and its
    # z column would be a constant, i.e. a dead input.  Dropping it reproduces
    # Jung's original 4-input trunk (x, y, t, gdf) exactly.
    space = ["x", "y"] + (["z"] if ndim >= 3 else [])
    # A 2D flow field has no z-component: it is exactly zero everywhere. Feeding
    # it as a third input channel costs convolution weights that receive no
    # gradient and teaches the network nothing. Drop it.
    vel_ch = ["ux", "uy"] + (["uz"] if ndim >= 3 else [])

    if flow_proxy:
        if flow_mode not in FLOW_MODES:
            raise ValueError("flow_mode must be one of %s" % (FLOW_MODES,))
        branch = (["material"] if keep_geometry_channel else []) + vel_ch
        if dim_free:
            cols = (["t"] if with_time else []) + ["dwall", "tau"]
            label = "C dim-free  trunk (%s)" % ", ".join(cols)
        else:
            feat = {"tau": ["tau"], "speed": ["speed"], "both": ["gdf", "tau"]}[flow_mode]
            cols = space + (["t"] if with_time else []) + feat
            label = "A flow-proxy  trunk (%s)" % ", ".join(cols)
        return dict(trunk_cols=cols, branch_ch=branch, trunk_dim=len(cols),
                    in_channels=len(branch), film=True, needs_flow=True,
                    flow_proxy=True, dim_free=bool(dim_free), label=label)

    # ---- all switches off: the original behaviour, unchanged ----------------
    branch = ["material"] + (vel_ch if with_velocity else [])
    cols = space + (["t"] if with_time else [])
    if distance != "none":
        cols = cols + [distance]
    return dict(trunk_cols=cols, branch_ch=branch, trunk_dim=len(cols),
                in_channels=len(branch), film=(distance != "none"),
                needs_flow=False, flow_proxy=False, dim_free=False,
                label="OFF  trunk (%s)" % ", ".join(cols))


def dwall_scale(ny, nz):
    """Transverse half-width used to normalise the wall distance.

    Singleton axes are excluded: a 2D dataset has nz == 1 and that axis carries
    no transverse extent at all.
    """
    dims = [d for d in (ny, nz) if d > 1] or [max(ny, nz)]
    return max(min(dims) / 2.0, 1.0)


SWITCH_KEYS = ("flow_proxy", "dim_free", "flow_mode", "keep_geometry_channel",
               "u_floor", "distance", "with_velocity")


def dataset_kwargs_from_ckpt(ck):
    """Rebuild the exact dataset configuration a checkpoint was trained with.

    evaluate.py and predict.py both call this, so a model can never be scored
    against a differently-configured dataset.  Old checkpoints written before
    the switches existed simply have no switch keys, and every one of them
    defaults to off, so they keep loading unchanged.
    """
    ta = ck.get("args", {})
    kw = dict(distance=ta.get("distance", "gdf"),
              with_velocity=bool(ta.get("with_velocity", False)),
              with_time=ck.get("with_time", True),
              flow_proxy=bool(ta.get("flow_proxy", False)),
              dim_free=bool(ta.get("dim_free", False)),
              flow_mode=ta.get("flow_mode", "tau"),
              keep_geometry_channel=bool(ta.get("keep_geometry_channel", False)),
              u_floor=float(ta.get("u_floor", 0.01)))
    ndim = 2 if (ck.get("grid") and int(ck["grid"][2]) == 1) else 3
    cfg = resolve_switches(ndim=ndim,
                           **{k: v for k, v in kw.items() if k != "u_floor"})
    return kw, cfg


def _open(path):
    return h5py.File(path, "r")


def split_by_geometry(h5path, frac=0.15, seed=0):
    """Hold out whole GEOMETRIES, never individual samples.  A split by sample
    leaks pore structure between train and test and inflates the score."""
    with _open(h5path) as h:
        gidx = h["samples/geom_index"][:]
    uniq = np.unique(gidx)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_te = max(1, int(round(frac * len(uniq))))
    te_g = set(perm[:n_te].tolist())
    te = np.where(np.isin(gidx, list(te_g)))[0]
    tr = np.where(~np.isin(gidx, list(te_g)))[0]
    return tr, te


class PRT3DDataset(Dataset):
    """
    Parameters
    ----------
    h5path      : path to dataset_reader.h5
    indices     : sample indices to use (from split_by_geometry)
    n_points    : trunk collocation points per item (8192 recommended)
    full_grid   : return every pore voxel instead of a random subset (eval)
    with_velocity : add the 3 velocity channels to branch1 (Pipeline 2)
    distance    : 'gdf' | 'edt' | 'none' — the trunk's geometry input.
                  'edt' and 'none' exist for the ablation that shows the
                  GEODESIC field is what matters, not just any distance.
    with_time   : include t in the trunk. None (default) decides from the file:
                  a steady dataset (T=1) has a CONSTANT t, and feeding a constant
                  is a dead input, so it is dropped and the trunk is
                  (x,y,z,gdf) — the 3D analogue of Jung's p_S(x,y,GDF).
                  A transient dataset (T>1) keeps it: (x,y,z,t,gdf), the
                  analogue of p_T(x,y,GDF,t). Pass True/False to force it.
    normalize   : divide each species by its dataset-wide scale
    """

    def __init__(self, h5path, indices=None, n_points=8192, full_grid=False,
                 with_velocity=False, distance="gdf", normalize=True,
                 time_index=None, seed=0, with_time=None,
                 flow_proxy=False, dim_free=False, flow_mode="tau",
                 keep_geometry_channel=False, u_floor=0.01, source_tag=0):
        self.h5path = h5path
        self.n_points = int(n_points)
        self.full_grid = bool(full_grid)
        self.with_velocity = bool(with_velocity)
        self.distance = distance
        self.normalize = bool(normalize)
        self.time_index = time_index
        self.flow_proxy = bool(flow_proxy) or bool(dim_free)
        self.dim_free = bool(dim_free)
        self.flow_mode = flow_mode
        self.keep_geometry_channel = bool(keep_geometry_channel)
        self.u_floor = float(u_floor)
        self.source_tag = int(source_tag)      # 0 = 3D native, 1 = extruded 2D
        self._flow_cache = {}                  # geom index -> (vel_norm, tau)
        self.stagnant_by_geom = {}             # geom index -> stagnant fraction
        self._h = None
        self._rng = np.random.default_rng(seed)

        with _open(h5path) as h:
            self.n_all = int(h.attrs["n_samples"])
            self.shape = tuple(int(v) for v in h.attrs["shape"])
            self.species = [s.decode() for s in h.attrs["species"]]
            self.param_names = [s.decode() for s in h.attrs["param_names"]]
            # Written on the dataset by some writers and on the group by
            # others. dict.get(key, default) evaluates the DEFAULT EAGERLY, so
            # the obvious one-liner raised KeyError on the group even when the
            # dataset carried the attribute perfectly well -- a fallback that
            # crashed whenever it was not needed.
            if "conc_scale" in h["samples/conc"].attrs:
                cs = h["samples/conc"].attrs["conc_scale"]
            elif "conc_scale" in h["samples"].attrs:
                cs = h["samples"].attrs["conc_scale"]
            else:
                raise KeyError(
                    "this file records no conc_scale, on samples/conc or on "
                    "samples. It was written by something older than the "
                    "current collect_complab_output.py; rebuild it.")
            self.conc_scale = np.array(cs, np.float32)
            self.T = int(h["samples/conc"].shape[1])
            self.C = int(h["samples/conc"].shape[2])
            self.has_velocity = "velocity" in h["samples"]
            self._geom_keys = set(h["geom"].keys())
            self.geom_index = h["samples/geom_index"][:]
            self.params = h["samples/params"][:]
            self.t_norm = h["samples/t_norm"][:]
            # pore voxel lists are small and reused constantly -> cache them
            mat = h["geom/material"][:]
        self.pore_idx = [np.argwhere(m == PORE).astype(np.int32) for m in mat]
        self.indices = np.arange(self.n_all) if indices is None else np.asarray(indices)

        if self.with_velocity and not self.has_velocity:
            raise ValueError("with_velocity=True but the file has no /samples/velocity "
                             "(collect_complab_output.py was run with --no-velocity)")
        self.with_time = (self.T > 1) if with_time is None else bool(with_time)

        self.ndim = 2 if int(self.shape[2]) == 1 else 3
        self.cfg = resolve_switches(
            flow_proxy=self.flow_proxy, dim_free=self.dim_free, distance=distance,
            with_velocity=self.with_velocity, with_time=self.with_time,
            flow_mode=self.flow_mode, keep_geometry_channel=self.keep_geometry_channel,
            ndim=self.ndim)
        self.trunk_cols = self.cfg["trunk_cols"]
        self.branch_ch = self.cfg["branch_ch"]
        self.in_channels = self.cfg["in_channels"]
        self.trunk_dim = self.cfg["trunk_dim"]

        if self.cfg["needs_flow"] and not self.has_velocity:
            raise ValueError(
                "--flow-proxy / --dim-free need the velocity field, but this file has "
                "no /samples/velocity. Re-run collect_complab_output.py WITHOUT --no-velocity.")
        if "edt" not in self._geom_keys and self.dim_free:
            raise ValueError("--dim-free needs geom/edt (the wall distance); this file "
                             "has none. Re-run collect_complab_output.py.")

    # -- lazy per-worker file handle (h5py handles are not fork-safe) --------
    @property
    def h(self):
        if self._h is None:
            self._h = _open(self.h5path)
        return self._h

    def __len__(self):
        return len(self.indices) * (1 if self.time_index is not None or self.T == 1 else self.T)

    def _decode(self, k):
        if self.T == 1 or self.time_index is not None:
            return int(self.indices[k]), (0 if self.T == 1 else int(self.time_index))
        return int(self.indices[k // self.T]), int(k % self.T)

    # ---------------------------------------------------------------- flow --
    def _flow(self, s, g):
        """Normalised velocity and travel time for sample s on geometry g.

        Cached PER GEOMETRY, not per sample.  Stokes is linear, so every run on
        the same geometry has the same flow field up to a scale factor; the
        normalised field and the normalised travel time are therefore geometry
        properties and are computed exactly once.  On a 128x64x64 geometry the
        sweep costs about 1 s, so 20 geometries is 20 s for the whole complab_campaign.
        """
        if g in self._flow_cache:
            return self._flow_cache[g]
        mat = self.h["geom/material"][g]
        vel = self.h["samples/velocity"][s].astype(np.float32)
        # collect_complab_output.py writes ZEROS for any run whose flow field could not be
        # read. Caching that under the geometry would hand an all-zero branch
        # input and a constant trunk feature to every other run on the same
        # structure, silently. Look for a sibling run that does have a field.
        if not np.any(vel):
            sibs = np.where(self.geom_index == g)[0]
            for s2 in sibs:
                v2 = self.h["samples/velocity"][int(s2)].astype(np.float32)
                if np.any(v2):
                    vel = v2
                    break
            else:
                raise ValueError(
                    "geometry %d has no run with a usable flow field, so "
                    "--flow-proxy / --dim-free cannot be used on this dataset. "
                    "Re-run collect_complab_output.py and check the warning it prints about "
                    "the flow field." % g)
        vn = normalize_velocity(vel, mat)
        tau, stag = travel_time(vel, mat, u_floor=self.u_floor)
        tau = squash(np.nan_to_num(tau, nan=0.0))          # -> [0, 1)
        m = (mat == PORE)
        # recorded per geometry, not overwritten only on a cache miss
        self.stagnant_by_geom[g] = float(stag[m].mean()) if m.any() else 0.0
        self.stagnant_frac = self.stagnant_by_geom[g]
        self._flow_cache[g] = (vn, tau.astype(np.float32))
        return self._flow_cache[g]

    def __getitem__(self, k):
        s, t = self._decode(k)
        g = int(self.geom_index[s])
        nx, ny, nz = self.shape

        vn = tau = None
        if self.cfg["needs_flow"]:
            vn, tau = self._flow(s, g)

        # ---------------- branch1: whatever resolve_switches asked for -------
        mat = self.h["geom/material"][g]
        chans = []
        for name in self.branch_ch:
            if name == "material":
                chans.append((mat == PORE).astype(np.float32)[None])
            elif name == "ux":                      # the velocity arrives whole
                v = (vn if vn is not None
                     else self.h["samples/velocity"][s].astype(np.float32))
                nvel = sum(1 for c in self.branch_ch if c in ("ux", "uy", "uz"))
                chans.append(v[:nvel])              # 2 components in 2D, 3 in 3D
            elif name in ("uy", "uz"):
                continue                            # consumed by the 'ux' branch
        branch1 = np.concatenate(chans, 0)

        branch2 = self.params[s].astype(np.float32).copy()
        branch2[:3] = np.log10(np.maximum(branch2[:3], 1e-12))   # Pe, Da span decades

        pts = self.pore_idx[g]
        if self.full_grid:
            sel = pts
        else:
            take = min(self.n_points, len(pts))
            sel = pts[self._rng.choice(len(pts), take, replace=take > len(pts))]
        xi, yi, zi = sel[:, 0], sel[:, 1], sel[:, 2]

        # ---------------- trunk: one branch per column name ------------------
        cols = []
        for name in self.trunk_cols:
            if name == "x":
                cols.append(xi / (nx - 1))
            elif name == "y":
                cols.append(yi / (ny - 1))
            elif name == "z":
                cols.append(zi / max(nz - 1, 1))
            elif name == "t":
                cols.append(np.full(len(sel), self.t_norm[s, t], np.float32))
            elif name in ("gdf", "edt"):
                cols.append(self.h["geom/" + name][g][xi, yi, zi] / max(nx - 1, 1))
            elif name == "dwall":
                # Normalised by the transverse half-width so it means the same
                # thing in 2D and 3D. min(ny, nz) collapses to 1 when nz == 1,
                # which left a 2D dataset carrying dwall in RAW VOXELS (0..~32)
                # while a 3D one carried 0..1 -- a thirty-fold scale difference
                # across exactly the transfer switch C exists to enable.
                cols.append(self.h["geom/edt"][g][xi, yi, zi]
                            / dwall_scale(ny, nz))
            elif name == "tau":
                cols.append(tau[xi, yi, zi])
            elif name == "speed":
                cols.append(np.sqrt((vn[:, xi, yi, zi] ** 2).sum(0)))
            else:
                raise KeyError("unknown trunk column %r" % name)
        trunk = np.stack([np.asarray(c, np.float32) for c in cols], 1)

        conc = self.h["samples/conc"][s, t].astype(np.float32)      # (C,nx,ny,nz)
        target = conc[:, xi, yi, zi].T                              # (P, C)
        if self.normalize:
            target = target / self.conc_scale[None, :]

        if torch is None:
            return branch1, branch2, trunk, target
        return (torch.from_numpy(branch1), torch.from_numpy(branch2),
                torch.from_numpy(trunk), torch.from_numpy(target.copy()))


def scatter_to_volume(values, points, shape, fill=np.nan):
    """(P,C) predictions at (P,3) voxel indices -> dense (C,nx,ny,nz) volume."""
    values = np.asarray(values); points = np.asarray(points)
    out = np.full((values.shape[1],) + tuple(shape), fill, np.float32)
    out[:, points[:, 0], points[:, 1], points[:, 2]] = values.T
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/dataset_reader.h5"
    tr, te = split_by_geometry(path)
    ds = PRT3DDataset(path, indices=tr)
    print("species     :", ds.species)
    print("params      :", ds.param_names)
    print("grid        :", ds.shape, " snapshots:", ds.T, " fields:", ds.C)
    print("dimension   : %dD  (grid %s)" % (ds.ndim, ds.shape))
    print("switches    : %s" % ds.cfg["label"])
    print("branch1     : %d ch  (%s)" % (ds.in_channels, ", ".join(ds.branch_ch)))
    print("trunk       : %d inputs  (%s)" % (ds.trunk_dim, ", ".join(ds.trunk_cols)))
    print("train / test: %d / %d samples (split by geometry)" % (len(tr), len(te)))
    b1, b2, tk, y = ds[0]
    print("branch1", tuple(b1.shape), "branch2", tuple(b2.shape),
          "trunk", tuple(tk.shape), "target", tuple(y.shape))
