#!/usr/bin/env python3
"""Build ONE file holding every result of a PRT-DeepONet campaign.

    python3 make_results_h5.py --data dataset.h5 --runs runs --eval eval \\
                               --out prt_results.h5

Everything a reader needs is inside: what was simulated, how the three sets were
chosen, how each model was fitted, how well it does on training, validation and
testing, the volumes themselves for the held out set, the weights, the figures,
and the notes explaining all of it. The file opens with h5py and nothing else.

WHY THE FILE LOOKS LIKE THIS
----------------------------
Three mistakes are easy to make with results of this kind, and each one shaped a
part of the layout.

  1. Averaging a per snapshot score. A species that starts at zero has almost no
     variance in its first frames, so a coefficient of determination computed on
     each snapshot separately and then averaged is dominated by those frames and
     can come out hugely negative while the prediction is in fact good. Both
     numbers are therefore stored, side by side, under names that say which is
     which, and the definition is written into /notes.

  2. Quoting normalised values. The stored concentrations carry two
     normalisations, one fitted across the whole campaign and one fitted on the
     training rows only. Every metric is therefore stored twice, once in the
     units the model works in and once as a fraction of the feed concentration,
     with both factors in the file.

  3. Scoring only the held out set. The progression from training to validation
     to testing is the evidence about overfitting, and it costs almost nothing.
     All three are scored the same way.

WHAT IT NEEDS
-------------
  the dataset .h5 the models were fitted to
  one checkpoint per model, and its summary.json
  optionally the evaluation directory, for the figures
  torch, h5py, numpy, and the PRT-DeepONet code on the path

SHAPES
------
The schema is written for either dimensionality. A field is stored as
(n_row, nz, ny, nx) in 3D and (n_row, ny, nx) in 2D, and the root attribute
`dim` says which. Everything else is identical, so one reader handles both.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import textwrap

import numpy as np

try:
    import h5py
except ImportError:                                   # pragma: no cover
    sys.exit("h5py is required: pip install h5py")

SCHEMA = "prt-results"
SCHEMA_VERSION = "1.0"

# Compression is worth it here: the fields are mostly NaN outside the pore space
# and compress to a fraction of their size, and nothing in this file is read in
# an inner loop.
COMP = dict(compression="gzip", compression_opts=4, shuffle=True)


# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #

def _utf8(x):
    return x.decode("utf-8", "replace") if isinstance(x, (bytes, np.bytes_)) else str(x)


def _sha256(path, cap=None):
    """Checksum of a file. `cap` limits how much is read, for very large ones."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
            n += len(blk)
            if cap and n >= cap:
                break
    return h.hexdigest(), n


def _put(g, name, value, **attrs):
    """Write one dataset, choosing a sensible storage for what it is."""
    if isinstance(value, str):
        d = g.create_dataset(name, data=np.bytes_(value.encode("utf-8")))
    else:
        a = np.asarray(value)
        if a.dtype.kind in "US":
            a = np.array([str(x).encode("utf-8") for x in a.ravel()]
                         ).reshape(a.shape)
            d = g.create_dataset(name, data=a)
        elif a.ndim == 0:
            d = g.create_dataset(name, data=a)
        else:
            d = g.create_dataset(name, data=a,
                                 **(COMP if a.size > 256 else {}))
    for k, v in attrs.items():
        d.attrs[k] = v
    return d


def _table(group, name, columns, note=""):
    """Write a table as one dataset per column, which is what makes it readable.

    A compound dtype would be tidier on paper and much worse in practice: every
    tool that opens an .h5 shows columns, and almost none show compound fields
    well. The row order is the same in every column.
    """
    g = group.create_group(name)
    n = None
    for k, v in columns.items():
        a = np.asarray(v)
        if n is None:
            n = len(a)
        elif len(a) != n:
            raise ValueError("column %r has %d rows, expected %d" % (k, len(a), n))
        _put(g, k, a)
    g.attrs["n_rows"] = int(n or 0)
    g.attrs["columns"] = [c.encode() for c in columns]
    if note:
        g.attrs["note"] = note
    return g


# --------------------------------------------------------------------------- #
#  the one definition of error, used everywhere in this file
# --------------------------------------------------------------------------- #

METRIC_DEF = """\
Let t be the simulated value and p the predicted value, taken over a stated set
of pore voxels. Grain voxels are never included: outside the pore space there is
no value, not a value of zero.

    n     the number of values in the set
    rmse  sqrt(mean((p - t)^2))
    mae   mean(|p - t|)
    bias  mean(p - t)                 positive means the prediction runs high
    r2    1 - sum((p - t)^2) / sum((t - mean(t))^2)
    max   max(|p - t|)

POOLED versus PER SNAPSHOT. A pooled metric takes the set to be every pore voxel
of every stored time of every run in the split, all at once. A per snapshot
metric computes the same quantity separately on each (run, time) and the summary
is then the mean over snapshots.

These are different numbers and for r2 the difference is severe. A species that
begins at zero has almost no variance in its first frames, so the denominator of
r2 is tiny there and a small absolute error produces a large negative r2 which
then dominates the mean over snapshots. Campaign 02, species C: pooled r2 is
0.961 while the mean of the per snapshot r2 is -97.3, on exactly the same
predictions.

QUOTE THE POOLED NUMBER. The per snapshot table is kept because it shows WHERE
the error is, in time and in condition, which the pooled number cannot. It is a
diagnostic, not a headline."""

UNITS_DEF = """\
A value stored in the dataset is not a concentration. It carries two
normalisations and both must be undone before a number is quoted or two species
are compared.

    conc_scale    fitted across the whole campaign when the dataset was built,
                  one factor per species, stored at /design/scales/conc_scale
    target_scale  fitted on the TRAINING rows only when a model was fitted,
                  stored at /models/<TAG>/target_scale

    fraction of the feed concentration = stored value x target_scale x conc_scale

Campaign 02 factors: A x 1.30612, B x 0.99993, C x 0.24718.

Every metric group in this file holds both forms. A dataset named `rmse` is in
the units the model works in; the one named `rmse_feed` is a fraction of the
feed concentration. The second is the one to quote. Read the first by mistake
and species C looks four times worse than it is."""

PITFALLS = """\
Things that have caught us, kept here so they do not catch anyone again.

1. The per snapshot r2 is not the r2. See /notes/metric_definitions.

2. Normalised values are not concentrations. See /notes/units.

3. Some predicted values are negative. How many, and how negative, is measured
   and stored for every split at /results/<TAG>/<split>/sign, so read it there
   rather than quoting a remembered figure. They sit in the earliest stored
   times, where the true field is near zero. Nothing clips them, because
   clipping would hide the behaviour rather than fix it, and a claim of "no
   negative predictions" taken from the last snapshot alone is wrong.

4. The held out pore spaces sit INSIDE the trained range of porosity, not
   outside it. The test score is therefore evidence of interpolation across pore
   structure, not of extrapolation to porosities never seen.

5. A speed comparison must say what it compares. The honest figure is the whole
   transient for all three species against one simulation of the same case. The
   per sample figure that an evaluation script prints is a single species at a
   single time and is about three hundred times larger.

6. Asking for a condition outside the fitted range returns an answer without
   complaint unless the extrapolation flag is read. /prediction/<name> stores
   that flag."""


# --------------------------------------------------------------------------- #
#  metrics
# --------------------------------------------------------------------------- #

def pooled(t, p):
    """The pooled statistics of one set of values. NaN entries are dropped."""
    t = np.asarray(t, np.float64).ravel()
    p = np.asarray(p, np.float64).ravel()
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    if t.size == 0:
        return dict(n=0, rmse=np.nan, mae=np.nan, bias=np.nan, r2=np.nan,
                    max_abs_err=np.nan)
    d = p - t
    den = np.sum((t - t.mean()) ** 2)
    return dict(n=int(t.size),
                rmse=float(np.sqrt(np.mean(d ** 2))),
                mae=float(np.mean(np.abs(d))),
                bias=float(np.mean(d)),
                r2=float(1.0 - np.sum(d ** 2) / den) if den > 0 else np.nan,
                max_abs_err=float(np.max(np.abs(d))))


def _write_stats(group, name, stats, factor, note=""):
    """One metric group, in both units.

    Every scalar appears twice. The plain name is in the units the model works
    in; the `_feed` name is a fraction of the feed concentration. r2 and n are
    scale free and appear once.
    """
    g = group.create_group(name)
    for k in ("rmse", "mae", "bias", "max_abs_err"):
        g.create_dataset(k, data=np.float64(stats[k]))
        g.create_dataset(k + "_feed", data=np.float64(stats[k] * factor))
    g.create_dataset("r2", data=np.float64(stats["r2"]))
    g.create_dataset("n", data=np.int64(stats["n"]))
    g.attrs["to_feed_factor"] = float(factor)
    g.attrs["definition"] = "see /notes/metric_definitions"
    if note:
        g.attrs["note"] = note
    return g


# --------------------------------------------------------------------------- #
#  reading the campaign side
# --------------------------------------------------------------------------- #

def read_design(src):
    """Everything about the experiment itself, from the dataset file."""
    d = {}
    d["attrs"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in src.attrs.items()}
    d["species"] = [_utf8(x) for x in src.attrs["species"]]
    d["param_names"] = [_utf8(x) for x in src.attrs["param_names"]]
    d["shape"] = [int(v) for v in src.attrs["shape"]]
    d["spacing"] = [float(v) for v in src.attrs["spacing"]]

    d["gid"] = src["geom/gid"][:].astype(np.int64)
    d["material"] = src["geom/material"]
    d["gdf"] = src["geom/gdf"]
    d["edt"] = src["geom/edt"]

    d["geom_index"] = src["samples/geom_index"][:].astype(np.int64)
    d["params"] = src["samples/params"][:].astype(np.float64)
    d["run_id"] = src["samples/run_id"][:].astype(np.int64)
    d["t_norm"] = src["samples/t_norm"][:].astype(np.float64)
    d["t_seconds"] = src["samples/t_seconds"][:].astype(np.float64)
    d["wall_s"] = src["samples/wall_s"][:].astype(np.float64)

    # porosity and tortuosity come from each run's own recorded parameters, not
    # recomputed here, so they are the numbers the campaign actually used.
    por, tor = [], []
    for s in range(len(d["geom_index"])):
        try:
            pj = json.loads(_utf8(src["inputs/params_json"][s]))
        except Exception:
            pj = {}
        por.append(float(pj.get("porosity", np.nan)))
        tor.append(float(pj.get("tortuosity", pj.get("tau_geom", np.nan))))
    d["run_porosity"] = np.asarray(por)
    d["run_tortuosity"] = np.asarray(tor)

    ng = len(d["gid"])
    gp = np.full(ng, np.nan)
    gt = np.full(ng, np.nan)
    for s, g in enumerate(d["geom_index"]):
        if np.isnan(gp[g]):
            gp[g] = d["run_porosity"][s]
            gt[g] = d["run_tortuosity"][s]
    d["geom_porosity"] = gp
    d["geom_tortuosity"] = gt

    # The collector writes conc_scale as an attribute of the /samples GROUP,
    # not of the file. Missing it is not a cosmetic problem: every metric then
    # comes out in the wrong units and species C looks four times worse than it
    # is, so the places it could be are all checked and a failure is loud.
    cs = None
    for where, key in (("samples", "conc_scale"), ("samples", "concentration_scale"),
                       ("", "conc_scale"), ("", "concentration_scale")):
        node = src[where] if where and where in src else src
        if key in node.attrs:
            cs = np.asarray(node.attrs[key], np.float64)
            break
    if cs is None and "ancillary/conc_scale" in src:
        cs = src["ancillary/conc_scale"][:].astype(np.float64)
    d["conc_scale"] = cs

    rs = None
    if "samples" in src and "rate_scale" in src["samples"].attrs:
        rs = np.asarray(src["samples"].attrs["rate_scale"], np.float64)
    d["rate_scale"] = rs
    return d


def read_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
#  prediction
# --------------------------------------------------------------------------- #

def predict_split(data_path, ck, gids, chunk=65536):
    """Predict every pore voxel of every stored time of the given pore spaces.

    Returns the predicted and simulated volumes with NaN outside the pore space,
    and one row of metadata per (run, time).
    """
    import torch
    from dataset_reader import (PRT3DDataset, indices_for_geometries,
                                dataset_kwargs_from_ckpt)
    from deeponet_model import PRT_DeepONet3D

    idx = indices_for_geometries(data_path, list(gids))
    if len(idx) == 0:
        return None

    kw, cfg = dataset_kwargs_from_ckpt(ck)
    if ck.get("target_scale") is not None:
        kw["target_scale"] = np.asarray(ck["target_scale"], np.float32)
    ds = PRT3DDataset(data_path, indices=idx, full_grid=True, **kw)

    model = PRT_DeepONet3D(
        in_channels=ck.get("in_channels", cfg["in_channels"]),
        n_params=len(ck["param_names"]),
        trunk_in_dim=int(ck["trunk_in_dim"]),
        grid=ds.shape).eval()
    model.load_state_dict(ck["model"])

    shape = tuple(int(v) for v in ds.shape)
    n = len(ds)
    P = np.full((n,) + shape, np.nan, np.float32)
    T = np.full((n,) + shape, np.nan, np.float32)
    rows = []
    # Time the forward passes alone. Reading the dataset, scattering back to a
    # volume and allocating the arrays are not what a speed claim is about, so
    # the clock is started and stopped around the model call and nothing else.
    import time
    t_infer = 0.0
    n_points = 0
    for k in range(n):
        b1, b2, tk, y = ds[k]
        t0 = time.perf_counter()
        with torch.no_grad():
            parts = [model(b1[None], b2[None], tk[None, a:a + chunk])[0].numpy()
                     for a in range(0, tk.shape[0], chunk)]
        t_infer += time.perf_counter() - t0
        n_points += int(tk.shape[0])
        pred = np.concatenate(parts, 0)
        s, ti = ds._decode(k)
        pts = ds.pore_idx[int(ds.geom_index[s])].astype(int)
        if len(shape) == 3:
            P[k][pts[:, 0], pts[:, 1], pts[:, 2]] = pred
            T[k][pts[:, 0], pts[:, 1], pts[:, 2]] = y.numpy()
        else:
            P[k][pts[:, 0], pts[:, 1]] = pred
            T[k][pts[:, 0], pts[:, 1]] = y.numpy()
        # _decode already returns the GLOBAL sample index, so indexing through
        # ds.indices a second time is wrong. It happens to give the right answer
        # whenever the selected rows are 0..n-1, which is why it can survive a
        # test on a dataset trimmed to one split and fail on a real one.
        row = dict(row=k,
                   sample=int(s),
                   run_id=int(ds.h["samples/run_id"][s]),
                   geom_row=int(ds.geom_index[s]),
                   gid=int(ds.h["geom/gid"][int(ds.geom_index[s])]),
                   t_index=int(ti),
                   t_norm=float(ds.t_norm[s, ti]))
        for nm, v in zip(ds.param_names, ds.params[s]):
            row[str(nm)] = float(v)
        rows.append(row)
    return dict(pred=P, truth=T, rows=rows, shape=shape,
                species=ds.target_species,
                inference_seconds=t_infer, points_evaluated=n_points,
                device=str(next(model.parameters()).device))


# --------------------------------------------------------------------------- #
#  the per split result group
# --------------------------------------------------------------------------- #

def write_split_results(parent, split_name, res, factor, design,
                        store_fields, note=""):
    """One /results/<TAG>/<split> group: the metrics at four resolutions."""
    g = parent.create_group(split_name)
    T, P = res["truth"], res["pred"]
    rows = res["rows"]
    g.attrs["n_runs"] = len(set(r["sample"] for r in rows))
    g.attrs["n_snapshots"] = len(rows)
    g.attrs["pore_spaces"] = sorted(set(int(r["gid"]) for r in rows))
    g.attrs["definition"] = "see /notes/metric_definitions"
    if note:
        g.attrs["note"] = note

    # ---- how long the forward passes took, measured here ------------------ #
    tm = g.create_group("timing")
    tm.create_dataset("inference_seconds_total",
                      data=np.float64(res.get("inference_seconds", np.nan)))
    tm.create_dataset("points_evaluated",
                      data=np.int64(res.get("points_evaluated", 0)))
    tm.create_dataset("seconds_per_snapshot",
                      data=np.float64(res.get("inference_seconds", np.nan)
                                      / max(len(rows), 1)))
    pts = max(int(res.get("points_evaluated", 0)), 1)
    tm.create_dataset("microseconds_per_pore_voxel",
                      data=np.float64(1e6 * res.get("inference_seconds", np.nan)
                                      / pts))
    tm.attrs["device"] = res.get("device", "unknown")
    tm.attrs["measured"] = ("the model forward passes only. Reading the "
                            "dataset, scattering values back into a volume and "
                            "allocating arrays are excluded.")
    tm.attrs["one_species_only"] = ("this is ONE species. A comparison against "
                                    "a simulation must use all species of the "
                                    "full transient: see /comparisons/speed.")

    # ---- pooled, the headline -------------------------------------------- #
    st = pooled(T, P)
    _write_stats(g, "pooled", st, factor,
                 "every pore voxel of every stored time of every run in "
                 "this split, taken at once. THIS IS THE NUMBER TO QUOTE.")

    # ---- per snapshot, the diagnostic ------------------------------------ #
    cols = dict(row=[], sample=[], run_id=[], gid=[], t_index=[], t_norm=[],
                rmse=[], rmse_feed=[], mae=[], bias=[], r2=[], n=[])
    for name in design["param_names"]:
        cols[name] = []
    for k, r in enumerate(rows):
        s = pooled(T[k], P[k])
        cols["row"].append(k)
        cols["sample"].append(r["sample"])
        cols["run_id"].append(r["run_id"])
        cols["gid"].append(r["gid"])
        cols["t_index"].append(r["t_index"])
        cols["t_norm"].append(r["t_norm"])
        cols["rmse"].append(s["rmse"])
        cols["rmse_feed"].append(s["rmse"] * factor)
        cols["mae"].append(s["mae"])
        cols["bias"].append(s["bias"])
        cols["r2"].append(s["r2"])
        cols["n"].append(s["n"])
        for name in design["param_names"]:
            cols[name].append(r.get(name, np.nan))
    _table(g, "per_snapshot", cols,
           "one row per (run, stored time). The mean of the r2 column is NOT "
           "the r2 of this split: see /notes/metric_definitions.")

    # the summary of the per snapshot numbers, kept so the difference from the
    # pooled number is visible in the file rather than having to be recomputed
    ps = g.create_group("per_snapshot_summary")
    r2a = np.asarray(cols["r2"], np.float64)
    rma = np.asarray(cols["rmse"], np.float64)
    ps.create_dataset("rmse_mean", data=np.float64(np.nanmean(rma)))
    ps.create_dataset("rmse_mean_feed", data=np.float64(np.nanmean(rma) * factor))
    ps.create_dataset("r2_mean", data=np.float64(np.nanmean(r2a)))
    ps.create_dataset("r2_median", data=np.float64(np.nanmedian(r2a)))
    ps.create_dataset("r2_undefined", data=np.int64(int(np.sum(~np.isfinite(r2a)))))
    ps.attrs["warning"] = ("r2_mean is dominated by early snapshots where the "
                           "true field has almost no variance. The pooled r2 "
                           "is the one to quote.")

    # ---- per run ---------------------------------------------------------- #
    by = {}
    for k, r in enumerate(rows):
        by.setdefault(r["sample"], []).append(k)
    cols = dict(sample=[], run_id=[], gid=[], rmse=[], rmse_feed=[], mae=[],
                bias=[], r2=[], n=[])
    for name in design["param_names"]:
        cols[name] = []
    for s in sorted(by):
        ks = by[s]
        st = pooled(T[ks], P[ks])
        r0 = rows[ks[0]]
        cols["sample"].append(s)
        cols["run_id"].append(r0["run_id"])
        cols["gid"].append(r0["gid"])
        for k2, v in (("rmse", st["rmse"]), ("mae", st["mae"]),
                      ("bias", st["bias"]), ("r2", st["r2"]), ("n", st["n"])):
            cols[k2].append(v)
        cols["rmse_feed"].append(st["rmse"] * factor)
        for name in design["param_names"]:
            cols[name].append(r0.get(name, np.nan))
    _table(g, "per_run", cols, "one row per simulation, pooled over its 21 times")

    # ---- per pore space --------------------------------------------------- #
    byg = {}
    for k, r in enumerate(rows):
        byg.setdefault(r["gid"], []).append(k)
    gcols = dict(gid=[], porosity=[], tortuosity=[], rmse=[], rmse_feed=[],
                 bias=[], r2=[], n_runs=[], n=[])
    for gid in sorted(byg):
        ks = byg[gid]
        st = pooled(T[ks], P[ks])
        grow = rows[ks[0]]["geom_row"]
        gcols["gid"].append(gid)
        gcols["porosity"].append(design["geom_porosity"][grow])
        gcols["tortuosity"].append(design["geom_tortuosity"][grow])
        gcols["rmse"].append(st["rmse"])
        gcols["rmse_feed"].append(st["rmse"] * factor)
        gcols["bias"].append(st["bias"])
        gcols["r2"].append(st["r2"])
        gcols["n_runs"].append(len(set(rows[k]["sample"] for k in ks)))
        gcols["n"].append(st["n"])
    _table(g, "per_geometry", gcols,
           "pooled over every run of each pore space")

    # ---- by condition ----------------------------------------------------- #
    for name in design["param_names"]:
        byc = {}
        for k, r in enumerate(rows):
            byc.setdefault(round(float(r.get(name, np.nan)), 6), []).append(k)
        ccols = {name: [], "rmse": [], "rmse_feed": [], "bias": [], "r2": [],
                 "n": []}
        for v in sorted(byc):
            st = pooled(T[byc[v]], P[byc[v]])
            ccols[name].append(v)
            ccols["rmse"].append(st["rmse"])
            ccols["rmse_feed"].append(st["rmse"] * factor)
            ccols["bias"].append(st["bias"])
            ccols["r2"].append(st["r2"])
            ccols["n"].append(st["n"])
        _table(g, "by_%s" % name, ccols, "pooled at each level of %s" % name)

    # ---- by time ---------------------------------------------------------- #
    byt = {}
    for k, r in enumerate(rows):
        byt.setdefault(round(float(r["t_norm"]), 6), []).append(k)
    tcols = dict(t_norm=[], rmse=[], rmse_feed=[], bias=[], r2=[], n=[])
    for v in sorted(byt):
        st = pooled(T[byt[v]], P[byt[v]])
        tcols["t_norm"].append(v)
        tcols["rmse"].append(st["rmse"])
        tcols["rmse_feed"].append(st["rmse"] * factor)
        tcols["bias"].append(st["bias"])
        tcols["r2"].append(st["r2"])
        tcols["n"].append(st["n"])
    _table(g, "by_time", tcols,
           "pooled across all runs at each stored time. This is where the early "
           "time behaviour shows up honestly, as a small rmse rather than as a "
           "wild r2.")

    # ---- the parity histogram, so the headline figure needs no fields ----- #
    t = T[np.isfinite(T)].ravel() * factor
    p = P[np.isfinite(P)].ravel() * factor
    m = min(t.size, p.size)
    hi = float(max(np.max(t[:m]) if m else 1.0, np.max(p[:m]) if m else 1.0)) * 1.02
    H, xe, ye = np.histogram2d(t[:m], p[:m], bins=160,
                               range=[[min(0.0, float(np.min(p[:m]))), hi],
                                      [min(0.0, float(np.min(p[:m]))), hi]])
    pg = g.create_group("parity")
    _put(pg, "counts", H.astype(np.int64))
    _put(pg, "simulated_edges", xe)
    _put(pg, "predicted_edges", ye)
    pg.attrs["units"] = "fraction of the feed concentration"
    pg.attrs["note"] = ("the two dimensional histogram the parity figure draws. "
                        "Plot counts on a log colour scale with the 1:1 line.")

    # ---- the sign of the error, and how much of it is negative ------------ #
    neg = float(np.mean(p[:m] < 0.0)) if m else np.nan
    sg = g.create_group("sign")
    sg.create_dataset("fraction_predicted_negative", data=np.float64(neg))
    sg.create_dataset("min_predicted_feed",
                      data=np.float64(float(np.min(p[:m])) if m else np.nan))
    sg.create_dataset("max_predicted_feed",
                      data=np.float64(float(np.max(p[:m])) if m else np.nan))
    sg.attrs["note"] = ("nothing is clipped. A negative prediction is reported "
                        "rather than hidden: see /notes/pitfalls.")

    # ---- the volumes themselves ------------------------------------------- #
    if store_fields:
        fg = g.create_group("fields")
        _put(fg, "truth", T.astype(np.float16))
        _put(fg, "pred", P.astype(np.float16))
        _put(fg, "t_norm", np.asarray([r["t_norm"] for r in rows]))
        _put(fg, "run_id", np.asarray([r["run_id"] for r in rows], np.int64))
        _put(fg, "gid", np.asarray([r["gid"] for r in rows], np.int64))
        fg.attrs["units"] = ("model units. Multiply by the to_feed_factor of "
                             "this model for fractions of the feed.")
        fg.attrs["to_feed_factor"] = float(factor)
        fg.attrs["fill"] = "NaN outside the pore space"
        fg.attrs["axis_order"] = ("row, z, y, x" if T.ndim == 4 else "row, y, x")
    return g


# --------------------------------------------------------------------------- #
#  the builder
# --------------------------------------------------------------------------- #

def build(args):
    sys.path[:0] = [p for p in (args.code_model, args.code_tools, args.code_root)
                    if p and os.path.isdir(p)]

    src = h5py.File(args.data, "r")
    design = read_design(src)
    dim = len(design["shape"])
    if dim not in (2, 3):
        sys.exit("the dataset declares a %dd grid, which this schema does not "
                 "cover" % dim)

    models = dict(args.model or [])
    if args.runs:
        for tag in sorted(os.listdir(args.runs)):
            ck = os.path.join(args.runs, tag, "best.pt")
            if os.path.isfile(ck):
                models.setdefault(tag, ck)
    if not models:
        sys.exit("no models given: use --runs <dir> or --model TAG=path/best.pt")

    print("dataset   %s" % args.data)
    print("grid      %s   (%dd)" % ("x".join(str(v) for v in design["shape"]), dim))
    print("models    %s" % ", ".join(sorted(models)))
    print("fields    %s" % args.fields)

    out = h5py.File(args.out, "w")

    # ---------------------------------------------------------------- header
    now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
    sha, nread = _sha256(args.data, cap=args.hash_cap)
    out.attrs["schema"] = SCHEMA
    out.attrs["schema_version"] = SCHEMA_VERSION
    out.attrs["title"] = args.title
    out.attrs["campaign"] = args.campaign
    out.attrs["created_utc"] = now.isoformat()
    out.attrs["created_by"] = "make_results_h5.py"
    out.attrs["dim"] = dim
    out.attrs["grid"] = np.asarray(design["shape"], np.int64)
    out.attrs["spacing"] = np.asarray(design["spacing"], np.float64)
    out.attrs["spacing_unit"] = _utf8(design["attrs"].get("spacing_unit", "um"))
    out.attrs["species"] = [s.encode() for s in design["species"]]
    out.attrs["param_names"] = [s.encode() for s in design["param_names"]]
    out.attrs["reaction"] = _utf8(design["attrs"].get("reaction", ""))
    out.attrs["dataset_file"] = os.path.abspath(args.data)
    out.attrs["dataset_sha256"] = sha
    out.attrs["dataset_sha256_bytes"] = int(nread)
    out.attrs["n_models"] = len(models)
    out.attrs["n_runs_in_dataset"] = int(len(design["run_id"]))
    out.attrs["n_pore_spaces"] = int(len(design["gid"]))

    # ----------------------------------------------------------------- notes
    n = out.create_group("notes")
    _put(n, "overview", textwrap.dedent(args.overview or DEFAULT_OVERVIEW).strip())
    _put(n, "metric_definitions", METRIC_DEF)
    _put(n, "units", UNITS_DEF)
    _put(n, "pitfalls", PITFALLS)
    _put(n, "glossary", GLOSSARY)
    _put(n, "provenance", args.provenance or
         "built by make_results_h5.py; see /models/<TAG>/args for the exact "
         "command each model was fitted with")
    n.attrs["read_me_first"] = "overview"

    # ---------------------------------------------------------------- design
    d = out.create_group("design")
    gg = d.create_group("geometry")
    _put(gg, "gid", design["gid"])
    _put(gg, "porosity", design["geom_porosity"])
    _put(gg, "tortuosity", design["geom_tortuosity"])
    if args.store_geometry:
        _put(gg, "material", design["material"][:],
             note="0 grain interior, 1 grain surface, 2 pore")
        _put(gg, "geodesic_distance", design["gdf"][:])
        _put(gg, "euclidean_distance", design["edt"][:])
    gg.attrs["note"] = ("one row per pore space. porosity and tortuosity are "
                        "read from each run's own recorded parameters, not "
                        "recomputed here.")

    rcols = dict(sample=np.arange(len(design["run_id"])),
                 run_id=design["run_id"],
                 gid=design["gid"][design["geom_index"]],
                 geom_row=design["geom_index"],
                 porosity=design["run_porosity"],
                 wall_seconds=design["wall_s"],
                 t_end_seconds=design["t_seconds"][:, -1])
    for j, name in enumerate(design["param_names"]):
        rcols[name] = design["params"][:, j]
    _table(d, "runs", rcols, "one row per simulation in the dataset")

    sc = d.create_group("scales")
    if design["conc_scale"] is not None:
        _put(sc, "conc_scale", design["conc_scale"],
             note="campaign wide, one factor per species, in the order of "
                  "the root `species` attribute")
    else:
        print("WARNING: the dataset carries no conc_scale, so every metric in "
              "this file is in model units and the _feed columns are NOT "
              "fractions of the feed concentration.")
        sc.attrs["WARNING"] = ("conc_scale was not found in the dataset. The "
                               "_feed columns in this file are therefore not "
                               "fractions of the feed concentration.")
    if design["rate_scale"] is not None:
        _put(sc, "rate_scale", design["rate_scale"])
    sc.attrs["how_to_use"] = "see /notes/units"

    # ------------------------------------------------- models and the results
    mg = out.create_group("models")
    rg = out.create_group("results")
    split_record = {}

    for tag in sorted(models):
        ckpath = models[tag]
        print("\n%s" % ("-" * 60))
        print("model %s   %s" % (tag, ckpath))
        ck = read_checkpoint(ckpath)
        m = mg.create_group(tag)

        species = ck.get("target_species") or ck.get("species") or tag[0]
        ts = np.asarray(ck.get("target_scale", [1.0]), np.float64)
        si = design["species"].index(species) if species in design["species"] else 0
        cs = (float(design["conc_scale"][si])
              if design["conc_scale"] is not None else 1.0)
        factor = float(ts[si] if ts.size > si else ts[0]) * cs

        m.attrs["species"] = species
        m.attrs["distance"] = _utf8(ck.get("args", {}).get("distance", ""))
        m.attrs["trunk_in_dim"] = int(ck.get("trunk_in_dim", 0))
        m.attrs["in_channels"] = int(ck.get("in_channels", 0))
        m.attrs["param_names"] = [_utf8(x).encode() for x in ck.get("param_names", [])]
        m.attrs["to_feed_factor"] = factor
        m.attrs["checkpoint_path"] = os.path.abspath(ckpath)
        _put(m, "target_scale", ts,
             note="fitted on the training rows only: see /notes/units")
        _put(m, "args", json.dumps(ck.get("args", {}), indent=1, default=str),
             note="the exact arguments this model was fitted with")

        # ---- the weights, so the file alone can predict
        if args.store_weights:
            wg = m.create_group("weights")
            total = 0
            for k, v in ck["model"].items():
                a = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
                _put(wg, k.replace("/", "."), a)
                total += int(a.size)
            wg.attrs["n_parameters"] = total
            wg.attrs["note"] = (
                "the fitted state dict, one dataset per tensor, named exactly as "
                "the state dict names it. To rebuild the model: construct "
                "PRT_DeepONet3D with in_channels, n_params and trunk_in_dim from "
                "this group's parent attributes, then load_state_dict on a dict "
                "built from these datasets.")
            m.attrs["n_parameters"] = total

        # ---- the fitting history
        tr = m.create_group("training")
        hist = ck.get("history") or []
        sm = {}
        smp = os.path.join(os.path.dirname(ckpath), "summary.json")
        if os.path.isfile(smp):
            sm = json.load(open(smp))
            hist = sm.get("history", hist)
        if hist:
            _table(tr, "history",
                   dict(epoch=[int(h.get("epoch", i)) for i, h in enumerate(hist)],
                        train_loss=[float(h.get("train", np.nan)) for h in hist],
                        val_loss=[float(h.get("val", np.nan)) for h in hist],
                        seconds=[float(h.get("sec", np.nan)) for h in hist],
                        lr=[float(h.get("lr", np.nan)) for h in hist]),
                   "one row per epoch. val_loss is what chose the checkpoint; "
                   "no weight was ever updated from it.")
            v = np.asarray([h.get("val", np.nan) for h in hist], np.float64)
            best = int(np.nanargmin(v))
            cg = tr.create_group("checkpoint")
            cg.create_dataset("best_epoch", data=np.int64(best))
            cg.create_dataset("best_val_loss", data=np.float64(v[best]))
            cg.create_dataset("epochs_run", data=np.int64(len(hist)))
            cg.create_dataset("compute_seconds", data=np.float64(
                float(np.nansum([h.get("sec", np.nan) for h in hist]))))
            cg.attrs["stopped_because"] = (
                "no improvement for %d epochs" % (len(hist) - 1 - best)
                if len(hist) - 1 - best > 0 else "ran to the epoch limit")
        if sm:
            _put(tr, "summary_json", json.dumps(sm, indent=1, default=str))

        # ---- the splits as this model recorded them
        sp = ck.get("split_geometries") or {}
        sg = m.create_group("split")
        for k in ("train", "val", "test"):
            _put(sg, k, np.asarray(sp.get(k, []), np.int64))
        sg.attrs["kind"] = _utf8(sm.get("split_kind", ck.get("split_kind", "")))
        sg.attrs["unit"] = "pore space, not run"
        sg.attrs["note"] = ("every run of a pore space goes to the same set. "
                            "Splitting runs at random instead would put the "
                            "same pore space on both sides.")
        split_record[tag] = sp

        # ---- score each split the same way
        tg = rg.create_group(tag)
        tg.attrs["species"] = species
        tg.attrs["to_feed_factor"] = factor
        for split in ("train", "val", "test"):
            if split not in args.splits:
                continue
            gids = sp.get(split, [])
            if not len(gids):
                continue
            res = predict_split(args.data, ck, gids, chunk=args.chunk)
            if res is None:
                print("  %-5s  not present in this dataset" % split)
                continue
            keep = (args.fields == "all"
                    or (args.fields == "test" and split == "test"))
            write_split_results(tg, split, res, factor, design, keep)
            st = pooled(res["truth"], res["pred"])
            print("  %-5s  %2d pore spaces  %7d snapshots  "
                  "rmse %.5f feed  r2 %.4f%s"
                  % (split, len(gids), len(res["rows"]),
                     st["rmse"] * factor, st["r2"],
                     "  (fields stored)" if keep else ""))
            del res

    # ------------------------------------------------------------ comparisons
    cmpg = out.create_group("comparisons")
    write_ablation(cmpg, rg, models, out)
    if args.sim_seconds:
        write_speed(cmpg, rg, design, args)

    # --------------------------------------------------------------- figures
    if args.figures:
        fg = out.create_group("figures")
        nfig = 0
        for root, _, files in os.walk(args.figures):
            for fn in sorted(files):
                if not fn.lower().endswith(".png"):
                    continue
                rel = os.path.relpath(os.path.join(root, fn), args.figures)
                with open(os.path.join(root, fn), "rb") as fh:
                    blob = np.frombuffer(fh.read(), np.uint8)
                dset = fg.create_dataset(rel.replace(os.sep, "__"), data=blob)
                dset.attrs["content_type"] = "image/png"
                dset.attrs["source"] = rel
                nfig += 1
        fg.attrs["n_figures"] = nfig
        fg.attrs["how_to_read"] = ("each dataset is the raw bytes of a .png. "
                                   "open(name,'wb').write(f[...][:].tobytes())")
        print("\nfigures embedded: %d" % nfig)

    out.close()
    src.close()
    size = os.path.getsize(args.out)
    print("\nwrote %s   %.1f MB" % (args.out, size / 1e6))
    return 0


def write_ablation(cmpg, rg, models, out):
    """The three C models against each other, if all three are present."""
    want = [t for t in ("C", "C_edt", "C_none") if t in models]
    if len(want) < 2:
        return
    g = cmpg.create_group("geometry_ablation")
    g.attrs["question"] = ("does the GEODESIC distance buy the accuracy, or "
                           "would any distance field do")
    g.attrs["models"] = [t.encode() for t in want]
    g.attrs["meaning"] = ("C uses the geodesic distance through the pore "
                          "space, C_edt the straight line distance, C_none no "
                          "distance at all. Nothing else differs.")
    rows = dict(model=[], rmse_feed=[], r2=[], n=[])
    per = {}
    for t in want:
        p = "%s/test/pooled" % t
        if p not in rg:
            continue
        rows["model"].append(t)
        rows["rmse_feed"].append(float(rg[p + "/rmse_feed"][()]))
        rows["r2"].append(float(rg[p + "/r2"][()]))
        rows["n"].append(int(rg[p + "/n"][()]))
        gp = "%s/test/per_geometry" % t
        if gp in rg:
            per[t] = dict(zip(rg[gp + "/gid"][:].tolist(),
                              rg[gp + "/rmse_feed"][:].tolist()))
    if rows["model"]:
        _table(g, "pooled", rows, "all three scored on the same held out set")
    if "C" in per and "C_none" in per:
        gids = sorted(set(per["C"]) & set(per["C_none"]))
        wins = sum(1 for k in gids if per["C"][k] < per["C_none"][k])
        cols = dict(gid=gids)
        for t in want:
            if t in per:
                cols[t] = [per[t][k] for k in gids]
        cols["geodesic_wins"] = [int(per["C"][k] < per["C_none"][k]) for k in gids]
        _table(g, "per_geometry", cols,
               "rmse as a fraction of the feed, per held out pore space")
        from math import comb
        p = sum(comb(len(gids), i) for i in range(wins, len(gids) + 1)) / 2 ** len(gids)
        s = g.create_group("sign_test")
        s.create_dataset("pore_spaces_won", data=np.int64(wins))
        s.create_dataset("pore_spaces_total", data=np.int64(len(gids)))
        s.create_dataset("p_one_sided", data=np.float64(p))
        s.attrs["null"] = ("the geodesic distance is no better than no "
                           "distance, so each pore space is a coin flip")


def write_speed(cmpg, rg, design, args):
    """The one speed comparison that is defensible, built from what was measured.

    A simulation produces every species at every stored time, in one run. The
    only like for like comparison is therefore the cost of predicting every
    species over the full transient of ONE case, against the wall time of one
    simulation. Dividing a single species at a single time by a whole simulation
    is what produces the six figure speed-ups that evaluation scripts print, and
    they are not a comparison of the same thing.
    """
    g = cmpg.create_group("speed")
    g.create_dataset("simulation_seconds_median", data=np.float64(args.sim_seconds))

    # cost per snapshot per species, averaged over the production models present
    per = []
    dev = "unknown"
    for tag in ("A", "B", "C"):
        p = "%s/test/timing" % tag
        if p in rg:
            per.append(float(rg[p + "/seconds_per_snapshot"][()]))
            dev = rg[p].attrs.get("device", dev)
    if per:
        n_t = int(design["t_norm"].shape[1])
        n_sp = len(per)
        one_case = float(np.sum(per)) * n_t      # every species, every time, one case
        g.create_dataset("seconds_per_snapshot_per_species",
                         data=np.float64(float(np.mean(per))))
        g.create_dataset("seconds_one_case_all_species_all_times",
                         data=np.float64(one_case))
        g.create_dataset("n_species_compared", data=np.int64(n_sp))
        g.create_dataset("n_stored_times", data=np.int64(n_t))
        g.create_dataset("speedup_honest",
                         data=np.float64(args.sim_seconds / one_case))
        g.attrs["device"] = dev
        g.attrs["speedup_honest_means"] = (
            "the median wall time of one simulation divided by the time to "
            "predict %d species at all %d stored times of one case, on %s"
            % (n_sp, n_t, dev))
        n_all = len(design["species"])
        if n_sp < n_all:
            g.attrs["INCOMPLETE"] = (
                "only %d of the %d species in this campaign had a model in this "
                "file, so the figure above is NOT the cost of a full case and "
                "overstates the speed-up by roughly %.1f times."
                % (n_sp, n_all, n_all / max(n_sp, 1)))
    g.attrs["honest_comparison"] = (
        "one simulation produces ALL species at ALL stored times. A speed "
        "comparison must therefore put the full transient of every species "
        "against it, not one species at one time.")
    g.attrs["caution"] = (
        "these timings belong to the machine this file was built on. A card "
        "and a processor differ by more than an order of magnitude, so quote "
        "the device with the number.")


DEFAULT_OVERVIEW = """\
WHAT THIS FILE IS

One file holding every result of a PRT-DeepONet campaign: what was simulated,
how the pore spaces were divided, how each model was fitted, how well each one
does on the training, validation and test sets, the predicted and simulated
volumes for the held out set, the fitted weights, the figures, and these notes.

HOW TO READ IT

  /notes            start here. metric_definitions and units in particular.
  /design           what the experiment was: pore spaces, runs, conditions
  /models/<TAG>     one group per fitted model: weights, arguments, history
  /results/<TAG>/<train|val|test>
                    the metrics, at four resolutions:
                      pooled          one number for the whole split, QUOTE THIS
                      per_run         one row per simulation
                      per_snapshot    one row per (run, stored time), diagnostic
                      per_geometry    one row per pore space
                    plus by_pe, by_da, by_time, parity and sign
  /comparisons      the geometry ablation and the speed comparison
  /figures          the figures, as embedded .png bytes

THE ONE THING TO GET RIGHT

Read `pooled/rmse_feed` and `pooled/r2`. Do not average the r2 column of
per_snapshot, and do not quote a value that has not been converted to a fraction
of the feed concentration. Both traps are explained in /notes.

A MINIMAL READ

    import h5py
    f = h5py.File("prt_results.h5")
    print(f["notes/overview"][()].decode())
    for tag in f["results"]:
        g = f["results"][tag]["test"]["pooled"]
        print(tag, float(g["rmse_feed"][()]), float(g["r2"][()]))"""


GLOSSARY = """\
FIELD NAMES USED THROUGHOUT THIS FILE

  gid              the identifier of a pore space, stable across the campaign
  geom_row         the row of that pore space in /design/geometry
  sample           the row of a run in the dataset the models were fitted to
  run_id           the identifier of the run directory the simulation came from
  t_index          0 to 20, which stored time
  t_norm           t / t_end, so 0.00, 0.05, ... 1.00 in every run
  pe               Peclet number, advection against diffusion
  da               Damkohler number, reaction against diffusion
  n                the number of pore voxel values a statistic was taken over

  rmse             root mean square error, in the units the model works in
  rmse_feed        the same, as a fraction of the feed concentration, QUOTE THIS
  mae              mean absolute error
  bias             mean(predicted - simulated); positive means predicted high
  r2               coefficient of determination, see /notes/metric_definitions
  max_abs_err      the single worst voxel

  pooled           taken over every value of the split at once
  per_snapshot     taken on each (run, time) separately; a diagnostic only
  to_feed_factor   multiply a stored value by this to get a fraction of the feed

  train            the pore spaces the weights were fitted to
  val              seen every epoch, but only to choose which epoch to keep
  test             never seen until the end, and read once

  C                the product model, using the geodesic distance
  C_edt            the same model told the straight line distance instead
  C_none           the same model told no distance at all"""


def main(argv=None):
    p = argparse.ArgumentParser(
        description="build one .h5 holding every result of a PRT-DeepONet "
                    "campaign",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True,
                   help="the dataset .h5 the models were fitted to")
    p.add_argument("--runs",
                   help="a directory holding one subdirectory per model, each "
                        "with best.pt and summary.json")
    p.add_argument("--model", action="append", type=lambda s: tuple(s.split("=", 1)),
                   metavar="TAG=PATH", help="name one model explicitly; repeatable")
    p.add_argument("--out", default="prt_results.h5")
    p.add_argument("--figures", help="a directory of .png files to embed")
    p.add_argument("--fields", choices=["none", "test", "all"], default="test",
                   help="which splits get the volumes themselves stored "
                        "(default: test)")
    p.add_argument("--splits", default="train,val,test",
                   type=lambda s: [x.strip() for x in s.split(",")])
    p.add_argument("--no-weights", dest="store_weights", action="store_false",
                   help="do not store the fitted weights")
    p.add_argument("--no-geometry", dest="store_geometry", action="store_false",
                   help="do not store the pore space volumes and distance fields")
    p.add_argument("--sim-seconds", type=float,
                   help="median wall time of one simulation, for the speed group")
    p.add_argument("--title", default="PRT-DeepONet results")
    p.add_argument("--campaign", default="")
    p.add_argument("--overview", help="replace the default overview note")
    p.add_argument("--provenance", help="text recording how this was produced")
    p.add_argument("--chunk", type=int, default=65536,
                   help="trunk points per forward pass")
    p.add_argument("--hash-cap", type=int, default=0,
                   help="stop checksumming the dataset after this many bytes; "
                        "0 reads all of it")
    p.add_argument("--code-root", default=os.environ.get("PRT_CODE", ""))
    p.add_argument("--code-model", default="")
    p.add_argument("--code-tools", default="")
    a = p.parse_args(argv)
    if a.code_root and not a.code_model:
        a.code_model = os.path.join(a.code_root, "3D", "model")
        a.code_tools = os.path.join(a.code_root, "3D", "tools")
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
