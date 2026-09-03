#!/usr/bin/env python3
"""
test_documented_numbers.py — check, by measurement, every claim the documents make
about the feature switches. Run it after touching anything in tools/ or model/.

    python test_documented_numbers.py
    python test_documented_numbers.py --keep /tmp/fixtures   reuse the fixtures

WHY IT EXISTS
    Every number quoted in the READMEs, the guides and the tutorial is supposed to
    come from this code. Numbers in prose go stale silently: the code changes, the
    sentence does not, and nobody finds out until somebody trusts it. This measures
    them again.

WHAT IT CHECKS

  1  with every switch off, the data pipeline is bit-for-bit what it was before
     the switches existed -- all six distance x velocity combinations
  2  a checkpoint written before the switches existed still loads, and its
     switches resolve to off; tested in two forms, including the oldest
  3  the travel time solves in about a second, lands in [0,1) after squashing,
     has the long tail that motivated the squashing, and is independent of the
     Peclet number, which it must be because Stokes is linear
  4  the extruded 2D flow field really is z-invariant, so switch B's training
     data is exact rather than approximate
  5  the dimension-free trunk has the SAME width in 2D and 3D, which is the
     whole reason it transfers, while the Cartesian trunk does not (4 vs 5)

WHAT CHANGED, AND WHY IT USED TO FAIL FOR EVERYONE
    This file used to begin by inserting /home/claude/GeometryAware3D/tools onto
    sys.path and reading /tmp/test3d.h5, /tmp/reg3d/best.pt, /tmp/tg.npz,
    /tmp/tvel.npy, /tmp/real2d.h5 and /tmp/data2d.h5. Those are paths on the
    machine it was written on. On any other machine it stopped on the first
    missing file, and because check_everything.py runs it, that failure was
    reported to every user of this project as though something were wrong with
    their installation.

    Now every path is resolved relative to THIS FILE, and every fixture is built
    here, into a temporary folder that is removed at the end. It takes a few
    minutes because two datasets and a short training run have to happen first.
    Use --keep to build them once and reuse them while working.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

# =============================================================================
#  BLOCK 1.  WHERE THINGS ARE
#
#  Everything from this file, never from an absolute path. tools/ and model/ sit
#  beside each other, and a checkout can live anywhere.
# =============================================================================
TOOLS = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(os.path.dirname(TOOLS), "model")
sys.path.insert(0, TOOLS)
sys.path.insert(0, MODEL)

import h5py                                                    # noqa: E402
import torch                                                   # noqa: E402
from dataset_reader import (PRT3DDataset, resolve_switches,     # noqa: E402
                            dataset_kwargs_from_ckpt)
from flow_coordinates import travel_time, squash, stats         # noqa: E402


R = []
SKIPPED = []


def chk(name, ok, detail=""):
    R.append((name, ok, detail))
    print("%-58s %s  %s" % (name, "PASS" if ok else "FAIL", detail))


def skip(name, why):
    # A skip is NOT a pass and is printed differently, because a run where half
    # the claims were skipped must not read like a run where they all held.
    SKIPPED.append((name, why))
    print("%-58s %s  %s" % (name, "SKIP", why))


def run(cmd, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run([str(c) for c in cmd], **kw)


# =============================================================================
#  BLOCK 2.  THE FIXTURES
#
#  Built here rather than assumed. Each one is the SMALLEST thing that can carry
#  the claim it is used for: four tiny rocks, a handful of runs, two epochs.
#  These fixtures measure nothing about accuracy and are not meant to.
# =============================================================================
def build_fixtures(d, verbose=True):
    """The four files the claims below need. Returns a dict of paths."""
    F = {}
    t0 = time.time()

    # (a) a small but REAL 3D dataset: geometry generated, flow solved,
    #     transport solved. Claims 1, 2, 3 and 5 all read it.
    F["h3"] = os.path.join(d, "test3d.h5")
    if not os.path.exists(F["h3"]):
        say(verbose, "  building the 3D practice dataset")
        # Deliberately smaller and shorter than that script's own defaults.
        # None of the claims below is about the flow field being converged; they
        # are about shapes, keys, scaling and linearity, and a converged field
        # would cost minutes per fixture for nothing.
        r = run([sys.executable, os.path.join(TOOLS, "build_practice_dataset.py"),
                 "--out", F["h3"], "--n-geom", "3", "--n-sets", "2",
                 "--n-times", "3", "--shape", "20", "14", "14",
                 "--stokes-iters", "1500"])
        if r.returncode:
            return None, "build_practice_dataset.py failed: %s" % r.stderr[-300:]

    # (b) a small 2D dataset, for the claim that the dimension-free trunk is the
    #     same width in both. It has to be a REAL 2D file, not a squeezed 3D one.
    F["h2"] = os.path.join(d, "data2d.h5")
    if not os.path.exists(F["h2"]):
        say(verbose, "  building the 2D dataset")
        r = run([sys.executable, os.path.join(TOOLS, "build_dataset_2d.py"),
                 "--out", F["h2"], "--n-geom", "4", "--n-sets", "2",
                 "--n-times", "3", "--n-species", "2",
                 "--shape", "48", "24", "--stokes-iters", "2000"])
        if r.returncode:
            return None, "build_dataset_2d.py failed: %s" % r.stderr[-300:]

    # (c) an extruded 2D set, for the claim that extrusion is EXACT. --synthetic
    #     rather than the published domains, because those are large and this
    #     claim is about the extrusion, not about whose rocks were extruded.
    F["ext"] = os.path.join(d, "real2d.h5")
    if not os.path.exists(F["ext"]):
        say(verbose, "  building the extruded 2D set")
        r = run([sys.executable,
                 os.path.join(TOOLS, "build_transfer_set_2d_to_3d.py"),
                 "--out", F["ext"], "--synthetic", "2", "--n-sets", "1",
                 "--n-times", "3", "--n-species", "2",
                 "--target-shape", "48", "24", "24", "--stokes-iters", "2000"])
        if r.returncode:
            return None, "build_transfer_set_2d_to_3d.py failed: %s" % r.stderr[-300:]

    # (d) a checkpoint, for the claim that an OLD one still loads. Two epochs is
    #     plenty: the claim is about the keys in the file, not about the weights.
    F["ck"] = os.path.join(d, "run", "best.pt")
    if not os.path.exists(F["ck"]):
        say(verbose, "  training a two-epoch checkpoint")
        r = run([sys.executable, os.path.join(MODEL, "train.py"),
                 "--data", F["h3"], "--out", os.path.join(d, "run"),
                 "--epochs", "2", "--batch-size", "2", "--n-points", "256",
                 "--workers", "0"])
        if r.returncode or not os.path.exists(F["ck"]):
            return None, "train.py failed: %s" % (r.stderr or r.stdout)[-300:]

    say(verbose, "  fixtures ready in %.0f s" % (time.time() - t0))
    return F, None


def say(verbose, s):
    if verbose:
        print(s, flush=True)


# =============================================================================
#  BLOCK 3.  CLAIM 1.  OFF IS THE ORIGINAL, BIT FOR BIT
#
#  Delegated to test_three_switches.py, which holds a verbatim copy of the
#  pre-switch data path and compares against it sample by sample. What is
#  checked HERE is that it covers all six combinations, because a version of it
#  that quietly stopped testing one of them would still print ALL TESTS PASSED.
# =============================================================================
def claim_1(F):
    print("=" * 90)
    print("CLAIM 1  OFF == original, bit-exact")
    print("=" * 90)
    r = run([sys.executable, os.path.join(TOOLS, "test_three_switches.py"),
             "--data", F["h3"]])
    n = r.stdout.count("PASS")
    chk("test_three_switches.py: all checks pass",
        "ALL TESTS PASSED" in r.stdout, "%d checks" % n)
    combos = re.findall(
        r"^  distance=(gdf|edt|none)\s+with_velocity=(True|False)\s+identical\s+(PASS|FAIL)",
        r.stdout, re.M)
    chk("  covers gdf/edt/none x with/without velocity",
        len(combos) == 6 and all(c[2] == "PASS" for c in combos),
        "%d of 6 combinations, all %s"
        % (len(combos), "PASS" if all(c[2] == "PASS" for c in combos) else "NOT pass"))


# =============================================================================
#  BLOCK 4.  CLAIM 2.  OLD CHECKPOINTS STILL LOAD
#
#  Two forms are made by STRIPPING keys out of a fresh checkpoint, rather than by
#  keeping an old file around. A kept file would be a museum piece nobody could
#  regenerate; this way the test stays true as the checkpoint format grows.
# =============================================================================
OLD_SWITCH_KEYS = ("flow_proxy", "dim_free", "flow_mode", "keep_geometry_channel",
                   "u_floor", "transfer_2d", "transfer_2d_frac", "init_from",
                   "freeze_trunk", "velocity_informed", "geom_features")


def claim_2(F, d):
    print()
    print("=" * 90)
    print("CLAIM 2  old checkpoints still load")
    print("=" * 90)
    ck = torch.load(F["ck"], map_location="cpu", weights_only=False)

    # (a) a checkpoint from before the switches existed: no switch keys at all
    a = dict(ck)
    a["args"] = {k: v for k, v in ck["args"].items() if k not in OLD_SWITCH_KEYS}
    for k in ("trunk_cols", "branch_ch", "switch_label"):
        a.pop(k, None)
    pa = os.path.join(d, "old_a.pt")
    torch.save(a, pa)

    # (b) the very oldest form: also missing the shape keys added later
    b = dict(a)
    for k in ("trunk_in_dim", "in_channels", "grid", "n_times"):
        b.pop(k, None)
    pb = os.path.join(d, "old_b.pt")
    torch.save(b, pb)

    for tag, p in (("no switch keys", pa), ("oldest form", pb)):
        kw, cfg = dataset_kwargs_from_ckpt(
            torch.load(p, map_location="cpu", weights_only=False))
        ok = (cfg["flow_proxy"] is False and cfg["dim_free"] is False
              and cfg.get("velocity_informed", "off") == "off"
              and cfg["trunk_cols"] == ["x", "y", "z", "t", "gdf"])
        chk("  %-16s -> switches default OFF" % tag, ok, cfg["label"])
        rr = run([sys.executable, os.path.join(MODEL, "evaluate.py"),
                  "--checkpoint", p, "--data", F["h3"],
                  "--out", os.path.join(d, "ev_" + tag.split()[0]),
                  "--no-3d", "--save-fields", "0"])
        chk("  %-16s -> evaluate.py runs it" % tag,
            rr.returncode == 0 and "mean RMSE" in rr.stdout,
            (rr.stdout.strip().splitlines()[-1][:60] if rr.returncode == 0
             else (rr.stderr or rr.stdout).strip()[-70:]))


# =============================================================================
#  BLOCK 5.  CLAIM 3.  SWITCH A'S TRAVEL TIME
#
#  The geometry and the velocity come out of the fixture dataset rather than out
#  of loose .npz files beside it, so there is one source of truth and no way for
#  the rock and its flow field to be from different runs.
# =============================================================================
def claim_3(F):
    print()
    print("=" * 90)
    print("CLAIM 3  switch A: tau")
    print("=" * 90)
    with h5py.File(F["h3"], "r") as h:
        gi = np.asarray(h["samples/geom_index"]).astype(int)
        mat = np.asarray(h["geom/material"][gi[0]])
        vel = np.asarray(h["samples/velocity"][0], np.float32)
        if "velocity" in h["samples"] and "scale" in h["samples/velocity"].attrs:
            vel = vel * float(h["samples/velocity"].attrs["scale"])
    if mat.ndim == 4:
        mat = mat[..., 0]

    t0 = time.time()
    raw, _stag = travel_time(vel, mat)
    el = time.time() - t0
    st = stats(raw, mat)
    m = mat == 2
    chk("  tau solves in about a second", el < 5.0,
        "%.2f s on %s" % (el, "x".join(map(str, mat.shape))))

    sq = squash(raw)
    chk("  squashed tau lands in [0,1)",
        float(sq[m].min()) >= 0 and float(sq[m].max()) < 1,
        "range %.3f..%.3f" % (sq[m].min(), sq[m].max()))
    chk("  raw tau has the long tail I quoted", st["max"] / st["median"] > 50,
        "median %.2f  p99 %.1f  max %.1f" % (st["median"], st["p99"], st["max"]))

    # Stokes is LINEAR, so scaling the velocity scales the time by the inverse
    # and the normalised travel time is unchanged. If this ever fails, either
    # the normalisation has gone or the solver is not solving Stokes.
    a2, _ = travel_time(vel * 7.3, mat)
    chk("  tau independent of Pe (u scaled x7.3)",
        float(np.nanmax(np.abs(a2[m] - raw[m]))) < 1e-4,
        "max diff %.1e" % float(np.nanmax(np.abs(a2[m] - raw[m]))))

    ds = PRT3DDataset(F["h3"], n_points=256, flow_proxy=True)
    gidx = ds.geom_index
    same = np.where(gidx == gidx[0])[0]
    c = ds.trunk_cols.index("tau")
    if len(same) >= 2:
        # Two runs on ONE rock at different Peclet numbers must give the same
        # tau column. Same seed, so the two draw the same voxels.
        x = PRT3DDataset(F["h3"], indices=[int(same[0])], n_points=999,
                         flow_proxy=True, seed=1)[0][2].numpy()
        y = PRT3DDataset(F["h3"], indices=[int(same[1])], n_points=999,
                         flow_proxy=True, seed=1)[0][2].numpy()
        chk("  dataset gives identical tau for two Pe, same geometry",
            float(np.abs(np.sort(x[:, c]) - np.sort(y[:, c])).max()) < 1e-6)
    else:
        skip("  dataset gives identical tau for two Pe, same geometry",
             "the fixture has only one run per rock")
    chk("  switch A trunk swaps gdf -> tau",
        ds.trunk_cols == ["x", "y", "z", "t", "tau"], str(ds.trunk_cols))
    chk("  switch A branch is the velocity field",
        ds.branch_ch == ["ux", "uy", "uz"], str(ds.branch_ch))


# =============================================================================
#  BLOCK 6.  CLAIM 4.  EXTRUSION IS EXACT
#
#  This is the claim switch B rests on. An extruded 2D domain is prismatic, so
#  the 3D solution IS the 2D solution repeated, which makes 3000 cheap domains
#  genuine members of the 3D problem class rather than an approximation of one.
#  The builder measures the z-variation of the flow it produced and stores it;
#  anything but exactly zero means the claim is wrong.
# =============================================================================
def claim_4(F):
    print()
    print("=" * 90)
    print("CLAIM 4  switch B: extrusion is exact")
    print("=" * 90)
    with h5py.File(F["ext"], "r") as h:
        zv = float(h.attrs["max_z_variation"])
        src = h.attrs["source"]
        src = src.decode() if isinstance(src, bytes) else str(src)
        nzs = int(h.attrs["nz_solve"])
        shape = tuple(int(v) for v in h.attrs["shape"])
    chk("  measured z-variation of the extruded flow", zv == 0.0,
        "%.2e (0 = exact)" % zv)
    chk("  the file says how it was built", src == "extruded_2d",
        "source=%s  nz_solve=%d  grid=%s" % (src, nzs, shape))


# =============================================================================
#  BLOCK 7.  CLAIM 5.  THE DIMENSION-FREE TRUNK IS THE SAME WIDTH IN BOTH
#
#  This is the whole argument for switch C. A network transfers between 2D and
#  3D unchanged only if its trunk has the same number of inputs in both, and the
#  Cartesian trunk does not: (x,y,t,gdf) against (x,y,z,t,gdf).
# =============================================================================
def claim_5(F):
    print()
    print("=" * 90)
    print("CLAIM 5  switch C: 3 columns in BOTH 2D and 3D")
    print("=" * 90)
    d3 = PRT3DDataset(F["h3"], n_points=64, dim_free=True)
    d2 = PRT3DDataset(F["h2"], n_points=64, dim_free=True)
    chk("  3D dim-free trunk", d3.trunk_cols == ["t", "dwall", "tau"],
        "%s  (ndim=%d)" % (d3.trunk_cols, d3.ndim))
    chk("  2D dim-free trunk", d2.trunk_cols == ["t", "dwall", "tau"],
        "%s  (ndim=%d)" % (d2.trunk_cols, d2.ndim))
    chk("  identical width, so the net transfers unchanged",
        d3.trunk_dim == d2.trunk_dim == 3)
    chk("  and the CARTESIAN trunk does NOT (4 vs 5)",
        resolve_switches(ndim=2)["trunk_dim"] == 4
        and resolve_switches(ndim=3)["trunk_dim"] == 5,
        "2D=%d  3D=%d" % (resolve_switches(ndim=2)["trunk_dim"],
                          resolve_switches(ndim=3)["trunk_dim"]))


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", default=None,
                    help="build the fixtures here and leave them, so a second "
                         "run starts immediately. Without it they go to a "
                         "temporary folder that is removed at the end.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    d = a.keep or tempfile.mkdtemp(prefix="prt_documented_")
    os.makedirs(d, exist_ok=True)
    print("fixtures: %s%s" % (d, "" if a.keep else "  (removed at the end)"))
    F, why = build_fixtures(d, verbose=not a.quiet)
    if F is None:
        # A fixture that will not build is a real failure, not a skip: every
        # claim below reads one, so there is nothing left to measure.
        print("\nCOULD NOT BUILD THE FIXTURES: %s" % why)
        if not a.keep:
            shutil.rmtree(d, ignore_errors=True)
        return 1

    try:
        claim_1(F)
        claim_2(F, d)
        claim_3(F)
        claim_4(F)
        claim_5(F)
    finally:
        if not a.keep:
            shutil.rmtree(d, ignore_errors=True)

    print()
    print("=" * 90)
    bad = [n for n, ok, _ in R if not ok]
    print("%d checks, %d passed, %d FAILED, %d skipped"
          % (len(R), len(R) - len(bad), len(bad), len(SKIPPED)))
    if bad:
        print("FAILURES:")
        for n in bad:
            print("   -", n)
    if SKIPPED:
        print("SKIPPED (weaker than passing):")
        for n, w in SKIPPED:
            print("   -", n, "--", w)
    # EXIT NON-ZERO WHEN SOMETHING FAILED. Without this the shell status was
    # always 0, so check_everything.py reported this group as ok no matter how
    # many claims had drifted -- a test that cannot fail is not a test.
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
