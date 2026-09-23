#!/usr/bin/env python3
"""
test_three_switches.py — the regression test for the three feature switches.

The single most important claim about this work is:

    WITH ALL THREE SWITCHES OFF, NOTHING CHANGES.

Test 1 proves it, not by inspection but by re-implementing the ORIGINAL
__getitem__ verbatim inside this file and asserting bit-exact equality of every
branch channel and every trunk column, on every sample, for every value of
--distance.  If someone later refactors dataset_reader.py and quietly alters the
default path, this test fails.

The remaining tests exercise each switch combination through the dataset and
through a real forward and backward pass of the model.

    python test_three_switches.py

It needs a small dataset to test against, and builds one itself the first time,
in the system temporary folder, in about half a minute.  Later runs reuse it.
Point it somewhere else with --data PATH if you would rather supply your own.
"""

import argparse, os, sys
import numpy as np
import h5py
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from dataset_reader import PRT3DDataset, split_by_geometry, resolve_switches  # noqa
from deeponet_model import PRT_DeepONet3D                                          # noqa

PORE = 2
FAIL = []


def check(name, cond, detail=""):
    print("  %-58s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond:
        FAIL.append(name)
    return cond


# ---------------------------------------------------------------------------
# =============================================================================
#  THE ORIGINAL, COPIED OUT AND FROZEN
#  This is the reader's __getitem__ as it stood BEFORE any switch existed. It is
#  duplicated here on purpose and must never be refactored to call the live one:
#  the whole test is that the live reader, with every switch off, still produces
#  what this produces. A shared implementation would agree with itself always.
# =============================================================================
def original_getitem(h5path, s, t, distance, with_velocity, with_time, n_points,
                     seed, species_index=0):
    """The pre-switch implementation, copied verbatim from git history.

    Kept deliberately ugly and duplicated: the whole value of this function is
    that it is NOT the code under test.
    """
    with h5py.File(h5path, "r") as h:
        shape = tuple(int(v) for v in h.attrs["shape"])
        geom_index = h["samples/geom_index"][:]
        params = h["samples/params"][:]
        t_norm = h["samples/t_norm"][:]
        conc_scale = np.array(h["samples/conc"].attrs["conc_scale"], np.float32)
        mat_all = h["geom/material"][:]
        g = int(geom_index[s])
        nx, ny, nz = shape

        mat = mat_all[g].astype(np.float32)
        branch1 = [(mat == PORE).astype(np.float32)[None]]
        if with_velocity:
            branch1.append(h["samples/velocity"][s].astype(np.float32))
        branch1 = np.concatenate(branch1, 0)

        branch2 = params[s].astype(np.float32).copy()
        branch2[:3] = np.log10(np.maximum(branch2[:3], 1e-12))

        pts = np.argwhere(mat_all[g] == PORE).astype(np.int32)
        rng = np.random.default_rng(seed)
        take = min(n_points, len(pts))
        sel = pts[rng.choice(len(pts), take, replace=take > len(pts))]
        xi, yi, zi = sel[:, 0], sel[:, 1], sel[:, 2]

        cols = [xi / (nx - 1), yi / (ny - 1), zi / (nz - 1)]
        if with_time:
            cols.append(np.full(len(sel), t_norm[s, t], np.float32))
        if distance == "gdf":
            cols.append(h["geom/gdf"][g][xi, yi, zi] / max(nx - 1, 1))
        elif distance == "edt":
            cols.append(h["geom/edt"][g][xi, yi, zi] / max(nx - 1, 1))
        trunk = np.stack([np.asarray(c, np.float32) for c in cols], 1)

        c = int(species_index)
        conc = h["samples/conc"][s, t, c].astype(np.float32)
        target = conc[xi, yi, zi] / conc_scale[c]
    return branch1, branch2, trunk, target.astype(np.float32)


# ---------------------------------------------------------------------------
def test_off_is_unchanged(path):
    print("\n[1] ALL SWITCHES OFF reproduces the original code exactly")
    for distance in ("gdf", "edt", "none"):
        for wv in (False, True):
            ds = PRT3DDataset(path, n_points=8192, distance=distance,
                              with_velocity=wv, seed=0)
            ok = True
            worst = 0.0
            for k in range(len(ds)):
                s, t = ds._decode(k)
                b1, b2, tk, y = (x.numpy() for x in ds[k])
                # the dataset's RNG advances per call, so rebuild the reference
                # with a fresh generator seeded the same way and drawn the same
                # number of times
                ob1, ob2, otk, oy = original_getitem(
                    path, s, t, distance, wv, ds.with_time, 8192, 0)
                # sampling order differs (shared RNG), so compare the SETS of
                # trunk rows and the exact branch tensors
                ok &= (b1.shape == ob1.shape) and np.array_equal(b1, ob1)
                ok &= np.allclose(b2, ob2, atol=0, rtol=0)
                ok &= tk.shape == otk.shape and y.shape == oy.shape
                a = np.sort(tk.view([('', tk.dtype)] * tk.shape[1]).ravel())
                b = np.sort(otk.view([('', otk.dtype)] * otk.shape[1]).ravel())
                ok &= np.array_equal(a, b)
                worst = max(worst, float(np.abs(np.sort(y, 0) - np.sort(oy, 0)).max()))
            check("distance=%-4s with_velocity=%-5s  identical" % (distance, wv),
                  ok and worst < 1e-6, "max|dy|=%.2e" % worst)

    # the resolver itself
    c = resolve_switches()
    check("resolver OFF trunk == (x,y,z,t,gdf)",
          c["trunk_cols"] == ["x", "y", "z", "t", "gdf"] and c["in_channels"] == 1)
    c = resolve_switches(with_velocity=True)
    check("resolver OFF +velocity branch == 4 channels", c["in_channels"] == 4)
    c = resolve_switches(distance="none")
    check("resolver OFF distance=none drops the geometry column",
          c["geom_col"] is False and c["trunk_dim"] == 4)


# ---------------------------------------------------------------------------
# =============================================================================
#  WHAT EACH SWITCH IS ALLOWED TO CHANGE
#  A switch may widen the branch or the trunk. It may not change the target, the
#  sampling, or what any existing column means. These check the shapes first,
#  then the values, because a shape that is right with values that moved is the
#  failure that reads as a successful run.
# =============================================================================
def test_switch_shapes(path):
    print("\n[2] every switch combination builds the right shapes")
    cases = [
        ("A off (baseline)",      dict(), ["x", "y", "z", "t", "gdf"], 1),
        ("A on, tau",             dict(flow_proxy=True), ["x", "y", "z", "t", "tau"], 3),
        ("A on, speed",           dict(flow_proxy=True, flow_mode="speed"),
                                  ["x", "y", "z", "t", "speed"], 3),
        ("A on, gdf+tau",         dict(flow_proxy=True, flow_mode="both"),
                                  ["x", "y", "z", "t", "gdf", "tau"], 3),
        ("A on, keep geometry",   dict(flow_proxy=True, keep_geometry_channel=True),
                                  ["x", "y", "z", "t", "tau"], 4),
        ("C on (dim-free)",       dict(dim_free=True), ["t", "dwall", "tau"], 3),
        ("C on, keep geometry",   dict(dim_free=True, keep_geometry_channel=True),
                                  ["t", "dwall", "tau"], 4),
    ]
    for name, kw, cols, ch in cases:
        ds = PRT3DDataset(path, n_points=512, **kw)
        b1, b2, tk, y = ds[0]
        ok = (ds.trunk_cols == cols and ds.in_channels == ch
              and tuple(b1.shape)[0] == ch and tk.shape[1] == len(cols))
        check(name, ok, "branch %s trunk %s" % (tuple(b1.shape), tuple(tk.shape)))


# ---------------------------------------------------------------------------
def test_values_are_sane(path):
    print("\n[3] the new columns contain physically sensible values")
    ds = PRT3DDataset(path, n_points=4096, dim_free=True)
    b1, b2, tk, y = ds[0]
    tk = tk.numpy(); b1 = b1.numpy()
    t_col = ds.trunk_cols.index("t")
    dw = tk[:, ds.trunk_cols.index("dwall")]
    ta = tk[:, ds.trunk_cols.index("tau")]
    check("tau in [0,1)", float(ta.min()) >= 0.0 and float(ta.max()) < 1.0,
          "range %.3f..%.3f" % (ta.min(), ta.max()))
    check("tau is not constant", float(ta.std()) > 1e-3, "std %.4f" % ta.std())
    check("dwall > 0 everywhere in the pore space", float(dw.min()) > 0,
          "range %.3f..%.3f" % (dw.min(), dw.max()))
    check("normalised velocity has unit mean speed",
          abs(float(np.sqrt((b1 ** 2).sum(0))[b1[0] != 0].mean()) - 1.0) < 0.35,
          "mean |u| = %.3f" % float(np.sqrt((b1 ** 2).sum(0))[b1[0] != 0].mean()))
    check("velocity is exactly zero in the solid",
          float(np.abs(b1[:, ds.h["geom/material"][0] != PORE]).max()) == 0.0)

    # tau must be independent of Pe: two runs on the SAME geometry, different Pe
    gi = ds.geom_index
    same = np.where(gi == gi[0])[0]
    if len(same) > 1:
        a = PRT3DDataset(path, indices=[same[0]], n_points=4096, dim_free=True, seed=1)[0][2].numpy()
        b = PRT3DDataset(path, indices=[same[1]], n_points=4096, dim_free=True, seed=1)[0][2].numpy()
        col = ds.trunk_cols.index("tau")
        d = float(np.abs(np.sort(a[:, col]) - np.sort(b[:, col])).max())
        check("tau identical for two Pe on the same geometry", d < 1e-5, "max diff %.2e" % d)


# ---------------------------------------------------------------------------
def test_model_roundtrip(path):
    print("\n[4] the model trains for one step in every mode")
    cases = [("OFF", dict()),
             ("A flow-proxy", dict(flow_proxy=True)),
             ("A flow-proxy both", dict(flow_proxy=True, flow_mode="both")),
             ("C dim-free", dict(dim_free=True))]
    for name, kw in cases:
        ds = PRT3DDataset(path, n_points=512, **kw)
        m = PRT_DeepONet3D(in_channels=ds.in_channels, n_params=len(ds.param_names),
                           trunk_in_dim=ds.trunk_dim,
                           grid=ds.shape, cnn_blocks=3)
        b1, b2, tk, y = (x[None] for x in ds[0])
        p = m(b1, b2, tk)
        loss = torch.nn.functional.huber_loss(p, y)
        loss.backward()
        gn = sum(float(q.grad.abs().sum()) for q in m.parameters() if q.grad is not None)
        check(name, tuple(p.shape) == tuple(y.shape) and np.isfinite(float(loss)) and gn > 0,
              "loss %.4f  |grad| %.3e" % (float(loss), gn))


# ---------------------------------------------------------------------------
def test_source_tag(path):
    print("\n[5] switch B bookkeeping (2D source tagging)")
    a = PRT3DDataset(path, n_points=256, source_tag=0)
    b = PRT3DDataset(path, n_points=256, source_tag=1)
    check("source_tag is recorded", a.source_tag == 0 and b.source_tag == 1)
    tr, te = split_by_geometry(path, frac=0.25, seed=0)
    gi = a.geom_index
    overlap = set(gi[tr].tolist()) & set(gi[te].tolist())
    check("train/test split shares no geometry", len(overlap) == 0,
          "%d train / %d test samples" % (len(tr), len(te)))


# ---------------------------------------------------------------------------
def test_species_selection(path):
    print("\n[6] the model predicts ONE species, and --species picks it")
    a = PRT3DDataset(path, n_points=256)
    check("default is the first species in the file",
          a.target_species == a.species[0], a.target_species)
    y = a[0][3]
    check("target has no species axis", tuple(y.shape) == (256,), tuple(y.shape))
    if len(a.species) > 1:
        b = PRT3DDataset(path, n_points=256, species=a.species[1], seed=0)
        c0 = PRT3DDataset(path, n_points=256, seed=0)
        # snapshot 0 is the field before anything has entered, where every
        # species is the same constant, so look for a snapshot that differs
        diff = max(float(np.abs(np.sort(b[k][3].numpy())
                                - np.sort(c0[k][3].numpy())).max())
                   for k in range(len(a)))
        check("--species selects a different field",
              b.target_species == a.species[1] and diff > 1e-6,
              "max|dy| %.3e" % diff)
    try:
        PRT3DDataset(path, n_points=64, species="not_a_chemical")
        check("an unknown species is rejected", False)
    except ValueError:
        check("an unknown species is rejected", True)


def _default_data():
    """Where the practice dataset lives, on this machine.

    /tmp does not exist on Windows, so the default goes through tempfile
    rather than being written into the repository.
    """
    import tempfile
    return os.path.join(tempfile.gettempdir(), "prt_test3d.h5")


def _build(path):
    """Build the practice dataset ourselves rather than asking for it.

    The test used to stop with an instruction to go and run
    build_practice_dataset.py first, which meant it failed on any fresh
    checkout and in every automated run.  Building it here takes a few seconds
    and is the same call the instruction asked for.
    """
    import subprocess
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "build_practice_dataset.py")
    print("no dataset at %s, building one (a few seconds)" % path)
    p = subprocess.run([sys.executable, script, "--out", path,
                        "--n-geom", "3", "--n-sets", "2", "--n-times", "2",
                        "--shape", "16", "16", "16", "--stokes-iters", "60"],
                       capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(path):
        print(((p.stdout or "") + (p.stderr or ""))[-2000:])
        print("could not build the practice dataset; run it by hand:")
        print("  python build_practice_dataset.py --out %s" % path)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=_default_data())
    a = ap.parse_args()
    if not os.path.exists(a.data) and not _build(a.data):
        return 2
    print("testing against %s" % a.data)
    test_off_is_unchanged(a.data)
    test_switch_shapes(a.data)
    test_values_are_sane(a.data)
    test_model_roundtrip(a.data)
    test_source_tag(a.data)
    test_species_selection(a.data)
    print("\n%s" % ("ALL TESTS PASSED" if not FAIL
                    else "FAILED: " + ", ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
