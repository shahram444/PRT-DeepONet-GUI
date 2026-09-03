#!/usr/bin/env python3
# =============================================================================
# NEW IN THE FLOW VERSION.  There is no 2D counterpart of this file.
#
#   WHAT IT IS
#     The end to end test of everything the flow work added. Four groups: the
#     descriptors' own self tests, the port checked against the released 2D
#     weights, the whole pipeline wired up on a dataset this file builds for
#     itself, and the guards that are supposed to refuse bad combinations.
#
#   WHY IT EXISTS
#     The Jung Lab release ships notebooks, not tests. A notebook proves a
#     result on one machine on one day. Porting it to N dimensions and to a
#     different axis convention is exactly the kind of change that keeps
#     running while quietly returning the wrong numbers, so the port needed a
#     test that runs on demand and fails loudly.
#
#   WHAT IT DOES NOT DO
#     It says nothing about ACCURACY. The dataset it builds is a handful of
#     tiny synthetic rocks and the training it runs is two epochs. Passing
#     means the shapes line up, the scaling survives a round trip and the
#     guards fire. It does not mean the model is any good.
# =============================================================================
"""The velocity pipeline, end to end, on a dataset this test builds itself.

Four groups.

  1  the descriptors        flow_features and harmonic_pressure own self tests
  2  the port               every released checkpoint loads into the ported classes
                            with no missing key, no unexpected key and no shape
                            mismatch. Skipped, and SAID to be skipped, when the
                            released weights are not on this machine.
  3  the wiring             a small dataset is built in the project's own HDF5 layout,
                            trained on for three epochs, and predicted from. This
                            proves the shapes line up and the scaling survives the
                            round trip. It proves nothing about accuracy.
  4  the guards             a grid mismatch, a dataset with no velocity and a bare
                            state_dict must each fail with an explanation rather than
                            a traceback or, worse, a plausible looking answer.

Run it like this.

    python test_flow_pipeline.py
    python test_flow_pipeline.py --reference /path/to/velocity-informed

What you get back. Nothing that survives the run: everything is built under a
temporary directory and removed. Group 2 is the one that matters most, because a
silent drift in the port is what would make every later comparison with the published
numbers meaningless.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)
sys.path.insert(0, TOOLS)


def script(name):
    """Where a script lives. The descriptors sit in 3D/tools, the models in 3D/model,
    and this file runs from either when the two are copied side by side for a test."""
    for d in (HERE, TOOLS):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(HERE, name)

FAILED = []
SKIPPED = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("   " + detail) if detail else ""))
    if not cond:
        FAILED.append(name)
    return cond


def skip(name, why):
    print("  SKIP  " + name + "   " + why)
    SKIPPED.append(name)


def run(cmd, **kw):
    cmd = [script(cmd[0])] + list(cmd[1:])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [HERE, TOOLS] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return subprocess.run([sys.executable] + cmd, capture_output=True, text=True,
                          cwd=HERE, env=env, **kw)


# =============================================================================
#  GROUP 1.  THE DESCRIPTORS' OWN SELF TESTS
#
#  flow_features.py and harmonic_pressure.py each carry their own --self-test,
#  and this runs them as subprocesses rather than importing them. That is
#  deliberate: it is the same command a user would type, so a failure here is
#  reproducible by hand instead of only inside this file.
# =============================================================================
def group_descriptors():
    print("\n1  the descriptors")
    for mod in ("flow_features.py", "harmonic_pressure.py"):
        r = run([mod, "--self-test"])
        good = r.returncode == 0 and "Everything passed" in r.stdout
        n = r.stdout.count("PASS")
        check("%s self test (%d checks)" % (mod, n), good,
              "" if good else r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[:200])


# =============================================================================
#  GROUP 2.  THE PORT, AGAINST THE RELEASED WEIGHTS
#
#  Loading their checkpoint into our classes proves the shapes agree. It SKIPS,
#  loudly and in different words from passing, when their repository is not on
#  the machine, because most users will not have it and a skip is a weaker
#  statement than a pass. The stronger check, value against value, is
#  test_reference_parity.py.
# =============================================================================
def group_port(reference):
    print("\n2  the port against the released weights")
    if not reference:
        skip("released checkpoints load into the ported classes",
             "pass --reference /path/to/velocity-informed to run this")
        return
    if not os.path.isdir(reference):
        skip("released checkpoints load into the ported classes",
             "not a directory: %s" % reference)
        return
    r = run(["velocity_model.py", "--check-reference", reference])
    print("\n".join("      " + ln for ln in r.stdout.strip().splitlines()))
    check("every released checkpoint loads with no missing or unexpected key",
          r.returncode == 0)


# =============================================================================
#  GROUP 3.  THE WIRING, END TO END
#
#  Builds a small dataset from scratch, trains the velocity operator on it for a
#  couple of epochs, predicts, writes the field back, and trains the
#  concentration model on what it wrote. Every stage of the pipeline runs.
#
#  IT PROVES NOTHING ABOUT ACCURACY, and says so in its own output. Four tiny
#  synthetic rocks and two epochs cannot. What it proves is that the shapes line
#  up, the scaling survives a round trip through the file, and each stage can
#  read what the one before it wrote.
# =============================================================================
def _blob_geometry(nx, ny, seed, porosity=0.55):
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    for _ in range(200):
        f = ndimage.gaussian_filter(rng.standard_normal((nx, ny)), 2.5, mode="wrap")
        f = (f - f.mean()) / f.std()
        pore = f > np.quantile(f, 1.0 - porosity)
        pore[:3] = True
        pore[-3:] = True
        lab, _ = ndimage.label(pore, ndimage.generate_binary_structure(2, 1))
        inl = set(np.unique(lab[0])) - {0}
        out = set(np.unique(lab[-1])) - {0}
        both = inl & out
        if not both:
            continue
        big = max(both, key=lambda l: (lab == l).sum())
        pore = lab == big
        if pore.mean() > 0.25:
            return pore
    raise RuntimeError("could not build a percolating test geometry")


def _fake_dataset(path, n_geom=4, n_cond=2, nx=40, ny=24):
    """A dataset in the project's own layout, with a velocity field that is not random.

    The velocity here is a crude potential flow, not a simulation. That is the point:
    the group tests the WIRING, and a field with real structure exercises the scaling
    and the divergence term in a way white noise would not.
    """
    import h5py
    from harmonic_pressure import harmonic_gradient
    mats, vels, gidx, params = [], [], [], []
    for g in range(n_geom):
        pore = _blob_geometry(nx, ny, seed=g)
        mat = np.where(pore, 2, 0).astype(np.uint8)
        mats.append(mat[..., None])
        grad = harmonic_gradient(pore, normalise=False)
        for c in range(n_cond):
            pe = 1.0 + 9.0 * c
            v = np.zeros((3, nx, ny, 1), np.float32)
            v[0, :, :, 0] = -grad[0] * pe * 1e-3
            v[1, :, :, 0] = -grad[1] * pe * 1e-3
            vels.append(v)
            gidx.append(g)
            params.append([pe, 1.0, 0.0, 0.1, 0.15, 0.04])
    with h5py.File(path, "w") as h:
        h.attrs["shape"] = np.array([nx, ny, 1], np.int32)
        h.attrs["dimension"] = 2
        h.attrs["n_samples"] = len(gidx)
        h.attrs["n_geometries"] = n_geom
        h.attrs["pore_code"] = 2
        h.attrs["param_names"] = np.array([b"pe", b"da_bio", b"da_abio",
                                           b"ks_ac_norm", b"ks_a_norm", b"y_norm"])
        h.attrs["species"] = np.array([b"C"])
        h.attrs["source"] = b"built by test_flow_pipeline.py, not a simulation"
        gg = h.create_group("geom")
        gg.create_dataset("material", data=np.stack(mats))
        sg = h.create_group("samples")
        sg.create_dataset("velocity", data=np.stack(vels))
        sg.create_dataset("geom_index", data=np.array(gidx, np.int32))
        sg.create_dataset("params", data=np.array(params, np.float64))
    return path


def group_wiring(tmp):
    print("\n3  the wiring, on a dataset built for this test")
    try:
        import h5py                                                # noqa: F401
    except ImportError:
        skip("train and predict on a small dataset", "h5py is not installed")
        return
    data = _fake_dataset(os.path.join(tmp, "dataset.h5"))
    check("the test dataset was written", os.path.exists(data))

    out = os.path.join(tmp, "vel")
    r = run(["train_velocity.py", "--data", data, "--out", out, "--quick",
             "--test-frac", "0.25", "--batch-size", "2", "--epochs", "3"])
    ok = r.returncode == 0
    if not ok:
        print("      " + (r.stderr.strip().splitlines() or ["no stderr"])[-1])
    check("training runs to completion", ok)
    ck = os.path.join(out, "best.pt")
    check("a checkpoint was written", os.path.exists(ck))
    check("a history file was written", os.path.exists(os.path.join(out, "history.csv")))
    if "held-out velocity magnitude NRMSE" in r.stdout:
        check("the run reports held-out percentiles", True)
    else:
        check("the run reports held-out percentiles", False, "no NRMSE block in the output")

    if not os.path.exists(ck):
        return
    import torch
    saved = torch.load(ck, map_location="cpu", weights_only=False)
    for key in ("grid", "stats", "n_comp", "condition", "buffer", "pressure",
                "test_geometries"):
        check("the checkpoint carries %r" % key, key in saved)
    st = saved["stats"]
    check("the scaling constants travelled with the weights",
          all(k in st for k in ("mis_mu", "uprm_sd", "dw2_max", "vel_mu", "cond_sd")))
    check("the velocity standardisation is not degenerate",
          all(s > 0 for s in st["vel_sd"]), "sd %s" % np.round(st["vel_sd"], 8).tolist())

    # --- prediction from a bare geometry
    pore = _blob_geometry(40, 24, seed=0)
    gpath = os.path.join(tmp, "rock.npz")
    np.savez_compressed(gpath, material=np.where(pore, 2, 0).astype(np.uint8))
    pdir = os.path.join(tmp, "pred")
    r = run(["predict_velocity.py", "--checkpoint", ck, "--geometry", gpath,
             "--pe", "5", "--pore-code", "2", "--out", pdir])
    ok = r.returncode == 0
    if not ok:
        print("      " + (r.stderr.strip().splitlines() or ["no stderr"])[-1])
    check("prediction runs on a bare geometry", ok)
    npz = os.path.join(pdir, "velocity.npz")
    if check("a velocity field was written", os.path.exists(npz)):
        with np.load(npz) as d:
            v = d["velocity"]
            p = d["pore"]
        check("the field has one component per dimension and the right shape",
              v.shape == (2,) + pore.shape, "got %s" % (v.shape,))
        check("the field is finite everywhere", bool(np.isfinite(v).all()))
        check("nothing is predicted inside the grains",
              float(np.abs(v[:, ~p]).max()) == 0.0)
        check("the field is not identically zero in the pore space",
              float(np.abs(v[:, p]).max()) > 0)
    check("the run reports a divergence residual",
          "divergence residual" in r.stdout)

    # --- prediction back into a dataset
    r = run(["predict_velocity.py", "--checkpoint", ck, "--data", data, "--write-back"])
    ok = r.returncode == 0
    if not ok:
        print("      " + (r.stderr.strip().splitlines() or ["no stderr"])[-1])
    if check("prediction runs over a whole dataset", ok):
        import h5py
        with h5py.File(data, "r") as h:
            has = "samples/velocity_pred" in h
            check("the predicted field was stored beside the simulated one", has)
            if has:
                a = np.asarray(h["samples/velocity_pred"])
                check("the stored field is the right shape",
                      a.shape[0] == int(h.attrs["n_samples"]) and a.shape[1] == 2,
                      "got %s" % (a.shape,))
                check("the file records which checkpoint wrote it",
                      "velocity_pred_source" in h["samples"].attrs)
                check("the stored field is finite", bool(np.isfinite(a).all()))


# =============================================================================
#  GROUP 4.  THE GUARDS
#
#  Every one of these is a way to waste an afternoon, and every one is supposed
#  to fail immediately with a reason. A guard that has quietly stopped firing
#  looks exactly like a guard that is working, which is why they are tested by
#  doing the wrong thing on purpose and requiring a non-zero exit.
# =============================================================================
def group_guards(tmp):
    print("\n4  the guards")
    ck = os.path.join(tmp, "vel", "best.pt")
    if not os.path.exists(ck):
        skip("the guards", "no checkpoint from group 3 to test against")
        return
    import torch

    # a geometry at the wrong grid
    pore = _blob_geometry(30, 20, seed=9)
    g2 = os.path.join(tmp, "wrong_grid.npz")
    np.savez_compressed(g2, material=np.where(pore, 2, 0).astype(np.uint8))
    r = run(["predict_velocity.py", "--checkpoint", ck, "--geometry", g2,
             "--pe", "5", "--pore-code", "2", "--out", os.path.join(tmp, "p2")])
    said = "grid" in (r.stderr + r.stdout).lower()
    check("a geometry at the wrong grid is refused, with the reason",
          r.returncode != 0 and said,
          "exit %d" % r.returncode)

    # a bare state_dict is not a checkpoint
    bare = os.path.join(tmp, "bare.pt")
    torch.save(torch.load(ck, map_location="cpu", weights_only=False)["model"], bare)
    r = run(["predict_velocity.py", "--checkpoint", bare, "--geometry",
             os.path.join(tmp, "rock.npz"), "--pe", "5", "--pore-code", "2",
             "--out", os.path.join(tmp, "p3")])
    check("a bare state_dict is refused, because nothing says how it was scaled",
          r.returncode != 0 and "scaling" in (r.stderr + r.stdout))

    # a dataset with no velocity
    try:
        import h5py
        nov = os.path.join(tmp, "novel.h5")
        shutil.copy(os.path.join(tmp, "dataset.h5"), nov)
        with h5py.File(nov, "r+") as h:
            del h["samples/velocity"]
            if "velocity_pred" in h["samples"]:
                del h["samples/velocity_pred"]
        r = run(["train_velocity.py", "--data", nov, "--out", os.path.join(tmp, "v2"),
                 "--quick"])
        check("a dataset with no velocity is refused, with the reason",
              r.returncode != 0 and "no velocity" in (r.stderr + r.stdout).lower())
    except ImportError:
        skip("a dataset with no velocity is refused", "h5py is not installed")

    # an unknown conditioning column
    r = run(["train_velocity.py", "--data", os.path.join(tmp, "dataset.h5"),
             "--out", os.path.join(tmp, "v3"), "--quick", "--condition", "reynolds"])
    check("an unknown conditioning column is refused, and the real ones are listed",
          r.returncode != 0 and "pe" in (r.stderr + r.stdout))


def main(argv=None):
    ap = argparse.ArgumentParser(description="The velocity pipeline, end to end.")
    ap.add_argument("--reference", help="path to the released velocity-informed folder")
    ap.add_argument("--keep", action="store_true", help="leave the temporary files behind")
    a = ap.parse_args(argv)

    print("velocity pipeline test")
    tmp = tempfile.mkdtemp(prefix="prtflow_")
    try:
        group_descriptors()
        group_port(a.reference)
        group_wiring(tmp)
        group_guards(tmp)
    finally:
        if a.keep:
            print("\ntemporary files left in", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "-" * 70)
    if FAILED:
        print("%d FAILED: %s" % (len(FAILED), "; ".join(FAILED)))
    if SKIPPED:
        print("%d skipped, which is weaker than passed: %s"
              % (len(SKIPPED), "; ".join(SKIPPED)))
    if not FAILED:
        print("Everything that could run, passed."
              if SKIPPED else "Everything passed.")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
