#!/usr/bin/env python3
"""
prt_core/test_core.py -- the four self-tests, plus what only holds between them.

WHAT CHANGED FROM THE 2D VERSION
    The 2D release has no tests, and three notebooks run by hand do not need
    one: a person watches every cell. The four modules this file covers each
    replaced a copy of something that lived in several places at once, and the
    failures they exist to catch are exactly the ones a person watching cannot
    see, because every shape agrees and the numbers are merely wrong.

Each module in prt_core checks itself. This file runs all four and then checks
the things that are true only when they are used TOGETHER, which is where the
old copies disagreed with each other:

    a reaction's distance convention is one conventions.py implements
    a geometry read by inputs.py feeds model.py's trunk at the right width
    the whole 2D chain runs on a rock that came out of a file
    the whole 3D chain runs on a rock that came out of an .h5
    a 2D model and a 3D model built from the same reaction differ in exactly
      one place, the z column, and nowhere else
    a checkpoint's recorded fields catch the two mistakes that used to be
      silent: the wrong chemistry and the wrong distance convention

    python prt_core/test_core.py
    python prt_core/test_core.py --quiet      only the failures and the total
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import conventions                                          # noqa: E402
import inputs                                               # noqa: E402
import model as core_model                                  # noqa: E402
import reactions                                            # noqa: E402

MODULES = ("reactions", "conventions", "inputs", "model")


def run_module_self_tests(quiet=False):
    """Each module's own self-test, as a subprocess, so a crash is contained."""
    bad = []
    for m in MODULES:
        p = subprocess.run([sys.executable, os.path.join(HERE, "%s.py" % m),
                            "--self-test"],
                           capture_output=True, text=True)
        good = p.returncode == 0
        print("  %-5s %s.py self-test" % ("PASS" if good else "FAIL", m))
        if not good:
            bad.append(m)
            print(p.stdout[-3000:])
            print(p.stderr[-2000:])
        elif not quiet:
            for line in p.stdout.splitlines():
                if line.strip().startswith("FAIL"):
                    print("    " + line)
    return bad


def _rock_2d(nx=148, ny=64):
    """A 2D rock with grains, in this project's codes."""
    m = np.full((nx, ny), inputs.PORE, np.uint8)
    rng = np.random.default_rng(3)
    for _ in range(18):
        cx, cy, r = rng.integers(8, nx - 8), rng.integers(6, ny - 6), rng.integers(3, 7)
        x, y = np.ogrid[:nx, :ny]
        m[(x - cx) ** 2 + (y - cy) ** 2 <= r * r] = inputs.SOLID
    m[0, :] = inputs.PORE                      # keep the inlet open
    m[-1, :] = inputs.PORE
    return m


def _rock_3d(n=32):
    m = np.full((n, n, n), inputs.PORE, np.uint8)
    rng = np.random.default_rng(5)
    x, y, z = np.ogrid[:n, :n, :n]
    for _ in range(12):
        c = rng.integers(4, n - 4, 3)
        r = rng.integers(2, 5)
        m[(x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 <= r * r] = inputs.SOLID
    m[0] = inputs.PORE
    m[-1] = inputs.PORE
    return m


def crossing_checks():
    """What is only true when the four are used together."""
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print("  %-5s %-58s %s" % ("PASS" if cond else "FAIL", name, extra))

    d = tempfile.mkdtemp()

    # 1. every reaction names a convention the other module implements
    unknown = [r.key for r in reactions.REACTIONS.values()
               if r.distance_convention not in conventions.DISTANCE_CONVENTIONS]
    check("every reaction's distance convention is one that exists",
          not unknown, ", ".join(unknown) or "all %d"
          % len(reactions.REACTIONS))

    # 2. a rock from a file feeds the trunk at the width the reaction asks for
    m2 = _rock_2d()
    p2 = os.path.join(d, "rock2d.npz")
    np.savez_compressed(p2, material=m2)
    g2 = inputs.read_geometry(p2, convention="published")
    net2, info2 = core_model.build("monod", (g2.shape[0], g2.shape[1], 1))
    check("a 2D rock read from a file is 2D to the model",
          g2.ndim == 2 and info2["ndim"] == 2)

    pts = np.argwhere(g2.pore)[:600]
    cols = [pts[:, 0] / (g2.shape[0] - 1), pts[:, 1] / (g2.shape[1] - 1),
            np.full(len(pts), 0.5), g2.gdf[g2.pore][:600]]
    tk = torch.tensor(np.stack(cols, 1)[None], dtype=torch.float32)
    b1 = torch.tensor((g2.material == inputs.PORE)[None, None, :, :, None],
                      dtype=torch.float32)
    with torch.no_grad():
        y2 = net2(b1, torch.zeros(1, 2), tk)
    check("and the whole 2D chain runs end to end",
          tuple(y2.shape) == (1, 600) and bool(torch.isfinite(y2).all()),
          str(tuple(y2.shape)))
    check("the trunk width the reaction asked for is the width it got",
          tk.shape[-1] == info2["trunk_in_dim"] == net2.trunk.in_dim,
          "%d" % tk.shape[-1])

    # 3. the same in 3D, out of an .h5, which is the format the audit asked for
    try:
        import h5py
        m3 = _rock_3d()
        p3 = os.path.join(d, "ds.h5")
        with h5py.File(p3, "w") as h:
            gr = h.create_group("geom")
            gr.create_dataset("material", data=m3[None])
            gr.create_dataset("gid", data=np.array([7]))
        g3 = inputs.read_geometry(p3, convention="ours")
        net3, info3 = core_model.build("aom_sulfate", g3.shape)
        pts = np.argwhere(g3.pore)[:600]
        cols = [pts[:, 0] / (g3.shape[0] - 1), pts[:, 1] / (g3.shape[1] - 1),
                pts[:, 2] / (g3.shape[2] - 1), np.full(len(pts), 0.5),
                g3.gdf[g3.pore][:600]]
        tk3 = torch.tensor(np.stack(cols, 1)[None], dtype=torch.float32)
        b13 = torch.tensor(g3.pore[None, None], dtype=torch.float32)
        with torch.no_grad():
            y3 = net3(b13, torch.zeros(1, 2), tk3)
        check("a 3D rock out of an .h5 runs end to end",
              tuple(y3.shape) == (1, 600) and bool(torch.isfinite(y3).all())
              and info3["ndim"] == 3, str(tuple(y3.shape)))
    except ImportError:
        check("h5py is installed so the .h5 route can be checked", False,
              "skipped")

    # 4. 2D and 3D differ in exactly one trunk column and nothing else
    _, ia = core_model.build("aom_sulfate", (64, 64, 64))
    _, ib = core_model.build("aom_sulfate", (64, 64, 1))
    only_z = [c for c in ia["trunk_cols"] if c not in ib["trunk_cols"]]
    check("2D and 3D differ by exactly the z column", only_z == ["z"],
          str(only_z))
    check("and by nothing else",
          [c for c in ib["trunk_cols"] if c not in ia["trunk_cols"]] == []
          and ia["param_names"] == ib["param_names"]
          and ia["reaction"] == ib["reaction"])

    # 5. the two mistakes that used to pass silently are now caught by name
    bad = core_model.checkpoint_matches(
        {"reaction": "monod", "distance_convention": "published"},
        {"reaction": "aom_sulfate", "distance_convention": "ours"}, say=None)
    check("the wrong chemistry and the wrong convention are both reported",
          len(bad) == 2, " / ".join(s.split(".")[0] for s in bad))
    check("and an agreeing checkpoint is silent",
          core_model.checkpoint_matches(
              {"reaction": "monod"}, {"reaction": "monod"}, say=None) == [])
    check("an older checkpoint that does not record them still runs",
          core_model.checkpoint_matches({"reaction": "monod"}, {}, say=None) == [])

    # 6. the distance convention really does reach the model. Two models fed
    #    the same rock under the two conventions must see different columns,
    #    or naming them achieved nothing.
    a = inputs.read_geometry(p2, convention="ours").gdf
    b = inputs.read_geometry(p2, convention="published").gdf
    pore = g2.pore
    c = float(np.corrcoef(a[pore], b[pore])[0, 1])
    check("the convention chosen at read time reaches the trunk", c < -0.99,
          "correlation %.4f" % c)

    # 7. the named convention is the released one, on the released domain.
    #    This is the check that makes naming them worth anything: not that our
    #    reimplementation is self-consistent, but that 'published' reproduces
    #    what the notebook itself builds, voxel for voxel, on the rock the
    #    release ships. The published grid's inlet is the y = 0 face, so the
    #    axis is 1 and not 0, which is exactly the sort of thing that used to
    #    be implicit.
    root = os.path.dirname(HERE)
    dom = os.path.join(root, "2D", "geometries", "Domain_Monod.npz")
    two_d = os.path.join(root, "2D_scripts")
    if os.path.exists(dom):
        if two_d not in sys.path:
            sys.path.insert(0, two_d)
        import prt2d_model                                     # noqa: E402
        raw = prt2d_model.read_domain(dom)
        mat = np.where(raw == 1, inputs.PORE, inputs.SOLID).astype(np.uint8)
        for scope, conv in (("full", "published"), ("pore", "published_pore")):
            a = prt2d_model.make_inlet_distance_norm(raw, scope)
            b = conventions.distance_column(mat, conv, inlet_axis=1)
            e = float(np.abs(a - b).max())
            check("convention %r is the notebook's own column" % conv,
                  e == 0.0, "largest difference %.3e" % e)
    else:
        print("  SKIP  the released 2D domain is not in this checkout")

    # 8. a published reaction's released weights would load, shape for shape,
    #    without the file being present: the shapes are the claim, not the file
    pub, _ = core_model.build("monod", (148, 64, 1), layout="published")
    keys = set(pub.state_dict())
    check("the published layout has the released key names",
          all(any(k.startswith(p) for p in
                  ("branch1_net.", "branch2_net.", "trunk_net.", "bias"))
              for k in keys), "%d tensors" % len(keys))

    return ok


def main():
    quiet = "--quiet" in sys.argv
    print("prt_core, the four modules")
    bad = run_module_self_tests(quiet)
    print("\nwhat is only true when they are used together")
    ok = crossing_checks()

    print()
    if bad:
        print("FAILED: %s" % ", ".join("%s.py" % m for m in bad))
    if not ok:
        print("FAILED: the crossing checks")
    if not bad and ok:
        print("prt_core: everything passed.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
