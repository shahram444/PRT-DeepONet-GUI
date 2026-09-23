#!/usr/bin/env python3
"""
prt_core/conventions.py -- the distance column, and the two ways it is scaled.

WHAT CHANGED FROM THE 2D VERSION
    Both codebases give the trunk a geodesic distance from the inlet through
    the pore space, and they scale it in opposite directions. Measured on the
    published release's own Monod domain:

        published : 0.005 to 1.000 over the pore, HIGH at the inlet, solid 0
        ours      : 0.000 to 3.349 over the pore, ZERO at the inlet, solid 0
        correlation over the pore space: -1.0000

    Perfectly anti-correlated, and on different scales. Both are defensible.
    Neither is wrong. What is wrong is having two of them and no name for
    either, because then a warm start from the published weights hands a
    trained trunk a column that means the reverse of what it was fitted to,
    the shapes all match, nothing raises, and the prediction is merely poor.

    So the conventions are named here, a model records which one it was trained
    under, and converting between them is one function rather than an
    afterthought. That is the whole file.

THE THREE
    ours             geodesic distance from the inlet in voxels, divided by
                     nx - 1. Zero at the inlet, rising downstream, zero in the
                     solid. What build_dataset_2d.py and build_dataset_3d.py
                     store and what dataset_reader.py scales.
    published        the same walk, then set the solid one step past the
                     furthest pore, negate, and scale the WHOLE GRID onto 0
                     to 1. High at the inlet. What the Monod and reversible
                     sorption notebooks build.
    published_pore   the same, but scaled over the pore only. What the
                     irreversible sorption notebook builds, and the reason a
                     single implementation for all three would have been wrong.

WHY THE PUBLISHED TWO DIFFER AT ALL
    Once the solid has been set past the furthest pore and negated, it becomes
    the minimum of the whole grid. Scaling over the whole grid therefore lets
    the solid pin the bottom of the range, and the pore never quite reaches 0.
    Scaling over the pore only does not. On a rock with much solid the two
    differ by several per cent everywhere, which is plenty to matter to a
    trained trunk and not enough to notice by eye.

    python -m prt_core.conventions --self-test
"""

from collections import deque

import numpy as np

DISTANCE_CONVENTIONS = ("ours", "published", "published_pore")

# The material codes this project uses everywhere. Repeated here rather than
# imported so that this file has no dependency on the rest of the tree: it is
# the piece most likely to be copied into a notebook.
SOLID, WALL, PORE = 0, 1, 2


def pore_mask(material):
    """True where a chemical can be, from CompLaB codes or from 0/1."""
    m = np.asarray(material)
    if m.dtype == bool:
        return m
    u = set(np.unique(m).tolist())
    if u <= {0, 1}:                       # a plain mask: 1 is pore
        return m == 1
    return m == PORE


# ---------------------------------------------------------------- the walk
def geodesic_steps(pore, inlet_axis=0, seed_value=0.0):
    """Steps through the pore space from the inlet face, four or six connected.

    Not the straight-line distance: a chemical cannot travel through rock. The
    published notebooks seed the inlet at 0.5 and add 1 per step; ours seeds at
    0 and adds 1. seed_value carries that difference rather than hiding it.

    Solid voxels come back as not-a-number, so the caller decides what they
    become. That decision is exactly what separates the conventions.
    """
    pore = np.asarray(pore, bool)
    d = np.full(pore.shape, np.nan, np.float32)
    q = deque()

    face = [slice(None)] * pore.ndim
    face[inlet_axis] = 0
    idx = np.argwhere(pore[tuple(face)])
    for p in idx:
        full = list(p)
        full.insert(inlet_axis, 0)
        full = tuple(full)
        d[full] = seed_value
        q.append(full)

    offsets = []
    for ax in range(pore.ndim):
        for step in (1, -1):
            o = [0] * pore.ndim
            o[ax] = step
            offsets.append(tuple(o))

    while q:
        cur = q.popleft()
        base = d[cur] + 1.0
        for o in offsets:
            nxt = tuple(c + s for c, s in zip(cur, o))
            if any(v < 0 or v >= n for v, n in zip(nxt, pore.shape)):
                continue
            if not pore[nxt]:
                continue
            if np.isnan(d[nxt]) or d[nxt] > base:
                d[nxt] = base
                q.append(nxt)
    return d


# ------------------------------------------------------------- the conventions
def distance_column(material, convention="ours", inlet_axis=0):
    """The trunk's distance column, under one named convention.

    Always returns float32 the shape of the geometry, zero in the solid.
    """
    if convention not in DISTANCE_CONVENTIONS:
        raise ValueError(
            "no distance convention called %r. The three are: %s. They are not "
            "interchangeable: 'ours' is zero at the inlet and 'published' is "
            "one there, so using the wrong one gives a trained trunk a column "
            "that runs backwards." % (convention, ", ".join(DISTANCE_CONVENTIONS)))

    pore = pore_mask(material)
    if convention == "ours":
        d = geodesic_steps(pore, inlet_axis, seed_value=0.0)
        n = max(material.shape[inlet_axis] - 1, 1)
        out = np.nan_to_num(d, nan=0.0) / float(n)
        out[~pore] = 0.0
        return out.astype(np.float32)

    # both published variants: seed at 0.5, push the solid one past the
    # furthest pore, negate, then scale
    d = geodesic_steps(pore, inlet_axis, seed_value=0.5)
    reach = float(np.nanmax(d[pore])) if pore.any() else 0.0
    d = np.where(pore, d, reach + 1.0)
    d = -d

    out = np.zeros(d.shape, np.float32)
    if convention == "published":
        valid = np.isfinite(d)                       # the whole grid
    else:
        valid = pore & np.isfinite(d)                # the pore only
    if np.any(valid):
        lo, hi = float(d[valid].min()), float(d[valid].max())
        if hi > lo:
            out[valid] = (d[valid] - lo) / (hi - lo)
        else:
            out[valid] = 1.0 if convention == "published_pore" else 0.0
    out[~pore] = 0.0
    return out.astype(np.float32)


def convert(column, material, frm, to, inlet_axis=0):
    """Rebuild a distance column under a different convention.

    The conversion is not arithmetic on the column: the two conventions differ
    in what they scale over as well as in direction, so the honest conversion
    is to walk the geometry again. That is cheap, and it cannot be subtly
    wrong the way a remembered formula can.
    """
    if frm == to:
        return np.asarray(column, np.float32)
    return distance_column(material, to, inlet_axis)


def describe(convention):
    """One sentence, for a log line or an error message."""
    return {
        "ours": "zero at the inlet, rising downstream, in voxels over nx - 1",
        "published": "one at the inlet, falling downstream, scaled over the "
                     "whole grid",
        "published_pore": "one at the inlet, falling downstream, scaled over "
                          "the pore only",
    }[convention]


def default_for(source):
    """Which convention a model from this source was fitted under."""
    return "published" if source == "published" else "ours"


# =============================================================================
#  SELF TEST
# =============================================================================
def _self_test():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print("  %-5s %-58s %s" % ("PASS" if cond else "FAIL", name, extra))

    # a straight channel along the inlet axis, walls either side
    m = np.zeros((12, 7), np.uint8)
    m[:, 2:5] = PORE
    p = pore_mask(m)

    ours = distance_column(m, "ours")
    pub = distance_column(m, "published")
    pore_only = distance_column(m, "published_pore")

    check("ours is zero at the inlet", float(ours[0][p[0]].max()) == 0.0)
    check("and rises downstream", float(ours[-1][p[-1]].min()) > 0.5,
          "%.3f" % float(ours[-1][p[-1]].min()))
    check("published is one at the inlet",
          abs(float(pub[0][p[0]].max()) - 1.0) < 1e-6,
          "%.4f" % float(pub[0][p[0]].max()))
    check("and falls downstream", float(pub[-1][p[-1]].max()) < 0.5,
          "%.3f" % float(pub[-1][p[-1]].max()))

    c = float(np.corrcoef(ours[p], pub[p])[0, 1])
    check("the two run in opposite directions", c < -0.999, "correlation %.4f" % c)

    check("every convention is zero in the solid",
          all(float(np.abs(x[~p]).max()) == 0.0 for x in (ours, pub, pore_only)))

    # the two published variants differ, which is why both exist
    d = float(np.abs(pub[p] - pore_only[p]).max())
    check("the two published variants are not the same column", d > 1e-3,
          "max difference %.4f" % d)
    check("and the pore-only one reaches zero at the outlet",
          abs(float(pore_only[p].min())) < 1e-6,
          "%.2e" % float(pore_only[p].min()))

    # converting is a rebuild, and a round trip returns the original
    back = convert(convert(ours, m, "ours", "published"), m, "published", "ours")
    check("a round trip through the other convention returns the original",
          np.allclose(back, ours, atol=1e-6))

    # 3D works the same way
    m3 = np.zeros((10, 6, 6), np.uint8)
    m3[:, 2:4, 2:4] = PORE
    o3 = distance_column(m3, "ours")
    p3 = pore_mask(m3)
    check("3D: zero at the inlet and rising", float(o3[0][p3[0]].max()) == 0.0
          and float(o3[-1][p3[-1]].min()) > 0.5)

    # a dead end is reached, and a disconnected pocket is not
    m4 = np.zeros((10, 5), np.uint8)
    m4[:, 2] = PORE
    m4[4:7, 3] = PORE                      # a stub off the side
    m4[8, 0] = PORE                        # an island, unreachable
    o4 = distance_column(m4, "ours")
    check("a side branch is reached through the pore",
          float(o4[5, 3]) > 0.0)
    check("an unreachable pocket stays at zero rather than becoming a number",
          float(o4[8, 0]) == 0.0)

    check("an unknown convention is refused by name",
          "published" in _err(lambda: distance_column(m, "inlet-high")))

    print("\n%s" % ("Everything passed." if ok else "SOMETHING FAILED."))
    return 0 if ok else 1


def _err(fn):
    try:
        fn()
    except Exception as e:                                     # noqa: BLE001
        return str(e)
    return ""


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(_self_test())
    for c in DISTANCE_CONVENTIONS:
        print("  %-16s %s" % (c, describe(c)))
