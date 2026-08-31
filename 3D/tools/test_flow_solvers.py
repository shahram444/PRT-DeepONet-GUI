#!/usr/bin/env python3
"""
test_flow_solvers.py — check the flow solvers against answers that do not come from them.

    python test_flow_solvers.py

WHY
    Reading a lattice-Boltzmann implementation and agreeing with it proves
    nothing: the mistakes that matter (a wrong weight, a mis-paired opposite
    direction, a forcing term on the wrong axis, too few iterations) all
    produce a field that still looks like flow. The only useful test is an
    answer worked out independently.

    Flow between two flat plates driven by a body force has one. It is a
    parabola, and its peak is

        u_max = g L^2 / (8 nu)          nu = (tau - 1/2)/3

    where L is the distance between the walls and g the force per unit mass.
    With mid-link bounce-back the wall sits half a voxel outside the last
    fluid node, so a channel of H fluid nodes has L = H + 1. Getting that
    half-voxel wrong is itself one of the classic errors, and it shows up here
    as a few percent that will not go away however long the solver runs.

WHAT IS CHECKED
    1. the lattice itself: weights sum to 1, first moment zero, second moment
       (1/3) delta_ab, and every direction paired with its exact opposite
    2. Poiseuille flow in 2D against the parabola, profile and peak
    3. the same in 3D, where a slit reduces to the same analytic answer
    4. mass conservation
    5. no flow inside solid
    6. the Mach number, which has to stay small or the incompressible limit
       this all rests on does not hold
    7. whether the DEFAULT iteration counts are actually enough to converge
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from prtlb_2d import SOLID, WALL, PORE, stokes_d2q9           # noqa: E402
from prtlb_3d import stokes_d3q19                            # noqa: E402

CS2 = 1.0 / 3.0          # lattice speed of sound squared


# ---------------------------------------------------------------------------
def lattice_moments(e, w, name):
    """Weights and velocity set must satisfy three identities exactly."""
    out = []
    s = float(w.sum())
    out.append(("%s weights sum to 1" % name, abs(s - 1.0) < 1e-12,
                "sum = %.15f" % s))

    m1 = (w[:, None] * e).sum(0)
    out.append(("%s first moment is zero" % name,
                np.abs(m1).max() < 1e-12, "max |sum w e| = %.2e" % np.abs(m1).max()))

    nd = e.shape[1]
    m2 = np.einsum("i,ia,ib->ab", w, e.astype(float), e.astype(float))
    err = np.abs(m2 - CS2 * np.eye(nd)).max()
    out.append(("%s second moment is (1/3) delta_ab" % name, err < 1e-12,
                "max error %.2e" % err))
    return out


def opposite_table(e, opp, name):
    bad = [i for i in range(len(e)) if not np.array_equal(e[opp[i]], -e[i])]
    return [("%s opposite directions are exact" % name, not bad,
             "all %d paired" % len(e) if not bad else "wrong: %s" % bad)]


# ---------------------------------------------------------------------------
def parabola(H, force, tau):
    """The analytic profile, with the walls where the solver actually puts them.

    Halfway bounce-back places the no-slip surface at the mid-link, half a
    voxel outside the last fluid node. Fluid nodes sit at grid y = 1..H, so the
    walls are at y = 0.5 and y = H + 0.5 and the gap between them is H.

    Getting that half voxel wrong is one of the classic errors and it is worth
    stating in full, because it costs 9% at the edges of a 20-voxel channel
    while leaving the peak within 1% and the shape a perfect parabola. It looks
    like nothing.
    """
    nu = (tau - 0.5) / 3.0
    y = np.arange(1, H + 1) - 0.5        # distance of each fluid node from the wall
    return (force / (2 * nu)) * y * (H - y)


def wall_offset(ux):
    """Where the fitted parabola vanishes, relative to the last fluid node.

    0.5 means the mid-link, which is what Palabos and therefore CompLaB do.
    1.0 means the solid node centre, which is a valid wall in a different
    place and makes every channel one voxel wider.
    """
    H = len(ux)
    y = np.arange(1, H + 1) - 0.5
    c = np.polyfit(y, ux, 2)
    r = np.sort(np.roots(c))
    return float(((0.5 - r[0]) + (r[1] - (H - 0.5))) / 2), c


def poiseuille_2d(H=20, nit=40000, force=2e-6, tau=0.8):
    """A slit channel: solid rows top and bottom, everything else open."""
    nx = 12
    g = np.full((nx, H + 2), PORE, np.uint8)
    g[:, 0] = SOLID
    g[:, -1] = SOLID
    v = stokes_d2q9(g, nit=nit, force=force, tau_lb=tau)
    ux = v[0][nx // 2, 1:-1]                    # profile across the channel

    return ux, parabola(H, force, tau)


def poiseuille_3d(H=16, nit=40000, force=2e-6, tau=0.8):
    """The same slit in three dimensions: walls in y only, open in z.

    A slit is the 3D case with a known answer. A square duct has one too, but
    it is an infinite series, and a test whose reference needs its own
    convergence study is a poor test.
    """
    nx, nz = 8, 8
    g = np.full((nx, H + 2, nz), PORE, np.uint8)
    g[:, 0, :] = SOLID
    g[:, -1, :] = SOLID
    v = stokes_d3q19(g, nit=nit, force=force, tau_lb=tau)
    ux = v[0][nx // 2, 1:-1, nz // 2]

    return ux, parabola(H, force, tau)


def _duct3d(nx=16, H=14):
    """A duct walled in y and z, OPEN along x.

    Padding all six faces seals the inlet and outlet, and with periodic
    streaming a sealed duct barely flows at all. The convergence figure it
    produced was measuring a plug, not the solver.
    """
    g = np.full((nx, H + 2, H + 2), PORE, np.uint8)
    g[:, 0, :] = g[:, -1, :] = SOLID
    g[:, :, 0] = g[:, :, -1] = SOLID
    return g


# ---------------------------------------------------------------------------

def _block(path, name):
    """The text between the BEGIN and END markers of one shared block."""
    src = open(path, encoding="utf-8").read()
    a = src.index("=== PRTLB %s BLOCK BEGIN" % name)
    a = src.index("\n", a) + 1
    a = src.index("\n", a) + 1                      # past the closing rule
    b = src.index("=== PRTLB %s BLOCK END" % name)
    b = src.rindex("# ====", a, b)
    return src[a:b]


def check_blocks_identical(check):
    """The two simulator files must share their physics WORD FOR WORD.

    prtlb_2d.py and prtlb_3d.py each hold flow, transport and reactions, so
    each one is readable on its own -- which is what they are for. The price
    is two copies of everything that is not the lattice, and this is what
    stops that price being paid in bugs.

    It is not a hypothetical risk. There used to be two independent transport
    solvers in this project, and six bugs were found and fixed in one of them
    while every one survived in the other: a periodic boundary that leaked the
    inlet into the outlet, a diffusion coefficient wrong by a factor of nx, a
    hard step count that stopped before the front entered the sample, a clip
    that hid an instability, and a per-species term that made three of the
    four chemicals byte-identical copies. They survived because fixing one
    copy makes the other LOOK fixed.

    So: edit one, copy it to the other. This fails loudly until you do, and
    prints the first line that differs.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    f2 = os.path.join(here, "prtlb_2d.py")
    f3 = os.path.join(here, "prtlb_3d.py")
    for name in ("SHARED", "TRANSPORT"):
        a, b = _block(f2, name), _block(f3, name)
        if a == b:
            check("the %s block is identical in prtlb_2d and prtlb_3d" % name,
                  True, "%d bytes" % len(a))
            continue
        la, lb = a.splitlines(), b.splitlines()
        first = next((i for i in range(max(len(la), len(lb)))
                      if i >= len(la) or i >= len(lb) or la[i] != lb[i]), 0)
        check("the %s block is identical in prtlb_2d and prtlb_3d" % name,
              False,
              "they differ from line %d: 2d has %r, 3d has %r -- copy the "
              "block across" % (first + 1,
                                la[first][:50] if first < len(la) else "<end>",
                                lb[first][:50] if first < len(lb) else "<end>"))


def main():
    results = []

    def check(name, cond, note=""):
        results.append((name, bool(cond), note))

    print("=" * 78)
    print("FLOW SOLVER CHECK")
    print("=" * 78)

    # ---- 0. the two simulator files still agree with each other ----------
    check_blocks_identical(check)

    # ---- 1. the lattice --------------------------------------------------
    e2 = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                   [1, 1], [-1, 1], [-1, -1], [1, -1]], float)
    w2 = np.array([4/9.] + [1/9.]*4 + [1/36.]*4)
    opp2 = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    e3 = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                   [0, 0, 1], [0, 0, -1], [1, 1, 0], [-1, -1, 0], [1, -1, 0],
                   [-1, 1, 0], [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
                   [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]], float)
    w3 = np.array([1/3.] + [1/18.]*6 + [1/36.]*12)
    opp3 = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])

    for nm, cond, note in (lattice_moments(e2, w2, "D2Q9")
                           + opposite_table(e2, opp2, "D2Q9")
                           + lattice_moments(e3, w3, "D3Q19")
                           + opposite_table(e3, opp3, "D3Q19")):
        check(nm, cond, note)

    # ---- 2. Poiseuille, 2D ------------------------------------------------
    print("\n2D channel against the analytic parabola")
    ux, exact = poiseuille_2d()
    rel = np.abs(ux - exact).max() / exact.max()
    peak = ux.max() / exact.max()
    print("   peak: solver %.4e   exact %.4e   ratio %.4f" %
          (ux.max(), exact.max(), peak))
    print("   worst point on the profile: %.2f%% off" % (100 * rel))
    check("2D Poiseuille peak within 1%", abs(peak - 1) < 0.01,
          "ratio %.4f" % peak)
    check("2D Poiseuille profile within 1.5%", rel < 0.015,
          "worst %.2f%%" % (100 * rel))
    off, c = wall_offset(ux)
    nu_fit = -2e-6 / (2 * c[0])
    print("   viscosity from the curvature: %.5f  against (tau-1/2)/3 = %.5f"
          % (nu_fit, 0.1))
    print("   no-slip wall sits %.3f voxel beyond the last fluid node "
          "(mid-link = 0.5)" % off)
    check("2D viscosity within 1% of (tau-1/2)/3", abs(nu_fit / 0.1 - 1) < 0.01,
          "%.5f" % nu_fit)
    check("2D wall is at the mid-link, as in Palabos", abs(off - 0.5) < 0.1,
          "offset %.3f" % off)

    # a parabola and nothing else: fit one and look at what is left
    yh = np.arange(1, len(ux) + 1) - 0.5
    coef = np.polyfit(yh, ux, 2)
    resid = np.abs(ux - np.polyval(coef, yh)).max() / ux.max()
    print("   residual after fitting a parabola: %.3f%%" % (100 * resid))
    check("2D profile IS a parabola", resid < 0.01, "residual %.3f%%" % (100 * resid))

    # ---- 3. Poiseuille, 3D ------------------------------------------------
    print("\n3D slit against the same parabola")
    ux3, exact3 = poiseuille_3d()
    rel3 = np.abs(ux3 - exact3).max() / exact3.max()
    peak3 = ux3.max() / exact3.max()
    print("   peak: solver %.4e   exact %.4e   ratio %.4f" %
          (ux3.max(), exact3.max(), peak3))
    print("   worst point on the profile: %.2f%% off" % (100 * rel3))
    check("3D Poiseuille peak within 1.5%", abs(peak3 - 1) < 0.015,
          "ratio %.4f" % peak3)
    check("3D Poiseuille profile within 2%", rel3 < 0.02,
          "worst %.2f%%" % (100 * rel3))
    off3, c3 = wall_offset(ux3)
    print("   no-slip wall sits %.3f voxel beyond the last fluid node" % off3)
    check("3D wall is at the mid-link, as in Palabos", abs(off3 - 0.5) < 0.1,
          "offset %.3f" % off3)

    # ---- 4, 5, 6. conservation, solids, Mach ------------------------------
    print("\nconservation, solids and the incompressible limit")
    rng = np.random.default_rng(3)
    from build_dataset_2d import blob_2d, keep_spanning_cluster, percolates
    while True:
        gg = blob_2d((64, 32), 0.70, rng, 3.0)
        gg, _ = keep_spanning_cluster(gg)
        if percolates(gg):
            break
    v = stokes_d2q9(gg, nit=3000)
    solid = (gg != PORE)
    check("no flow inside solid (2D)", float(np.abs(v[:, solid]).max()) == 0.0)

    g3 = np.full((10, 10, 10), PORE, np.uint8)
    g3[:, 0, :] = g3[:, -1, :] = SOLID
    v3 = stokes_d3q19(g3, nit=2000)
    check("no flow inside solid (3D)", float(np.abs(v3[:, g3 != PORE]).max()) == 0.0)

    sp = np.sqrt((v[:2] ** 2).sum(0))
    ma = float(sp.max()) / np.sqrt(CS2)
    print("   Mach number of the porous 2D run: %.2e" % ma)
    check("Mach number stays below 0.05", ma < 0.05, "Ma = %.2e" % ma)

    # ---- THE WALL MUST NOT MOVE WHEN TAU DOES -------------------------------
    #
    # This is the check the whole two-relaxation-time collision exists for.
    # With plain BGK the bounce-back wall sits at the mid-link only when
    # (tau-1/2)^2 = 3/16, i.e. tau = 0.9330, and drifts everywhere else:
    # measured offsets were 0.437 at tau = 0.55 and 1.100 at tau = 2.0 on a
    # four-voxel channel, so a four-voxel throat carried 73% too much flow and
    # the permeability of a rock depended on a numerical parameter.
    print()
    print("does the wall stay at the mid-link as tau changes?")
    worst_tau = 0.0
    for tau in (0.55, 0.8, 0.9330127, 1.2, 2.0):
        for H in (4, 8, 20):
            ux, ex = poiseuille_2d(H=H, nit=20000, force=2e-6, tau=tau)
            err = float(np.abs(ux - ex).max() / ex.max())
            worst_tau = max(worst_tau, err)
            print("   tau=%-7.4f H=%-2d  profile error %.3f%%"
                  % (tau, H, 100 * err))
    check("the profile is exact for every tau from 0.55 to 2.0",
          worst_tau < 0.005, "worst %.3f%%" % (100 * worst_tau))

    # ---- THE TWO SOLVERS MUST AGREE ----------------------------------------
    # A 2D channel and a 3D channel of the same aperture are the same problem.
    # If they disagree, one of them is wrong and no single-solver test can say
    # which.
    print()
    print("do the two solvers agree on the same problem?")
    worst_dim = 0.0
    for H in (4, 8):
        u2 = float(poiseuille_2d(H=H, nit=20000, force=2e-6)[0].max())
        u3 = float(poiseuille_3d(H=H, nit=20000, force=2e-6)[0].max())
        worst_dim = max(worst_dim, abs(u2 - u3) / max(u2, 1e-30))
        print("   H=%-2d  2D %.6e   3D %.6e   ratio %.5f"
              % (H, u2, u3, u2 / u3))
    check("D2Q9 and D3Q19 agree to 0.1% on the same channel",
          worst_dim < 1e-3, "worst %.4f%%" % (100 * worst_dim))

    # ---- MASS. Advertised at the top of this file and never actually run. ---
    print()
    g = _duct3d()
    v = stokes_d3q19(g, nit=400, force=2e-6)
    pore = (g == PORE)
    flux = [float(v[0, i][pore[i]].sum()) for i in range(2, g.shape[0] - 2)]
    spread = (max(flux) - min(flux)) / max(abs(np.mean(flux)), 1e-30)
    check("the flux through every plane is the same to 1%", spread < 0.01,
          "spread %.3f%%" % (100 * spread))

    # ---- AND THE FIELD MUST BE USABLE --------------------------------------
    # A diverged flow used to be returned as not-a-number and the TRANSPORT
    # solver then reported "the integration diverged", blaming the wrong
    # solver entirely.
    check("the returned field is finite", bool(np.isfinite(v).all()))
    # A force large enough to leave the low-Mach regime has to be reported.
    # In double precision it no longer overflows -- which is the point of the
    # precision change -- so the guard that matters is the Mach warning, not a
    # crash. Silence here would mean a compressible flow field being handed to
    # a transport solver that assumes an incompressible one.
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        vfast = stokes_d3q19(g, nit=200, force=3e-3, tau_lb=0.8)
    said = "Mach" in buf.getvalue()
    ma_fast = float(np.sqrt((vfast ** 2).sum(0))[pore].max()) * np.sqrt(3.0)
    check("a force that leaves the low-Mach regime is reported", said,
          "Ma = %.3f, warned = %s" % (ma_fast, said))

    # ---- 7. are the DEFAULT iteration counts enough? ----------------------
    print("\nis the default iteration count enough to converge?")
    for label, solver, geom, default in (
            ("2D, 1200 its", stokes_d2q9, gg, 1200),
            ("3D, 400 its", stokes_d3q19, _duct3d(), 400)):
        a = solver(geom, nit=default)
        b = solver(geom, nit=default * 8)
        m = (geom == PORE)
        sa = np.sqrt((a[:2] ** 2).sum(0))[m].mean()
        sb = np.sqrt((b[:2] ** 2).sum(0))[m].mean()
        drift = abs(sa - sb) / max(sb, 1e-30)
        print("   %-14s mean speed %.4e -> %.4e at 8x the iterations "
              "(%.1f%% short)" % (label, sa, sb, 100 * drift))
        check("%s is within 5%% of converged" % label, drift < 0.05,
              "%.1f%% short" % (100 * drift))

    # ---------------------------------------------------------------------
    print()
    print("=" * 78)
    width = max(len(n) for n, _, _ in results)
    for n, ok, note in results:
        print("  %-*s %s %s" % (width, n, "PASS" if ok else "FAIL", note))
    bad = [n for n, ok, _ in results if not ok]
    print("=" * 78)
    print("%d checks, %d passed, %d FAILED" % (len(results),
                                               len(results) - len(bad), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
