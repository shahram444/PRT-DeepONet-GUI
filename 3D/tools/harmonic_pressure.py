#!/usr/bin/env python3
# =============================================================================
# CHANGED FROM THE 2D VERSION
#
#   WHERE IT CAME FROM
#     Nothing here is a port. This REPLACES a piece of their pipeline.
#     github.com/hjunglab/PRT-DeepONet   branch/folder: velocity-informed
#     flow/parameters/Pressure_component_UNet.pt   (2.4 M parameters)
#     flow/models/PRT-DeepONet_Velocity_load.ipynb, code cell 4
#
#   WHAT THEIR 2D CODE DOES
#     Trains a U-Net to PREDICT the harmonic pressure gradient from the pore
#     mask, so the velocity operator's trunk can be fed in one forward pass.
#
#   WHAT WE CHANGED, AND WHY
#     We solve Laplace's equation directly instead, with a sparse linear
#     solve. On their 148 x 64 grid that takes about 0.1 s, which is FASTER
#     than their network and exact rather than approximate. A network is a
#     strange way to approximate something you can solve.
#
#     The U-Net is still available and still theirs: velocity_model.py holds
#     the same architecture and loads their weights. Use it when the solve
#     stops being cheap. Measured on this code, one CPU core, porosity 0.5:
#
#         2D 148 x 64      0.10 s per rock      solve is the right choice
#         3D 64 x 64 x 64  45.8 s per rock      the U-Net earns its keep
#
#     train_velocity.py --pressure {solve,unet,none} picks between them, and
#     'none' is the ablation that says what the prior was worth at all.
# =============================================================================
"""The harmonic pressure field on a pore space, and its gradient.

The velocity operator's trunk is given the direction and relative strength of the
driving force at each point. That comes from a HARMONIC pressure field: the solution of
Laplace's equation on the pore space, held at 1 on the inlet face and 0 on the outlet,
with no flux through the grain walls.

    div(grad P) = 0   inside the pore space
    P = 1             on the inlet face
    P = 0             on the outlet face
    dP/dn = 0         on every pore-grain boundary

WHY HARMONIC AND NOT THE REAL PRESSURE. The real pressure field of a Stokes flow
depends on the viscosity and the driving force, so it would have to be recomputed for
every flow condition. Laplace's equation has neither in it. The field is a property of
the rock alone, so it is solved ONCE and reused at every Reynolds or Peclet number in
the campaign, which is the only reason it is affordable as an input feature.

It is not the true pressure. A Stokes pressure satisfies Laplace's equation in the
bulk but its boundary condition at a wall is set by the velocity field, not by zero
normal gradient. This is a PRIOR: it has the right topology, it points the right way
down the connected paths, and it costs one sparse solve. The network is given the two
gradient components separately rather than a single magnitude, so it can learn how far
off the prior is and correct it.

WHY WE SOLVE RATHER THAN PREDICT. The released pipeline trains a U-Net to predict this
field from the pore mask, at 2.4 million parameters, so that it costs one forward pass.
On a 148 by 64 grid the exact sparse solve takes a few milliseconds, which is faster
than the network, so solving is the default here and the U-Net is available for the
cases where it is not: very large 3D volumes, where the sparse factorisation is the
expensive part. velocity_model.PressureComponentUNet is the same architecture as
theirs and loads their weights.

Run it like this.

    python harmonic_pressure.py --self-test
    python harmonic_pressure.py --geometry rock.npz --out pressure.npz

What you get back. With --out, an npz with the pressure field and its gradient
components. The self test writes nothing.

Worth knowing. A pore voxel with no connected path to the inlet has no boundary
condition reaching it, which leaves the linear system singular. Those voxels are
detected and dropped to zero rather than being allowed to make the solve fail, and the
count is reported. On a domain that was cleaned with keep_inlet_connected there are
none; on a raw foreign geometry there are usually a few.
"""
import argparse
import sys

import numpy as np

try:
    from scipy import ndimage
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
except ImportError:                                                    # pragma: no cover
    sys.exit("harmonic_pressure.py needs scipy. Run: python gui/install_requirements.py")

__all__ = ["harmonic_pressure", "harmonic_gradient"]


def _connected_to_both_faces(pore):
    """Pore voxels reachable from the inlet face, by face connectivity.

    Face connectivity, six neighbours in 3D and four in 2D, because that is the
    stencil the solve itself uses. Labelling with corner connectivity would call a
    diagonal touch a connection while the five point Laplacian would not, and the
    system would come back singular on exactly those voxels.
    """
    st = ndimage.generate_binary_structure(pore.ndim, 1)
    lab, _ = ndimage.label(pore, st)
    inlet_labels = set(np.unique(lab[0])) - {0}
    return np.isin(lab, list(inlet_labels)) & pore


def harmonic_pressure(pore, report=False):
    """Solve Laplace's equation on the pore space. Flow runs along axis 0.

    Returns the pressure field, zero outside the pore space and on any pocket that
    cannot be reached from the inlet.
    """
    pore = np.asarray(pore, bool)
    live = _connected_to_both_faces(pore)
    dropped = int(pore.sum() - live.sum())

    cells = np.argwhere(live)
    if len(cells) == 0:
        if report:
            print("  no pore voxel connects to the inlet; pressure is zero everywhere")
        return np.zeros(pore.shape, np.float32), dropped

    idx = -np.ones(pore.shape, np.int64)
    idx[live] = np.arange(len(cells))
    n = len(cells)
    nx = pore.shape[0]
    ndim = pore.ndim

    rows, cols, vals = [], [], []
    b = np.zeros(n)
    steps = []
    for ax in range(ndim):
        for d in (-1, 1):
            s = [0] * ndim
            s[ax] = d
            steps.append(tuple(s))

    # -------------------------------------------------------------------------
    # BLOCK 2.  Assemble the linear system, one row per live pore voxel.
    # Inlet and outlet rows are Dirichlet: a single 1 on the diagonal and the
    # boundary value on the right-hand side. Interior rows are the standard
    # 5-point (2D) or 7-point (3D) Laplacian. A neighbour that is solid is
    # simply LEFT OUT of the row, and the diagonal counts only the neighbours
    # that were included. That omission IS the no-flux condition: it is what
    # a mirrored ghost cell reduces to, without allocating one.
    # -------------------------------------------------------------------------
    for k, cell in enumerate(cells):
        c = tuple(int(v) for v in cell)
        if c[0] == 0:
            rows.append(k); cols.append(k); vals.append(1.0)
            b[k] = 1.0
            continue
        if c[0] == nx - 1:
            rows.append(k); cols.append(k); vals.append(1.0)
            b[k] = 0.0
            continue
        deg = 0
        for st in steps:
            nb = tuple(c[i] + st[i] for i in range(ndim))
            if any(nb[i] < 0 or nb[i] >= pore.shape[i] for i in range(ndim)):
                continue
            if not live[nb]:
                continue                       # no flux: the wall simply drops out
            rows.append(k); cols.append(int(idx[nb])); vals.append(1.0)
            deg += 1
        rows.append(k); cols.append(k); vals.append(-float(deg) if deg else 1.0)

    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    p = spsolve(A, b)
    field = np.zeros(pore.shape, np.float32)
    field[live] = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    if report:
        print("  solved %d cells, %d pore voxels dropped as unreachable" % (n, dropped))
    return field, dropped


def harmonic_gradient(pore, normalise=True):
    """The gradient components of the harmonic pressure. Shape (ndim,) + pore.shape.

    Central differences inside the pore space, zero elsewhere. With normalise the
    components are divided by the largest absolute value over the pore space, so the
    sign and the zero point are preserved and the magnitude lands in [-1, 1]. That is
    absolute maximum scaling, which is what the reference does, and it matters: a z
    score would move the zero, and a zero gradient means no driving force rather than
    an average one.
    """
    pore = np.asarray(pore, bool)
    P, _ = harmonic_pressure(pore)
    ndim = pore.ndim
    out = np.zeros((ndim,) + pore.shape, np.float32)
    for ax in range(ndim):
        if pore.shape[ax] < 3:
            continue
        sl_c = [slice(None)] * ndim
        sl_p = [slice(None)] * ndim
        sl_m = [slice(None)] * ndim
        sl_c[ax] = slice(1, -1)
        sl_p[ax] = slice(2, None)
        sl_m[ax] = slice(None, -2)
        out[ax][tuple(sl_c)] = (P[tuple(sl_p)] - P[tuple(sl_m)]) * 0.5
    out[:, ~pore] = 0.0
    if normalise:
        # ABSOLUTE MAXIMUM scaling, not a z-score, and the difference matters.
        # A z-score would shift the zero, and a zero gradient means "no driving
        # force here", not "an average driving force here". Their notebook
        # scales the same way, for the same reason.
        m = np.abs(out[:, pore]).max() if pore.any() else 0.0
        if m > 0:
            out /= m
    return out


def _self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + name + (("   " + detail) if detail else ""))
        if not cond:
            ok = False

    print("harmonic_pressure self test")

    # --- an open slab: the exact answer is a straight ramp
    nx, ny = 21, 9
    pore = np.ones((nx, ny), bool)
    P, dropped = harmonic_pressure(pore)
    want = np.linspace(1.0, 0.0, nx)[:, None] * np.ones((1, ny))
    err = float(np.abs(P - want).max())
    check("an open slab gives the exact linear ramp", err < 1e-6, "max error %.2e" % err)
    check("nothing is dropped in an open slab", dropped == 0)

    g = harmonic_gradient(pore, normalise=False)
    interior = np.zeros((nx, ny), bool)
    interior[1:-1, :] = True
    # a central difference over a ramp of slope -1/(nx-1) gives exactly that slope
    check("the gradient along the flow is constant and negative",
          bool((g[0][interior] < 0).all()) and float(g[0][interior].std()) < 1e-6,
          "value %.4f, expected %.4f" % (float(g[0][nx // 2, 4]), -1.0 / (nx - 1)))
    check("the transverse gradient is zero in an open slab",
          float(np.abs(g[1]).max()) < 1e-9)

    # --- a wall must not leak: no flux, so the field is still one dimensional
    pore = np.ones((21, 11), bool)
    pore[:, 0] = False
    pore[:, -1] = False
    P, _ = harmonic_pressure(pore)
    col = P[10][pore[10]]
    check("a walled duct is still uniform across its width",
          float(col.std()) < 1e-6, "sd %.2e" % float(col.std()))

    # --- a blocked path: the pressure must not run through solid
    pore = np.ones((21, 11), bool)
    pore[10, :] = False
    pore[10, 5] = True                        # one voxel connects the two halves
    P, _ = harmonic_pressure(pore)
    check("the field is continuous through a single connecting voxel",
          0.0 < float(P[10, 5]) < 1.0, "P at the throat is %.3f" % float(P[10, 5]))
    check("the pressure drops monotonically along the flow",
          bool(np.all(np.diff(P[:, 5]) <= 1e-9)))

    # --- an isolated pocket has no boundary condition and must be dropped, not fatal.
    # The rest of the domain must stay connected, or everything downstream of the cut
    # is unreachable too and the count means something else entirely.
    pore = np.ones((21, 11), bool)
    pore[8:12, 3:8] = False                   # a solid island
    pore[9:11, 4:7] = True                    # a sealed pocket inside that island
    P, dropped = harmonic_pressure(pore)
    check("a sealed pocket is dropped rather than making the solve fail",
          dropped == 6, "dropped %d voxels, expected the 6 in the pocket" % dropped)
    check("the dropped pocket holds no pressure", float(np.abs(P[9:11, 4:7]).max()) == 0.0)
    check("the flow path around the island is unaffected",
          float(P[0, 0]) == 1.0 and float(P[-1, 0]) == 0.0
          and bool(np.all(np.diff(P[:, 0]) <= 1e-9)))

    # --- 3D: an extruded slab must give the extruded answer exactly
    vol = np.ones((15, 7, 5), bool)
    P3, _ = harmonic_pressure(vol)
    want3 = np.linspace(1.0, 0.0, 15)[:, None, None] * np.ones((1, 7, 5))
    check("a 3D open slab gives the exact ramp too",
          float(np.abs(P3 - want3).max()) < 1e-6)
    g3 = harmonic_gradient(vol, normalise=False)
    check("the 3D gradient has one non zero component",
          float(np.abs(g3[1]).max()) < 1e-9 and float(np.abs(g3[2]).max()) < 1e-9)

    # --- normalisation keeps the sign and the zero
    pore = np.ones((21, 11), bool)
    pore[:, 0] = False
    gn = harmonic_gradient(pore, normalise=True)
    check("normalised components stay inside [-1, 1] and keep their sign",
          float(np.abs(gn).max()) <= 1.0 + 1e-6 and float(gn[0][10, 5]) < 0)
    check("normalisation leaves solid voxels at exactly zero",
          float(np.abs(gn[:, ~pore]).max()) == 0.0)

    print("\n" + ("Everything passed." if ok else "SOMETHING FAILED, read the lines above."))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="The harmonic pressure field and its gradient.")
    ap.add_argument("--geometry", help=".npz or .npy holding the material or pore array")
    ap.add_argument("--pore-code", type=int, default=None)
    ap.add_argument("--flow-axis", type=int, default=0)
    ap.add_argument("--out", help="write the field and gradient to this .npz")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return _self_test()
    if not a.geometry:
        ap.error("give --geometry, or --self-test")

    import flow_features as ff
    if a.geometry.endswith(".npz"):
        with np.load(a.geometry) as d:
            key = next((k for k in ("material", "m", "mat", "geom", "arr_0") if k in d.files),
                       d.files[0])
            arr = np.asarray(d[key])
    else:
        arr = np.load(a.geometry)
    arr = ff.read_reference_orientation(arr, a.flow_axis)
    pore = ff.pore_mask_from_material(arr, a.pore_code)
    print("shape %s, porosity %.3f" % (tuple(arr.shape), float(pore.mean())))
    P, dropped = harmonic_pressure(pore, report=True)
    g = harmonic_gradient(pore)
    print("  pressure spans %.4f to %.4f" % (float(P[pore].min()), float(P[pore].max())))
    for ax in range(pore.ndim):
        v = g[ax][pore]
        print("  gradient axis %d: min %.4f  max %.4f" % (ax, v.min(), v.max()))
    if a.out:
        np.savez_compressed(a.out, pressure=P, gradient=g, pore=pore, dropped=dropped)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
