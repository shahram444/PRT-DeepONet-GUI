#!/usr/bin/env python3
"""
flow_coordinates.py — FLOW-SPACE COORDINATES.  The support module for switch A
(--flow-proxy) and switch C (--dim-free).

WHAT THIS IS FOR
----------------
The 3D PRT-DeepONet currently tells the trunk where it is with the GEODESIC
DISTANCE FIELD: "you are N voxels from the inlet, measured through pore space".
That is a property of the GEOMETRY.

Christof's suggestion was to use the FLOW FIELD instead, so that the geometry
encoding problem can be handed to the state of the art (Geo-ONet, PNNL/EMSL)
and we spend our effort on the reactions.  The flow-space analogue of the
geodesic distance is not the speed |u|.  It is the ADVECTIVE TRAVEL TIME

        tau(x)  =  time for a fluid parcel to reach x from the inlet

obtained from the steady transport equation

        u . grad(tau)  =  1        with   tau = 0 on the inlet face.

WHY TRAVEL TIME AND NOT SPEED
-----------------------------
Along a streamline the transport-reaction equation collapses to

        dC/dtau  =  -R(C)      (plus transverse mixing)

which contains no geometry and no dimension at all.  In tau coordinates the
reaction front sits at a FIXED tau regardless of the pore structure or the
Peclet number, so the trunk is being handed a coordinate in which the answer is
nearly trivial.  That is exactly the regime where DeepONets normally struggle:
sharp advective fronts in Cartesian space.

It is also why tau makes 2D and 3D data interchangeable (switch C).  The trunk
(tau, d_wall, t) has THREE inputs in both 2D and 3D, whereas the Cartesian trunk
needs four in 2D and five in 3D and therefore cannot transfer at all.

THE ONE HONEST WEAKNESS — READ THIS BEFORE TRUSTING A RESULT
-------------------------------------------------------------
Where the fluid does not move, tau is undefined and the flow field carries NO
information about the pore space.  That is not a numerical detail, it is
physics: in a dead-end pore, or anywhere at Pe = 0.3, solute arrives by
DIFFUSION, and a velocity field of zero cannot distinguish "deep dead-end pore"
from "solid".  The geodesic distance still can.

So the honest prediction is that --flow-proxy will match or beat --distance gdf
at Pe 10 and 30 and LOSE at Pe 0.3.  The new scenario grid spans both, so this
comes out of the complab_campaign as a measurable curve rather than an opinion.  Use
--flow-mode both to give the trunk gdf AND tau and let it choose.

We keep tau finite everywhere by adding a small velocity floor, `u_floor`,
expressed as a fraction of the mean pore speed.  The default 0.01 means "no
parcel is treated as taking longer than about 100 mean transit times".  The
voxels where the floor is what determined tau are returned as `stagnant`, so
they can be masked, weighted down, or simply counted and reported.

USAGE
-----
    from flow_coordinates import travel_time, normalize_velocity, wall_distance

    tau, stagnant = travel_time(vel, material)        # vel (3,nx,ny,nz)
    un            = normalize_velocity(vel, material) # divided by mean pore speed
    dw            = wall_distance(material)           # == the 'edt' already in the h5

    python flow_coordinates.py --self-test        # runs on an analytic channel + a blob
"""

import numpy as np

SOLID, WALL, PORE = 0, 1, 2


# --------------------------------------------------------------------------
# =============================================================================
#  BLOCK 1.  THE SMALL PIECES
#
#  A pore mask, a mean speed and a normalised velocity. Every one of them is
#  taken over PORE VOXELS ONLY. Averaging over the whole grid would make the
#  same rock look slower at lower porosity, which is a property of the array
#  and not of the flow.
# =============================================================================
def pore_mask(material):
    return np.asarray(material) == PORE


def mean_pore_speed(vel, material):
    """Mean |u| over pore voxels.  Returns 0.0 if there is no flow at all."""
    m = pore_mask(material)
    if not m.any():
        return 0.0
    sp = np.sqrt((np.asarray(vel, np.float64) ** 2).sum(0))
    return float(sp[m].mean())


def normalize_velocity(vel, material, eps=1e-30):
    """u / <|u|>_pore, with zeros in the solid.

    Dividing by the mean pore speed is not cosmetic.  The branch CNN must see
    the SHAPE of the flow field, not its magnitude, because the magnitude is
    just the Peclet number, which the parameter branch already receives.  If we
    fed the raw field the network would get Pe twice — once as a scalar and once
    as the scale of the velocity — and could not disentangle "faster flow" from
    "different pore structure".
    """
    vel = np.asarray(vel, np.float32)
    m = pore_mask(material)
    s = mean_pore_speed(vel, material)
    out = vel / max(s, eps)
    out[:, ~m] = 0.0
    return out.astype(np.float32)


def wall_distance(material):
    """Euclidean distance from each pore voxel to the nearest non-pore voxel.

    This is the transverse coordinate for switch C.  It is identical to the
    'edt' field already stored in dataset.h5 by collect_complab_output.py, so on a real
    dataset we read it rather than recompute it; this function exists so the
    module is self-contained and testable.
    """
    from scipy import ndimage
    return ndimage.distance_transform_edt(pore_mask(material)).astype(np.float32)


# --------------------------------------------------------------------------
# =============================================================================
#  BLOCK 2.  THE TRAVEL TIME
#
#  Switch A's replacement for the geodesic distance: how long the FLOW takes to
#  carry solute from the inlet to each voxel, rather than how far it is. Solved
#  by the same neighbour relaxation the geodesic field uses, so the two are
#  comparable, with the edge cost set by the local speed instead of by 1.
#
#  u_floor is what makes it finite. A dead-end pocket has a speed of essentially
#  zero and would otherwise take infinite time to reach, so the speed is floored
#  at a fraction of the mean. That floor IS the model of diffusion here, which
#  is why switch A is expected to lose at low Peclet, where diffusion is the
#  thing actually delivering the solute.
# =============================================================================
def travel_time(vel, material, u_floor=0.01, max_iter=4000, tol=1e-5,
                inlet_axis=0, normalize=True, verbose=False):
    """Advective travel time from the inlet face, by upwind fast sweeping.

    Solves   sum_d  u_d dtau/dx_d  =  1,   tau = 0 on the inlet plane,
    with first-order upwinding.  Rearranged for the centre value that is being
    updated, the scheme is

        tau_c  =  ( 1 + sum_d a_d tau_upwind_d )  /  sum_d a_d

    where a_d = |u_d| in lattice units (dx = 1) and tau_upwind_d is the
    neighbour the flow in direction d is coming FROM.  Every coefficient is
    non-negative, so the update is a weighted average of upwind neighbours plus
    a positive source: it is monotone and cannot oscillate.

    Sweeping alternates direction each iteration, which is what makes this
    converge in O(nx) passes rather than O(nx^2) as plain Jacobi would.

    Parameters
    ----------
    vel       : (3, nx, ny, nz) velocity, any units
    material  : (nx, ny, nz) CompLaB codes
    u_floor   : velocity floor as a FRACTION of the mean pore speed.  Keeps tau
                finite in stagnant zones.  0.01 caps tau at roughly 100 mean
                transit times.  Set 0.0 to disable, at the cost of infinities.
    normalize : divide by the mean advection time L/<u>, so tau is O(1) and
                independent of the Peclet number.  Leave this on: it is what
                makes tau comparable across runs, across geometries and between
                2D and 3D.

    Returns
    -------
    tau      : (nx, ny, nz) float32, NaN in the solid
    stagnant : (nx, ny, nz) bool, True where the velocity floor is what set the
               value, i.e. where the flow field carries no real information
    """
    vel = np.asarray(vel, np.float32)
    mat = np.asarray(material)
    m = pore_mask(mat)
    nx, ny, nz = mat.shape

    if not m.any():
        return np.full(mat.shape, np.nan, np.float32), np.zeros(mat.shape, bool)

    ubar = mean_pore_speed(vel, mat)
    if ubar <= 0:                       # no flow at all: tau is meaningless
        return np.full(mat.shape, np.nan, np.float32), m.copy()

    floor = float(u_floor) * ubar
    a = np.abs(vel).astype(np.float32)
    speed = np.sqrt((vel ** 2).sum(0))
    stagnant = m & (speed < floor)

    # The floor is applied along the INLET axis only.  Adding it isotropically
    # would invent transverse transport that does not exist; adding it along the
    # flow direction is the statement "a stagnant parcel still eventually moves
    # downstream", which is what a small diffusivity would do.
    a[inlet_axis] = np.maximum(a[inlet_axis], np.where(stagnant, floor, 0.0))

    # sign masks: is the flow in direction d positive (coming from index-1)?
    pos = [(vel[d] >= 0) for d in range(3)]

    BIG = np.float32(1e30)

    # inlet plane: the first pore voxel layer at index 0 along inlet_axis
    inlet = np.zeros(mat.shape, bool)
    sl = [slice(None)] * 3
    sl[inlet_axis] = 0
    inlet[tuple(sl)] = True
    inlet &= m
    if not inlet.any():                 # inlet plane is all wall: use the first
        for i in range(mat.shape[inlet_axis]):   # plane that has any pore voxel
            sl[inlet_axis] = i
            cand = np.zeros(mat.shape, bool); cand[tuple(sl)] = True
            if (cand & m).any():
                inlet = cand & m
                break

    tau = np.full(mat.shape, BIG, np.float32)   # NaN restored at the end
    tau[inlet] = 0.0
    HALF = BIG * 0.5

    def plane_shift(p, axis, shift):
        """In-plane neighbour: out[j] = p[j - shift]; boundary = BIG."""
        out = np.roll(p, shift, axis=axis)
        if shift > 0:
            if axis == 0: out[0, :] = BIG
            else:         out[:, 0] = BIG
        else:
            if axis == 0: out[-1, :] = BIG
            else:         out[:, -1] = BIG
        return out

    def update_plane(i):
        """Godunov upwind update of one x-plane.

        A direction only contributes if its UPWIND neighbour is already known.
        Excluding unknown neighbours from BOTH numerator and denominator is what
        makes the front propagate one layer per pass instead of stalling at 1e30.
        """
        pm = m[i]
        if not pm.any() or inlet[i].all():
            return 0.0
        cur = tau[i]
        num = np.ones_like(cur)
        den = np.zeros_like(cur)
        # x direction: upwind plane is i-1 if u_x >= 0 else i+1
        prev_p = tau[i - 1] if i > 0 else np.full_like(cur, BIG)
        next_p = tau[i + 1] if i < nx - 1 else np.full_like(cur, BIG)
        srcs = [np.where(pos[0][i], prev_p, next_p),
                np.where(pos[1][i], plane_shift(cur, 0, 1), plane_shift(cur, 0, -1)),
                np.where(pos[2][i], plane_shift(cur, 1, 1), plane_shift(cur, 1, -1))]
        for d in range(3):
            ok_d = srcs[d] < HALF
            ad = np.where(ok_d, a[d][i], 0.0)
            num += ad * np.where(ok_d, srcs[d], 0.0)
            den += ad
        new = np.where(den > 1e-20, num / np.maximum(den, 1e-20), BIG)
        new = np.where(inlet[i], 0.0, new)
        new = np.where(pm, np.minimum(new, cur), BIG)     # monotone: only decrease
        d_max = float(np.abs(np.where((new < HALF) & (cur < HALF), new - cur, 0.0)).max())
        moved = float(((cur >= HALF) & (new < HALF)).sum())
        tau[i] = new
        return d_max + moved

    for it in range(int(max_iter)):
        ch = 0.0
        for i in range(nx):                 # forward sweep, with the flow
            ch += update_plane(i)
        for i in range(nx - 1, -1, -1):     # backward sweep, for recirculation
            ch += update_plane(i)
        fin = m & (tau < HALF)
        scale = float(tau[fin].max()) if fin.any() else 1.0
        if ch / max(scale, 1e-12) < tol and it >= 1:
            if verbose:
                print("converged in %d sweeps" % (it + 1))
            break

    tau = np.where(m, tau, np.nan).astype(np.float32)
    unreached = m & (tau > BIG * 0.5)
    if unreached.any():
        # voxels the flow never reaches even with the floor: fall back to the
        # largest finite value so downstream code never sees 1e30
        fin = tau[m & ~unreached]
        fill = float(np.nanmax(fin)) if fin.size else 0.0
        tau[unreached] = fill
        stagnant |= unreached

    if normalize:
        # L / <u>  in lattice units:  L = extent along the inlet axis
        t_ref = mat.shape[inlet_axis] / max(ubar, 1e-30)
        tau = (tau / t_ref).astype(np.float32)

    return tau, stagnant


# --------------------------------------------------------------------------
# =============================================================================
#  BLOCK 3.  MAKING IT AN INPUT
#
#  The raw travel time spans orders of magnitude between a channel and a pocket,
#  so it is squashed before the trunk ever sees it. A network fed the raw field
#  spends its capacity on the dynamic range instead of on the transport.
# =============================================================================
def squash(tau):
    """tau / (1 + tau) — the form actually fed to the trunk.

    Measured on a real D3Q19 Stokes field at porosity 0.45, the normalised
    travel time has median 0.76, 90th percentile 7.8 and a maximum of 186: a
    long tail from the deepest dead-end pores.  Feeding that raw would put the
    reaction front, which lives at tau ~ 1, into the bottom 0.5 percent of the
    input range and the network would never resolve it.

    tau/(1+tau) is monotone, smooth, maps 0 -> 0, 1 -> 0.5, infinity -> 1, and
    compresses the tail while KEEPING resolution where the physics is.  It is a
    reparameterisation, not a clip: no information is destroyed and the ordering
    is preserved exactly.
    """
    t = np.asarray(tau, np.float32)
    return (t / (1.0 + t)).astype(np.float32)


def stats(tau, material):
    """Percentiles of tau over the pore space, for logging and sanity checks."""
    m = pore_mask(material)
    t = np.asarray(tau)[m]
    t = t[np.isfinite(t)]
    if not t.size:
        return {}
    p = np.percentile(t, [50, 90, 99, 100])
    return dict(median=float(p[0]), p90=float(p[1]), p99=float(p[2]), max=float(p[3]))


def flow_fields(vel, material, u_floor=0.01, edt=None):
    """Everything switch A and switch C need, in one call.

    Returns dict with 'vel_norm' (3,nx,ny,nz), 'tau' (nx,ny,nz),
    'dwall' (nx,ny,nz), 'stagnant' (nx,ny,nz), 'stagnant_frac' float.
    """
    tau, stag = travel_time(vel, material, u_floor=u_floor)
    m = pore_mask(material)
    return dict(
        vel_norm=normalize_velocity(vel, material),
        tau=tau,
        dwall=(wall_distance(material) if edt is None else np.asarray(edt, np.float32)),
        stagnant=stag,
        stagnant_frac=float(stag[m].mean()) if m.any() else 0.0,
    )


# --------------------------------------------------------------------------
# =============================================================================
#  BLOCK 4.  THE SELF TEST
#
#  Run with --self-test. Built-in rather than in a separate file because these
#  are properties of the functions above, not of a pipeline, and a test that
#  ships inside the module cannot be left behind when the module moves.
# =============================================================================
def _self_test():
    """Two cases with a known answer, plus one realistic blob geometry."""
    ok = True

    # ---- 1. uniform plug flow in an open duct -----------------------------
    nx, ny, nz = 32, 8, 8
    mat = np.full((nx, ny, nz), PORE, np.uint8)
    mat[:, 0, :] = mat[:, -1, :] = mat[:, :, 0] = mat[:, :, -1] = WALL
    vel = np.zeros((3, nx, ny, nz), np.float32)
    vel[0][mat == PORE] = 1.0
    tau, stag = travel_time(vel, mat, normalize=False)
    # exact answer: tau = x  (unit speed, unit spacing)
    x = np.arange(nx, dtype=np.float32)[:, None, None] * np.ones((1, ny, nz), np.float32)
    m = mat == PORE
    err = np.abs(tau[m] - x[m]).max()
    print("1. plug flow      max|tau - x| = %.3e   stagnant %.1f%%"
          % (err, 100 * stag[m].mean()))
    ok &= err < 1e-3

    # ---- 2. same duct at half speed: tau doubles --------------------------
    tau2, _ = travel_time(vel * 0.5, mat, normalize=False)
    r = np.nanmax(tau2[m]) / max(np.nanmax(tau[m]), 1e-12)
    print("2. half speed     tau ratio    = %.4f   (expect 2.0000)" % r)
    ok &= abs(r - 2.0) < 1e-3

    # ---- 3. normalisation makes tau Pe-independent ------------------------
    a, _ = travel_time(vel, mat, normalize=True)
    b, _ = travel_time(vel * 7.3, mat, normalize=True)
    d = np.abs(a[m] - b[m]).max()
    print("3. normalised     max|tau(u) - tau(7.3u)| = %.3e   (expect ~0)" % d)
    ok &= d < 1e-4

    # ---- 4. a dead-end pocket is flagged stagnant, not infinite -----------
    mat2 = mat.copy()
    mat2[10:14, 1:4, 1:4] = PORE
    vel2 = np.zeros((3,) + mat2.shape, np.float32)
    vel2[0][mat2 == PORE] = 1.0
    vel2[0][10:14, 1:4, 1:4] = 0.0            # the pocket is stagnant
    tau3, stag3 = travel_time(vel2, mat2, normalize=False)
    m2 = mat2 == PORE
    print("4. dead end       finite everywhere = %s   stagnant %.1f%%"
          % (bool(np.isfinite(tau3[m2]).all()), 100 * stag3[m2].mean()))
    ok &= bool(np.isfinite(tau3[m2]).all())
    ok &= stag3[m2].mean() > 0

    # ---- 5. a real blob geometry, monotone increase downstream -----------
    from scipy import ndimage
    rng = np.random.default_rng(0)
    f = ndimage.gaussian_filter(rng.standard_normal((48, 24, 24)), 3.0, mode="wrap")
    g = np.where(f > np.quantile(f, 0.45), PORE, SOLID).astype(np.uint8)
    g[:3] = g[-3:] = PORE
    g[:, 0, :] = g[:, -1, :] = g[:, :, 0] = g[:, :, -1] = WALL
    v = np.zeros((3,) + g.shape, np.float32)
    v[0][g == PORE] = 1.0                      # crude but adequate for the test
    t5, s5 = travel_time(v, g, normalize=True)
    mm = g == PORE
    prof = [np.nanmean(t5[i][mm[i]]) for i in range(g.shape[0]) if mm[i].any()]
    mono = all(prof[i + 1] >= prof[i] - 1e-4 for i in range(len(prof) - 1))
    print("5. blob geometry  tau increases downstream = %s   range %.3f..%.3f"
          % (mono, np.nanmin(t5[mm]), np.nanmax(t5[mm])))
    ok &= mono

    # ---- 6. wall distance -------------------------------------------------
    dw = wall_distance(g)
    print("6. wall distance  range %.2f .. %.2f voxels" % (dw[mm].min(), dw[mm].max()))
    ok &= dw[mm].min() >= 1.0

    print("\nSELF TEST %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(_self_test())
    ap.print_help()
