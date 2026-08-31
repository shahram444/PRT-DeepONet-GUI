#!/usr/bin/env python3
# =============================================================================
# build_geometry_3d.py — 3D pore geometries in JUNG'S STYLE, for CompLaB3D +
#                       PRT-DeepONet-3D.
# =============================================================================
# GEOMETRY STYLE — measured directly from the PRT-DeepONet release, not guessed.
# -----------------------------------------------------------------------------
# I read geometries/Domain_Monod.npz and several Domains/domain_*.dat from the
# PRT-DeepONet repository.  What they actually are:
#
#   shape         (148, 64) -- x = 148 is the FLOW direction, y = 64.  The .dat
#                 files are 9472 values, one per line, C-order with y fastest.
#                 (Reshaping them as (64,148) produces horizontal streaks, which
#                 is how you can tell the ordering is wrong.)
#
#   codes         Domain_*.npz stores a flat uint8 array under the key 'm' with
#                 values {0,1,2}.  Measured:  the first and last x-slices are
#                 100 % code 1, i.e. pure-fluid inlet/outlet buffers.  So
#                       JUNG:  0 = SOLID, 1 = PORE, 2 = INTERFACE
#                 which is NOT CompLaB's convention (see below).  Code 2 is only
#                 6.7 % of the domain and forms 1-voxel outlines -> interface.
#
#   morphology    NOT sphere packs.  Organic, bicontinuous blobs -- the signature
#                 of a THRESHOLDED CORRELATED GAUSSIAN RANDOM FIELD.  Measured
#                 two-point correlation length 6-8 pixels in both directions
#                 (isotropic).  A Gaussian filter of sigma = 4.0 reproduces 6-7
#                 voxels in 3D; sigma is exposed as --sigma.
#
#   porosity      0.486, 0.495, 0.524, 0.609 across the four domains I sampled,
#                 so the 2000-domain training set spans roughly 0.45-0.65.
#
#   transverse    Jung's 2D medium fills the whole cross-section -- there are no
#                 solid walls at y = 0 or y = 63.
#
# -----------------------------------------------------------------------------
# TWO DELIBERATE DIFFERENCES FROM JUNG, BOTH FORCED BY CompLaB
# -----------------------------------------------------------------------------
# [A] CODES.  geometry.dat is written in CompLaB's convention, which the XML
#     declares as <pore>2</pore><solid>0</solid><bounce_back>1</bounce_back>:
#            HERE:  0 = solid, 1 = interface/bounce-back, 2 = pore
#            JUNG:  0 = solid, 1 = pore,                  2 = interface
#     Codes 1 and 2 are swapped relative to the PRT-DeepONet release.  The npz
#     written alongside carries BOTH: 'material' (CompLaB) and 'material_jung'
#     (Jung), so the ML side can use whichever it prefers.  Do not mix them --
#     an unknown code silently becomes reactive fluid (complab_functions.hh:243).
#
# [B] TRANSVERSE WALLS.  CompLaB has NO periodic boundary option anywhere
#     (grep -r periodic src/ returns nothing); the y and z faces are handled
#     ONLY by what the geometry marks as wall (complab_functions.hh:243-279 sets
#     up only west/east).  Pore voxels on the domain edge would stream from
#     unallocated space.  So a --wall voxel-thick bounce-back shell is added,
#     making this a walled column rather than Jung's unbounded medium.  Set
#     --wall 1 to minimise the perturbation; --wall 0 is refused.
#
# -----------------------------------------------------------------------------
# WHY NOT SPHERE PACKS (--style spheres is available but is NOT the default)
# -----------------------------------------------------------------------------
# Sphere packs give a different pore-size distribution and lower tortuosity than
# thresholded random fields at the same porosity.  Since the point of the paper
# is that the GEODESIC encoding beats a Euclidean one, and that gap grows with
# tortuosity, matching Jung's morphology matters.  Use --style spheres only if
# you want a second, independent geometry family for a generalisation test.
#
# -----------------------------------------------------------------------------
# BUGS FIXED FROM THE ORIGINAL make_geometry.py (all verified against the source)
# -----------------------------------------------------------------------------
# [1] add_interface() was DEAD CODE: build() only ever wrote WALL(1) and PORE(2),
#     never SOLID(0), so `g == SOLID` was always empty and no interface layer was
#     produced.  Harmless for an empty duct whose walls were already code 1, but
#     with grains written as code 0 and no code-1 layer,
#     calculateDistanceFromSolid (complab_functions.hh:160) runs `while(lp==1)`
#     with NO upper bound on the search radius and terminates only on a code-1
#     neighbour -> CompLaB HANGS AT STARTUP, forever, with no error message.
# [2] `g[:2,:,:] = PORE` opened the transverse walls at the inlet/outlet planes,
#     putting fluid on the domain edge with nothing to bounce off.
# [3] Codes 5 and 6 (MT, SRB) have no <material_numbers> entry, so per
#     complab_functions.hh:243-274 they fell through a catch-all and silently
#     became ordinary REACTIVE FLUID rather than biofilm.
# [4] add_interface() used np.roll, which wraps periodically.
# [5] Domain was 64x48x20, not cubic.
# [6] One geometry, one hard-coded seed.
# [7] It was an EMPTY DUCT.  In an empty duct the geodesic distance from the
#     inlet is exactly the x-coordinate -- identical to Euclidean -- so the
#     paper's central claim would have been a guaranteed null result.
# [8] Biofilm painted as material codes.  For PLANKTONIC biomass that is backwards
#     twice over: biomass must be a mobile ADE field, and the presence of a
#     <microbeN> tag in <material_numbers> is precisely what flips a microbe from
#     planktonic to biofilm (complab_functions.hh:965-978).
#
# KEPT, because it was right: the write order.  x outer, y middle, z inner
# (z fastest) is a C-order ravel of (nx,ny,nz) and matches readGeometry at
# complab_functions.hh:118.  --verify round-trips the file and asserts it.
#
# usage
#   python build_geometry_3d.py --n 140 --out ./geometries --render --render3d
#   python build_geometry_3d.py --n 140 --out ./geometries --nx 148   # Jung aspect
# =============================================================================

import argparse, csv, json, os, sys
import numpy as np
from scipy import ndimage

SOLID, WALL, PORE = 0, 1, 2          # CompLaB <material_numbers>


# ---------------------------------------------------------------- geometry ---
def blob_field(shape, sigma, rng):
    """Correlated Gaussian random field — the generator behind Jung's morphology.
    Smoothing white noise with a Gaussian of width sigma gives a field whose
    two-point correlation length is ~sqrt(2)*sigma; thresholding it produces the
    organic bicontinuous blobs seen in Domain_*.npz.  mode='wrap' keeps the
    statistics stationary right up to the domain edge."""
    return ndimage.gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")


def build_blobs(nx, ny, nz, target_phi, rng, sigma=4.0,
                wall=2, inlet_buf=5, outlet_buf=5):
    f = blob_field((nx, ny, nz), sigma, rng)
    interior = np.zeros((nx, ny, nz), bool)
    interior[inlet_buf:nx - outlet_buf, wall:ny - wall, wall:nz - wall] = True
    # threshold chosen on the interior only, so the buffers do not bias porosity
    thr = np.quantile(f[interior], 1.0 - target_phi)
    solid = (f < thr) & interior          # pore where f >= thr
    solid[:, :wall, :] = True; solid[:, ny - wall:, :] = True
    solid[:, :, :wall] = True; solid[:, :, nz - wall:] = True
    return solid


def build_spheres(nx, ny, nz, target_phi, rng, r_min=3.0, r_max=7.0,
                  wall=2, inlet_buf=5, outlet_buf=5, max_tries=20000):
    """Alternative morphology: random polydisperse overlapping spheres.  Kept as
    a second, independent geometry family for a generalisation test."""
    solid = np.zeros((nx, ny, nz), bool)
    solid[:, :wall, :] = True; solid[:, ny - wall:, :] = True
    solid[:, :, :wall] = True; solid[:, :, nz - wall:] = True
    interior = np.zeros((nx, ny, nz), bool)
    interior[inlet_buf:nx - outlet_buf, wall:ny - wall, wall:nz - wall] = True
    n_int = int(interior.sum())
    xg, yg, zg = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    phi = lambda: 1.0 - (solid & interior).sum() / n_int
    t = 0
    while phi() > target_phi and t < max_tries:
        t += 1
        r = rng.uniform(r_min, r_max)
        c = (rng.uniform(inlet_buf, nx - outlet_buf), rng.uniform(0, ny), rng.uniform(0, nz))
        blob = ((xg - c[0])**2 + (yg - c[1])**2 + (zg - c[2])**2) <= r * r
        cand = solid | (blob & interior)
        if _percolates(~cand, inlet_buf, nx - outlet_buf - 1):
            solid = cand
    return solid


def build_fibers(nx, ny, nz, target_phi, rng, length=None, thickness=None,
                 wall=2, inlet_buf=5, outlet_buf=5, max_tries=20000):
    """Fibrous medium: straight cylindrical fibres laid down in random
    directions until the target porosity is reached.

    The third morphology family, matching the fibrous medium of Asgari and
    Meile: it stands for filter media, biological tissue and engineered mats,
    and it is the one whose pore space is strongly channelled along the fibre
    directions rather than bicontinuous (blobs) or granular (spheres). Fibre
    length defaults to 1.5 times the shortest transverse dimension and the
    radius thickens as the medium closes, so a low
    porosity is reached with fewer, fatter fibres rather than a felt of
    hair-thin ones that no grid could resolve.
    """
    if length is None:
        length = max(6.0, 0.5 * min(ny, nz) * 3.0)
    if thickness is None:                       # fatter fibres at low porosity
        thickness = float(np.interp(target_phi, [0.4, 0.6, 0.8], [2.0, 1.5, 1.0]))
    solid = np.zeros((nx, ny, nz), bool)
    solid[:, :wall, :] = True; solid[:, ny - wall:, :] = True
    solid[:, :, :wall] = True; solid[:, :, nz - wall:] = True
    interior = np.zeros((nx, ny, nz), bool)
    interior[inlet_buf:nx - outlet_buf, wall:ny - wall, wall:nz - wall] = True
    n_int = int(interior.sum())
    xg, yg, zg = np.meshgrid(np.arange(nx, dtype=np.float32),
                             np.arange(ny, dtype=np.float32),
                             np.arange(nz, dtype=np.float32), indexing="ij")
    phi = lambda: 1.0 - (solid & interior).sum() / n_int
    t = 0
    while phi() > target_phi and t < max_tries:
        t += 1
        # a random start point and a random direction on the unit sphere
        p0 = np.array([rng.uniform(inlet_buf, nx - outlet_buf),
                       rng.uniform(0, ny), rng.uniform(0, nz)], np.float32)
        d = rng.standard_normal(3).astype(np.float32)
        d /= max(np.linalg.norm(d), 1e-9)
        # distance from every voxel centre to the fibre axis, clamped to the
        # segment so the fibre has ends rather than running on for ever
        rx, ry, rz = xg - p0[0], yg - p0[1], zg - p0[2]
        s = np.clip(rx * d[0] + ry * d[1] + rz * d[2], 0.0, length)
        dist2 = (rx - s * d[0]) ** 2 + (ry - s * d[1]) ** 2 + (rz - s * d[2]) ** 2
        fibre = dist2 <= thickness * thickness
        cand = solid | (fibre & interior)
        if _percolates(~cand, inlet_buf, nx - outlet_buf - 1):
            solid = cand
    return solid


def _percolates(pore, x_in, x_out):
    lab, n = ndimage.label(pore)
    if n == 0:
        return False
    return bool((set(np.unique(lab[x_in])) - {0}) & (set(np.unique(lab[x_out])) - {0}))


def keep_percolating_cluster(solid, x_in, x_out):
    """Isolated pore pockets become solid: they would give the LBM dead fluid and
    leave the geodesic field infinite there.  Connected DEAD ENDS are kept — they
    are where planktonic biomass accumulates, which is exactly the physics the
    geodesic encoding exists to see."""
    pore = ~solid
    lab, n = ndimage.label(pore)
    if n == 0:
        return solid, -1
    good = (set(np.unique(lab[x_in])) - {0}) & (set(np.unique(lab[x_out])) - {0})
    if not good:
        return solid, -1
    keep = np.isin(lab, list(good))
    return ~keep, int((pore & ~keep).sum())


def label_materials(solid):
    """solid interior -> 0, solid touching pore -> 1 (bounce-back), pore -> 2.
    Padded dilation, NOT np.roll, so nothing wraps around the domain edge."""
    pore = ~solid
    pad = np.pad(pore, 1, mode="constant", constant_values=False)
    touch = np.zeros_like(pore)
    for ax in range(3):
        for s in (-1, 1):
            sl = [slice(1, -1)] * 3
            sl[ax] = slice(1 + s, pad.shape[ax] - 1 + s)
            touch |= pad[tuple(sl)]
    g = np.full(solid.shape, SOLID, np.uint8)
    g[solid & touch] = WALL
    g[pore] = PORE
    return g


def to_jung_codes(g):
    """CompLaB (0 solid, 1 interface, 2 pore) -> Jung (0 solid, 1 pore, 2 interface)."""
    j = np.zeros_like(g)
    j[g == PORE] = 1
    j[g == WALL] = 2
    return j


def no_bare_solid_touching_pore(g):
    pore = (g == PORE)
    pad = np.pad(pore, 1, mode="constant", constant_values=False)
    touch = np.zeros_like(pore)
    for ax in range(3):
        for s in (-1, 1):
            sl = [slice(1, -1)] * 3
            sl[ax] = slice(1 + s, pad.shape[ax] - 1 + s)
            touch |= pad[tuple(sl)]
    return not ((g == SOLID) & touch).any()


# -------------------------------------------------------------------- GDF ---
def geodesic_from_inlet(g, x_src=0, method="relax"):
    """Tortuous distance from the inlet plane THROUGH the pore space, by fast
    marching on the masked eikonal equation.  This is the trunk's `dist_inlet`.
    x_src = 0 because readGeometry duplicates the first file slice into the ghost
    inlet plane and Palabos applies the pressure boundary on the x=0 face."""
    pore = (g == PORE)
    if not pore[x_src].any():
        return None

    # WHY THE DEFAULT IS THE SLOWER, CRUDER SOLVER.
    #
    # Fast marching solves the eikonal equation and gives a smooth, sub-voxel
    # distance. Neighbour relaxation only steps along axes, so its paths run
    # systematically LONGER -- measured on three 32^3 structures, the mean
    # distance is 1.17, 1.18 and 1.19 times the fast-marching one.
    #
    # That ratio is the problem. Every dataset in this project computes the
    # geodesic by relaxation, so a model's trunk learns THAT scale. Generating
    # a prediction geometry with fast marching would hand it a distance field
    # a fifth smaller than anything it was trained on -- silently, with no
    # error and no warning, just a quietly worse prediction. Consistency
    # between training and prediction beats smoothness.
    #
    # Pass method="fast-marching" deliberately if you are building geometries
    # for something that does not go through this project's trunk.
    if method != "fast-marching":
        from build_practice_dataset import geodesic
        d = geodesic(g)
        return np.where(pore, d, np.nan).astype(np.float32)

    try:
        import skfmm
    except ImportError:
        # scikit-fmm solves the eikonal equation properly and gives a smooth,
        # sub-voxel distance. It is worth having, but it is one more thing to
        # install and it was a HARD requirement of this file alone -- so a
        # missing wheel stopped the geometry generator dead with
        # "ModuleNotFoundError: No module named 'skfmm'" and no suggestion of
        # what to do about it.
        #
        # The six-neighbour relaxation below is the same solver build_practice_dataset
        # and build_dataset_2d already use, and it is what every dataset in this
        # project was in fact built with. Its distances are quantised to whole
        # voxel steps, so they run a few percent long on diagonal paths; for
        # the trunk input that is a scale factor, not a change of shape.
        print("     note: scikit-fmm is not installed, so the geodesic field is "
              "computed by neighbour relaxation instead of fast marching. Same "
              "field, quantised to whole voxel steps. Install scikit-fmm for "
              "the smoother version.", flush=True)
        from build_practice_dataset import geodesic
        d = geodesic(g)
        return np.where(pore, d, np.nan).astype(np.float32)

    phi = np.ones(g.shape, np.float64)
    phi[x_src][pore[x_src]] = 0.0
    d = skfmm.distance(np.ma.MaskedArray(phi, mask=~pore))
    return np.array(d.filled(np.nan), np.float32)


def euclidean_to_solid(g):
    """Straight-line distance to the nearest solid.  Not a model input — it is
    the control for the geodesic-vs-Euclidean ablation."""
    return ndimage.distance_transform_edt(g == PORE).astype(np.float32)


def correlation_length(binary, axis=0):
    a = binary.astype(float); a -= a.mean()
    F = np.fft.fft(a, axis=axis)
    C = np.real(np.fft.ifft(F * np.conj(F), axis=axis))
    C = C.mean(axis=tuple(i for i in range(C.ndim) if i != axis))
    if C[0] == 0:
        return -1
    C /= C[0]
    for k, v in enumerate(C[:len(C) // 2]):
        if v < np.exp(-1):
            return k
    return -1


# ------------------------------------------------------------------- I/O ---
def write_dat(g, path):
    """C-order ravel of (nx,ny,nz): x outer, y middle, z inner (z fastest),
    matching Palabos IndexOrdering::forward as read by readGeometry
    (complab_functions.hh:118).  ONE INTEGER PER LINE — the format the user's
    working geometry files use.  Palabos reads via plb_ifstream's operator>>,
    which treats newlines and spaces identically, so rows-per-line would also
    parse, but there is no reason to differ from a format known to work."""
    with open(path, "w") as f:
        f.write("\n".join(map(str, g.reshape(-1).tolist())))
        f.write("\n")


def read_dat(path, shape):
    return np.loadtxt(path, dtype=np.uint8).reshape(shape)


def write_dims(g, path):
    with open(path, "w") as f:
        f.write("%d %d %d\n" % g.shape)


# ------------------------------------------------------------------ render ---
def render_slices(g, gdf, meta, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    nx, ny, nz = g.shape
    cmap = ListedColormap(["#8a8f98", "#3b3f46", "#dbe4ef"])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], 3)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.4))
    ax[0, 0].imshow(g[:, :, nz // 2].T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    ax[0, 0].set_title("material, x-y slice at z=%d   (grey solid, dark interface, pale pore)" % (nz // 2))
    ax[0, 1].imshow(g[:, ny // 2, :].T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    ax[0, 1].set_title("material, x-z slice at y=%d" % (ny // 2))
    im = ax[1, 0].imshow(gdf[:, :, nz // 2].T, origin="lower", cmap="viridis", interpolation="nearest")
    ax[1, 0].set_title("geodesic distance from inlet"); fig.colorbar(im, ax=ax[1, 0], fraction=.045)
    xc = np.broadcast_to(np.arange(nx, dtype=np.float32)[:, None, None], g.shape)
    ex = gdf - xc
    im2 = ax[1, 1].imshow(ex[:, :, nz // 2].T, origin="lower", cmap="magma", interpolation="nearest")
    ax[1, 1].set_title("tortuosity excess: geodesic minus straight-line x")
    fig.colorbar(im2, ax=ax[1, 1], fraction=.045)
    for a in ax.ravel():
        a.set_xlabel("x (flow +)")
    fig.suptitle("geom %04d  |  %dx%dx%d  |  porosity %.3f  |  corr length %d vox  |  "
                 "max geodesic excess %.0f vox"
                 % (meta["gid"], nx, ny, nz, meta["phi_interior"],
                    meta["corr_len_x"], meta["gdf_excess_max"]), fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=115, bbox_inches="tight"); plt.close(fig)


def _surf(ax, mask, color, alpha):
    if mask.sum() == 0:
        return
    from skimage import measure
    m = np.pad(mask.astype(np.float32), 1)
    try:
        v, f, _, _ = measure.marching_cubes(m, 0.5)
    except Exception:
        return
    v -= 1.0
    ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2], color=color, alpha=alpha,
                    linewidth=0, antialiased=False, shade=True)


def render_3d(g, gdf, meta, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    nx, ny, nz = g.shape
    w = int(meta.get("wall", 2))
    shell = np.zeros(g.shape, bool)
    shell[:, :w, :] = True; shell[:, ny - w:, :] = True
    shell[:, :, :w] = True; shell[:, :, nz - w:] = True
    grains = ((g == SOLID) | (g == WALL)) & ~shell
    pore = (g == PORE)
    half = np.zeros(g.shape, bool); half[:, ny // 2:, :] = True

    fig = plt.figure(figsize=(16.5, 5.6))
    ax = fig.add_subplot(131, projection="3d")
    _surf(ax, grains, "#8a8f98", 0.95)
    ax.set_title("grain structure (duct walls removed)")

    ax2 = fig.add_subplot(132, projection="3d")
    _surf(ax2, grains & ~half, "#c8ccd2", 0.16)
    _surf(ax2, pore & ~half, "#2f7fd6", 0.95)
    ax2.set_title("open pore space, half cut away")

    ax3 = fig.add_subplot(133, projection="3d")
    _surf(ax3, grains, "#c8ccd2", 0.12)
    slab = np.zeros(g.shape, bool); slab[:, ny // 2 - 3:ny // 2 + 3, :] = True
    gg = np.where(pore & slab, gdf, np.nan)
    if np.isfinite(gg).any():
        p = np.argwhere(np.isfinite(gg)); p = p[:: max(1, len(p) // 12000)]
        sc = ax3.scatter(p[:, 0], p[:, 1], p[:, 2], c=gg[p[:, 0], p[:, 1], p[:, 2]],
                         cmap="viridis", s=4, alpha=0.85, linewidths=0)
        fig.colorbar(sc, ax=ax3, fraction=0.03, pad=0.02, label="geodesic distance")
    ax3.set_title("geodesic field on a mid-plane slab")

    for a in (ax, ax2, ax3):
        a.set_xlim(0, nx); a.set_ylim(0, ny); a.set_zlim(0, nz)
        try: a.set_box_aspect((nx / max(nx, ny, nz), ny / max(nx, ny, nz), nz / max(nx, ny, nz)))
        except Exception: pass
        a.set_xlabel("x (flow +)"); a.set_ylabel("y"); a.set_zlabel("z")
        a.view_init(elev=18, azim=-62)
    fig.suptitle("geom %04d  |  %dx%dx%d  |  porosity %.3f  |  tortuosity %.2f  |  "
                 "max geodesic excess %.0f vox"
                 % (meta["gid"], nx, ny, nz, meta["phi_interior"],
                    meta["tortuosity"], meta["gdf_excess_max"]), fontsize=12)
    fig.legend(handles=[Patch(facecolor="#8a8f98", label="solid grains"),
                        Patch(facecolor="#2f7fd6", label="open pore space")],
               loc="lower center", ncol=2, frameon=False)
    fig.savefig(path, dpi=118, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ driver ---
def build_one(gid, args, target_phi, rng):
    nx, ny, nz = args.nx, args.ny, args.nz
    for attempt in range(15):
        sigma = rng.uniform(*args.sigma_range) if args.style == "blobs" else 0.0
        if args.style == "blobs":
            solid = build_blobs(nx, ny, nz, target_phi, rng, sigma,
                                args.wall, args.inlet_buf, args.outlet_buf)
        elif args.style == "fibers":
            solid = build_fibers(nx, ny, nz, target_phi, rng, args.fiber_length,
                                 args.fiber_thickness, args.wall, args.inlet_buf,
                                 args.outlet_buf)
        else:
            solid = build_spheres(nx, ny, nz, target_phi, rng, args.r_min, args.r_max,
                                  args.wall, args.inlet_buf, args.outlet_buf)
        solid, removed = keep_percolating_cluster(solid, args.inlet_buf,
                                                  nx - args.outlet_buf - 1)
        if removed < 0:
            continue
        g = label_materials(solid)
        gdf = geodesic_from_inlet(g, 0, method=args.geodesic)
        if gdf is None or np.isnan(gdf[g == PORE]).any():
            continue
        assert no_bare_solid_touching_pore(g), "bare code-0 solid touching pore"

        pore = (g == PORE)
        interior = np.zeros(g.shape, bool)
        interior[args.inlet_buf:nx - args.outlet_buf,
                 args.wall:ny - args.wall, args.wall:nz - args.wall] = True
        xc = np.broadcast_to(np.arange(nx, dtype=np.float32)[:, None, None], g.shape)
        excess = (gdf - xc)[pore]

        meta = dict(
            gid=gid, nx=nx, ny=ny, nz=nz, style=args.style, sigma=float(sigma),
            spawn_key=list(map(int, rng.bit_generator.seed_seq.spawn_key)),
            target_phi=float(target_phi),
            phi_interior=float(pore[interior].mean()),
            phi_total=float(pore.mean()),
            n_solid=int((g == SOLID).sum()), n_wall=int((g == WALL).sum()),
            n_pore=int(pore.sum()), isolated_removed=int(removed),
            corr_len_x=int(correlation_length(pore[interior.any(2).any(1)], 0)),
            corr_len_y=int(correlation_length(pore, 1)),
            corr_len_z=int(correlation_length(pore, 2)),
            gdf_max=float(np.nanmax(gdf)), tortuosity=float(np.nanmax(gdf) / (nx - 1)),
            gdf_excess_mean=float(excess.mean()), gdf_excess_max=float(excess.max()),
            attempts=attempt + 1, wall=args.wall,
            inlet_buf=args.inlet_buf, outlet_buf=args.outlet_buf,
            codes_complab="0=solid 1=interface/bounce_back 2=pore",
            codes_jung="0=solid 1=pore 2=interface",
        )
        return g, gdf, euclidean_to_solid(g), meta
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=140)
    p.add_argument("--out", default="./geometries")
    p.add_argument("--style", choices=["blobs", "spheres", "fibers"], default="blobs",
                   help="'blobs' reproduces Jung's thresholded-random-field morphology")
    p.add_argument("--nx", type=int, default=64, help="flow direction; 148 matches Jung's aspect")
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--phi-min", type=float, default=0.45, help="Jung's set spans ~0.45-0.65")
    p.add_argument("--phi-max", type=float, default=0.65)
    p.add_argument("--sigma-range", type=float, nargs=2, default=[3.0, 4.5],
                   help="Gaussian filter width; 4.0 gives Jung's 6-7 voxel correlation length")
    p.add_argument("--fiber-length", type=float, default=None,
                   help="fibers only: fibre length in voxels; default half the\nshortest transverse dimension, times three")
    p.add_argument("--fiber-thickness", type=float, default=None,
                   help="fibers only: fibre radius in voxels; default thickens\nas the target porosity falls")
    p.add_argument("--r-min", type=float, default=3.0, help="--style spheres only")
    p.add_argument("--r-max", type=float, default=7.0, help="--style spheres only")
    p.add_argument("--wall", type=int, default=2,
                   help="transverse bounce-back shell. CompLaB has no periodic option, "
                        "so this cannot be 0")
    p.add_argument("--inlet-buf", type=int, default=5, help="pure-pore buffer, as Jung has")
    p.add_argument("--outlet-buf", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--geodesic", choices=["relax", "fast-marching"],
                   default="relax",
                   help="how to compute the distance from the inlet. 'relax' "
                        "matches every dataset this project builds, so a model "
                        "sees the same scale in training and in prediction. "
                        "'fast-marching' is smoother but gives distances about "
                        "17 percent shorter, which would be a silent mismatch.")
    p.add_argument("--render", action="store_true", help="2D slice + geodesic figure")
    p.add_argument("--render3d", action="store_true", help="3D marching-cubes render (~35 s each)")
    p.add_argument("--render3d-every", type=int, default=1)
    p.add_argument("--verify", action="store_true", help="round-trip the .dat and assert equality")
    p.add_argument("--no-dat", action="store_true")
    args = p.parse_args()

    if args.wall < 1:
        sys.exit("--wall must be >= 1: CompLaB has no periodic boundary option, so pore "
                 "voxels on the y/z domain edge would stream from unallocated space.")

    os.makedirs(args.out, exist_ok=True)
    targets = np.linspace(args.phi_min, args.phi_max, args.n)
    children = np.random.SeedSequence(args.seed).spawn(args.n)

    rows, ok = [], 0
    for gid in range(args.n):
        res = build_one(gid, args, float(targets[gid]), np.random.default_rng(children[gid]))
        if res is None:
            print("  geom %04d FAILED after 15 attempts" % gid, file=sys.stderr); continue
        g, gdf, edt, meta = res
        d = os.path.join(args.out, "geom_%04d" % gid)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "geom_%04d.npz" % gid),
                            material=g, material_jung=to_jung_codes(g),
                            gdf=gdf, edt=edt, meta=json.dumps(meta))
        if not args.no_dat:
            dat = os.path.join(d, "geometry.dat")
            write_dat(g, dat)
            write_dims(g, os.path.join(d, "geometry.dims"))
            if args.verify:
                assert np.array_equal(read_dat(dat, g.shape), g), "round-trip mismatch"
        if args.render:
            render_slices(g, gdf, meta, os.path.join(d, "geom_%04d_slices.png" % gid))
        if args.render3d and (args.render3d_every <= 1 or gid % args.render3d_every == 0):
            render_3d(g, gdf, meta, os.path.join(d, "geom_%04d_3d.png" % gid))

        rows.append(meta); ok += 1
        print("  geom %04d  phi=%.3f  sigma=%.1f  corr=(%d,%d,%d)  tortuosity=%.2f  "
              "GDF-excess mean=%.2f max=%.0f"
              % (gid, meta["phi_interior"], meta["sigma"], meta["corr_len_x"],
                 meta["corr_len_y"], meta["corr_len_z"], meta["tortuosity"],
                 meta["gdf_excess_mean"], meta["gdf_excess_max"]))

    if rows:
        keys = [k for k in rows[0] if k != "spawn_key"]
        with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        phi = [r["phi_interior"] for r in rows]
        cl = [r["corr_len_y"] for r in rows]
        tor = [r["tortuosity"] for r in rows]
        ex = [r["gdf_excess_mean"] for r in rows]
        print("\n%d/%d geometries -> %s   (style=%s)" % (ok, args.n, args.out, args.style))
        print("  porosity            %.3f - %.3f   (Jung's set: ~0.45 - 0.65)" % (min(phi), max(phi)))
        print("  correlation length  %d - %d voxels (Jung's 2D: 6 - 8 pixels)" % (min(cl), max(cl)))
        print("  tortuosity factor   %.2f - %.2f" % (min(tor), max(tor)))
        print("  geodesic excess     %.2f voxels mean  (0 would mean an empty duct; this is"
              % float(np.mean(ex)))
        print("                      the signal the geodesic-vs-Euclidean ablation needs)")
        print("\n  CompLaB.xml must say  <nx>%d</nx><ny>%d</ny><nz>%d</nz>" % (args.nx, args.ny, args.nz))
        print("  (readGeometry adds the 2 ghost x-planes itself — do not include them)")


if __name__ == "__main__":
    main()
