#!/usr/bin/env python3
# =============================================================================
# CHANGED FROM THE 2D VERSION
#
#   WHERE IT CAME FROM
#     github.com/hjunglab/PRT-DeepONet   branch/folder: velocity-informed
#     flow/models/PRT-DeepONet_Velocity_load.ipynb, code cell 3
#     (functions uprm_map, _local_thickness, _sphere_paint, mis_map, dw2_map)
#
#   WHAT THEIR 2D CODE DOES
#     Computes three maps of a pore image: MIS (how wide the pore is at each
#     pixel), UPRM (how wide the narrowest throat between the inlet and that
#     pixel is) and the squared wall distance. Fixed 64 x 148 grid, flow along
#     the SECOND axis, a hard-coded 10 pixel buffer, and z-scoring constants
#     baked into the notebook as literals.
#
#   WHAT WE CHANGED, AND WHY
#     1. N DIMENSIONS. Every function now works on a 2D image or a 3D volume.
#        Their disk structuring element became an n-ball; their four-neighbour
#        walk became a 2*ndim face walk. The 2D answers are unchanged, and
#        3D/model/test_reference_parity.py proves it against their own code on
#        their own example domain, value for value.
#     2. FLOW ON AXIS 0. They store (64, 148) with the flow along axis 1. This
#        whole project puts the flow first. read_reference_orientation() does
#        the transpose in ONE place instead of at every call site.
#     3. BUFFER IS AN ARGUMENT, not the constant BUF = 10. Their campaign pads
#        10 pixels; ours pads 5; a hand-built one may pad none. Getting this
#        wrong measures the padding instead of the rock, so it cannot be a
#        constant.
#     4. SCALING IS RETURNED, NOT BAKED IN. Their notebook hard-codes
#        UPRM_MU = 2.4316839948181954 and friends, which are properties of
#        their 3000-domain campaign. zscore_stats() computes them for yours.
# =============================================================================
"""The geometry descriptors the velocity operator needs, in two and three dimensions.

The geodesic distance already in this project answers a transport question: how far
must a molecule travel from the inlet to reach here, through open pore space. It says
nothing about how fast anything moves along that path, which is why a network given
only the geometry and the geodesic distance cannot tell a preferential channel from a
pocket beside it.

Three descriptors answer the flow question instead.

    MIS   the maximum inscribed sphere radius. The radius of the largest sphere that
          fits inside the pore space and contains this voxel. Purely LOCAL: it says
          how wide the pore is here and nothing else.

    UPRM  the upstream constrained pore radius. The radius of the largest sphere that
          could be delivered from the inlet to this voxel, which is set by the
          narrowest throat along the best available path. NON LOCAL: two voxels in
          identically shaped pockets get different values if one sits behind a
          bottleneck. This is the part a convolution cannot work out for itself,
          because its receptive field is smaller than the path.

    dw2   the squared distance to the nearest wall, scaled to [0, 1]. The no slip
          profile across a duct is parabolic in the wall distance, so handing the
          trunk dw squared saves it from having to discover that shape from data.

PROVENANCE. The 2D forms of uprm_map, _local_thickness, _sphere_paint, mis_map and
dw2_map are ports of the reference implementation released with the velocity informed
PRT-DeepONet by Jo and Jung (github.com/hjunglab/PRT-DeepONet, velocity-informed,
flow/models/PRT-DeepONet_Velocity_load.ipynb). The generalisation to three dimensions
is ours, and test_flow_features.py checks that a 3D volume built by extruding a 2D
image reproduces the 2D answer plane for plane, which is the only claim that can be
made without a 3D reference to compare against.

A NOTE ON AXIS ORDER, because it has bitten this project before. The reference code
stores an image as (nx, ny) = (64, 148) with FLOW ALONG THE SECOND AXIS. Everything in
this project stores (nx, ny, nz) with FLOW ALONG THE FIRST. Every function here takes
the project convention and takes the flow axis as an argument, so nothing has to be
transposed at the call site. read_reference_orientation() converts, and says so.

Run it like this.

    python flow_features.py --self-test
    python flow_features.py --geometry rock.npz --out features.npz --render

What you get back. With --out, one npz holding mis, uprm and dw2 as float32 arrays the
shape of the geometry, plus the scaling constants used. Without it, nothing: the
self test writes no file.

Worth knowing. MIS is computed on the INTERIOR only and then painted outward into the
inlet and outlet buffers. Those buffers are fully open by construction, so a sphere
placed in one would report an enormous radius that has nothing to do with the rock.
The reference implementation does this and so does ours; the visible consequence is a
band of constant value at each end of the MIS and UPRM maps, which is correct rather
than an artefact.
"""
import argparse
import heapq
import os
import sys

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt, binary_dilation
except ImportError:                                                   # pragma: no cover
    sys.exit("flow_features.py needs scipy. Run: python gui/install_requirements.py")

# The buffer width at the inlet and the outlet, in voxels. The published 2D release
# pads 10 voxels at each end of a 128 wide domain to make 148; our own generator
# defaults to 5. The value only matters for where MIS stops being computed and starts
# being painted, so it is an argument everywhere and this is only the fallback.
DEFAULT_BUFFER = 10

__all__ = [
    "mis_map", "uprm_map", "dw2_map", "all_features",
    "pore_mask_from_material", "read_reference_orientation",
]


# --------------------------------------------------------------------------- masks
def pore_mask_from_material(mat, pore_code=None):
    """Boolean pore mask from a material array.

    This project writes 0 solid, 1 interface, 2 pore, following CompLaB. The published
    2D release writes 0 solid, 1 pore, 2 interface, which is the opposite for codes 1
    and 2 and has caused a silent porosity error before now. Pass pore_code when you
    know it; with pore_code=None the array is read as the project's own convention.
    """
    mat = np.asarray(mat)
    if pore_code is not None:
        return mat == pore_code
    if mat.dtype == bool:
        return mat
    # project convention: anything that is not solid and not the wall skin is pore
    return mat >= 2


def read_reference_orientation(arr, flow_axis=0):
    """Convert an array stored the reference way into this project's axis order.

    The reference stores (64, 148) with flow along axis 1. This project puts flow on
    axis 0. Returns a view with the flow axis first.
    """
    arr = np.asarray(arr)
    if flow_axis == 0:
        return arr
    return np.moveaxis(arr, flow_axis, 0)


# -----------------------------------------------------------------------------
# BLOCK 1.  The structuring element
# Their 2D code builds a disk with np.ogrid over two axes. This is the same
# thing over ndim axes, so one function serves an image and a volume.
# -----------------------------------------------------------------------------
def _disk(radius, ndim):
    """A boolean ball of the given radius, as a structuring element."""
    r = int(np.ceil(radius))
    grid = np.ogrid[tuple(slice(-r, r + 1) for _ in range(ndim))]
    d2 = sum(g * g for g in grid)
    return d2 <= radius * radius


# ----------------------------------------------------------------------- local size
def _local_thickness(pore, r_step=0.5):
    """The maximum inscribed sphere radius at every pore voxel.

    The standard granulometry: take the distance transform, which gives the radius of
    the largest sphere CENTRED at each voxel, then sweep radii from large to small and
    paint each voxel with the largest sphere that both fits and covers it. A voxel
    near a wall can still sit inside a large sphere centred further in, which is the
    difference between this and the distance transform itself.
    """
    pore = np.asarray(pore, bool)
    # The distance transform gives the radius of the largest sphere CENTRED at
    # each voxel. That is not what we want: a voxel one step from a grain can
    # still lie inside a large sphere centred further in. So use the transform
    # only to find where big spheres can sit, then paint outward from there.
    edt = distance_transform_edt(pore).astype(np.float32)
    rmax = float(edt.max()) if pore.any() else 0.0
    if rmax <= 0:
        return np.zeros_like(edt)
    out = np.zeros_like(edt)
    # Largest radius first. A voxel keeps the FIRST radius that reaches it,
    # which is therefore the largest, and later smaller spheres cannot
    # overwrite it. Their code relies on the same ordering.
    for r in np.arange(rmax, 0, -r_step):
        centers = pore & (edt >= r - 1e-6)
        if not centers.any():
            continue
        painted = binary_dilation(centers, structure=_disk(r, pore.ndim)) & pore
        newly = painted & (out == 0)
        out[newly] = r
    out[~pore] = 0.0
    return out.astype(np.float32)


def _paint_into_buffers(pore, interior_lt, buf):
    """Extend the interior MIS values into the inlet and outlet buffers.

    The buffers are open channels. A sphere placed in one is limited only by the
    domain height, so computing MIS there would report a radius the rock never has and
    would dominate the z scoring of the whole map. Instead the interior values are
    grown outward into the buffers, largest radius first, and any buffer voxel no
    interior sphere reaches takes the smallest interior value.
    """
    nx = pore.shape[0]
    full = np.zeros(pore.shape, np.float32)
    core = (slice(buf, nx - buf),) + (slice(None),) * (pore.ndim - 1)
    full[core] = interior_lt
    seed = np.zeros_like(full)
    seed[core] = interior_lt

    in_buf = np.zeros(pore.shape, bool)
    in_buf[:buf] = True
    out_buf = np.zeros(pore.shape, bool)
    out_buf[nx - buf:] = True

    for r in np.unique(seed[seed > 0])[::-1]:
        centers = seed >= r - 1e-6
        reached = binary_dilation(centers, structure=_disk(r, pore.ndim))
        for region in (in_buf, out_buf):
            grow = reached & region & (full < r)
            full[grow] = r
    return full, (in_buf | out_buf)


def mis_map(pore, buf=DEFAULT_BUFFER, r_step=0.5):
    """MIS: how wide the pore is at each voxel. Local.

    pore is a boolean array with FLOW ALONG AXIS 0. buf is the number of voxels of
    open buffer at each end; pass 0 for a domain with no padding.
    """
    pore = np.asarray(pore, bool)
    nx = pore.shape[0]
    if buf <= 0 or 2 * buf >= nx:
        out = _local_thickness(pore, r_step)
        return out
    core = (slice(buf, nx - buf),) + (slice(None),) * (pore.ndim - 1)
    interior_lt = _local_thickness(pore[core], r_step)
    full, buf_region = _paint_into_buffers(pore, interior_lt, buf)
    if (interior_lt > 0).any():
        vmin = float(interior_lt[interior_lt > 0].min())
    else:
        vmin = 0.0
    unreached = pore & buf_region & (full <= 0)
    full[unreached] = vmin
    full[~pore] = 0.0
    return full.astype(np.float32)


# -------------------------------------------------------------------- upstream size
def uprm_map(pore, connectivity=1):
    """UPRM: the largest sphere the inlet could deliver to each voxel. Non local.

    A widest path problem, solved with a maximum first priority queue. Start every
    inlet voxel at its own inscribed radius, then relax: the value that can reach a
    neighbour is the smaller of the value here and the neighbour's own radius, because
    a sphere cannot pass a throat narrower than itself. Keep the largest such value
    over all paths.

    Face connectivity by default, which is what the lattice Boltzmann stencil uses.
    A diagonal step would let a sphere squeeze through a corner the solver treats as
    closed, and this project has already been caught once by labelling with 26
    connectivity while the physics used 6.
    """
    # This is a WIDEST PATH problem, not a shortest path one. Dijkstra
    # minimises a sum along the route; here we maximise the MINIMUM along it,
    # because a sphere cannot pass a throat narrower than itself. Same heap,
    # different relaxation rule.
    pore = np.asarray(pore, bool)
    edt = distance_transform_edt(pore).astype(np.float32)
    best = np.zeros(pore.shape, np.float32)      # best bottleneck found so far
    seen = np.zeros(pore.shape, bool)            # settled, never revisited
    heap = []                                    # max-heap, via negated keys

    inlet = np.argwhere(pore[0])
    for cell in inlet:
        idx = (0,) + tuple(int(c) for c in cell)
        best[idx] = float(edt[idx])
        heapq.heappush(heap, (-best[idx], idx))

    ndim = pore.ndim
    steps = []
    for ax in range(ndim):
        for d in (-1, 1):
            s = [0] * ndim
            s[ax] = d
            steps.append(tuple(s))
    if connectivity != 1:
        raise ValueError("uprm_map only supports face connectivity, to match the solver")

    shape = pore.shape
    while heap:
        neg, idx = heapq.heappop(heap)
        if seen[idx]:
            continue
        seen[idx] = True
        here = -neg
        for st in steps:
            nb = tuple(idx[k] + st[k] for k in range(ndim))
            if any(nb[k] < 0 or nb[k] >= shape[k] for k in range(ndim)):
                continue
            if not pore[nb] or seen[nb]:
                continue
            # The relaxation. What can reach the neighbour is the smaller of
            # what reached here and the neighbour's own inscribed radius.
            cand = min(here, float(edt[nb]))
            if cand > best[nb]:
                best[nb] = cand
                heapq.heappush(heap, (-cand, nb))

    best[~pore] = 0.0
    return best.astype(np.float32)


# ------------------------------------------------------------------- wall distance
def dw2_map(pore, dw2_min=None, dw2_max=None):
    """The squared wall distance, scaled to [0, 1] over the pore space.

    Returns the map and the (min, max) actually used, so the same scaling can be
    reapplied to another geometry. Training must reuse ONE pair of constants across
    the whole campaign; rescaling each rock to its own range would tell the network
    that a wide rock and a narrow one look the same, which is exactly the information
    the descriptor exists to carry.
    """
    pore = np.asarray(pore, bool)
    edt = distance_transform_edt(pore).astype(np.float32)
    f2 = (edt * edt).astype(np.float32)
    vals = f2[pore]
    lo = float(dw2_min) if dw2_min is not None else (float(vals.min()) if vals.size else 0.0)
    hi = float(dw2_max) if dw2_max is not None else (float(vals.max()) if vals.size else 1.0)
    if hi <= lo:
        hi = lo + 1.0
    out = np.zeros_like(f2)
    out[pore] = np.clip((f2[pore] - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32), (lo, hi)


# ------------------------------------------------------------------------ the lot
def all_features(pore, buf=DEFAULT_BUFFER, dw2_range=None, r_step=0.5):
    """MIS, UPRM and dw2 for one geometry, in one call.

    Returns a dict with 'mis', 'uprm', 'dw2' and 'dw2_range'. Nothing is z scored
    here: the scaling constants belong to the campaign, not to one rock, so they are
    computed once over the training set and applied by the caller.
    """
    pore = np.asarray(pore, bool)
    lo_hi = dw2_range if dw2_range is not None else (None, None)
    dw2, used = dw2_map(pore, lo_hi[0], lo_hi[1])
    return {
        "mis": mis_map(pore, buf=buf, r_step=r_step),
        "uprm": uprm_map(pore),
        "dw2": dw2,
        "dw2_range": used,
    }


def zscore_stats(maps):
    """Mean and standard deviation over the pore voxels of a list of maps.

    Used once per campaign to fix the MIS and UPRM scaling. Solid voxels are zero by
    construction and are excluded, because including them would move the mean with the
    porosity of the rock rather than with the pore size.
    """
    vals = np.concatenate([m[m > 0].ravel() for m in maps if (m > 0).any()])
    if vals.size == 0:
        return 0.0, 1.0
    sd = float(vals.std())
    return float(vals.mean()), sd if sd > 0 else 1.0


# ----------------------------------------------------------------------- self test
def _self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + name + (("   " + detail) if detail else ""))
        if not cond:
            ok = False

    print("flow_features self test")

    # --- a straight duct of known width
    nx, ny = 40, 21
    pore = np.zeros((nx, ny), bool)
    pore[:, 5:16] = True                      # 11 voxels wide, walls at 4 and 16
    mis = mis_map(pore, buf=0)
    uprm = uprm_map(pore)
    dw2, rng = dw2_map(pore)
    centre = mis[nx // 2, 10]
    check("duct MIS at the centre line is about half the width",
          abs(centre - 5.5) <= 0.5, "got %.2f, half width is 5.5" % centre)
    check("duct UPRM is constant along the duct",
          float(uprm[5:-5, 10].std()) < 1e-5,
          "sd %.2e" % float(uprm[5:-5, 10].std()))
    check("dw2 peaks on the centre line",
          np.argmax(dw2[nx // 2]) == 10, "peak at %d" % int(np.argmax(dw2[nx // 2])))
    check("nothing is written outside the pore space",
          float(np.abs(mis[~pore]).max()) == 0.0 and float(np.abs(uprm[~pore]).max()) == 0.0)

    # --- the bottleneck case, which is the whole point of UPRM
    nx, ny = 60, 31
    pore = np.zeros((nx, ny), bool)
    pore[:, 5:26] = True                      # a wide duct, 21 voxels
    pore[28:32, :] = False
    pore[28:32, 14:17] = True                 # pinched to 3 voxels in the middle
    mis = mis_map(pore, buf=0)
    uprm = uprm_map(pore)
    up_before = float(uprm[10, 15])
    up_after = float(uprm[50, 15])
    mis_after = float(mis[50, 15])
    check("UPRM downstream of a throat is smaller than upstream",
          up_after < up_before - 0.5, "%.2f after, %.2f before" % (up_after, up_before))
    check("UPRM downstream is capped by the throat, not the local width",
          up_after < mis_after, "UPRM %.2f, local MIS %.2f" % (up_after, mis_after))
    check("MIS does not notice the throat from downstream",
          mis_after > up_after + 0.5)

    # --- a dead end pocket: same shape, different access
    nx, ny = 50, 31
    pore = np.zeros((nx, ny), bool)
    pore[:, 13:18] = True                     # a narrow spine, 5 voxels
    pore[20:30, 2:12] = True                  # a wide pocket, reached only through the spine
    pore[20:30, 12:14] = True                 # the connection
    uprm = uprm_map(pore)
    mis = mis_map(pore, buf=0)
    pocket = (25, 6)
    check("a wide pocket behind a narrow spine has MIS larger than UPRM",
          mis[pocket] > uprm[pocket] + 0.5,
          "MIS %.2f, UPRM %.2f" % (mis[pocket], uprm[pocket]))

    # --- 3D against 2D: an extruded image must give the extruded answer
    nx, ny = 34, 21
    rng_ = np.random.default_rng(0)
    base = np.zeros((nx, ny), bool)
    base[:, 4:17] = True
    base[15:19, 4:9] = False
    nz = 7
    vol = np.repeat(base[:, :, None], nz, axis=2)
    u2 = uprm_map(base)
    u3 = uprm_map(vol)
    # In 3D the sphere is limited by the z walls as well, so the values are not equal.
    # What must hold is the ORDERING, since that is what the descriptor encodes.
    a = u2[base]
    b = u3[:, :, nz // 2][base]
    if a.size > 2 and a.std() > 0 and b.std() > 0:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = 1.0
    check("3D UPRM on an extruded image ranks the pore space the same way as 2D",
          corr > 0.95, "correlation %.4f" % corr)

    m2 = mis_map(base, buf=0)
    m3 = mis_map(vol, buf=0)
    check("3D MIS never exceeds the 2D MIS of the same slice",
          bool((m3[:, :, nz // 2][base] <= m2[base] + 1e-4).all()),
          "the z walls can only make the sphere smaller")

    # --- the buffer treatment
    nx, ny = 60, 31
    pore = np.zeros((nx, ny), bool)
    pore[:, 8:23] = True
    pore[:10, :] = True                       # a fully open inlet buffer
    pore[-10:, :] = True
    with_buf = mis_map(pore, buf=10)
    without = mis_map(pore, buf=0)
    check("painting the buffers keeps MIS below the open channel value",
          float(with_buf[:10].max()) < float(without[:10].max()) - 0.5,
          "%.2f painted, %.2f raw" % (float(with_buf[:10].max()), float(without[:10].max())))
    check("the buffer band is constant across its own width",
          float(with_buf[0][pore[0]].std()) < 1e-5)

    # --- dw2 scaling is reusable
    d1, r1 = dw2_map(pore)
    d2, r2 = dw2_map(pore, r1[0], r1[1])
    check("reapplying a stored dw2 range reproduces the map",
          float(np.abs(d1 - d2).max()) < 1e-6)
    check("dw2 stays inside [0, 1]",
          float(d1.min()) >= 0.0 and float(d1.max()) <= 1.0)

    # --- all_features agrees with the individual calls
    f = all_features(pore, buf=10)
    check("all_features returns the same maps as the individual functions",
          float(np.abs(f["mis"] - with_buf).max()) < 1e-6
          and float(np.abs(f["uprm"] - uprm_map(pore)).max()) < 1e-6)

    print("\n" + ("Everything passed." if ok else "SOMETHING FAILED, read the lines above."))
    return 0 if ok else 1


# ----------------------------------------------------------------------------- cli
def _load_geometry(path, pore_code=None):
    if path.endswith(".npz"):
        with np.load(path) as d:
            for key in ("material", "m", "mat", "geom", "arr_0"):
                if key in d.files:
                    return np.asarray(d[key]), key
            return np.asarray(d[d.files[0]]), d.files[0]
    if path.endswith(".npy"):
        return np.load(path), "npy"
    raise SystemExit("give a .npz or .npy geometry, got %s" % path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="MIS, UPRM and the squared wall distance for a pore geometry.")
    ap.add_argument("--geometry", help=".npz or .npy holding the material or pore array")
    ap.add_argument("--pore-code", type=int, default=None,
                    help="which material code means pore. Read from your input file "
                         "rather than guessed; omit for a boolean array or this "
                         "project's own 0 solid, 1 wall, 2 pore convention")
    ap.add_argument("--buffer", type=int, default=DEFAULT_BUFFER,
                    help="open buffer voxels at each end of the flow axis (default %d). "
                         "Use 0 for a domain with no padding." % DEFAULT_BUFFER)
    ap.add_argument("--flow-axis", type=int, default=0,
                    help="which axis the flow runs along in the file. Default 0, this "
                         "project's convention. The published 2D release uses 1.")
    ap.add_argument("--out", help="write the three maps to this .npz")
    ap.add_argument("--render", action="store_true", help="also write a .png of each map")
    ap.add_argument("--self-test", action="store_true", help="check the descriptors and exit")
    a = ap.parse_args(argv)

    if a.self_test:
        return _self_test()
    if not a.geometry:
        ap.error("give --geometry, or --self-test")

    arr, key = _load_geometry(a.geometry)
    arr = read_reference_orientation(arr, a.flow_axis)
    pore = pore_mask_from_material(arr, a.pore_code)
    print("read %s from %s, shape %s, porosity %.3f"
          % (key, a.geometry, tuple(arr.shape), float(pore.mean())))
    if not pore[0].any():
        print("WARNING: no pore voxel on the inlet face. UPRM will be zero everywhere, "
              "which usually means the flow axis is wrong. Try --flow-axis 1.")

    f = all_features(pore, buf=a.buffer)
    for name in ("mis", "uprm", "dw2"):
        m = f[name]
        v = m[pore]
        print("  %-5s min %.3f  mean %.3f  max %.3f" % (name, v.min(), v.mean(), v.max()))
    print("  dw2 scaled over [%.3f, %.3f]" % f["dw2_range"])

    if a.out:
        np.savez_compressed(a.out, mis=f["mis"], uprm=f["uprm"], dw2=f["dw2"],
                            dw2_range=np.asarray(f["dw2_range"], np.float64),
                            pore=pore, buffer=a.buffer)
        print("wrote", a.out)
    if a.render:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed, skipping --render")
            return 0
        stem = os.path.splitext(a.out or a.geometry)[0]
        for name in ("mis", "uprm", "dw2"):
            m = f[name]
            if m.ndim == 3:
                m = m[:, :, m.shape[2] // 2]
                pm = pore[:, :, pore.shape[2] // 2]
            else:
                pm = pore
            img = np.ma.masked_where(~pm, m)
            fig, ax = plt.subplots(figsize=(8, 8 * m.shape[1] / max(m.shape[0], 1)))
            im = ax.imshow(img.T, origin="lower", cmap="viridis")
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.03)
            fig.savefig(stem + "_" + name + ".png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            print("wrote", stem + "_" + name + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
