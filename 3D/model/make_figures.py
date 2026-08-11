#!/usr/bin/env python3
"""
make_figures.py — shared 2D and 3D field rendering for predict.py and evaluate.py.

Every figure shows the same three settings_and_units quantities the paper is about:
    FLOW      velocity magnitude, what carries everything
    BIOTIC    the microbial reaction rate field, R_bio
    ABIOTIC   the surface reaction rate field, R_abio
plus the individual species concentrations.
"""

import numpy as np

PORE = 2


# ---------------------------------------------------------------- rate fields
# The writers do not agree on parameter names: collect_complab_output.py emits
# 'ks_ac_norm', the 2D and ingest tools emit 'Ks_Ac_over_Ac0'. A plain
# dictionary lookup silently fell through to the DEFAULTS, so every rate field
# from those datasets was computed with Da = 1.0 and the stock half-saturation
# constants, while the figure title displayed the run's real values. Aliases are
# resolved here so old and new files both work.
_PARAM_ALIASES = {
    "pe": ("pe", "peclet"),
    "da_bio": ("da_bio", "dabio", "da"),
    "da_abio": ("da_abio", "daabio"),
    "ks_ac_norm": ("ks_ac_norm", "ks_ac_over_ac0", "ks_ac", "ksac"),
    "ks_a_norm": ("ks_a_norm", "ks_a_over_a0", "ks_a", "ksa"),
    "y_norm": ("y_norm", "y", "yield"),
}


def _resolve_params(param_names, params):
    raw = {str(n).lower(): float(v) for n, v in zip(param_names, params)}
    out = dict(raw)
    for canon, names in _PARAM_ALIASES.items():
        for n in names:
            if n in raw:
                out[canon] = raw[n]
                break
    return out


def reaction_rates(conc, species, params, param_names):
    """Turn predicted concentrations into the reaction-rate fields.

    These are what a reactive-transport reader actually wants to see, and they
    are the direct analogue of the rate laws in defineKinetics.hh:

        R_bio  = Da_bio  * Bio * [Ac/(Ks_Ac+Ac)] * [A/(Ks_A+A)]
        R_abio = Da_abio * P

    Written in the dimensionless groups the model was conditioned on, so the
    magnitudes are comparable across runs. Returns (R_bio, R_abio), both NaN
    outside the pore space, exactly like the concentration fields.
    """
    p = _resolve_params(param_names, params)
    idx = {s: i for i, s in enumerate(species)}
    g = lambda k: conc[idx[k]] if k in idx else None

    Ac, A, P, Bio = g("Ac"), g("A"), g("P"), g("Bio")
    R_bio = R_abio = None
    if Ac is not None and A is not None and Bio is not None:
        ks_ac = max(p.get("ks_ac_norm", 0.1), 1e-12)
        ks_a = max(p.get("ks_a_norm", 0.15), 1e-12)
        acp, ap, bp = np.clip(Ac, 0, None), np.clip(A, 0, None), np.clip(Bio, 0, None)
        R_bio = p.get("da_bio", 1.0) * bp * (acp / (ks_ac + acp)) * (ap / (ks_a + ap))
    if P is not None:
        R_abio = p.get("da_abio", 1.0) * np.clip(P, 0, None)
    return R_bio, R_abio


def velocity_magnitude(vel):
    return None if vel is None else np.sqrt((np.asarray(vel, np.float32) ** 2).sum(0))


# ------------------------------------------------------------------------ 2D
def _norm(panels):
    """Accept (name, array, cmap) or (name, array, cmap, (vmin, vmax)).

    The 4th element lets a truth panel and its prediction share one colour
    scale. Without that they are auto-scaled independently and two fields that
    differ by a factor of two can be drawn identically, which is exactly the
    error a reader is trying to see."""
    out = []
    for p in panels:
        n, a, c = p[0], p[1], p[2]
        lim = p[3] if len(p) > 3 else None
        if a is not None:
            out.append((n, a, c, lim))
    return out


def shared_limits(*arrays):
    """(vmin, vmax) over every finite value in the given arrays."""
    vals = [np.asarray(a)[np.isfinite(a)] for a in arrays if a is not None]
    vals = [v for v in vals if v.size]
    if not vals:
        return None
    v = np.concatenate(vals)
    return float(v.min()), float(v.max())


def render_2d(material, panels, path, title, slice_axis=2, slice_index=None):
    """panels: list of (name, 3D array, cmap[, (vmin, vmax)]). Mid-plane slices."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = _norm(panels)
    if not panels:
        return
    sh = material.shape
    k = sh[slice_axis] // 2 if slice_index is None else slice_index
    take = lambda a: (a[k] if slice_axis == 0 else a[:, k] if slice_axis == 1 else a[:, :, k]).T

    ncol = min(4, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.9 * nrow), squeeze=False)
    ax = ax.ravel()
    for j in range(len(panels), len(ax)):
        ax[j].axis("off")
    solid = take((material != PORE).astype(float))
    for i, (nm, arr, cm, lim) in enumerate(panels):
        img = np.where(solid > 0.5, np.nan, take(arr))
        kw = dict(origin="lower", cmap=cm, interpolation="nearest")
        if lim is not None:
            kw.update(vmin=lim[0], vmax=lim[1])
        im = ax[i].imshow(img, **kw)
        ax[i].imshow(np.where(solid > 0.5, 1.0, np.nan), origin="lower",
                     cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
        ax[i].set_title(nm, fontsize=11)
        ax[i].set_xlabel("x (flow +)")
        fig.colorbar(im, ax=ax[i], fraction=.046)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ 3D
def _grain_surface(ax, solid, color="#8d949d", alpha=0.30, offset=0):
    """Translucent grain skin. edgecolor must be switched off explicitly:
    Poly3DCollection otherwise strokes every triangle and a 64^3 marching-cubes
    mesh then reads as a wire grid rather than a surface."""
    if solid.sum() == 0:
        return
    try:
        from skimage import measure
        m = np.pad(solid.astype(np.float32), 1)
        v, f, _, _ = measure.marching_cubes(m, 0.5, step_size=2)
        v += float(offset) - 1.0
        s = ax.plot_trisurf(v[:, 0], v[:, 1], f, v[:, 2], color=color, alpha=alpha,
                            linewidth=0, antialiased=False, shade=True)
        s.set_edgecolor("none")
    except Exception:
        pass


def render_3d(material, panels, path, title, max_points=14000, cut=True, trim=3):
    """One 3D panel per field: grains translucent, field as a coloured point
    cloud through the half-cut pore space.

    `trim` peels the outer voxels off every face before rendering. The geometries
    carry a bounce-back shell on the transverse faces, and that shell is a closed
    box in the marching-cubes surface: leave it in and it hides the entire
    interior. Trimming it is a rendering choice only, nothing is trimmed from the
    prediction or the metrics."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = _norm(panels)
    if not panels:
        return
    # A 3D render of a field with one voxel along z is a picture of nothing,
    # and worse, trimming would empty it: slice(3, 1-3) is slice(3, -2), which
    # selects zero elements on a length-1 axis. The result was a blank axes
    # frame written at full rendering cost for every native 2D dataset. Say so
    # and decline, rather than producing an empty figure that looks like a
    # result.
    if min(material.shape) <= 1:
        print("  (skipping the 3D render: this field is %s, which has no third "
              "dimension to cut away. The 2D figures are the ones to look at.)"
              % "x".join(str(s) for s in material.shape))
        return

    t = int(max(0, trim))
    # Never trim an axis down to nothing.
    t = min(t, (min(material.shape) - 1) // 2)
    sl = (slice(t, material.shape[0] - t or None),
          slice(t, material.shape[1] - t or None),
          slice(t, material.shape[2] - t or None))
    material = material[sl]
    panels = [(n, np.asarray(a)[sl], c, lim) for n, a, c, lim in panels]

    nx, ny, nz = material.shape
    solid = (material != PORE)
    keep = np.ones(material.shape, bool)
    if cut:
        keep[:, ny // 2:, :] = False
    show = (~solid) & keep

    fig = plt.figure(figsize=(5.0 * len(panels), 5.0))
    for i, (nm, arr, cm, lim) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        _grain_surface(ax, solid & keep, offset=t)
        pts = np.argwhere(show & np.isfinite(arr))
        if len(pts):
            step = max(1, len(pts) // max_points)
            q = pts[::step]
            v = arr[q[:, 0], q[:, 1], q[:, 2]]
            kw = dict(vmin=lim[0], vmax=lim[1]) if lim is not None else {}
            sc = ax.scatter(q[:, 0] + t, q[:, 1] + t, q[:, 2] + t, c=v, cmap=cm,
                            s=4, alpha=0.70, linewidths=0, **kw)
            fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.02)
        ax.set_xlim(0, nx + 2 * t); ax.set_ylim(0, ny + 2 * t); ax.set_zlim(0, nz + 2 * t)
        try: ax.set_box_aspect((1, 1, 1))
        except Exception: pass
        ax.set_xlabel("x (flow +)"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(nm, fontsize=11)
        ax.view_init(elev=18, azim=-62)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=112, bbox_inches="tight")
    plt.close(fig)


def standard_panels(conc, species, vel, params, param_names):
    """The panel set used everywhere: flow, the two rate fields, then species."""
    umag = velocity_magnitude(vel)
    R_bio, R_abio = reaction_rates(conc, species, params, param_names)
    panels = [("FLOW  |u|", umag, "cividis"),
              ("BIOTIC  R_bio", R_bio, "YlGn"),
              ("ABIOTIC  R_abio", R_abio, "OrRd")]
    panels += [(s, conc[i], "viridis") for i, s in enumerate(species)]
    # R_bio needs Ac, A and Bio; R_abio needs P. On a two-chemical dataset both
    # are None, the renderer drops them, and what was billed as a physics
    # figure quietly becomes a single picture of the flow field -- at the full
    # cost of a 3D render. Say so once, where it can be acted on.
    if R_bio is None or R_abio is None:
        missing = []
        if R_bio is None:
            missing.append("the biotic rate needs Ac, A and Bio")
        if R_abio is None:
            missing.append("the abiotic rate needs P")
        print("   (rate fields not drawn: %s, and this dataset has %s. Build "
              "one with --n-species 4 to get them.)"
              % ("; ".join(missing), ", ".join(species)))
    return panels, R_bio, R_abio
