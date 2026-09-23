#!/usr/bin/env python3
"""
write_vti_and_png.py -- the pictures the dataset builders offer to write.

WHY THIS FILE EXISTS
    AUDIT VTI-01. build_dataset_2d.py and build_dataset_3d.py both take
    --save-vti and --save-png, and both imported this module inside a
    try/except that set it to None when it was missing. It WAS missing, so the
    two flags accepted the request, printed a note and wrote nothing. A flag
    that is accepted and then ignored is worse than a flag that does not exist,
    because the run looks like it did what was asked.

    The file is restored here rather than the flags removed, because looking at
    a few runs is how a builder run gets checked at all: an .h5 full of numbers
    does not show a front that never entered the sample.

WHAT IT WRITES
    <prefix>/run_%04d_<species>_t%02d.vti     one scalar field per snapshot
    <prefix>/run_%04d_velocity.vti            the velocity, as a vector
    <prefix>/run_%04d_material.vti            the geometry, once per run
    <prefix>/run_%04d.png                     a contact sheet: every species
                                              down the page, time across it

    The .vti files are VTK ImageData with inline base64, byte for byte the
    format 3D/model/predict.py writes, so a simulated field and a predicted one
    open side by side in ParaView with no conversion.

WHAT IT IS CAREFUL ABOUT
    2D and 3D. The 2D builder holds (T, C, nx, ny) and the 3D builder holds
    (T, C, nx, ny, nz). Both are accepted; a 2D field is written as a single
    z plane, which is what ParaView expects and what predict.py does.

    The solid. Grains are masked out of the .png rather than drawn as zero
    concentration, because a zero inside a grain and a zero in the pore mean
    different things and a picture that shows them the same colour hides the
    front.

    Doing nothing. want_vti and want_png are honoured separately, and nothing
    is created that was not asked for.

USED BY
    3D/tools/build_dataset_2d.py  --save-vti / --save-png / --save-runs
    3D/tools/build_dataset_3d.py  the same
"""

import base64
import os
import struct

import numpy as np


# ---------------------------------------------------------------------- VTI
def _b64(raw):
    """VTK's appended-data convention: a uint32 byte count, then the bytes."""
    return base64.b64encode(struct.pack("<I", len(raw)) + raw).decode()


# =============================================================================
#  THE VTK FORMAT, WRITTEN BY HAND
#  Deliberately not through a VTK library. This has to produce byte for byte
#  what 3D/model/predict.py produces, so that a simulated field and a predicted
#  one open side by side in ParaView, and the surest way to guarantee that is
#  for both to write the same few lines of XML.
# =============================================================================
def write_vti(path, array, name, spacing=1.0):
    """One scalar field as VTK ImageData.

    array is (nx, ny) or (nx, ny, nz). A 2D field is written with nz = 1, which
    is how predict.py writes one, so the two are directly comparable.
    """
    a = np.asarray(array, np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    nx, ny, nz = a.shape
    raw = a.transpose(2, 1, 0).ravel().tobytes()
    sp = float(spacing)
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n'
                '<VTKFile type="ImageData" version="0.1" '
                'byte_order="LittleEndian" header_type="UInt32">\n'
                '<ImageData WholeExtent="0 %d 0 %d 0 %d" Origin="0 0 0" '
                'Spacing="%g %g %g">\n<Piece Extent="0 %d 0 %d 0 %d">\n'
                '<PointData>\n'
                % (nx - 1, ny - 1, nz - 1, sp, sp, sp,
                   nx - 1, ny - 1, nz - 1))
        f.write('<DataArray type="Float64" Name="%s" NumberOfComponents="1" '
                'format="binary">%s</DataArray>\n' % (name, _b64(raw)))
        f.write("</PointData><CellData></CellData></Piece></ImageData>"
                "</VTKFile>\n")


def write_vti_vector(path, comps, name, spacing=1.0):
    """A vector field as VTK ImageData. comps is (ncomp, nx, ny[, nz]).

    A 2D field is padded to three components with zeros, because ParaView's
    vector glyphs expect three and a two-component array is read as something
    else entirely.
    """
    c = np.asarray(comps, np.float64)
    if c.ndim == 3:
        c = c[:, :, :, None]
    ncomp, nx, ny, nz = c.shape
    if ncomp < 3:
        pad = np.zeros((3 - ncomp, nx, ny, nz), np.float64)
        c = np.concatenate([c, pad], axis=0)
    inter = c.transpose(3, 2, 1, 0).ravel()          # z, y, x, component
    sp = float(spacing)
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n'
                '<VTKFile type="ImageData" version="0.1" '
                'byte_order="LittleEndian" header_type="UInt32">\n'
                '<ImageData WholeExtent="0 %d 0 %d 0 %d" Origin="0 0 0" '
                'Spacing="%g %g %g">\n<Piece Extent="0 %d 0 %d 0 %d">\n'
                '<PointData>\n'
                % (nx - 1, ny - 1, nz - 1, sp, sp, sp,
                   nx - 1, ny - 1, nz - 1))
        f.write('<DataArray type="Float64" Name="%s" NumberOfComponents="3" '
                'format="binary">%s</DataArray>\n'
                % (name, _b64(inter.tobytes())))
        f.write("</PointData><CellData></CellData></Piece></ImageData>"
                "</VTKFile>\n")


# ---------------------------------------------------------------------- PNG
def _mid_plane(a):
    """A 2D picture of a 2D or 3D field: the field, or its middle z plane."""
    a = np.asarray(a)
    return a if a.ndim == 2 else a[:, :, a.shape[2] // 2]


def save_png(path, conc, times, species, mat=None, note=""):
    """A contact sheet: one row per species, one column per snapshot.

    Returns False and writes nothing when matplotlib is not installed, because
    a missing drawing library should not stop a dataset being built.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    c = np.asarray(conc, np.float32)                  # (T, C, nx, ny[, nz])
    nt, nc = c.shape[0], c.shape[1]
    solid = None if mat is None else (_mid_plane(np.asarray(mat)) == 0)

    fig, axes = plt.subplots(nc, nt, squeeze=False,
                             figsize=(1.9 * nt + 1.2, 1.9 * nc + 0.9),
                             constrained_layout=True)
    for ci in range(nc):
        # one colour scale per species, over the whole run, so the snapshots
        # can be compared with each other rather than each being restretched
        lo, hi = float(np.nanmin(c[:, ci])), float(np.nanmax(c[:, ci]))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, max(hi, 1e-12)
        for ti in range(nt):
            ax = axes[ci][ti]
            f = _mid_plane(c[ti, ci]).astype(np.float32)
            if solid is not None:
                f = np.where(solid, np.nan, f)        # grains are not zero
            ax.imshow(f.T, origin="lower", vmin=lo, vmax=hi, cmap="viridis",
                      interpolation="nearest", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_title("t = %.3g" % float(times[ti]), fontsize=8)
            if ti == 0:
                name = species[ci] if ci < len(species) else "field %d" % ci
                ax.set_ylabel("%s\n%.3g to %.3g" % (name, lo, hi), fontsize=8)
    if note:
        fig.suptitle(note, fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ the entry
# =============================================================================
#  ONE RUN'S PICTURES
#  Returns the LIST of files written rather than assuming any were. That is the
#  whole of AUDIT VTI-01: the caller used to assume, the module was absent, and
#  every run reported pictures it had not produced.
# =============================================================================
def save_run(prefix, k, conc, times, species, mat=None, vel=None,
             want_vti=True, want_png=True, spacing=1.0, note=""):
    """Write the pictures for one run of a dataset build.

    prefix    a directory; it is created if it does not exist
    k         the run number, used in every file name
    conc      (T, C, nx, ny) in 2D or (T, C, nx, ny, nz) in 3D
    times     the T normalised times, used to label the columns
    species   the C names, in the order of the second axis of conc
    mat       the geometry, so the grains can be masked out of the picture
    vel       (ncomp, nx, ny[, nz]), written once per run as a vector field

    Returns the list of files written, so the caller can report it rather than
    assume it. AUDIT VTI-01: the caller used to assume, and the assumption was
    wrong for every run.
    """
    os.makedirs(prefix, exist_ok=True)
    written = []
    c = np.asarray(conc, np.float32)
    nt, nc = c.shape[0], c.shape[1]
    tag = os.path.join(prefix, "run_%04d" % int(k))

    if want_vti:
        for ci in range(nc):
            name = species[ci] if ci < len(species) else "field%d" % ci
            for ti in range(nt):
                p = "%s_%s_t%02d.vti" % (tag, name, ti)
                write_vti(p, c[ti, ci], name, spacing)
                written.append(p)
        if mat is not None:
            p = tag + "_material.vti"
            write_vti(p, np.asarray(mat, np.float64), "material", spacing)
            written.append(p)
        if vel is not None:
            p = tag + "_velocity.vti"
            write_vti_vector(p, np.asarray(vel), "velocity", spacing)
            written.append(p)

    if want_png:
        p = tag + ".png"
        if save_png(p, c, times, species, mat=mat, note=note):
            written.append(p)

    return written


# ------------------------------------------------------------------ self test
def _self_test():
    """Writes both formats for a 2D and a 3D run and reads the .vti back."""
    import tempfile
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print("  %-5s %-52s %s" % ("PASS" if cond else "FAIL", name, extra))

    rng = np.random.default_rng(0)
    out = tempfile.mkdtemp()

    # 2D: (T, C, nx, ny)
    c2 = rng.random((3, 2, 12, 8)).astype(np.float32)
    m2 = np.ones((12, 8), np.uint8) * 2
    m2[5:7, 2:5] = 0
    v2 = rng.random((2, 12, 8)).astype(np.float32)
    w = save_run(out, 0, c2, [0.1, 0.5, 1.0], ["Ac", "A"], mat=m2, vel=v2,
                 spacing=1e-5, note="2D self test")
    check("a 2D run writes its files", len(w) >= 3 * 2 + 2, "%d files" % len(w))
    check("and one of them is the contact sheet",
          any(p.endswith(".png") for p in w))

    # 3D: (T, C, nx, ny, nz)
    c3 = rng.random((2, 1, 6, 5, 4)).astype(np.float32)
    m3 = np.ones((6, 5, 4), np.uint8) * 2
    v3 = rng.random((3, 6, 5, 4)).astype(np.float32)
    w3 = save_run(out, 1, c3, [0.5, 1.0], ["Ac"], mat=m3, vel=v3)
    check("a 3D run writes its files", len(w3) >= 2 + 2, "%d files" % len(w3))

    # read one back, the same way the collector reads CompLaB's output
    try:
        from collect_complab_output import read_vti
        arrays, dims = read_vti(os.path.join(out, "run_0000_Ac_t00.vti"))
        back = list(arrays.values())[0]
        check("the .vti reads back at the right shape",
              tuple(back.shape[:2]) == (12, 8), str(tuple(back.shape)))
        check("and with the same numbers",
              np.allclose(np.asarray(back).reshape(12, 8, -1)[:, :, 0],
                          c2[0, 0], atol=1e-9))
    except Exception as e:
        check("the .vti reads back with the project's own reader", False, str(e))

    check("nothing is written when nothing is asked for",
          save_run(out, 2, c2, [0.1, 0.5, 1.0], ["Ac", "A"],
                   want_vti=False, want_png=False) == [])

    print("\n%s" % ("Everything passed." if ok else "SOMETHING FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(_self_test())
    ap.print_help()
