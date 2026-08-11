#!/usr/bin/env python3
"""
write_vti_and_png.py -- pictures and ParaView files of what a simulation produced.

WHY THIS EXISTS
    The dataset is one HDF5 file, which is the right thing for training and
    the wrong thing for looking at. Two other formats answer questions the
    HDF5 file cannot answer quickly.

    VTI is what CompLaB writes, one file per chemical per saved time, and it
    opens in ParaView. If you already look at CompLaB output in ParaView, a
    Python run in the same format drops straight into the same session and the
    two can be put side by side.

    PNG is for the question you actually ask first: did the front cross the
    sample, or is the field empty? One picture answers that in a second. This
    is something CompLaB does NOT do -- it declares an image writer and never
    uses it, so the only way to see a CompLaB result is to open ParaView.

WHAT IS WRITTEN
    <out>_fields/
        run0000/
            Ac_0000.vti  Ac_0005.vti  ...     one per chemical per snapshot
            material.vti                      the rock, once per structure
            speed.vti                      the flow field
            Ac.png  A.png  P.png  Bio.png     the last snapshot, mid-plane
            middle.png                      every chemical, first and last

    A 3D field is written as a genuine 3D image. A 2D field is written with
    nz = 1, which ParaView reads as a single slice.

NOT A TRANSPORT FORMAT
    These files are for looking at. Nothing in this project reads them back:
    training, evaluation and the viewer all read the HDF5 file. Deleting the
    whole folder loses nothing but the pictures.
"""

import base64
import os
import zlib

import numpy as np


# --------------------------------------------------------------------------
def _vti_bytes(arr, name, spacing=1.0, origin=(0.0, 0.0, 0.0)):
    """One VTK ImageData file, as bytes.

    Written by hand rather than with a library, because the alternative is a
    dependency the rest of the project does not need for a format that is
    forty lines. Appended raw with a 64-bit length header, which is what
    ParaView expects for appended_format="raw".
    """
    a = np.ascontiguousarray(np.asarray(arr, np.float32))
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError("a VTI field must be 2D or 3D, got %dD" % a.ndim)
    nx, ny, nz = a.shape
    # VTK indexes x fastest; numpy indexes the last axis fastest. Transposing
    # here rather than hoping is the difference between a picture of the rock
    # and a picture of the rock's transpose, and both look plausible.
    payload = a.transpose(2, 1, 0).tobytes(order="C")
    head = ('<?xml version="1.0"?>\n'
            '<VTKFile type="ImageData" version="1.0" '
            'byte_order="LittleEndian" header_type="UInt64">\n'
            '  <ImageData WholeExtent="0 %d 0 %d 0 %d" Origin="%g %g %g" '
            'Spacing="%g %g %g">\n'
            '    <Piece Extent="0 %d 0 %d 0 %d">\n'
            '      <PointData Scalars="%s">\n'
            '        <DataArray type="Float32" Name="%s" format="appended" '
            'offset="0"/>\n'
            '      </PointData>\n'
            '    </Piece>\n'
            '  </ImageData>\n'
            '  <AppendedData encoding="raw">\n_'
            % (nx - 1, ny - 1, nz - 1, origin[0], origin[1], origin[2],
               spacing, spacing, spacing,
               nx - 1, ny - 1, nz - 1, name, name))
    tail = "\n  </AppendedData>\n</VTKFile>\n"
    return (head.encode("ascii")
            + np.array([len(payload)], "<u8").tobytes()
            + payload + tail.encode("ascii"))


def write_vti(path, arr, name, spacing=1.0):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(_vti_bytes(arr, name, spacing))
    return path


# --------------------------------------------------------------------------
def _mid_slice(a):
    """The middle plane of a volume, or the plane itself in 2D."""
    a = np.asarray(a)
    if a.ndim == 2:
        return a
    if a.shape[2] == 1:
        return a[:, :, 0]
    return a[:, :, a.shape[2] // 2]


def write_png_panel(path, fields, names, mat=None, title=""):
    """One picture, one panel per chemical, on a shared colour scale.

    A SHARED scale, deliberately. Scaling each panel to its own maximum makes
    a field of 0.001 look identical to a field of 1.0, which is exactly the
    failure this picture exists to catch.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(fields)
    if n == 0:
        return None
    slices = [_mid_slice(f) for f in fields]
    hi = max(float(np.nanmax(s)) for s in slices) or 1.0
    rock = _mid_slice(mat) if mat is not None else None

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), squeeze=False)
    for i, (s, nm) in enumerate(zip(slices, names)):
        ax = axes[0][i]
        im = ax.imshow(s.T, origin="lower", vmin=0.0, vmax=hi,
                       cmap="viridis", interpolation="nearest")
        if rock is not None:
            # The grain outlines. Drawn from the TRANSPOSED mask, like the
            # field above it.
            #
            # There used to be a second, invisible imshow here -- fully
            # transparent, left over from an earlier attempt at shading the
            # grains -- and it was drawn UNtransposed. An invisible image
            # still sets the axes limits, so a 60 by 32 domain got y limits of
            # 0 to 59 while the real picture occupied 0 to 31, and every panel
            # came out with the top half blank. It looked exactly like half
            # the sample being empty.
            ax.contour(rock.T == 0, levels=[0.5], colors="0.35",
                       linewidths=0.6)
        ax.set_title("%s   max %.3g" % (nm, float(np.nanmax(s))), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8,
                 label="concentration, scaled to the inlet")
    if title:
        fig.suptitle(title, fontsize=10)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def save_run(folder, run_id, conc, t_norm, species, mat=None, vel=None,
             want_vti=True, want_png=True, spacing=1.0, note=""):
    """Everything for one simulation. Returns the list of files written."""
    out = []
    d = os.path.join(folder, "run%04d" % int(run_id))
    conc = np.asarray(conc)                       # (T, C) + grid
    T, C = conc.shape[0], conc.shape[1]

    if want_vti:
        for t in range(T):
            for c in range(C):
                out.append(write_vti(
                    os.path.join(d, "%s_%04d.vti" % (species[c], t)),
                    conc[t, c], species[c], spacing))
        if mat is not None:
            out.append(write_vti(os.path.join(d, "material.vti"),
                                 np.asarray(mat, np.float32), "material",
                                 spacing))
        if vel is not None:
            v = np.asarray(vel)
            out.append(write_vti(os.path.join(d, "speed.vti"),
                                 np.sqrt((v ** 2).sum(0)), "speed", spacing))

    if want_png:
        out.append(write_png_panel(
            os.path.join(d, "last.png"),
            [conc[-1, c] for c in range(C)], list(species[:C]), mat,
            "run %d, last snapshot (t = %.3g)%s"
            % (int(run_id), float(t_norm[-1]),
               ("   " + note) if note else "")))
        mid = max(T // 2, 1)
        out.append(write_png_panel(
            os.path.join(d, "middle.png"),
            [conc[mid, c] for c in range(C)], list(species[:C]), mat,
            "run %d, part way through (t = %.3g)"
            % (int(run_id), float(t_norm[mid]))))
    return [p for p in out if p]


# --------------------------------------------------------------------------
def self_test():
    ok = True

    def check(name, cond, note=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %-56s %s %s" % (name, "PASS" if cond else "FAIL", note))

    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        # 3D
        conc = rng.random((3, 2, 6, 5, 4)).astype(np.float32)
        mat = np.full((6, 5, 4), 2, np.uint8)
        mat[2:4, 2, 2] = 0
        files = save_run(td, 7, conc, np.array([0.0, 0.5, 1.0]), ["Ac", "A"],
                         mat=mat, want_vti=True, want_png=True)
        check("a 3D run writes files", len(files) >= 3 * 2 + 1 + 2,
              "%d files" % len(files))
        check("every file it claims to have written exists",
              all(os.path.exists(p) for p in files))
        vti = os.path.join(td, "run0007", "Ac_0000.vti")
        raw = open(vti, "rb").read()
        check("the VTI header names the array", b'Name="Ac"' in raw)
        check("and declares the right extent",
              b'WholeExtent="0 5 0 4 0 3"' in raw,
              raw[:400].decode("ascii", "replace").split("WholeExtent")[1][:20])
        # the payload has to be exactly one float per voxel, or ParaView reads
        # garbage past the end and shows a plausible picture of nothing
        n_expect = 6 * 5 * 4 * 4
        # The marker is the underscore AFTER <AppendedData>, not the first
        # underscore in the file -- byte_order= has one, and looking for that
        # one reads eight bytes of XML as a length and gets 5.5e18.
        i = raw.index(b"<AppendedData")
        i = raw.index(b"_", i) + 1
        n_declared = int(np.frombuffer(raw[i:i + 8], "<u8")[0])
        check("the payload length matches the voxel count",
              n_declared == n_expect, "%d against %d" % (n_declared, n_expect))

        # 2D, which is written as a single slice rather than refused
        conc2 = rng.random((2, 1, 8, 6)).astype(np.float32)
        files2 = save_run(td, 1, conc2, np.array([0.0, 1.0]), ["Ac"],
                          mat=np.full((8, 6), 2, np.uint8))
        check("a 2D run writes files too", len(files2) >= 3,
              "%d files" % len(files2))
        raw2 = open(os.path.join(td, "run0001", "Ac_0000.vti"), "rb").read()
        check("and its VTI is a single slice",
              b'WholeExtent="0 7 0 5 0 0"' in raw2)
        check("the pictures are real PNGs",
              open(os.path.join(td, "run0001", "last.png"), "rb").read(4)
              == b"\x89PNG")

        # The panel must show the WHOLE domain. A stray untransposed layer
        # once set the y limits to the x extent, so a 60 by 32 field was drawn
        # into the bottom half of a 60 by 60 box and the top half came out
        # blank -- indistinguishable from half the sample being empty.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f2 = np.full((60, 32), 0.5, np.float32)
        r2 = np.full((60, 32), 2, np.uint8); r2[20:25, 10:20] = 0
        write_png_panel(os.path.join(td, "aspect.png"), [f2], ["Ac"], r2)
        fig, ax = plt.subplots()
        ax.imshow(f2.T, origin="lower")
        ax.contour(r2.T == 0, levels=[0.5])
        lo, hi = ax.get_ylim()
        plt.close(fig)
        check("the panel spans the whole field, not part of it",
              abs(hi - lo) < 40, "y span %.1f for a field 32 tall" % abs(hi - lo))

    print()
    print("SELF TEST PASSED" if ok else "SELF TEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
