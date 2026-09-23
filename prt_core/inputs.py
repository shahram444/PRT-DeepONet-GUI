#!/usr/bin/env python3
"""
prt_core/inputs.py -- one reader for every geometry file this project accepts.

WHAT CHANGED FROM THE 2D VERSION
    The published release reads one thing: a .npz holding 9472 zeros and ones,
    flat, under a key called "m". Our 3D side grew four more readers in four
    places. predict.py has an if-chain over .h5, .npz, .vti and .dat.
    build_transfer_set_2d_to_3d.py has its own. import_2d_simulations.py has a
    third, and 2D_scripts/prt2d_model.py a fourth. They disagree about what a
    material code means, about whether a flat array is (nx, ny) or (ny, nx), and
    about what to do when the geodesic field is missing.

    They are one function here. Every script calls it, so a file that loads in
    one place loads everywhere, and a file that does not gives the same
    sentence about why wherever it is tried.

WHAT IT READS
    .h5 / .hdf5   a dataset written by build_dataset_2d.py or _3d.py, or by
                  either collector. --geom-index picks the rock. A file that is
                  just one array per dataset is also accepted.
    .npz          build_geometry_3d.py's output (material, gdf, edt), or the
                  published 2D domain, which is flat and has no material codes.
    .npy          one array, the geometry.
    .vti          CompLaB's own geometry echo, so a rock goes from a run folder
                  into a prediction with no conversion step.
    .dat / .txt   raw text. 9472 values is the published 64 by 148 grid and is
                  recognised; anything else needs its shape given.

MATERIAL CODES, WHICH ARE THE THING THAT GOES WRONG
    This project uses CompLaB's codes: 0 solid, 1 wall, 2 pore. The published
    2D release uses 1 for pore and 0 for solid, with no wall. A file holding
    only 0 and 1 is therefore read as the published convention and converted, a
    file holding a 2 is read as ours, and read_geometry says which it decided.
    Guessing silently is how a rock ends up inside out, with every field
    computed over the grains.

WHAT COMES BACK
    A Geometry: material, pore, gdf, edt, ndim, shape, source, note. The gdf is
    computed under a NAMED convention if the file does not carry one, never
    under whichever happened to be in the calling script.

    python -m prt_core.inputs <file>              what is in it
    python -m prt_core.inputs --self-test         check the reader
"""

import os

import numpy as np

try:                                   # the package, when imported normally
    from . import conventions
except ImportError:                    # run as a file, for the self-test
    import conventions

SOLID, WALL, PORE = 0, 1, 2

# The published 2D grid. A flat file of this length is that grid and nothing
# else, which is worth recognising because the release ships 3000 of them.
PUBLISHED_2D = (64, 148)
PUBLISHED_2D_LEN = PUBLISHED_2D[0] * PUBLISHED_2D[1]


class Geometry:
    """One rock, however it arrived.

    material   (nx, ny) or (nx, ny, nz) uint8 in our codes: 0 solid, 1 wall,
               2 pore. Always at least 2D, never flat.
    pore       the boolean mask, computed once so no caller re-derives it
    gdf        the trunk's distance column, under self.convention
    edt        distance to the nearest grain, in voxels, or None
    source     the path it came from
    note       one line saying what was decided while reading, including which
               material convention was recognised and whether the distance had
               to be computed
    """

    def __init__(self, material, gdf=None, edt=None, source="", note="",
                 convention="ours", gid=None):
        self.material = np.asarray(material, np.uint8)
        self.gdf = None if gdf is None else np.asarray(gdf, np.float32)
        self.edt = None if edt is None else np.asarray(edt, np.float32)
        self.source = source
        self.note = note
        self.convention = convention
        self.gid = gid

    # ------------------------------------------------------------- properties
    @property
    def shape(self):
        return tuple(self.material.shape)

    @property
    def ndim(self):
        """2 or 3, by what the rock IS and not by how it is stored.

        A 3D array with a third dimension of 1 is a two-dimensional problem,
        and the encoder, the trunk and the channel list all follow this number.
        Deciding it here rather than in five scripts is the point of the file.
        """
        s = self.shape
        if len(s) == 2:
            return 2
        return 2 if s[2] == 1 else 3

    @property
    def pore(self):
        return self.material == PORE

    @property
    def porosity(self):
        return float(self.pore.mean())

    def as_3d(self):
        """The material with a third axis, for code that wants one shape."""
        m = self.material
        return m if m.ndim == 3 else m[:, :, None]

    def distance(self, convention=None):
        """The distance column, rebuilt if a different convention is asked for.

        Asking for the one it already holds costs nothing. Asking for another
        walks the geometry again, which is cheap and cannot be subtly wrong the
        way scaling the stored column would be.
        """
        want = convention or self.convention
        if self.gdf is not None and want == self.convention:
            return self.gdf
        return conventions.distance_column(self.material, want)

    def __repr__(self):
        return "<Geometry %s %dD porosity %.3f from %s>" % (
            "x".join(str(v) for v in self.shape), self.ndim,
            self.porosity, os.path.basename(self.source) or "memory")


# =============================================================================
#  MATERIAL CODES
# =============================================================================
def to_material_codes(arr):
    """Any plausible geometry array in our codes, and a sentence about how.

    Returns (material, note). The three cases:
        holds a 2          already our codes, taken as they are
        holds only 0 and 1 the published convention, 1 is pore, converted
        anything else      thresholded at the midpoint, and the note says so,
                           because a float geometry is somebody's grey image
                           and the caller should know a threshold was picked
    """
    a = np.asarray(arr)
    if a.dtype == bool:
        return np.where(a, PORE, SOLID).astype(np.uint8), \
            "a boolean mask: True is pore"

    u = np.unique(a)
    if u.size <= 8 and set(u.tolist()) <= {0, 1, 2}:
        if 2 in u.tolist():
            return a.astype(np.uint8), "already in this project's codes (2 is pore)"
        # only 0 and 1: the published convention
        return np.where(a == 1, PORE, SOLID).astype(np.uint8), \
            "the published convention (1 is pore), converted to 2 for pore"

    lo, hi = float(a.min()), float(a.max())
    mid = 0.5 * (lo + hi)
    return np.where(a > mid, PORE, SOLID).astype(np.uint8), \
        ("values from %g to %g, so thresholded at %g. Check this is the right "
         "way round." % (lo, hi, mid))


# =============================================================================
#  THE READERS
# =============================================================================
def _reshape_flat(flat, shape):
    """A flat array onto a grid, with an error that does the arithmetic."""
    n = int(np.asarray(flat).size)
    if shape is not None:
        want = int(np.prod(shape))
        if want != n:
            raise ValueError(
                "this file holds %d values and the shape given is %s, which "
                "needs %d. One of the two is wrong." % (n, tuple(shape), want))
        return np.asarray(flat).reshape(shape)
    if n == PUBLISHED_2D_LEN:
        return np.asarray(flat).reshape(PUBLISHED_2D)
    raise ValueError(
        "this file holds %d values in one flat array and does not say what "
        "grid they are. Pass the shape. (%d values would be the published 2D "
        "grid, %d by %d, and is recognised without being told.)"
        % (n, PUBLISHED_2D_LEN, PUBLISHED_2D[0], PUBLISHED_2D[1]))


def _read_h5(path, index=0):
    import h5py
    with h5py.File(path, "r") as h:
        if "geom" in h and "material" in h["geom"]:
            g = h["geom"]
            n = int(g["material"].shape[0])
            i = int(index)
            if not -n <= i < n:
                raise IndexError(
                    "index %d, but %s holds %d geometries (0 to %d)"
                    % (i, os.path.basename(path), n, n - 1))
            mat = np.asarray(g["material"][i])
            gdf = np.nan_to_num(np.asarray(g["gdf"][i])) if "gdf" in g else None
            edt = np.asarray(g["edt"][i]) if "edt" in g else None
            gid = int(np.asarray(g["gid"][i])) if "gid" in g else i
            return mat, gdf, edt, gid, "geometry %d of %d (gid %d)" % (i % n, n, gid)

        # not one of ours: take the first array that looks like a grid
        cands = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3, 4):
                cands.append((name, obj.shape))
        h.visititems(visit)
        if not cands:
            raise ValueError(
                "%s has no /geom/material and no array of 2, 3 or 4 dimensions, "
                "so there is nothing in it that could be a rock."
                % os.path.basename(path))
        name = sorted(cands)[0][0]
        a = np.asarray(h[name])
        if a.ndim == 4:
            a = a[int(index)]
        return a, None, None, None, "read %s, which is not this project's layout" % name


def _read_npz(path, shape=None):
    z = np.load(path, allow_pickle=False)
    keys = list(z.files)
    if "material" in keys:
        mat = z["material"]
        gdf = np.nan_to_num(z["gdf"]) if "gdf" in keys else None
        edt = z["edt"] if "edt" in keys else None
        return mat, gdf, edt, "material, and %s" % ", ".join(
            k for k in ("gdf", "edt") if k in keys) if (
            "gdf" in keys or "edt" in keys) else "material only"
    # the published domains: one flat array, usually under "m"
    key = "m" if "m" in keys else keys[0]
    a = np.asarray(z[key])
    if a.ndim == 1:
        a = _reshape_flat(a, shape)
    return a, None, None, "one array, %r" % key


def _read_vti(path):
    """CompLaB's geometry echo, through the project's own .vti reader."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    tools = os.path.join(os.path.dirname(here), "3D", "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from collect_complab_output import read_vti          # noqa: E402
    arrays, dims = read_vti(path)
    if not arrays:
        raise ValueError("no arrays in %s" % os.path.basename(path))
    name = list(arrays)[0]
    return np.asarray(arrays[name]), "the %r array" % name


def _read_text(path, shape=None):
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        return _reshape_flat(raw, shape), "a flat text file"
    if shape is not None and tuple(raw.shape) != tuple(shape):
        return _reshape_flat(raw.ravel(), shape), "text, reshaped as asked"
    return raw, "text, %s as written" % "x".join(str(v) for v in raw.shape)


# =============================================================================
#  THE ENTRY POINT
# =============================================================================
def read_geometry(path, index=0, shape=None, convention="ours",
                  want_edt=False, quiet=True):
    """Any geometry file this project accepts, as a Geometry.

    path        .h5, .hdf5, .npz, .npy, .vti, .dat, .txt or no extension
    index       which rock, for a file holding several
    shape       the grid, for a flat file that does not say
    convention  which distance convention to compute the gdf under if the file
                does not carry one. NAMED, never assumed: see conventions.py
    want_edt    also compute the wall distance if the file has none. Needs
                scipy, so it is off unless asked for

    The note on the returned Geometry says every decision that was made, and
    quiet=False prints it. Nothing about the read is silent.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s is not there. A geometry can be a .h5 dataset, a .npz from "
            "build_geometry_3d.py, a published 2D domain, a CompLaB .vti or a "
            "raw .dat with the shape given." % path)

    low = path.lower()
    gdf = edt = gid = None
    if low.endswith((".h5", ".hdf5")):
        raw, gdf, edt, gid, how = _read_h5(path, index)
    elif low.endswith(".npz"):
        raw, gdf, edt, how = _read_npz(path, shape)
    elif low.endswith(".npy"):
        raw = np.load(path)
        if raw.ndim == 1:
            raw = _reshape_flat(raw, shape)
        how = "one .npy array"
    elif low.endswith(".vti"):
        raw, how = _read_vti(path)
    else:
        raw, how = _read_text(path, shape)

    if raw.ndim == 1:
        raw = _reshape_flat(raw, shape)
    if raw.ndim not in (2, 3):
        raise ValueError(
            "%s gave an array of %d dimensions, %s. A geometry is 2 or 3."
            % (os.path.basename(path), raw.ndim, tuple(raw.shape)))

    mat, code_note = to_material_codes(raw)
    notes = [how, code_note]

    if gdf is not None and gdf.shape != mat.shape:
        notes.append("the stored gdf is %s and the rock is %s, so it was "
                     "recomputed" % (tuple(gdf.shape), tuple(mat.shape)))
        gdf = None
    if gdf is None:
        gdf = conventions.distance_column(mat, convention)
        notes.append("distance computed here, convention %r: %s"
                     % (convention, conventions.describe(convention)))
    else:
        notes.append("distance taken from the file, assumed to be %r"
                     % convention)

    if edt is None and want_edt:
        from scipy import ndimage
        edt = ndimage.distance_transform_edt(mat == PORE).astype(np.float32)
        notes.append("wall distance computed here")

    g = Geometry(mat, gdf, edt, source=path, note="; ".join(notes),
                 convention=convention, gid=gid)
    if not quiet:
        print("  %s" % g)
        for n in notes:
            print("    %s" % n)
    return g


def add_arguments(ap):
    """The standard flags, so every script spells them the same way."""
    ap.add_argument("--geometry", required=True,
                    help="a .h5 dataset, a .npz, a .npy, a CompLaB .vti or a "
                         "raw .dat")
    ap.add_argument("--geom-index", type=int, default=0,
                    help="which rock, for a file holding several")
    ap.add_argument("--geom-shape", type=int, nargs="+", default=None,
                    help="the grid, for a flat file that does not say it")
    ap.add_argument("--distance-convention", default="ours",
                    choices=list(conventions.DISTANCE_CONVENTIONS),
                    help="how to scale the trunk's distance column when the "
                         "file does not carry one. 'ours' is zero at the inlet, "
                         "the published two are one there")
    return ap


def from_arguments(a, want_edt=False, quiet=False):
    """A Geometry from the flags add_arguments put on the parser."""
    return read_geometry(a.geometry, index=getattr(a, "geom_index", 0),
                         shape=getattr(a, "geom_shape", None),
                         convention=getattr(a, "distance_convention", "ours"),
                         want_edt=want_edt, quiet=quiet)


# =============================================================================
#  SELF TEST
# =============================================================================
def _self_test():
    import tempfile
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print("  %-5s %-58s %s" % ("PASS" if cond else "FAIL", name, extra))

    d = tempfile.mkdtemp()

    # a rock in our own codes, 3D
    m3 = np.zeros((16, 10, 8), np.uint8)
    m3[:, 2:8, 2:6] = PORE
    p = os.path.join(d, "rock.npz")
    np.savez_compressed(p, material=m3)
    g = read_geometry(p)
    check("a .npz in this project's codes keeps them",
          g.material.max() == PORE and g.ndim == 3, g.note.split(";")[1].strip())
    check("and gets a distance column computed for it",
          g.gdf is not None and float(g.gdf.max()) > 0.0)

    # the published 2D domain: flat, 9472 values, 1 is pore
    flat = np.zeros(PUBLISHED_2D_LEN, np.int32)
    flat.reshape(PUBLISHED_2D)[:, 3:] = 1
    p2 = os.path.join(d, "Domain_test.npz")
    np.savez_compressed(p2, m=flat)
    g2 = read_geometry(p2)
    check("a published flat domain is recognised without being told its grid",
          g2.shape == PUBLISHED_2D, str(g2.shape))
    check("and its 1 becomes this project's 2",
          int(g2.material.max()) == PORE and g2.porosity > 0.9,
          "porosity %.3f" % g2.porosity)
    check("a 2D rock reports 2 dimensions", g2.ndim == 2)

    # the same thing as raw text, which is what Input_domains.zip holds
    p3 = os.path.join(d, "geometry.dat")
    np.savetxt(p3, flat.reshape(PUBLISHED_2D), fmt="%d")
    g3 = read_geometry(p3)
    check("the same rock as a .dat reads the same",
          np.array_equal(g3.material, g2.material))

    # a flat .dat of an unknown length says so rather than guessing
    p4 = os.path.join(d, "odd.dat")
    np.savetxt(p4, np.ones(77, np.int32))
    check("a flat file of an unfamiliar length asks for its shape",
          "Pass the shape" in _err(lambda: read_geometry(p4)))
    g4 = read_geometry(p4, shape=(7, 11))
    check("and reads once it is given", g4.shape == (7, 11))

    # a 3D array with a flat third axis is a 2D problem
    m2in3 = np.full((12, 9, 1), PORE, np.uint8)
    m2in3[4:6, 3:5, 0] = SOLID
    p5 = os.path.join(d, "flat3d.npz")
    np.savez_compressed(p5, material=m2in3)
    g5 = read_geometry(p5)
    check("a rock stored as 3D with nz = 1 is a 2D problem",
          g5.ndim == 2 and len(g5.shape) == 3, str(g5.shape))

    # the convention is honoured and is not the stored one by accident
    a = read_geometry(p, convention="ours").gdf
    b = read_geometry(p, convention="published").gdf
    pore = m3 == PORE
    c = float(np.corrcoef(a[pore], b[pore])[0, 1])
    check("the two conventions really do come out opposite",
          c < -0.999, "correlation %.4f" % c)
    check("and asking a Geometry for the other one rebuilds it",
          np.allclose(read_geometry(p).distance("published"), b, atol=1e-6))

    # a boolean mask
    m6, note6 = to_material_codes(pore)
    check("a boolean mask converts and says so",
          int(m6.max()) == PORE and np.array_equal(m6 == PORE, pore), note6)

    # an .h5 in this project's layout
    try:
        import h5py
        p7 = os.path.join(d, "ds.h5")
        with h5py.File(p7, "w") as h:
            gr = h.create_group("geom")
            gr.create_dataset("material", data=np.stack([m3, m3[::-1]]))
            gr.create_dataset("gid", data=np.array([11, 12]))
        g7 = read_geometry(p7, index=1)
        check("an .h5 dataset gives the rock that was asked for",
              g7.gid == 12 and np.array_equal(g7.material, m3[::-1]))
        check("and an index past the end says how many there are",
              "holds 2 geometries" in _err(lambda: read_geometry(p7, index=9)))
    except ImportError:
        check("h5py is installed", False, "skipped: h5py not available")

    # a missing file
    check("a missing file lists what a geometry can be",
          "CompLaB" in _err(lambda: read_geometry(os.path.join(d, "nope.npz"))))

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
    ap.add_argument("path", nargs="?", help="a geometry file to describe")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--shape", type=int, nargs="+", default=None)
    ap.add_argument("--convention", default="ours",
                    choices=list(conventions.DISTANCE_CONVENTIONS))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(_self_test())
    if not a.path:
        ap.print_help()
        sys.exit(0)
    read_geometry(a.path, index=a.index, shape=a.shape,
                  convention=a.convention, quiet=False)
