#!/usr/bin/env python3
"""
_common.py — the bits every test file in this folder needs.

WHAT THIS IS FOR
    The tests in this folder must run from anywhere, on any machine, without the
    person running them having to set PYTHONPATH or know where anything lives.
    This file works out the layout once and everything else imports it.

    It also holds the small helpers for building tiny pore structures by hand.
    "By hand" is the point: a test whose expected answer was produced by the
    same code it is testing proves only that the code is deterministic.
"""

import os
import subprocess
import sys

import numpy as np

# =============================================================================
#  BLOCK 1.  WHERE EVERYTHING IS
#
#  Derived from this file's own location, never from the working directory, so
#  the tests run the same whether you launch them from the project root, from
#  inside tests/, or from your home folder.
# =============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "3D", "tools")
MODEL = os.path.join(ROOT, "3D", "model")
GUI = os.path.join(ROOT, "gui")
BRIDGE = os.path.join(ROOT, "bridge")

for _d in (TOOLS, MODEL, GUI):
    if _d not in sys.path:
        sys.path.insert(0, _d)


# The material codes this project uses, following CompLaB.
SOLID, WALL, PORE = 0, 1, 2


def script(folder, name):
    """The absolute path of one script, checked."""
    p = os.path.join(folder, name)
    if not os.path.exists(p):
        raise AssertionError("this project is missing %s" % p)
    return p


def run(cmd, timeout=300):
    """Run a command and hand back (exit code, everything it printed)."""
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"                 # never try to open a window
    env["PYTHONUNBUFFERED"] = "1"
    try:
        p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return 124, "TIMED OUT after %d seconds" % timeout
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# =============================================================================
#  BLOCK 2.  TINY GEOMETRIES, DRAWN BY HAND
#
#  Every array below is small enough to check with a pencil. That is what makes
#  the unit tests meaningful: the expected answers are worked out from the
#  definition of the quantity, not produced by running the code and writing
#  down whatever came out.
# =============================================================================
def open_channel(nx=12, ny=7, ndim=2):
    """A completely open box with solid walls on the sides only.

    Flow runs along axis 0, which is open at both ends. Every other face is
    solid. This is the simplest thing with a right answer: the pressure falls
    linearly from inlet to outlet, and the widest sphere that fits is set by
    the channel width.
    """
    shape = (nx, ny) if ndim == 2 else (nx, ny, ny)
    m = np.full(shape, PORE, np.uint8)
    m[:, 0] = m[:, -1] = SOLID
    if ndim == 3:
        m[:, :, 0] = m[:, :, -1] = SOLID
    return m


def two_channels(nx=16, ny=11):
    """Two open channels of DIFFERENT widths, separated by a solid block.

    The wide one is five voxels across, the narrow one one voxel across. Both
    run the full length, so both are connected to the inlet, and any descriptor
    that measures pore width has to tell them apart.
    """
    m = np.full((nx, ny), SOLID, np.uint8)
    m[:, 1:6] = PORE           # the wide channel: 5 voxels
    m[:, 9] = PORE             # the narrow channel: 1 voxel
    return m


def bottleneck(nx=14, ny=9):
    """One wide chamber reachable only through a one-voxel throat.

    This is the geometry the whole flow capability exists for. The chamber is
    as wide as the inlet channel, so MIS cannot tell them apart, while UPRM
    must: nothing wider than the throat can be delivered to the chamber.
    """
    m = np.full((nx, ny), SOLID, np.uint8)
    m[0:5, 2:7] = PORE         # the inlet chamber, 5 wide
    m[5:9, 4] = PORE           # the throat, 1 wide
    m[9:nx, 2:7] = PORE        # the far chamber, 5 wide again
    return m


def blocked(nx=10, ny=7):
    """A domain with NO path from inlet to outlet. Nothing should percolate."""
    m = np.full((nx, ny), PORE, np.uint8)
    m[:, 0] = m[:, -1] = SOLID
    m[nx // 2, :] = SOLID      # a wall right across the middle
    return m


def all_solid(nx=6, ny=6):
    """No pore space at all. Every function must survive this without raising."""
    return np.zeros((nx, ny), np.uint8)
