#!/usr/bin/env python3
"""
prt_core -- the four things every script in this project needs, in one place.

WHY THIS PACKAGE EXISTS
    The repository grew outwards from three published notebooks, and each new
    script carried its own copy of whatever it needed from them. By the time
    the audit was written there were three copies of the network, four readers
    for a geometry file, two distance conventions with no name for either, and
    the chemistry hard-wired as constructor arguments in five places.

    None of that is a bug on its own. The bugs are what the copies allow: a
    warm start that hands a trained trunk a distance column running the other
    way; a geometry read as inside out by one script and correctly by another;
    a checkpoint that cannot say which chemistry it is for, so the wrong one
    loads and merely predicts badly.

    Four files fix the class of problem rather than the instances:

    reactions.py    which chemistry, as data. Species, dimensionless numbers
                    with their trained ranges, steady or transient. Adding one
                    is a table entry or a .json, not a new script.
    conventions.py  the distance column, and the two ways it is scaled. Named,
                    recorded in the checkpoint, converted by one function.
    inputs.py       one reader for .h5, .npz, .npy, .vti and .dat, which also
                    decides 2D or 3D from the rock rather than from a flag.
    model.py        the network, defined once, in both key layouts, built from
                    a reaction rather than from typed-in widths.

WHAT USES IT
    Everything. 3D/model/deeponet_model.py and 2D_scripts/prt2d_model.py
    re-export from model.py, so every existing import still works and there is
    one definition behind them. train.py, evaluate.py and predict.py take
    --reaction and --distance-convention from here and record both.

CHECK IT
    python prt_core/test_core.py
    python -m prt_core.reactions --self-test
    python -m prt_core.conventions --self-test
    python -m prt_core.inputs --self-test
    python -m prt_core.model --self-test
"""

# =============================================================================
#  THE SUBMODULES, AND THE NAMES WORTH HAVING AT THE TOP
#  Both are exported: `from prt_core import conventions` for the whole module,
#  and `from prt_core import read_geometry` for the handful of names a script
#  actually calls. Importing the package imports all four, which is cheap: none
#  of them touches a file or builds a network at import time.
# =============================================================================
from . import conventions, inputs, model, reactions
# ---------------------------------------------------------------------------
#  THE ORDER MATTERS ONE PLACE ONLY
#  model.py imports reactions.py, and inputs.py imports conventions.py. Both
#  are written to work when run as plain files too, for their self-tests, so
#  neither depends on this package being imported first.
# ---------------------------------------------------------------------------
from .conventions import (DISTANCE_CONVENTIONS, convert, distance_column,
                          pore_mask)                      # the distance column
from .inputs import Geometry, read_geometry               # every geometry format
from .model import PRT2D, PRT_DeepONet3D, build, count_parameters, load_published
from .reactions import REACTIONS, Reaction, get, resolve  # which chemistry

# Named explicitly, so a star import gives the four modules and the handful of
# functions above, and not every name they happen to import themselves.
__all__ = [
    "conventions", "inputs", "model", "reactions",
    "DISTANCE_CONVENTIONS", "distance_column", "convert", "pore_mask",
    "Geometry", "read_geometry",
    "PRT_DeepONet3D", "PRT2D", "build", "load_published", "count_parameters",
    "REACTIONS", "Reaction", "get", "resolve",
]

# Bumped with the repository's tag. Version 3 is the one that introduced this
# package; a checkpoint written by an earlier version has no reaction recorded
# in it, and everything here is written to keep working on those.
__version__ = "3.0.0"
