#!/usr/bin/env python3
"""
build_transfer_set.py — 2D/ -> a 3D training file, for switch B.

A thin wrapper over the transfer-set builder that knows where the two halves of
this repository live, so no paths have to be typed.  Run it from anywhere:

    python build_transfer_set.py                     # 200 domains, 128x64x64
    python build_transfer_set.py --limit 1000        # more domains
    python build_transfer_set.py --target-shape 128 64 64 --n-sets 4

Output goes to bridge/train2d.h5 by default.  Feed it to training with:

    python ../3D/model/train.py --data <3d.h5> \\
           --transfer-2d train2d.h5 --transfer-2d-frac 0.3 --dim-free

WHAT IT COSTS
-------------
The extruded medium is prismatic, so the flow and transport are solved on a thin
z-slab (--nz-solve, default 8) and tiled up to the full nz.  That is exact, not
interpolated, and it makes the ingest about eight times cheaper.  Measured on a
128 x 64 x 64 target: roughly ten seconds per domain at 300 Stokes iterations,
so 200 domains is about half an hour on one core and trivially parallelisable by
splitting --limit across jobs and merging.

Compare that with the 3D campaign it is meant to reduce: 5.5 to 16 hours PER RUN.
"""

import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TWO_D = os.path.join(ROOT, "2D")
# AUDIT BRIDGE-ROOT-01A. This pointed at 3D/tools/ingest_2d.py, which was
# renamed to build_transfer_set_2d_to_3d.py, so the wrapper exited on its own
# existence check and the GUI button that calls it did nothing. The older name
# is still tried, so a checkout that predates the rename keeps working.
INGEST = os.path.join(ROOT, "3D", "tools", "build_transfer_set_2d_to_3d.py")
INGEST_OLD = os.path.join(ROOT, "3D", "tools", "ingest_2d.py")
if not os.path.exists(INGEST) and os.path.exists(INGEST_OLD):
    INGEST = INGEST_OLD


# =============================================================================
#  EVERY ARGUMENT IS PASSED THROUGH
#  This file adds paths and defaults and nothing else. It does not interpret the
#  builder's flags, so anything that works here works when the builder is called
#  directly, which is what makes the two impossible to drift apart.
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "train2d.h5"))
    ap.add_argument("--limit", type=int, default=200,
                    help="how many 2D domains to use. 2D/Domains holds 3000.")
    ap.add_argument("--target-shape", type=int, nargs=3, default=[128, 64, 64],
                    help="MUST match the 3D dataset exactly or train.py refuses")
    ap.add_argument("--n-sets", type=int, default=3,
                    help="parameter sets (Pe, Da) per domain")
    ap.add_argument("--n-times", type=int, default=21)
    ap.add_argument("--n-species", type=int, default=4)
    ap.add_argument("--params", choices=["pe_da", "full"], default="pe_da",
                    help="which dimensionless groups the transfer set records. "
                         "Must match the 3D dataset it is mixed into.")
    ap.add_argument("--species", nargs="*", default=None,
                    help="species names; must match the 3D file")
    ap.add_argument("--nz-solve", type=int, default=8)
    ap.add_argument("--stokes-iters", type=int, default=1200)
    ap.add_argument("--adr-steps", type=int, default=200)
    ap.add_argument("--interface", choices=["pore", "solid"], default="pore")
    ap.add_argument("--synthetic", type=int, default=None,
                    help="skip 2D/ and synthesise N domains of the same "
                         "morphology instead. Useful to test the pipeline.")
    a = ap.parse_args()          # nothing below reads sys.argv again

    if not os.path.exists(INGEST):
        sys.exit("cannot find the transfer-set builder.\n"
                 "  looked for : %s\n"
                 "  and also   : %s\n"
                 "Is this script still inside PRT-DeepONet/bridge/?"
                 % (INGEST, INGEST_OLD))
    if a.synthetic is None and not os.path.isdir(TWO_D):
        sys.exit("cannot find %s\nUse --synthetic N to build a set without it."
                 % TWO_D)

    # ---------------------------------------------------------------------
    #  THE COMMAND, PRINTED BEFORE IT RUNS
    #  So that a run started from the window can be repeated by hand on a
    #  cluster, by copying the line rather than reconstructing it.
    # ---------------------------------------------------------------------
    cmd = [sys.executable, INGEST, "--out", a.out,
           "--target-shape", *map(str, a.target_shape),
           "--n-sets", str(a.n_sets), "--n-times", str(a.n_times),
           "--n-species", str(a.n_species), "--nz-solve", str(a.nz_solve),
           "--params", a.params,
           "--stokes-iters", str(a.stokes_iters), "--adr-steps", str(a.adr_steps),
           "--interface", a.interface]
    if a.species:
        # Only passed when given: the builder's own default has to match the 3D
        # file, and repeating it here would be a second place to keep in step.
        cmd += ["--species", *a.species]
    if a.synthetic is not None:
        # --synthetic builds domains of the same morphology instead of reading
        # the released ones, so the pipeline can be checked without 2D/ present.
        cmd += ["--synthetic", str(a.synthetic)]
    else:
        cmd += ["--jung-dir", TWO_D, "--limit", str(a.limit)]

    print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        sys.exit(r.returncode)

    print("\nnext:")            # the command that consumes what was just built
    print("  python %s --data <3d.h5> --transfer-2d %s --transfer-2d-frac 0.3 --dim-free"
          % (os.path.join(ROOT, "3D", "model", "train.py"), a.out))


if __name__ == "__main__":
    main()
