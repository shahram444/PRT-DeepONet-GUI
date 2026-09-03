#!/usr/bin/env python3
"""
run_ablation_sweep.py — run the experiment the three switches exist to settle.

Trains one model per configuration, evaluates them all on the SAME held-out
geometries, and writes a comparison table plus a per-Peclet breakdown.  That
breakdown is the point: the honest prediction is that the flow-proxy switch wins
at high Peclet and LOSES at Pe = 0.3, because a stagnant velocity field carries
no information about dead-end pores while the geodesic field still does.  This
script measures that rather than arguing about it.

    python run_ablation_sweep.py --data /data/dataset_reader.h5 --out ./sweep

    # with 2D transfer as well
    python run_ablation_sweep.py --data 3d.h5 --data-2d train2d.h5 --out ./sweep

    # a fast smoke test on the tiny synthetic file
    python build_practice_dataset.py --out /tmp/test3d.h5          # in ../tools
    python run_ablation_sweep.py --data /tmp/test3d.h5 --out /tmp/sweep --quick

WHAT EACH ROW IS
----------------
  baseline        all switches off: trunk (x,y,z,t,gdf).  The reference.
  edt             the existing ablation control: Euclidean instead of geodesic.
  flow-tau        SWITCH A. trunk (x,y,z,t,tau), branch sees the velocity field.
  flow-both       SWITCH A with gdf AND tau in the trunk. The safe version.
  dim-free        SWITCH C. trunk (t,dwall,tau). The one that transfers 2D->3D.
  dim-free+2D     SWITCH C fed extra extruded-2D training data (needs --data-2d).
  dim-free+2Dpre  SWITCH C pretrained on 2D, fine-tuned on 3D with a frozen trunk.

The last two are what answer "how few 3D geometries can we get away with".
"""

import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        raise SystemExit("command failed: %s" % " ".join(cmd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the 3D dataset")
    ap.add_argument("--data-2d", default=None,
                    help="an extruded-2D dataset from tools/build_transfer_set_2d_to_3d.py")
    ap.add_argument("--out", default="./sweep")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-points", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--transfer-2d-frac", type=float, default=0.3)
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these configurations by name")
    ap.add_argument("--quick", action="store_true",
                    help="3 epochs, tiny batches: a plumbing check, not science")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse checkpoints already in --out and only evaluate")
    a = ap.parse_args()

    # =========================================================================
    # BLOCK 1.  ONE BASE COMMAND, ONE EXTRA LIST PER CONFIGURATION
    #
    # Every run below is train.py with the SAME base arguments and a different
    # tail. That is the whole design: a comparison is only a comparison if the
    # only thing that differs between the rows is the thing being compared, and
    # building each command by hand is how that stops being true.
    # =========================================================================
    if a.quick:
        # Three epochs on 512 points settles nothing about accuracy. It answers
        # the only question --quick is for: does the whole sweep run to the end?
        a.epochs, a.batch_size, a.n_points, a.workers = 3, 2, 512, 0

    os.makedirs(a.out, exist_ok=True)
    base = [sys.executable, os.path.join(HERE, "train.py"), "--data", a.data,
            "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
            "--n-points", str(a.n_points), "--workers", str(a.workers)]

    configs = [
        ("baseline", []),
        ("edt", ["--distance", "edt"]),
        ("flow-tau", ["--flow-proxy"]),
        ("flow-both", ["--flow-proxy", "--flow-mode", "both"]),
        ("dim-free", ["--dim-free"]),
    ]
    # The 2D rows need a transfer set. Without one they are LEFT OUT rather than
    # run on nothing, so a missing file costs you rows in the table and not a
    # row of meaningless numbers.
    if a.data_2d:
        configs += [
            ("dim-free+2D", ["--dim-free", "--transfer-2d", a.data_2d,
                             "--transfer-2d-frac", str(a.transfer_2d_frac)]),
        ]
    if a.only:
        configs = [c for c in configs if c[0] in a.only]

    for name, extra in configs:
        d = os.path.join(a.out, name)
        if a.skip_train and os.path.exists(os.path.join(d, "best.pt")):
            print("skipping %s (checkpoint exists)" % name)
            continue
        run(base + ["--out", d] + extra)

    # ---- the two-stage form of switch B: pretrain on 2D, fine-tune on 3D ----
    if a.data_2d and (not a.only or "dim-free+2Dpre" in a.only):
        pre = os.path.join(a.out, "_pretrain2d")
        if not (a.skip_train and os.path.exists(os.path.join(pre, "best.pt"))):
            run([sys.executable, os.path.join(HERE, "train.py"),
                 "--data", a.data_2d, "--out", pre, "--dim-free",
                 "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
                 "--n-points", str(a.n_points), "--workers", str(a.workers)])
        d = os.path.join(a.out, "dim-free+2Dpre")
        if not (a.skip_train and os.path.exists(os.path.join(d, "best.pt"))):
            run(base + ["--out", d, "--dim-free",
                        "--init-from", os.path.join(pre, "best.pt"), "--freeze-trunk"])
        configs.append(("dim-free+2Dpre", []))

    # ---- evaluate everything on the same held-out geometries ---------------
    # =========================================================================
    # BLOCK 2.  EVALUATE THEM ALL AT ONCE, ON THE SAME ROCKS
    #
    # One evaluate.py call over every checkpoint, not one call each. The split
    # is derived from the dataset and the seed, so separate calls would agree
    # anyway, but a single call cannot silently stop agreeing.
    #
    # A configuration whose training failed has no best.pt and is dropped here
    # rather than crashing the comparison; the table then shows which rows are
    # missing, which is the useful failure.
    # =========================================================================
    pairs = ["%s=%s" % (n, os.path.join(a.out, n, "best.pt")) for n, _ in configs
             if os.path.exists(os.path.join(a.out, n, "best.pt"))]
    run([sys.executable, os.path.join(HERE, "evaluate.py"), "--data", a.data,
         "--compare"] + pairs + ["--out", os.path.join(a.out, "compare"),
                                 "--save-fields", "2", "--no-3d"])

    # ---- the breakdown that actually answers the question ------------------
    summarise(os.path.join(a.out, "compare", "rmse_table.csv"), a.out)


def summarise(csv_path, out):
    """Mean RMSE per configuration, overall and split by Peclet decade."""
    import csv as _csv
    if not os.path.exists(csv_path):
        print("no rmse_table.csv to summarise")
        return
    rows = list(_csv.DictReader(open(csv_path)))
    if not rows:
        return
    # =========================================================================
    # BLOCK 3.  THE PECLET BREAKDOWN
    #
    # A single average hides the answer. The claim being tested is that the flow
    # field wins at high Peclet and loses at low, so the number that settles it
    # is the split, not the mean.
    # =========================================================================
    # Found by name, case insensitively. A dataset with no Pe column still gets
    # the overall table rather than an exception.
    pe_key = next((k for k in rows[0] if k.lower() == "pe"), None)
    tags = sorted({r["tag"] for r in rows})

    def mean(rs):
        v = [float(r["rmse_mean"]) for r in rs]
        # NaN for an empty band, never 0.0, which would print as the best score
        # in the table for a band that holds no runs at all.
        return sum(v) / len(v) if v else float("nan")

    lines = []
    lines.append("%-16s %10s" % ("configuration", "RMSE all"))
    bands = []
    if pe_key:
        bands = [("Pe<1", lambda p: p < 1.0), ("1<=Pe<10", lambda p: 1.0 <= p < 10.0),
                 ("Pe>=10", lambda p: p >= 10.0)]
        lines[0] += "".join("%12s" % b[0] for b in bands)
    for t in tags:
        rs = [r for r in rows if r["tag"] == t]
        line = "%-16s %10.4f" % (t, mean(rs))
        for _, f in bands:
            line += "%12.4f" % mean([r for r in rs if f(float(r[pe_key]))])
        lines.append(line)

    txt = "\n".join(lines)
    print("\n" + "=" * 70)
    print("SWITCH SWEEP RESULTS   (lower is better; the 2D paper's bar was 0.04)")
    print("=" * 70)
    print(txt)
    print("""
HOW TO READ THE PECLET COLUMNS
  The flow-proxy rows are expected to beat the baseline in the Pe>=10 column and
  to LOSE in the Pe<1 column.  At low Peclet the solute arrives by diffusion, the
  velocity field is near zero in the dead-end pores that matter, and tau carries
  no information there while the geodesic field still does.  If flow-both is not
  at least as good as the better of baseline and flow-tau in EVERY column, the
  extra trunk column is not being used and something is wrong.""")
    with open(os.path.join(out, "sweep_summary.txt"), "w") as f:
        f.write(txt + "\n")
    print("\nwrote %s" % os.path.join(out, "sweep_summary.txt"))


if __name__ == "__main__":
    main()
