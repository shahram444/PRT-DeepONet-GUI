#!/usr/bin/env python3
"""Read a results file written by make_results_h5.py.

    python3 read_results_h5.py prt_results.h5                 the summary
    python3 read_results_h5.py prt_results.h5 --tree          what is in it
    python3 read_results_h5.py prt_results.h5 --notes         the notes
    python3 read_results_h5.py prt_results.h5 --table C test per_run
    python3 read_results_h5.py prt_results.h5 --csv out/      every table as csv
    python3 read_results_h5.py prt_results.h5 --figures out/  the embedded pngs

Nothing here is required to use the file: h5py alone opens it, and the layout is
in /notes/overview. This is a convenience, and a worked example of the three or
four lines that answer most questions.
"""
import argparse
import csv
import os
import sys

import numpy as np

try:
    import h5py
except ImportError:                                   # pragma: no cover
    sys.exit("h5py is required: pip install h5py")


def _s(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray) and v.dtype.kind == "S":
        return [x.decode("utf-8", "replace") for x in v.tolist()]
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def summary(f):
    print("%s" % _s(f.attrs.get("title", "")))
    print("%s" % ("=" * 72))
    for k in ("campaign", "created_utc", "dim", "grid", "spacing",
              "species", "reaction", "n_models", "n_runs_in_dataset",
              "n_pore_spaces"):
        if k in f.attrs:
            print("  %-20s %s" % (k, _s(f.attrs[k])))
    print("  %-20s %s" % ("dataset", _s(f.attrs.get("dataset_file", ""))))

    if "results" not in f:
        return
    print("\nACCURACY, as fractions of the feed concentration")
    print("  %-8s %-6s %10s %9s %11s %10s %12s"
          % ("model", "split", "rmse", "r2", "bias", "mae", "n"))
    for tag in f["results"]:
        for split in ("train", "val", "test"):
            p = "results/%s/%s/pooled" % (tag, split)
            if p not in f:
                continue
            g = f[p]
            print("  %-8s %-6s %10.5f %9.4f %+11.6f %10.5f %12d"
                  % (tag, split, g["rmse_feed"][()], g["r2"][()],
                     g["bias_feed"][()], g["mae_feed"][()], g["n"][()]))

    for tag in f["results"]:
        p = "results/%s/test/per_snapshot_summary" % tag
        if p in f and np.isfinite(f[p]["r2_mean"][()]):
            a = f[p]["r2_mean"][()]
            b = f["results/%s/test/pooled/r2" % tag][()]
            if abs(a - b) > 0.5:
                print("\n  note: for %s the MEAN of the per snapshot r2 is "
                      "%.1f while the pooled r2 is %.4f." % (tag, a, b))
                print("        Quote the pooled one. See /notes/metric_definitions.")
                break

    if "comparisons/geometry_ablation/pooled" in f:
        g = f["comparisons/geometry_ablation/pooled"]
        print("\nGEOMETRY ABLATION, held out, rmse as a fraction of the feed")
        for m, r, r2 in zip(_s(g["model"][:]), g["rmse_feed"][:], g["r2"][:]):
            print("  %-10s %.5f   r2 %.4f" % (m, r, r2))
        st = f.get("comparisons/geometry_ablation/sign_test")
        if st is not None:
            print("  the geodesic distance wins on %d of %d pore spaces, "
                  "p = %.4f" % (st["pore_spaces_won"][()],
                                st["pore_spaces_total"][()],
                                st["p_one_sided"][()]))

    if "comparisons/speed" in f:
        g = f["comparisons/speed"]
        if "speedup_honest" in g:
            print("\nSPEED")
            print("  one simulation            %.0f s"
                  % g["simulation_seconds_median"][()])
            print("  predicting one whole case %.3f s"
                  % g["seconds_one_case_all_species_all_times"][()])
            print("  speed-up                  %.0f x   on %s"
                  % (g["speedup_honest"][()], _s(g.attrs.get("device", "?"))))
            if "INCOMPLETE" in g.attrs:
                print("  WARNING: %s" % _s(g.attrs["INCOMPLETE"]))


def tree(f, maxdepth=4):
    def walk(g, d=0):
        if d > maxdepth:
            return
        for k in g:
            o = g[k]
            if isinstance(o, h5py.Group):
                n = o.attrs.get("n_rows")
                print("%s%s/%s" % ("  " * d, k, "" if n is None else "  (%d rows)" % n))
                walk(o, d + 1)
            else:
                print("%s%-28s %-16s %s" % ("  " * d, k, str(o.shape), o.dtype))
    walk(f)


def notes(f, which=None):
    for k in f["notes"]:
        if which and k != which:
            continue
        print("\n%s\n%s\n%s" % (k.upper(), "=" * 72, _s(f["notes"][k][()])))


def show_table(f, path):
    g = f[path]
    cols = _s(g.attrs["columns"])
    data = {c: g[c][:] for c in cols}
    print("  ".join("%-12s" % c for c in cols))
    n = g.attrs["n_rows"]
    for i in range(min(n, 40)):
        row = []
        for c in cols:
            v = data[c][i]
            row.append("%-12.5g" % v if isinstance(v, (float, np.floating))
                       else "%-12s" % _s(v))
        print("  ".join(row))
    if n > 40:
        print("... %d more rows" % (n - 40))


def dump_csv(f, out):
    os.makedirs(out, exist_ok=True)
    n = 0
    def visit(name, obj):
        nonlocal n
        if isinstance(obj, h5py.Group) and "columns" in obj.attrs:
            cols = _s(obj.attrs["columns"])
            rows = zip(*[obj[c][:] for c in cols])
            p = os.path.join(out, name.replace("/", "__") + ".csv")
            with open(p, "w", newline="") as fh:
                w = csv.writer(fh)
                if "note" in obj.attrs:
                    fh.write("# %s\n" % _s(obj.attrs["note"]))
                w.writerow(cols)
                for r in rows:
                    w.writerow([_s(x) for x in r])
            n += 1
    f.visititems(visit)
    print("wrote %d csv files to %s" % (n, out))


def dump_figures(f, out):
    if "figures" not in f:
        print("this file carries no figures")
        return
    os.makedirs(out, exist_ok=True)
    for k in f["figures"]:
        p = os.path.join(out, k.replace("__", os.sep))
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(f["figures"][k][:].tobytes())
    print("wrote %d figures to %s" % (len(f["figures"]), out))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("file")
    p.add_argument("--tree", action="store_true")
    p.add_argument("--notes", nargs="?", const="", default=None)
    p.add_argument("--table", nargs=3, metavar=("TAG", "SPLIT", "NAME"))
    p.add_argument("--csv", metavar="DIR")
    p.add_argument("--figures", metavar="DIR")
    a = p.parse_args(argv)

    f = h5py.File(a.file, "r")
    if _s(f.attrs.get("schema", "")) != "prt-results":
        print("warning: this does not declare itself a prt-results file")
    if a.tree:
        tree(f)
    elif a.notes is not None:
        notes(f, a.notes or None)
    elif a.table:
        show_table(f, "results/%s/%s/%s" % tuple(a.table))
    elif a.csv:
        dump_csv(f, a.csv)
    elif a.figures:
        dump_figures(f, a.figures)
    else:
        summary(f)
    f.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:        # piping into head is not an error
        try:
            sys.stdout.close()
        finally:
            os._exit(0)
