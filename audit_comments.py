#!/usr/bin/env python3
"""
audit_comments.py — is every file in this version actually commented?

WHY THIS EXISTS
    The rule for version 1.2 is that anyone opening a file can tell what is
    going on without reading the code:

      1  a FIRST BLOCK at the top of every new or changed file saying what
         changed from the 2D version and where its code came from
      2  BLOCK COMMENTS marking the sections inside it
      3  LINE COMMENTS on the parts that look wrong until explained

    A rule nobody checks decays on the first busy afternoon. This checks it,
    file by file, and prints what is missing rather than a pass or a fail.

WHAT IT DOES NOT DO
    It cannot judge whether a comment is any good. It counts and it locates.
    A file can pass every check here and still be badly explained; what it
    catches is the file that was edited and never documented at all.

    python audit_comments.py
    python audit_comments.py --quiet     only the files with something missing
"""

import argparse
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The files the flow work created or touched. Every one of these must carry the
# top-of-file block. Kept explicit rather than discovered, so that adding a file
# to the version without documenting it FAILS here instead of being ignored.
FLOW_NEW = [
    "3D/tools/flow_features.py",
    "3D/tools/harmonic_pressure.py",
    "3D/tools/add_flow_features.py",
    "3D/model/velocity_model.py",
    "3D/model/train_velocity.py",
    "3D/model/predict_velocity.py",
    "3D/model/test_flow_pipeline.py",
    "3D/model/test_reference_parity.py",
    "gui/test_flow_panel.py",
]
FLOW_CHANGED = [
    "3D/tools/dataset_reader.py",
    "3D/tools/collect_complab_output.py",
    "3D/tools/collect_foreign_complab.py",
    "3D/model/train.py",
    "gui/prt_gui.py",
    "gui/test_gui_commands.py",
    "check_everything.py",
]

BANNER = re.compile(r"CHANGED FROM THE 2D VERSION|NEW IN THE FLOW VERSION")
# A block comment is a run of at least three full-line comments, which is what
# separates a section marker from an aside about the next line. A run of one or
# two is a LINE comment: it explains the line under it, not the section.
#
# That distinction is the whole of the counting here, and it took one wrong
# version to get right. Counting only comments that sit AFTER code on the same
# line called a carefully explained file uncommented, because the explanations
# were on their own lines above the code rather than trailing it, which is
# where most of them belong once they are longer than a few words.
BLOCK_MIN = 3
# Below this, a file is under-commented whatever else it has. One comment line
# for every twenty of code is not a target, it is a floor.
DENSITY_FLOOR = 0.05


def measure(path):
    src = io.open(path, encoding="utf-8", errors="replace").read()
    lines = src.split("\n")
    code = 0
    full = 0            # whole-line comments, of any kind
    trail = 0           # comments after code on the same line
    near = 0            # runs of 1 or 2 full lines: an aside about the next line
    blocks = 0          # runs of 3 or more: a section marker
    run = 0
    in_doc = False
    quote = None
    for ln in lines:
        s = ln.strip()
        # Docstrings are documentation but they are not what was asked for, so
        # they are neither counted as comments nor as code.
        if in_doc:
            if quote in s:
                in_doc = False
            continue
        if s.startswith('"""') or s.startswith("'''"):
            quote = s[:3]
            if not (len(s) > 3 and s.endswith(quote)):
                in_doc = True
            continue
        if s.startswith("#"):
            full += 1
            run += 1
            continue
        if run >= BLOCK_MIN:
            blocks += 1
        elif run:
            near += 1
        run = 0
        if not s:
            continue
        code += 1
        # A '#' inside a string is not a comment. Strip the obvious cases
        # rather than parsing Python: this is a density estimate, not a lint.
        bare = re.sub(r"'[^']*'|\"[^\"]*\"", "", ln)
        if "#" in bare:
            trail += 1
    if run >= BLOCK_MIN:
        blocks += 1
    elif run:
        near += 1
    return dict(total=len(lines), code=code, full=full, trail=trail, near=near,
                lines_=trail + near, blocks=blocks,
                banner=bool(BANNER.search(src[:6000])),
                density=(full + trail) / max(code, 1))


def report(rel, m, need_banner, quiet):
    miss = []
    if need_banner and not m["banner"]:
        miss.append("NO TOP BLOCK saying what changed from the 2D version")
    if m["blocks"] < 2:
        miss.append("only %d block comment(s); the sections are not marked"
                    % m["blocks"])
    if m["lines_"] < 5:
        miss.append("only %d line comment(s): %d beside code, %d explaining the "
                    "line below" % (m["lines_"], m["trail"], m["near"]))
    if m["density"] < DENSITY_FLOOR:
        miss.append("one comment per %d lines of code, below the floor of 1 in %d"
                    % (round(1 / max(m["density"], 1e-9)),
                       round(1 / DENSITY_FLOOR)))
    if quiet and not miss:
        return miss
    mark = "ok  " if not miss else "MISS"
    print("  %s %-42s %5d lines %4d block %4d line %5.0f%%"
          % (mark, rel, m["code"], m["blocks"], m["lines_"],
             100 * m["density"]))
    for w in miss:
        print("           %s" % w)
    return miss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true",
                    help="print only the files with something missing")
    args = ap.parse_args()

    print("=" * 78)
    print("COMMENT AUDIT")
    print("=" * 78)
    print("BLOCK  a run of %d or more comment lines: a section marker."
          % BLOCK_MIN)
    print("LINE   a comment beside code, or a run of one or two above it: an")
    print("       aside about the line it sits on.")
    print("%%      comment lines as a share of code lines.")
    print("Docstrings count as neither. This cannot judge whether a comment is")
    print("any good, only whether one is there.\n")

    bad = []
    for title, group, need in (
            ("NEW IN THIS VERSION", FLOW_NEW, True),
            ("CHANGED FOR THIS VERSION", FLOW_CHANGED, True)):
        print("%s  (must carry the top block)" % title)
        for rel in group:
            p = os.path.join(HERE, rel)
            if not os.path.exists(p):
                print("  MISS %-42s NOT FOUND" % rel)
                bad.append(rel)
                continue
            if report(rel, measure(p), need, args.quiet):
                bad.append(rel)
        print()

    # Everything else. No top block is required of a file the flow work never
    # touched, but the density floor still applies: a file nobody can read is
    # a problem whoever wrote it.
    seen = set(FLOW_NEW) | set(FLOW_CHANGED)
    rest = []
    # tests/ is included: the test files are read by whoever is asked to run
    # them, more often than the code is, so they are held to the same bar.
    for sub in ("3D/tools", "3D/model", "gui", "bridge", "tests", "."):
        d = os.path.join(HERE, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py"):
                continue
            rel = os.path.normpath(os.path.join(sub, f)).replace("\\", "/")
            if rel not in seen:
                rest.append(rel)
    print("EVERY OTHER FILE  (no top block required; the floor still applies)")
    for rel in rest:
        if report(rel, measure(os.path.join(HERE, rel)), False, args.quiet):
            bad.append(rel)
    print()

    print("-" * 78)
    if bad:
        print("%d file(s) fall short:" % len(bad))
        for b in bad:
            print("   " + b)
        return 1
    print("Every file carries what it should: the top block where one is owed,")
    print("block comments marking its sections, and line comments beside the")
    print("code that needs them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
