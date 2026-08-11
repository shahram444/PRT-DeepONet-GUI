#!/usr/bin/env python3
"""
complab_campaign.py — build and manage a batch CompLaB3D complab_campaign for PRT-DeepONet-3D.

Every run is fully isolated in its own directory with its own CompLaB.xml,
input/geometry.dat and output/.  A failure in one run cannot affect any other:
Slurm array tasks are independent processes, run_one.sh never propagates a
non-zero exit, and every run writes a status.json whether it succeeded or not.

    # 1. build the run matrix and the Slurm array script
    python complab_campaign.py build \
        --geometries ./geometries \
        --out ./complab_campaign \
        --complab /path/to/complab \
        --per-geom 20

    # 2. submit
    cd complab_campaign && sbatch submit.sbatch

    # 3. watch
    python complab_campaign.py status --out ./complab_campaign

    # 4. retry only the failures (writes a second array script)
    python complab_campaign.py retry --out ./complab_campaign

Design notes
------------
Peclet is set in the XML and CompLaB auto-rescales delta_P to hit it
(complab.cpp:429-445), so Pe changes the flow field.  Damkohler is set through
PRT_* environment variables read by defineKinetics.hh / defineAbioticKinetics.hh,
so it changes NOTHING about the flow and needs no recompile.

Dimensionless groups (diffusive Damkohler, independent of velocity):
    Pe       = u * L / D_Ac                     <- <Peclet> in the XML
    Da_bio   = (Vmax * B0 / Ac0) * L^2 / D_Ac   ->  PRT_VMAX
    Da_abio  = k_surf * L^2 / D_P               ->  PRT_KSURF
    Ks_Ac/Ac0 , Ks_A/A0 , Y*Ac0/B0              ->  PRT_KS_AC, PRT_KS_A, PRT_Y
That is a 6-dimensional branch-2 input for the DeepONet.

ade_dt = ((tau-0.5)/3) * dx_m^2 / D_Ac is computed here and exported as PRT_DT,
which removes the hand-synchronisation bug that cost the T9 runs a 60x error.
Cross-check it against the printed "[ADE] dt" on the first run of a complab_campaign.
"""

import argparse, csv, json, math, os, shutil, stat, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "CompLaB.xml.template")

# fixed settings_and_units constants of the system (shared by every run)
D_AC  = 1.0e-9      # donor diffusion, m^2/s   -- substrate 0, sets ade_dt
D_A   = 1.0e-9      # acceptor
D_P   = 1.5e-9      # product
D_BIO = 2.0e-10     # planktonic cells diffuse ~5x slower than small solutes
AC0   = 2.0e-3      # inlet donor,    mol/L
A0    = 5.0e-3      # inlet acceptor, mol/L
B0    = 1.0e-2      # uniform initial biomass
KD    = 0.006       # biomass decay, 1/s
ALPHA = 1.0         # acceptor:donor stoichiometry


# --------------------------------------------------------------------- utils
def render(template_text, mapping):
    out = template_text
    for k, v in mapping.items():
        out = out.replace("@%s@" % k, str(v))
    leftover = [w for w in out.split("@") if w.isupper() and w.isidentifier()]
    if leftover:
        raise RuntimeError("unfilled template keys: %s" % sorted(set(leftover)))
    return out


def loguniform(rng, lo, hi, n):
    return np.exp(rng.uniform(math.log(lo), math.log(hi), n))


def lhs_log(rng, ranges, n):
    """Latin hypercube in log space — better coverage than a random draw and far
    fewer runs than a full grid for the same parameter-space resolution."""
    d = len(ranges)
    out = np.empty((n, d))
    for j, (lo, hi) in enumerate(ranges):
        cut = (rng.permutation(n) + rng.random(n)) / n
        out[:, j] = np.exp(math.log(lo) + cut * (math.log(hi) - math.log(lo)))
    return out


def derive(pe, da_bio, da_abio, ks_ac, ks_a, Y, char_len_vox, dx_um, tau):
    """Map dimensionless groups to the settings_and_units constants CompLaB and the
    kinetics headers actually consume."""
    L = char_len_vox * dx_um * 1e-6                 # characteristic length, m
    dt = ((tau - 0.5) / 3.0) * (dx_um * 1e-6) ** 2 / D_AC
    vmax = da_bio * D_AC * AC0 / (B0 * L * L)       # 1/s per unit biomass
    ksurf = da_abio * D_P / (L * L)                 # 1/s
    return dict(L_m=L, dt=dt, vmax=vmax, ksurf=ksurf,
                ks_ac=ks_ac * AC0, ks_a=ks_a * A0, Y=Y)


# --------------------------------------------------------------------- build
def cmd_build(args):
    with open(TEMPLATE) as f:
        tpl = f.read()

    geoms = sorted(d for d in os.listdir(args.geometries)
                   if d.startswith("geom_") and
                   os.path.isdir(os.path.join(args.geometries, d)))
    if args.max_geom:
        geoms = geoms[:args.max_geom]
    if not geoms:
        sys.exit("no geom_* directories found under %s" % args.geometries)

    os.makedirs(args.out, exist_ok=True)

    # Split by GEOMETRY, not by run, so every batch holds whole geometries.
    # That keeps each batch independently collectable and keeps the
    # split-by-geometry train/test rule valid within a batch.
    nb = max(1, int(args.batches))
    groups = [list(g) for g in np.array_split(np.array(geoms, dtype=object), nb)]
    batch_of = {}
    for bi, grp in enumerate(groups):
        for gname in grp:
            batch_of[gname] = bi
    bdirs = [args.out if nb == 1 else os.path.join(args.out, "batch_%d" % (bi + 1))
             for bi in range(nb)]
    for d in bdirs:
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    rid = 0

    for gi, gname in enumerate(geoms):
        gdir = os.path.abspath(os.path.join(args.geometries, gname))
        gid = int(gname.split("_")[1])
        dat = os.path.join(gdir, "geometry.dat")
        if not os.path.isfile(dat):
            print("  SKIP %s: no geometry.dat" % gname, file=sys.stderr)
            continue

        if args.sampling == "grid":
            pts = [(p, b, a) for p in args.pe for b in args.da_bio for a in args.da_abio]
        else:
            s = lhs_log(rng, [tuple(args.pe_range), tuple(args.dabio_range),
                              tuple(args.daabio_range)], args.per_geom)
            pts = [tuple(r) for r in s]

        for (pe, da_bio, da_abio) in pts:
            # the two Monod ratios and the yield are drawn per run as well, so
            # the 6-D branch-2 space is covered rather than just its 3-D slice
            ks_ac = float(loguniform(rng, *args.ksac_range, 1)[0])
            ks_a  = float(loguniform(rng, *args.ksa_range, 1)[0])
            Y     = float(loguniform(rng, *args.y_range, 1)[0])

            d = derive(pe, da_bio, da_abio, ks_ac, ks_a, Y,
                       args.char_len, args.dx_um, args.tau)

            bi = batch_of[gname]
            rdir = os.path.join(bdirs[bi], "runs", "run_%06d" % rid)
            os.makedirs(os.path.join(rdir, "input"), exist_ok=True)
            os.makedirs(os.path.join(rdir, "output"), exist_ok=True)

            link = os.path.join(rdir, "input", "geometry.dat")
            if os.path.lexists(link):
                os.remove(link)
            if args.copy_geometry:
                shutil.copyfile(dat, link)
            else:
                os.symlink(dat, link)

            xml = render(tpl, dict(
                RUN_ID="%06d" % rid, GID=gid,
                PE="%.6g" % pe, DA_BIO="%.6g" % da_bio, DA_ABIO="%.6g" % da_abio,
                NX=args.nx, NY=args.ny, NZ=args.nz,
                DX_UM=args.dx_um, CHAR_LEN=args.char_len,
                DELTA_P=args.delta_p, TAU=args.tau,
                NS_MAX=args.ns_max, ADE_MAX=args.ade_max, ADE_CONV=args.ade_conv,
                D_AC="%.6g" % D_AC, D_AC_BF="%.6g" % (D_AC / 2),
                D_A="%.6g" % D_A,   D_A_BF="%.6g" % (D_A / 2),
                D_P="%.6g" % D_P,   D_P_BF="%.6g" % (D_P / 2),
                D_BIO="%.6g" % D_BIO,
                AC0="%.6g" % AC0, A0="%.6g" % A0, B0="%.6g" % B0, KD="%.6g" % KD,
                KS_AC="%.6g" % d["ks_ac"], KS_A="%.6g" % d["ks_a"],
                VMAX="%.6g" % d["vmax"],
                VTK_INTERVAL=args.vtk_interval,
                READ_NS="true" if args.reuse_flow else "false",
                PRECIP_BLOCK=(
                    "    <precipitation>\n"
                    "        <enabled>true</enabled>\n"
                    "        <surface_only>1</surface_only>\n"
                    "        <update_interval>1000000</update_interval>\n"
                    "    </precipitation>" if args.surface_abiotic else
                    "    <precipitation>\n        <enabled>false</enabled>\n    </precipitation>"),
            ))
            with open(os.path.join(rdir, "CompLaB.xml"), "w") as f:
                f.write(xml)

            env = "\n".join([
                "#!/bin/bash",
                "# rate constants for defineKinetics.hh / defineAbioticKinetics.hh.",
                "# These used to be constexpr, which meant a rebuild per Damkohler value.",
                "export PRT_VMAX=%.10g"    % d["vmax"],
                "export PRT_KSURF=%.10g"   % d["ksurf"],
                "export PRT_KS_AC=%.10g"   % d["ks_ac"],
                "export PRT_KS_A=%.10g"    % d["ks_a"],
                "export PRT_ALPHA=%.10g"   % ALPHA,
                "export PRT_Y=%.10g"       % d["Y"],
                "export PRT_KD=%.10g"      % KD,
                "# ade_dt = ((tau-0.5)/3)*dx_m^2/D_Ac  -- cross-check the printed [ADE] dt",
                "export PRT_DT=%.10g"      % d["dt"],
                "export PRT_MAXFRAC=0.5",
                "export COMPLAB_BIN=%s"    % os.path.abspath(args.complab),
                "export PRT_TIMEOUT=%d"    % args.timeout,
                "export PRT_LAUNCHER=%s"   % args.launcher,
                "",
            ])
            with open(os.path.join(rdir, "env.sh"), "w") as f:
                f.write(env)

            params = dict(
                run_id=rid, gid=gid, geometry_dir=gdir,
                pe=float(pe), da_bio=float(da_bio), da_abio=float(da_abio),
                ks_ac_norm=float(ks_ac), ks_a_norm=float(ks_a), y_norm=float(Y),
                vmax=d["vmax"], ksurf=d["ksurf"], ks_ac=d["ks_ac"], ks_a=d["ks_a"],
                ade_dt=d["dt"], L_m=d["L_m"],
                ac0=AC0, a0=A0, b0=B0, kd=KD, alpha=ALPHA,
                nx=args.nx, ny=args.ny, nz=args.nz,
                species=["Ac", "A", "P"], microbes=["Bio"],
                ade_max=args.ade_max, vtk_interval=args.vtk_interval,
                n_snapshots=args.ade_max // args.vtk_interval + 1,
            )
            with open(os.path.join(rdir, "params.json"), "w") as f:
                json.dump(params, f, indent=2)

            params["batch"] = bi + 1
            rows.append(params)
            rid += 1

    # ---- per-batch run list, manifest, driver, submit script ---------------
    keys = ["run_id", "batch", "gid", "pe", "da_bio", "da_abio", "ks_ac_norm",
            "ks_a_norm", "y_norm", "vmax", "ksurf", "ade_dt"]
    for bi, d in enumerate(bdirs):
        br = [r for r in rows if r["batch"] == bi + 1]
        with open(os.path.join(d, "runs.txt"), "w") as f:
            for r in br:
                f.write("runs/run_%06d\n" % r["run_id"])
        with open(os.path.join(d, "runs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in br:
                w.writerow({k: r[k] for k in keys})
        _write_runner(d)
        jn = args.job_name if nb == 1 else "%s_b%d" % (args.job_name, bi + 1)
        _write_sbatch(args, len(br), "submit.sbatch", "runs.txt", out=d, job=jn)

    if nb > 1:
        with open(os.path.join(args.out, "runs_all.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in keys})
        _write_batch_index(args, bdirs, rows, groups)

    dt_all = {r["ade_dt"] for r in rows}
    print("\nbuilt %d runs over %d geometries in %d batch(es) -> %s"
          % (len(rows), len(geoms), nb, args.out))
    for bi, d in enumerate(bdirs):
        br = [r for r in rows if r["batch"] == bi + 1]
        print("   batch %d: %4d runs, %3d geometries (%s .. %s)"
              % (bi + 1, len(br), len(groups[bi]), groups[bi][0], groups[bi][-1]))
    print("  ade_dt (PRT_DT) = %s s" % ", ".join("%.4g" % d for d in sorted(dt_all)))
    print("  Pe      %.3g .. %.3g" % (min(r["pe"] for r in rows), max(r["pe"] for r in rows)))
    print("  Da_bio  %.3g .. %.3g" % (min(r["da_bio"] for r in rows), max(r["da_bio"] for r in rows)))
    print("  Da_abio %.3g .. %.3g" % (min(r["da_abio"] for r in rows), max(r["da_abio"] for r in rows)))
    if nb == 1:
        print("\n  cd %s && sbatch submit.sbatch" % args.out)
    else:
        print("\n  submit one batch at a time:")
        for bi in range(nb):
            print("     cd %s && sbatch submit.sbatch" % bdirs[bi])
        print("  or all at once:   bash %s" % os.path.join(args.out, "submit_all.sh"))
    print("  IMPORTANT: on the first completed run, check that the printed")
    print("  '[ADE] dt' matches PRT_DT above, and compare output/inputGeom.vti")
    print("  against the geometry npz to confirm the index ordering.")


def _write_batch_index(args, bdirs, rows, groups):
    """A submit_all.sh and a human-readable BATCHES.md at the complab_campaign root."""
    sh = os.path.join(args.out, "submit_all.sh")
    with open(sh, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submits every batch. Each is an independent Slurm array;\n")
        f.write("# a failure in one cannot affect another.\n")
        f.write("set -u\n")
        for d in bdirs:
            f.write('(cd "%s" && sbatch submit.sbatch)\n' % os.path.abspath(d))
    os.chmod(sh, os.stat(sh).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    L = ["# Campaign batches\n\n",
         "%d runs over %d geometries, split into %d independent batches.\n\n"
         % (len(rows), len({r["gid"] for r in rows}), len(bdirs)),
         "The split is by GEOMETRY, so each batch holds whole geometries and can be "
         "collected and trained on by itself. Run ids are globally unique, so batches "
         "merge without collision.\n\n",
         "| batch | runs | geometries | gid range | submit |\n",
         "|---|---:|---:|---|---|\n"]
    for bi, d in enumerate(bdirs):
        br = [r for r in rows if r["batch"] == bi + 1]
        gids = sorted({r["gid"] for r in br})
        L.append("| %d | %d | %d | %04d to %04d | `cd %s && sbatch submit.sbatch` |\n"
                 % (bi + 1, len(br), len(gids), gids[0], gids[-1],
                    os.path.basename(d)))
    L += ["\n## Watch\n\n```bash\npython3 complab_campaign.py status --out <campaign_root>\n",
          "python3 complab_campaign.py status --out <campaign_root>/batch_1\n```\n",
          "\n## Re-queue crashes, per batch\n\n```bash\n",
          "python3 complab_campaign.py retry --out <campaign_root>/batch_1\n",
          "cd <campaign_root>/batch_1 && sbatch retry.sbatch\n```\n",
          "\n## Collect\n\nOne dataset from everything finished so far:\n\n```bash\n",
          "python3 collect_complab_output.py --complab_campaign <campaign_root>/batch_* \\\n",
          "                   --geometries ../geometries --out ./dataset --mode steady\n```\n",
          "\nOr one batch alone:\n\n```bash\n",
          "python3 collect_complab_output.py --complab_campaign <campaign_root>/batch_1 \\\n",
          "                   --geometries ../geometries --out ./dataset_b1 --mode steady\n```\n"]
    with open(os.path.join(args.out, "BATCHES.md"), "w") as f:
        f.writelines(L)


def _write_runner(out):
    """run_one.sh — never propagates a failure, always writes status.json."""
    txt = r'''#!/bin/bash
# run_one.sh <run_dir>   — executes ONE CompLaB simulation in isolation.
# Deliberately does NOT use `set -e`: a failing run must record its failure and
# exit 0 so that neither Slurm nor the surrounding array treats it as fatal.

RUN_DIR="$1"
if [ -z "$RUN_DIR" ]; then echo "usage: run_one.sh <run_dir>"; exit 2; fi
cd "$RUN_DIR" || { echo "cannot cd to $RUN_DIR"; exit 2; }

STATUS="status.json"
write_status () {   # state reason exit_code wall_s
  cat > "$STATUS" <<EOF
{"state": "$1", "reason": "$2", "exit_code": $3, "wall_s": $4,
 "n_vti": $(ls -1 output/*.vti 2>/dev/null | wc -l),
 "host": "$(hostname)", "slurm_job": "${SLURM_JOB_ID:-none}",
 "array_task": "${SLURM_ARRAY_TASK_ID:-none}"}
EOF
}

if [ ! -f env.sh ];      then write_status fail missing_env 90 0;  exit 0; fi
if [ ! -f CompLaB.xml ]; then write_status fail missing_xml 91 0;  exit 0; fi
# shellcheck disable=SC1091
source ./env.sh
if [ ! -x "$COMPLAB_BIN" ]; then write_status fail missing_binary 92 0; exit 0; fi
if [ ! -e input/geometry.dat ]; then write_status fail missing_geometry 93 0; exit 0; fi

mkdir -p output
rm -f output/*.vti output/*.chk 2>/dev/null

t0=$(date +%s)
# PRT_LAUNCHER is set in env.sh: "mpirun" on UGA GACRC Sapelo2, "srun" on some
# other sites, "none" to run the binary directly (serial / debugging).
case "${PRT_LAUNCHER:-mpirun}" in
  mpirun) timeout "${PRT_TIMEOUT:-14400}" mpirun -np "${SLURM_NTASKS:-1}" "$COMPLAB_BIN" > run.log 2>&1 ;;
  srun)   timeout "${PRT_TIMEOUT:-14400}" srun --ntasks="${SLURM_NTASKS:-1}" "$COMPLAB_BIN" > run.log 2>&1 ;;
  *)      timeout "${PRT_TIMEOUT:-14400}" "$COMPLAB_BIN" > run.log 2>&1 ;;
esac
rc=$?
t1=$(date +%s); wall=$((t1-t0))

nvti=$(ls -1 output/*.vti 2>/dev/null | wc -l)

if   [ $rc -eq 124 ]; then write_status fail timeout 124 "$wall"
elif [ $rc -ne 0 ];   then
  reason=$(grep -m1 -iE "terminating|error|exception|assert" run.log 2>/dev/null \
           | tr -d '"' | tr '\n' ' ' | cut -c1-160)
  [ -z "$reason" ] && reason="nonzero_exit"
  write_status fail "$reason" $rc "$wall"
elif [ "$nvti" -eq 0 ]; then write_status fail no_output 95 "$wall"
else write_status ok "" 0 "$wall"
fi

# Free the checkpoints: they are large and collect_complab_output.py reads only the .vti files.
rm -f output/*.chk 2>/dev/null
exit 0
'''
    p = os.path.join(out, "run_one.sh")
    with open(p, "w") as f:
        f.write(txt)
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_sbatch(args, n, fname, listfile, out=None, job=None):
    mods = "\n".join("module load %s" % m for m in args.modules) if args.modules else \
           "# module load <your compiler/MPI>   # e.g. on Sapelo2: ml spider OpenMPI"
    out = out or args.out
    job = job or args.job_name
    txt = """#!/bin/bash
#SBATCH --job-name=%(job)s
#SBATCH --array=0-%(last)d%%%(conc)d
#SBATCH --partition=%(part)s
#SBATCH --nodes=1
#SBATCH --ntasks=%(cores)d
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=%(mem)s
#SBATCH --time=%(time)s
%(acct)s#SBATCH --output=logs/slurm_%%A_%%a.out
#SBATCH --error=logs/slurm_%%A_%%a.err
%(mail)s
# A Slurm job ARRAY of many SMALL jobs, not one big job.  A 64^3 domain does not
# need 36 ranks, and array tasks are separate processes, so one crash cannot take
# down the rest.  The %%N suffix on --array caps how many run at once.

cd $SLURM_SUBMIT_DIR
mkdir -p logs
%(mods)s

RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" %(list)s)
if [ -z "$RUN" ]; then echo "no run for task $SLURM_ARRAY_TASK_ID"; exit 0; fi
bash run_one.sh "$RUN"
exit 0
""" % dict(job=job, last=max(n - 1, 0), conc=args.concurrent,
           cores=args.cores, time=args.time, part=args.partition, mem=args.mem_per_cpu,
           acct=("#SBATCH --account=%s\n" % args.account) if args.account else "",
           mail=("#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user=%s\n" % args.email) if args.email else "",
           mods=mods, list=listfile)
    p = os.path.join(out, fname)
    with open(p, "w") as f:
        f.write(txt)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)


# -------------------------------------------------------------------- status
def _batch_dirs(out):
    """Return the complab_campaign dirs under `out`: itself, or its batch_* children."""
    bs = sorted(d for d in os.listdir(out)
                if d.startswith("batch_") and
                os.path.isdir(os.path.join(out, d, "runs")))
    return [os.path.join(out, b) for b in bs] if bs else [out]


def _scan(out):
    runs_dir = os.path.join(out, "runs")
    if not os.path.isdir(runs_dir):
        res = []
        for d in _batch_dirs(out):
            res += [(os.path.basename(d) + "/" + a, b, c, e) for a, b, c, e in _scan(d)]
        return res
    res = []
    for d in sorted(os.listdir(runs_dir)):
        rd = os.path.join(runs_dir, d)
        sp = os.path.join(rd, "status.json")
        if not os.path.isfile(sp):
            res.append((d, "pending", "", 0))
            continue
        try:
            with open(sp) as f:
                s = json.load(f)
            res.append((d, s.get("state", "?"), s.get("reason", ""), s.get("wall_s", 0)))
        except Exception as e:
            res.append((d, "fail", "unreadable_status:%s" % e, 0))
    return res


def cmd_status(args):
    res = _scan(args.out)
    from collections import Counter
    c = Counter(r[1] for r in res)
    print("total %d  |  " % len(res) + "  ".join("%s=%d" % kv for kv in sorted(c.items())))
    walls = [r[3] for r in res if r[1] == "ok" and r[3]]
    if walls:
        print("  wall time (ok runs): median %.0f s  max %.0f s  total %.1f node-h"
              % (float(np.median(walls)), max(walls), sum(walls) / 3600.0))
    fails = [r for r in res if r[1] == "fail"]
    if fails:
        print("\nfailures (%d):" % len(fails))
        fc = Counter(r[2][:70] for r in fails)
        for reason, k in fc.most_common(15):
            print("  %4d x  %s" % (k, reason or "(no reason recorded)"))


def cmd_retry(args):
    if not os.path.isdir(os.path.join(args.out, "runs")):
        for d in _batch_dirs(args.out):
            sub = argparse.Namespace(**vars(args)); sub.out = d
            print("--- %s" % os.path.basename(d)); cmd_retry(sub)
        return
    res = _scan(args.out)
    bad = [d for d, st, _, _ in res if st != "ok"]
    if not bad:
        print("nothing to retry")
        return
    with open(os.path.join(args.out, "retry.txt"), "w") as f:
        for d in bad:
            f.write("runs/%s\n" % d)
    class A: pass
    a = A()
    for k in ("out", "job_name", "concurrent", "cores", "time", "partition",
              "account", "email", "mem_per_cpu", "modules", "launcher"):
        setattr(a, k, getattr(args, k))
    a.job_name = args.job_name + "_retry"
    _write_sbatch(a, len(bad), "retry.sbatch", "retry.txt")
    print("wrote retry.txt (%d runs) and retry.sbatch" % len(bad))
    print("  cd %s && sbatch retry.sbatch" % args.out)


# ---------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--out", default="./complab_campaign")
        q.add_argument("--job-name", default="prt3d")
        q.add_argument("--concurrent", type=int, default=40,
                       help="max array tasks running at once")
        q.add_argument("--cores", type=int, default=8, help="MPI ranks per run")
        q.add_argument("--time", default="04:00:00")
        q.add_argument("--partition", default="batch",
                       help="Slurm partition. UGA GACRC Sapelo2: 'batch' (7-day limit)")
        q.add_argument("--account", default=None,
                       help="omit on Sapelo2 - it does not require an account flag")
        q.add_argument("--email", default=None, help="for --mail-user")
        q.add_argument("--mem-per-cpu", default="2gb")
        q.add_argument("--modules", nargs="*", default=[],
                       help="module load lines, e.g. --modules foss/2023a")
        q.add_argument("--launcher", choices=["mpirun", "srun", "none"], default="mpirun",
                       help="mpirun on Sapelo2; srun at some other sites")

    b = sub.add_parser("build"); common(b)
    b.add_argument("--geometries", required=True)
    b.add_argument("--complab", required=True, help="path to the complab binary")
    b.add_argument("--max-geom", type=int, default=None)
    b.add_argument("--per-geom", type=int, default=20)
    b.add_argument("--batches", type=int, default=1,
                   help="split into N independent campaigns, by geometry. Each "
                        "gets its own runs/, runs.txt, runs.csv, run_one.sh and "
                        "submit.sbatch, and can be submitted and collected alone.")
    b.add_argument("--sampling", choices=["lhs", "grid"], default="lhs")
    b.add_argument("--pe-range", type=float, nargs=2, default=[1.0, 100.0])
    b.add_argument("--dabio-range", type=float, nargs=2, default=[0.1, 100.0])
    b.add_argument("--daabio-range", type=float, nargs=2, default=[0.1, 100.0])
    b.add_argument("--ksac-range", type=float, nargs=2, default=[0.01, 1.0])
    b.add_argument("--ksa-range", type=float, nargs=2, default=[0.01, 1.0])
    b.add_argument("--y-range", type=float, nargs=2, default=[0.01, 0.2])
    b.add_argument("--pe", type=float, nargs="+", default=[1, 5, 20, 100])
    b.add_argument("--da-bio", type=float, nargs="+", default=[0.1, 1, 10, 100])
    b.add_argument("--da-abio", type=float, nargs="+", default=[0.1, 1, 10])
    b.add_argument("--nx", type=int, default=64)
    b.add_argument("--ny", type=int, default=64)
    b.add_argument("--nz", type=int, default=64)
    b.add_argument("--dx-um", type=float, default=1.0)
    b.add_argument("--char-len", type=float, default=60.0)
    b.add_argument("--tau", type=float, default=0.8)
    b.add_argument("--delta-p", type=float, default=2.0e-3)
    b.add_argument("--ns-max", type=int, default=50000)
    b.add_argument("--ade-max", type=int, default=50000)
    b.add_argument("--ade-conv", default="1e-6")
    b.add_argument("--vtk-interval", type=int, default=2500)
    b.add_argument("--timeout", type=int, default=14400, help="seconds per run")
    b.add_argument("--seed", type=int, default=20260728)
    b.add_argument("--copy-geometry", action="store_true",
                   help="copy geometry.dat instead of symlinking")
    b.add_argument("--surface-abiotic", action="store_true",
                   help="make the abiotic reaction fire only on pore voxels "
                        "touching a grain. OFF by default (volumetric). This is "
                        "the only path CompLaB exposes for surface gating and it "
                        "goes through the <precipitation> block, but nothing "
                        "precipitates: <solid_substrate> is omitted so the "
                        "geometry stays frozen.")
    b.add_argument("--reuse-flow", action="store_true",
                   help="set read_NS_file=true. OFF by default: the Peclet "
                        "auto-rescale at complab.cpp:429 may re-solve the flow "
                        "anyway, so validate on one run before trusting it.")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("status"); common(s); s.set_defaults(func=cmd_status)
    r = sub.add_parser("retry");  common(r); r.set_defaults(func=cmd_retry)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
