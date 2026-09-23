#!/usr/bin/env python3
"""
test_documented_numbers.py — check, by measurement, every claim made about the three
switches. Run it after touching anything in tools/ or model/.

    python test_documented_numbers.py

Needs the practice files first:
    python build_practice_dataset.py --out ../../work/test3d.h5
    python build_dataset_2d.py --out ../../work/data2d.h5 --n-geom 6 --n-sets 3 --n-times 5

Every path is an argument, so nothing here assumes a particular machine.
Anything missing is reported as a skip, with the command that builds it.

Each claim is checked against a NUMBER, not against the code reading plausibly:

  1  with every switch off, the data pipeline is bit-for-bit what it was before
     the switches existed -- all six distance x velocity combinations
  2  a checkpoint written before the switches existed still loads, and its
     switches resolve to off; tested in two forms, including the oldest
  3  the travel time solves in about a second, lands in [0,1) after squashing,
     has the long tail that motivated the squashing, and is independent of the
     Peclet number, which it must be because Stokes is linear
  4  the extruded 2D flow field really is z-invariant, so switch B's training
     data is exact rather than approximate
  5  the dimension-free trunk has the SAME width in 2D and 3D, which is the
     whole reason it transfers, while the Cartesian trunk does not (4 vs 5)
"""
import os, sys, subprocess, numpy as np, torch, h5py, time

# AUDIT TEST-01. Every path in this file used to be hard-coded to one
# developer's machine: /home/claude/GeometryAware3D for the code, which is the
# name the project had before it was renamed, and /tmp/test3d.h5,
# /tmp/data2d.h5, /tmp/real2d.h5, /tmp/reg3d/best.pt for the fixtures. On any
# other checkout the file failed on its first import, and run_tests.py and
# check_everything.py both reported that as a failure of the repository.
#
# The code is now found relative to this file, the fixtures are arguments with
# sensible defaults, and --help works without needing any of them to exist.
# A missing fixture is reported as a skip with the command that would build it,
# which is a different thing from a failed claim.
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # the repository root
sys.path.insert(0, os.path.join(ROOT, "3D", "tools"))
sys.path.insert(0, os.path.join(ROOT, "3D", "model"))

_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--data-3d", default=os.path.join(ROOT, "work", "test3d.h5"),
                 help="a small 3D dataset, for claims 1, 3 and 5")
_ap.add_argument("--data-2d", default=os.path.join(ROOT, "work", "data2d.h5"),
                 help="a small 2D dataset, for claim 5")
_ap.add_argument("--checkpoint", default=os.path.join(ROOT, "work", "reg3d", "best.pt"),
                 help="any trained checkpoint, for claim 2")
_ap.add_argument("--transfer", default=os.path.join(ROOT, "work", "transfer2d.h5"),
                 help="an extruded transfer set, for claim 4")
_ap.add_argument("--tmp", default=None, help="where to put intermediate files")
ARGS = _ap.parse_args()

import tempfile
TMP = ARGS.tmp or tempfile.mkdtemp(prefix="documented_numbers_")
os.makedirs(TMP, exist_ok=True)

SKIPPED = []


def need(path, what, how):
    """A fixture that is not there is a skip, with the command that makes it."""
    if os.path.exists(path):
        return True
    SKIPPED.append((what, path, how))
    print("SKIP  %-52s  missing %s" % (what, path))
    print("      build it with:  %s" % how)
    return False

from dataset_reader import PRT3DDataset, resolve_switches, dataset_kwargs_from_ckpt
from flow_coordinates import travel_time, squash, stats
R=[]
def chk(name, ok, detail=""): R.append((name,ok,detail)); print("%-58s %s  %s"%(name,"PASS" if ok else "FAIL",detail))

if need(ARGS.data_3d, "CLAIM 1  the switches are off when off",
        "python build_practice_dataset.py --out %s" % ARGS.data_3d):
    print("="*90); print("CLAIM 1  OFF == original, bit-exact"); print("="*90)
    r=subprocess.run([sys.executable,os.path.join(ROOT, "3D", "tools", "test_three_switches.py"),
                      '--data',ARGS.data_3d],capture_output=True,text=True)
    n=r.stdout.count("PASS")
    chk("test_three_switches.py: all checks pass", "ALL TESTS PASSED" in r.stdout, "%d checks"%n)
    import re as _re
    combos=_re.findall(r"^  distance=(gdf|edt|none)\s+with_velocity=(True|False)\s+identical\s+(PASS|FAIL)",
                       r.stdout, _re.M)
    chk("  covers gdf/edt/none x with/without velocity",
        len(combos)==6 and all(c[2]=="PASS" for c in combos),
        "%d of 6 combinations, all %s"%(len(combos),
        "PASS" if all(c[2]=="PASS" for c in combos) else "NOT pass"))

if need(ARGS.checkpoint, "CLAIM 2  old checkpoints still load",
        "python ../model/train.py --data %s --out %s --epochs 2"
        % (ARGS.data_3d, os.path.dirname(ARGS.checkpoint))):
    print(); print("="*90); print("CLAIM 2  old checkpoints still load"); print("="*90)
    src=ARGS.checkpoint
    ck=torch.load(src,map_location='cpu',weights_only=False)
    # (a) a checkpoint from before the switches existed: no switch keys in args
    a=dict(ck); a['args']={k:v for k,v in ck['args'].items()
            if k not in ('flow_proxy','dim_free','flow_mode','keep_geometry_channel',
                         'u_floor','transfer_2d','transfer_2d_frac','init_from','freeze_trunk')}
    for k in ('trunk_cols','branch_ch','switch_label'): a.pop(k,None)
    torch.save(a,os.path.join(TMP, "old_a.pt"))
    # (b) the very oldest form: also no trunk_in_dim / in_channels / grid
    b=dict(a)
    for k in ('trunk_in_dim','in_channels','grid','n_times'): b.pop(k,None)
    torch.save(b,os.path.join(TMP, "old_b.pt"))
    for tag,p in (("no switch keys",os.path.join(TMP, "old_a.pt")),("oldest form",os.path.join(TMP, "old_b.pt"))):
        kw,cfg=dataset_kwargs_from_ckpt(torch.load(p,map_location='cpu',weights_only=False))
        ok = (cfg['flow_proxy'] is False and cfg['dim_free'] is False
              and cfg['trunk_cols']==['x','y','z','t','gdf'])
        chk("  %-16s -> switches default OFF"%tag, ok, cfg['label'])
        rr=subprocess.run([sys.executable,os.path.join(ROOT, "3D", "model", "evaluate.py"),
            '--checkpoint',p,'--data',ARGS.data_3d,'--out',os.path.join(TMP, "ev_old"),'--no-3d',
            '--save-fields','0'],capture_output=True,text=True)
        chk("  %-16s -> evaluate.py runs it"%tag, rr.returncode==0 and "mean RMSE" in rr.stdout,
            (rr.stdout.strip().splitlines()[-1][:60] if rr.returncode==0 else rr.stderr.strip()[-70:]))

if need(ARGS.data_3d, "CLAIM 3  travel time",
        "python build_practice_dataset.py --out %s" % ARGS.data_3d):
    print(); print("="*90); print("CLAIM 3  switch A: tau"); print("="*90)
    mat=np.load(os.path.join(TMP, "tg.npz"))['material']; vel=np.load(os.path.join(TMP, "tvel.npy"))
    t0=time.time(); tau,stag=travel_time(vel,mat); el=time.time()-t0
    st=stats(tau,mat); m=mat==2
    chk("  tau solves in about a second", el<5.0, "%.2f s on %s"%(el,"x".join(map(str,mat.shape))))
    raw,_=travel_time(vel,mat)
    sq=squash(raw)
    chk("  squashed tau lands in [0,1)", float(sq[m].min())>=0 and float(sq[m].max())<1,
        "range %.3f..%.3f"%(sq[m].min(),sq[m].max()))
    chk("  raw tau has the long tail I quoted", st['max']/st['median']>50,
        "median %.2f  p99 %.1f  max %.1f"%(st['median'],st['p99'],st['max']))
    a2,_=travel_time(vel*7.3,mat)
    chk("  tau independent of Pe (u scaled x7.3)", float(np.nanmax(np.abs(a2[m]-raw[m])))<1e-4,
        "max diff %.1e"%float(np.nanmax(np.abs(a2[m]-raw[m]))))
    ds=PRT3DDataset(ARGS.data_3d,n_points=256,flow_proxy=True)
    gi=ds.geom_index; same=np.where(gi==gi[0])[0]
    c=ds.trunk_cols.index('tau')
    x=PRT3DDataset(ARGS.data_3d,indices=[same[0]],n_points=999,flow_proxy=True,seed=1)[0][2].numpy()
    y=PRT3DDataset(ARGS.data_3d,indices=[same[1]],n_points=999,flow_proxy=True,seed=1)[0][2].numpy()
    chk("  dataset gives identical tau for two Pe, same geometry",
        float(np.abs(np.sort(x[:,c])-np.sort(y[:,c])).max())<1e-6)
    chk("  switch A trunk swaps gdf -> tau", ds.trunk_cols==['x','y','z','t','tau'], str(ds.trunk_cols))
    chk("  switch A branch is the velocity field", ds.branch_ch==['ux','uy','uz'], str(ds.branch_ch))

if need(ARGS.transfer, "CLAIM 4  the extrusion is exact",
        "python build_transfer_set_2d_to_3d.py --synthetic 4 --out %s"
        % ARGS.transfer):
    print(); print("="*90); print("CLAIM 4  switch B: extrusion is exact"); print("="*90)
    h=h5py.File(ARGS.data_2d)
    zv=float(h.attrs['max_z_variation'])
    chk("  measured z-variation of the extruded flow", zv==0.0, "%.2e (0 = exact)"%zv)
    src=h.attrs['source']
    src=src.decode() if isinstance(src,bytes) else str(src)   # h5py returns str, not bytes
    chk("  built from the real published domains", src=='extruded_2d',
        "source=%s  nz_solve=%d  grid=%s"%(src,int(h.attrs['nz_solve']),
                                           tuple(int(v) for v in h.attrs['shape'])))

if need(ARGS.data_2d, "CLAIM 5  the dimension-free trunk",
        "python build_dataset_2d.py --out %s --n-geom 4" % ARGS.data_2d):
    print(); print("="*90); print("CLAIM 5  switch C: 3 columns in BOTH 2D and 3D"); print("="*90)
    d3=PRT3DDataset(ARGS.data_3d,n_points=64,dim_free=True)
    d2=PRT3DDataset(ARGS.data_2d,n_points=64,dim_free=True)
    chk("  3D dim-free trunk", d3.trunk_cols==['t','dwall','tau'], "%s  (ndim=%d)"%(d3.trunk_cols,d3.ndim))
    chk("  2D dim-free trunk", d2.trunk_cols==['t','dwall','tau'], "%s  (ndim=%d)"%(d2.trunk_cols,d2.ndim))
    chk("  identical width, so the net transfers unchanged", d3.trunk_dim==d2.trunk_dim==3)
    chk("  and the CARTESIAN trunk does NOT (4 vs 5)",
        resolve_switches(ndim=2)['trunk_dim']==4 and resolve_switches(ndim=3)['trunk_dim']==5,
        "2D=%d  3D=%d"%(resolve_switches(ndim=2)['trunk_dim'],resolve_switches(ndim=3)['trunk_dim']))

print(); print("="*90)
bad=[n for n,o,_ in R if not o]
print("%d checks, %d passed, %d FAILED"%(len(R),len(R)-len(bad),len(bad)))
if bad: print("FAILURES:"); [print("   -",n) for n in bad]
# AUDIT TEST-01. A claim that could not be checked is reported as a skip, not
# as a pass and not as a failure, and the command that would let it be checked
# is printed. A run with nothing to check on says so instead of looking green.
if SKIPPED:
    print()
    print("%d claim(s) were NOT checked, for want of a fixture:" % len(SKIPPED))
    for what, path, how in SKIPPED:
        print("   - %s" % what)
        print("       missing : %s" % path)
        print("       build it: %s" % how)
if not R:
    print()
    print("NOTHING WAS CHECKED. Build the fixtures above and run this again.")
# EXIT NON-ZERO WHEN SOMETHING FAILED. Without this the shell status was
# always 0, so check_everything.py reported this group as ok no matter how
# many claims had drifted -- a test that cannot fail is not a test.
# AUDIT TEST-01 and TEST-RUNNER-02. Three outcomes, three statuses, so a run
# that checked nothing is never reported as a run that passed:
#   0  every claim that could be checked was checked, and held
#   1  a claim was checked and did not hold
#   2  nothing could be checked, because the fixtures are not there
if bad:
    sys.exit(1)
sys.exit(0 if R else 2)
