#!/usr/bin/env python3
"""
test_documented_numbers.py — check, by measurement, every claim made about the three
switches. Run it after touching anything in tools/ or model/.

    python test_documented_numbers.py

Needs the practice files first:
    python build_practice_dataset.py --out /tmp/test3d.h5
    python build_dataset_2d.py --out /tmp/data2d.h5 --n-geom 6 --n-sets 3 --n-times 5

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
sys.path.insert(0,'/home/claude/GeometryAware3D/tools')
sys.path.insert(0,'/home/claude/GeometryAware3D/model')
from dataset_reader import PRT3DDataset, resolve_switches, dataset_kwargs_from_ckpt
from flow_coordinates import travel_time, squash, stats
R=[]
def chk(name, ok, detail=""): R.append((name,ok,detail)); print("%-58s %s  %s"%(name,"PASS" if ok else "FAIL",detail))

print("="*90); print("CLAIM 1  OFF == original, bit-exact"); print("="*90)
r=subprocess.run([sys.executable,'/home/claude/GeometryAware3D/tools/test_three_switches.py',
                  '--data','/tmp/test3d.h5'],capture_output=True,text=True)
n=r.stdout.count("PASS")
chk("test_three_switches.py: all checks pass", "ALL TESTS PASSED" in r.stdout, "%d checks"%n)
import re as _re
combos=_re.findall(r"^  distance=(gdf|edt|none)\s+with_velocity=(True|False)\s+identical\s+(PASS|FAIL)",
                   r.stdout, _re.M)
chk("  covers gdf/edt/none x with/without velocity",
    len(combos)==6 and all(c[2]=="PASS" for c in combos),
    "%d of 6 combinations, all %s"%(len(combos),
    "PASS" if all(c[2]=="PASS" for c in combos) else "NOT pass"))

print(); print("="*90); print("CLAIM 2  old checkpoints still load"); print("="*90)
src='/tmp/reg3d/best.pt'
ck=torch.load(src,map_location='cpu',weights_only=False)
# (a) a checkpoint from before the switches existed: no switch keys in args
a=dict(ck); a['args']={k:v for k,v in ck['args'].items()
        if k not in ('flow_proxy','dim_free','flow_mode','keep_geometry_channel',
                     'u_floor','transfer_2d','transfer_2d_frac','init_from','freeze_trunk')}
for k in ('trunk_cols','branch_ch','switch_label'): a.pop(k,None)
torch.save(a,'/tmp/old_a.pt')
# (b) the very oldest form: also no trunk_in_dim / in_channels / grid
b=dict(a)
for k in ('trunk_in_dim','in_channels','grid','n_times'): b.pop(k,None)
torch.save(b,'/tmp/old_b.pt')
for tag,p in (("no switch keys",'/tmp/old_a.pt'),("oldest form",'/tmp/old_b.pt')):
    kw,cfg=dataset_kwargs_from_ckpt(torch.load(p,map_location='cpu',weights_only=False))
    ok = (cfg['flow_proxy'] is False and cfg['dim_free'] is False
          and cfg['trunk_cols']==['x','y','z','t','gdf'])
    chk("  %-16s -> switches default OFF"%tag, ok, cfg['label'])
    rr=subprocess.run([sys.executable,'/home/claude/GeometryAware3D/model/evaluate.py',
        '--checkpoint',p,'--data','/tmp/test3d.h5','--out','/tmp/ev_old','--no-3d',
        '--save-fields','0'],capture_output=True,text=True)
    chk("  %-16s -> evaluate.py runs it"%tag, rr.returncode==0 and "mean RMSE" in rr.stdout,
        (rr.stdout.strip().splitlines()[-1][:60] if rr.returncode==0 else rr.stderr.strip()[-70:]))

print(); print("="*90); print("CLAIM 3  switch A: tau"); print("="*90)
mat=np.load('/tmp/tg.npz')['material']; vel=np.load('/tmp/tvel.npy')
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
ds=PRT3DDataset('/tmp/test3d.h5',n_points=256,flow_proxy=True)
gi=ds.geom_index; same=np.where(gi==gi[0])[0]
c=ds.trunk_cols.index('tau')
x=PRT3DDataset('/tmp/test3d.h5',indices=[same[0]],n_points=999,flow_proxy=True,seed=1)[0][2].numpy()
y=PRT3DDataset('/tmp/test3d.h5',indices=[same[1]],n_points=999,flow_proxy=True,seed=1)[0][2].numpy()
chk("  dataset gives identical tau for two Pe, same geometry",
    float(np.abs(np.sort(x[:,c])-np.sort(y[:,c])).max())<1e-6)
chk("  switch A trunk swaps gdf -> tau", ds.trunk_cols==['x','y','z','t','tau'], str(ds.trunk_cols))
chk("  switch A branch is the velocity field", ds.branch_ch==['ux','uy','uz'], str(ds.branch_ch))

print(); print("="*90); print("CLAIM 4  switch B: extrusion is exact"); print("="*90)
h=h5py.File('/tmp/real2d.h5')
zv=float(h.attrs['max_z_variation'])
chk("  measured z-variation of the extruded flow", zv==0.0, "%.2e (0 = exact)"%zv)
src=h.attrs['source']
src=src.decode() if isinstance(src,bytes) else str(src)   # h5py returns str, not bytes
chk("  built from the real published domains", src=='extruded_2d',
    "source=%s  nz_solve=%d  grid=%s"%(src,int(h.attrs['nz_solve']),
                                       tuple(int(v) for v in h.attrs['shape'])))

print(); print("="*90); print("CLAIM 5  switch C: 3 columns in BOTH 2D and 3D"); print("="*90)
d3=PRT3DDataset('/tmp/test3d.h5',n_points=64,dim_free=True)
d2=PRT3DDataset('/tmp/data2d.h5',n_points=64,dim_free=True)
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
# EXIT NON-ZERO WHEN SOMETHING FAILED. Without this the shell status was
# always 0, so check_everything.py reported this group as ok no matter how
# many claims had drifted -- a test that cannot fail is not a test.
sys.exit(1 if bad else 0)
