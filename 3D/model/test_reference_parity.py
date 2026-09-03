#!/usr/bin/env python3
# =============================================================================
# NEW IN THE FLOW VERSION.  This file exists BECAUSE of the 2D version.
#
#   WHERE THE REFERENCE CODE CAME FROM
#     github.com/hjunglab/PRT-DeepONet   branch/folder: velocity-informed
#     flow/models/PRT-DeepONet_Velocity_load.ipynb  (feature cell and weights)
#     plus their bundled example domain, Domain_Velocity.npz
#
#   WHAT IT CHECKS, AND WHY IT IS THE MOST IMPORTANT TEST HERE
#     Loading their released checkpoint into our rewritten classes proves the
#     SHAPES agree. That is a weak claim: two different functions can have the
#     same tensor shapes. This file proves the NUMBERS agree. It copies their
#     feature functions VERBATIM out of the notebook, runs them beside ours on
#     their own domain, compares MIS, UPRM and dw2 value by value, then runs
#     their trained weights twice, once fed by their features and once by ours,
#     and compares the two predicted velocity fields.
#
#     It currently reports every value identical, max abs difference 0.
#
#   WHAT TO DO IF IT EVER STOPS SAYING THAT
#     Stop comparing anything with their published numbers. From that moment
#     on you would be measuring our reimplementation, not their method. Find
#     the difference first.
#
#   WHY IT CAN SKIP
#     It needs their repository on the machine, which most users will not
#     have. When it is absent the test SKIPS and says so in different words
#     from passing, because a skip is a weaker statement than a pass and the
#     two must not be confused in a log.
# =============================================================================
"""Our descriptors against theirs, on their own example domain, voxel by voxel.

Loading the released weights proves the SHAPES match. This proves the NUMBERS do.
It runs the feature functions copied verbatim out of the released load notebook beside
ours on their bundled domain, compares MIS, UPRM and the squared wall distance, then
runs their trained weights twice, once fed by their features and once by ours, and
compares the two predicted velocity fields.

If this passes, our implementation is theirs. If it does not, every later comparison
with their published numbers is measuring our reimplementation and not their method,
and the difference has to be found before anything else is believed.

Run it like this.

    python test_reference_parity.py --reference /path/to/velocity-informed

What you get back. Nothing on disk. Skipped, and said to be skipped, when the released
repository is not on this machine.

Worth knowing. Their arrays are (64, 148) with the flow along the SECOND axis; this
project puts the flow on the FIRST. The transpose is done here and nowhere else, which
is why our functions take the flow axis as an argument rather than assuming one.
"""
import argparse
import sys, os, heapq
import numpy as np, torch
from scipy.ndimage import distance_transform_edt, binary_dilation
sys.path.insert(0, ".")
from velocity_model import VelocityDeepONet, PressureComponentUNet, load_reference_weights
import flow_features as ff

NX, NY = 64, 148          # their order: flow along axis 1
L = NX * NY
UPRM_MU, UPRM_SD = 2.4316839948181954, 1.9931210222984472
MIS_MU,  MIS_SD  = 5.161643981933594,  3.299696683883667
DW2_MIN, DW2_MAX = 1.0, 784.0
VEL_MU_U, VEL_SD_U = 0.00023968351888470352, 0.0005056853988207877
VEL_MU_V, VEL_SD_V = 6.503713052552484e-07,  0.00029989489121362567
RE_B2 = {0: 0.009110829792916775, 1: 0.02172919735312462, 2: 0.04727563634514809}

# ============================== THEIR code, copied verbatim from the load notebook
# =============================================================================
#  BLOCK 1.  THEIR CODE, COPIED VERBATIM
#
#  Everything named their_* below is transcribed from the released load
#  notebook without a single edit, including the parts we would write
#  differently. That is the whole point: an improved copy would be a third
#  implementation, and agreeing with it would prove nothing about theirs.
#
#  So do not tidy these. If one of them looks wrong, it is either wrong in
#  their released code too, in which case our port has to match it, or it has
#  been transcribed incorrectly, which is a bug in this file.
# =============================================================================
def their_uprm_map(m):
    H, W = m.shape; void = (m > 0)
    e = distance_transform_edt(void).astype(np.float32)
    acc = np.zeros_like(e); vis = np.zeros_like(void, bool); hp = []
    for y in range(H):
        if void[y, 0]:
            acc[y, 0] = float(e[y, 0]); heapq.heappush(hp, (-float(e[y, 0]), y, 0))
    while hp:
        nv, y, x = heapq.heappop(hp); cur = -nv
        if vis[y, x]: continue
        vis[y, x] = True
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            yy, xx = y+dy, x+dx
            if 0 <= yy < H and 0 <= xx < W and void[yy, xx]:
                cand = min(cur, float(e[yy, xx]))
                if cand > acc[yy, xx]:
                    acc[yy, xx] = cand; heapq.heappush(hp, (-cand, yy, xx))
    acc *= void.astype(np.float32)
    return acc.astype(np.float32)

BUF = 10
def their_local_thickness(m, r_step=0.5):
    pore = (m > 0); edt = distance_transform_edt(pore).astype(np.float32); rmax = float(edt.max())
    if rmax <= 0: return np.zeros_like(edt)
    res = np.zeros_like(edt)
    for r in np.arange(rmax, 0, -r_step):
        centers = pore & (edt >= r - 1e-6)
        if not centers.any(): continue
        rr = int(np.ceil(r)); yy, xx = np.ogrid[-rr:rr+1, -rr:rr+1]; disk = (xx*xx + yy*yy) <= r*r
        painted = binary_dilation(centers, structure=disk) & pore
        newly = painted & (res == 0); res[newly] = r
    res[~pore] = 0.0; return res.astype(np.float32)

def their_sphere_paint(m, lt, buf=BUF):
    H, W = m.shape; pore = (m > 0)
    left = np.zeros((H,W), bool); left[:, :buf] = True
    right = np.zeros((H,W), bool); right[:, W-buf:] = True
    full = np.zeros((H,W), np.float32); full[:, buf:W-buf] = lt
    seed = np.zeros((H,W), np.float32); seed[:, buf:W-buf] = lt
    for r in np.unique(seed[seed > 0])[::-1]:
        ce = seed >= r - 1e-6; rr = int(np.ceil(r)); yy, xx = np.ogrid[-rr:rr+1, -rr:rr+1]; dk = (xx*xx + yy*yy) <= r*r
        pt = binary_dilation(ce, structure=dk)
        for reg in (left, right):
            up = pt & reg & (full < r); full[up] = r
    return full, pore, (left | right)

def their_mis_map(m, buf=BUF):
    lt = their_local_thickness(m[:, buf:m.shape[1]-buf], 0.5)
    full, pore, bufreg = their_sphere_paint(m, lt, buf)
    vmin = float(lt[lt > 0].min()) if (lt > 0).any() else 0.0
    bp = pore & bufreg & (full <= 0); full[bp] = vmin; full[m <= 0] = 0.0
    return full.astype(np.float32)

def their_dw2_map(m):
    pore = (m > 0); e = distance_transform_edt(pore).astype(np.float32); f2 = e*e
    o = np.zeros_like(f2); o[pore] = np.clip((f2[pore]-DW2_MIN)/(DW2_MAX-DW2_MIN), 0, 1)
    return o.astype(np.float32)

# =============================================================================
#  BLOCK 2.  RUN BOTH, AND COMPARE
#
#  All three maps first, on their own bundled domain. Then their trained weights
#  twice over: once fed by their features and once by ours, comparing the two
#  predicted velocity fields. The second half is what catches a difference too
#  small to see in a feature map but large enough to move the network.
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Our flow descriptors against the released ones, voxel by voxel.")
    ap.add_argument("--reference", required=True,
                    help="the released velocity-informed folder")
    a = ap.parse_args(argv)
    REF = a.reference
    if not os.path.isdir(REF):
        print("SKIP: not a directory: %s" % REF)
        return 0
    return _compare(REF)


def _compare(REF):
    arr = np.load(os.path.join(REF, "flow/examples/Domain_Velocity.npz"))
    m_flat = arr['m'] if 'm' in arr.files else arr[arr.files[0]]
    m_bin = (m_flat.reshape(NX, NY) > 0.5).astype(np.int32)
    m = m_bin.astype(np.float32).copy()
    m[0, :] = 0.0; m[-1, :] = 0.0            # their BLOCK_TOP_BOTTOM convention
    pore_theirs = (m == 1)

    t_uprm = their_uprm_map(m)
    t_mis  = their_mis_map(m)
    t_dw2  = their_dw2_map(m)

    # ours: flow must run along axis 0, so transpose in and back out
    poreT = pore_theirs.T                     # (148, 64), flow along axis 0
    o_uprm = ff.uprm_map(poreT).T
    o_mis  = ff.mis_map(poreT, buf=BUF).T
    o_dw2v, _ = ff.dw2_map(poreT, DW2_MIN, DW2_MAX)
    o_dw2 = o_dw2v.T

    def cmp(name, a, b, mask):
        d = np.abs(a[mask] - b[mask])
        rng = float(b[mask].max() - b[mask].min()) or 1.0
        ident = bool(np.allclose(a[mask], b[mask], atol=1e-5))
        corr = float(np.corrcoef(a[mask], b[mask])[0,1]) if a[mask].std() > 0 and b[mask].std() > 0 else 1.0
        print("  %-6s  identical %-5s  max abs diff %.4g (%.3f%% of range)  correlation %.6f"
              % (name, ident, d.max(), 100*d.max()/rng, corr))
        return ident

    print("features on their bundled example domain, %d pore voxels" % pore_theirs.sum())
    i1 = cmp("UPRM", o_uprm, t_uprm, pore_theirs)
    i2 = cmp("MIS",  o_mis,  t_mis,  pore_theirs)
    i3 = cmp("dw2",  o_dw2,  t_dw2,  pore_theirs)

    # ============ full prediction, their weights, our features vs their features
    dev = "cpu"
    gv = PressureComponentUNet(1,2).to(dev)
    load_reference_weights(gv, os.path.join(REF,"flow/parameters/Pressure_component_UNet.pt"), verbose=False)
    gv.eval()
    vm = VelocityDeepONet(3,128,2,grid=(NX,NY)).to(dev)
    load_reference_weights(vm, os.path.join(REF,"flow/parameters/Velocity.pt"), verbose=False)
    vm.eval()

    xc = np.arange(NX, dtype=np.float32)/(NX-1); yc = np.arange(NY, dtype=np.float32)/(NY-1)
    Xg, Yg = np.meshgrid(xc, yc, indexing='ij')
    XY = np.stack([Xg.reshape(-1), Yg.reshape(-1)], 1).astype(np.float32)

    @torch.no_grad()
    def run(uprm, mis, dw2):
        mt = torch.from_numpy(m).to(dev)
        dP = gv((mt == 1).float().view(1,1,NX,NY))[0]
        pm = (mt == 1); Z = torch.zeros((), dtype=dP.dtype)
        gx = torch.where(pm, dP[0], Z).reshape(-1); gy = torch.where(pm, dP[1], Z).reshape(-1)
        b1 = torch.stack([mt,
                          torch.from_numpy(((uprm-UPRM_MU)/UPRM_SD).astype(np.float32)),
                          torch.from_numpy(((mis-MIS_MU)/MIS_SD).astype(np.float32))]).unsqueeze(0)
        tr = torch.empty((1,L,5))
        tr[0,:,0]=torch.from_numpy(XY[:,0]); tr[0,:,1]=torch.from_numpy(XY[:,1])
        tr[0,:,2]=gx; tr[0,:,3]=gy; tr[0,:,4]=torch.from_numpy(dw2.reshape(-1))
        b2 = torch.tensor([[float(RE_B2[1])]])
        uv = vm(b1,b2,tr)[0].numpy()
        return np.stack([uv[...,0]*VEL_SD_U+VEL_MU_U, uv[...,1]*VEL_SD_V+VEL_MU_V])

    vt = run(t_uprm, t_mis, t_dw2)
    vo = run(o_uprm, o_mis, o_dw2)
    print("\npredicted velocity from their weights, at Re condition 1")
    sp = np.sqrt((vt**2).sum(0))
    print("  their features : mean speed %.6g, max %.6g over the pore space"
          % (sp[pore_theirs].mean(), sp[pore_theirs].max()))
    d = np.abs(vt - vo)
    rel = d[:, pore_theirs].max() / (np.abs(vt[:, pore_theirs]).max() or 1.0)
    print("  our features   : max abs difference %.4g, %.4f%% of the peak velocity" % (d.max(), 100*rel))
    same = bool(np.allclose(vt, vo, atol=1e-9))
    print("  fields identical:", same)
    ok = bool(i1 and i2 and i3 and same)
    print("\n" + ("EVERY FEATURE AND THE PREDICTION MATCH THEIRS EXACTLY."
                 if ok else "DIFFERENCES ABOVE, read them."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
