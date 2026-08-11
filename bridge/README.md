# bridge/ — where 2D meets 3D

Two scripts. Everything that translates between the two halves lives here, so
`2D/` stays untouched and `3D/` never needs to know where Jung's files are.

```
python build_transfer_set.py --limit 200     # 2D/Domains -> train2d.h5
python run_experiment.py --data <3d.h5>      # trains every configuration, compares them
```

---

## Why extrusion is exact, not an approximation

Take a 2D domain and repeat it along z. The medium is now **prismatic**: nothing
varies in z, so there is no z-gradient, so the exact 3D solution is the 2D
solution repeated. An extruded 2D domain is a genuine member of the 3D problem
class — an unusually simple one, but not a fake one.

`build_transfer_set.py` does not ask you to take that on faith. It solves the
flow on the extruded volume and **measures** the z-variation of the resulting
speed field, printing it per domain and in the summary. On Jung's real domains it
comes out at exactly `0.00e+00`.

Two consequences:

**The z faces stay open.** Putting walls at `z = 0` and `z = nz-1` would make the
true 3D flow develop a Poiseuille profile across z, and the extruded field would
stop being the exact solution — the error would be a whole profile, not a
boundary layer. `--z-walls` forces the walled version if you would rather match
the 3D geometry convention than stay exact.

**The solve is z-redundant.** Since nothing varies in z, we solve on a thin slab
(`--nz-solve`, default 8 layers) and tile up to the full `nz`. Exact, and about
eight times cheaper. That is the difference between a laptop job and a cluster
job.

---

## Where the fields come from

Only the **geometries** are taken from `2D/`. The flow and concentration fields
are generated here, with the same D3Q19 Stokes solver and the same
advection-diffusion-reaction integrator used elsewhere in the 3D code.

That is deliberate. Jung's solution fields are for *his* chemistry and *his*
parameter ranges. Mixing two different reaction systems into one training set
would teach the network nothing useful — the whole premise of switch B is that
the **reaction physics** is what transfers, so both halves have to be the same
reaction physics.

---

## Two file-format traps, both handled

**Opposite array orderings.** Measured on the release:

```
Domains/domain_*.dat        reshape(148, 64) C-order     ->  1 pore cluster
geometries/Domain_*.npz     reshape(64, 148) then .T     ->  1 pore cluster
```

Apply the `.dat` convention to a `.npz` and you get 81 disconnected pore clusters
and a domain that fails percolation — silently. `ingest_2d.py` tries both
orderings per file and keeps whichever gives fewer connected components, then
prints which it chose.

**Swapped codes.** Jung: `0 = solid, 1 = pore, 2 = interface`. CompLaB:
`0 = solid, 1 = wall, 2 = pore`. Codes 1 and 2 mean opposite things. The
interface voxels are a one-voxel skin, 6.7 % of the domain; counting them as
pore gives porosity 0.493 on `Domain_Monod.npz`, inside the 0.486–0.565 range
measured on the `.dat` domains, while counting them as solid gives 0.426, which
is outside it. So pore is the default; `--interface solid` flips it.

---

## Cost

Measured on a 128 × 64 × 64 target with the default 8-layer slab and 300 Stokes
iterations: roughly **ten seconds per domain**. So 200 domains is about half an
hour on one core, and it parallelises trivially by splitting `--limit` across
jobs.

Set against the campaign it exists to shrink: **5.5 to 16 hours per 3D run**, and
240 runs per reaction formulation.

---

## Percolation

Every 2D domain is checked inlet-to-outlet before it is used, after resampling to
the target width and after keeping only the largest pore cluster. Non-percolating
domains are dropped and counted.

This is also where the topology gap shows up most clearly. A thresholded Gaussian
field percolates near porosity **0.20 in 3D** and near **0.59 in 2D** — nearly
three times higher, same morphology, same generator. That gap is exactly why
switch B transfers the trunk and retrains the geometry branch rather than
transferring the whole network.
