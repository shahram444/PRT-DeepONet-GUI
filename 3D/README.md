# GeometryAware3D - the PRT-DeepONet-3D side

**Shahram Asgari and Christof Meile, Meile Lab, Department of Marine Sciences,
University of Georgia.** Everything in this folder is ours, apart from the two
dimensional velocity classes and flow descriptors noted in `../LICENSING.md`.

Everything that is not CompLaB. CompLaB itself is a separate C++ project; what
lives here builds its input, reads its output, and trains on the result.

## Every file in here, and what it is for

Lines are given so you can tell a one-page utility from a solver. **NEW** and
**CHANGED** mark what the flow work added or touched.

```
3D/
├── README.md                              this file
├── SWITCHES.md                            the switches and the flow pipeline, in full
│
├── tools/    geometry, simulation, datasets
│   │
│   │  ---- the two Python simulators ----------------------------------------
│   ├── prtlb_2d.py                 2566   flow, transport and reactions in 2D
│   ├── prtlb_3d.py                 2535   the same in 3D. The transport and reaction
│   │                                      code is word for word identical to the 2D
│   │                                      file, and test_flow_solvers.py checks it
│   ├── settings_and_units.py       2974   every physical number in one place, with
│   │                                      its units and its provenance
│   │
│   │  ---- making pore structures -------------------------------------------
│   ├── build_geometry_3d.py         608   Gaussian random media at a chosen porosity
│   ├── build_toy_pack_2d.py         279   a small 2D packing, for tests and demos
│   │
│   │  ---- building datasets ------------------------------------------------
│   ├── build_dataset_2d.py          973   runs the 2D simulator over a sweep of
│   │                                      conditions and writes one dataset.h5
│   ├── build_dataset_3d.py          934   the same in 3D
│   ├── build_practice_dataset.py    203   a small but REAL dataset in five seconds
│   ├── make_demo_complab.py         739   writes a demo campaign in the exact layout
│   │                                      finished CompLaB output has. The numbers are
│   │                                      constructed and it says so
│   │
│   │  ---- the CompLaB interface --------------------------------------------
│   ├── complab_campaign.py          585   writes the input files a cluster run needs,
│   │                                      in batches, and tracks and retries them
│   ├── collect_complab_output.py    718   a campaign this project set up -> dataset.h5  CHANGED
│   ├── collect_foreign_complab.py  1686   runs somebody else set up, by hand, with or
│   │                                      without their input files beside them        CHANGED
│   ├── import_2d_simulations.py     396   2D simulation output somebody sent you
│   │
│   │  ---- 2D into 3D, for switch B -----------------------------------------
│   ├── build_transfer_set_2d_to_3d.py 458  extrudes 2D domains into 3D blocks
│   ├── load_pretrained_2d_weights.py  207  converts the published 2D weights into ours
│   │
│   │  ---- what the model reads ---------------------------------------------
│   ├── dataset_reader.py            597   the PyTorch Dataset. THE HINGE OF THE
│   │                                      PROJECT: the one file that knows both what
│   │                                      an .h5 holds and what the model wants, so
│   │                                      every switch passes through it            CHANGED
│   ├── flow_coordinates.py          437   travel time and stream coordinates, switch A
│   │
│   │  ---- the flow descriptors ---------------------------------------------
│   ├── flow_features.py             573   MIS, UPRM and the squared wall distance,
│   │                                      in two and three dimensions                   NEW
│   ├── harmonic_pressure.py         337   the sparse Laplace solve that gives the
│   │                                      velocity operator's trunk its prior           NEW
│   ├── add_flow_features.py         231   puts all three into a dataset you already
│   │                                      have, without recollecting it                 NEW
│   │
│   │  ---- the tests --------------------------------------------------------
│   ├── test_flow_solvers.py         402   the 2D and 3D solvers against each other
│   ├── test_three_switches.py       263   proves switches-off is bit-identical to the
│   │                                      original, by re-implementing it inside itself
│   └── test_documented_numbers.py   418   every number quoted in the docs, against
│                                          the code that is supposed to produce it
│
└── model/    the neural operator
    │
    │  ---- concentration ----------------------------------------------------
    ├── deeponet_model.py            200   the architecture: CNN branch, dense branch,
    │                                      trunk, FiLM
    ├── train.py                     372   the training loop, and every feature flag     CHANGED
    ├── evaluate.py                  535   scores a checkpoint on structures it never saw
    ├── predict.py                   485   answers about one new structure, in under a
    │                                      second, with no simulation
    ├── run_ablation_sweep.py        204   trains every switch configuration and
    │                                      compares them on the SAME held-out rocks
    ├── make_figures.py              273   the figures, from a finished run
    ├── train.sbatch                       Slurm submit. Set your own partition
    │
    │  ---- velocity --------------------------------------------------------
    ├── velocity_model.py            665   the velocity operator, the pressure U-Net,
    │                                      the ROI Huber loss and the divergence
    │                                      penalty, in 2D and 3D                         NEW
    ├── train_velocity.py            538   trains the operator                           NEW
    ├── predict_velocity.py          412   runs it and writes samples/velocity_pred
    │                                      beside samples/velocity, never over it        NEW
    │
    │  ---- the tests -------------------------------------------------------
    ├── test_flow_pipeline.py        394   the whole velocity path end to end, on a
    │                                      dataset it builds for itself. 33 checks       NEW
    └── test_reference_parity.py     244   our descriptors and our predicted field
                                           against THEIRS, on their own bundled
                                           domain, value by value                        NEW
```

Nothing generated is kept here. Geometries, datasets, checkpoints and figures
are all written under `work/` at the project root.

A per-file guide to both folders is in `docs/`: `GUIDE_3D_tools.docx` and
`GUIDE_3D_model.docx`.

**Every file marked NEW or CHANGED says so at the top of itself**, in a block headed
`CHANGED FROM THE 2D VERSION`: where the code came from, what their version does, and
what was changed and why.

---


## The three feature switches, and the flow pipeline  (all default OFF)

A, B and C were added after the July meeting; the flow pipeline after the velocity
informed follow-up. It is listed here because it is a flag on the same script, not
because it works like the other three.
With all of them off the code behaves exactly as it did before; `tools/test_three_switches.py` proves it bit-exactly.  Full documentation
in **SWITCHES.md**.

| switch | flag | what it does |
|--------|------|--------------|
| A | `--flow-proxy` | use the FLOW FIELD instead of the geodesic distance. Trunk gets the advective travel time `tau`, branch gets the normalised velocity. This is Christof's "can we just take the flow field instead of the GDF?" |
| B | `--transfer-2d H5` | mix extruded 2D domains into training. An extruded 2D domain is an EXACT 3D problem, so this is free training data, not an approximation. Build the file with `tools/build_transfer_set_2d_to_3d.py`. |
| C | `--dim-free` | A and B together: the flow-space trunk `(t, dwall, tau)`, which has the SAME number of inputs in 2D and 3D and therefore transfers with no modification. |
| the flow pipeline | `--velocity-informed` | give the CONCENTRATION branch the velocity field as extra image channels and leave the trunk alone. `simulated` uses the field the solver produced, `predicted` uses one from the velocity operator. This is the published follow-up, and it is NOT switch A. |

**The flow pipeline is reached differently from the other three.** A, B and C are tick boxes on
the Train page. D is a panel: **The flow field -> The flow pipeline**, in the sidebar,
in both modes. Six shared boxes down the left so every step reads the same dataset, five
cards down the right, each with its own state, its own command line and its own Run
button, and one button that runs them in order and stops at the first failure. The flag
on `train.py` is unchanged; only the way in has moved. The reason is that A, B and C
each change one training run and need nothing prepared, while D needs three earlier
stages to have run against the same dataset first.

```
python tools/build_practice_dataset.py --out /tmp/test3d.h5     # a small but real dataset
python tools/test_three_switches.py --data /tmp/test3d.h5   # 29 checks
python model/run_ablation_sweep.py --data 3d.h5 --data-2d train2d.h5 --out sweep
```


## The geometries

None ship with the repository. `tools/build_geometry_3d.py` writes them, one
directory per geometry, each holding what CompLaB reads and what the model
reads:

```
geom_0000/
    geometry.dat            <- CompLaB reads this
    geometry.dims
    geom_0000.npz           <- material, material_jung, gdf, edt, meta
    geom_0000_slices.png    <- 2D slices, geodesic field, tortuosity excess (--render)
    geom_0000_3d.png        <- 3D render (--render3d)
manifest.csv                <- porosity, correlation length, tortuosity for all of them
```

### These match Jung's morphology, measured not guessed

The generator was written against measurements of the published release, not
against a description of it. `2D/geometries/Domain_Monod.npz` and several
`2D/Domains/domain_*.dat` were read directly. What they actually are:

**Shape (148, 64)**, x = 148 being the flow direction, stored one value per line
in C order with y fastest. Reshaping them as (64, 148) produces horizontal
streaks — that is how you can tell the ordering is wrong.

**Codes 0 = solid, 1 = pore, 2 = interface.** I could determine this because the
first and last x-slices are 100% code 1 — pure-fluid inlet and outlet buffers.
Code 2 is only 6.7% of the domain and forms 1-voxel outlines.

**Morphology: not sphere packs.** Organic, bicontinuous blobs — the signature of
a thresholded correlated Gaussian random field. Measured two-point correlation
length 6–8 pixels, isotropic. A Gaussian filter of σ ≈ 4 reproduces that in 3D.

**Porosity 0.486 / 0.495 / 0.524 / 0.609** across the four domains sampled, so
the 3000-domain training set spans roughly 0.45 to 0.65.

The generator reproduces all of it: `--style blobs` (the default) thresholds a
correlated random field with σ drawn from 3.0–4.5, porosity stratified 0.45–0.65,
and pure-pore inlet/outlet buffers. `--style spheres` gives a second, independent
morphology if you want a generalisation test.

### Two deliberate differences from Jung, both forced by CompLaB

**Codes are swapped.** `geometry.dat` uses CompLaB's convention, which the XML
declares as `<pore>2</pore><solid>0</solid><bounce_back>1</bounce_back>`:

| | 0 | 1 | 2 |
|---|---|---|---|
| **here** | solid | interface / bounce-back | pore |
| **Jung** | solid | pore | interface |

The `.npz` carries both — `material` (CompLaB) and `material_jung` (Jung) — so
the ML side can use either. Do not mix them: an unknown code silently becomes
reactive fluid (`complab_functions.hh:243-274`).

**Transverse walls exist here, and do not in Jung's 2D.** CompLaB has no periodic
boundary option anywhere — `grep -r periodic src/` returns nothing, and
`complab_functions.hh:243-279` sets up only the west and east faces. Pore voxels
on the y or z domain edge would stream from unallocated space. So a 2-voxel
bounce-back shell is added, making this a walled column rather than Jung's
unbounded medium. `--wall 1` minimises the perturbation; `--wall 0` is refused.

### Regenerate

```bash
python tools/build_geometry_3d.py --n 140 --out ./geometries --render --render3d --render3d-every 10
python tools/build_geometry_3d.py --n 140 --out ./geometries --nx 148     # Jung's aspect ratio
```

About 2 s per geometry, plus ~28 s for each 3D render. Deterministic from
`--seed`.

---

## tools/

```bash
python complab_campaign.py build --geometries ../geometries --out ./complab_campaign \
       --complab /path/to/CompLB3D/build/complab \
       --per-geom 20 --batches 5 \
       --partition batch --cores 8 --time 04:00:00 \
       --modules foss/2023a --launcher mpirun --concurrent 100

cd complab_campaign/batch_1 && sbatch submit.sbatch    # or: bash complab_campaign/submit_all.sh
python complab_campaign.py status --out ./complab_campaign
python complab_campaign.py retry  --out ./complab_campaign

python collect_complab_output.py --campaign ./complab_campaign --geometries ../geometries \
                  --out ./dataset --mode steady
```

Each run gets its own directory with its own `CompLaB.xml`, `input/geometry.dat`,
`output/` and `env.sh`. Nothing is shared, so nothing can collide.

`--batches 5` splits the campaign into five independent sub-campaigns of 560 runs
each, **by geometry**, so any batch can be submitted, collected and trained on by
itself. Run ids stay globally unique, so batches merge without collision.
`complab_campaign.py status` and `retry` walk every batch, and `collect_complab_output.py --campaign`
accepts the campaign root, several paths, or one batch alone.

`CompLaB.xml.template` is **not a CompLaB file** — CompLaB never sees it. It is an
input to `complab_campaign.py`, which renders one real `CompLaB.xml` per run with a
different Peclet number.

### Steady state or a time series

CompLaB writes a `.vti` every `--vtk-interval` iterations, so a run already
holds a full time series; `collect_complab_output.py` decides what is kept.

```bash
python collect_complab_output.py --campaign ./complab_campaign --geometries ../geometries \
                  --out ./dataset --mode steady               # T=1, final state

python collect_complab_output.py --campaign ./complab_campaign --geometries ../geometries \
                  --out ./dataset --mode transient --n-times 6   # T=6, evolution
```

`--n-times` is not optional in practice. `<ade_converge_iT>` lets each run stop
when it reaches steady state, so runs end at different iterations and hold
different snapshot counts; stacking those raw either fails on the shape or, far
worse, puts different simulated times in the same slot. `--n-times K` resamples
every run onto `t = 0, 1/(K-1), ..., 1` of its own run length, so slot k means
the same fraction of the approach to steady state in every run — which is what
the trunk's `t` input has to mean for the model to learn anything from it.

The file grows linearly in K. One batch of 560 runs at K=6 is about 7 GB in
float16; all five batches at K=6 is about 35 GB, so collect one batch at a time unless the
training node has the room.

The trunk follows the file automatically. A steady dataset drops `t`, because a
constant input is a dead input, giving `(x, y, z, gdf)` — the 3D analogue of the
2D paper's `p_S(x, y, GDF)`. A transient dataset keeps it: `(x, y, z, t, gdf)`,
the analogue of `p_T(x, y, GDF, t)`. `--with-time` / `--no-time` on `train.py`
force it either way. The checkpoint records which it was, so `predict.py` and
`evaluate.py` cannot silently disagree with it.

### Failure isolation

Slurm array tasks are independent processes. `run_one.sh` deliberately avoids
`set -e` and always exits 0 — it records the failure and gets out of the way.
Every run writes `status.json` whether it worked or not.

| stage | catches | example reason |
|---|---|---|
| run | crash, timeout, missing input, no output | `timeout`, `missing_geometry`, the first `ERROR`/`Terminating` line from `run.log` |
| files | `.vti` missing or unreadable | `no .vti for species 'Bio'` |
| data | finished but the fields are unusable | `all concentration fields are identically zero`, `non-finite values`, `concentration 3.2 exceeds bound 1.0 (blow-up)`, `12.4% of voxels negative` |

`campaign_report.md` groups every failure by cause then lists them run by run;
`failures.csv` has the raw detail. Tested on 12 runs with three injected failures
— missing geometry, solver crash, all-zero output — all 12 tasks completed, 9
collected, each failure reported with its own specific cause.

Run-stage failures are worth re-queueing. Data-stage failures are physics
failures: re-running the same parameters reproduces them exactly, so the fix is
to check `PRT_DT`, lower `PRT_MAXFRAC`, or exclude that corner of parameter
space. `retry` deliberately skips them.

### Parameter space

Six dimensionless groups, Latin-hypercube sampled in log space
(`--sampling grid` gives a full grid for the Pe–Da heatmap figure):

| group | range | set through |
|---|---|---|
| `Pe = uL/D_Ac` | 1 – 100 | `<Peclet>`; CompLaB rescales ΔP to hit it |
| `Da_bio = (Vmax·B0/Ac0)·L²/D_Ac` | 0.1 – 100 | `PRT_VMAX` |
| `Da_abio = k_surf·L²/D_P` | 0.1 – 100 | `PRT_KSURF` |
| `Ks_Ac/Ac0` | 0.01 – 1 | `PRT_KS_AC` |
| `Ks_A/A0` | 0.01 – 1 | `PRT_KS_A` |
| `Y·Ac0/B0` | 0.01 – 0.2 | `PRT_Y` |

Damköhler is defined diffusively, so it is velocity-independent — changing it
does not touch the flow field.

---

## model/

```bash
python model/train.py --data tools/dataset/dataset.h5 --out runs/gdf
python model/train.py --data ... --out runs/edt  --distance edt      # ablation
python model/train.py --data ... --out runs/none --distance none     # ablation
python model/train.py --data ... --out runs/vel  --with-velocity     # flow-conditioned
```

Those three `--distance` runs are the ablation that carries the paper: they show
that the *geodesic* field, not just any distance field, is what buys the accuracy.

### Using a trained model

```bash
# 1. accuracy on held-out geometries, plus the paper's figures
python model/evaluate.py --data dataset/dataset.h5 --out figs/ablation \
    --compare geodesic=runs/gdf/best.pt euclidean=runs/edt/best.pt none=runs/none/best.pt

# 2. predict on a brand-new geometry, no simulation at all
python model/predict.py --checkpoint runs/gdf/best.pt \
    --geometry geometries/geom_0007/geom_0007.npz \
    --pe 12.5 --da-bio 3.2 --da-abio 0.8 --ks-ac 0.1 --ks-a 0.15 --y 0.04 \
    --out pred_geom0007

# 3. the same, as a time series: one column per t, the 3D analogue of the
#    snapshot strip on the right of the 2D paper's architecture figure
python model/predict.py --checkpoint runs/gdf/best.pt \
    --geometry geometries/geom_0007/geom_0007.npz \
    --pe 12.5 --da-bio 3.2 --da-abio 0.8 --t-series 6 --out pred_geom0007
```

`--t-norm` picks one time; `--t-series K` sweeps K of them and writes
`time_series.png` (rows = biotic rate, abiotic rate, then each species; columns
= time; one colour scale per row so the row reads as an evolution) and
`time_series.npz`. Both are ignored with a note on a steady checkpoint, which
has no time input to sweep.

`evaluate.py` writes `metrics.json` (per-species RMSE and R², speedup),
`rmse_table.csv` (one row per held-out sample), `fields_*.png`
(truth / prediction / absolute error slices), `physics_*_2d.png` and
`physics_*_3d.png` (flow, biotic rate and abiotic rate, truth against
prediction), `rmse_vs_params.png`, and `ablation.png` when `--compare` is used.
`--no-3d` skips the 3D renders, which are the slow part.

`predict.py` writes `pred.npz` (all fields as a dense `(n,nx,ny,nz)` volume),
one `.vti` per species plus `R_bio_pred.vti` and `R_abio_pred.vti` that open in
ParaView next to CompLaB's own output, and the figures below. It accepts either
the geometry `.npz` or a raw CompLaB `geometry.dat` with `--nx --ny --nz`,
computing the geodesic field itself if the file does not already carry one.

### Flow, biotic and abiotic fields, in 2D and in 3D

Every figure from `predict.py` and `evaluate.py` leads with the same three
quantities, because those are what a reactive-transport reader looks for:

| panel | what it is | where it comes from |
|---|---|---|
| **FLOW** `\|u\|` | velocity magnitude | the field you pass with `--velocity`, or CompLaB's own `u.vti` |
| **BIOTIC** `R_bio` | `Da_bio · Bio · [Ac/(Ks_Ac+Ac)] · [A/(Ks_A+A)]` | derived from the predicted concentrations with the same double-Monod law as `defineKinetics.hh` |
| **ABIOTIC** `R_abio` | `Da_abio · P` | the same first-order law as `defineAbioticKinetics.hh` |

then the individual species. `*_2d.png` is a mid-plane slice with the solid
masked grey; `*_3d.png` is a half-cut 3D point cloud through the pore space with
the grains drawn as a translucent marching-cubes skin. In `evaluate.py` the
truth and prediction panels of a given quantity share one colour scale, so what
you see is a real difference and not two independently auto-scaled images.

`predict.py` will not draw a flow field that does not belong to the geometry:
if more than 2% of solid voxels in the supplied velocity file carry non-zero
velocity it stops with an error instead of producing a plausible-looking but
wrong FLOW panel.

Measured on CPU with an untrained-quality checkpoint: 0.17 s geometry prep
(cacheable) and 1.08 s inference over 125,432 pore voxels at 64³. Against a
30-minute CompLaB run that is a speedup of roughly 1700x on CPU alone.

### Checked against the 2D paper's architecture figure, block by block

| 2D paper | here | same? |
|---|---|---|
| `B_C` geometry branch: 5 × Conv2D(3×3) + SiLU + AvgPool2D(2×2), channels 16/32/64/128/256, then flatten → Linear → SiLU, 128 nodes | 5 × Conv3D(3×3×3) + SiLU + AvgPool3D(2×2×2), same channel ladder, flatten → Linear, 128 | yes, one dimension up. 64³ through 5 halving blocks is 2×2×2×256 = 2048, exactly the 2D model's 2×4×256 flatten width |
| `B_F` parameter branch: 3 × Linear, SiLU on 1–2, none on 3, in 2 or 3 → 128 | 3 × Linear, SiLU, none on the last, in **6** → 128 | yes, wider input |
| `T` trunk: 8 × Linear, SiLU on 1–7, none on 8, in 3 or 4 → 128 | 8 × Linear, same activation pattern, in **4 or 5** → 128, plus FiLM re-injection of the geodesic column every 3rd layer | yes, plus one addition |
| `s = G(w, r)(p)` — branch ⊙ branch, dotted with the trunk | `branch1(w) * branch2(r)` → per-species coefficients → `einsum` with the trunk | yes |
| `p_S(x, y, GDF)` steady / `p_T(x, y, GDF, t)` transient | `(x, y, z, gdf)` steady / `(x, y, z, t, gdf)` transient, chosen automatically from the dataset | yes |
| `r_A(Pe, Da_A)`, `r_D(Pe, Da_A, Da_D)`, `r_M(Pe, Da_M)` — separate models per reaction system | ONE model, `r = (Pe, Da_bio, Da_abio, Ks_Ac/Ac0, Ks_A/A0, Y·Ac0/B0)` | deliberately different |
| output: one field per snapshot, snapshots along t | `(B, P, n_species)` for 4 species at once, swept over t by `--t-series` | yes, multi-species |

Two deliberate departures, both stated so a reviewer sees them coming.

**One model instead of three.** The 2D paper trains a separate operator per
reaction system: abiotic only, abiotic-plus-decay, Monod. Here biotic and
abiotic run in the same simulation and the same network, conditioned on both
`Da_bio` and `Da_abio` in one 6-vector. That is what "biotic and abiotic
capability" means for this project, and it is why the parameter branch takes 6
inputs rather than 2 or 3. The cost is that the model must learn a
higher-dimensional parameter manifold from the same number of runs; the benefit
is that coupled biotic–abiotic behaviour is representable at all, which three
separate single-mechanism models cannot do.

**FiLM re-injection in the trunk.** An addition, not a port. See below.

### What changed from the 2D model, and why

**Conv2d → Conv3d, AvgPool2d → AvgPool3d.** Nothing else in the branch changes: a
64³ volume through 5 halving blocks gives 2×2×2 at 256 channels = **2048 values,
exactly the flatten dimension of the 2D model's 2×4×256**. The encoder ports over
verbatim.

**The trunk takes 5 inputs and is subsampled.** Your 2D model evaluates the trunk
at all 9 472 grid points. The same thing at 64³ is 262 144 points — a 26.8 GB
activation tensor at batch 25. At 8 192 sampled pore voxels the trunk costs
**944 MMACs, slightly less than your 2D model's 1 091**, and memory stays near
0.3 GB. That is why the 3D model trains on a laptop. Whole model: 1.66 M
parameters.

**Multi-species output through one shared trunk.** The branch emits
`n_species × p` coefficients against a single trunk. The trunk is the part
exposed to the per-voxel cost, so duplicating it per species would multiply the
dominant cost for no benefit.

**Geometry re-injection.** `Trunk(inject_every=3)` FiLM-modulates every third
hidden layer with the geodesic column. Deep networks with a single early geometry
input provably lose that information with depth — "Do Neural Operators Forget
Geometry?" (2026) formalises it via a data-processing-inequality argument. Your
2D trunk is shallow enough not to care; a 3D one is not. Set `--inject-every 0`
to reproduce the plain 2D behaviour and ablate it.

### The dataset layer

`dataset_reader.py` produces exactly the 4-tuple your 2D model already takes:

```python
from dataset_reader import PRT3DDataset, split_by_geometry
tr, te = split_by_geometry("dataset/dataset.h5", frac=0.15)
train = PRT3DDataset("dataset/dataset.h5", indices=tr, n_points=8192)
# branch1 (1,64,64,64)   branch2 (6,)   trunk (8192,5)   target (8192,4)
```

It splits by **geometry**, never by sample — a sample split leaks pore structure
between train and test and inflates the score. Use `full_grid=True` at evaluation
and `scatter_to_volume()` to rebuild a dense volume.
