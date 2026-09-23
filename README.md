# PRT-DeepONet Studio, flow-aware

A geometry-aware neural operator for pore-scale reactive transport, together
with **PRT-LB**, the lattice Boltzmann simulator that generates its training
data, and a desktop application that drives the whole workflow.

Pore-scale reactive transport is accurate and slow: one three-dimensional
simulation takes hours. This project builds, trains and checks a neural
operator that predicts the same fields in milliseconds.

```
PRT-DeepONet/
├── prt.py        one command line for everything, with no window
├── prt_core/     the network, the chemistry, the distance conventions, the
│                 geometry reader. Defined once, used by 2D and 3D alike
├── 2D/           the published PRT-DeepONet release of Kim and Jung, unmodified
├── 2D_scripts/   that release as runnable modules instead of notebooks
├── 3D/           the geometry-aware 3D extension
├── bridge/       the one place the two halves meet
├── gui/          the desktop application
├── docs/         one guide per part of the code
├── tests/        unit tests that need no simulation and no training
└── examples/     example input files, with a README
```

Nothing here needs the window. `prt.py` runs every step from a terminal with
the same flags and the same defaults the buttons use, which is what a cluster
job needs and what to fall back on if the window will not start:

```bash
python prt.py                 the commands, grouped in the order they are run
python prt.py help train      the full help for one of them
python prt.py core            check the shared core
```

**Already have simulation output?** Go straight to
[From the data you already have to a trained model](#from-the-data-you-already-have-to-a-trained-model).
It runs from a folder of `.vti` files or an existing `dataset.h5` through to a
prediction, and says after each command what you should see and what to do
about it.

---

## Check the checkout before you use it

Two commands, neither of which trains anything.

```bash
python smoke_test.py        # about 40 seconds, the deploy gate
python run_tests.py         # everything, including the slow physics
```

`smoke_test.py` is the fast one: it starts with `prt_core/test_core.py`, which
checks the four things everything else imports, and then imports every module,
builds the network,
builds a small synthetic dataset from analytic fields, writes an untrained
checkpoint, and drives `evaluate.py` and `predict.py` end to end on them.
Random weights travel the same load, build, inference and write path that
trained weights do, so a renamed checkpoint key, a shape that stopped
matching, or a missing package fails in seconds rather than after an hour of
training.

`run_tests.py` runs the same checks in named groups and adds the lattice
Boltzmann self-tests, which take minutes. Its report says, for anything that
fails, what the check was protecting and what to do about it, and it ends with
an order to fix things in. Useful flags:

```bash
python run_tests.py --quick              # skip the slow physics group
python run_tests.py --group model --group data
python run_tests.py --report report.md   # write the report to a file
python run_tests.py --window             # also test the window, needs a display
```

A green run means the plumbing holds: the code imports, the shapes agree, every
feature switch is genuinely off when off, and the buttons send commands the
scripts accept. It says nothing about accuracy. That comes from training and
reading the held-out error.

---

## From the data you already have to a trained model

This is the whole workflow, in the order you do it. It assumes you already have
simulation output: either a folder of `.vti` files from CompLaB, or a
`dataset.h5` somebody handed you.

Everything below runs from the repository root, with no window. The same
commands work in the window, and the window shows you the command it will run.

---

### Step 1. What do you have

| what is on disk | which step to start at |
|---|---|
| folders of `.vti` from CompLaB, one folder per run | step 2 |
| a CompLaB campaign this project wrote, with `params.json` beside each run | step 2, the campaign route |
| a `dataset.h5` already | step 3 |
| nothing yet, and you want to see the chain work | `python prt.py practice --out work/d.h5 --n-geom 6 --shape 24 24 24` |

A `.vti` on its own is anonymous: it holds an array of numbers on a grid and
nothing else. No Peclet number, no Damkohler number, no note of which rock it
was computed on. Step 2 is where those get attached, and it is the step that
decides whether anything downstream means what you think it does.

---

### Step 2. Turn the `.vti` files into one `dataset.h5`

**Look before you write.** `--inspect` reads everything and writes nothing.

```bash
python prt.py collect-foreign --inspect --runs work/my_output
```

**What you see,** one block per run:

```
run_Pe10_Da1               rock 0  grid (64, 64, 64)
     pore code 2, read from the run's CompLaB.xml
     input file        CompLaB.xml
     kinetics          defineKinetics.hh
     snapshots at [0, 2000, 4000, 6000, 8000]
     highest concentration in pore space 1.002
     conditions: pe=10 (CompLaB.xml), da_bio=1 (kinetics)
======================================================================
usable: 24 of 25
   rejected  run_Pe3_Da10             every value is zero
```

**What to do with it.** Three lines are worth reading carefully.

- **`pore code`.** CompLaB's material codes are set per run, so a geometry file
  written by one setup can mean the opposite of one written by another. If this
  says the wrong thing, pass `--pore-code N`. Getting it wrong trains the
  network on the rock instead of the pore space and nothing later complains.
- **`conditions`.** Anything listed under `NOT FOUND` goes in as zero. Supply it
  with `--pe`, `--da-bio`, `--da-abio`, or point at the input file with `--xml`.
- **`rejected`.** Each rejection carries its reason. A few are normal; most of
  the runs rejected means something systematic, usually the wrong `--species`
  glob.

**Then write it.**

```bash
python prt.py collect-foreign --runs work/my_output --out work/dataset \
              --species Ac=Ac_*.vti A=A_*.vti
```

**What to write, and what it means.**

| you write | what it means |
|---|---|
| `--runs work/my_output` | the folder that HOLDS the run folders, not one run |
| `--out work/dataset` | a folder. `dataset.h5` is written inside it |
| `--inspect` | read everything, write nothing. Always do this first |
| `--species Ac=Ac_*.vti A=A_*.vti` | one entry per chemical: the NAME the dataset will use, then the file pattern. Default is `C=subsLattice0_*.vti` |
| `--geometry-file inputGeom.vti` | the rock inside each run folder. If absent, `geometry.vti` and `maskLattice*.vti` are tried too |
| `--pore-code 2` | which number in the geometry means pore. Normally read from the run's `CompLaB.xml`; set it only when inspect got it wrong |
| `--xml PATH` | the `CompLaB.xml` these runs used, if it is not beside them. This is where the conditions come from |
| `--pe 10 --da-bio 1 --da-abio 1` | supply a condition by hand when no file carries it. Otherwise it goes in as zero |
| `--flow-glob nsLattice*.vti` | the velocity snapshots. `--no-velocity` skips them |
| `--rate-glob rateLattice*.vti` | the reaction rate fields, if the run wrote them |
| `--params pe_da` | two dimensionless numbers, which is what the network expects. `full` records six |

You get `work/dataset/dataset.h5`.

**If the runs came from this project's own campaign builder**, they carry a
`params.json` and a `status.json`, so the conditions need no recovering:

```bash
python prt.py collect --campaign work/campaign --geometries work/geometries \
              --out work/dataset --mode transient --n-times 6
```

| you write | what it means |
|---|---|
| `--campaign work/campaign` | the campaign folder. Several are allowed and they merge into one dataset |
| `--geometries work/geometries` | where the rocks live |
| `--mode transient` | keep a time series, which is what makes the trunk's `t` input mean anything. `steady` keeps the final snapshot only |
| `--n-times 6` | resample every run onto 6 equally spaced times. Needed in practice, because runs converge at different iterations and otherwise hold different snapshot counts. The file grows linearly in it |
| `--max-conc 2.0` | reject a run whose peak concentration exceeds this, in mol/L |

That route also writes `failures.csv` and `campaign_report.md` beside the
dataset, one row per rejected run with its reason.

---

### Step 3. Look inside the `.h5` before you train on it

```bash
python prt.py dataset-info work/dataset/dataset.h5
```

**What you see.** Two reports. The first is what the FILE holds, which is a
fact. The second is what the MODEL will be handed, which is an interpretation.
A dataset that behaves strangely is almost always a disagreement between them.

```
attributes
  param_names      ['pe', 'da']
  shape            [16 16 16]
  species          ['Ac', 'A']

arrays
  /geom/edt               (2, 16, 16, 16)          float32
  /geom/gdf               (2, 16, 16, 16)          float32
  /geom/gid               (2,)                     int32
  /geom/material          (2, 16, 16, 16)          uint8
  /samples/conc           (2, 3, 2, 16, 16, 16)    float16
  /samples/geom_index     (2,)                     int32
  /samples/params         (2, 2)                   float32
  /samples/run_id         (2,)                     int32
  /samples/settled        (2,)                     bool
  /samples/t_norm         (2, 3)                   float32
  /samples/velocity       (2, 3, 16, 16, 16)       float16

samples attributes
  conc_scale       [0.9999651 0.9999856]

species     : ['Ac', 'A']  predicting: Ac
grid        : (16, 16, 16)  snapshots: 3  fields: 2
dimension   : 3D  (grid (16, 16, 16))
switches    : OFF  trunk (x, y, z, t, gdf)
branch1     : 1 ch  (material)
trunk       : 5 inputs  (x, y, z, t, gdf)
train / test: 1 / 1 samples (split by geometry)
branch1 (1, 16, 16, 16) branch2 (2,) trunk (2349, 5) target (2349,)
```

**How to read it.**

| line | what it means |
|---|---|
| `/geom/material` | one entry per ROCK, not per run. `(G, nx, ny, nz)` |
| `/geom/gdf`, `/geom/edt` | the travel distance from the inlet through the pore, and the straight-line distance to the nearest grain |
| `/samples/conc` | `(runs, snapshots, species, nx, ny, nz)`. Stored as float16 and already divided by `conc_scale` |
| `/samples/params` | the conditions of each run, named by `param_names` |
| `/samples/geom_index` | which rock each run used. This is what lets whole rocks be held out |
| `/samples/settled` | False means the run hit the step cap rather than reaching steady state |
| `dimension` | 3D, or 2D when the third grid number is 1. Nothing selects this by hand |

**What to do.** If `species`, `param_names` or the grid is not what you expect,
go back to step 2. Training on a file whose species list is wrong costs hours
and produces a model nobody can interpret.

---

### Step 3b. Or open the same file in the viewer

```bash
python prt.py gui
```

Viewer tab, then **Dataset (.h5)**. The left side is the file itself, laid out
the way an HDF5 browser lays it out, with a plain-language line beside each
entry:

```
geom          the rocks, as they started. One entry per rock.
  material    the pore structure. 0 solid, 1 interface, 2 pore.
  gdf         distance from the inlet, measured through pore space.
  edt         distance to the nearest solid.
  gid         identity number of each rock.
samples       the runs. One entry each.
  conc        every species field: the chemicals and the microbes.
  velocity    the flow field.
  params      the conditions of each run, one row per run.
  t_norm      time of each snapshot, 0 to 1.
  geom_index  which rock each run used.
inputs        what each run was TOLD to do. Four tables.
  xml         every setting in the run's input file, as a table.
  kinetics    every rate constant in the .hh files, with its note.
  order       the chemical order those files expect. Compare against species.
```

Nothing has to be present. A file with no input files and no rate fields opens
the same way; the tree is simply shorter.

The right side shows whatever you clicked, and what "shows" means follows what
the thing is:

| what you select | what you get |
|---|---|
| an array with a grid in it | a picture, with a slider for every axis not on screen |
| a short list or table | the numbers |
| text | the text, which is how `CompLaB.xml` and the kinetics file are read |
| a group | what is inside it and what it is called |

The controls above the picture:

- **across** and **up** choose which two axes are on screen. The rest become
  sliders, named rather than numbered: `run`, `snapshot`, `species`,
  `component`, `rock`. So `samples/conc` gives a slider for the run, one for the
  snapshot and one for the chemical.
- **View**, in 3D: `Solid` for a pore structure, because the object is its outer
  surface. `Cloud` for a field, so what is inside shows through what is in
  front. `Surface` with the level slider to sweep an isosurface. `Plane` for a
  single slice.
- **Colours**, **hide solid**, **log scale**, **Save picture**.
- **Inspect** replaces the picture with what the thing is: shape, type,
  attributes.

**One trap.** `samples/conc` is stored already divided by `conc_scale`, so the
viewer shows 0 to 1 and not mol/L. Multiply by `samples/conc_scale` for real
values. The line beside it in the tree says so.

---

### Step 4. Train

```bash
python prt.py train --data work/dataset/dataset.h5 --out work/runs/A \
              --species Ac --reaction acetate_sulfate --epochs 300
```

The model predicts ONE field, so a dataset with four chemicals needs four
training runs, one per `--species`.

**What to write, and what it means.**

| you write | what it means |
|---|---|
| `--data .../dataset.h5` | the file from step 2 |
| `--out work/runs/A` | a folder. `best.pt` and `summary.json` are written in it |
| `--species Ac` | which chemical THIS model predicts. Leave it out and the first in the file is used, and the run says which |
| `--reaction acetate_sulfate` | which chemistry. It changes no tensor shape; it is recorded so evaluate and predict can refuse the wrong pairing by name |
| `--epochs 300` | an upper bound. Training stops early when the held-out error stops improving |
| `--patience 20` | how many epochs without improvement before it stops |
| `--batch-size 8` | runs per step. Larger uses more memory |
| `--n-points 8192` | pore voxels sampled per step. Asking for more than the rock has simply uses all of them |
| `--lr 1e-3` | learning rate. 0.001 is the usual choice |
| `--val-frac 0.15 --test-frac 0.15` | the share of GEOMETRIES used to choose the checkpoint, and the share held back to report on |
| `--seed 42` | fixes the split and the initialisation, so the run repeats |
| `--workers 4` | loader processes. Use `0` on Windows if it misbehaves |
| `--distance gdf` | what the trunk senses geometry with. `edt` is the straight-line control, `none` removes it. This is the ablation that carries the paper |
| `--distance-convention ours` | how that column is scaled. Leave it alone unless warm-starting from the published 2D weights |

**What you see:**

```
target scale: fitted on the training split, [1. 1.]
device      : cpu
reaction    : acetate_sulfate  (Acetate oxidation with sulfate reduction (ours))
distance    : convention 'ours', zero at the inlet, rising downstream, in voxels over nx - 1
species     : predicting 'Ac'  (this file holds Ac, A)
switches    : OFF  trunk (x, y, z, t, gdf)
branch1 ch  : 1  (material)   params: 0.46M
parameters  : pe, da   (layout pe_da, da = da_bio)
trunk inputs: x, y, z, t, gdf   (snapshots per run: 2)
train/val/test: 16 / 4 / 4 samples over three disjoint sets of geometries
              geometries  train 4, val 1, test 1
epoch   0  train 0.238167  val 0.074643  (0.4s)
epoch   1  train 0.058619  val 0.067728  (0.3s)
...

RMSE (normalised units), on geometries held out of both training and checkpoint selection:
  Ac     test 0.3227    <- the 2D paper's bar was 0.04
         val  0.3082    (what the checkpoint was chosen on; report the test number)
```

**What to check, in this order.**

1. **`train/val/test`** are three disjoint sets of whole GEOMETRIES, not runs.
   If you see fewer than about 10 geometries in training, the score below means
   very little.
2. **`species`** is the one you meant.
3. **`reaction`** matches the chemistry. If it does not list your species the
   run says so and continues; the checkpoint records what you asked for, and
   evaluate and predict will later refuse to pretend otherwise.
4. **`val` stops falling** while `train` keeps falling: that is overfitting.
   More geometries, not more epochs.
5. **Report the `test` number, never the `val` one.** Validation chose the
   checkpoint, so it is optimistic by construction.

**What you get:** `work/runs/A/best.pt` and `work/runs/A/summary.json`. The
checkpoint carries the split, the scalings, the reaction and the distance
convention, so nothing downstream has to be told them again.

---

### Step 5. Validate on the rocks it never saw

```bash
python prt.py evaluate --checkpoint work/runs/A/best.pt \
              --data work/dataset/dataset.h5 --out work/eval/A
```

**What to write, and what it means.**

| you write | what it means |
|---|---|
| `--checkpoint work/runs/A/best.pt` | the model to score |
| `--data .../dataset.h5` | the dataset it was trained from. The test rocks are named inside the checkpoint, so they are not re-chosen here |
| `--out work/eval/A` | where the metrics and figures go |
| `--sim-seconds 1800` | the wall clock of ONE simulation of the kind this model replaces. The speedup is meaningless until you measure your own and pass it |
| `--save-fields 4` | how many per-sample field figures to write |
| `--no-3d` | skip the 3D renders, which are the slow part |
| `--compare A=runs/A/best.pt B=runs/B/best.pt` | score several checkpoints on the same rocks, in one table |

**What you see:**

```
test set    : the 1 geometries recorded in the checkpoint
  switches : OFF  trunk (x, y, z, t, gdf)
  reaction : acetate_sulfate, distance convention 'ours'
model        Ac     mean RMSE 0.2982   0.076 s/sample   speedup 23723x
             R2 -0.5676
             speedup is against an assumed 1800 s per 3D simulation (the default).
             Measure your own and pass --sim-seconds before quoting it.
```

**What to check.**

- **`test set`** should say *the geometries recorded in the checkpoint*. If it
  says *recomputed here*, the checkpoint predates that field and the set is only
  the same one if `--test-frac` and `--seed` match the training run.
- **`reaction`** must be the chemistry of this dataset. A mismatch is printed
  by name and the score below it is meaningless.
- **The speedup is an assumption until you measure it.** Time one of your own
  CompLaB runs and pass `--sim-seconds`.
- **R2 below zero** means the model is worse than predicting the mean. That is
  a training problem, not an evaluation one: go back to step 4.

**What you get** in `work/eval/A/`: `metrics.json`, `rmse_table.csv`,
`rmse_vs_params.png`, and per-sample field and physics figures. The figures are
what to look at first: an RMSE hides where the error is, and the pictures do
not.

---

### Step 6. Predict on a new rock

The geometry can be a rock out of a dataset, a `.npz`, a CompLaB `.vti`, or a
raw `.dat`.

```bash
python prt.py predict --checkpoint work/runs/A/best.pt \
              --geometry work/dataset/dataset.h5 --geom-index 0 \
              --pe 10 --da-bio 1 --da-abio 1 --out work/pred/A
```

**What to write, and what it means.**

| you write | what it means |
|---|---|
| `--checkpoint work/runs/A/best.pt` | the model |
| `--geometry <file>` | the rock: a `.h5` dataset, a `.npz`, a CompLaB `.vti`, or a raw `.dat` |
| `--geom-index 0` | which rock, when `--geometry` is a `.h5` holding several |
| `--nx 64 --ny 64 --nz 64` | only for a raw `.dat`, which does not say its own grid |
| `--pe 10 --da-bio 1 --da-abio 1` | the conditions to predict AT. They must be inside the range the model was trained over |
| `--t-norm 1` | which moment. 1.0 is the final or steady state |
| `--t-series 5` | instead of one moment, five spanning 0 to 1, written as `time_series.png` and `.npz` |
| `--out work/pred/A` | where the `.vti`, the `.npz` and the figures go |
| `--velocity vel.npy` | required only for a flow-aware checkpoint, shape `(3, nx, ny, nz)` |

**What you see:**

```
checkpoint : work/runs/A/best.pt
  species  : Ac
  reaction : acetate_sulfate, distance convention 'ours'
  switches : OFF  trunk (x, y, z, t, gdf)
  time     : this is a TIME-DEPENDENT model; --t-norm 1 selects the snapshot
  geometry 0 of 6 from work/dataset/dataset.h5 (gid 0)
  grid     : 24x24x24, 12306 pore voxels (porosity 0.890)
  figures  : pred_2d.png, pred_3d.png, pred_3d_conc.png

predicted range (normalised units):
  Ac    min +0.7364  mean +0.7382  max +0.7404

timing on cpu
  geometry prep (incl. geodesic) :   0.004 s   <- once per geometry, cacheable
  network inference              :   0.088 s   <- this is the number to quote
```

**What to check.**

- **`grid`** must match the grid the checkpoint was trained on. It refuses
  outright otherwise, because the geometry encoder is sized by the grid.
- **`time`.** A time-dependent model needs `--t-norm`. Use `--t-series 5` for a
  sequence instead of one snapshot.
- **`predicted range`.** A field that is nearly constant, as above, means the
  model has learned the mean and nothing else. That is the same finding as a
  negative R2 in step 5.
- **The conditions you pass must be inside the range the model was trained
  over.** Outside it, this is extrapolation and not prediction.

**What you get** in `work/pred/A/`: `Ac_pred.vti` for ParaView, `pred.npz` with
the raw field and the geometry, and the figures.

---

### The optional flow-aware path

Off unless asked for. Five steps, in this order, and step 2 is a control that
costs one training run and tells you whether steps 3 to 5 are worth doing:

```bash
python prt.py flow-features --data work/dataset/dataset.h5 --buffer 5
python prt.py train --data work/dataset/dataset.h5 --out work/runs/sim \
              --velocity-informed simulated
python prt.py velocity-train --data work/dataset/dataset.h5 --out work/runs/vel
python prt.py velocity-predict --data work/dataset/dataset.h5 \
              --checkpoint work/runs/vel/best.pt --write-back
python prt.py train --data work/dataset/dataset.h5 --out work/runs/pred \
              --velocity-informed predicted
```

`--buffer` must match the open padding your campaign used: 10 for the published
2D set, 5 for the geometries this project generates, 0 for none.

---

### The same chain in 2D

Identical commands. A dataset whose third grid number is 1 is a 2D problem, and
the geometry branch, the trunk and the channels all follow the data. There is no
2D fork to keep in step.

To run the PUBLISHED 2D model instead, with its own released weights:

```bash
python prt.py predict2d --reaction monod --out work/pred2d
```

---

### Nothing yet, and no cluster

A generator writes a demo campaign in the exact layout finished CompLaB output
has, so the whole chain above can be exercised in a few minutes:

```bash
pip install -r requirements.txt
python 3D/tools/make_demo_complab.py --out work/demo
python prt.py collect-foreign --runs work/demo/demo_2D --out work/demo/ds2d
```

The numbers in the demo are constructed rather than simulated and the script
says so. The geometry is a real percolating packing and the file layout is
exactly what CompLaB writes, so it exercises the collector, the viewer and the
training code. It settles nothing about physics.

---

## What it can do

**Build pore structures.** Gaussian random media at a chosen porosity in two
or three dimensions, or read a domain from the published 2D set or from a
CompLaB file. Unreachable pore space is filled in, a biofilm layer can be
grown on the grain surfaces, and two distance maps are stored with every
domain: the true travel distance from the inlet through the pore space, and
the straight-line distance to the nearest grain.

**Solve the flow.** Stokes flow by a two-relaxation-time lattice Boltzmann
scheme on D2Q9 or D3Q19, driven by a body force rather than by end pressures,
with the wall pinned exactly halfway between the last water voxel and the
first solid one at every viscosity. The solver runs until the mean velocity
has actually settled rather than for a fixed number of iterations.

**Transport and react.** Species carried through that flow by a flux-form
finite volume method with a minmod slope limiter, harmonic-mean diffusivities
at every face, and a divergence correction. Biotic (dual Monod) and abiotic
reactions can each be switched on or off independently.

**Collect somebody else's CompLaB output.** Any folder of runs, with or
without the input files beside them. Each run's own `CompLaB.xml` and kinetics
headers are found automatically, the pore code is read from the input file
rather than guessed, both distance fields are computed, and every absence is
recorded rather than being a reason to fail. Reaction rate fields are read
where the run wrote them.

**Screen every run before keeping it.** A run is rejected if it has no
geometry, if an array has the wrong shape, if every value is zero, if a value
exceeds the blow-up bound, or if its grid differs from the rest. Each
rejection is reported with its own reason.

**Train, evaluate and predict.** A DeepONet with a convolutional branch for
the geometry, a fully connected branch for the dimensionless numbers, and a
trunk taking position, time and distance. The architecture matches the
published 2D model block for block; the only differences are the ones the third
dimension forces, namely 3D convolutions and pooling, one more trunk
coordinate, and a trunk evaluated on a sampled subset of pore voxels so the
activations fit in memory. It predicts one concentration field, so one model is
trained per reaction system and per chemical species, as in the 2D release;
`train.py --species NAME` picks which one. Training, ablation sweeps,
evaluation against held-out rocks, figures, and prediction on new geometries.

The flow-aware, velocity-informed capability is available as an optional switch
and is off by default. With it off the model is the 2D one lifted one
dimension, and nothing in the default path is touched by it.

**Transfer from 2D to 3D.** Extruded 2D domains are prismatic, so the exact 3D
solution is the 2D solution repeated. That makes 3000 cheap 2D domains genuine
members of the 3D problem class rather than an approximation.

**Predict the flow itself.** A second operator maps the pore structure and one flow
condition to the velocity field, with no flow solve at all. It is conditioned on two
descriptors the geometry alone does not give a convolution: how wide the pore is at
each voxel, and how wide the narrowest throat between that voxel and the inlet is. The
loss adds a penalty on how far the predicted field is from conserving mass.

**Give the concentration model the flow.** The velocity field, simulated or predicted,
joins the pore structure as extra image channels. This is what separates a preferential
channel from the pocket beside it, which geometry and a geodesic distance cannot do.

**Do all of it behind buttons.** The desktop application states what goes in,
what happens and what comes out before anything runs, and no command is ever
hidden: every page shows the exact command line it will execute.

---

## Inputs and outputs

**What goes in.** A pore structure as an integer voxel array, either generated
or supplied. Then the dimensionless numbers: the Peclet number, a Damkohler
number for each reaction, the half-saturation constants and the yield.
Optionally a biofilm thickness, a choice of reference length, and which
species the abiotic reaction consumes.

**What comes out.** For every run: the concentration field of each species at
each snapshot, the steady velocity field, the reaction rate of each reaction
where the run wrote it, the two distance maps, the two pore size maps, the material
array, the input files the run was given, and the conditions that produced it.

**What the trained network gives you.** The same concentration fields for a
geometry it has never seen, in milliseconds instead of hours.

---

## The dataset format

Everything is stored in **HDF5** (`.h5`), one file per campaign. HDF5 is a
container format for scientific data: a single file holds many named arrays
arranged in a folder-like tree, each array can be read without loading the
rest, and metadata rides along with the data instead of in a separate README
that goes stale.

That last point is why it was chosen here. A dataset of a few thousand runs
becomes one file rather than a directory of hundreds of thousands, and the
conditions of every run sit in the same file as its fields: the Peclet number,
the Damkohler numbers, the half-saturation constants and the yield are rows of
`samples/params`, named by the `param_names` attribute, and whether the run
settled or hit the step limit is `samples/settled`. A field can never be
separated from the numbers that produced it. Each rock is stored once and every
run records which rock it used, which is what makes it possible to hold whole
rocks out of training rather than a random selection of runs.

`docs/HDF5_dataset_format_2D_and_3D_v3.docx` is the full description: the
layout, what every entry means, what it costs on disk, and what is deliberately
not in it. To look inside one, `python prt.py dataset-info <file.h5>` prints the
arrays and then what the model will be handed. Step 3 above shows the output and
how to read it.

If HDF5 is new to you, this is a good introduction:
https://www.neonscience.org/resources/learning-hub/tutorials/about-hdf5

---

## The simulator

| file | what it holds |
|------|---------------|
| `3D/tools/prtlb_2d.py` | the 2D simulator: flow, transport and reactions |
| `3D/tools/prtlb_3d.py` | the 3D simulator, same structure |

Both files are organised into numbered blocks with a map at the top, so the
flow solver, the pressure gradient, advection, dispersion, the boundary and
initial conditions, and the biotic and abiotic reactions can each be found
without reading the rest. The transport and reaction code is word for word
identical between the two, and `test_flow_solvers.py` compares them on every
run.

Neither has been validated against CompLaB. Treat their numbers as synthetic
until that comparison is done.

---

## Installing and running

Python 3.10 or later.

```bash
pip install -r requirements.txt
python gui/prt_gui.py
```

The application has its own install button if you prefer that route. To
confirm everything works:

```bash
python check_everything.py
```

---

## What is in each folder

| folder | contents |
|--------|----------|
| `3D/tools/` | the two simulators, dataset builders, geometry generation, the CompLaB interface, the flow descriptors, the demo generator, the test suite |
| `3D/model/` | the DeepONet: training, prediction, evaluation, figures, and the velocity operator |
| `gui/` | the desktop application that drives every script above |
| `bridge/` | transfer learning from 2D to 3D |
| `2D/` | the published two-dimensional predecessor of Kim and Jung, unmodified |
| `docs/` | one guide per part of the code, and the dataset format |
| `examples/` | a settings file, a CompLaB.xml and the two kinetics headers, with a README |
| `check_everything.py` | runs every self-check in the project |

Anything the code produces (geometries, datasets, trained weights, figures) is
written under `work/`, which is deliberately not tracked here.

---

## The four feature switches

All four default to **off**. With all four off the 3D code behaves exactly
as it did before they existed, and `3D/tools/test_three_switches.py` proves
that bit for bit against a copy of the original implementation.

| switch | flag | what changes | needs 2D data |
|--------|------|--------------|---------------|
| A | `--flow-proxy` | the trunk receives the advective travel time instead of the geodesic distance, and the branch receives the velocity field instead of the pore mask | no |
| B | `--transfer-2d H5` | extruded 2D domains are mixed into training | yes |
| C | `--dim-free` | the trunk becomes (t, dwall, tau), the same three inputs in 2D and 3D, so the network transfers unchanged | yes, to be useful |
| D | `--velocity-informed` | the velocity field joins the geometry as extra branch channels, and the trunk is unchanged | no |

Switch A asks whether the flow field can replace the geodesic distance
function. Switch B is the compute argument. Switch C is why they belong
together: what makes 2D to 3D transfer hard is the geometry domain gap, and
switch A removes geometry from the input.

Switch D asks a narrower question and gets a clearer answer. It does not replace
anything: the velocity is added to the branch and the trunk stays as it was. In two
dimensions the published follow-up measured the median concentration RMSE falling from
0.0304 to 0.0263 and the worst case from 0.1739 to 0.1126. Their own control is the
sharpest part: the same velocity fed to the trunk pointwise recovered almost nothing,
so what matters is that the field arrives as a field a convolution can read.

A and D are alternatives, not companions, and `train.py` refuses to run both at once
rather than silently picking one. The full write-up is in `3D/SWITCHES.md`.

---

## The flow capability, end to end

The five commands are in the workflow above, under *The optional flow-aware
path*. Two things about them are worth stating on their own.

**The control comes first, and it is cheap.** Your campaigns already store the
velocity the solver produced, so switch D can be tested with no velocity model
at all. Training on the SIMULATED field measures the ceiling a predicted field
is chasing. If that run does not beat the baseline, nothing after it will.

**`flow-features` is only sometimes needed.** The collectors write MIS and UPRM
themselves now. That step is for `--geom-features` on a dataset collected before
they did, and the `--buffer` must match the open padding your campaign used: 10
for the published 2D set, 5 for the geometries this project generates, 0 for
none.

---

## What is ours and what is theirs

The velocity operator, the pressure U-Net and the three flow descriptors are ports of
the reference implementation released with the velocity informed PRT-DeepONet by Jo and
Jung, at github.com/hjunglab/PRT-DeepONet under `velocity-informed`. The two
dimensional forms are exact: their released checkpoints load into our classes with no
missing key, and `3D/model/test_reference_parity.py` runs their own feature code beside
ours on their bundled domain and reports every value identical, including the full
predicted velocity field. The three dimensional forms are ours, and there is no
released 3D reference to check them against.

---

## Licence and citation

GNU General Public License, version 3 or later. The full text is in `LICENSE`.

`LICENSING.md` says who wrote what and under which terms each part may be used.

The two-dimensional half is not ours. `2D/` is the published PRT-DeepONet of
**Yehoon Kim and Heewon Jung**, Jung Lab, Chungnam National University,
Republic of Korea, included unmodified and under the same licence. It supplies
the 3000 pore domains, the trained weights for three reaction types, and the
architecture this project extends into three dimensions. If you use anything
that touches the two-dimensional side, including the transfer set and the
warm start, please cite their work as well as this one.

`CITATION.cff` has both entries in a form GitHub and reference managers read
directly.

Shahram Asgari, shahram.asgari@uga.edu
Christof Meile, cmeile@uga.edu
Meile Lab, Department of Marine Sciences, University of Georgia
