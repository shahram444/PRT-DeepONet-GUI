# PRT-DeepONet Studio, version 1.2

A neural operator that predicts pore-scale reactive transport in milliseconds instead
of hours, **PRT-LB**, the lattice Boltzmann simulator that generates its training data,
and a desktop application that drives both.

**Version 1.2 adds the flow capability**: the network can be shown the velocity field,
and a second network can predict that field with no flow solve at all. Version 1.1 is
in the folder beside this one, unchanged. With the flow capability off, this version
behaves exactly as that one does, and a test proves it bit for bit.

---

## Install and check

Python 3.10 or later.

```bash
pip install -r requirements.txt
python check_everything.py          # about ten minutes
python gui/prt_gui.py               # the window
```

`check_everything.py` runs fifteen groups of self-checks as separate subprocesses, so
a crash in one cannot hide the rest. If it passes, the installation is sound.

**If you are handing this to somebody else to check**, point them at `tests/` instead.
It has its own README addressed to a reader who does not work on this project and does
not need to: one command, a table of ok / FAILED / SKIPPED, and a report file to send
back.

```bash
cd tests
python run_all_tests.py --fast     # about a quarter of an hour
python run_all_tests.py            # about half an hour
```

The window has an install button too, which runs `gui/install_requirements.py`. That
script always installs through the **same interpreter that is running it**, never
through a `pip` found on PATH, because a `pip` on PATH belongs to whichever Python was
installed last and packages put there are invisible to the one the window actually
uses.

## Start here: the whole chain in five minutes

No cluster, no CompLaB, no waiting.

```bash
cd 3D/tools
python make_demo_complab.py --out ../../work/demo
python collect_foreign_complab.py --runs ../../work/demo/demo_2D \
                                  --out  ../../work/demo/demo_2D_dataset
```

25 runs over 5 rocks, collected into one `dataset.h5`. Open it in the window's Viewer.
The geometry is a real percolating packing and the layout is exactly what CompLaB
writes, so this exercises the collector, the viewer and the training code. The numbers
in it are constructed rather than simulated, and the script says so. **It settles
nothing about physics.**

---

# How the project fits together

```
      a pore structure                you generate one, or CompLaB reads one,
              │                       or you extrude a published 2D domain
              ▼
      flow, by lattice Boltzmann      D2Q9 in 2D, D3Q19 in 3D
              │
              ▼
      transport and reactions         advection, diffusion, biotic + abiotic
              │
              ▼
      one dataset.h5                  every writer here produces the same layout
              │
              ▼
      train  ──►  evaluate  ──►  predict on a rock nobody ever simulated
```

Everything in `3D/tools/` makes data. Everything in `3D/model/` uses it. Nothing in
`tools/` trains and nothing in `model/` simulates, and the one file that knows both
sides is `3D/tools/dataset_reader.py`.

```
PRT-DeepONet-v1.2/
├── 3D/tools/    make data:  geometry, the simulators, campaigns, collectors, descriptors
├── 3D/model/    use it:     the network, training, evaluation, prediction, the velocity operator
├── gui/         the desktop application that drives every script above
├── bridge/      the two shortcuts across the 2D/3D join
├── 2D/          the published release of Kim and Jung, unmodified
├── tests/       the test suite, with a README for whoever you ask to run it
├── docs/        the tutorial, the dataset format, one guide per part of the code
├── examples/    one of each input file, with a README
└── work/        everything the code produces. Not tracked.
```

---

# Part 1. Putting your own numbers in

The solver works in six dimensionless numbers. You think in mol/L, micrometres and
m²/s. **`settings_and_units.py` is the only place that conversion happens**, so there
is one answer to what a number means rather than one per script.

| file | what it is |
|---|---|
| `3D/tools/settings_and_units.py` | every physical number, its units, and the conversion. 2974 lines |

```bash
python settings_and_units.py --template my_run.xml     # write a template, then edit it
python settings_and_units.py --check my_run.xml        # what it means, before running
python settings_and_units.py --capabilities            # what this can and cannot do
```

The template carries every supported tag with a comment saying what it changes and in
what unit. `--check` prints three lists and they mean different things:

- **complaints** stop a run
- **legal, but probably not what you meant** does not stop anything, and is worth
  reading anyway: a blank chemical, two chemicals that will be identical, a biomass in
  a run with no microbes
- **notes** record that something overrode a default

**Worth knowing.** Péclet and both Damköhler numbers carry a length, so which reference
length you choose decides what all three mean. The file says which it used, in the
report and in the dataset.

---

# Part 2. Getting data

Four ways in. Pick one.

## A. Simulate it yourself, with PRT-LB

| file | what it is |
|---|---|
| `3D/tools/prtlb_2d.py` | the 2D simulator: flow, transport, reactions. 2566 lines |
| `3D/tools/prtlb_3d.py` | the 3D simulator, same structure. 2535 lines |
| `3D/tools/build_dataset_2d.py` | runs the 2D simulator over a sweep, writes one `dataset.h5` |
| `3D/tools/build_dataset_3d.py` | the same in 3D |

**Flow** is Stokes by a two-relaxation-time lattice Boltzmann scheme, D2Q9 or D3Q19,
driven by a body force rather than by end pressures, with the wall pinned exactly
halfway between the last water voxel and the first solid one at every viscosity. **It
runs until the mean velocity has settled**, so `--stokes-iters` is a cap and not the
number of iterations.

**Transport** is flux-form finite volume with a minmod slope limiter, harmonic-mean
diffusivities at every face, and a divergence correction. Two reactions run on top and
each switches on or off independently: a biotic one obeying dual Monod kinetics with a
yield, and an abiotic one.

**Before every write, `check_physics` asserts what a correct solution must satisfy**:
nothing above what it could be made from, nothing negative, mass accounted for. A run
that fails is reported and **no file is written**.

```bash
python build_dataset_2d.py --out d.h5 --n-geom 16 --n-sets 5 \
       --shape 148 64 --n-times 8 --n-species 4 \
       --pe-min 3 --pe-max 50 --da-min 0.1 --da-max 10

python build_dataset_3d.py --out d3.h5 --n-geom 16 --n-sets 5 --shape 32 32 32
```

**Run `--show-settings` first.** It prints what your numbers mean and returns before
anything is simulated.

**Worth knowing.** 2D is a slice through the rock, not a thin rock, and it is roughly
fifty times cheaper than 3D. In 3D the cost goes as about the **fifth** power of the
grid: the voxel count as the cube, and an explicit scheme needs more steps as it grows.
Double the grid and think in weeks, not hours.

**Neither simulator has been validated against CompLaB. Treat their numbers as
synthetic until that comparison is done.**

## B. Drive CompLaB on a cluster

| file | what it is |
|---|---|
| `3D/tools/build_geometry_3d.py` | writes pore structures as files, in CompLaB's material codes |
| `3D/tools/complab_campaign.py` | builds and manages a batch campaign on Slurm |
| `3D/tools/collect_complab_output.py` | reads the finished campaign back into one training file |

```bash
python build_geometry_3d.py --n 140 --out ./geometries \
       --nx 64 --ny 64 --nz 64 --phi-min 0.45 --phi-max 0.65

python complab_campaign.py build --geometries ./geometries \
       --complab /path/to/CompLaB3D --out ./campaign --per-geom 20
# submit, then:
python complab_campaign.py status --out ./campaign
python complab_campaign.py retry  --out ./campaign

python collect_complab_output.py --campaign ./campaign/batch_* \
       --geometries ./geometries --out ./dataset --mode transient --n-times 11
```

**Every run gets its own fully isolated directory** with its own `CompLaB.xml`,
`env.sh`, `params.json` and `input/geometry.dat`. Nothing is shared, so nothing can
collide. `--batches` splits the campaign by geometry into independent sub-campaigns,
any of which can be submitted, collected and trained on alone; run ids stay globally
unique so batches merge without collision.

The collector **triages every run twice**: from `params.json` and `status.json` before
reading anything, then on the fields themselves. Runs stop at different iterations and
hold different numbers of snapshots, so they are resampled onto one common normalised
time grid.

**What you get back.** `dataset/dataset.h5`, `dataset/campaign_report.md`, and
`dataset/failures.csv` when there were failures. **Every rejection is reported with its
own reason.**

**Worth knowing.** `--wall 0` is refused outright by the geometry builder, because
CompLaB has no periodic boundary anywhere and pore voxels on the domain face would
leak.

## C. Collect somebody else's CompLaB output

| file | what it is |
|---|---|
| `3D/tools/collect_foreign_complab.py` | any folder of runs, set up by anyone. 1686 lines |

The campaign collector knows where everything is because it put it there. **This one is
given a folder and works the rest out.** It finds each run's own `CompLaB.xml` and
kinetics headers rather than being told about them, and reads the XML for what it
actually says: the Péclet number, **the pore material code**, the number of substrates
and their names. Reaction rate fields are read where the run wrote them.

```bash
python collect_foreign_complab.py --runs ./their_runs --inspect      # writes nothing
python collect_foreign_complab.py --runs ./their_runs --out ./dataset
```

**Start with `--inspect`.** It prints what it found and what is missing, and writes
nothing at all.

**Every absence is recorded rather than being a reason to fail.** Anything it could not
find goes into the `provenance` group along with the paths it looked at.

**Worth knowing.** Species files are found by **number**, not by name, because the two
disagree in real output more often than not.

## D. Reuse 2D data

An extruded 2D structure is prismatic, so **the exact 3D solution is the 2D solution
repeated**. That is not an approximation, and it makes 3000 cheap domains genuine
members of the 3D problem class.

| file | what it is |
|---|---|
| `3D/tools/build_transfer_set_2d_to_3d.py` | extrudes 2D domains into 3D blocks |
| `3D/tools/import_2d_simulations.py` | reads 2D output a collaborator sent you |
| `3D/tools/load_pretrained_2d_weights.py` | converts the published 2D weights into ours, as a warm start |
| `bridge/build_transfer_set.py` | the same, with the paths filled in |

```bash
python build_transfer_set_2d_to_3d.py --jung-dir ../../2D --out t.h5 \
       --target-shape 128 64 64 --n-sets 3 --n-species 4

python import_2d_simulations.py --src one_example.npz --dry-run     # always first
```

It exploits prismatic geometry twice: it solves on a thin slab of a few layers and
tiles the result to the full `nz`, which is about eight times cheaper, and it measures
the z-variation of what it produced and stores it. On the real domains that number
comes out at exactly zero.

**Worth knowing.** `--target-shape` must match your 3D dataset exactly or training
refuses the file. And an extruded domain has zero tortuosity in z, so a network fed too
much of it learns that nothing ever happens in that direction: keep the 2D fraction at
or below 0.3.

## Quick data, for testing

| file | what it is |
|---|---|
| `3D/tools/build_practice_dataset.py` | a tiny but entirely real dataset, in about a minute |
| `3D/tools/build_toy_pack_2d.py` | one command for a complete 2D playground |
| `3D/tools/make_demo_complab.py` | a campaign in CompLaB's exact layout, without CompLaB |

The practice dataset is small enough to run after any change and real enough to mean
something: genuine geometry, genuine D3Q19 flow, the genuine transport solver. The demo
campaign is **not**: its numbers are constructed, and it says so in its own output.

---

# Part 3. The dataset

**One HDF5 file per campaign, one layout, whoever wrote it.** A dataset of a few
thousand runs becomes one file rather than a directory of hundreds of thousands.

The point of the format is that **a field can never be separated from the numbers that
produced it**. The Péclet number, the Damköhler numbers, the half-saturation constants
and the yield are rows of `samples/params`, named by the `param_names` attribute, and
whether it settled or hit the step limit is `samples/settled`, all in the same file as
the fields.

**Each rock is stored once and every run records which rock it used.** That is what
makes it possible to hold whole rocks out of training rather than a random selection of
runs, which is the difference between a real score and a memorisation check.

| what is in it | |
|---|---|
| `geom/material` | the pore structure, one per rock |
| `geom/gdf`, `geom/edt` | the geodesic distance from the inlet, and the straight-line distance to the nearest grain |
| `geom/mis`, `geom/uprm`, `geom/dw2` | the three pore size maps. **New in 1.2** |
| `samples/conc` | every species at every snapshot |
| `samples/velocity` | the steady flow field |
| `samples/velocity_pred` | a predicted flow field, when one has been written. **New in 1.2** |
| `samples/rate` | the reaction rates, when the run wrote them and the collector found them |
| `samples/params`, `param_names` | the conditions that produced each run |
| `samples/settled` | whether the run settled or hit the step limit |

`docs/HDF5_dataset_format_2D_and_3D_v3.docx` is the full description: what every entry
means, what it costs on disk, and what is deliberately not in it.

```bash
python 3D/tools/dataset_reader.py path/to/dataset.h5     # prints the tree
```

---

# Part 4. Training

| file | what it is |
|---|---|
| `3D/model/deeponet_model.py` | the network |
| `3D/model/train.py` | fits it, and writes `best.pt` |
| `3D/tools/dataset_reader.py` | the PyTorch Dataset, and where every feature flag is resolved |

**The network is three parts multiplied together.** A convolutional branch reads the
pore structure and compresses it to a short vector every voxel of that sample shares. A
dense branch reads the dimensionless numbers. A trunk reads one point: its position,
the time, and the geometry feature. The encoder is dimension-aware by side effect: if
the grid's third axis is 1 it swaps its 3D convolutions for 2D ones.

```bash
python 3D/model/train.py --data d.h5 --out runs/first --epochs 60
```

**It splits by structure and never by sample.** Two runs on the same rock share its
entire geometry, so a model that memorised the rock would score well on a random split
and tell you nothing.

**`best.pt` is not just weights.** It carries the species names, the parameter names
and their order, the grid, and every flag the run used. That is why prediction and
evaluation can rebuild the exact training configuration from it rather than from
whatever you type later.

**Defaults**: 300 epochs, batch 8, 8192 trunk points, learning rate 0.001, patience 20.
AdamW with cosine annealing, a Huber loss, mixed precision. `best.pt` is rewritten
whenever the held-out loss improves, so a run is usable mid-flight.

**Worth knowing.** It samples pore voxels rather than evaluating the whole grid. A 148
by 64 2D domain holds about 7,000 pore voxels, so any `--n-points` above that means
"the whole domain every step"; a 64³ domain holds of order 100,000, where the number
really is a sample.

---

# Part 5. Evaluating and predicting

| file | what it is |
|---|---|
| `3D/model/evaluate.py` | held-out accuracy, and the figures |
| `3D/model/predict.py` | a trained network on a rock that was never simulated |
| `3D/model/make_figures.py` | the shared rendering, so the two produce comparable figures |
| `3D/model/run_ablation_sweep.py` | trains every configuration and compares them |

```bash
python evaluate.py --data d.h5 --checkpoint runs/first/best.pt --out figs

python predict.py --checkpoint runs/first/best.pt --geometry new_rock.npz \
       --pe 10 --da-bio 1 --da-abio 1 --out predictions/A --t-series 8
```

`evaluate.py` **rebuilds the dataset configuration from the checkpoint and re-derives
the same split**, then refuses to continue if it cannot. It reports error per chemical
and against each of the three main numbers, rather than one number, because one number
hides where the model is weak.

`predict.py` is the point of the project. It reads a geometry, computes the distance
fields if they are not already there, and answers in under a second. **It checks the
parameters you give against the ranges the model was actually trained on** and says so
when you are outside them.

**Worth knowing, three things.**

- **Two RMSE numbers are printed and they differ.** One is the mean of the per-snapshot
  errors; the other pools every voxel first. Neither is wrong; say which you quoted.
- **`--save-fields` is forced to zero whenever more than one model is compared**, so a
  `--compare` run never produces field figures.
- **Ignore the speedup unless you set `--sim-seconds`** to a simulation time you
  measured yourself. The default is a placeholder.

For the sweep, **run `--quick` once first**. Three epochs and tiny batches: it proves
the thing runs and proves nothing else.

---

# Part 6. The flow capability

## The problem it solves

Take two points in the same rock, the same distance along the flow path from the inlet,
in pores of the same width. One is on a preferential channel. The other is in a
stagnant pocket behind a narrow throat. They receive completely different amounts of
solute and react at completely different rates.

Version 1.1 could not tell them apart. It saw the pore geometry and the geodesic
distance, which is how far a molecule must travel. Neither says anything about how fast
it moves once it gets going. The error was systematic: **concentrations overestimated
in pore space that is geometrically connected but hydraulically dead**, which is the
immobile fraction that matters most for tailing and for anything upscaled from the pore
field.

## What it does about it

The velocity field joins the pore structure as extra image channels on the
convolutional branch. The trunk is left exactly as it was.

That is the whole intervention. Not a new architecture, not a new loss. The published
follow-up that introduced it also tried feeding the same velocity to the trunk instead,
as two numbers per query point, and it recovered almost nothing. What matters is that
the field arrives **as a field** a convolution can read across a neighbourhood.

| file | what it is |
|---|---|
| `3D/tools/flow_features.py` | MIS, UPRM and the squared wall distance, in 2D and 3D |
| `3D/tools/harmonic_pressure.py` | Laplace's equation on the pore space, and its gradient |
| `3D/tools/add_flow_features.py` | puts the three maps into a dataset you already have |
| `3D/model/velocity_model.py` | the velocity operator, the pressure U-Net, the losses |
| `3D/model/train_velocity.py` | trains the operator |
| `3D/model/predict_velocity.py` | runs it, and writes `samples/velocity_pred` back |

**Three descriptors**, stored with every rock:

| | what it is | why it is not redundant |
|---|---|---|
| **MIS** | radius of the largest sphere that fits in the pore space and contains this voxel | not the wall distance. A voxel near a grain can still lie inside a large sphere centred further in |
| **UPRM** | radius of the largest sphere the inlet could deliver here, set by the narrowest throat on the best path | **not local at all.** Two identical pockets differ if one sits behind a bottleneck, and the limiting throat can be far outside a convolution's receptive field |
| **dw²** | squared distance to the nearest wall | a no-slip profile is parabolic in the wall distance, so the square is the natural input |

UPRM is the interesting one for a transport reader: it is the critical throat radius on
the widest path from the inlet, computed voxel by voxel. What percolation and mercury
intrusion measure globally, made local.

## How to run it: the flow pipeline panel

In the window, **The flow field → The flow pipeline**, in both 2D and 3D mode.

It is a panel and not a tick box, and the reason is worth one sentence: switches A, B
and C each change one training run and need nothing prepared, while this needs four
stages to have run against the same dataset before the fifth can read what they wrote.

**Fill in six boxes down the left, once.** They are shared by every step, which is what
stops step 4 writing to one file while step 5 reads another.

| box | what to put in it |
|---|---|
| Dataset | the one `.h5`. It must hold a velocity field, so it was collected without "skip the flow field" |
| Folder for the results | each step gets its own subfolder inside it |
| Open buffer at each end, in voxels | **10** for the published 2D set, **5** for geometries this program makes, **0** for none |
| Which condition drives the flow | `pe`. Ours record the Péclet number; the published model uses Reynolds, which we do not store |
| How many passes over the data | applies to all three training steps |
| Also use the pore size maps | optional. Two more branch channels. Needs step 1 |

**Five cards down the right.** Each shows the exact command it will run, has its own
Run button, its own state, and a tick box.

| # | step | skip it when |
|---|---|---|
| 1 | Add the flow descriptors — writes MIS, UPRM and dw² into the dataset, in place | it was collected with v1.2, which already writes them |
| 2 | **Train on the simulated flow** — the control | **never** |
| 3 | Train the velocity operator | you only wanted the control |
| 4 | Predict the flow and store it — writes `samples/velocity_pred` beside the simulated field, never over it | same |
| 5 | Train on the predicted flow | same |

**Run the whole pipeline** does the ticked ones in order and **stops at the first
failure**, rather than running a step whose input was never written. Two things it
fills in for you: step 4 is always told to write the field back, and step 4 always
reads step 3's checkpoint.

**All settings for this step** on any card opens that script's ordinary page, with
every flag it accepts, for when the six boxes are not enough.

## Do step 2 before you build anything

Step 2 needs no velocity model. It uses the flow your solver already stored, so it
costs one training run.

That measures the **ceiling**. A predicted field can only ever approach the real one,
so if the real one does not help your chemistry and your Péclet range, a network
trained to approximate it will not either. Three numbers settle it:

| | |
|---|---|
| **no flow** | your existing baseline, from Learn → Train the network |
| **simulated flow** | step 2. The ceiling |
| **predicted flow** | step 5. What you actually get |

**If `simulated` does not beat `no flow`, stop.** If it does, the gap between
`simulated` and `predicted` is what the velocity operator costs you.

## The one box to get wrong

**"Open buffer at each end, in voxels"** is how many voxels of fully open channel sit at
the inlet and the outlet of your domain.

MIS is computed on the interior and then grown outward into that buffer, because a
sphere placed in an open channel would report a radius the rock never has. Set it wrong
and the pore size map measures your padding instead of your rock.

## The same thing from a terminal

```bash
DS=work/demo/dataset.h5
R=work/runs/flow

# 1. only if the dataset does not already carry the descriptors
python 3D/tools/add_flow_features.py --data $DS --buffer 5

# 2. THE CONTROL. Run this first, and compare against the same run with the flag off.
python 3D/model/train.py --data $DS --out $R/concentration_simulated \
                         --velocity-informed simulated

# 3. the velocity operator
python 3D/model/train_velocity.py --data $DS --out $R/velocity_operator

# 4. run it. Without --write-back, step 5 has nothing to read.
python 3D/model/predict_velocity.py --checkpoint $R/velocity_operator/best.pt \
                                    --data $DS --write-back --out $R/velocity_fields

# 5. step 2 again, on the predicted field
python 3D/model/train.py --data $DS --out $R/concentration_predicted \
                         --velocity-informed predicted
```

Add `--geom-features` to steps 2 and 5 for the MIS and UPRM channels.

**Two more things worth knowing about the operator.** Its loss is a Huber term averaged
over the **pore voxels only**, plus lambda times the mean squared divergence; averaging
over the whole grid would make a low-porosity rock look easy. And early stopping
watches the **data term alone**, never the sum, because a run getting worse at velocity
would otherwise look like it was improving as the divergence term fell.

## What it improves

**Every number here is from the published two-dimensional study.** Nobody has measured
any of it in three dimensions. That is the experiment this version exists to make
possible, not one it has already done.

Steady irreversible sorption, 2700 held-out samples, concentration RMSE:

| | v1.1 | v1.2 | change |
|---|---|---|---|
| median | 0.0304 | 0.0263 | 13% better |
| 90th percentile | 0.0570 | 0.0442 | 22% better |
| **worst case** | **0.1739** | **0.1126** | **35% better** |

Read the shape of that column, not the middle of it. **The gain grows toward the hard
cases**, which is what you would expect if the mechanism is real.

Transient Monod kinetics: mean RMSE 0.0480 to 0.0378, 21% better. Better in eight of
nine Péclet–Damköhler combinations, and worse in exactly one: Pe = 1 with the fastest
reaction, where diffusion and reaction govern the field and the flow barely matters.
That is the honest boundary of the method.

## What it costs

| | v1.1 | v1.2 |
|---|---|---|
| inference, per geometry, one CPU core | about 1.5 ms | about 190 ms |

Roughly 130 times slower per prediction, because the new figure covers the descriptors,
the pressure prior, the velocity operator and then the concentration operator. Even so,
a sweep that would take a week of solver time still finishes in half an hour.

Preprocessing, measured on this code, one CPU core, porosity around 0.5:

| grid | MIS | UPRM | pressure solve | total per rock | disk per rock |
|---|---|---|---|---|---|
| 2D, 148 × 64 | 0.09 s | 0.04 s | 0.10 s | **0.23 s** | 0.11 MB |
| 3D, 64³ | 12.6 s | 2.3 s | 45.8 s | **61 s** | 3.15 MB |

**The pressure solve dominates in 3D, and the cheap path avoids it entirely.** It feeds
the velocity operator's trunk, so running only step 2 never touches it and the cost per
rock falls from 61 seconds to about 15. On a large 3D campaign, `--pressure unet`
replaces the sparse solve with one forward pass: four hours becomes minutes on 240
rocks.

A predicted velocity field is three float32 components per run, so a 240-run campaign
at 64³ adds roughly 750 MB. It is written beside the simulated field, never over it.

## What is unchanged

Everything. With `--velocity-informed off`, every branch channel and every trunk column
is bit-identical to version 1.1. `3D/tools/test_three_switches.py` proves it by
re-implementing the original data path inside itself and asserting equality on every
sample. If a v1.1 result will not reproduce in v1.2, that test is where to look first.

---

# Part 7. The three feature switches

Separate from the flow pipeline, and unchanged since v1.1. All default **off**, and
with all three off the code behaves exactly as it did before they existed.

| switch | flag | what changes | needs 2D data |
|---|---|---|---|
| A | `--flow-proxy` | the trunk gets the advective travel time instead of the geodesic distance, and the branch gets the velocity field instead of the pore mask | no |
| B | `--transfer-2d H5` | extruded 2D domains are mixed into training | yes |
| C | `--dim-free` | the trunk becomes (t, dwall, tau): the same three inputs in 2D and 3D, so the network transfers unchanged | yes, to be useful |

Switch A asks whether the flow field can replace the geodesic distance. Switch B is the
compute argument. Switch C is why they belong together: what makes 2D to 3D transfer
hard is the geometry domain gap, and switch A removes geometry from the input.

Switch A's travel time is computed by `3D/tools/flow_coordinates.py`. **Read the section
in it headed THE ONE HONEST WEAKNESS before trusting a result**: where the fluid does
not move, the travel time carries no information, which is exactly the low-Péclet
regime where the geodesic distance still does.

**A and the flow pipeline cannot both be on.** A *replaces* the trunk's geodesic
distance; the flow pipeline leaves the trunk alone and adds to the branch. Two different
answers to one question, so `train.py` refuses the combination rather than picking one
silently.

The full write-up is in `3D/SWITCHES.md`. To settle which is worth it on your data:

```bash
python 3D/model/run_ablation_sweep.py --data d3.h5 --out sweep --quick   # once
python 3D/model/run_ablation_sweep.py --data d3.h5 --data-2d t.h5 --out sweep

python bridge/run_experiment.py --data d3.h5          # the same, paths filled in
```

It trains one model per configuration, evaluates them all against the **same** held-out
structures, and splits the error by Péclet band. **That breakdown is the point.** Write
down your prediction before you look: the flow field should win at high Péclet and lose
below 1.

---

# Part 8. The window

```bash
python gui/prt_gui.py
```

Every script above is a page. **Each page states what goes in, what happens and what
comes out before anything runs, and shows the exact command line it will execute.** You
can copy that line into a terminal; nothing the window does is hidden from you.

| | |
|---|---|
| **Left** | the pipeline as a tree: Set up, Simulate, Learn, The flow field, Predict, Look at results, Help |
| **A page** | settings on the left, the three coloured boxes on the right, the command at the bottom |
| **Bottom** | Log is the child process as it arrives. Problems collects every error with the command that produced it. Monitor plots the training and held-out curves live |

**The Viewer** is a general HDF5 browser rather than a fixed set of plots. The left half
is a tree of everything in the file; select an array and it draws. The axis that selects
a field is **named rather than numbered**: it reads the species and reaction labels out
of the file, so the picker says `species 2 P` and not `channel 2`.

For a 3D field there are four ways of looking:

| | |
|---|---|
| **Plane** | one slice. The one to measure against |
| **Cloud** | composites the volume front to back at low opacity, so a front inside the rock is visible |
| **Surface** | the grains as a marching-cubes surface |
| **Solid** | opaque voxels |

**Two rules the window always follows.** Nothing is ever written on top of an earlier
result: run the same page twice and the second goes to `work/2D_new_2` rather than back
into `work/2D_new`, and it says so. And every field is validated before launch, with a
list of what needs attention naming the box and the section it is in, rather than an
argparse error three seconds into a run.

**Help → Tutorial** is six worked projects, start to finish. `gui/TUTORIAL.txt`.

---

# Part 9. The tests

Nothing here is decoration. Each one was written after a specific bug.

**There are two layers.** `tests/` is a suite you can hand to somebody else; the rest
live beside the code they test.

## `tests/` — for somebody who does not work on this project

| file | what it asks |
|---|---|
| `tests/test_01_smoke.py` | does everything start at all? Every file compiles, every library imports, all 26 scripts answer `--help`, the window builds with the display faked |
| `tests/test_02_units.py` | one function at a time, against an answer worked out by hand. A 5-voxel channel must read width 3; a chamber behind a 1-voxel throat must read 3 locally and 1 for what the inlet can deliver; a straight channel's pressure must fall in a straight line with exactly zero sideways gradient |
| `tests/test_03_data.py` | files written here read back correctly, the dataset layout is what everything downstream assumes, and the one operation that edits a file in place refuses to overwrite |
| `tests/test_04_repeatable.py` | same seed, same answer — **and** different seed, different answer, because a function ignoring the seed would pass the first alone |
| `tests/test_05_refusals.py` | twelve wrong things done on purpose. Each must fail rather than carry on, and each message must name what to fix |
| `tests/run_all_tests.py` | one command that runs all of the above plus the groups below, and writes `test_report.txt` |

`tests/README.md` explains each of them in plain language, says what a skip means, and
says what these tests do **not** tell you: nothing about whether the physics is right,
and nothing about whether the model is accurate.

## Beside the code

| file | what it proves |
|---|---|
| `3D/tools/test_flow_solvers.py` | both flow solvers against answers that do not come from them: the analytic parabola between parallel plates, the lattice identities, and the two simulator files compared byte for byte. 28 checks |
| `3D/tools/test_three_switches.py` | with the switches off, the data path is bit-for-bit what it was before they existed. It holds a verbatim copy of the original and compares against it |
| `3D/tools/test_documented_numbers.py` | every number the documents claim, measured again |
| `3D/model/test_flow_pipeline.py` | the whole velocity path end to end, on a dataset it builds itself. 33 checks |
| `3D/model/test_reference_parity.py` | our descriptors and our predicted field against **theirs**, value by value |
| `gui/test_gui_commands.py` | every button emits a command line its script's real argument parser accepts. 50 commands |
| `gui/test_flow_panel.py` | the flow panel's five steps line up with each other, and a failed step stops the sequence. 35 checks |
| `gui/test_gui_widgets.py` | every page builds, every field has a home, every action is reachable from the sidebar |
| `gui/test_gui_sweep_modes.py` | both halves of every range-or-exact-numbers box |
| `audit_comments.py` | every file carries its top block, its section markers and its line comments |

```bash
python check_everything.py       # all fifteen groups, including tests/
```

**`test_gui_commands.py` is worth understanding.** Seven separate bugs in the window
were the same bug: the button sent a flag the script does not have, or omitted one it
requires, or packed several values into one token. None could be seen by reading the
button; all appeared three seconds into a run. This asks each script's **own** parser
whether the command is acceptable, so it cannot drift out of date. It also checks
itself, by feeding the checker four commands known to be bad and one known to be good.

---

# Reading the code

**Start with `3D/tools/dataset_reader.py`.** It is the one file that knows both what an
HDF5 dataset holds and what the model expects, so everything passes through it. Then
`3D/model/train.py` for the flags, `3D/model/deeponet_model.py` for the architecture,
and `FlowPipelinePage` in `gui/prt_gui.py` for the panel.

**Every file the flow work added or touched explains itself at the top**, in a block
headed `CHANGED FROM THE 2D VERSION`: which repository, notebook and cell the code came
from, what their version does, and what was changed and why. Inside, block comments
mark the sections and line comments explain the parts that look wrong until explained.
`audit_comments.py` checks that this stays true and runs as part of the self-check.

**Four bugs were found and fixed while writing those comments.**

| file | what was wrong |
|---|---|
| `3D/tools/settings_and_units.py` | wrote a settings XML it could not read back: one paragraph of its own explanatory text used a double hyphen, which is illegal inside an XML comment |
| `bridge/build_transfer_set.py` | pointed at `ingest_2d.py`, renamed long before. It exited on its first check every time |
| `bridge/run_experiment.py` | the same fault: `run_switch_sweep.py`, renamed to `run_ablation_sweep.py` |
| `3D/tools/test_documented_numbers.py` | hardcoded to absolute paths on the machine it was written on, so it could never have passed anywhere else |

---

# What is ours and what is theirs

The velocity operator, the pressure U-Net and the two-dimensional flow descriptors are
ports of the reference implementation released with the velocity informed PRT-DeepONet
by Jo and Jung, at github.com/hjunglab/PRT-DeepONet under `velocity-informed`.

**The ports are exact, and this was checked rather than assumed.** Their released
checkpoints load into our classes with no missing key, no unexpected key and no shape
mismatch. Beyond that, `3D/model/test_reference_parity.py` runs their own feature code
beside ours on their own bundled domain:

```
UPRM   identical True   max abs diff 0   correlation 1.000000
MIS    identical True   max abs diff 0   correlation 1.000000
dw2    identical True   max abs diff 0   correlation 1.000000
predicted velocity, their weights, our features vs theirs:
       max abs difference 0,  fields identical: True
```

That test runs as part of `check_everything.py`. If it ever stops reporting identical,
any comparison with their published numbers is measuring our reimplementation rather
than their method.

The three-dimensional forms are ours, and there is no released 3D reference to check
them against. `LICENSING.md` names exactly which files are ports and which parts of
them.

## Who wrote what

**The three-dimensional extension and the desktop application are the work of Shahram
Asgari and Christof Meile, Meile Lab, Department of Marine Sciences, University of
Georgia.** That covers the 3D formulation, the geometry generator, the D2Q9 and D3Q19
flow solvers, the shared advection, diffusion and reaction solver, the CompLaB3D
campaign and collection tools, the HDF5 dataset layout, the training, evaluation and
prediction pipeline, the feature switches, the three-dimensional velocity operator, and
the whole desktop application.

Two parts come from the Jung Lab at Chungnam National University, included under the
same licence:

- `2D/` is the published PRT-DeepONet of **Yehoon Kim and Heewon Jung**, unmodified. It
  supplies the 3000 pore domains, the trained weights for three reaction types, and the
  architecture this work extends. Cite it if you use anything that touches the
  two-dimensional side, including the transfer set and the warm start.
- The two-dimensional velocity operator, the pressure network and the flow descriptors
  are ports of the velocity informed release of **Hyegyeong Jo and Heewon Jung**. Cite
  it if you use the flow pipeline or the velocity operator.

To cite the three-dimensional work:

> Asgari, S. and Meile, C. PRT-DeepONet Studio, version 1.2. Meile Lab, Department of
> Marine Sciences, University of Georgia, 2026.

`CITATION.cff` holds this and both upstream entries in a form GitHub and reference
managers read directly.

---

GNU General Public License, version 3 or later. Full text in `LICENSE`; `LICENSING.md`
says who wrote what and under which terms each part may be used.

Shahram Asgari, shahram.asgari@uga.edu
Christof Meile, cmeile@uga.edu
Meile Lab, Department of Marine Sciences, University of Georgia
