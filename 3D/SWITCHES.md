# The three feature switches, and the flow pipeline

All of them default to **OFF**. With all of them off the code does exactly what it
did before they existed — same branch channels, same trunk columns, same order,
bit for bit. `tools/test_three_switches.py` proves that by re-implementing the original
`__getitem__` and asserting equality on every sample.

```
python tools/test_three_switches.py --data /tmp/test3d.h5
```

---

## Switch A — `--flow-proxy`

**Christof's question:** *"can we just take the flow field instead of the GDF?
That would allow us to focus on the reactions and simply use the state of the art
of the flow field AI/ML field rather than having to work on that ourselves."*

**What it does.** Replaces the trunk's geodesic column with the **advective
travel time** `tau`, and feeds the branch the **normalised velocity field**
instead of the binary pore mask.

|                | OFF                     | `--flow-proxy`          |
|----------------|-------------------------|-------------------------|
| branch1        | material (1 ch)         | ux, uy, uz (3 ch)       |
| trunk          | x, y, z, t, **gdf**     | x, y, z, t, **tau**     |

**Why travel time and not speed.** `tau(x)` is the time for a fluid parcel to
reach `x` from the inlet, from `u·grad(tau) = 1`. Along a streamline the
transport-reaction equation collapses to `dC/dtau = -R(C)`, which contains no
geometry and no dimension. In `tau` coordinates the reaction front sits at a
fixed `tau` whatever the pore structure — the trunk is handed a coordinate in
which the answer is nearly trivial, and sharp advective fronts are exactly where
DeepONets normally struggle.

**The honest weakness.** Where the fluid does not move, the flow field carries
*no* information about the pore space. In a dead-end pore, or anywhere at
Pe = 0.3, solute arrives by diffusion and a zero velocity cannot distinguish
"deep dead-end pore" from "solid" — the geodesic field still can. So expect
`--flow-proxy` to match or beat `--distance gdf` at Pe 10 and 30 and to **lose at
Pe 0.3**. The new scenario grid spans both, so this is measurable rather than
arguable. Measured on a real D3Q19 Stokes field at porosity 0.45: 8 % of pore
voxels stagnant, median `tau` 0.76, 99th percentile 24, maximum 186.

**Options**

```
--flow-mode tau          (default) trunk gets tau
--flow-mode speed        trunk gets |u| instead
--flow-mode both         trunk gets gdf AND tau -- the safe choice, six columns
--keep-geometry-channel  keep the pore mask as a 4th branch channel
--u-floor 0.01           velocity floor as a fraction of the mean pore speed,
                         which keeps tau finite in stagnant zones
```

**At inference you now need a velocity field.** That is the point, not a
limitation: `geometry -> Geo-ONet -> u -> PRT-DeepONet -> C` is a cascade where
we own only the second stage. `predict.py` requires `--velocity vel.npy` for a
flow-proxy checkpoint and refuses with an explanation otherwise.

---

## Switch B — `--transfer-2d`

**Not from Christof — this one is ours.** It attacks the real constraint: 3D runs
cost 5.5 to 16 h each, so 3D geometries are the scarce resource, while Jung's
release ships 2000 2D domains and a 2D solve is roughly fifty times cheaper.

**What it does.** Extrudes 2D domains along z. The extruded medium is prismatic,
so nothing varies in z and **the exact 3D solution is the 2D solution repeated**.
This is not domain adaptation and not an approximation — an extruded 2D domain is
a genuine member of the 3D problem class. `build_transfer_set_2d_to_3d.py` measures the z-variation
of the solved flow field and reports it; on our test set it is exactly 0.

```
python tools/build_transfer_set_2d_to_3d.py --jung-dir /path/to/PRT-DeepONet-main \
                          --out train2d.h5 --target-shape 128 64 64
python tools/build_transfer_set_2d_to_3d.py --synthetic 40 --out train2d.h5   # no 2D source yet
```

**Two ways to use it.**

```
# joint: mix 2D into every epoch
python model/train.py --data 3d.h5 --transfer-2d train2d.h5 --transfer-2d-frac 0.3

# two-stage: pretrain on 2D, fine-tune on 3D with the trunk frozen
python model/train.py --data train2d.h5 --out runs/pre --dim-free
python model/train.py --data 3d.h5 --init-from runs/pre/best.pt --freeze-trunk --dim-free
```

**Why freeze the trunk.** The reaction physics transfers completely — rate laws,
the Pe and Da response, front shapes, Monod saturation; none of it contains a
dimension. The **topology transfers not at all**: a thresholded Gaussian field
percolates near porosity 0.20 in 3D and near 0.59 in 2D, and a 3D pore network
has far more routes around an obstacle. So transfer the trunk and the parameter
branch, retrain the geometry branch. That split happens to be exactly the
DeepONet factorisation.

**Keep `--transfer-2d-frac` at or below ~0.3.** An extruded domain has zero
z-tortuosity; at a high mixing fraction the network learns the shortcut "nothing
ever happens in z".

---

## Switch C — `--dim-free`

Switches A and B together, and the reason they belong together: **the thing that
makes 2D→3D transfer hard is the geometry domain gap, and switch A removes
geometry from the input.**

A 2D velocity field and a 3D velocity field are far more alike than a 2D binary
image and a 3D binary volume — both divergence-free, both channelling, both with
comparable `|u|/<u>` statistics. The binary phase function is not.

**What it does.** Replaces the Cartesian trunk with the flow-space trunk.

| trunk    | 2D                | 3D                   | transfers? |
|----------|-------------------|----------------------|------------|
| OFF      | x, y, t, gdf (4)  | x, y, z, t, gdf (5)  | no — different widths |
| `--dim-free` | t, dwall, tau (3) | t, dwall, tau (3) | **yes — identical** |

`dwall` is the distance to the nearest solid, normalised by the transverse
half-width so it means the same thing in both. The trunk needs no kernel
inflation, no architecture surgery, no zero-padded z column: it is literally the
same network. `--dim-free` implies `--flow-proxy`.

---

## Running the experiment

```
python model/run_ablation_sweep.py --data 3d.h5 --data-2d train2d.h5 --out sweep
```

Trains one model per configuration, evaluates them all on the same held-out
geometries, and prints mean RMSE **split by Peclet band** — which is what
actually settles the question:

```
configuration      RMSE all        Pe<1    1<=Pe<10      Pe>=10
baseline             ...           ...        ...         ...
flow-tau             ...           ...        ...         ...
flow-both            ...           ...        ...         ...
dim-free             ...           ...        ...         ...
dim-free+2D          ...           ...        ...         ...
dim-free+2Dpre       ...           ...        ...         ...
```

Expected shape of the result: flow-proxy wins the `Pe>=10` column, loses the
`Pe<1` column, and `flow-both` is at least as good as the better of the two
everywhere. If `flow-both` is not, the extra trunk column is not being used and
something is wrong.

The row that matters for the compute argument is `dim-free+2Dpre`: pretrain on
free 2D data, fine-tune on a handful of 3D geometries. If it beats `baseline`
trained on all of them, we need far fewer 16-hour runs.

---

## New and changed files

| file | what |
|------|------|
| `tools/flow_coordinates.py` | **new** — travel time, normalised velocity, wall distance, `--self-test` |
| `tools/build_transfer_set_2d_to_3d.py` | **new** — 2D domains → extruded 3D h5 |
| `tools/build_practice_dataset.py` | **new** — a small but real dataset for testing without the cluster |
| `tools/test_three_switches.py` | **new** — the regression test, including OFF == original |
| `tools/dataset_reader.py` | `resolve_switches()` decides the layout; one place, no drift |
| `model/train.py` | the switch flags; records them in the checkpoint |
| `model/evaluate.py` | rebuilds the exact training configuration from the checkpoint |
| `model/predict.py` | flow-proxy inference path, `--velocity`, travel-time diagnostics |
| `model/deeponet_model.py` | clamps CNN blocks to what the grid supports |
| `model/run_ablation_sweep.py` | **new** — runs the whole comparison and the Peclet breakdown |

Old checkpoints keep loading: they carry no switch keys and every switch
defaults to off.

The flow pipeline added eleven more files and changed ten. They are listed with the rest of the
project in `README.md` at the top of this folder, each marked **NEW** or **CHANGED**,
and every one of them carries a block at the top of itself saying where its code came
from and what was changed.

| file | what |
|------|------|
| `tools/flow_features.py` | **new** — MIS, UPRM and dw², in two and three dimensions, `--self-test` |
| `tools/harmonic_pressure.py` | **new** — the sparse Laplace solve behind the velocity trunk's prior, `--self-test` |
| `tools/add_flow_features.py` | **new** — writes all three into a dataset you already have |
| `model/velocity_model.py` | **new** — the velocity operator, the pressure U-Net, the ROI Huber loss, the divergence penalty |
| `model/train_velocity.py` | **new** — trains the operator |
| `model/predict_velocity.py` | **new** — runs it, and writes `samples/velocity_pred` back |
| `model/test_flow_pipeline.py` | **new** — the whole velocity path end to end, 33 checks |
| `model/test_reference_parity.py` | **new** — our descriptors and predicted field against theirs, value by value |
| `../gui/test_flow_panel.py` | **new** — the window's flow pipeline panel: its steps, and whether they line up |
| `tools/dataset_reader.py` | the velocity-informed branch layout, the velocity z scoring, four guards |
| `model/train.py` | `--velocity-informed`, `--geom-features`, and the guard against A and D together |
| `tools/collect_complab_output.py`, `tools/collect_foreign_complab.py` | write the three descriptors at collection time |
| `../gui/prt_gui.py` | the flow pipeline panel, five new actions, a new sidebar group |
| `../check_everything.py` | four more check groups |


---

## The flow pipeline — `--velocity-informed`

Not a fourth switch, and it is worth being plain about why. A, B and C are one flag
each: you set one, you press Run, you get a model. This is four stages that have to
run against the same dataset before the fifth can read what they wrote, so the window
gives it a page of its own rather than a box beside the other three.

**The question it answers:** does giving the concentration network the flow field,
and changing nothing else, make it better?

**Where to find it.** On the command line it is a flag on `train.py`, like the other
three. In the window it is not: it has a panel of its own, **The flow field → The flow
pipeline**. A, B and C each change one training run and need nothing prepared. D needs
three earlier stages to have run against the same dataset first, and a tick box beside
the other three would let you start the last stage without them. The failure arrives
minutes later as a missing key inside an HDF5 file, which is why the window shows the
sequence as a sequence.

**What it does.** Adds the velocity components to the CNN branch as extra image
channels. The trunk is left exactly as it was.

|                | OFF                 | `--velocity-informed`        |
|----------------|---------------------|------------------------------|
| branch1        | material (1 ch)     | material, ux, uy, uz (4 ch)  |
| trunk          | x, y, z, t, **gdf** | x, y, z, t, **gdf**          |

**This is not switch A.** A REPLACES the geodesic distance in the trunk with a flow
coordinate. D leaves the trunk untouched and adds to the branch. Asking for both is
asking for two different branch layouts at once, so `train.py` refuses the
combination rather than silently picking one.

**Where it comes from.** Jo, Kim, Kim, Lim, Choi, Ryu and Jung, *PRT-DeepONet with
Sequentially Coupled Velocity and Concentration Prediction for Pore-Scale Reactive
Transport*, preprint SSRN 7388394. In 2D they measured the median concentration RMSE
falling from 0.0304 to 0.0263, the 90th percentile from 0.0570 to 0.0442, and the
worst case from 0.1739 to 0.1126.

**The sharpest result is a negative one.** They fed the same velocity to the TRUNK
instead, as two numbers at each query point, and it recovered almost nothing: median
0.0297 against the 0.0304 baseline. The whole gain depends on the field arriving as a
field the convolution can read. A pointwise velocity says how fast the fluid moves
here; it says nothing about what this point is connected to.

**Two ways to get the field.**

```
--velocity-informed simulated    samples/velocity, what the solver produced
--velocity-informed predicted    samples/velocity_pred, from the velocity operator
```

`simulated` needs no velocity model at all and measures the CEILING a predicted field
is chasing. It is the cheaper experiment and it should be run first: if the simulated
field does not help, a network trained to approximate it will not help either.
`predicted` is the published two stage pipeline, and needs
`predict_velocity.py --write-back` to have run over the dataset.

**Optionally, the pore size maps.** `--geom-features` adds MIS and UPRM to the branch
as well. MIS is how wide the pore is at this voxel; UPRM is how wide the narrowest
throat between the inlet and this voxel is, which is not local and is the part a
convolution cannot work out for itself. The dataset must carry them:
`add_flow_features.py` writes them into an existing file without recollecting it.

**Why the velocity is z scored on the way in.** The raw field is around 1e-4 in
lattice units while the pore mask channel is 0 or 1. Fed unscaled, the velocity
channel arrives four orders of magnitude below its neighbour and the first convolution
barely registers it. The reference implementation z scores it and so does this.

### The velocity operator

The `predicted` route needs a trained flow model. That is a second operator with
the same branch-trunk architecture and different inputs:

| | |
|---|---|
| branch CNN | pore mask, MIS, UPRM |
| branch FNN | one flow condition (Pe here; the published model uses Re, which our datasets do not record) |
| trunk | position, squared wall distance, the harmonic pressure gradient components |
| output | ux, uy [, uz] |

The loss is a Huber term over the pore space plus `lambda` times the mean squared
divergence. Their sweep found the velocity error flat up to lambda 10 and clearly
degrading past 100 while the divergence error fell throughout, so 10 is the default.

The whole pipeline on the command line, in the order the window's panel runs it:

```
# 1. the descriptors, only if the dataset predates the collectors writing them
python tools/add_flow_features.py --data dataset.h5 --buffer 5

# 2. THE CONTROL. No velocity model at all. Run this one first.
python model/train.py --data dataset.h5 --out runs/D_sim --velocity-informed simulated

# 3. the operator
python model/train_velocity.py --data dataset.h5 --out runs/vel

# 4. run it, and write the fields back beside the simulated ones
python model/predict_velocity.py --checkpoint runs/vel/best.pt --data dataset.h5 --write-back

# 5. the same run as step 2, on the predicted field instead
python model/train.py --data dataset.h5 --out runs/D_pred --velocity-informed predicted
```

Three numbers at the end: the run with the flag off, `runs/D_sim`, and `runs/D_pred`.
If `D_sim` does not beat the baseline, stop at step 2. If it does, the gap between
`D_sim` and `D_pred` is what the velocity operator costs you.

`model/test_reference_parity.py --reference <their repo>` checks our descriptors
against theirs on their own bundled domain. It currently reports every feature and the
full predicted velocity field identical to theirs, which is what makes a comparison
with their published numbers meaningful rather than a comparison with our
reimplementation.
