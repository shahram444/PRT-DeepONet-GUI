# PRT-DeepONet Studio

A geometry-aware neural operator for pore-scale reactive transport, together
with **PRT-LB**, the lattice Boltzmann simulator that generates its training
data, and a desktop application that drives the whole workflow.

Pore-scale reactive transport is accurate and slow: one three-dimensional
simulation takes hours. This project builds, trains and checks a neural
operator that predicts the same fields in milliseconds.

```
PRT-DeepONet/
├── 2D/         the published PRT-DeepONet release of Kim and Jung, unmodified
├── 3D/         the geometry-aware 3D extension
├── bridge/     the one place the two halves meet
├── gui/        the desktop application
├── docs/       one guide per part of the code
└── examples/   example input files, with a README
```

---

## Getting started in five minutes, with no cluster

You do not need CompLaB, a cluster, or hours of simulation to see the whole
chain work. There is a generator that writes a demo campaign in the exact
layout finished CompLaB output has.

```bash
pip install -r requirements.txt

cd 3D/tools
python make_demo_complab.py --out ../../work/demo

python collect_foreign_complab.py --runs ../../work/demo/demo_2D \
                                  --out  ../../work/demo/demo_2D_dataset
```

That gives you 25 two-dimensional runs over 5 distinct rocks, collected into
one `dataset.h5`. Repeat with `demo_3D` for the three-dimensional case. Open
either in the viewer:

```bash
python gui/prt_gui.py
```

The numbers in the demo are constructed rather than simulated, and the script
says so in its own output. The geometry is a real percolating packing and the
file layout is exactly what CompLaB writes, so it exercises the collector, the
viewer and the training code. It settles nothing about physics.

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
trunk taking position, time and distance. Training, ablation sweeps,
evaluation against held-out rocks, figures, and prediction on new geometries.

**Transfer from 2D to 3D.** Extruded 2D domains are prismatic, so the exact 3D
solution is the 2D solution repeated. That makes 3000 cheap 2D domains genuine
members of the 3D problem class rather than an approximation.

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
where the run wrote it, the two distance maps, the material array, the input
files the run was given, and the conditions that produced it.

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
not in it. Read one file with `3D/tools/dataset_reader.py`, which prints the
tree and pulls out any run by index.

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
| `3D/tools/` | the two simulators, dataset builders, geometry generation, the CompLaB interface, the demo generator, the test suite |
| `3D/model/` | the DeepONet: training, prediction, evaluation, figures |
| `gui/` | the desktop application that drives every script above |
| `bridge/` | transfer learning from 2D to 3D |
| `2D/` | the published two-dimensional predecessor of Kim and Jung, unmodified |
| `docs/` | one guide per part of the code, and the dataset format |
| `examples/` | a settings file, a CompLaB.xml and the two kinetics headers, with a README |
| `check_everything.py` | runs every self-check in the project |

Anything the code produces (geometries, datasets, trained weights, figures) is
written under `work/`, which is deliberately not tracked here.

---

## The three feature switches

All three default to **off**. With all three off the 3D code behaves exactly
as it did before they existed, and `3D/tools/test_three_switches.py` proves
that bit for bit against a copy of the original implementation.

| switch | flag | what changes | needs 2D data |
|--------|------|--------------|---------------|
| A | `--flow-proxy` | the trunk receives the advective travel time instead of the geodesic distance, and the branch receives the velocity field instead of the pore mask | no |
| B | `--transfer-2d H5` | extruded 2D domains are mixed into training | yes |
| C | `--dim-free` | the trunk becomes (t, dwall, tau), the same three inputs in 2D and 3D, so the network transfers unchanged | yes, to be useful |

Switch A asks whether the flow field can replace the geodesic distance
function. Switch B is the compute argument. Switch C is why they belong
together: what makes 2D to 3D transfer hard is the geometry domain gap, and
switch A removes geometry from the input. The full write-up is in
`3D/SWITCHES.md`.

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
