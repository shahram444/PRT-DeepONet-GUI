# PRT-DeepONet Studio

A geometry-aware neural operator for pore-scale reactive transport, together
with **PRT-LB**, the lattice Boltzmann simulator that generates its training
data, and a desktop application that drives the whole workflow.

Pore-scale reactive transport is accurate and slow: one three-dimensional
simulation takes hours. This project builds, trains and checks a neural
operator that predicts the same fields in milliseconds.

```
PRT-DeepONet/
├── 2D/         the published PRT-DeepONet release, unmodified
├── 3D/         the geometry-aware 3D extension
├── bridge/     the one place the two halves meet
├── gui/        the desktop application
└── docs/       guides for each part, and the methods paper
```

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

**Transport and react.** Up to four species carried through that flow by a
flux-form finite volume method with a minmod slope limiter, harmonic-mean
diffusivities at every face, and a divergence correction. Biotic (dual Monod)
and abiotic reactions can each be switched on or off independently, so an
abiotic-only run means no microbes, no biomass, and a reaction that still
happens.

**Screen every run before keeping it.** A simulation is rejected if any
species rises above what it could have been made from, goes negative,
oscillates voxel to voxel along the flow, reaches the outlet before crossing
the middle, turns out to be a rescaled copy of another species, stays at zero
while being fed, or carries a reaction rate for a species that is absent
everywhere.

**Train, evaluate and predict.** A DeepONet with a convolutional branch for
the geometry, a fully connected branch for the dimensionless numbers, and a
trunk taking position, time and distance. Training, ablation sweeps,
evaluation against held-out runs, figures, and prediction on new geometries.

**Transfer from 2D to 3D.** Extruded 2D domains are prismatic, so the exact 3D
solution is the 2D solution repeated. That makes 3000 cheap 2D domains
genuine members of the 3D problem class rather than an approximation.

**Do all of it behind buttons.** The desktop application states what goes in,
what happens and what comes out before anything runs, and no command is ever
hidden: every page shows the exact command line it will execute, so you can
copy it and run it yourself.

---

## What the published 2D release does not do

`2D/` is a complete and working piece of work, and everything below is an
addition rather than a correction.

| capability | 2D release | this project |
|---|---|---|
| Three-dimensional domains | no | yes, D3Q19 |
| Its own flow and transport solver | no, domains and fields are supplied | yes, PRT-LB generates everything |
| Biotic and abiotic reactions together | one reaction type per model | both, each switchable |
| Transient four-species output | Monod, sorption variants | donor, acceptor, product, biomass, all in time |
| Surface-weighted abiotic kinetics | no | yes, weighted by solid-facing faces |
| Biofilm as a slower-diffusing phase | no | yes, with harmonic-mean face diffusivity |
| Automatic dataset generation and screening | no | yes, with eleven rejection tests |
| A flow-field input in place of the geodesic distance | no | yes, switch A |
| One set of trunk inputs valid in 2D and 3D | no | yes, switch C |
| A desktop application over the whole workflow | notebooks | yes |
| Verification against analytic solutions | not shipped | ten groups of checks |

---

## Inputs and outputs

**What goes in.** A pore structure as an integer voxel array, either generated
or supplied, using 0 for solid, 1 for the solid surface, 2 for pore and 3 for
biofilm. Then six dimensionless numbers: the Péclet number, a Damköhler number
for each of the two reactions, the two half-saturation constants and the
yield. Optionally a biofilm thickness, a choice of reference length, and which
species the abiotic reaction consumes.

**What comes out.** For every run: the concentration field of each species at
logarithmically spaced times, the steady velocity field, the two distance
maps, the material array, and the full parameter set that produced them,
including the pressure gradient actually used and whether the run settled or
hit the step limit. Optionally VTI files and PNG slices for viewing in
ParaView.

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
becomes one file rather than a directory of hundreds of thousands, and each
run carries its own Péclet number, Damköhler numbers, porosity, reference
length convention and settling status as attributes on the group, so a field
can never be separated from the parameters that produced it.

Read one with `3D/tools/dataset_reader.py`, which prints the tree and pulls
out any run by index.

If HDF5 is new to you, this is a good introduction:
https://www.neonscience.org/resources/learning-hub/tutorials/about-hdf5

---

## The simulator

| file | what it holds |
|------|---------------|
| `3D/tools/prtlb_2d.py` | the 2D simulator: flow, transport and reactions |
| `3D/tools/prtlb_3d.py` | the 3D simulator, same structure |

Both files are organised into eleven numbered blocks with a map at the top, so
the flow solver, the pressure gradient, advection, dispersion, the boundary
and initial conditions, and the biotic and abiotic reactions can each be found
without reading the rest.

`docs/METHODS_PRT-LB.docx` describes the method in full, with equations,
verification tables and references.

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

A short end-to-end run, about five seconds:

```bash
cd 3D/tools
python build_practice_dataset.py --out practice.h5
python test_three_switches.py --data practice.h5
```

---

## What is in each folder

| folder | contents |
|--------|----------|
| `3D/tools/` | the two simulators, dataset builders, geometry generation, the CompLaB interface, the test suite |
| `3D/model/` | the DeepONet: training, prediction, evaluation, figures |
| `gui/` | the desktop application that drives every script above |
| `bridge/` | transfer learning from 2D to 3D |
| `2D/` | the published two-dimensional predecessor, unmodified |
| `docs/` | one guide per part of the code, plus the methods paper |
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

## Why 2D data is worth reusing

A 3D run costs between five and sixteen hours, while `2D/Domains` already
holds 3000 domains and a 2D solve is far cheaper. Extrude a 2D domain along z
and the medium becomes prismatic: nothing varies in z, so the exact 3D
solution is the 2D solution repeated. `bridge/build_transfer_set.py` measures
the z-variation of the solved flow field and reports it rather than asserting
it; on the real domains it comes out at exactly zero.

```bash
cd bridge
python build_transfer_set.py --limit 200
python run_experiment.py --data ../3D/dataset/dataset_reader.h5
```

The second command prints mean RMSE per configuration split by Péclet band,
which is what settles the question: the flow field should win at high Péclet
and lose at Pe = 0.3, where solute arrives by diffusion and a stagnant
velocity field cannot tell a deep dead-end pore from solid.

---

## A trap worth knowing about

The two 2D file formats use **opposite array orderings**:

```
2D/Domains/domain_*.dat       reshape(148, 64) C-order      ->  1 pore cluster
2D/geometries/Domain_*.npz    reshape(64, 148) then .T      ->  1 pore cluster
```

Apply the `.dat` convention to a `.npz` and the domain is shredded into 81
disconnected pore clusters that fail any percolation test, silently and with
no error. Everything in `bridge/` detects the ordering per file rather than
assuming it.

The material codes differ too. The 2D release uses 0 = solid, 1 = pore,
2 = interface, while CompLaB uses 0 = solid, 1 = wall, 2 = pore, so codes 1
and 2 are swapped. `bridge/` handles the translation; nothing else needs to.

---

## The domain files

`2D/Domains` ships as a single `Input_domains.zip` rather than as 3000 loose
`.dat` files. Unzip it in place before running anything that reads them.

---

## Licence, attribution and citation

### Authors

| | | |
|---|---|---|
| Shahram Asgari | author | shahram.asgari@uga.edu |
| Christof Meile | principal investigator | cmeile@uga.edu |

Meile Lab, Department of Marine Sciences, University of Georgia, Athens,
Georgia 30602, USA.

Everything in `2D/` is the published PRT-DeepONet release of Yehoon Kim
(wnsla7323@naver.com) and Heewon Jung (hjung@cnu.ac.kr), Jung Lab, Chungnam
National University, redistributed unmodified.

### Licence

The whole project is released under the **GNU General Public License, version
3 or later**, which is the licence `2D/` carries and which this work therefore
inherits. Commercial and industrial use is not covered by those terms and
requires a separate licence; write to the addresses above.

The legal text is in `LICENSE`, with the authorship and citation header at the
top, and in `COPYING` without it.

### How to cite

> Asgari, S., Meile, C., 2026. PRT-DeepONet Studio, version 1.0. Meile Lab,
> Department of Marine Sciences, University of Georgia, Athens, Georgia, USA.

> Kim, Y., Jung, H., 2025. PRT-DeepONet. Jung Lab, Chungnam National
> University, Republic of Korea.

If you use both halves, cite both. `CITATION.cff` holds the same information
in machine-readable form, which is what puts a "Cite this repository" button
on the GitHub page.

`LICENSING.md` gives the full picture, including two unresolved questions
about dual licensing that should be settled before this is distributed outside
the group.
