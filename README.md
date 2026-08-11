# PRT-DeepONet

A geometry-aware neural operator for pore-scale reactive transport, together
with **PRT-LB**, the lattice Boltzmann simulator that generates its training
data. Two- and three-dimensional versions of the same project.

```
PRT-DeepONet/
├── 2D/         the published PRT-DeepONet release, unmodified
├── 3D/         the geometry-aware 3D extension
├── bridge/     the one place the two halves meet
├── gui/        a desktop application that drives everything
└── docs/       guides for each part, and the methods paper
```

---

## The simulator

`PRT-LB` solves Stokes flow through a voxelised pore structure with a
two-relaxation-time lattice Boltzmann scheme, then transports up to four
chemical species through the resulting velocity field using a flux-form finite
volume method, applying biotic (Monod) and abiotic reactions at every step.
Each dimension lives in one self-contained file.

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

The graphical interface has its own install button if you prefer that route.
To confirm everything works:

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
| `3D/tools/` | the two simulators, dataset builders, geometry generation, the CompLaB interface, and the test suite |
| `3D/model/` | the DeepONet itself: training, prediction, evaluation, figures |
| `gui/` | a Tkinter desktop application that drives every script above |
| `bridge/` | transfer learning from 2D to 3D |
| `2D/` | the published two-dimensional predecessor, unmodified |
| `docs/` | one guide per part of the code, plus the methods paper |
| `check_everything.py` | runs every self-check in the project |

Anything the code produces (geometries, datasets, trained weights, figures)
is written under `work/`, which is deliberately not tracked here.

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
solution is the 2D solution repeated. An extruded 2D domain is therefore a
genuine member of the 3D problem class rather than an approximation.
`bridge/build_transfer_set.py` measures the z-variation of the solved flow
field and reports it rather than asserting it; on the real domains it comes
out at exactly zero.

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

## Attribution and licence

The code in `3D/`, `bridge/`, `gui/` and `docs/` is released under the MIT
licence; see `LICENSE`.

`2D/` is the published PRT-DeepONet release of Jung et al., redistributed
unmodified under its own terms. See `2D/LICENSE`.

## Citation

To be added on publication.
