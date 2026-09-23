# prt_core

Four things every script in this project needs, written down once.

The repository grew outwards from three published notebooks, and each new
script carried its own copy of whatever it needed from them. By the time the
audit was written there were three copies of the network, four readers for a
geometry file, two distance conventions with no name for either, and the
chemistry hard-wired as constructor arguments in five places.

None of that is a bug on its own. The bugs are what the copies allow, and they
share a shape: every tensor agrees, nothing raises, and the answer is merely
wrong.

| file | what it owns | what it replaced |
|---|---|---|
| `reactions.py` | which chemistry a model is for: species, dimensionless numbers with the ranges they were trained over, steady or transient | widths typed into five scripts |
| `conventions.py` | the trunk's distance column and the two ways it is scaled | two anti-correlated conventions with no name for either |
| `inputs.py` | one reader for `.h5`, `.npz`, `.npy`, `.vti` and `.dat` | four readers that disagreed |
| `model.py` | the network, in both key layouts, built from a reaction | three copies of one architecture |

## The distance convention, which is the one worth reading about

Measured on the published release's own Monod domain:

```
published : 0.005 to 1.000 over the pore, HIGH at the inlet, solid 0
ours      : 0.000 to 3.349 over the pore, ZERO at the inlet, solid 0
correlation over the pore space: -1.0000
```

Perfectly anti-correlated, and on different scales. Both are defensible.
Neither is wrong. What was wrong is having two of them and no name for either,
because then a warm start from the published weights hands a trained trunk a
column that means the reverse of what it was fitted to.

There are three, not two. `irreversible_sorption` scales over the pore only and
the other two published reactions scale over the whole grid, which is the
notebooks' own difference and not ours.

```
ours             geodesic distance from the inlet in voxels, over nx - 1
published        the same walk, negated, scaled over the WHOLE GRID
published_pore   the same, scaled over the PORE only
```

A model records which one it was trained under. `predict.py` rebuilds the
column under that convention rather than under whichever the calling script
assumed, and `evaluate.py` says by name when a checkpoint and a run disagree.

The check that makes the naming worth anything is in `test_core.py`: the
convention named `published` must reproduce the released notebook's own column
on the released rock, voxel for voxel. It does, with a difference of exactly
zero.

## Adding a chemistry

A table entry in `reactions.py`, or a file, with no change to any code:

```bash
python prt.py train --data d.h5 --out runs/a --reaction my_chemistry.json
python prt.py train --data d.h5 --out runs/a --reaction CompLaB.xml
```

The `.xml` route reads `<name_of_substrates>`, so a CompLaB settings file that
already exists defines the chemistry without anything being retyped.

## 2D and 3D are one code path

A grid whose third number is 1 is a two-dimensional problem. The geometry
branch switches to `Conv2d` and `AvgPool2d`, the trunk drops its `z` column,
and the velocity channels drop `uz`. Nothing selects this by hand; it follows
the data, so there is no 2D fork to keep in step.

The identity that makes the published weights transferable:

```
148 x 64      halved five times  ->  4 x 2 at 256 channels  = 2048
64 x 64 x 64  halved five times  ->  2 x 2 x 2 at 256       = 2048
```

`model.py`'s self-test asserts it rather than trusting it.

## Checking it

```bash
python prt_core/test_core.py           # all four, plus what only holds between them
python prt_core/reactions.py --self-test
python prt_core/conventions.py --self-test
python prt_core/inputs.py --self-test
python prt_core/model.py --self-test
```

`test_core.py` is the first entry in `smoke_test.py` and in `run_tests.py`,
because everything below it imports from here: if it fails, the failures
further down are echoes.
