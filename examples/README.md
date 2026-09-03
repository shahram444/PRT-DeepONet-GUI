# examples

Four input files you can read without running anything, and hand back to the
code when you want to.

| file | read by | what it is |
|---|---|---|
| `example_settings_2d.xml` | `3D/tools/build_dataset_2d.py`, `build_dataset_3d.py` | every setting the Python generators take, with a comment on each saying what it changes and in what units |
| `example_CompLaB.xml` | CompLaB, and `3D/tools/collect_foreign_complab.py` | one CompLaB run's input file, in the campaign dialect |
| `example_defineKinetics.hh` | CompLaB, compiled in | the biotic rate law: `Ac + A -> P`, driven by the microbe `Bio` |
| `example_defineAbioticKinetics.hh` | CompLaB, compiled in | the abiotic rate law, off in this example |

## The settings file

It was written by the generator itself:

```bash
python 3D/tools/build_dataset_2d.py --out work/scratch.h5 \
                                    --save-settings examples/example_settings_2d.xml
```

so it is always exactly the set of settings the code actually reads, not a
list somebody kept up to date by hand. Edit it and pass it back:

```bash
python 3D/tools/build_dataset_2d.py --out work/mine.h5 \
                                    --settings examples/example_settings_2d.xml
```

`--show-settings` prints the same thing to the screen and writes nothing.

## The CompLaB files

These three came out of `3D/tools/make_demo_complab.py`, which writes a
campaign in the exact layout finished CompLaB output has. They are the real
dialect: the collector reads this XML for the pore material code, the species
names and their file prefix, the Peclet number, the relaxation time and the
voxel size.

The chemistry is one microbial reaction:

```
Ac + A -> P        R = Vmax * Bio * Ac/(Ks_ac+Ac) * A/(Ks_a+A)
```

with the chemical order `C[0]=Ac, C[1]=A, C[2]=P` and one microbe `B[0]=Bio`.
That order must match between the XML and the header, and it is the first
thing to check when a collected dataset has the right shape and the wrong
chemicals in it.

The numbers in the demo campaign these came from are constructed rather than
simulated. The input files themselves are not: they are what a real run of
this chemistry would be given.

## Nothing here is needed for the flow capability

Switch D and the velocity operator read the geometry out of a dataset and work
everything else out themselves. There is no extra input file to write, and no setting
in `example_settings_2d.xml` that turns them on: they are command line flags on
`train.py` and on `train_velocity.py`. The one number you do have to get right is the
open buffer at each end of the flow axis, and it is a flag rather than a setting
because it describes the campaign you already have rather than one you are about to
run.
