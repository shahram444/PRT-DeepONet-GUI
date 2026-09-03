# Contributing

Bug reports, questions and patches are all welcome. This file says what makes
each of them easy to act on.

## Before you change anything

Run the whole self-test suite and note what it says:

```bash
python check_everything.py
```

It runs every check in the project as a separate process, so one crash cannot
hide the others. Run it again after your change and compare. A change that
turns a passing check red needs either a fix or an explanation of why the old
expectation was wrong.

## Reporting a problem

The useful report has four parts: what you ran, what you expected, what
happened, and the exact command. Every page of the desktop application shows
the command line it is about to execute, so copying that line costs nothing and
saves a round trip.

Please include the version of Python, the operating system, and, when a dataset
is involved, the output of

```bash
python 3D/tools/dataset_reader.py your.h5
```

which prints the layout and what it would hand the network, without the data
itself.

If you cannot share the data, the demo campaign reproduces most problems:

```bash
cd 3D/tools
python make_demo_complab.py --out ../../work/demo
```

## Sending a change

Keep one change per pull request, and say in the description what measurement
told you the change was needed. "Fixes the axis order" is hard to review;
"the y and z axes were swapped in the 3D writer, which the round trip test in
test_flow_solvers.py now catches" is not.

Two conventions the code follows and that a patch should keep.

**Nothing is invented to fill a gap.** When a value cannot be found, it is
recorded as absent along with where the search looked. A missing input file, a
missing chemical, a missing snapshot: each is noted and the rest is kept. What
must not happen is a plausible number appearing where no number was measured.

**Comments say why, not what.** The line above a fix should say what went
wrong and how it was found, because that is the part nobody can reconstruct
later from the code.

## Ported code is not ours to refactor

`3D/model/velocity_model.py` and the two dimensional half of
`3D/tools/flow_features.py` are ports of code released by the Jung Lab. Their channel
ladders, block counts, layer widths and the latent split reproduce the release exactly,
and that is load bearing: their trained checkpoints load into these classes, and
`3D/model/test_reference_parity.py` checks our descriptors against theirs value for
value on their own example domain.

Change any of it and both stop being true, and every comparison with their published
numbers silently becomes a comparison with our reimplementation. If you need different
behaviour, add it beside the port rather than inside it. The three dimensional forms
are ours and are fair game.

## The two halves have different owners

`2D/` is the published release of Kim and Jung, kept exactly as published and
never written to. If a change belongs there, it belongs upstream rather than
here. Everything else is ours and is fair game.

## Licence

By contributing you agree that your contribution is licensed under the GNU
General Public License, version 3 or later, the same as the rest of the
project. See `LICENSE` and `LICENSING.md`.
