# Version 3.0.0, and how to tag it

Run these on Windows, in Git Bash or PowerShell, from inside
`PRT-DeepONet-GUI`. They are not run from the cloud session: git cannot write
inside the OneDrive mount from there, which is why a few `tmp_obj_*` files may
be sitting in `.git/objects/`. The first command clears them.

```bash
git gc --prune=now
git add -A
git commit -m "v3.0.0: one shared core, the checkpoint says what it is for, every file commented"
git tag -a v3.0.0 -m "Version 3.0.0"
git push origin main
git push origin v3.0.0
```

## What is in it

**prt_core/**, four files that replace copies that lived in several places.

| file | what it owns | what it replaced |
|---|---|---|
| `reactions.py` | which chemistry a model is for | widths typed into five scripts |
| `conventions.py` | the trunk's distance column and the two ways it is scaled | two anti-correlated conventions with no name for either |
| `inputs.py` | one reader for `.h5`, `.npz`, `.npy`, `.vti`, `.dat` | four readers that disagreed |
| `model.py` | the network, in both key layouts | three copies of one architecture |

`3D/model/deeponet_model.py` and `2D_scripts/prt2d_model.py` now re-export from
`prt_core/model.py`, so every existing import still works and there is one
definition behind them.

**The checkpoint records what it is for.** `--reaction` and
`--distance-convention` go into `best.pt` and `summary.json`. Neither changes a
tensor shape, which is why they had to be written down: `evaluate.py` and
`predict.py` compare them and say by name when they disagree, and `predict.py`
rebuilds the distance column under the convention the checkpoint records.

**Running without the window.** `prt.py` is one command line for everything,
with the same flags and defaults the buttons use.

**Every file is commented.** `python audit_comments.py --quiet` reported 33
files short of the rule and now reports none. The audit itself was extended to
cover `prt_core/` and `2D_scripts/`.

## Check it

```bash
python prt_core/test_core.py                  # the core, 4 self-tests + 13 crossing checks
python 2D_scripts/test_against_notebooks.py   # the published notebooks, difference 0.000e+00
python smoke_test.py                          # 14 checks, about a minute
python run_tests.py                           # everything, including the slow physics
python audit_comments.py --quiet              # every file carries what it should
```

Last full run in the cloud session, on 2 cores:

```
every file parses        ok     0.4s
the shared core          ok    11.6s   prt_core: everything passed.
the network              ok     4.0s   15 tests
the dataset layer        ok     4.0s   11 tests
evaluate and predict     ok    28.3s    8 tests
your own settings        ok    14.5s
flow coordinates         ok     1.6s
the flow descriptors     ok     1.2s
the pressure solve       ok     0.9s
the three switches       ok     6.5s
the velocity pipeline    ok    44.6s
what the buttons send    ok     5.0s
the flow pipeline panel  ok     3.4s   35 checks
the sweep boxes          ok     3.8s
the 2D simulator         ok   569.0s
the 3D simulator         ok   404.0s
the flow solvers         ok    43.5s   28 checks, 28 passed
the documented numbers   skipped        its fixtures are not built in that checkout
```

The three published reactions still reproduce their notebooks exactly, at all
four stages: the distance column, the three input tensors, the raw prediction
and the notebook's own plotted output.
