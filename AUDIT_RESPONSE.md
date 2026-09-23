# Response to the code audit

Every item in the audit, what was done about it, and how to see that it was
done. Each fix carries an `AUDIT <ID>` comment at the place it was made, so
`grep -rn "AUDIT " .` walks the whole list in the code itself.

Nothing here was fixed by reading. Every entry names the command that shows the
old behaviour is gone.

## The findings

| ID | what was wrong | what was done | check it |
|---|---|---|---|
| TRAIN-04 | one held-out set both chose the checkpoint and was reported as the final score | three disjoint sets of whole geometries. Validation chooses, test is opened once afterwards and is what gets reported. The exact geometry ids go into the checkpoint so `evaluate.py` reproduces the split instead of recomputing it | `prt train --data d.h5 --out r` prints train, val and test separately, and the two numbers differ |
| DATA-05 | each split fitted its own velocity mean and standard deviation | fitted once on the training split, passed to the others, recorded in the checkpoint and reused by evaluation and prediction. The 64-row sample is now spread across the split instead of being the first 64 in order | the run prints `velocity : scaled by the training split, mu ... sd ...` |
| DATA-04 | the concentration scale came from every run in the file, test rows included | the target is divided by a scale fitted on the training rows, recorded in the checkpoint. The file's own scale stays as the physical unit | `--target-scale file` restores the old behaviour; the default prints the fitted one |
| PRESS-04 | a central difference at a pore voxel read the zero stored in the grain, inventing a wall-normal gradient | one-sided differences where a neighbour is solid, which is what the no-flux wall condition means | `python 3D/tools/harmonic_pressure.py --self-test` now includes "a straight channel has no gradient across it" |
| FLOW-03 | `np.concatenate` on an empty list raised `ValueError` when every map was zero | collect first, test, then concatenate | `zscore_stats([])` returns the neutral scaling |
| FLOWCOORD-07 | `inlet_axis` was an argument but nothing checked that the rest honoured it | a self-test that rotates the rock and asks for the matching axis, and requires the same field rotated | `python 3D/tools/flow_coordinates.py --self-test`, checks 5b |
| VELMODEL-09 | shape mismatches were printed after a call that raises on them | the mismatches are found first, dropped from the load with the reason said out loud, and the tensors that fit are still loaded | load a checkpoint of the wrong width: it now reports instead of raising |
| VELTRAIN-09 | the final held-out NRMSE described the first eight samples | the whole held-out set by default, and a limited run says how many of how many | the report line reads "over all N samples" |
| VELPRED-08 | `--pe` named the conditioning variable Peclet whatever the checkpoint said | `--condition`, with `--pe` kept as an alias, and the checkpoint's own name printed and used in the error | `prt velocity-predict --help` |
| PRTLB3D-03 | a self-test assertion ended in `or True`, so it could not fail | the `or True` is gone and the check passes on its own merits | `python 3D/tools/prtlb_3d.py`, and the same in 2D |
| TRANSFER-08 | runs that hit the step cap were written into the transfer set beside the settled ones | dropped by default, with the count printed; `--keep-unsettled` puts them back, and the choice is recorded in the file | `prt transfer --synthetic 4 --out t.h5` reports what it dropped |
| PRACTICE-12 | the metadata recorded an abiotic Damkohler the solver was never given | the recorded value is the one that was used | `3D/tools/build_practice_dataset.py`, the `da_abio` line |
| TOY2D-10 | more than eight prediction structures indexed past a fixed string | names that grow without limit, A to Z then AA | `--n-predict 30` works |
| IMPORT2D-03 | an unnamed array was chosen by shape, silently | guessing is off unless `--guess-arrays`, and an ambiguous guess refuses and names the candidates | shown in the response note below |
| IMPORT2D-04 | source times were replaced by an even ladder | the file's own times are kept, scaled and checked for being increasing; the even ladder is the fallback and the run says which happened | the import prints "every run kept the snapshot times recorded in its own file" |
| IMPORT2D-09 | `allow_pickle=True` on every load | off unless `--allow-pickle` | `prt import-2d --help` |
| BRIDGE-ROOT-01A | pointed at `3D/tools/ingest_2d.py`, renamed long ago | points at the real script, still accepts the old name, and says what it looked for | `python bridge/build_transfer_set.py --help` |
| BRIDGE-ROOT-01B | pointed at `3D/model/run_switch_sweep.py` | the same treatment | `python bridge/run_experiment.py --help` |
| GUI-02 | `test_flow_panel.py` tested a `FlowPipelinePage` that did not exist | the panel exists: five steps, one set of shared settings, stops at the first failure. Switch D is off the training page, because it is a sequence and not a box. The test is now in both runners | `python gui/test_flow_panel.py` — 35 checks |
| TESTCOMMON-08 | a fixture wrote to column 9 whatever the width was | the width is checked and the channel placed relative to it | `tests/_common.py`, `two_channels` |
| TEST-01 | hard-coded paths from one machine and from the project's old name | every path is an argument, the code is found relative to the file, a missing fixture is a skip with the command that builds it | `python 3D/tools/test_documented_numbers.py --help` works on a fresh checkout |
| TEST-RUNNER-02 | `MISSING` rows were reported and then left out of the failure set | a configured check that cannot be found is a failure. `skipped` now means only skipped | `run_tests.py`, `FAILING_STATUSES` |
| TEST-RUNNER-03 | the static scan walked into virtual environments and compiled other people's code | an explicit skip list: venvs, caches, site-packages, build output | `run_tests.py`, `SKIP_DIRS` |
| GUIINST-10 | `--check` returned success while reporting missing packages | it returns 0 only when everything is present | `python gui/install_requirements.py --check; echo $?` |
| GUILAUNCH-08 | with `set -u` and no interpreter found, bash died on an unset variable | it says what it looked for and what to install | `gui/prt-deeponet.sh` |
| VTI-01 | `--save-vti` and `--save-png` were accepted and wrote nothing, because `write_vti_and_png.py` was absent | the module is restored, with a self-test, and writes the same `.vti` format `predict.py` writes so simulated and predicted fields open side by side | `prt dataset2d --save-vti --save-png --save-runs 1 --out d.h5`, then look in `d_fields/` |

## Three more, found while working through the audit

| what | why it matters | what was done |
|---|---|---|
| the two collectors disagreed on normalisation | `collect_complab_output.py` stored raw mol/L and `collect_foreign_complab.py` stored the field already divided by `conc_scale`, and both wrote the same attribute. The reader divides regardless, so one route was scaled twice and nothing complained: both files load, train and score | both store the scaled field. Multiply by `conc_scale` for mol/L |
| the campaign collector looked for the wrong files | it searched `output/<species>_*.vti`, but the template it ships sets `<subs_filename>subsLattice</subs_filename>`, so CompLaB writes `subsLattice0_*.vti` and every run failed at "no .vti for species 'Ac'" | the species name is tried first, then the positional `subsLattice`/`bioLattice` name, and the error says where it looked |
| the foreign collector never wrote the flow descriptors | a variable name slip inside a broad `except`, so it printed a note and carried on. `geom/mis`, `geom/uprm` and `geom/dw2` were therefore always absent, and switch D with `--geom-features` was unusable on any file it produced | one identifier |
| the warm-start message was untrue | it promised that "shape changes are skipped, not silently reshaped". `load_state_dict` raises on a size mismatch whatever `strict` is set to | the mismatched tensors are dropped deliberately and listed, and the message says which route does work |

## The four defects the audit did not reach, fixed in version 3

The audit found bugs. Underneath them were four places where the same thing was
written down more than once, and each of those is a supply of future bugs of
exactly the kind the audit listed. They are one place each now, in `prt_core/`.

| what was duplicated | where it was | what it cost | where it is now |
|---|---|---|---|
| the network | `3D/model/deeponet_model.py`, `2D_scripts/prt2d_model.py`, and again inside each published notebook | the reversible sorption notebook's class default says four convolution blocks and it is built with five. Three copies means three chances for the next such difference to go unnoticed | `prt_core/model.py`. The other two files re-export from it, so every existing import still works and there is one definition behind them |
| the distance column | the dataset builders, `predict.py`, `prt2d_model.py` | two conventions with no name for either, anti-correlated at exactly -1.0000 on the published Monod domain. A warm start under the wrong one hands a trained trunk a column running backwards: the shapes all match, nothing raises, the prediction is merely poor | `prt_core/conventions.py`. Three named conventions, recorded in the checkpoint, converted by one function |
| which chemistry a model is for | constructor arguments typed into five scripts | a checkpoint could not say what it was trained for, so a model fitted on methane and sulfate loads into a run predicting acetate with every shape agreeing | `prt_core/reactions.py`. Five reactions as data; a sixth is a table entry or a `.json`, and a CompLaB `.xml` defines one without anything being retyped |
| reading a geometry file | `predict.py`, `build_transfer_set_2d_to_3d.py`, `import_2d_simulations.py`, `prt2d_model.py` | four readers that disagreed about what a material code means and about what to do when the geodesic field is absent | `prt_core/inputs.py`. One reader for `.h5`, `.npz`, `.npy`, `.vti` and `.dat`, which also decides 2D or 3D from the rock rather than from a flag |

Two things follow from this that are worth stating on their own.

**The checkpoint now records what it is for.** `--reaction` and
`--distance-convention` go into `best.pt` and into `summary.json`. Neither
changes a tensor shape, which is exactly why they had to be written down:
`evaluate.py` and `predict.py` compare them and say by name when they disagree,
and `predict.py` rebuilds the distance column under the convention the
checkpoint records rather than under whichever the calling script assumed. A
checkpoint written before this field existed still loads, and says so.

**The claim is checked against the published code, not against itself.**
`prt_core/test_core.py` requires the convention named `published` to reproduce
the released notebook's own distance column on the released Monod domain,
voxel for voxel, and gets a difference of exactly zero. The notebook parity test
still reports zero difference at all four stages for all three reactions after
the move.

Run it:

```
python prt_core/test_core.py
python 2D_scripts/test_against_notebooks.py
python prt.py reactions
python prt.py conventions
```

## Running without the window

Everything in this project already ran without the GUI: the window has never
done anything except build a command line and start it. What was missing was one
entry point that knows where the scripts are, which is what `prt.py` is. It is a
dispatcher and nothing else. It does not parse the arguments of the script it
calls, add defaults, or reorder anything, so a command written on a cluster
works verbatim in the window and the other way round.

```
python prt.py                    the commands, grouped in the order they are run
python prt.py help train         the full help for one
python prt.py core               check the shared core
python prt.py test               the fast gate
```

`gui/test_gui_commands.py` is what keeps the two honest: it takes every button
in the window and requires the script it calls to accept the flags it sends.

## Every file is commented

`python audit_comments.py --quiet` reported 33 files short of the rule: a top
block saying what changed from the 2D version, block comments marking the
sections, and line comments on the parts that look wrong until explained. It now
reports none, and the audit itself was extended to cover `prt_core/` and
`2D_scripts/`, so the new code is held to the same bar as the old.

## What the audit measured, and what it did not

The audit's runtime section is right that its numbers settle nothing about
accuracy: two geometries, four simulations, two epochs, and three of the four
transport runs stopped at the step cap. The speedup it quotes came from the
script's assumed 30 seconds per 2D simulation, not from a measurement.
`evaluate.py` states that assumption on every run and takes `--sim-seconds`;
nothing should be quoted until a real CompLaB wall time is put in.

The CompLaB path itself is still untested end to end. That needs the executable
and the module environment on GACRC, and it is the next thing worth doing.
