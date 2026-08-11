# What was wrong, and what is different now

A record of one round of review and repair. It is written down because most of
these were **silent**: the code ran, produced plausible-looking files, and
reported encouraging numbers. Nothing crashed. That is the dangerous kind.

Run `python check_everything.py` at any time. It runs every self-check in the
project and prints one summary.

---

## The one that mattered most: there were two solvers

`adr_2d()` in `build_dataset_2d.py` and `adr()` in `build_practice_dataset.py` were two
independent implementations of the same advection–diffusion–reaction step. Six
bugs were found and fixed in the first. **All six were still in the second**,
and the second is the solver behind the 2D→3D transfer set — so switch B was
training on data with a periodic boundary leak, a Peclet number 24 times the
one requested, and four "chemicals" that were byte-identical copies.

Two implementations of one thing is not a tidiness problem. It is a
correctness problem, because fixing one makes the other *look* fixed.

There is now one solver, held word for word in `tools/prtlb_2d.py` and
`tools/prtlb_3d.py` and compared byte for byte on every test run, which works in two and three
dimensions. Both callers are three-line wrappers around it. It has a self-test
that checks it against answers that can be written down without it — plug flow
through an open channel must leave `exp(-Da)` at the outlet, and it does, to
within 2%.

### The six

| | What it was | What it did |
|---|---|---|
| 1 | `D = 1/Pe` instead of `nx/Pe` | simulated Peclet × grid width |
| 2 | a fixed step count | the front never entered the sample |
| 3 | evenly spaced snapshots | the whole transient in the first interval |
| 4 | `np.roll` boundaries | **periodic**: inlet fed straight into the outlet |
| 5 | solid neighbour treated as zero | every grain surface a perfect sink |
| 6 | `dt = 1/(4D)` | *exactly* the stability limit — checkerboard |

Measured on the data that had already been generated and delivered: 93% of
adjacent pore cells disagreed in sign, a tenth of the field was above the
inlet concentration (impossible in a purely consumptive system), and solute
was at the outlet at 0.37 while the mid-plane was still exactly 0.

The clip at 2.0 that used to sit at the end of the step is gone. A correct
scheme obeys the maximum principle without help, and `check_physics()` now
asserts on every run that it does.

---

## Chemistry that was not chemistry

The second species was `C[0] * 0.6**j` — a rescaled copy. A "two-chemical"
dataset held one chemical twice, the network's second output head learned
nothing the first had not, and the two per-species errors it reported were the
same number printed twice (Ac 0.1320, A 0.1325).

Now the donor, acceptor, product and biomass are genuinely coupled, with the
acceptor consumed at 0.4 per donor rather than 1 — the stoichiometric ratio
times `Ac0/A0 = 2/5` from `complab_campaign.py`. That factor is what makes them
different equations; with a ratio of exactly 1 and equal inlets they are
identical no matter how the code is written.

The biomass is **attached**: it does not advect. With a mobile biomass and a
zero-concentration inlet, it was pinned to exactly zero on the inlet face —
the one place where donor and acceptor are both at maximum — so the biotic
rate field was identically zero precisely where the chemistry is most active,
and three quarters of the biomass washed out rather than decaying (0.100 →
0.019 with transport, 0.100 → 0.071 without).

---

## Geometry: 55% of a domain was unreachable

`extrude()` seals the transverse faces with wall **after** the percolation
check. On one domain that split the pore space into three clusters, and the
largest (6936 voxels) did not touch the inlet while the one that did held
5736. The geodesic distance left the unreachable 55% at its infinity
sentinel, `1e9` — an ordinary float as far as everything downstream is
concerned, which then went into the trunk as a coordinate and flattened every
real distance to nothing once normalised.

Two changes: `keep_inlet_connected()` keeps what the inlet can reach rather
than what is biggest, and runs *last*, after everything that can change
connectivity; and `assert_finite_distance()` refuses to write a dataset whose
distance field is not finite on every pore voxel.

---

## Seventeen GUI faults

Eleven of these were the same shape: the button sent a flag the script does
not have, omitted one it requires, or packed several values into one token.

- The **Campaign** button could never run. `complab_campaign.py` needs a subcommand
  (`build` / `status` / `retry`) and `build` requires `--complab`; the button
  sent neither.
- `--grid "148 64"`, `--species "Ac A P Bio"` and `--compare "a=x.pt b=y.pt"`
  went as single tokens, so the grid was rejected and the four chemicals
  became one chemical with a long name.
- `--adr-steps 200` was hard-coded, silently overriding the convergence logic.
- **Predict** promised raw `.dat` support in its help text but had no
  `--nx/--ny/--nz` fields.
- The 2D viewer drew `show[:, y, :]`, which on a field with `nz = 1` is a
  one-pixel strip.

The underlying cause was that `Field` guessed how many tokens to send from the
field's *name* (`key.endswith("shape")`). It now carries an explicit `nargs`
that must match the receiving script's, and `test_gui_commands.py` checks
every button against the real `argparse` parsers — not a copy of them, so it
cannot drift. That test also carries its own sensitivity check: it feeds
itself the known-bad commands and fails if it stops catching them. A test that
cannot fail is worse than no test, because it is trusted.

---

## Reporting that was quietly wrong

- **Row labels.** The time-series figure labelled the last species
  `BIOMASS Bio` whatever it was. On an `(Ac, A)` dataset it called the
  acceptor biomass — a caption a reader has no way to check.
- **Species mismatch.** `evaluate.py` labels every number from the *dataset*
  while the model's outputs come from the *checkpoint*, and never compared
  them. A model trained on `(A, Ac)` scored against `(Ac, A)` reported
  `rmse_A` for the head that learned the donor. It now refuses.
- **Empty 3D renders.** `render_3d` trims 3 voxels off each face; on `nz = 1`
  that selects nothing, so every native-2D dataset produced a blank figure at
  full rendering cost. It now says why it is skipping.
- **`t_norm` overwritten.** `build_practice_dataset` replaced the solver's real
  integration times with a straight 0→1 ramp, telling the trunk the snapshots
  were evenly spaced when they were not.
- **`--n-times 2`** never selected the end of the run, because
  `np.geomspace(a, b, num=1)` returns `[a]`.
- **Duplicate snapshots.** Short runs padded the snapshot list past the end,
  so several snapshots shared one `t_norm` and one field.
- **Parameter names.** `build_practice_dataset` still wrote `Pe`, `Da_bio`,
  `Ks_Ac_over_Ac0`; every other writer had moved to lowercase. The reader
  accepted both, so nothing complained — and two files disagreed about what
  their own columns were called.

---

## Honest costs

The toy pack now takes about twenty minutes, not four. The old four minutes
was not real: the periodic-boundary bug filled the domain from the wrong end
almost immediately and the convergence test believed it. An explicit scheme's
step limit and its time to steady state both scale with the grid width
squared, so the step count is set by the grid, not by the computer.

Runs that hit the step cap are recorded in `samples/settled` rather than
passed off as steady. They are valid transients and fine to train on; their
last snapshot is just "as far as we got".

---

## The 2D release was not touched

`2D/` is a third party's published work and is read-only. Every reference to
it in the code is a read, a file-open dialog's starting folder, or a
"show me this folder" menu item. There is no code path that writes there.
