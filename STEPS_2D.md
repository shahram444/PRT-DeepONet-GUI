# Training in 2D, from the beginning

Every step below has a button in the GUI. The command line is given too, so you
can see what the button does and paste it into a terminal if you prefer.

Start the GUI by double-clicking `gui/PRT-DeepONet.bat`, then set **Working in:
2D** at the top left.

---

## Step 0 — install the packages, once

**GUI:** Set up → Install the Python packages → torch build `cpu` → Run.

Takes a few minutes. It installs into the Python shown in the bottom-left
corner, which is the same one the analysis scripts use. That distinction is what
usually goes wrong: `pip install` on its own lands in the system Python while
the project's environment stays empty.

Confirm with **Tools → Check this computer can run everything**.

Then run **Set up → Check everything at once**, or from a terminal:

```bash
python check_everything.py
```

Six checks, under a minute. It is the fastest way to tell a broken
installation from a broken idea, and it is worth running again after moving
the folder or any time something behaves oddly. What it covers is listed at
the top of that file; `FIXES.md` explains why each check exists.

---

## Step 0b — look at the data before you trust it

This is the habit worth forming, and it is cheap.

Every dataset this project writes is checked as it is built: no concentration
may exceed the inlet value, adjacent pore voxels may not alternate in sign,
nothing may reach the outlet before it has crossed the sample, and no two
"chemicals" may be rescaled copies of each other. If any of those fail, the
build says so loudly and, for the practice and transfer sets, refuses to
write the file at all.

Those four checks exist because a delivered dataset once failed all of them
at once and nothing said a word — the training run completed, reported a mean
RMSE of 0.03, and the number meant nothing. See `FIXES.md`.

So: open the Viewer, step through the snapshots, and satisfy yourself that
the front enters at the left and moves right. Five minutes there is worth
more than any amount of reading.

---

## Step 1 — decide which of the three you actually have

This is the step people skip, and it decides everything downstream.

### Route A — his **geometries**, our chemistry  ← works today

The release ships 3,000 real pore domains. We solve the flow and transport on
them ourselves, with the same rate law our 3D runs use. The advantage is that
2D and 3D results become directly comparable, because the chemistry is the same
on both sides. The limitation is that it is not his physics.

**GUI:** Set up → Make a 2D dataset → *Where the pore structures come from* =
`use the published 2D domains`.

```bash
cd 3D/tools
python build_dataset_2d.py --out ../../dataset/data2d.h5 \
       --jung-dir ../../2D --limit 200 \
       --shape 148 64 --n-sets 6 --n-times 11 --n-species 2
```

About ten seconds per domain, so 200 domains is roughly half an hour.

### Route B — his **simulation output**  ← needs him to send it

Not in the release. See `HEEWON_DATA.md` for the email to send and the exact
list to ask for. When it arrives:

```bash
python import_2d_simulations.py --src /path/to/one_run.npz --dry-run     # always first
python import_2d_simulations.py --src /path/to/all_runs --out ../../dataset/heewon.h5 \
                         --n-species 1 --species C
```

### Route C — our **own** 2D structures

No dependency on anyone. Useful for method questions.

```bash
python build_dataset_2d.py --out ../../dataset/data2d.h5 \
       --n-geom 40 --n-sets 6 --n-times 11 --phi-min 0.55 --phi-max 0.85
```

The porosity range starts at 0.55 rather than 0.20 because a thresholded
Gaussian field stops percolating near 0.59 in 2D and near 0.20 in 3D. Same
morphology, same generator, threshold nearly three times higher. That gap is
also why a geometry encoder trained on 2D cannot simply be reused in 3D.

### Route D — output from **CompLaB2D**

If you run CompLaB itself in 2D, collect_complab_output it exactly as in 3D. A 2D complab_campaign is
simply one whose grid has `nz = 1`, and everything downstream adapts on its own.

**GUI:** Simulate → Collect CompLaB output into a dataset.

```bash
python collect_complab_output.py --complab_campaign /path/to/complab_campaign --geometries /path/to/geometries \
                  --out ../../dataset --mode transient --n-times 11
```

You can rehearse this before any real run finishes:
Set up → Make a practice CompLaB complab_campaign, with grid `96 48 1`.

---

## Step 2 — look at the data before training on it

**GUI:** Look at results → Viewer → Open Dataset (.h5).

Step through the chemicals in space and in time. Then switch *Show* to
**geodesic distance from the inlet** — that field is how the network senses the
pore structure, and five minutes looking at it is worth more than any
explanation of the method.

If the concentration field is empty, check the snapshot spinner: snapshot 0 is
the state before anything has entered the sample.

---

## Step 3 — train

**GUI:** Learn → Train the network.

```bash
python ../model/train.py --data ../../dataset/data2d.h5 --out ../../runs/2d \
       --epochs 300 --batch-size 8 --n-points 8192
```

It will print

```
dimension   : 2D  (grid (148, 64, 1))
switches    : OFF  trunk (x, y, t, gdf)
branch1 ch  : 1  (material)
```

That trunk — `(x, y, t, gdf)` — **is** the published architecture. It is not an
approximation of it. The code notices `nz = 1`, switches the encoder from 3D to
2D convolutions and drops the z column, so the same Train button serves 2D and
3D.

Watch the **Monitor** tab. Two curves: the error on data it is learning from,
and on structures it has never seen. When the second stops falling, it has
learned all it can from this much data. If the first keeps falling while the
second flattens or rises, it has started memorising.

The split is always by **structure**, never by sample. If the same pore
structure appeared in both halves the score would be flattering and meaningless.

### Optional — warm-start from his published weights

His branch CNN and trunk transfer into ours unchanged, so training can start
from a geometry encoder that already works instead of from noise.

```bash
python load_pretrained_2d_weights.py --checkpoint ../../2D/parameters/Monod.pt \
       --n-params 6 --n-species 1 --save ../../runs/warmstart.pt

python ../model/train.py --data <your 2D dataset> --out ../../runs/2d_warm \
       --init-from ../../runs/warmstart.pt --freeze-trunk
```

Two conditions: the dataset must be on the **148 × 64** grid his `fc` layer was
trained on, and it must have the species count you passed. `--freeze-trunk`
keeps his reaction response and retrains only the geometry branch.

Honest note: whether this actually helps is an open question, not a claim. His
trunk was fitted to *his* chemistry. Run it both ways and compare.

---

## Step 4 — check it on structures it has never seen

**GUI:** Learn → Evaluate a trained network.

```bash
python ../model/evaluate.py --checkpoint ../../runs/2d/best.pt \
       --data ../../dataset/data2d.h5 --out ../../figs/2d --no-3d
```

Produces truth against prediction as slices, the reaction-rate fields derived
from both, and — most useful — the error plotted against Peclet and Damkohler,
so you can see *where* the model is weak rather than only how weak it is on
average.

---

## Step 5 — predict on a structure that was never simulated

**GUI:** Predict → Predict on a new geometry.

```bash
python ../model/predict.py --checkpoint ../../runs/2d/best.pt \
       --geometry /path/to/geom.npz --pe 10 --da-bio 1 --da-abio 1 \
       --out ../../prediction/2d
```

Under a second, against minutes for the 2D simulation and hours for the 3D one.
That is the whole point of the exercise.

---

## Step 6 — the switches, in 2D

All three work in 2D exactly as in 3D, and 2D is where you should test them
first, because a 2D run is roughly fifty times cheaper.

```bash
python ../model/train.py --data data2d.h5 --out runs/a --flow-proxy
python ../model/train.py --data data2d.h5 --out runs/ab --flow-proxy --flow-mode both
python ../model/train.py --data data2d.h5 --out runs/c --dim-free
python ../model/train.py --data data2d.h5 --out runs/edt --distance edt
```

- `--flow-proxy` → trunk `(x, y, t, tau)`, branch sees the velocity field
- `--flow-mode both` → trunk `(x, y, t, gdf, tau)`, the safe option
- `--dim-free` → trunk `(t, dwall, tau)` — **three columns in both 2D and 3D**,
  which is what lets the same network transfer between them
- `--distance edt` → the control that shows the geodesic field is what buys the
  accuracy, not just any distance

To run them all and compare on the same held-out structures:

**GUI:** Learn → Run the switch comparison.

```bash
python ../model/run_ablation_sweep.py --data data2d.h5 --out ../../sweep2d
```

It prints mean error per configuration, **split by Peclet band**. That
breakdown is the point: the flow field should win at high Peclet and lose below
Pe = 1, where solute arrives by diffusion and a stagnant velocity field cannot
tell a deep dead-end pore from solid rock.

---

## The whole thing, condensed

```bash
cd 3D/tools

# 1. dataset from his geometries, our chemistry
python build_dataset_2d.py --out ../../dataset/data2d.h5 --jung-dir ../../2D \
       --limit 200 --shape 148 64 --n-sets 6 --n-times 11 --n-species 2

# 2. train
python ../model/train.py --data ../../dataset/data2d.h5 --out ../../runs/2d --epochs 300

# 3. check
python ../model/evaluate.py --checkpoint ../../runs/2d/best.pt \
       --data ../../dataset/data2d.h5 --out ../../figs/2d --no-3d

# 4. predict
python ../model/predict.py --checkpoint ../../runs/2d/best.pt \
       --geometry ../../2D/geometries/Domain_Monod.npz --pe 10 \
       --da-bio 1 --da-abio 1 --out ../../prediction/2d
```

---

## If something goes wrong

The **Problems** tab collects every error with the command that produced it, and
the same command is shown on every action page before you run it. The usual
causes, in order:

- a path that does not exist yet
- a dataset built on a different grid than the checkpoint expects
- `--flow-proxy` or `--dim-free` with no velocity field in the dataset; those
  switches replace the geometry with the flow, so it has to be there
- warm-starting from the published weights with a grid other than 148 × 64
