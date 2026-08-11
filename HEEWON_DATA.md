# What to ask Heewon for

## The short version

The published release contains **geometries, trained weights and notebooks — no
simulation output**. We checked: `2D/Domains/` holds 3,000 pore masks,
`2D/geometries/` holds 3 example domains, `2D/parameters/` holds 3 trained
`.pt` files. There are no concentration fields and no flow fields anywhere.

So there are three separate things we might want from him, and they are worth
asking for separately because they have very different costs to him.

| | What it is | Do we have it? | What it unlocks |
|---|---|---|---|
| 1 | his **geometries** | **yes**, 3,000 of them, in the release | training on realistic pore structures, today |
| 2 | his **trained weights** | **yes**, in the release | a warm start: his geometry encoder drops into ours unchanged |
| 3 | his **simulation output** | **no** | training on *his physics*, and a real comparison against his results |

Items 1 and 2 work now. Item 3 is the ask.

---

## What we already verified about his trained weights

Running `3D/tools/load_pretrained_2d_weights.py` on `2D/parameters/Monod.pt`:

```
TRANSFERRED                       34 of 35 tensors
   branch1    all      Conv2d 1->16->32->64->128->256, fc 2048 -> 128
   branch2    all      Linear 2 -> 128 -> 128 -> 128    (2 inputs: Pe, Da)
   trunk      all      Linear 4 -> 128, six 128 -> 128, 128 -> 128
FRESHLY INITIALISED    head.weight, head.bias, bias
forward pass           output (1, 256, 1), all finite: True
```

His trunk takes **4 inputs**, which is `(x, y, t, GDF)` — exactly what our 2D
mode builds. So his geometry-sensing network transfers with no surgery at all.
That is a direct answer to Christof's todo item *"can we just use his GDF
weights directly, so that we can focus on the reactions only?"* — measurably,
yes.

Two caveats. The `fc` layer's 2048 inputs are `256 × (148/32) × (64/32)`, so it
is tied to the 148 × 64 grid; a different grid and that layer will not load.
And his network predicts **one** species with **two** parameters, so with our
six-parameter datasets the first layer of the parameter branch is skipped
(reported explicitly, not silently reshaped).

---

## The request to send him

> Hi Heewon,
>
> We are building a 3D version of PRT-DeepONet on CompLaB3D, and we would like
> to put your 2D results and ours into one database so the two can be compared
> directly. Your published domains and trained parameters have been extremely
> useful already — his branch CNN and trunk load straight into our
> implementation.
>
> What we do not have is the underlying **simulation output**. Could you share
> the fields your networks were trained on?
>
> For each simulation run, what we need is:
>
> - **the pore geometry** as an array, with a note on which value means pore
> - **the concentration field**, ideally as a time series rather than a single
>   steady state, shaped `(n_times, n_species, nx, ny)` or anything close to it
> - **the Peclet and Damkohler numbers** for that run, and the definitions you
>   used for them, which is the part we most need in writing
>
> Very useful if it exists, optional if not:
>
> - **the velocity field**, `(2, nx, ny)` or `(nx, ny, 2)`
> - **the reaction rate field**, if it was written out rather than derived
> - the geodesic distance field, though we can recompute that
>
> Format does not matter much — one `.npz` per run with whatever key names you
> already use is ideal, and we have a reader that auto-detects the layout.
>
> **Could you send ONE example run first?** We will confirm we can read it and
> come back within a day, before you spend time exporting the whole set.
>
> On the Damkohler definition specifically: you quote Da = 74, 296 and 740,
> while our grid is 0.1, 1 and 10. That is about three orders of magnitude, so
> we are certainly not normalising the same way. Knowing which length, which
> velocity and which rate constant you divide by would let us put both sets of
> results on the same axis. That one paragraph may be the most valuable thing in
> this email.

---

## When the data arrives

**Always dry-run the first file.** It reports what it found and writes nothing:

```bash
cd 3D/tools
python import_2d_simulations.py --src /path/to/one_run.npz --dry-run
```

You get back something like:

```
run_00.npz
   arrays present : domain, C, vel, Pe, Da
   geometry       : domain (148, 64)   porosity 0.685
   concentration  : C -> (T=6, C=1, 148, 64)
   velocity       : vel (2, 148, 64)
   Pe, Da         : 4.69, 4.18
USABLE: 1 of 1
```

If anything is missing it says which, and what to do about it — `--pe` and
`--da` for conditions carried outside the arrays, `--params-json` for a lookup
table, and a specific warning if no run has a velocity field, because without
one the flow-proxy and dimension-free switches cannot be used at all.

Then ingest the lot and train:

```bash
python import_2d_simulations.py --src /path/to/all_runs --out data/heewon.h5 \
                         --n-species 1 --species C
python ../model/train.py --data data/heewon.h5 --out runs/heewon
```

---

## Two things the reader watches for

**All runs on the same structure.** If every run uses one geometry, a split by
structure is impossible and any score is optimistic. The reader counts distinct
geometries and warns.

**One snapshot per run.** That is a steady-state dataset; the trunk's time
input is dropped automatically. Fine, but say so when reporting results, and
ask for a time series if the transient behaviour is what you care about.
