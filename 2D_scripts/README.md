# The published 2D model, as scripts

`2D/` holds the Kim and Jung release exactly as published, and its models live
in Jupyter notebooks. This folder is the same three models as plain Python, so
the 2D side can be run, edited, imported and put under test the way the 3D side
already is. Nothing in `2D/` is touched.

    prt2d_model.py             the architecture, the geodesic distance, the inputs
    prt2d_predict.py           a command line predictor
    test_against_notebooks.py  proof that this is a port and not a rewrite

## Run it

    cd 2D_scripts
    python prt2d_predict.py --reaction monod

That loads the released weights and domain, predicts at the published
conditions and times, and writes `prediction.npz` and `prediction.png`.

    python prt2d_predict.py --reaction monod --pe 10 --da 0.5 --times 0.25 0.5 1.0
    python prt2d_predict.py --reaction irreversible_sorption
    python prt2d_predict.py --reaction reversible_sorption --domain ../2D/Domains/domain_7.dat

Or import it:

    import prt2d_model as M
    m_bin = M.read_domain("../2D/geometries/Domain_Monod.npz")
    model, missing, unexpected = M.load_model("monod", "../2D/parameters/Monod.pt")
    b1, b2, trunk = M.build_inputs("monod", m_bin, [(10.0, 0.5)], t_norm=1.0)
    field = M.predict(model, b1, b2, trunk)          # (1, 64, 148)

## It is a port, and here is the proof

    python test_against_notebooks.py

This runs each published notebook cell by cell and compares it with these
scripts on the same weights and the same domain, at four points: the geodesic
distance column, the three input tensors, the raw prediction, and the field the
notebook itself produces. Every one must agree to zero difference.

    monod                    all match
    irreversible_sorption    all match
    reversible_sorption      all match

Run it after any change here. A notebook is the reference, not the other way
round.

## The three reactions are not the same network

| reaction | conv blocks | trunk | parameter branch | conditions |
|---|---|---|---|---|
| `monod` | 5 | x, y, t, gdf | Pe, Da | 2 x 2, 21 times |
| `irreversible_sorption` | 5 | x, y, gdf | Pe, Da_A | 3 x 3, steady state |
| `reversible_sorption` | 5 | x, y, t, gdf | Pe, Da_A, Da_D | 2 x 2 x 2, 6 times |

Two traps are worth naming, because both are silent:

- **The geodesic distance is scaled differently.** `monod` and
  `reversible_sorption` scale it over the whole grid, so the solid, set to one
  past the furthest pore and then negated, pins the bottom of the range.
  `irreversible_sorption` scales it over the pore only. Using one rule for all
  three changes the input to a trained network without any error being raised.
- **`reversible_sorption`'s class default says four convolution blocks, and it
  is built with five.** Four gives a first fully connected layer of 4608 and the
  checkpoint wants 2048.

## Raw output against what the figures show

`predict()` returns the raw field. Every notebook clips to 0 and 1 and sets the
solid to zero before plotting, which is presentation rather than model output,
so it is kept separate as `as_displayed()`. The `.npz` carries both:
`prediction` is raw, `prediction_display` is what the published figures show.

## What this does not do

It does not train. The three training notebooks at the top of `2D/` read their
data from the original author's machine, and those datasets were never
published, so there is nothing to train against here. The release ships
geometries, weights and notebooks but no concentration fields.
