#!/usr/bin/env python3
"""
evaluate.py — held-out accuracy and the paper's figures.

    # one model
    python evaluate.py --checkpoint ../runs/gdf/best.pt \
                       --data ../dataset/dataset_reader.h5 --out ../figs/gdf

    # the ablation that carries the paper
    python evaluate.py --data ../dataset/dataset_reader.h5 --out ../figs/ablation \
        --compare geodesic=../runs/gdf/best.pt \
                  euclidean=../runs/edt/best.pt \
                  none=../runs/none/best.pt

WHAT IT TESTS ON
    The geometries held out by split_by_geometry, i.e. pore structures the model
    has never seen. Never a sample split: that leaks structure and inflates the
    score.

WHAT IT PRODUCES
    metrics.json         RMSE, R2 and the speedup for the predicted field
    rmse_table.csv       one row per held-out sample, for your own plotting
    fields_*.png         truth / prediction / absolute error, mid-plane slices
    physics_*_2d.png     FLOW, BIOTIC rate and ABIOTIC rate: truth vs prediction
    physics_*_3d.png     the same three fields as 3D half-cut renders
    time_*_2d.png        transient models: transport and both reaction rates
                         marching in time, truth above prediction, one column
                         per snapshot
    time_*_3d.png        the same evolution as 3D half-cut renders
    rmse_vs_params.png   RMSE against the dimensionless groups the dataset has
    ablation.png + .csv  side-by-side bars when --compare is used
"""

import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from dataset_reader import (PRT3DDataset, split_by_geometry,        # noqa
                            indices_for_geometries,
                            scatter_to_volume, dataset_kwargs_from_ckpt,
                            species_of_ckpt)
from deeponet_model import PRT_DeepONet3D                                            # noqa
import make_figures                                                                     # noqa


def trunk_dim_of(ck):
    """Trunk width, from the checkpoint. Checkpoints written before the field
    existed always carried t, hence the default."""
    if "trunk_in_dim" in ck:
        return int(ck["trunk_in_dim"])
    return 3 + int(ck.get("with_time", True)) + int(ck["args"]["distance"] != "none")


def switch_label(ck):
    return ck.get("switch_label", dataset_kwargs_from_ckpt(ck)[1]["label"])


def load_model(path, ds, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    ta = ck["args"]
    _, cfg = dataset_kwargs_from_ckpt(ck)
    m = PRT_DeepONet3D(in_channels=ck.get("in_channels", cfg["in_channels"]),
                       n_params=len(ck["param_names"]),
                       trunk_in_dim=trunk_dim_of(ck),
                       grid=ds.shape).to(device).eval()
    m.load_state_dict(ck["model"])
    return m, ta


def run_one(ckpt, args, device, tag, save_fields=0):
    """Full-grid prediction on every held-out sample. Returns per-sample RMSE."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    # rebuild EXACTLY the dataset configuration this checkpoint was trained with,
    # switches included; old checkpoints have no switch keys and default to off
    kw, cfg = dataset_kwargs_from_ckpt(ck)

    # AUDIT TRAIN-04. Use the geometries the checkpoint says were held out,
    # rather than recomputing a split here from a fraction and a seed. If those
    # two ever disagree, this scores the model on rocks it was trained on and
    # nothing says so. Older checkpoints carry no split, and then the old
    # behaviour is used and the output says which happened.
    split = ck.get("split_geometries")
    if split and split.get("test"):
        te = indices_for_geometries(args.data, split["test"])
        print("test set    : the %d geometries recorded in the checkpoint"
              % len(split["test"]))
        if ck.get("split_kind") == "two-way":
            print("              two-way split: these rocks also chose the "
                  "checkpoint, so this score is optimistic")
    else:
        _, te = split_by_geometry(args.data, frac=args.test_frac, seed=args.seed)
        print("test set    : recomputed here, because this checkpoint records "
              "no split.")
        print("              It is only the same set if --test-frac and "
              "--seed match the training run.")

    # AUDIT DATA-05 and DATA-04. Score under the scalings the network was
    # trained with, not under scalings refitted on the test rows.
    vs = ck.get("vel_stats")
    if vs:
        kw["vel_stats"] = (np.asarray(vs[0], np.float32),
                           np.asarray(vs[1], np.float32))
    if ck.get("target_scale"):
        kw["target_scale"] = np.asarray(ck["target_scale"], np.float32)

    ds = PRT3DDataset(args.data, indices=te, full_grid=True, **kw)
    print("  switches : %s" % cfg["label"])

    # What the checkpoint says it is for. Neither of these changes a tensor
    # shape, so neither is caught by the trunk-width check below: a model
    # trained on one chemistry scores perfectly happily against another, and
    # the number means nothing. Print them, and say plainly when the
    # checkpoint does not record them.
    if ck.get("reaction"):
        print("  reaction : %s, distance convention %r"
              % (ck["reaction"], ck.get("distance_convention", "not recorded")))
        if ck.get("reaction_species") and ds.species:
            missing = [s for s in ds.species if s not in ck["reaction_species"]]
            if missing:
                print("             NOTE: this dataset holds %s, which %s does "
                      "not list. The score below is against fields the model "
                      "was not trained for."
                      % (", ".join(missing), ck["reaction"]))
    else:
        print("  reaction : not recorded in this checkpoint, so nothing here "
              "can check that the dataset is the same chemistry")
    if ds.trunk_dim != trunk_dim_of(ck):
        sys.exit("checkpoint expects a %d-input trunk but this dataset gives %d. "
                 "The checkpoint and the dataset disagree about the time axis: "
                 "a model trained on a transient dataset cannot be evaluated on a "
                 "steady one, or the other way round."
                 % (trunk_dim_of(ck), ds.trunk_dim))
    if ds.T > 1 and not ds.with_time:
        print("WARNING: this checkpoint has no time input but the dataset has %d "
              "snapshots per run. Every snapshot is being scored against the same "
              "time-blind prediction, so the RMSE will look bad for a reason that "
              "is not the model's accuracy. Train with a transient dataset, or "
              "collect_complab_output with --mode steady." % ds.T)
    # The checkpoint's output heads and the dataset's column labels are two
    # different lists, and every number in rmse_table.csv, metrics.json and
    # the figures is labelled from the DATASET. If they disagree, the reported
    # rmse_A belongs to the head that learned some other chemical -- a wrong
    # answer that looks entirely reasonable. train.py guards this for the
    # transfer set; it has to be guarded here too.
    # species_of_ckpt also accepts an older checkpoint that stored a LIST here,
    # by taking its first entry.
    ck_species = species_of_ckpt(ck)
    if ck_species and ds.target_species != ck_species:
        sys.exit("the checkpoint was trained on chemical %r but this dataset "
                 "run selected %r. Every error in the table is labelled from "
                 "the dataset while the model's output comes from the "
                 "checkpoint, so the two would be silently mismatched. "
                 "Evaluate against the species it was trained on."
                 % (ck_species, ds.target_species))

    model, _ = load_model(ckpt, ds, device)
    species = ds.target_species

    rows, t_inf = [], 0.0
    for k in range(len(ds)):
        b1, b2, tk, y = ds[k]
        b1 = b1[None].to(device); b2 = b2[None].to(device)
        t0 = time.time()
        with torch.no_grad():
            pred = torch.cat([model(b1, b2, tk[None, s:s + args.chunk].to(device))[0]
                              for s in range(0, tk.shape[0], args.chunk)], 0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_inf += time.time() - t0
        p = pred.cpu().numpy(); t = y.numpy()          # both (P,)
        se = float(((p - t) ** 2).mean())
        var = float(t.var())
        # R-SQUARED IS UNDEFINED ON A CONSTANT FIELD, and snapshot 0 IS a
        # constant field: the sample before anything has entered. R2 divides by
        # the variance of the truth, so a floor of 1e-30 does not rescue it, it
        # manufactures a number of order 1e26 and one such row destroys the
        # average of all the others. Reported R2 was -4.2e26, which reads as a
        # catastrophic model and is in fact a division by nothing.
        #
        # NaN is the honest value here, and it is excluded from the average
        # rather than propagated.
        r2 = float(1 - se / var) if var > 1e-12 else float("nan")
        s, ti = ds._decode(k)          # sample index and snapshot index
        rows.append(dict(sample=s, t_index=ti,
                         t_norm=float(ds.t_norm[s, ti]), tag=tag,
                         species=species,
                         rmse=float(np.sqrt(se)), r2=r2,
                         rmse_mean=float(np.sqrt(se)),
                         **{n: float(v) for n, v in zip(ds.param_names, ds.params[s])}))
        if save_fields and k < save_fields:
            # take the voxel indices from the pore list, NOT from trunk columns
            # 0-2: with --dim-free the trunk is (t, dwall, tau) and carries no
            # Cartesian coordinates at all
            pts = ds.pore_idx[int(ds.geom_index[s])].astype(int)
            truth_v = scatter_to_volume(t, pts, ds.shape)
            pred_v = scatter_to_volume(p, pts, ds.shape)
            _field_fig(truth_v, pred_v, species, rows[-1],
                       os.path.join(args.out, "fields_%s_%02d.png" % (tag, k)))
            g = int(ds.geom_index[s])
            material = ds.h["geom/material"][g]
            velfield = ds.h["samples/velocity"][s] if ds.has_velocity else None
            _physics_fig(material, velfield, truth_v, pred_v, species,
                         ds.params[s], ds.param_names, rows[-1],
                         os.path.join(args.out, "physics_%s_%02d" % (tag, k)),
                         do_3d=not args.no_3d)
    if save_fields and ds.T > 1:
        time_fig(ds, model, device, 0, args, tag)
    return rows, t_inf, species


def time_fig(ds, model, device, k0, args, tag):
    """How the fields EVOLVE, truth against prediction, one column per snapshot.

    This is the figure that answers 'does the operator carry time'. A model that
    has only memorised the steady state produces identical columns; a model that
    has learned the transient reproduces the front advancing and the reaction
    front trailing behind it."""
    if ds.T < 2:
        return
    nT = ds.T
    truth, pred = [], []
    for ti in range(nT):
        k = k0 * nT + ti                       # same sample, successive snapshots
        b1, b2, tk, y = ds[k]
        with torch.no_grad():
            p = torch.cat([model(b1[None].to(device), b2[None].to(device),
                                 tk[None, c:c + args.chunk].to(device))[0]
                           for c in range(0, tk.shape[0], args.chunk)], 0).cpu().numpy()
        # NOT decoded from trunk columns 0-2: under --dim-free those are
        # (t, dwall, tau) and the scatter would place every value at a
        # meaningless voxel, giving a figure that looks plausible and is noise.
        pts = ds.pore_idx[int(ds.geom_index[int(ds.indices[k0])])].astype(int)
        truth.append(scatter_to_volume(y.numpy(), pts, ds.shape))
        pred.append(scatter_to_volume(p, pts, ds.shape))

    s = int(ds.indices[k0])
    params, pnames, species = ds.params[s], ds.param_names, ds.target_species
    material = ds.h["geom/material"][int(ds.geom_index[s])]
    ts = ds.t_norm[s]
    rb_t, ra_t = zip(*[make_figures.reaction_rates(v, species, params, pnames) for v in truth])
    rb_p, ra_p = zip(*[make_figures.reaction_rates(v, species, params, pnames) for v in pred])
    # Label the row with the species that is actually in it, and with the role
    # that name carries when it is one this project knows. Labelling by
    # position was how the acceptor of an (Ac, A) dataset came to be captioned
    # "BIOMASS Bio" -- not merely untidy but wrong, and wrong in a way a reader
    # has no way to detect from the figure.
    ROLE = {"Ac": "DONOR", "A": "ACCEPTOR", "P": "PRODUCT", "Bio": "BIOMASS"}
    CMAP = {"Ac": "viridis", "A": "cividis", "P": "magma", "Bio": "BuPu"}

    head = "%s  %s" % (ROLE.get(species, "CHEMICAL"), species)
    cm = CMAP.get(species, "viridis")
    rows = [(head + "  truth", list(truth), cm),
            (head + "  pred", list(pred), cm)]
    rows += [("BIOTIC  R_bio  truth", list(rb_t), "YlGn"),
             ("BIOTIC  R_bio  pred", list(rb_p), "YlGn"),
             ("ABIOTIC  R_abio  truth", list(ra_t), "OrRd"),
             ("ABIOTIC  R_abio  pred", list(ra_p), "OrRd")]
    rows = [r for r in rows if all(a is not None for a in r[1])]

    title = ("time evolution, held-out geometry   %s   %s"
             % (species, "  ".join(
                 "%s=%.3g" % (n, float(v)) for n, v in zip(pnames, params)
                 if str(n).lower() == "pe" or str(n).lower().startswith("da"))))
    _time_grid(material, rows, ts, os.path.join(args.out, "time_%s_2d.png" % tag), title)

    if not args.no_3d:
        pick = [0, nT // 2, nT - 1]
        # The biotic rate needs Ac, A AND Bio at once, which one model does not
        # predict, so every rate panel is None, the renderer drops them all and
        # writes NOTHING -- ticking "3D renders" then produced no 3D time
        # figure and no explanation. Fall back to the predicted field itself,
        # which is what a reader wants to see in 3D anyway, and say so.
        have_rate = all(a is not None for a in rb_t) and \
            all(a is not None for a in rb_p)
        if have_rate:
            src_t, src_p, what, cmap = rb_t, rb_p, "biotic rate", "YlGn"
        else:
            src_t, src_p = list(truth), list(pred)
            what, cmap = species, "viridis"
            print("   (3D time figure shows %s: the biotic rate needs Ac, A "
                  "and Bio together, and one model predicts one field.)"
                  % what)
        panels = []
        for j in pick:
            panels.append(("%s truth  t=%.2f" % (what, ts[j]), src_t[j], cmap))
            panels.append(("%s pred  t=%.2f" % (what, ts[j]), src_p[j], cmap))
        lim = make_figures.shared_limits(*[p[1] for p in panels])
        panels = [(n, a, c, lim) for n, a, c in panels]
        make_figures.render_3d(material, panels,
                      os.path.join(args.out, "time_%s_3d.png" % tag),
                      title + "   (half-cut 3D, %s)" % what)


def _time_grid(material, rows, ts, path, title):
    """rows of (label, [volume per time], cmap); columns are time."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = material.shape[2] // 2
    solid = (material[:, :, z] != make_figures.PORE).T
    nT = len(ts)
    fig, ax = plt.subplots(len(rows), nT, figsize=(2.75 * nT, 2.75 * len(rows)),
                           squeeze=False)
    # rows come in truth/prediction pairs; a pair must share one scale, otherwise
    # a prediction that is uniformly too low still looks identical to the truth
    pair = [make_figures.shared_limits(*(rows[2 * k][1] + rows[2 * k + 1][1]))
            for k in range(len(rows) // 2)]
    for i, (nm, vols, cm) in enumerate(rows):
        lim = pair[i // 2]
        for j in range(nT):
            img = np.where(solid, np.nan, vols[j][:, :, z].T)
            kw = dict(origin="lower", cmap=cm, interpolation="nearest")
            if lim: kw.update(vmin=lim[0], vmax=lim[1])
            im = ax[i, j].imshow(img, **kw)
            ax[i, j].imshow(np.where(solid, 1.0, np.nan), origin="lower", cmap="Greys",
                            vmin=0, vmax=1.6, interpolation="nearest")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title("t = %.2f" % ts[j], fontsize=12)
            if j == 0:
                ax[i, j].set_ylabel(nm, fontsize=9)
        fig.colorbar(im, ax=ax[i, :].tolist(), fraction=.02, pad=.01)
    fig.suptitle(title + "   (mid-plane slice; truth and prediction share a colour scale)",
                 fontsize=13, y=0.995)
    fig.savefig(path, dpi=112, bbox_inches="tight")
    plt.close(fig)


def _physics_fig(material, velfield, truth, pred, species, params, param_names,
                 meta, stem, do_3d=True):
    """FLOW, BIOTIC rate, ABIOTIC rate — truth against prediction, 2D and 3D.

    The concentrations are what the network emits; the rate fields are what a
    reactive-transport reader wants to look at. Both are derived here with the
    same rate laws the solver used, so a rate error is a real error and not a
    plotting artefact."""
    umag = make_figures.velocity_magnitude(velfield)
    Rb_t, Ra_t = make_figures.reaction_rates(truth, species, params, param_names)
    Rb_p, Ra_p = make_figures.reaction_rates(pred, species, params, param_names)
    d = lambda a, b: None if (a is None or b is None) else np.abs(a - b)

    # truth and prediction share one colour scale per quantity, so the eye is
    # comparing values and not two independently auto-scaled images
    lb = make_figures.shared_limits(Rb_t, Rb_p)
    la = make_figures.shared_limits(Ra_t, Ra_p)

    title = ("held-out sample %d   %s   t=%.2f   %s   RMSE=%.4f"
             % (meta["sample"], species, meta.get("t_norm", 1.0),
                _conditions(meta, param_names), meta["rmse_mean"]))
    # With neither rate available every panel below except FLOW is None, the
    # renderer drops them, and a figure billed as "the physics" becomes one
    # picture of the velocity field -- at the full cost of a 3D render. Show
    # the predicted field instead, which is always there, and say what happened.
    bare = Rb_t is None and Ra_t is None
    if bare:
        _physics_fig._said = getattr(_physics_fig, "_said", False)
        if not _physics_fig._said:
            print("   (no rate fields: the biotic rate needs Ac, A and Bio "
                  "together and the abiotic rate needs P, while one model "
                  "predicts one field, here %s. The physics figures show that "
                  "field instead.)" % species)
            _physics_fig._said = True

    panels_2d = [("FLOW  |u|", umag, "cividis", None),
                 ("BIOTIC  R_bio  truth", Rb_t, "YlGn", lb),
                 ("BIOTIC  R_bio  predicted", Rb_p, "YlGn", lb),
                 ("BIOTIC  |error|", d(Rb_t, Rb_p), "magma", None),
                 ("ABIOTIC  R_abio  truth", Ra_t, "OrRd", la),
                 ("ABIOTIC  R_abio  predicted", Ra_p, "OrRd", la),
                 ("ABIOTIC  |error|", d(Ra_t, Ra_p), "magma", None)]
    panels_3d = [("FLOW  |u|", umag, "cividis", None),
                 ("BIOTIC  R_bio  truth", Rb_t, "YlGn", lb),
                 ("BIOTIC  R_bio  predicted", Rb_p, "YlGn", lb),
                 ("ABIOTIC  R_abio  truth", Ra_t, "OrRd", la),
                 ("ABIOTIC  R_abio  predicted", Ra_p, "OrRd", la)]
    if bare:
        # the field itself, truth against prediction, on a shared scale
        ls = make_figures.shared_limits(truth, pred)
        panels_2d.append((species + "  truth", truth, "viridis", ls))
        panels_2d.append((species + "  predicted", pred, "viridis", ls))
        panels_2d.append((species + "  |error|", d(truth, pred), "magma", None))
        panels_3d.append((species + "  truth", truth, "viridis", ls))
        panels_3d.append((species + "  predicted", pred, "viridis", ls))
    make_figures.render_2d(material, panels_2d, stem + "_2d.png", title + "   (mid-plane slice)")
    if do_3d:
        make_figures.render_3d(material, panels_3d, stem + "_3d.png",
                      title + "   (half-cut 3D)")


def _p(meta, name, default=0.0):
    """Case-insensitive parameter lookup: collect_complab_output.py and hand-built files
    disagree about whether the groups are called 'Pe' or 'pe'."""
    for k, v in meta.items():
        if isinstance(k, str) and k.lower() == name.lower():
            return v
    return default


def _conditions(meta, param_names):
    """The run's conditions as a caption, naming the columns the DATASET has.

    A two-column dataset records (pe, da) and a six-column one records
    (pe, da_bio, da_abio, ...). Hard-coding Da_bio and Da_abio printed
    'Da_bio=0  Da_abio=0' on every figure from a two-column file.
    """
    out = []
    for n in param_names:
        n = str(n)
        if n.lower() == "pe" or n.lower().startswith("da"):
            out.append("%s=%.3g" % (n, float(_p(meta, n))))
    return "  ".join(out)


def _field_fig(truth, pred, species, meta, path):
    """Truth, prediction and absolute error of the ONE predicted field."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nz = truth.shape[2]; z = nz // 2
    fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.4))
    t, p = truth[:, :, z].T, pred[:, :, z].T
    vmin, vmax = np.nanmin(t), np.nanmax(t)
    for c, (img, lab, cm) in enumerate([(t, "ground truth", "viridis"),
                                        (p, "prediction", "viridis"),
                                        (np.abs(p - t), "|error|", "magma")]):
        kw = dict(origin="lower", interpolation="nearest", cmap=cm)
        if c < 2: kw.update(vmin=vmin, vmax=vmax)
        im = ax[c].imshow(img, **kw)
        fig.colorbar(im, ax=ax[c], fraction=.046)
        ax[c].set_title("%s, %s" % (species, lab), fontsize=10)
        ax[c].set_xticks([]); ax[c].set_yticks([])
    fig.suptitle("held-out sample %d   t=%.2f   Pe=%.3g  Da_bio=%.3g  Da_abio=%.3g"
                 "   mean RMSE=%.4f"
                 % (meta["sample"], meta.get("t_norm", 1.0), _p(meta, "pe"),
                    _p(meta, "da_bio"), _p(meta, "da_abio"),
                    meta["rmse_mean"]), fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--compare", nargs="*", default=[], metavar="LABEL=CKPT")
    ap.add_argument("--out", default="./figs")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--save-fields", type=int, default=4)
    ap.add_argument("--no-3d", action="store_true",
                    help="skip the 3D renders (they are the slow part)")
    ap.add_argument("--sim-seconds", type=float, default=None,
                    help="wall clock of ONE simulation of the kind this model "
                         "replaces, for the speedup number. Defaults to 1800 s "
                         "for a 3D dataset and 30 s for a 2D one. MEASURE YOUR "
                         "OWN and pass it before quoting the speedup anywhere.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # THE SPEEDUP IS ONLY AS HONEST AS ITS BASELINE.
    #
    # The default used to be a flat 1800 s -- the wall clock of a 3D CompLaB
    # run -- and it was applied to 2D datasets too. That divided a 2D
    # prediction by a 3D simulation and reported speedups of 100,000x, a
    # number that compares two different things and would not survive a
    # question at a talk. A 2D run on a 148 x 64 grid takes seconds, not half
    # an hour, so the 2D default is 30 s and the printout now always says what
    # it assumed.
    import h5py as _h5
    with _h5.File(args.data, "r") as _f:
        _nz = int(tuple(_f.attrs["shape"])[2])
    _auto = args.sim_seconds is None
    if _auto:
        args.sim_seconds = 1800.0 if _nz > 1 else 30.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = ([("model", args.checkpoint)] if args.checkpoint else []) + \
             [tuple(c.split("=", 1)) for c in args.compare]
    if not models:
        sys.exit("give --checkpoint or --compare LABEL=path ...")

    all_rows, summary = [], {}
    for tag, ck in models:
        rows, t_inf, species = run_one(ck, args, device, tag,
                                       args.save_fields if len(models) == 1 else 0)
        all_rows += rows
        arr = np.array([r["rmse"] for r in rows])          # one per snapshot
        r2 = np.array([r["r2"] for r in rows])
        n_undef = int(np.isnan(r2).sum())
        if n_undef:
            print("   (R2 undefined on %d of %d snapshots -- the truth is "
                  "constant there, usually t = 0 before anything has entered. "
                  "Those are left out of the R2 average, not counted as zero.)"
                  % (n_undef, len(rows)))

        # TWO DIFFERENT AVERAGES, BOTH REPORTED, BECAUSE THEY DIFFER.
        #
        # rmse_mean is the mean of the per-snapshot RMSEs. train.py instead
        # pools every point of every snapshot and takes one square root at the
        # end. Square root is concave, so the mean of the roots is always the
        # SMALLER of the two -- measured here, 0.0476 against 0.0538 for the
        # same weights on the same data. Neither is wrong, but quoting one
        # number in training and a different one in evaluation, both called
        # "held-out RMSE", invites exactly the confusion it caused.
        pooled = float(np.sqrt((arr ** 2).mean()))
        with np.errstate(invalid="ignore"):
            r2_mean = float(np.nanmean(r2)) if np.isfinite(r2).any() \
                else float("nan")
        summary[tag] = dict(checkpoint=ck, n_samples=len(rows),
                            species=species,
                            rmse=round(float(arr.mean()), 5),
                            rmse_pooled=round(pooled, 5),
                            r2=round(r2_mean, 4) if r2_mean == r2_mean else None,
                            r2_snapshots_undefined=n_undef,
                            rmse_mean=float(arr.mean()),
                            rmse_pooled_mean=pooled,
                            inference_seconds_total=t_inf,
                            inference_seconds_per_sample=t_inf / max(len(rows), 1),
                            speedup_vs_simulation=args.sim_seconds / (t_inf / max(len(rows), 1)))
        print("%-12s %-6s mean RMSE %.4f   %.3f s/sample   speedup %.0fx"
              % (tag, species, arr.mean(),
                 t_inf / max(len(rows), 1),
                 args.sim_seconds / (t_inf / max(len(rows), 1))))
        print("             (that is the mean of the per-snapshot RMSEs. "
              "Pooled over every point at once it is %.4f, which is the "
              "number train.py prints.)" % pooled)
        if r2_mean == r2_mean:
            print("             R2 %.4f" % r2_mean)
        print("             speedup is against an assumed %.0f s per %s "
              "simulation%s. Measure your own and pass --sim-seconds before "
              "quoting it." % (args.sim_seconds, "3D" if _nz > 1 else "2D",
                               " (the default)" if _auto else ""))

    import csv
    keys = sorted({k for r in all_rows for k in r})
    with open(os.path.join(args.out, "rmse_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_rows)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tag0 = models[0][0]
        r0 = [r for r in all_rows if r["tag"] == tag0]
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
        # the dimensionless columns this dataset actually has: (pe, da) for
        # the default layout, (pe, da_bio, da_abio) for the full one
        pcols = [c for c in keys
                 if c.lower() == "pe" or c.lower().startswith("da")][:3]
        for i in range(3):
            if i >= len(pcols):
                ax[i].axis("off")       # a blank frame explains nothing
                continue
            pc = pcols[i]
            ax[i].scatter([r[pc] for r in r0], [r["rmse_mean"] for r in r0], s=26, alpha=.75)
            ax[i].set_xscale("log"); ax[i].set_xlabel(pc); ax[i].set_ylabel("mean RMSE")
            ax[i].axhline(0.04, ls="--", c="crimson", lw=1, label="2D paper bar 0.04")
            ax[i].legend(fontsize=8)
        fig.suptitle("held-out RMSE against the dimensionless groups (%s)" % tag0)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "rmse_vs_params.png"), dpi=115)
        plt.close(fig)

        if len(models) > 1:
            labs = [t for t, _ in models]
            means = [summary[t]["rmse_mean"] for t in labs]
            fig, a = plt.subplots(figsize=(1.7 * len(labs) + 2.5, 4))
            b = a.bar(labs, means, color=["#1b7a4b", "#1f4e79", "#8a8f98", "#c2410c"][:len(labs)])
            a.axhline(0.04, ls="--", c="crimson", lw=1)
            a.set_ylabel("mean held-out RMSE"); a.set_title("Ablation: what the trunk distance buys")
            for r, v in zip(b, means):
                a.text(r.get_x() + r.get_width() / 2, v, " %.4f" % v, ha="center", va="bottom", fontsize=9)
            fig.tight_layout(); fig.savefig(os.path.join(args.out, "ablation.png"), dpi=115)
            plt.close(fig)
    except Exception as e:
        print("  (figures skipped: %s)" % e)

    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
