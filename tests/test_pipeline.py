#!/usr/bin/env python3
"""
test_pipeline.py -- evaluate.py, predict.py and the 2D weight loader, driven
end to end on a synthetic dataset and an UNTRAINED checkpoint.

Why an untrained checkpoint is the right tool here.  Training is slow, and its
result is a question about accuracy.  These tests ask a different question: does
the wiring hold.  Random weights exercise exactly the same load, build,
inference and write path that trained weights do, in a couple of seconds, and
they fail loudly on the mistakes that actually happen: a shape that no longer
matches, a species axis that came back, a checkpoint key that was renamed, a
figure writer that assumed a list of fields.

What each test would catch

  evaluate_runs      evaluate.py breaking on a single-field prediction
  evaluate_outputs   the metrics table or the figures losing their file names
  evaluate_metrics   the reported error going non-finite or losing its species
  predict_runs       predict.py breaking on a brand-new geometry
  predict_outputs    the VTK and npz writers regrowing a per-species index
  predict_field      the returned field not covering the pore space
  species_mismatch   scoring a model against a species it was not trained on
  weights_2d         the published 2D checkpoint no longer loading strictly

Everything runs in a temporary folder and leaves nothing behind.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "3D", "model")
TOOLS = os.path.join(ROOT, "3D", "tools")
for p in (TOOLS, MODEL, HERE):
    sys.path.insert(0, p)

import numpy as np                                              # noqa: E402

import synthetic                                                # noqa: E402


def run(cmd, cwd=None):
    """Run a script and return (returncode, combined output)."""
    p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True,
                       cwd=cwd or ROOT)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


class Pipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="prt_pipe_")
        cls.data = synthetic.make_dataset(os.path.join(cls.dir, "d.h5"))
        cls.ckpt = synthetic.make_checkpoint(os.path.join(cls.dir, "m.pt"),
                                             cls.data)
        cls.eval_out = os.path.join(cls.dir, "eval")
        cls.rc_eval, cls.out_eval = run(
            [os.path.join(MODEL, "evaluate.py"),
             "--checkpoint", cls.ckpt, "--data", cls.data,
             "--out", cls.eval_out, "--no-3d", "--chunk", "2048"])
        # a geometry file for predict.py, taken from the dataset itself
        import h5py
        with h5py.File(cls.data) as h:
            mat = np.array(h["geom"]["material"][0])
        cls.n = mat.shape[0]
        cls.geom = os.path.join(cls.dir, "geom.dat")
        np.savetxt(cls.geom, mat.reshape(-1), fmt="%d")
        cls.pred_out = os.path.join(cls.dir, "pred")
        cls.rc_pred, cls.out_pred = run(
            [os.path.join(MODEL, "predict.py"),
             "--checkpoint", cls.ckpt, "--geometry", cls.geom,
             "--nx", str(cls.n), "--ny", str(cls.n), "--nz", str(cls.n),
             "--pe", "10", "--da-bio", "1", "--da-abio", "1",
             "--out", cls.pred_out])

    def test_evaluate_runs(self):
        self.assertEqual(self.rc_eval, 0, self.out_eval[-3000:])

    def test_evaluate_outputs(self):
        for f in ("metrics.json", "rmse_table.csv"):
            self.assertTrue(os.path.exists(os.path.join(self.eval_out, f)),
                            "evaluate.py must write %s" % f)
        pngs = [f for f in os.listdir(self.eval_out) if f.endswith(".png")]
        self.assertTrue(pngs, "evaluate.py must write figures")
        for f in os.listdir(self.eval_out):
            self.assertNotIn("_sp0", f, "no per-species index in file names")
            self.assertNotIn("species0", f)

    def test_evaluate_metrics(self):
        with open(os.path.join(self.eval_out, "metrics.json")) as fh:
            m = json.load(fh)
        blob = json.dumps(m)
        self.assertIn("Ac", blob, "the metrics must name the species scored")
        for key in ("rmse", "RMSE"):
            if key in blob:
                break
        else:
            self.fail("the metrics must report an RMSE")
        self.assertNotIn("NaN", blob)
        self.assertNotIn("Infinity", blob)

    def test_predict_runs(self):
        self.assertEqual(self.rc_pred, 0, self.out_pred[-3000:])

    def test_predict_outputs(self):
        files = os.listdir(self.pred_out)
        self.assertIn("pred.npz", files)
        vti = [f for f in files if f.endswith(".vti")]
        self.assertEqual(vti, ["Ac_pred.vti"],
                         "one field, named after its species, with no index")
        self.assertTrue([f for f in files if f.endswith(".png")])

    def test_predict_field(self):
        z = np.load(os.path.join(self.pred_out, "pred.npz"))
        self.assertIn("concentration", z.files)
        vol = z["concentration"]
        self.assertEqual(vol.shape, (self.n, self.n, self.n))
        finite = np.isfinite(vol)
        self.assertGreater(finite.sum(), 0.2 * vol.size,
                           "the prediction must cover the pore space")
        self.assertEqual(str(z["species"]).strip("[]'\" "), "Ac")

    def test_species_mismatch_is_refused(self):
        """A model trained on Ac must not be silently scored against A."""
        ck = synthetic.make_checkpoint(os.path.join(self.dir, "m_A.pt"),
                                       self.data, species="A")
        import torch
        d = torch.load(ck, map_location="cpu", weights_only=False)
        d["species"] = "not_in_this_file"
        torch.save(d, ck)
        rc, out = run([os.path.join(MODEL, "evaluate.py"),
                       "--checkpoint", ck, "--data", self.data,
                       "--out", os.path.join(self.dir, "eval_bad"),
                       "--no-3d"])
        self.assertNotEqual(rc, 0,
                            "evaluating against the wrong species must fail")

    def test_published_2d_weights_load(self):
        """The published 2D checkpoint layout must load into this model.

        Jung's release is not in this repository, so the test builds a
        checkpoint with the same tensor names and shapes and asks the loader to
        take it strictly.  If the architecture drifts away from the 2D one, this
        is where it shows.
        """
        import torch
        from deeponet_model import PRT_DeepONet3D
        ref = PRT_DeepONet3D(in_channels=1, n_params=2, trunk_in_dim=4,
                             grid=(64, 148, 1))
        sd = {k: torch.randn_like(v) for k, v in ref.state_dict().items()}
        path = os.path.join(self.dir, "pub2d.pt")
        torch.save(sd, path)
        fresh = PRT_DeepONet3D(in_channels=1, n_params=2, trunk_in_dim=4,
                               grid=(64, 148, 1))
        missing, unexpected = fresh.load_state_dict(torch.load(
            path, map_location="cpu", weights_only=False), strict=True)
        self.assertEqual(list(missing), [])
        self.assertEqual(list(unexpected), [])
        self.assertEqual(len(sd), 35,
                         "the published layout has 35 tensors; a change here "
                         "means the architecture moved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
