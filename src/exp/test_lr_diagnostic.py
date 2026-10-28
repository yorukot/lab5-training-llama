"""Fairness, scheduling, and interpretation checks for the LR diagnostic."""

import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from exp.common import DEFAULT, atomic_json, calibrated_decay, schedule
from exp.lr_diagnostic import POINTS, configurations, initialize, report, submit, worker


class LRDiagnosticTests(unittest.TestCase):
    def test_only_two_lrs_vary_and_methods_get_identical_points(self):
        configs = list(configurations())
        self.assertEqual(len(configs), 24)
        self.assertEqual(len({c["run_id"] for c in configs}), 24)
        for a, b in zip(configs[::2], configs[1::2], strict=True):
            for key in a:
                if key not in ("method", "meta_lr", "run_id"):
                    self.assertEqual(a[key], b[key])
            for key in DEFAULT:
                if key not in ("method", "meta_lr", "matrix_lr", "embed_lr", "phase"):
                    self.assertEqual(a[key], DEFAULT[key])
        self.assertIn((1.25, 1.25), POINTS)  # BU transfer candidate
        self.assertIn((1.5, 1.5), POINTS)  # BV brackets BU in the source study

    def test_decay_rescaling_preserves_every_step_not_just_final_product(self):
        values, _ = schedule(2289, 0.01, 0.8)
        eta = DEFAULT["matrix_lr"] * np.asarray(values)
        wd = calibrated_decay(DEFAULT["matrix_lr"], values, 1.5)
        for factor in (0.5, 0.825, 1.25, 2.0):
            scaled_wd = calibrated_decay(DEFAULT["matrix_lr"] * factor, values, 1.5)
            np.testing.assert_allclose(1 - eta * wd,
                                       1 - factor * eta * scaled_wd,
                                       rtol=0, atol=2e-16)

    def test_fixed_multiplier_equivalence_and_finite_budget_counterexample(self):
        # Exact-real-arithmetic equivalence; BF16 need not be bitwise identical.
        w, u, lr, wd, factor = 0.7, -0.3, 0.02, 0.1, 0.825
        adaptive = (1 - lr * wd) * w - lr * factor * u
        scaled_lr = lr * factor
        fixed_lr = (1 - scaled_lr * wd / factor) * w - scaled_lr * u
        self.assertAlmostEqual(adaptive, fixed_lr, places=15)
        # f(w)=w^2/2: shrinkage can worsen finite-horizon loss despite stability.
        self.assertGreater((1 - 0.1 * factor)**20, (1 - 0.1)**20)

    def test_report_does_not_claim_a_win_and_rejects_partial_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = initialize(root, "/tmp/data", "/tmp/assets")
            rid = state["runs"][0]
            cfg = json.loads((root / "configs" / f"{rid}.json").read_text())
            value = {"status": "completed", "config": cfg, "actual_tokens": 100,
                     "total_tokens": 100, "metrics": {
                         "eval_loss": 3.0, "eval_perplexity": math.exp(3.0)}}
            path = root / "runs" / rid / "result.json"
            atomic_json(path, value)
            result = report(root)
            self.assertFalse(result["complete"])
            self.assertIn("no superiority", result["claim"])
            value["actual_tokens"] = 99
            atomic_json(path, value)
            with self.assertRaisesRegex(ValueError, "Incomplete token"):
                report(root)

    def test_allocation_rejects_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = initialize(root, "/tmp/data", "/tmp/assets")
            state["jobs"] = ["123"]
            atomic_json(root / "study.json", state)
            with patch("exp.lr_diagnostic.subprocess.check_output", return_value="123\n"):
                with self.assertRaisesRegex(RuntimeError, "already queued"):
                    submit(root)

    def test_workers_isolate_gpus_rotate_methods_and_skip_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initialize(root, "/tmp/data", "/tmp/assets")
            assignments = []

            def train(command, **kwargs):
                cfg = json.loads(Path(command[command.index("--config") + 1]).read_text())
                output = Path(command[command.index("--output") + 1])
                assignments.append((cfg["method"], kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
                self.assertNotIn("--tiny-smoke", command)
                atomic_json(output / "result.json", {
                    "status": "completed", "config": cfg, "actual_tokens": 100,
                    "total_tokens": 100, "metrics": {"eval_loss": 3.0,
                                                       "eval_perplexity": math.exp(3.0)},
                })
                return subprocess.CompletedProcess(command, 0)

            with (
                patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2,5", "SLURM_JOB_ID": "999"}),
                patch("exp.lr_diagnostic.subprocess.run", side_effect=train) as invoke,
            ):
                worker(root)
                self.assertEqual(invoke.call_count, 24)
                worker(root)
                self.assertEqual(invoke.call_count, 24)
            for method in ("normuon", "hg_normuon"):
                self.assertEqual({device for m, device in assignments if m == method}, {"2", "5"})
            self.assertTrue(report(root)["complete"])


if __name__ == "__main__":
    unittest.main()
