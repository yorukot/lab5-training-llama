"""Offline trainer, HPO protocol and report integration checks."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import load_file
import torch

from exp.campaign import configuration, initialize, maybe_expand, select_finalists, write_config
from exp.common import atomic_json
from exp.optimizer import METHODS
from exp.report import paired_summary
from exp.train import run


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def train(self, name, method, micro_batch=2, stop_after=None, resume=None):
        config = self.root / f"{name}.json"
        atomic_json(config, {"method": method, "micro_batch": micro_batch})
        args = argparse.Namespace(config=str(config), tiny_smoke=True, stop_after=stop_after,
                                  output=str(self.root / name), no_wandb=True, resume=resume,
                                  data_dir="unused", cache_dir="unused")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = run(args)
        return result, load_file(str(self.root / name / "final" / "model.safetensors"))

    def test_token_weighted_accumulation_and_evaluation(self):
        a, wa = self.train("micro2", "adamw", micro_batch=2)
        b, wb = self.train("micro4", "adamw", micro_batch=4)
        self.assertEqual(a["actual_tokens"], b["actual_tokens"])
        self.assertAlmostEqual(a["metrics"]["eval_loss"], b["metrics"]["eval_loss"], places=5)
        for name in wa:
            torch.testing.assert_close(wa[name], wb[name], atol=2e-6, rtol=2e-5)

    def test_full_trainer_resume_is_identical(self):
        full, weights = self.train("full", "normuon")
        stopped, _ = self.train("resumed", "normuon", stop_after=3)
        self.assertEqual(stopped["status"], "interrupted")
        resumed, restored = self.train("resumed", "normuon", resume="auto")
        self.assertEqual(full["actual_tokens"], resumed["actual_tokens"])
        for name in weights:
            torch.testing.assert_close(weights[name], restored[name], atol=0, rtol=0)

    def test_search_coordinates_are_matched(self):
        for trial in range(16):
            reference = configuration("normuon", trial)
            for method in ("hg_normuon_global", "hg_normuon"):
                candidate = configuration(method, trial)
                for key in reference:
                    if key not in ("method", "meta_lr"):
                        self.assertEqual(reference[key], candidate[key])

    def test_boundary_expands_all_methods_equally(self):
        state = initialize(self.root, "/tmp/prepared-data")
        state.update(phase="search", search_count=16)
        for method in METHODS:
            for trial in range(16):
                cfg = configuration(method, trial)
                cfg["matrix_lr"] = 1e-4 if trial == 0 else 1e-3
                rid = write_config(self.root, cfg)
                atomic_json(self.root / "runs" / rid / "result.json",
                            {"status": "completed", "metrics": {"eval_loss": 3 + trial / 100}, "config": cfg})
        maybe_expand(self.root, state)
        self.assertEqual(state["search_count"], 24)
        self.assertEqual(len(state["active_runs"]), 32)
        for method in METHODS:
            self.assertEqual(sum(method == json.loads((self.root / "configs" / f"{rid}.json").read_text())["method"]
                                 for rid in state["active_runs"]), 8)

    def test_finalists_selected_without_confirmation(self):
        state = initialize(self.root, "/tmp/prepared-data")
        state["shortlist"] = {m: [configuration(m, i) for i in range(3)] for m in METHODS}
        for method in METHODS:
            for trial in range(3):
                for seed, phase in ((42, "search"), (43, "selection"), (44, "selection")):
                    cfg = configuration(method, trial) | {"seed": seed, "phase": phase}
                    rid = write_config(self.root, cfg)
                    # Trial 0 has a lucky first seed, but trial 1 is better on average.
                    loss = (2 if seed == 42 else 5) if trial == 0 else 3 if trial == 1 else 4
                    atomic_json(self.root / "runs" / rid / "result.json",
                                {"status": "completed", "metrics": {"eval_loss": loss}})
        select_finalists(self.root, state)
        self.assertTrue(all(cfg["trial"] == 1 for cfg in state["winners"].values()))
        self.assertEqual(len(state["active_runs"]), 20)
        self.assertTrue((self.root / "locked_finalists.json").exists())

    def test_primary_criterion_rejects_lucky_win(self):
        self.assertTrue(paired_summary([2.99] * 5, [3.] * 5)["passes_primary_criterion"])
        self.assertFalse(paired_summary([2.5, 3.01, 3.01, 3.01, 3.01], [3.] * 5)["passes_primary_criterion"])


if __name__ == "__main__":
    unittest.main()
