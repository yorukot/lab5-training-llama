"""Locked confirmation must never turn missing or incompatible evidence into success."""

import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from exp.research_report import (
    CONFIRMATION_SEEDS,
    HOLDOUT_OFFSET,
    HOLDOUT_TOKENS,
    TRAINING_TOKENS,
    main,
    paired_summary,
    protocol_digest,
    read_json,
    render_markdown,
    summarize,
)


def digest(character):
    return character * 64


class ResearchReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shared = {
            "architecture": {"layers": 12, "width": 768},
            "tokenizer": "gpt2",
            "data_sha256": digest("a"),
            "training_config_sha256": digest("b"),
            "dataset_revision": "fixed-dataset-revision",
            "data_order": "deterministic_sorted_training_shards; seed controls initialization",
        }
        self.manifest = {
            "schema_version": 1,
            "locked": True,
            "baseline_method": "muon",
            "candidate_method": "candidate",
            "seeds": list(CONFIRMATION_SEEDS),
            "expected": {
                "training_tokens": TRAINING_TOKENS,
                "shared_identity": shared,
                "method_config_sha256": {"muon": digest("c"), "candidate": digest("d")},
                "method_source_sha256": {
                    "muon": {"gpt2_benchmark.py": digest("4"), "gpt2_evaluate.py": digest("6")},
                    "candidate": {"gpt2_benchmark.py": digest("4"), "candidate.py": digest("5"), "gpt2_evaluate.py": digest("6")},
                },
                "evaluation_settings": {"device": "cuda", "batch_sequences": 64},
                "evaluation_runtime": {"torch": "2.14.0", "cuda": "13.0", "gpu": "NVIDIA H200", "precision": "pinned evaluator precision"},
                "evaluation": {
                    "slice_id": f"start-{HOLDOUT_OFFSET}-count-{HOLDOUT_TOKENS}",
                    "token_offset": HOLDOUT_OFFSET,
                    "token_count": HOLDOUT_TOKENS,
                    "data_sha256": digest("e"),
                },
            },
            "records": [],
            "historical_reference": {"method": "muon", "seed": 42, "ppl": 26.50544},
        }
        for i, seed in enumerate(CONFIRMATION_SEEDS):
            for method in ("muon", "candidate"):
                nll = math.log(27 + i - (1 if method == "candidate" else 0))
                checkpoint_hash = digest(str(i + 1) if method == "muon" else "f")
                common = {
                    "status": "completed",
                    "method": method,
                    "shared_identity": copy.deepcopy(shared),
                    "method_config_sha256": self.manifest["expected"]["method_config_sha256"][method],
                    "method_source_sha256": self.manifest["expected"]["method_source_sha256"][method],
                }
                run = common | {
                    "metadata": {
                        "seed": seed,
                        "smoke_steps": None,
                        "comparison_eligible": True,
                        "protocol_sha256": digest("7"),
                        "dataset_manifest_sha256": digest("a"),
                        "dataset_revision": "fixed-dataset-revision",
                        "source": {
                            "commit": "pinned-source-commit",
                            "files": {"train_gpt_simple.py": {"sha256": digest("0")}},
                        },
                        "time_scope": "runner end-to-end",
                    },
                    "final": {"training_tokens": TRAINING_TOKENS},
                    "checkpoint": {"filename": "checkpoint.pt", "sha256": checkpoint_hash, "format_version": 1},
                    "end_to_end_seconds": 550 + i,
                    "end_to_end_gpu_hours": 1.2,
                    "peak_allocated_bytes_rank0": 123456,
                }
                evaluation = common | {
                    "seed": seed,
                    "training_tokens": TRAINING_TOKENS,
                    "checkpoint_sha256": checkpoint_hash,
                    "source_sha256": digest("0"),
                    "protocol_sha256": digest("7"),
                    "confirmation_protocol_sha256": protocol_digest(self.manifest),
                    "evaluator_sha256": digest("6"),
                    "device": "cuda",
                    "batch_sequences": 64,
                    "evaluation_runtime": self.manifest["expected"]["evaluation_runtime"],
                    "source_commit": "pinned-source-commit",
                    "dataset_manifest_sha256": digest("a"),
                    "dataset_revision": "fixed-dataset-revision",
                    "evaluation": self.manifest["expected"]["evaluation"] | {
                        "nll": nll,
                        "ppl": math.exp(nll),
                        "aggregation": "token_weighted_mean",
                    },
                }
                paths = {"run": f"{method}-{seed}-run.json", "evaluation": f"{method}-{seed}-eval.json"}
                self.manifest["records"].append(paths)
                (self.root / paths["run"]).write_text(json.dumps(run))
                (self.root / paths["evaluation"]).write_text(json.dumps(evaluation))

    def read_record(self, index=0, part="evaluation"):
        path = self.root / self.manifest["records"][index][part]
        return path, json.loads(path.read_text())

    def change_record(self, keys, value, index=0, part="evaluation"):
        path, record = self.read_record(index, part)
        cursor = record
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = value
        path.write_text(json.dumps(record))

    def test_complete_locked_comparison_confirms_absolute_per_seed_ppl_gain(self):
        report = summarize(self.manifest, self.root)
        self.assertEqual(report["status"], "confirmed")
        self.assertEqual(len(report["pairs"]), 5)
        self.assertAlmostEqual(report["confirmation"]["mean_ppl_reduction"], 1.0)
        self.assertTrue(report["confirmation"]["supports_true_reduction_at_least_target"])
        self.assertEqual(report["historical_reference"]["seed"], 42)
        self.assertEqual(len(report["run_diagnostics"]), 10)
        self.assertEqual(report["run_diagnostics"][0]["peak_allocated_bytes_rank0"], 123456)

    def test_point_target_and_lower_confidence_target_are_distinct(self):
        # mean=0.6, lower CI approximately 0.404: meets the registered criterion,
        # but does not support a claim that the true gain itself exceeds 0.5.
        deltas = [0.4, 0.5, 0.6, 0.7, 0.8]
        result = paired_summary([math.log(30)] * 5, [math.log(30 - d) for d in deltas])
        self.assertTrue(result["passes_primary_criterion"])
        self.assertFalse(result["supports_true_reduction_at_least_target"])
        self.assertAlmostEqual(result["mean_ppl_reduction"], 0.6)
        self.assertAlmostEqual(result["ci95_ppl_reduction"][0], 0.403675683852244, places=10)

    def test_statistically_positive_but_too_small_gain_is_inconclusive(self):
        for index in range(1, 10, 2):
            _, baseline = self.read_record(index - 1)
            nll = math.log(baseline["evaluation"]["ppl"] - 0.2)
            self.change_record(["evaluation", "nll"], nll, index)
            self.change_record(["evaluation", "ppl"], math.exp(nll), index)
        report = summarize(self.manifest, self.root)
        self.assertEqual(report["status"], "inconclusive")
        self.assertFalse(report["confirmation"]["meets_point_target"])

    def test_large_mean_with_uncertainty_is_not_confirmed(self):
        result = paired_summary([math.log(30)] * 5, [math.log(30 - d) for d in [-1, 0, 1, 2, 3]])
        self.assertGreater(result["mean_ppl_reduction"], 0.5)
        self.assertLess(result["ci95_ppl_reduction"][0], 0)
        self.assertFalse(result["passes_primary_criterion"])

    def test_mean_seed_ppl_is_not_exponential_of_mean_nll(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        report = paired_summary(values, [value - 0.1 for value in values])
        self.assertAlmostEqual(report["baseline_mean_ppl"], sum(math.exp(v) for v in values) / 5)
        self.assertNotAlmostEqual(report["baseline_mean_ppl"], math.exp(sum(values) / 5))

    def test_unlocked_results_remain_exploratory(self):
        self.manifest["locked"] = False
        report = summarize(self.manifest, self.root)
        self.assertEqual(report["status"], "exploratory")
        self.assertIsNone(report["confirmation"])

    def test_short_non_smoke_runs_can_be_reported_as_exploratory(self):
        self.manifest["locked"] = False
        tokens = 1000 * 524288
        self.manifest["expected"]["training_tokens"] = tokens
        for index in range(10):
            self.change_record(["training_tokens"], tokens, index)
            self.change_record(["final", "training_tokens"], tokens, index, "run")
            self.change_record(["metadata", "comparison_eligible"], False, index, "run")
        report = summarize(self.manifest, self.root)
        self.assertEqual(report["status"], "exploratory")
        self.assertIsNone(report["confirmation"])

    def test_missing_seed_and_duplicate_records_never_pass(self):
        for mode in ("missing", "duplicate", "missing_file"):
            with self.subTest(mode=mode):
                manifest = copy.deepcopy(self.manifest)
                if mode == "missing":
                    manifest["records"].pop()
                elif mode == "duplicate":
                    manifest["records"].append(manifest["records"][0])
                else:
                    manifest["records"][0]["evaluation"] = "absent.json"
                report = summarize(manifest, self.root)
                self.assertEqual(report["status"], "incomplete")
                self.assertIsNone(report["confirmation"])
                if mode == "missing_file":
                    self.assertEqual(len(report["run_diagnostics"]), 10)

    def test_identity_budget_and_evaluation_mismatches_block_confirmation(self):
        mutations = [
            ("evaluation", ["method"], "unknown"),
            ("evaluation", ["seed"], 42),
            ("evaluation", ["seed"], True),
            ("evaluation", ["status"], "failed"),
            ("evaluation", ["training_tokens"], TRAINING_TOKENS - 1),
            ("evaluation", ["shared_identity", "architecture", "width"], 1024),
            ("evaluation", ["shared_identity", "tokenizer"], "other"),
            ("evaluation", ["shared_identity", "data_sha256"], digest("1")),
            ("evaluation", ["shared_identity", "training_config_sha256"], digest("1")),
            ("evaluation", ["method_config_sha256"], digest("1")),
            ("evaluation", ["method_source_sha256", "gpt2_benchmark.py"], digest("1")),
            ("evaluation", ["checkpoint_sha256"], digest("2")),
            ("evaluation", ["source_sha256"], digest("1")),
            ("evaluation", ["protocol_sha256"], digest("1")),
            ("evaluation", ["confirmation_protocol_sha256"], digest("1")),
            ("evaluation", ["evaluator_sha256"], digest("1")),
            ("evaluation", ["device"], "cpu"),
            ("evaluation", ["batch_sequences"], 1),
            ("evaluation", ["evaluation_runtime", "torch"], "2.15.0"),
            ("evaluation", ["evaluation_runtime", "gpu"], "different GPU"),
            ("evaluation", ["source_commit"], "unreviewed"),
            ("evaluation", ["dataset_manifest_sha256"], digest("1")),
            ("evaluation", ["dataset_revision"], "other"),
            ("evaluation", ["evaluation", "token_offset"], 0),
            ("evaluation", ["evaluation", "token_count"], HOLDOUT_TOKENS - 1),
            ("evaluation", ["evaluation", "slice_id"], "development"),
            ("evaluation", ["evaluation", "data_sha256"], digest("1")),
            ("evaluation", ["evaluation", "aggregation"], "mean_of_batch_means"),
            ("evaluation", ["evaluation", "nll"], -1),
            ("evaluation", ["evaluation", "nll"], 1000),
            ("evaluation", ["evaluation", "ppl"], 0),
            ("run", ["metadata", "seed"], 202),
            ("run", ["metadata", "comparison_eligible"], False),
            ("run", ["metadata", "smoke_steps"], 2),
            ("run", ["final", "training_tokens"], TRAINING_TOKENS - 1),
            ("run", ["status"], "running"),
            ("run", ["shared_identity", "architecture", "width"], 1024),
            ("run", ["method_config_sha256"], digest("1")),
            ("run", ["metadata"], None),
        ]
        for part, keys, value in mutations:
            with self.subTest(part=part, keys=keys):
                path, original = self.read_record(part=part)
                self.change_record(keys, value, part=part)
                try:
                    report = summarize(self.manifest, self.root)
                    self.assertEqual(report["status"], "incomplete")
                    self.assertIsNone(report["confirmation"])
                    self.assertTrue(report["errors"])
                finally:
                    path.write_text(json.dumps(original))

    def test_unexpected_seed_cannot_replace_registered_seed(self):
        self.change_record(["seed"], 999)
        report = summarize(self.manifest, self.root)
        self.assertEqual(report["status"], "incomplete")
        self.assertIn({"method": "muon", "seed": 101}, report["missing_records"])

    def test_invalid_locked_protocol_is_rejected(self):
        mutations = [
            (["seeds"], [101, 202, 303, 404]),
            (["seeds"], [101, 101, 303, 404, 505]),
            (["seeds"], [101, 202, 303, 404, 999]),
            (["expected", "training_tokens"], 1000),
            (["expected", "method_config_sha256", "muon"], "unfrozen"),
            (["expected", "method_source_sha256"], {}),
            (["expected", "method_source_sha256", "candidate", "gpt2_evaluate.py"], digest("1")),
            (["expected", "evaluation_settings", "batch_sequences"], 0),
            (["expected", "evaluation", "token_offset"], 0),
            (["candidate_method"], "muon"),
            (["locked"], "yes"),
            (["schema_version"], True),
        ]
        for keys, value in mutations:
            with self.subTest(keys=keys):
                manifest = copy.deepcopy(self.manifest)
                cursor = manifest
                for key in keys[:-1]:
                    cursor = cursor[key]
                cursor[keys[-1]] = value
                with self.assertRaises(ValueError):
                    summarize(manifest, self.root)

    def test_protocol_digest_is_independent_of_record_paths_and_history(self):
        before = protocol_digest(self.manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest["records"] = []
        manifest["historical_reference"] = None
        self.assertEqual(before, protocol_digest(manifest))
        manifest["expected"]["method_config_sha256"]["candidate"] = digest("a")
        self.assertNotEqual(before, protocol_digest(manifest))

    def test_nonfinite_and_duplicate_json_fields_rejected(self):
        for raw in ('{"nll": NaN}', '{"nll": Infinity}', '{"seed": 101, "seed": 202}'):
            path = self.root / "bad.json"
            path.write_text(raw)
            with self.assertRaises(ValueError):
                read_json(path)
        for invalid in (float("nan"), float("inf"), -1.0, True, 1000):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                paired_summary([invalid] * 5, [1.0] * 5)

    def test_cli_resolves_paths_relative_to_manifest_and_writes_reports(self):
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest))
        output_json, output_md = self.root / "summary.json", self.root / "summary.md"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main(["--manifest", str(manifest_path), "--json", str(output_json), "--markdown", str(output_md)])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output_json.read_text())["status"], "confirmed")
        self.assertEqual(stdout.getvalue(), output_md.read_text())
        self.assertIn("Historical reference (excluded from confirmation)", stdout.getvalue())
        self.assertIn("| 101 |", stdout.getvalue())

    def test_incomplete_cli_has_nonzero_exit_and_preserves_failure_report(self):
        self.manifest["records"] = []
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest))
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main(["--manifest", str(manifest_path), "--stdout-format", "json"])
        self.assertEqual(status, 2)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "incomplete")

    def test_partial_results_show_all_seed_slots_without_statistics(self):
        self.manifest["records"] = self.manifest["records"][:2]
        report = summarize(self.manifest, self.root)
        self.assertEqual(len(report["pairs"]), 5)
        self.assertEqual(len(report["missing_records"]), 8)
        self.assertIsNone(report["confirmation"])
        self.assertIn("missing", render_markdown(report))


if __name__ == "__main__":
    unittest.main()
