"""Held-out slice, exact architecture and checkpoint provenance checks."""

import ast
import copy
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from exp.gpt2_benchmark import PROTOCOL, VENDOR, sha256, shard_metadata, verify_source
from exp.gpt2_evaluate import (
    build_model,
    confirmation_gate,
    evaluate,
    evaluate_batches,
    evaluation_runtime,
    export_checkpoint,
    load_checkpoint,
    model_classes,
    slice_tensors,
    validate_slice,
)


def write_shard(path, values):
    path.write_bytes(
        struct.pack("<iii", 20240520, 1, len(values))
        + bytes(1012)
        + struct.pack(f"<{len(values)}H", *values)
    )


class EvaluationTests(unittest.TestCase):
    def test_reserved_targets_require_a_lock_before_checkpoint_or_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with (
                mock.patch(
                    "exp.gpt2_evaluate.load_checkpoint",
                    side_effect=AssertionError("checkpoint must not load"),
                ),
                self.assertRaisesRegex(ValueError, "confirmation-manifest"),
            ):
                evaluate(
                    directory / "missing.pt",
                    directory,
                    directory / "result.json",
                    80000000,
                    1024,
                    device="cpu",
                )
            self.assertEqual(
                json.loads((directory / "result.json").read_text())["status"], "error"
            )
        self.assertIsNone(
            confirmation_gate({}, 80000000 - 1024, 1024, "", 8, "cpu", {})
        )
        with self.assertRaisesRegex(ValueError, "confirmation-manifest"):
            confirmation_gate({}, 80000000 - 1024, 2048, "", 8, "cpu", {})

    def test_confirmation_gate_checks_frozen_payload_slice_runtime_and_source(self):
        from exp import gpt2_evaluate
        from exp.research_report import protocol_digest

        digest = "a" * 64
        sources = {"gpt2_evaluate.py": sha256(gpt2_evaluate.__file__)}
        shared = {
            "architecture": "pinned",
            "tokenizer": "gpt2",
            "data_sha256": digest,
            "training_config_sha256": digest,
            "dataset_revision": "revision",
            "data_order": "fixed",
        }
        runtime = evaluation_runtime("cpu")
        payload = {
            "method": "muon",
            "seed": 101,
            "training_tokens": 1703936000,
            "shared_identity": shared,
            "method_config_sha256": digest,
            "method_source_sha256": sources,
        }
        manifest = {
            "schema_version": 1,
            "locked": True,
            "baseline_method": "muon",
            "candidate_method": "soft_polar",
            "seeds": [101, 202, 303, 404, 505],
            "records": [],
            "expected": {
                "training_tokens": 1703936000,
                "shared_identity": shared,
                "method_config_sha256": {"muon": digest, "soft_polar": digest},
                "method_source_sha256": {"muon": sources, "soft_polar": sources},
                "evaluation": {
                    "slice_id": "start-80000000-count-10485760",
                    "token_offset": 80000000,
                    "token_count": 10485760,
                    "data_sha256": digest,
                },
                "evaluation_settings": {"device": "cpu", "batch_sequences": 8},
                "evaluation_runtime": runtime,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "confirmation.json"
            path.write_text(json.dumps(manifest))

            def check(
                current=payload, start=80000000, count=10485760, settings=runtime
            ):
                return confirmation_gate(
                    current, start, count, digest, 8, "cpu", settings, path
                )

            self.assertEqual(check(), protocol_digest(manifest))
            for change in (
                {"method": "unknown"},
                {"seed": 42},
                {"training_tokens": 524288},
                {"shared_identity": {}},
                {"method_config_sha256": "b" * 64},
                {"method_source_sha256": {}},
            ):
                with self.assertRaises(ValueError):
                    check(payload | change)
            with self.assertRaisesRegex(ValueError, "slice"):
                check(count=1024)
            with self.assertRaisesRegex(ValueError, "runtime"):
                check(settings=runtime | {"torch": "changed"})
            changed = copy.deepcopy(manifest)
            changed["locked"] = False
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "must be locked"):
                check()
            changed = copy.deepcopy(manifest)
            for method in ("muon", "soft_polar"):
                changed["expected"]["method_source_sha256"][method][
                    "gpt2_evaluate.py"
                ] = "b" * 64
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "Evaluator source"):
                check(
                    payload | {"method_source_sha256": {"gpt2_evaluate.py": "b" * 64}}
                )

    def test_target_offsets_and_exact_end_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            write_shard(path, list(range(17)))
            x, y = slice_tensors(path, 0, 8, sequence_length=4)
            self.assertEqual(x.flatten().tolist(), list(range(8)))
            self.assertEqual(y.flatten().tolist(), list(range(1, 9)))
            x, y = slice_tensors(path, 8, 8, sequence_length=4)
            self.assertEqual(x.flatten().tolist(), list(range(8, 16)))
            self.assertEqual(y.flatten().tolist(), list(range(9, 17)))
            for offset, count in ((9, 8), (-1, 8), (0, 0), (0, 7), (True, 8)):
                with self.assertRaises(ValueError):
                    validate_slice(path, offset, count, sequence_length=4)

    def test_evaluation_uses_token_sum_and_partial_last_batch(self):
        class SumLoss(torch.nn.Module):
            def forward(self, x, y):
                return y.double().sum()

        x = torch.zeros(5, 4, dtype=torch.int32)
        y = torch.arange(20).reshape(5, 4)
        result = evaluate_batches(SumLoss(), x, y, batch_sequences=2, device="cpu")
        self.assertEqual(result["token_count"], 20)
        self.assertEqual(result["nll"], 9.5)
        self.assertEqual(result["ppl"], math.exp(9.5))

        class BadLoss(SumLoss):
            def forward(self, x, y):
                return torch.tensor(float("nan"))

        with self.assertRaises(FloatingPointError):
            evaluate_batches(BadLoss(), x, y, device="cpu")

    def test_only_model_classes_execute_and_architecture_is_identical(self):
        expected = json.loads(PROTOCOL.read_text())["model"]
        with (
            mock.patch(
                "torch.distributed.init_process_group",
                side_effect=AssertionError("trainer executed"),
            ),
            mock.patch(
                "torch.cuda.set_device",
                side_effect=AssertionError("CUDA setup executed"),
            ),
        ):
            model = build_model(expected)
        self.assertEqual(
            sum(p.numel() for p in model.parameters()), expected["parameters"]
        )
        self.assertEqual(model.embed.weight.dtype, torch.bfloat16)
        self.assertEqual(model.proj.weight.dtype, torch.float32)
        self.assertIsNot(model.embed.weight, model.proj.weight)
        with self.assertRaisesRegex(ValueError, "architecture"):
            build_model(expected | {"tied_embedding": True})
        definitions = [
            node
            for node in ast.parse((VENDOR / "train_gpt_simple.py").read_text()).body
            if isinstance(node, ast.ClassDef) and node.name != "Muon"
        ]
        namespace = {
            "torch": torch,
            "nn": torch.nn,
            "F": torch.nn.functional,
            "Tensor": torch.Tensor,
        }
        exec(  # noqa: S102 -- verified source, classes only
            compile(
                ast.Module(body=definitions, type_ignores=[]), "upstream_test", "exec"
            ),
            namespace,
        )
        standalone = model_classes()["GPT"](32, 1, 128)
        reference = namespace["GPT"](32, 1, 128)
        reference.load_state_dict(standalone.state_dict())
        inputs = torch.arange(8).reshape(2, 4)
        targets = inputs + 1
        torch.testing.assert_close(
            standalone(inputs, targets), reference(inputs, targets), rtol=0, atol=0
        )

    def test_checkpoint_roundtrip_provenance_and_mismatch(self):
        source = verify_source()
        model = torch.nn.Linear(2, 2)
        metadata = {
            "protocol": json.loads(PROTOCOL.read_text()),
            "source": source,
            "seed": 101,
            "experiment": {"method": "muon"},
            "dataset_manifest_sha256": "data",
            "dataset_revision": "revision",
            "protocol_sha256": "protocol",
            "method": "muon",
            "shared_identity": {"architecture": "test"},
            "method_config_sha256": "config",
            "method_source_sha256": {"source.py": "hash"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            record = export_checkpoint(path, model, metadata, 524288)
            self.assertEqual(record["format_version"], 1)
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_checkpoint(path, model, metadata, 524288)
            with mock.patch(
                "exp.gpt2_evaluate.build_model", return_value=torch.nn.Linear(2, 2)
            ):
                restored, payload = load_checkpoint(path)
            torch.testing.assert_close(restored.weight, model.weight)
            self.assertEqual(payload["training_tokens"], 524288)
            self.assertEqual(
                payload["method_source_sha256"], metadata["method_source_sha256"]
            )
            payload["source_sha256"] = "changed"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "source identity"):
                load_checkpoint(path)

    def test_wrong_checkpoint_dtype_and_error_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            source = verify_source()
            torch.save(
                {
                    "format_version": 1,
                    "source_commit": source["commit"],
                    "source_sha256": source["files"]["train_gpt_simple.py"]["sha256"],
                    "model": {},
                    "state_dict": torch.nn.Linear(2, 2).double().state_dict(),
                },
                path,
            )
            with (
                mock.patch(
                    "exp.gpt2_evaluate.build_model", return_value=torch.nn.Linear(2, 2)
                ),
                self.assertRaisesRegex(ValueError, "dtypes"),
            ):
                load_checkpoint(path)
            output = Path(directory) / "evaluation.json"
            with self.assertRaises(ValueError):
                evaluate(path, Path(directory), output, 0, 1024, device="cpu")
            self.assertEqual(json.loads(output.read_text())["status"], "error")
            with self.assertRaisesRegex(ValueError, "output exists"):
                evaluate(path, Path(directory), output, 0, 1024, device="cpu")

    def test_evaluation_result_binds_checkpoint_dataset_and_slice(self):
        class SumLoss(torch.nn.Module):
            def forward(self, inputs, targets):
                return targets.double().sum() / 1000

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "checkpoint.pt"
            checkpoint.write_bytes(b"mock checkpoint")
            shard = directory / "fineweb_val_000000.bin"
            write_shard(shard, list(range(2049)))
            manifest_path = directory / "dataset_manifest.json"
            manifest_path.write_text(
                json.dumps({"revision": "revision", "files": [shard_metadata(shard)]})
            )
            payload = {
                "seed": 101,
                "training_tokens": 1703936000,
                "experiment": {},
                "method": "muon",
                "shared_identity": {},
                "method_config_sha256": "config",
                "method_source_sha256": {},
                "protocol_sha256": "protocol",
                "source_commit": "commit",
                "source_sha256": "source",
                "dataset_manifest_sha256": sha256(manifest_path),
                "dataset_revision": "revision",
                "model": {},
            }
            output = directory / "evaluated.json"
            with mock.patch(
                "exp.gpt2_evaluate.load_checkpoint", return_value=(SumLoss(), payload)
            ):
                result = evaluate(
                    checkpoint, directory, output, 1024, 1024, device="cpu"
                )
                self.assertEqual(result["checkpoint_sha256"], sha256(checkpoint))
                self.assertEqual(result["evaluation"]["first_target_shard_index"], 1025)
                self.assertEqual(result["evaluation"]["token_count"], 1024)
                self.assertAlmostEqual(
                    result["evaluation"]["nll"], (1025 + 2048) / 2000
                )
                self.assertEqual(
                    result["evaluation"]["data_sha256"], shard_metadata(shard)["sha256"]
                )
                manifest_path.write_text(manifest_path.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "dataset identity"):
                    evaluate(
                        checkpoint,
                        directory,
                        directory / "bad.json",
                        0,
                        1024,
                        device="cpu",
                    )


if __name__ == "__main__":
    unittest.main()
