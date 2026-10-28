"""Provenance, exact recipe and honest time-to-quality accounting checks."""

import ast
import io
import json
import math
import struct
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from exp.gpt2_benchmark import (
    PROTOCOL,
    VENDOR,
    MetricStream,
    adaptive_optimizer_diagnostics,
    first_target_observation,
    parse_validation,
    research_identity,
    shard_metadata,
    smoke_source,
    validate_dataset_files,
    validation_configuration,
    validation_source,
    verify_source,
)


class BenchmarkTests(unittest.TestCase):
    def test_adaptive_diagnostics_include_only_owned_scalar_state(self):
        import torch

        model = torch.nn.Module()
        model.register_parameter("owned", torch.nn.Parameter(torch.zeros(3, 8)))
        model.register_parameter("other_rank", torch.nn.Parameter(torch.zeros(4, 4)))
        optimizer = torch.optim.Optimizer(
            list(model.parameters()),
            {"method": "shampoo_adaptive", "alpha_max": 0.1, "update_frequency": 10},
        )
        block = {
            "left": torch.full((2, 3, 3), 987654.0),
            "right": torch.full((2, 4, 4), 987654.0),
            "left_root": torch.full((2, 3, 3), 987654.0),
            "right_root": torch.full((2, 4, 4), 987654.0),
            "left_alpha": torch.tensor([0.05, 0.1], dtype=torch.float64),
            "right_alpha": torch.tensor([0.02, 0.03], dtype=torch.float64),
            "left_scatter": torch.tensor([100.0, 200.0]),
            "right_scatter": torch.tensor([300.0, 400.0]),
            "last_refresh_step": 21,
        }
        optimizer.state[model.owned] = {"step": 23, "blocks": [block]}
        result = adaptive_optimizer_diagnostics(model, optimizer, rank=3)
        self.assertEqual(result["rank"], 3)
        self.assertEqual(len(result["parameters"]), 1)
        parameter = result["parameters"][0]
        self.assertEqual(parameter["name"], "owned")
        self.assertEqual(parameter["shape"], [3, 8])
        self.assertEqual(parameter["optimizer_step"], 23)
        saved = parameter["blocks"][0]
        self.assertEqual(saved["last_root_refresh_step"], 21)
        self.assertEqual(saved["left_alpha"], [0.05, 0.1])
        self.assertEqual(saved["right_scatter"], [300.0, 400.0])
        self.assertEqual(saved["left_tile_dimension"], 3)
        self.assertEqual(saved["right_tile_dimension"], 4)
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn("987654", serialized)
        self.assertNotIn("left_root", serialized)
        self.assertIn("not a training-time average", serialized)
        self.assertEqual(len(optimizer.state), 1)
        torch.testing.assert_close(block["left_alpha"], torch.tensor([0.05, 0.1], dtype=torch.float64))
        block["left_alpha"] = torch.ones(2, 2)
        with self.assertRaisesRegex(ValueError, "one scalar per tile"):
            adaptive_optimizer_diagnostics(model, optimizer, rank=3)

    def test_pinned_source_and_actual_model_size(self):
        import torch

        manifest = verify_source()
        protocol = json.loads(PROTOCOL.read_text())
        self.assertEqual(manifest["commit"], protocol["upstream_commit"])
        tree = ast.parse((VENDOR / "train_gpt_simple.py").read_text())
        classes: list[ast.stmt] = [
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name != "Muon"
        ]
        namespace: dict[str, Any] = {
            "torch": torch,
            "nn": torch.nn,
            "F": torch.nn.functional,
            "Tensor": torch.Tensor,
        }
        with torch.device("meta"):
            exec(  # noqa: S102 -- immutable source verified above; classes only
                compile(ast.Module(body=classes, type_ignores=[]), "upstream", "exec"),
                namespace,
            )
            model = namespace["GPT"](50304, 12, 768)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 162354816)
        self.assertIsNot(model.embed.weight, model.proj.weight)
        self.assertEqual(3250 * 524288, protocol["training"]["total_training_tokens"])

    def test_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for p in VENDOR.iterdir():
                if p.is_file():
                    (root / p.name).write_bytes(p.read_bytes())
            with (root / "train_gpt_simple.py").open("a") as stream:
                stream.write("\n# accidental edit\n")
            with self.assertRaisesRegex(ValueError, "Pinned source changed"):
                verify_source(root)

    def test_shard_header_size_and_content_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard.bin"
            header = struct.pack("<iii", 20240520, 1, 4) + bytes(1012)
            path.write_bytes(header + struct.pack("<4H", 1, 2, 3, 4))
            before = shard_metadata(path)
            self.assertEqual(before["tokens"], 4)
            path.write_bytes(header + struct.pack("<4H", 4, 3, 2, 1))
            self.assertNotEqual(before["sha256"], shard_metadata(path)["sha256"])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                shard_metadata(path)

    def test_unverified_glob_files_and_duplicate_manifest_entries_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = ["fineweb_train_000001.bin", "fineweb_val_000000.bin"]
            manifest = {"shards": 1, "files": [{"filename": name} for name in names]}
            for name in names:
                (directory / name).touch()
            validate_dataset_files(directory, manifest)
            extra = directory / "fineweb_train_000000.bin"
            extra.touch()
            with self.assertRaisesRegex(ValueError, "globs differ"):
                validate_dataset_files(directory, manifest)
            extra.unlink()
            manifest["files"].append(manifest["files"][0])
            with self.assertRaisesRegex(ValueError, "exact benchmark shard set"):
                validate_dataset_files(directory, manifest)

    def test_smoke_changes_only_two_budget_constants(self):
        source = (VENDOR / "train_gpt_simple.py").read_text()
        patched = ast.parse(smoke_source(source, 2))
        original = ast.parse(source)
        for tree in (patched, original):
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and target.id in (
                        "train_steps",
                        "val_tokens",
                    ):
                        node.value = ast.Constant(0)
        self.assertEqual(ast.dump(patched), ast.dump(original))
        with self.assertRaises(ValueError):
            smoke_source(source, 0)

    def test_validation_patch_changes_only_declared_evaluation_nodes(self):
        source = (VENDOR / "train_gpt_simple.py").read_text()
        original, patched = (
            ast.parse(source),
            ast.parse(validation_source(source, 1024, 524288, 50)),
        )
        patched.body = [
            node
            for node in patched.body
            if not (
                isinstance(node, ast.ImportFrom) and node.module == "exp.gpt2_evaluate"
            )
        ]
        for tree in (original, patched):
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and ast.unparse(node.targets[0])
                    in ("val_tokens", "(val_inputs, val_targets)", "val_step_freq")
                ):
                    node.value = ast.Constant(None)
        self.assertEqual(ast.dump(original), ast.dump(patched))
        with self.assertRaisesRegex(ValueError, "acquisition changed"):
            validation_source(
                source.replace(
                    'next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))',
                    "next(other_loader())",
                ),
                0,
                524288,
            )

    def test_training_validation_cannot_read_confirmation(self):
        protocol = json.loads(PROTOCOL.read_text())
        protocol["evaluation"]["development"] = {
            "token_offset": 0,
            "token_count": 10485760,
        }
        protocol["evaluation"]["confirmation"] = {
            "token_offset": 80000000,
            "token_count": 10485760,
        }
        self.assertEqual(validation_configuration(protocol)["token_offset"], 0)
        self.assertEqual(
            validation_configuration(protocol, smoke_steps=2)["token_count"], 524288
        )
        for args in ((80000000, 524288), (-1, 524288), (0, 1), (10485760, 524288)):
            with self.assertRaises(ValueError):
                validation_configuration(protocol, *args)
        protocol["evaluation"]["confirmation"]["token_offset"] = 524288
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validation_configuration(protocol)

    def test_shared_identity_excludes_optimizer_but_includes_training_budget(self):
        protocol = json.loads(PROTOCOL.read_text())
        source = verify_source()
        base = {
            "method": "muon",
            "steps": 3250,
            "matrix_lr": 0.025,
            "matrix_decay": 0.05,
        }
        other = base | {"method": "shampoo", "matrix_lr": 0.001}
        a = research_identity(protocol, base, source, "datahash", "revision")
        b = research_identity(protocol, other, source, "datahash", "revision")
        self.assertEqual(a["shared_identity"], b["shared_identity"])
        self.assertNotEqual(a["method_config_sha256"], b["method_config_sha256"])
        c = research_identity(
            protocol, base | {"steps": 100}, source, "datahash", "revision"
        )
        self.assertNotEqual(a["shared_identity"], c["shared_identity"])
        self.assertIn("research_optimizers.py", b["method_source_sha256"])

    def test_timing_parser_keeps_wall_and_training_time_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            stream = MetricStream(io.StringIO(), output, time.perf_counter() - 100, 8)
            stream.write("step:125/3250 val_loss:3.28000 ")
            self.assertEqual(stream.history, [])
            stream.write("train_time:12.000s step_avg:96.00ms\n")
            row = stream.history[0]
            self.assertAlmostEqual(row["ppl"], math.exp(3.28))
            self.assertEqual(row["training_seconds"], 12)
            self.assertGreaterEqual(row["end_to_end_seconds"], 100)
            self.assertGreater(row["end_to_end_gpu_hours"], row["training_gpu_hours"])
            with self.assertRaises(FloatingPointError):
                parse_validation("step:1/10 val_loss:nan train_time:10s")

    def test_target_crossing_is_bracketed_and_unreached_is_censored(self):
        history = [
            {"step": 125, "ppl": 30, "end_to_end_gpu_hours": 1},
            {"step": 250, "ppl": 26, "end_to_end_gpu_hours": 2},
        ]
        reached = first_target_observation(history, 27)
        self.assertEqual(reached["cost_interval"], [1, 2])
        self.assertEqual(reached["first_observed_cost"], 2)
        self.assertEqual(
            first_target_observation(history, 25),
            {"reached": False, "right_censored_after": 2},
        )


if __name__ == "__main__":
    unittest.main()
"""Provenance, exact recipe and honest time-to-quality accounting checks."""

import ast
import io
import json
import math
import struct
import tempfile
import time
import unittest
from pathlib import Path

from exp.gpt2_benchmark import (
    PROTOCOL,
    VENDOR,
    MetricStream,
    first_target_observation,
    parse_validation,
    shard_metadata,
    smoke_source,
    verify_source,
)


class BenchmarkTests(unittest.TestCase):
    def test_pinned_source_and_actual_model_size(self):
        import torch

        manifest = verify_source()
        protocol = json.loads(PROTOCOL.read_text())
        self.assertEqual(manifest["commit"], protocol["upstream_commit"])
        tree = ast.parse((VENDOR / "train_gpt_simple.py").read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name != "Muon"]
        namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional, "Tensor": torch.Tensor}
        with torch.device("meta"):
            exec(compile(ast.Module(body=classes, type_ignores=[]), "upstream", "exec"), namespace)
            model = namespace["GPT"](50304, 12, 768)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 162354816)
        self.assertIsNot(model.embed.weight, model.proj.weight)
        self.assertEqual(3250 * 524288, protocol["training"]["total_training_tokens"])

    def test_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for p in VENDOR.iterdir():
                if p.is_file():
                    (root / p.name).write_bytes(p.read_bytes())
            with (root / "train_gpt_simple.py").open("a") as stream:
                stream.write("\n# accidental edit\n")
            with self.assertRaisesRegex(ValueError, "Pinned source changed"):
                verify_source(root)

    def test_shard_header_size_and_content_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard.bin"
            header = struct.pack("<iii", 20240520, 1, 4) + bytes(1012)
            path.write_bytes(header + struct.pack("<4H", 1, 2, 3, 4))
            before = shard_metadata(path)
            self.assertEqual(before["tokens"], 4)
            path.write_bytes(header + struct.pack("<4H", 4, 3, 2, 1))
            self.assertNotEqual(before["sha256"], shard_metadata(path)["sha256"])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                shard_metadata(path)

    def test_smoke_changes_only_two_budget_constants(self):
        source = (VENDOR / "train_gpt_simple.py").read_text()
        patched = ast.parse(smoke_source(source, 2))
        original = ast.parse(source)
        for tree in (patched, original):
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name) and target.id in ("train_steps", "val_tokens"):
                        node.value = ast.Constant(0)
        self.assertEqual(ast.dump(patched), ast.dump(original))
        with self.assertRaises(ValueError):
            smoke_source(source, 0)

    def test_timing_parser_keeps_wall_and_training_time_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            stream = MetricStream(io.StringIO(), output, time.perf_counter() - 100, 8)
            stream.write("step:125/3250 val_loss:3.28000 ")
            self.assertEqual(stream.history, [])
            stream.write("train_time:12.000s step_avg:96.00ms\n")
            row = stream.history[0]
            self.assertAlmostEqual(row["ppl"], math.exp(3.28))
            self.assertEqual(row["training_seconds"], 12)
            self.assertGreaterEqual(row["end_to_end_seconds"], 100)
            self.assertGreater(row["end_to_end_gpu_hours"], row["training_gpu_hours"])
            with self.assertRaises(FloatingPointError):
                parse_validation("step:1/10 val_loss:nan train_time:10s")

    def test_target_crossing_is_bracketed_and_unreached_is_censored(self):
        history = [
            {"step": 125, "ppl": 30, "end_to_end_gpu_hours": 1},
            {"step": 250, "ppl": 26, "end_to_end_gpu_hours": 2},
        ]
        reached = first_target_observation(history, 27)
        self.assertEqual(reached["cost_interval"], [1, 2])
        self.assertEqual(reached["first_observed_cost"], 2)
        self.assertEqual(first_target_observation(history, 25), {"reached": False, "right_censored_after": 2})


if __name__ == "__main__":
    unittest.main()
