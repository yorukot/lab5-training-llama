"""Independent checks of fold weighting, source patch scope and communication."""

import ast
import copy
import itertools
import math
import os
import sys
import unittest

import torch
import torch.distributed as dist

from exp import gpt2_candidate as candidate
from exp.gpt2_benchmark import VENDOR, verify_source
from exp.tcd import precondition_first_order


def config(method="tcd1", steps=3250):
    return {"method": method, "steps": steps, "matrix_lr": 0.025,
            "matrix_decay": 0.05, "rho": 0.0 if method == "fold_rms" else 3.0}


class CandidateTests(unittest.TestCase):
    def test_anchor_ast_is_identical_to_upstream(self):
        verify_source()
        source = (VENDOR / "train_gpt_simple.py").read_text()
        changed = candidate.experiment_source(source, config("muon"))
        self.assertEqual(ast.dump(ast.parse(changed)), ast.dump(ast.parse(source)))

    def test_candidate_changes_only_declared_optimizer_and_reduction(self):
        verify_source()
        source = (VENDOR / "train_gpt_simple.py").read_text()
        original, patched = ast.parse(source), ast.parse(candidate.experiment_source(source, config()))
        patched.body = [n for n in patched.body if not (
            isinstance(n, ast.ImportFrom) and n.module == "exp.gpt2_candidate"
        )]
        for tree in (original, patched):
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "optimizer2":
                    node.value = ast.Constant(None)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    if ast.unparse(node.value.func) == "reduce_gradient" or (
                        ast.unparse(node.value.func) == "dist.all_reduce" and ast.unparse(node.value.args[0]) == "p.grad"
                    ):
                        node.value = ast.Constant(None)
        self.assertEqual(ast.dump(original), ast.dump(patched))

    def test_zero_risk_identical_folds_and_nesterov(self):
        torch.manual_seed(7)
        g, old = torch.randn(2, 3, 5, dtype=torch.float64)
        direction, next_m = candidate.fold_direction(g.repeat(4, 1, 1), old, 0, 0.95)
        expected_m = 0.95 * old + 0.05 * g
        nest = 0.05 * g + 0.95 * expected_m
        expected = nest / nest.square().mean().sqrt() / math.sqrt(5)
        torch.testing.assert_close(next_m, expected_m)
        torch.testing.assert_close(direction, expected)
        zero, _ = candidate.fold_direction(torch.zeros(4, 3, 5), torch.zeros(3, 5))
        self.assertEqual(zero.count_nonzero().item(), 0)

    def test_cross_fitted_scale_preserves_conditional_mean_identity(self):
        choices = torch.tensor([[[1., 2.], [3., 1.]], [[2., -1.], [1., 2.]]], dtype=torch.float64)
        mu = choices.mean(0)
        actual = torch.stack([
            candidate.fold_direction(choices[list(ids)], torch.zeros_like(mu), 3, 0)[0]
            for ids in itertools.product(range(2), repeat=4)
        ]).mean(0)
        expected = torch.stack([
            precondition_first_order(mu, (a-b)/2, (a+b)/2, 3)[0]
            / ((a+b)/2).square().mean().sqrt() / math.sqrt(2)
            for a, b in itertools.product(choices, repeat=2)
        ]).mean(0)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_local_sum_to_fold_mean_token_weighting(self):
        sums = torch.arange(8*3*5, dtype=torch.float64).reshape(8, 3, 5) * 65536
        old = torch.zeros(3, 5, dtype=torch.float64)
        expected_samples = torch.stack([(sums[2*i]+sums[2*i+1])/131072 for i in range(4)])
        expected, momentum = candidate.fold_direction(expected_samples, old, 3, 0.95)
        actual_state = old.clone()
        actual = candidate.matrix_update._torchdynamo_orig_callable(sums, actual_state, 3, 0.95)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_state, momentum)
        torch.testing.assert_close(momentum, sums.sum(0) / 524288 * 0.05)

    def test_invalid_config_rejected(self):
        source = (VENDOR / "train_gpt_simple.py").read_text()
        for change in ({"steps": 0}, {"rho": -1}, {"matrix_lr": float("nan")}, {"method": "unknown"}):
            with self.assertRaises(ValueError):
                candidate.experiment_source(source, config() | change)

    def test_research_patch_preserves_sum_reduction_and_auxiliary_adamw(self):
        source = (VENDOR / "train_gpt_simple.py").read_text()
        for method in (
            "matrix_adamw",
            "shampoo",
            "shampoo_shrink",
            "shampoo_adaptive",
            "soft_polar",
        ):
            experiment = {
                "method": method,
                "matrix_lr": 0.001,
                "matrix_decay": 0.05,
                "steps": 3250,
            }
            patched = ast.parse(candidate.experiment_source(source, experiment))
            original = ast.parse(source)
            patched.body = [
                node
                for node in patched.body
                if not (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "exp.research_optimizers"
                )
            ]
            for tree in (original, patched):
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Assign)
                        and len(node.targets) == 1
                        and ast.unparse(node.targets[0]) == "optimizer2"
                    ):
                        node.value = ast.Constant(None)
            self.assertEqual(ast.dump(patched), ast.dump(original), method)

    def test_research_defaults_explicit_and_ambiguous_optimizer_rejected(self):
        source = (VENDOR / "train_gpt_simple.py").read_text()
        experiment = {
            "method": "shampoo_shrink",
            "matrix_lr": 0.001,
            "matrix_decay": 0.05,
            "steps": 100,
        }
        normalized = candidate.validate_experiment(experiment)
        self.assertGreater(normalized["optimizer_kwargs"]["shrinkage"], 0)
        self.assertEqual(candidate.validate_experiment(normalized), normalized)
        for change in (
            {"optimizer_kwargs": {"not_an_option": 1}},
            {"rho": 1},
            {"steps": True},
        ):
            with self.assertRaises(ValueError):
                candidate.validate_experiment(experiment | change)
        with self.assertRaisesRegex(ValueError, "arguments changed"):
            candidate.experiment_source(
                source.replace(
                    "lr=0.025, weight_decay=0.05)",
                    "lr=0.025, weight_decay=0.05, mu=0.95)",
                ),
                experiment,
            )

    def test_integrated_rule_retains_curvature_failure(self):
        g = torch.tensor([[0.0, -1.0], [0.0, 0.0]], dtype=torch.float64)
        e = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
        samples = torch.stack([g + e, g - e, g + e, g - e])
        losses = []
        for rho in (0.0, 3.0):
            d, _ = candidate.fold_direction(samples, torch.zeros_like(g), rho, 0.0)
            w = -0.1 * d
            losses.append(
                (0.5 * (w[0, 1] - 1).square() + 5000 * w[1, 0].square()).item()
            )
        self.assertLess(losses[0], 0.5)
        self.assertGreater(losses[1], 7.0)


def distributed_check():
    """Eight-rank end-to-end collector vs independent one-process calculation."""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 8
    # Check the collector without compilation; the subsequent model smoke checks
    # compiled execution on the real shapes under isolated per-rank caches.
    candidate.matrix_update = candidate.matrix_update._torchdynamo_orig_callable
    for rho in (0., 3.):
        params = [torch.nn.Parameter(torch.full((3, 5), 0.1*(i+1), device="cuda")) for i in range(8)]
        opt = candidate.MatrixTCD(params, lr=0.025, weight_decay=0.05, rho=rho)
        expected = [p.detach().clone() for p in params]
        momenta = [torch.zeros_like(p) for p in params]
        for step in range(2):
            for i, p in enumerate(params):
                grid = torch.arange(15, device="cuda", dtype=torch.float32).reshape(3, 5)
                p.grad = (torch.sin(grid + rank + i + step) + 0.2) * 65536
                local_before = p.grad.clone()
                candidate.reduce_gradient(p, opt)
                torch.testing.assert_close(p.grad, local_before, rtol=0, atol=0)
                all_means = torch.stack([torch.sin(grid + r + i + step) + 0.2 for r in range(8)])
                samples = all_means.reshape(4, 2, 3, 5).mean(1)
                direction, momenta[i] = candidate.fold_direction(samples, momenta[i], rho)
                expected[i].mul_(1 - 0.025 * 0.05).add_(direction, alpha=-0.025)
            aux = torch.nn.Parameter(torch.zeros(1, device="cuda"))
            aux.grad = torch.full_like(aux, rank + 1)
            candidate.reduce_gradient(aux, opt)
            torch.testing.assert_close(aux.grad, torch.full_like(aux, 36))
            opt.step()
            for p, answer in zip(params, expected, strict=True):
                torch.testing.assert_close(p, answer, rtol=2e-5, atol=2e-6)
        state = copy.deepcopy(opt.state_dict())
        restored = candidate.MatrixTCD(params, rho=rho)
        restored.load_state_dict(state)
        for p in opt.state:
            torch.testing.assert_close(restored.state[p]["momentum"], opt.state[p]["momentum"])
    dist.barrier()
    if rank == 0:
        print("Eight-rank fold collector, parameter sync, auxiliary sum, and momentum checks passed.", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    if "--distributed" in sys.argv:
        distributed_check()
    else:
        unittest.main()
