"""Run on nano4: PYTHONPATH=src python -m exp.test_research_optimizers.

Distributed check: PYTHONPATH=src torchrun --standalone --nproc-per-node=2
    -m exp.test_research_optimizers --distributed
Use --nccl with --distributed for CUDA collectives on an allocated GPU node.
"""

import copy
import json
import math
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist

from exp.research_optimizers import (
    RESEARCH_METHODS,
    MatrixResearchOptimizer,
    _soft_polar_kernel,
    ema_anisotropy_scatter,
    ema_shrinkage_alpha,
    ema_weight_statistics,
    inverse_fourth_root,
    make_matrix_optimizer,
    soft_polar_direction,
    soft_polar_reference,
    trace_shrink,
    validate_research_config,
)


class ResearchOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_adam_matches_pytorch_on_sum_gradients(self):
        torch.manual_seed(32)
        parameters = [torch.nn.Parameter(torch.randn(3, 5, dtype=torch.float64))]
        references = [torch.nn.Parameter(parameters[0].detach().clone())]
        actual = make_matrix_optimizer(
            parameters, "matrix_adamw", lr=0.003, weight_decay=0.07
        )
        expected = torch.optim.AdamW(
            references, lr=0.003, weight_decay=0.07, betas=(0.9, 0.95), eps=1e-8
        )
        for step in range(5):
            gradient = torch.randn_like(parameters[0]) * 524288
            parameters[0].grad = gradient.clone()
            references[0].grad = gradient.clone()
            actual.step()
            expected.step()
            torch.testing.assert_close(
                parameters[0], references[0], rtol=1e-12, atol=1e-12
            )
            self.assertEqual(actual.state[parameters[0]]["step"], step + 1)
        torch.testing.assert_close(
            actual.state[parameters[0]]["exp_avg"],
            expected.state[references[0]]["exp_avg"],
        )

    def test_soft_polar_matches_svd_for_rectangular_and_degenerate_inputs(self):
        torch.manual_seed(24)
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            for shape in ((3, 7), (7, 3), (4, 4)):
                random = torch.randn(shape).to(dtype)
                rank_one = torch.ones(shape, dtype=dtype)
                for matrix in (random, rank_one, torch.zeros_like(random)):
                    with self.subTest(dtype=dtype, shape=shape):
                        actual = soft_polar_direction(matrix)
                        expected = soft_polar_reference(matrix)
                        torch.testing.assert_close(
                            actual, expected, rtol=3e-4, atol=3e-5
                        )
                        self.assertLessEqual(
                            torch.linalg.svdvals(actual).max().item(), 1.0001
                        )
                        self.assertTrue(torch.isfinite(actual).all())

    def test_small_ridge_finite_iteration_error_is_measured_not_hidden(self):
        # 12 iterations are an approximation, particularly near minimum tau.
        # The actual transfer function is compared, not a residual on nullspace.
        matrix = torch.diag(
            torch.tensor([1.0, 0.05, 0.005, 1e-5, 0.0], dtype=torch.float64)
        )
        expected = soft_polar_reference(matrix, tau=0.01)
        actual = _soft_polar_kernel(matrix, tau=0.01, iterations=12)
        relative = (actual - expected).norm() / expected.norm()
        self.assertLess(relative.item(), 0.1)
        self.assertGreater(relative.item(), 0)
        with self.assertRaises(ValueError):
            soft_polar_direction(matrix, tau=0.01, iterations=12)
        torch.testing.assert_close(
            soft_polar_direction(matrix, tau=0.01, iterations=20),
            expected,
            rtol=1e-9,
            atol=1e-10,
        )
        self.assertTrue((actual.diagonal() <= expected.diagonal() + 1e-12).all())

    def test_soft_polar_equivariance_scaling_and_rank_aware_limit(self):
        matrix = torch.diag(torch.tensor([3.0, 0.2, 0.0], dtype=torch.float64))
        direction = soft_polar_direction(matrix)
        torch.testing.assert_close(direction, soft_polar_direction(matrix * 524288))
        torch.testing.assert_close(direction.mT, soft_polar_direction(matrix.mT))
        torch.testing.assert_close(
            soft_polar_reference(matrix, tau=0),
            torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)),
        )
        # Norm matching is an explicit ablation, separate from the bounded map.
        parameter = torch.nn.Parameter(torch.zeros(3, 3, dtype=torch.float64))
        optimizer = make_matrix_optimizer(
            [parameter], "soft_polar", lr=1, mu=0, match_update_norm=True
        )
        parameter.grad = matrix.clone()
        optimizer.step()
        self.assertAlmostEqual(parameter.norm().item(), math.sqrt(3), places=10)

    def test_shrinkage_preserves_trace_changes_inverse_and_leaves_input(self):
        factor = torch.diag(torch.tensor([100.0, 1.0, 0.0], dtype=torch.float64))
        before = factor.clone()
        shrunk = trace_shrink(factor, 0.25)
        torch.testing.assert_close(shrunk.trace(), factor.trace(), rtol=1e-14, atol=0)
        torch.testing.assert_close(factor, before, rtol=0, atol=0)
        expected = 0.75 * factor.diagonal() + 0.25 * factor.trace() / 3
        expected = (expected + 1e-6 * factor.trace() / 3).pow(-0.25)
        torch.testing.assert_close(
            inverse_fourth_root(factor, 0.25).diagonal(), expected
        )
        self.assertFalse(
            torch.allclose(
                inverse_fourth_root(factor, 0), inverse_fourth_root(factor, 0.25)
            )
        )

    def test_shampoo_uses_two_inverse_fourth_powers_and_bias_correction(self):
        gradient = torch.diag(torch.tensor([3.0, 0.5], dtype=torch.float64))
        parameter = torch.nn.Parameter(torch.zeros_like(gradient))
        optimizer = make_matrix_optimizer(
            [parameter], "shampoo", lr=1, graft="none", block_size=2
        )
        parameter.grad = gradient.clone()
        optimizer.step()
        factors = gradient.diagonal().square()
        ridge = 1e-6 * factors.mean()
        expected = gradient.diagonal() / (factors + ridge).sqrt()
        torch.testing.assert_close(
            -parameter.diagonal(), expected, rtol=1e-10, atol=1e-10
        )

    def test_batched_shampoo_tiles_match_independent_svd_including_edges(self):
        torch.manual_seed(108)
        gradient = torch.randn(5, 7, dtype=torch.float64)
        parameter = torch.nn.Parameter(torch.zeros_like(gradient))
        optimizer = make_matrix_optimizer(
            [parameter], "shampoo", lr=1, block_size=3, graft="none"
        )
        parameter.grad = gradient.clone()
        optimizer.step()
        expected = torch.empty_like(gradient)
        for row in range(0, 5, 3):
            for col in range(0, 7, 3):
                tile = gradient[row : row + 3, col : col + 3]
                u, singular, vh = torch.linalg.svd(tile, full_matrices=False)
                trace = singular.square().sum()
                left = singular.square() + 1e-6 * trace / tile.shape[0]
                right = singular.square() + 1e-6 * trace / tile.shape[1]
                filtered = singular / (left * right).pow(0.25)
                expected[row : row + 3, col : col + 3] = (
                    u * filtered.unsqueeze(0)
                ) @ vh
        torch.testing.assert_close(-parameter, expected, rtol=1e-9, atol=1e-10)

    def test_zero_shrinkage_recovers_parent_and_raw_factors_are_identical(self):
        torch.manual_seed(4)
        weights = [torch.nn.Parameter(torch.randn(3, 5)) for _ in range(3)]
        for p in weights[1:]:
            p.data.copy_(weights[0])
        methods = ("shampoo", "shampoo_shrink", "shampoo_shrink")
        optimizers = [
            make_matrix_optimizer(
                [p], method, lr=0.01, block_size=2, update_frequency=2, shrinkage=rho
            )
            for p, method, rho in zip(weights, methods, (0, 0, 0.3), strict=True)
        ]
        for _ in range(4):
            gradient = torch.randn(3, 5) * 1000
            for p, optimizer in zip(weights, optimizers, strict=True):
                p.grad = gradient.clone()
                optimizer.step()
            torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
            states = [
                optimizer.state[p]
                for optimizer, p in zip(optimizers, weights, strict=True)
            ]
            for a, b in zip(states[0]["blocks"], states[2]["blocks"], strict=True):
                for side in ("left", "right"):
                    torch.testing.assert_close(a[side], b[side], rtol=0, atol=0)
        self.assertFalse(torch.allclose(weights[0], weights[2], rtol=1e-6, atol=1e-6))

    def test_zero_gradients_dtype_and_checkpoint_continuation(self):
        torch.manual_seed(64)
        for method in RESEARCH_METHODS:
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(method=method, dtype=dtype):
                    parameter = torch.nn.Parameter(torch.ones(3, 5, dtype=dtype))
                    optimizer = make_matrix_optimizer(
                        [parameter], method, weight_decay=0
                    )
                    parameter.grad = torch.zeros_like(parameter)
                    optimizer.step()
                    torch.testing.assert_close(
                        parameter, torch.ones_like(parameter), rtol=0, atol=0
                    )
                    parameter.grad = torch.randn_like(parameter)
                    optimizer.step()
                    saved = copy.deepcopy(optimizer.state_dict())
                    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
                    restored = make_matrix_optimizer([restored_parameter], method)
                    restored.load_state_dict(saved)
                    parameter.grad = torch.randn_like(parameter)
                    restored_parameter.grad = parameter.grad.clone()
                    optimizer.step()
                    restored.step()
                    torch.testing.assert_close(
                        parameter, restored_parameter, rtol=0, atol=0
                    )
                    for key in ("momentum", "exp_avg", "exp_avg_sq"):
                        if key in optimizer.state[parameter]:
                            self.assertEqual(
                                restored.state[restored_parameter][key].dtype,
                                torch.float32,
                            )
                            torch.testing.assert_close(
                                optimizer.state[parameter][key],
                                restored.state[restored_parameter][key],
                                rtol=0,
                                atol=0,
                            )

    def test_invalid_settings_and_checkpoint_topology(self):
        invalid = (
            ("matrix_adamw", {"beta1": 1}),
            ("matrix_adamw", {"epsilon": float("nan")}),
            ("matrix_adamw", {"unused": 3}),
            ("shampoo", {"shrinkage": 0.1}),
            ("shampoo_shrink", {"shrinkage": 1.1}),
            ("shampoo", {"block_size": True}),
            ("shampoo", {"update_frequency": 0}),
            ("shampoo_adaptive", {"shrinkage": 0.1}),
            ("shampoo_adaptive", {"alpha_max": -0.1}),
            ("shampoo_adaptive", {"alpha_max": 1.1}),
            ("shampoo_adaptive", {"alpha_max": float("nan")}),
            ("shampoo_adaptive", {"alpha_max": True}),
            ("soft_polar", {"tau": 0}),
            ("soft_polar", {"iterations": 2.5}),
            ("soft_polar", {"match_update_norm": 1}),
        )
        for method, settings in invalid:
            with (
                self.subTest(method=method, settings=settings),
                self.assertRaises(ValueError),
            ):
                validate_research_config(method, settings)
        parameter = torch.nn.Parameter(torch.zeros(2, 2))
        optimizer = make_matrix_optimizer([parameter], "matrix_adamw")
        state = optimizer.state_dict()
        state["research_topology"]["world_size"] = 8
        with self.assertRaises(ValueError):
            optimizer.load_state_dict(state)
        with self.assertRaises(ValueError):
            optimizer.step()

    def test_adaptive_scatter_matches_explicit_weighted_history(self):
        torch.manual_seed(218)
        samples = torch.randn(8, 3, 3, 3, dtype=torch.float64)
        samples = samples @ samples.mT
        identity = torch.eye(3, dtype=torch.float64)
        for beta2 in (0.0, 0.5, 0.95, 0.999999):
            raw = torch.zeros_like(samples[0])
            scatter = torch.zeros(3, dtype=torch.float64)
            for step, sample in enumerate(samples, start=1):
                previous_mean = raw / (1 - beta2 ** (step - 1)) if step > 1 else raw
                scatter = ema_anisotropy_scatter(
                    sample, previous_mean, scatter, beta2, step
                )
                raw = beta2 * raw + (1 - beta2) * sample
                mean = raw / (1 - beta2**step)
                weights = torch.tensor(
                    [beta2 ** (step - 1 - i) for i in range(step)], dtype=torch.float64
                )
                weights /= weights.sum()
                history = samples[:step]
                trace_mean = history.diagonal(dim1=-2, dim2=-1).mean(-1)
                centered = history - trace_mean[..., None, None] * identity
                center_mean = (weights[:, None, None, None] * centered).sum(0)
                expected_scatter = (
                    weights[:, None] * (centered - center_mean).square().sum((-2, -1))
                ).sum(0)
                torch.testing.assert_close(
                    scatter, expected_scatter, rtol=2e-10, atol=2e-12
                )
                newest, squared_sum = ema_weight_statistics(beta2, step)
                self.assertAlmostEqual(newest, weights[-1].item(), places=12)
                self.assertAlmostEqual(
                    squared_sum, weights.square().sum().item(), places=12
                )
                alpha = ema_shrinkage_alpha(mean, scatter, beta2, step, alpha_max=1)
                if step == 1 or beta2 == 0:
                    torch.testing.assert_close(
                        alpha, torch.zeros_like(alpha), rtol=0, atol=0
                    )
                else:
                    uncertainty = squared_sum * expected_scatter / (1 - squared_sum)
                    expected_alpha = (
                        uncertainty / center_mean.square().sum((-2, -1))
                    ).clamp(0, 1)
                    torch.testing.assert_close(
                        alpha, expected_alpha, rtol=2e-9, atol=2e-12
                    )

    def test_adaptive_scale_invariance_and_tensor_shrinkage(self):
        torch.manual_seed(307)
        samples = torch.randn(6, 2, 3, 3, dtype=torch.float64)
        samples = samples @ samples.mT
        estimates = []
        # Scaling gradients changes Gram factors quadratically; the ratio must
        # cancel the resulting fourth-power change in scatter and anisotropy.
        for scale in (1e-4, 1.0, 1e6):
            raw = torch.zeros_like(samples[0])
            scatter = torch.zeros(2, dtype=torch.float64)
            for step, sample in enumerate(samples * scale, start=1):
                old_mean = raw / (1 - 0.8 ** (step - 1)) if step > 1 else raw
                scatter = ema_anisotropy_scatter(sample, old_mean, scatter, 0.8, step)
                raw = 0.8 * raw + 0.2 * sample
            estimates.append(
                ema_shrinkage_alpha(raw / (1 - 0.8**6), scatter, 0.8, 6, 1)
            )
        for alpha in estimates[1:]:
            torch.testing.assert_close(alpha, estimates[0], rtol=2e-12, atol=2e-13)
        factors = samples[0]
        alphas = torch.tensor([0.0, 0.2], dtype=torch.float64)
        torch.testing.assert_close(
            trace_shrink(factors, alphas),
            torch.stack(
                [
                    trace_shrink(factor, alpha.item())
                    for factor, alpha in zip(factors, alphas, strict=True)
                ]
            ),
        )
        torch.testing.assert_close(
            inverse_fourth_root(factors, alphas),
            torch.stack(
                [
                    inverse_fourth_root(factor, alpha.item())
                    for factor, alpha in zip(factors, alphas, strict=True)
                ]
            ),
        )
        for invalid in (
            torch.tensor([0.1, 1.1]),
            torch.tensor(float("nan")),
            torch.ones(2, 1),
        ):
            with self.assertRaises(ValueError):
                trace_shrink(factors, invalid)

    def test_adaptive_preserves_reliable_weak_mode_but_shrinks_noisy_isotropy(self):
        # Deterministic weak eigenvalues are useful structure, not automatically
        # noise. The estimator distinguishes temporal variation from magnitude.
        stable = torch.diag(torch.tensor([4.0, 2.0**-20], dtype=torch.float64))
        noisy = [
            torch.diag(torch.tensor(pair, dtype=torch.float64))
            for pair in ((4.0, 1.0), (1.0, 4.0))
        ]
        last_alphas = []
        for history in ([stable] * 12, noisy * 6):
            raw = torch.zeros_like(stable)
            scatter = torch.zeros((), dtype=torch.float64)
            for step, sample in enumerate(history, start=1):
                old_mean = raw / (1 - 0.5 ** (step - 1)) if step > 1 else raw
                scatter = ema_anisotropy_scatter(sample, old_mean, scatter, 0.5, step)
                raw = 0.5 * raw + 0.5 * sample
            mean = raw / (1 - 0.5**12)
            alpha = ema_shrinkage_alpha(mean, scatter, 0.5, 12, alpha_max=1)
            last_alphas.append(alpha.item())
            if history[0] is stable:
                torch.testing.assert_close(
                    trace_shrink(mean, alpha), stable, rtol=0, atol=0
                )
        self.assertEqual(last_alphas[0], 0)
        self.assertGreater(last_alphas[1], 0.9)
        isotropic = torch.eye(2, dtype=torch.float64)
        self.assertEqual(
            ema_shrinkage_alpha(isotropic, torch.tensor(0.0), 0.9, 3).item(), 0
        )
        self.assertAlmostEqual(
            ema_shrinkage_alpha(isotropic, torch.tensor(1.0), 0.9, 3).item(), 0.1
        )

    def test_adaptive_cap_zero_recovers_parent_and_preserves_raw_factors(self):
        torch.manual_seed(119)
        weights = [torch.nn.Parameter(torch.randn(5, 7)) for _ in range(3)]
        for parameter in weights[1:]:
            parameter.data.copy_(weights[0])
        optimizers = [
            make_matrix_optimizer(
                [weights[0]], "shampoo", block_size=3, update_frequency=2
            ),
            make_matrix_optimizer(
                [weights[1]],
                "shampoo_adaptive",
                block_size=3,
                update_frequency=2,
                alpha_max=0,
            ),
            make_matrix_optimizer(
                [weights[2]], "shampoo_adaptive", block_size=3, update_frequency=2
            ),
        ]
        for _ in range(5):
            gradient = torch.randn_like(weights[0]) * 1000
            for parameter, optimizer in zip(weights, optimizers, strict=True):
                parameter.grad = gradient.clone()
                optimizer.step()
            torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
            states = [
                optimizer.state[p]
                for optimizer, p in zip(optimizers, weights, strict=True)
            ]
            for parent, control, adaptive in zip(
                *(state["blocks"] for state in states), strict=True
            ):
                for side in ("left", "right"):
                    torch.testing.assert_close(
                        parent[side], control[side], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        parent[side], adaptive[side], rtol=0, atol=0
                    )
                    self.assertEqual(
                        adaptive[f"{side}_scatter"].shape, adaptive[side].shape[:-2]
                    )
                    self.assertTrue((adaptive[f"{side}_alpha"] <= 0.1).all())

    def test_adaptive_refresh_diagnostics_and_bfloat_checkpoint(self):
        torch.manual_seed(511)
        parameter = torch.nn.Parameter(torch.zeros(5, 7, dtype=torch.bfloat16))
        optimizer = make_matrix_optimizer(
            [parameter], "shampoo_adaptive", block_size=3, update_frequency=3
        )
        for step in range(1, 5):
            parameter.grad = torch.randn_like(parameter)
            optimizer.step()
            for block in optimizer.state[parameter]["blocks"]:
                self.assertEqual(block["last_refresh_step"], 1 if step < 4 else 4)
                for side in ("left", "right"):
                    if step < 4:
                        self.assertEqual(
                            block[f"{side}_alpha"].count_nonzero().item(), 0
                        )
                    self.assertEqual(block[f"{side}_scatter"].dtype, torch.float32)
        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored = make_matrix_optimizer(
            [restored_parameter], "shampoo_adaptive", block_size=3
        )
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for _ in range(3):
            parameter.grad = torch.randn_like(parameter)
            restored_parameter.grad = parameter.grad.clone()
            optimizer.step()
            restored.step()
            torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)
            for before, after in zip(
                optimizer.state[parameter]["blocks"],
                restored.state[restored_parameter]["blocks"],
                strict=True,
            ):
                self.assertEqual(
                    before["last_refresh_step"], after["last_refresh_step"]
                )
                for side in ("left", "right"):
                    for suffix in ("", "_root", "_scatter", "_alpha"):
                        torch.testing.assert_close(
                            before[side + suffix], after[side + suffix], rtol=0, atol=0
                        )


def distributed_check():
    """Real collectives: mixed shapes, partial ownership, SUM moments, resume."""
    torch.set_num_threads(1)
    cuda = "--nccl" in sys.argv
    cache_root = os.environ.get("BENCHMARK_CACHE_ROOT")
    if cache_root:
        # Set before the first kernel is compiled. Tests get a separate namespace
        # from the benchmark's exclusive rank directories in the same job.
        cache = Path(cache_root) / f"tests-rank-{os.environ['RANK']}"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    if cuda:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if cuda
        else torch.device("cpu")
    )
    for method in RESEARCH_METHODS:
        shapes = [(3, 5)] * (world + 1) + [(5, 3), (2, 2)]
        parameters = [
            torch.nn.Parameter(torch.full(shape, 0.2, device=device))
            for shape in shapes
        ]
        references = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
        method_settings = (
            {"update_frequency": 2} if method.startswith("shampoo") else {}
        )
        optimizer = make_matrix_optimizer(
            parameters, method, lr=0.01, **method_settings
        )
        local_topology = {"world_size": 1, "rank": 0}
        with mock.patch.object(
            MatrixResearchOptimizer,
            "_distributed_topology",
            return_value=local_topology,
        ):
            reference = make_matrix_optimizer(
                references, method, lr=0.01, **method_settings
            )
        expected_momenta = {p: torch.zeros_like(p) for p in parameters}
        beta = optimizer.param_groups[0].get("beta1", 0.95)
        for step in range(3):
            for index, (p, expected) in enumerate(
                zip(parameters, references, strict=True)
            ):
                grid = torch.arange(p.numel(), device=device).reshape(p.shape)
                p.grad = torch.sin(grid + index + step + rank) + 0.1 * (rank + 1)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                analytic_sum = sum(
                    torch.sin(grid + index + step + source) + 0.1 * (source + 1)
                    for source in range(world)
                )
                torch.testing.assert_close(
                    p.grad,
                    analytic_sum,
                    rtol=2e-6,
                    atol=2e-6,
                    msg=f"SUM reduction mismatch: {method}, step={step}, parameter={index}",
                )
                # NCCL and the independent sum legitimately use different FP32
                # addition orders. Test the optimizer with identical gradients
                # after testing SUM itself, especially for singular Shampoo.
                expected.grad = p.grad.clone()
                expected_momenta[p].lerp_(p.grad, 1 - beta)
            with mock.patch.object(
                MatrixResearchOptimizer,
                "_distributed_topology",
                return_value=local_topology,
            ):
                reference.step()
            optimizer.step()
            for index, (p, expected) in enumerate(
                zip(parameters, references, strict=True)
            ):
                torch.testing.assert_close(
                    p,
                    expected,
                    rtol=3e-5,
                    atol=3e-6,
                    msg=f"Optimizer mismatch: {method}, step={step}, parameter={index}",
                )
                if p in optimizer.state:
                    key = "momentum" if method == "soft_polar" else "exp_avg"
                    torch.testing.assert_close(
                        optimizer.state[p][key],
                        expected_momenta[p],
                        msg=f"SUM momentum mismatch: {method}, step={step}, parameter={index}",
                    )
                gathered = [torch.empty_like(p) for _ in range(world)]
                dist.all_gather(gathered, p)
                for other in gathered:
                    torch.testing.assert_close(p, other, rtol=0, atol=0)
            saved = copy.deepcopy(optimizer.state_dict())
            restored = make_matrix_optimizer(
                parameters, method, lr=0.01, **method_settings
            )
            restored.load_state_dict(saved)
            optimizer = restored
        if method == "shampoo_adaptive":
            any_alpha = torch.zeros((), dtype=torch.int32, device=device)
            for state in optimizer.state.values():
                for block in state["blocks"]:
                    if block["last_refresh_step"] != 3:
                        raise AssertionError(
                            "Adaptive distributed roots did not refresh at step 3"
                        )
                    for side in ("left", "right"):
                        any_alpha.copy_(
                            torch.maximum(
                                any_alpha,
                                (block[f"{side}_alpha"] > 0).any().to(torch.int32),
                            )
                        )
            dist.all_reduce(any_alpha, op=dist.ReduceOp.MAX)
            if any_alpha.item() != 1:
                raise AssertionError(
                    "Distributed test never applied nonzero adaptive shrinkage"
                )
    dist.barrier()
    if rank == 0:
        print(
            "Research optimizer sharding, SUM gradients, mixed-shape synchronization, and state restore passed.",
            flush=True,
        )
    dist.destroy_process_group()


def gpu_profile():
    """Production dimensions: validate spectra, then time 12 steps incl refresh.

    Run only in an allocated nano4 GPU job; output is machine-readable JSON.
    This is optimizer-only profiling, not a training speed or PPL claim.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu-profile requires an allocated CUDA GPU")
    cache_root = os.environ.get("BENCHMARK_CACHE_ROOT")
    if cache_root:
        cache = Path(cache_root) / "tests-profile"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    torch.set_num_threads(1)
    torch.manual_seed(91)
    device = torch.device("cuda")
    q, _ = torch.linalg.qr(torch.randn(768, 768, device=device))
    v, _ = torch.linalg.qr(torch.randn(768, 768, device=device))
    singular = torch.logspace(0, -6, 768, device=device)
    singular[384:] = 0
    base = (q * singular.unsqueeze(0)) @ v.mT
    normalized_singular = singular / singular.norm()
    checks = []
    for tau, iterations in ((0.1, 12), (0.01, 20)):
        filtered = (
            normalized_singular / (normalized_singular.square() + tau * tau).sqrt()
        )
        expected_base = (q * filtered.unsqueeze(0)) @ v.mT
        for shape in ((768, 768), (3072, 768), (768, 3072)):
            if shape == (768, 768):
                matrix, expected = base, expected_base
            elif shape == (3072, 768):
                matrix = torch.cat([base] * 4, dim=0) / 2
                expected = torch.cat([expected_base] * 4, dim=0) / 2
            else:
                matrix = torch.cat([base] * 4, dim=1) / 2
                expected = torch.cat([expected_base] * 4, dim=1) / 2
            actual = soft_polar_direction(matrix, tau=tau, iterations=iterations)
            relative_error = ((actual - expected).norm() / expected.norm()).item()
            if not torch.isfinite(actual).all() or relative_error > 0.005:
                raise AssertionError(
                    f"Production soft-polar error {relative_error} for {shape}, tau={tau}"
                )
            checks.append(
                {
                    "shape": shape,
                    "tau": tau,
                    "iterations": iterations,
                    "relative_error": relative_error,
                }
            )
    print(json.dumps({"spectral_checks": checks}), flush=True)
    for method in RESEARCH_METHODS:
        parameters = [
            torch.nn.Parameter(torch.zeros(shape, device=device))
            for shape in ((3072, 768), (768, 3072), (768, 768))
        ]
        optimizer = make_matrix_optimizer(parameters, method)
        for parameter in parameters:
            parameter.grad = torch.randn_like(parameter) * 1000
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(12):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        if any(not torch.isfinite(parameter).all() for parameter in parameters):
            raise AssertionError(f"Nonfinite {method} parameters during GPU profile")
        print(
            json.dumps(
                {
                    "method": method,
                    "three_matrix_step_seconds": times,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                    "includes_refresh_steps": [1, 11],
                }
            ),
            flush=True,
        )
        del parameters, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if "--distributed" in sys.argv:
        distributed_check()
    elif "--gpu-profile" in sys.argv:
        gpu_profile()
    else:
        unittest.main()
