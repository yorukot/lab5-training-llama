"""Exact-reference and preregistration checks; run on the authorized CPU host."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from exp import adaptive_shrink_probe as probe
from exp import research_probe as base
from exp.research_optimizers import trace_shrink


class AdaptiveShrinkProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gaussian_factor_formula_with_exact_second_moment_cubature(self):
        dtype, dim = torch.float64, 4
        mean = torch.arange(16, dtype=dtype).reshape(dim, dim) / 10
        left = torch.diag(torch.tensor([0.2, 0.5, 1.1, 2.0], dtype=dtype))
        right = torch.diag(torch.tensor([1.0, 0.3, 0.8, 1.7], dtype=dtype))
        sigma = 0.4
        # +/- sqrt(mn) coordinate matrices have exactly zero mean and
        # identity covariance in vec(Z), so this quadratic expectation is exact.
        basis = torch.eye(dim**2, dtype=dtype).reshape(dim**2, dim, dim) * dim
        white = torch.cat([basis, -basis])
        samples = mean + sigma * probe._sqrt_psd(left) @ white @ probe._sqrt_psd(right)
        expected_left, expected_right = probe.gaussian_factor_truth(mean, left, right, sigma)
        torch.testing.assert_close((samples @ samples.mT).mean(0), expected_left, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close((samples.mT @ samples).mean(0), expected_right, rtol=1e-12, atol=1e-12)

    def test_scalar_oracle_matches_constrained_quadratic_minimum(self):
        estimate = torch.diag(torch.tensor([6.0, 1.0, 0.5], dtype=torch.float64))
        truth = torch.diag(torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64))
        for cap in (0.1, 1.0):
            alpha = probe.oracle_shrinkage(estimate, truth, cap)
            self.assertGreaterEqual(alpha, 0)
            self.assertLessEqual(alpha, cap)
            optimal = (trace_shrink(estimate, alpha) - truth).square().sum().item()
            for competitor in torch.linspace(0, cap, 101, dtype=torch.float64):
                error = (trace_shrink(estimate, competitor.item()) - truth).square().sum().item()
                self.assertLessEqual(optimal, error + 1e-12)
        self.assertEqual(probe.oracle_shrinkage(torch.eye(3), torch.eye(3)), 0)

    def test_stationary_noise_levels_share_problem_and_standardized_draws(self):
        low = probe.draw_optimization_problem("stationary_low_noise", 91, 8)
        high = probe.draw_optimization_problem("stationary_high_noise", 91, 8)
        torch.testing.assert_close(low.initial, high.initial, rtol=0, atol=0)
        torch.testing.assert_close(low.first_hessian, high.first_hessian, rtol=0, atol=0)
        torch.testing.assert_close(low.noise / 0.03, high.noise / 0.7, rtol=1e-12, atol=1e-12)
        deterministic = probe.draw_optimization_problem("deterministic_weak_signal", 91, 8)
        self.assertEqual(torch.count_nonzero(deterministic.noise).item(), 0)
        for case in probe.CASES:
            problem = probe.draw_optimization_problem(case, 91, 8)
            parameter = problem.initial.clone().requires_grad_(True)
            loss, analytic = base.population_loss_and_gradient(parameter, problem.hessian(7))
            actual, = torch.autograd.grad(loss, parameter)
            torch.testing.assert_close(actual, analytic, rtol=1e-12, atol=1e-12)

    def test_factor_stream_replay_and_deterministic_drift(self):
        stream = probe.draw_factor_stream("deterministic_drift", 7, 20)
        replay = probe.draw_factor_stream("deterministic_drift", 7, 20)
        torch.testing.assert_close(stream.gradients, replay.gradients, rtol=0, atol=0)
        torch.testing.assert_close(stream.gradients[:10], stream.gradients[0].expand(10, -1, -1))
        torch.testing.assert_close(stream.gradients[10:], stream.gradients[10].expand(10, -1, -1))
        self.assertFalse(torch.allclose(stream.left_truth[0], stream.left_truth[10]))
        torch.testing.assert_close(stream.gradients @ stream.gradients.mT, stream.left_truth)
        torch.testing.assert_close(stream.gradients.mT @ stream.gradients, stream.right_truth)

    def test_constant_stream_has_no_temporal_uncertainty_but_drift_does(self):
        constant = probe.run_factor_diagnostic("constant_weak_signal", "shampoo_adaptive", 7, 20)
        drift = probe.run_factor_diagnostic("deterministic_drift", "shampoo_adaptive", 7, 20)
        self.assertEqual(constant["status"], "completed")
        self.assertEqual(drift["status"], "completed")
        self.assertLess(max(row["alpha"] for row in constant["records"]), 1e-20)
        after = [row for row in drift["records"] if row["after_drift"]]
        self.assertGreater(max(row["alpha"] for row in after), 0)
        # All observations are deterministic, so the corrected raw EMA equals
        # its expectation despite a changed instantaneous population factor.
        for row in after:
            self.assertLess(row["expected_ema_truth"]["raw_squared_error"], 1e-25)
        self.assertGreater(max(row["current_truth"]["raw_squared_error"] for row in after), 0.01)

    def test_preregister_then_lock_and_preserve_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "probe"
            original = probe.run_trial

            def checked(case, method, lr, seed, steps):
                self.assertTrue((output / "protocol.json").exists())
                self.assertEqual((output / "locked_configurations.json").exists(), seed == 2002)
                return original(case, method, lr, seed, steps)

            with mock.patch.object(probe, "run_trial", side_effect=checked):
                summary = probe.run_study(
                    output, steps=2, seeds=(2002,), tuning_seeds=(11,),
                    lrs=(0.01,), cases=("stationary_low_noise",),
                    streams=("constant_weak_signal",), methods=("shampoo",),
                )
            self.assertEqual(summary["trial_counts"]["optimization_total"], 2)
            self.assertEqual(summary["trial_counts"]["factor_stream_total"], 4)
            self.assertEqual(summary["trial_counts"]["factor_stream_failed"], 0)
            self.assertEqual(summary["h200_gpu_hours"], 0)
            spec = json.loads((output / "protocol.json").read_text())
            self.assertNotIn("adaptive_cap1_diagnostic", spec["methods"])
            self.assertEqual(spec["stream_methods"]["adaptive_cap1_diagnostic"]["settings"]["alpha_max"], 1)
            with self.assertRaises(FileExistsError):
                probe.run_study(output)

    def test_failed_test_seed_disables_mean_and_seed_overlap_rejected(self):
        selected = {"selected_lr": 0.1, "lr_boundary": False}
        trials = [
            {"seed": 1001, "status": "completed", "final_loss": 0.1},
            {"seed": 2002, "status": "diverged", "final_loss": None},
        ]
        result = probe._test_summary(selected, trials, (1001, 2002))
        self.assertEqual(result["test_status"], "incomplete_or_failed")
        self.assertIsNone(result["mean_test_loss"])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            probe.protocol(seeds=(11,))

    def test_nonfinite_diagnostics_are_recorded_without_invalid_json(self):
        with (
            mock.patch.object(probe, "_applied_alpha", return_value=float("nan")),
            mock.patch.object(probe, "factor_metrics") as metrics,
        ):
            failed = probe.run_factor_diagnostic("constant_weak_signal", "shampoo", 7, 2)
        metrics.assert_not_called()
        self.assertEqual(failed["status"], "nonfinite_factor")
        self.assertEqual(failed["records"], [])
        json.dumps(failed, allow_nan=False)
        with mock.patch.object(probe, "factor_metrics", return_value={"error": float("inf")}):
            failed = probe.run_factor_diagnostic("constant_weak_signal", "shampoo", 7, 2)
        self.assertEqual(failed["status"], "nonfinite_diagnostic")
        self.assertEqual(failed["records"], [])
        json.dumps(failed, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
