"""Analytic and protocol tests; run on the authorized CPU host."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from exp import research_probe as probe


class ResearchProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_population_gradient_matches_autograd_in_all_regimes(self):
        for case in probe.CASES:
            problem = probe.draw_problem(case, 31, 8)
            for step in (0, 7):
                hessian = problem.hessian(step)
                parameter = problem.initial.clone().requires_grad_(True)
                loss, analytic = probe.population_loss_and_gradient(parameter, hessian)
                (actual,) = torch.autograd.grad(loss, parameter)
                torch.testing.assert_close(actual, analytic, rtol=1e-12, atol=1e-12)
                self.assertGreaterEqual(loss.item(), 0)
                self.assertGreater(torch.linalg.eigvalsh(hessian).min().item(), 0)

    def test_problem_draws_preserve_pairing_and_do_not_share_test_seeds(self):
        first = probe.draw_problem("rotated_noise", 101, 7)
        repeat = probe.draw_problem("rotated_noise", 101, 7)
        other = probe.draw_problem("rotated_noise", 202, 7)
        for name in ("initial", "noise", "first_hessian", "final_hessian"):
            torch.testing.assert_close(
                getattr(first, name), getattr(repeat, name), rtol=0, atol=0
            )
        self.assertFalse(torch.equal(first.noise, other.noise))
        aligned = probe.draw_problem("aligned_noise", 101, 7)
        torch.testing.assert_close(first.initial, aligned.initial, rtol=0, atol=0)
        torch.testing.assert_close(
            first.first_hessian, aligned.first_hessian, rtol=0, atol=0
        )
        torch.testing.assert_close(
            first.noise.flatten(1).norm(dim=1),
            aligned.noise.flatten(1).norm(dim=1),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertFalse(torch.equal(first.noise, aligned.noise))
        with self.assertRaises(ValueError):
            probe.protocol(seeds=(11,))
        with self.assertRaises(ValueError):
            probe.protocol(seeds=(101, 101))

    def test_weak_modes_have_small_gradients_but_meaningful_loss(self):
        problem = probe.draw_problem("weak_singular_signal", 1, 4)
        _, gradient = probe.population_loss_and_gradient(
            problem.initial, problem.first_hessian
        )
        self.assertLess(gradient.diagonal()[-1].abs().item(), 0.04)
        mode_losses = 0.5 * problem.initial.diagonal() * gradient.diagonal()
        self.assertGreater(mode_losses.min().item(), 0.4)
        changing = probe.draw_problem("abrupt_curvature_rotation", 1, 4)
        self.assertFalse(torch.allclose(changing.hessian(1), changing.hessian(2)))
        torch.testing.assert_close(
            torch.linalg.eigvalsh(changing.first_hessian),
            torch.linalg.eigvalsh(changing.final_hessian),
        )

    def test_selection_rejects_failures_and_reports_boundary(self):
        rows = [
            {"lr": lr, "seed": seed, "status": "completed", "final_loss": loss}
            for lr, loss in ((0.01, 2), (0.1, 1), (1.0, 0.1))
            for seed in (1, 2)
        ]
        rows[-1] |= {"status": "diverged", "final_loss": None}
        selected = probe.select_configuration(rows, (0.01, 0.1, 1.0), (1, 2))
        self.assertEqual(selected["selected_lr"], 0.1)
        self.assertFalse(selected["lr_boundary"])
        self.assertFalse(selected["grid"][-1]["eligible"])
        edge = probe.select_configuration(rows[:2], (0.01, 0.1), (1, 2))
        self.assertEqual(edge["selected_lr"], 0.01)
        self.assertTrue(edge["lr_boundary"])

    def test_protocol_precedes_trials_and_configuration_lock_precedes_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "probe"
            original = probe.run_trial

            def checked_trial(case, method, lr, seed, steps):
                self.assertTrue((output / "protocol.json").exists())
                if seed == 2:
                    self.assertTrue((output / "locked_configurations.json").exists())
                else:
                    self.assertFalse((output / "locked_configurations.json").exists())
                return original(case, method, lr, seed, steps)

            with mock.patch.object(probe, "run_trial", side_effect=checked_trial):
                summary = probe.run_study(
                    output,
                    steps=2,
                    seeds=(2,),
                    tuning_seeds=(1,),
                    methods=("sgd",),
                    cases=("aligned_noise",),
                    lrs=(0.01,),
                )
            self.assertEqual(summary["trial_counts"]["total"], 2)
            self.assertEqual(summary["trial_counts"]["failed"], 0)
            self.assertEqual(summary["h200_gpu_hours"], 0)
            stored = json.loads((output / "protocol.json").read_text())
            self.assertEqual(stored["test_seeds"], [2])
            with self.assertRaises(FileExistsError):
                probe.run_study(output, steps=2)

    def test_divergence_is_recorded_without_nonfinite_json(self):
        trial = probe.run_trial("aligned_noise", "sgd", 1e8, 1, 3)
        self.assertEqual(trial["status"], "diverged")
        self.assertIsNone(trial["final_loss"])
        json.dumps(trial, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
