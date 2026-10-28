"""Independent algebraic checks of TCD, including its failure mechanism."""

import copy
import itertools
import unittest

import torch

from exp.tcd import (
    TCD,
    crossfit_direction,
    gram_risk,
    precondition,
    precondition_first_order,
    risk_operator,
)
from exp.tcd_probe import rank_one_noise_control


class TCDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        torch.set_num_threads(2)

    def test_penalty_gradient_and_risk_bound(self):
        e = torch.randn(3, 5, dtype=torch.float64)
        d = torch.randn_like(e, requires_grad=True)
        (actual,) = torch.autograd.grad(gram_risk(e, d), d)
        torch.testing.assert_close(actual, risk_operator(e, d), atol=1e-12, rtol=1e-12)
        bound = (1 / 3 + 1 / 5) * (e * d).sum().square()
        self.assertGreaterEqual(gram_risk(e, d).item(), bound.item())

    def test_exact_solver_against_vectorized_linear_system(self):
        for m, n in ((2, 2), (2, 4), (4, 2)):
            for coupled in (False, True):
                e, g, mean = [torch.randn(m, n, dtype=torch.float64) for _ in range(3)]
                d, info = precondition(
                    g, e, mean, 7, coupled=coupled, normalize_trace=False
                )
                eye = torch.eye(m * n, dtype=e.dtype)
                k = torch.stack(
                    [
                        risk_operator(e, col.reshape(m, n), coupled).flatten()
                        for col in eye
                    ],
                    dim=1,
                )
                h = eye + info["gamma"] * k
                expected = torch.linalg.solve(h, g.flatten()).reshape(m, n)
                torch.testing.assert_close(d, expected, atol=2e-12, rtol=2e-12)
                self.assertAlmostEqual(
                    info["inverse_trace_mean"].item(),
                    torch.linalg.inv(h).trace().item() / (m * n),
                    places=12,
                )

    def test_trace_normalization_spd_and_bounds(self):
        e, mean = [torch.randn(3, 3, dtype=torch.float64) for _ in range(2)]
        eye = torch.eye(9, dtype=e.dtype)
        p = torch.stack(
            [precondition(col.reshape(3, 3), e, mean, 3)[0].flatten() for col in eye],
            dim=1,
        )
        torch.testing.assert_close(p, p.T, atol=1e-12, rtol=1e-12)
        eigen = torch.linalg.eigvalsh(p)
        self.assertGreaterEqual(eigen.min().item(), 1 / 4 - 1e-12)
        self.assertLessEqual(eigen.max().item(), 4 + 1e-12)
        self.assertAlmostEqual(p.trace().item(), 9, places=12)

    def test_orthogonal_and_transpose_equivariance(self):
        samples = torch.randn(4, 3, 5, dtype=torch.float64)
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        r, _ = torch.linalg.qr(torch.randn(5, 5, dtype=torch.float64))
        d = crossfit_direction(samples)
        torch.testing.assert_close(
            crossfit_direction(q @ samples @ r.T), q @ d @ r.T, atol=2e-12, rtol=2e-12
        )
        torch.testing.assert_close(
            crossfit_direction(samples.mT), d.T, atol=2e-12, rtol=2e-12
        )

    def test_direction_can_gain_rank_and_can_increase_true_loss(self):
        e = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
        g = torch.tensor([[0.0, -1.0], [0.0, 0.0]], dtype=torch.float64)
        d, _ = precondition(g, e, torch.zeros_like(e), 3)
        self.assertEqual(torch.linalg.matrix_rank(g).item(), 1)
        self.assertEqual(torch.linalg.matrix_rank(d).item(), 2)
        self.assertGreater((g * d).sum().item(), 0)
        w = -0.1 * d
        loss = 0.5 * (w[0, 1] - 1).square() + 5000 * w[1, 0].square()
        self.assertGreater(loss.item(), 0.5)

    def test_crossfit_expectation_exact_for_discrete_independent_samples(self):
        choices = torch.tensor(
            [[[2.0, 0.0], [1.0, -1.0]], [[-1.0, 2.0], [0.0, 3.0]]], dtype=torch.float64
        )
        mu = choices.mean(0)
        samples = torch.stack(
            [choices[list(ids)] for ids in itertools.product(range(2), repeat=4)]
        )
        observed = crossfit_direction(samples, 5).mean(0)
        expected = torch.stack(
            [
                precondition(mu, (a - b) / 2, (a + b) / 2, 5)[0]
                for a, b in itertools.product(choices, repeat=2)
            ]
        ).mean(0)
        torch.testing.assert_close(observed, expected, atol=1e-12, rtol=1e-12)
        self.assertGreaterEqual(
            (mu * observed).sum().item(), mu.square().sum().item() / 6
        )

    def test_zero_noise_and_zero_rho_recover_average(self):
        g = torch.randn(3, 2, dtype=torch.float64)
        torch.testing.assert_close(
            crossfit_direction(g.repeat(4, 1, 1)), g, atol=1e-12, rtol=1e-12
        )
        samples = torch.randn(4, 3, 2, dtype=torch.float64)
        torch.testing.assert_close(
            crossfit_direction(samples, 0), samples.mean(0), atol=0, rtol=0
        )

    def test_optimizer_resume_and_explicit_samples(self):
        p = torch.nn.Parameter(torch.randn(3, 2, dtype=torch.float64))
        opt = TCD([p], lr=0.03, momentum=0.9, weight_decay=0.2)
        with self.assertRaises(RuntimeError):
            opt.step()
        for _ in range(3):
            opt.set_samples({p: torch.randn(4, 3, 2, dtype=torch.float64)})
            opt.step()
        q = torch.nn.Parameter(p.detach().clone())
        restored = TCD([q])
        restored.load_state_dict(copy.deepcopy(opt.state_dict()))
        for _ in range(3):
            samples = torch.randn(4, 3, 2, dtype=torch.float64)
            opt.set_samples({p: samples})
            restored.set_samples({q: samples})
            opt.step()
            restored.step()
            torch.testing.assert_close(p, q, atol=0, rtol=0)

    def test_first_order_spectral_bound_trace_and_objective_descent(self):
        for m, n in ((2, 2), (2, 4), (4, 2)):
            e, g, mean = [torch.randn(m, n, dtype=torch.float64) for _ in range(3)]
            eye = torch.eye(m * n, dtype=e.dtype)
            normalized_e = e / e.norm()
            k = torch.stack(
                [
                    risk_operator(normalized_e, col.reshape(m, n)).flatten()
                    for col in eye
                ],
                dim=1,
            )
            d, info = precondition_first_order(g, e, mean, 30, normalize_trace=False)
            self.assertLessEqual(
                torch.linalg.eigvalsh(k).max().item(), info["operator_bound"].item()
            )
            q = eye - info["coefficient"] * k
            torch.testing.assert_close(d.flatten(), q @ g.flatten())
            self.assertAlmostEqual(
                q.trace().item() / (m * n), info["trace_mean"].item(), places=12
            )
            p = q / info["trace_mean"]
            eigen = torch.linalg.eigvalsh(p)
            self.assertGreaterEqual(eigen.min().item(), 1 / 31 - 1e-12)
            self.assertLessEqual(eigen.max().item(), 31 + 1e-12)
            gamma = 30 * info["gate"] / info["operator_bound"]
            before = gamma * gram_risk(normalized_e, g)
            after = 0.5 * (d - g).square().sum() + gamma * gram_risk(normalized_e, d)
            self.assertLessEqual(after.item(), before.item())

    def test_first_order_equivariance_zero_noise_and_uncoupled_trace(self):
        samples = torch.randn(4, 3, 5, dtype=torch.float64)
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        r, _ = torch.linalg.qr(torch.randn(5, 5, dtype=torch.float64))
        d = crossfit_direction(samples, solver="first_order")
        torch.testing.assert_close(
            crossfit_direction(q @ samples @ r.T, solver="first_order"),
            q @ d @ r.T,
            atol=2e-12,
            rtol=2e-12,
        )
        torch.testing.assert_close(
            crossfit_direction(samples.mT, solver="first_order"), d.T
        )
        g = samples[0]
        torch.testing.assert_close(
            crossfit_direction(g.repeat(4, 1, 1), solver="first_order"), g
        )
        e = samples[1]
        p = torch.stack(
            [
                precondition_first_order(col.reshape(3, 5), e, g, coupled=False)[0]
                .flatten()
                for col in torch.eye(15, dtype=g.dtype)
            ],
            dim=1,
        )
        self.assertAlmostEqual(p.trace().item(), 15, places=12)

    def test_first_order_crossfit_expectation_and_optimizer_dispatch(self):
        choices = torch.randn(2, 2, 3, dtype=torch.float64)
        samples = torch.stack(
            [choices[list(ids)] for ids in itertools.product(range(2), repeat=4)]
        )
        observed = crossfit_direction(samples, 5, solver="first_order").mean(0)
        expected = torch.stack(
            [
                precondition_first_order(choices.mean(0), (a - b) / 2, (a + b) / 2, 5)[0]
                for a, b in itertools.product(choices, repeat=2)
            ]
        ).mean(0)
        torch.testing.assert_close(observed, expected, atol=1e-12, rtol=1e-12)
        p = torch.nn.Parameter(torch.randn(2, 3, dtype=torch.float64))
        before = p.detach().clone()
        opt = TCD([p], lr=0.03, solver="first_order")
        opt.set_samples({p: samples[7]})
        opt.step()
        torch.testing.assert_close(
            p, before - 0.03 * crossfit_direction(samples[7], solver="first_order")
        )

    def test_rank_one_control_matches_full_covariance_solve(self):
        samples = torch.randn(4, 2, 3, dtype=torch.float64)
        a, b = samples[:2].mean(0), samples[2:].mean(0)
        ea, eb = (samples[0] - samples[1]) / 2, (samples[2] - samples[3]) / 2
        expected = torch.zeros_like(a)
        for g, e, mean in ((b, ea, a), (a, eb, b)):
            energy = e.square().sum()
            gate = energy / (energy + mean.square().sum())
            h = torch.eye(6, dtype=e.dtype) + 3 * gate * torch.outer(
                e.flatten(), e.flatten()
            ) / energy
            inverse = torch.linalg.inv(h)
            expected += (inverse @ g.flatten()).reshape_as(g) / (inverse.trace() / 6)
        torch.testing.assert_close(rank_one_noise_control(samples, 3), expected / 2)


if __name__ == "__main__":
    unittest.main()
