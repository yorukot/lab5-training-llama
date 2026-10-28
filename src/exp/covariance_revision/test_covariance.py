"""Exact covariance, nullspace, objective and failure checks, independent of GPU."""

import itertools
import unittest

import torch

from exp.covariance_revision.candidate import fold_direction
from exp.covariance_revision.covariance import (
    covariance_direction,
    covariance_precondition,
    noise_factors,
)
from exp.tcd import gram_risk


class CovarianceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(2)

    def test_woodbury_against_full_inverse_and_objective(self):
        for shape in ((2, 3), (3, 2), (3, 3)):
            half = torch.randn(4, *shape, dtype=torch.float64)
            g = torch.randn(*shape, dtype=torch.float64)
            mean, b, var = noise_factors(half)
            strength = 3 * var / (var + mean.square().sum())
            h = torch.eye(g.numel(), dtype=g.dtype) + strength * b.T @ b
            expected = torch.linalg.solve(h, g.flatten()).reshape_as(g)
            actual = covariance_precondition(g, half, 3)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            d = actual.detach().requires_grad_()
            objective = .5 * (d-g).square().sum() + .5 * strength * (b@d.flatten()).square().sum()
            derivative, = torch.autograd.grad(objective, d)
            torch.testing.assert_close(derivative, torch.zeros_like(d), atol=1e-12, rtol=0)
            self.assertLessEqual(actual.norm().item(), g.norm().item()+1e-12)
            self.assertLessEqual((b@actual.flatten()).norm().item(), (b@g.flatten()).norm().item()+1e-12)
            eigen = torch.linalg.eigvalsh(torch.linalg.inv(h))
            self.assertGreaterEqual(eigen.min().item(), .25-1e-12)
            self.assertLessEqual(eigen.max().item(), 1+1e-12)

    def test_covariance_estimator_exact_discrete_expectation(self):
        choices = torch.randn(2, 2, 3, dtype=torch.float64)
        centered = (choices-choices.mean(0)).flatten(1)
        population = centered.T@centered/2
        estimates = []
        for indices in itertools.product(range(2), repeat=4):
            half = choices[list(indices)]
            _, b, var = noise_factors(half)
            estimates.append(var*b.T@b)
        torch.testing.assert_close(torch.stack(estimates).mean(0), population/4, atol=1e-12, rtol=1e-12)

    def test_identity_noise_preserves_symmetric_traceless_signal(self):
        g = torch.tensor([[1., .4], [.4, -1.]], dtype=torch.float64)
        eye = torch.eye(2, dtype=g.dtype)
        z = torch.tensor([-3., -1., 1., 3.], dtype=g.dtype)
        half = g+z[:, None, None]*eye
        d = covariance_precondition(g, half, 30)
        torch.testing.assert_close(d, g, rtol=1e-12, atol=1e-12)
        self.assertGreater(gram_risk(eye, g).item(), 0)
        samples = torch.cat([half, half])
        actual, _ = covariance_direction(samples, torch.zeros_like(g), 30, 0)
        control, _ = covariance_direction(samples, torch.zeros_like(g), 0, 0)
        torch.testing.assert_close(actual, control, rtol=1e-12, atol=1e-12)

    def test_zero_noise_and_zero_rho_control(self):
        g, old = torch.randn(2, 3, 4, dtype=torch.float64)
        torch.testing.assert_close(covariance_precondition(g, g.repeat(4,1,1)), g)
        rank_means = torch.randn(8, 3, 4, dtype=torch.float64)
        expected, m1 = fold_direction(rank_means.reshape(4,2,3,4).mean(1), old, 0)
        actual, m2 = covariance_direction(rank_means, old, 0)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(m1, m2, rtol=1e-12, atol=1e-12)
        zeros, _ = covariance_direction(torch.zeros_like(rank_means), torch.zeros_like(old))
        self.assertEqual(zeros.count_nonzero().item(), 0)

    def test_rectangular_orthogonal_equivariance(self):
        samples = torch.randn(8, 3, 5, dtype=torch.float64)
        old = torch.randn(3, 5, dtype=torch.float64)
        q, _ = torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
        r, _ = torch.linalg.qr(torch.randn(5,5,dtype=torch.float64))
        d, m = covariance_direction(samples, old)
        transformed, tm = covariance_direction(q@samples@r.T, q@old@r.T)
        torch.testing.assert_close(transformed, q@d@r.T, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(tm, q@m@r.T, rtol=1e-12, atol=1e-12)

    def test_contraction_still_has_high_curvature_failure(self):
        g = torch.tensor([[1.,0.]],dtype=torch.float64)
        noise = torch.tensor([[1.,1.]],dtype=torch.float64)
        half = torch.tensor([-3.,-1.,1.,3.],dtype=g.dtype)[:,None,None]*noise
        d = covariance_precondition(g,half,3)
        w = -.1*d
        after = .5*(w[0,0]+1).square()+5000*w[0,1].square()
        self.assertGreater(after.item(), .5)
        self.assertLess(d.norm().item(), g.norm().item())

    def test_rms_matching_does_not_match_spectral_step(self):
        rank_one = torch.zeros(8,8,dtype=torch.float64)
        rank_one[0,0] = 8**.5
        polar = torch.eye(8,dtype=torch.float64)
        self.assertEqual(rank_one.norm().item(), polar.norm().item())
        self.assertGreater(torch.linalg.matrix_norm(rank_one,ord=2).item(), torch.linalg.matrix_norm(polar,ord=2).item()*2)


if __name__ == "__main__":
    unittest.main()
