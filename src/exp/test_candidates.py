"""The optimizer candidates (note/optimizer-research/03-candidates.md): each reduces to NorMuon when
off, and does what its card says when on."""

import copy
import unittest

import torch
from transformers import AutoModelForCausalLM

import ema_nesterov
from exp.common import DEFAULT, model_config
from exp.optimizer import build_optimizer as build_old
from muon import muon_update
from optimizer import build_optimizer


def train(variant="normuon", ema=0.0, steps=5, total_steps=100, model=None):
    torch.manual_seed(0)
    model = model or AutoModelForCausalLM.from_config(model_config(tiny=True))
    opt = build_optimizer(
        model,
        "muon",
        2e-3,
        0.3,
        7e-3,
        0.05,
        7e-3,
        0.9873,
        variant,
        False,
        total_steps,
        ema,
    )
    torch.manual_seed(1)
    for _ in range(steps):
        x = torch.randint(3, 120, (2, 16))
        model.zero_grad()
        model(input_ids=x, labels=x).loss.backward()
        opt.step()
    return model, opt


class CandidateTests(unittest.TestCase):
    def test_lab_optimizer_matches_old_150m_normuon(self):
        torch.manual_seed(0)
        a = AutoModelForCausalLM.from_config(model_config(tiny=True))
        b = copy.deepcopy(a)
        cfg = DEFAULT | {"method": "normuon", "matrix_lr": 2e-3, "embed_lr": 7e-3}
        old = build_old(a, cfg, 0.3, 0.05, 0)
        _, new = train(model=b, steps=0)
        torch.manual_seed(1)
        for _ in range(5):
            x = torch.randint(3, 120, (2, 16))
            for m in (a, b):
                m.zero_grad()
                m(input_ids=x, labels=x).loss.backward()
            old.step()
            new.step()
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)

    def test_long_side_equals_normuon_except_on_wide_matrices(self):
        for shape, same in (((64, 32), True), ((32, 32), True), ((16, 64), False)):
            g = torch.randn(shape) * torch.rand(shape[-1]).add(0.1)
            a = muon_update(g, {}, "normuon")
            b = muon_update(g, {}, "normuon_long")
            self.assertEqual(torch.equal(a, b), same, shape)
            if not same:
                # Columns equalized: their norms spread much less than NorMuon's.
                self.assertLess(b.norm(dim=0).std(), 0.5 * a.norm(dim=0).std())

    def test_head_damping_off_at_start_and_damps_the_top_direction(self):
        g = torch.randn(48, 32) + 3 * torch.outer(torch.randn(48), torch.randn(32))
        self.assertTrue(
            torch.equal(
                muon_update(g, {}, "normuon"), muon_update(g, {}, "normuon_damped")
            )
        )
        damped_state, full_state = {}, {}
        for _ in range(20):  # the same history for both; the power iteration converges
            damped = muon_update(g, damped_state, "normuon_damped", progress=1.0)
            full = muon_update(g, full_state, "normuon")
        u, _, vh = torch.linalg.svd(g)  # the Nesterov input is a multiple of g here
        head = lambda x: u[:, 0] @ x @ vh[0]
        self.assertAlmostEqual((head(damped) / head(full)).item(), 1 / 5, delta=1e-3)

    def test_ema_nesterov_off_outside_window_and_gain_on_persistent_steps(self):
        a, _ = train(total_steps=1000)  # 5 steps: progress 0.005, before the window
        b, _ = train(ema=0.6, total_steps=1000)
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        # A constant step d inside the window: the iterate moves d / (1 - beta) per step.
        p, state, d, beta = torch.zeros(3), {}, torch.ones(3), 0.5
        for _ in range(3000):
            before = p.clone()
            p.add_(d)
            ema_nesterov.ema_nesterov_step(p, d, state, beta, beta)
        torch.testing.assert_close(p - before, d / (1 - beta), rtol=1e-4, atol=0)


if __name__ == "__main__":
    unittest.main()
