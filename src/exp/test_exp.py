"""CPU checks: PYTHONPATH=src python -m unittest exp.test_exp -v."""

import copy
import math
import unittest

import numpy as np
import torch

from exp.common import calibrated_decay, model_config, schedule
from exp.data import PackedCollator, budget_order, unpack_and_split
from exp.optimizer import ExperimentOptimizer
from muon import muon_update


def make_optimizer(p, method="hg_normuon", **kwargs):
    return ExperimentOptimizer(
        [{"params": [p], "kind": "matrix", "lr": 0.01, "weight_decay": 0.1}],
        method=method,
        **kwargs,
    )


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_chain_rule_with_decay(self):
        w = torch.tensor([[0.3, -0.2], [0.1, 0.25]], dtype=torch.float64)
        u = torch.tensor([[0.7, -0.1], [0.2, 0.9]], dtype=torch.float64)
        curvature = torch.tensor([[1.0, 4.0], [2.0, 3.0]], dtype=torch.float64)
        eta, wd, a, epsilon = 0.07, 0.2, 1.3, 1e-6
        v = eta * u

        def phi(factor):
            return 0.5 * (curvature * ((1 - eta * wd) * w - factor * v).square()).sum()

        h = -(curvature * ((1 - eta * wd) * w - a * v) * v).sum()
        fd = (phi(a + epsilon) - phi(a - epsilon)) / (2 * epsilon)
        self.assertLess(abs((h - fd).item()), 1e-9)

    def test_zero_meta_lr_is_original_normuon(self):
        a = torch.randn(5, 3)
        b = torch.nn.Parameter(a.clone())
        base = {}
        optimizer = make_optimizer(b, meta_lr=0.0)
        for _ in range(7):
            grad = torch.randn_like(a)
            update = muon_update(grad, base, True, momentum=0.95)
            a.mul_(0.999).add_(update, alpha=-0.01)
            b.grad = grad.clone()
            optimizer.observe_unclipped_gradients()
            optimizer.step()
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_observation_before_clipping_and_bounds(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        optimizer = make_optimizer(p, meta_lr=10.0)
        optimizer.state[p]["previous_base_step"] = torch.ones_like(p)
        p.grad = torch.ones_like(p) * 100
        optimizer.observe_unclipped_gradients()
        self.assertEqual(optimizer._pending[p].item(), -400.0)
        torch.nn.utils.clip_grad_norm_([p], 1.0)
        optimizer.step()
        self.assertEqual(optimizer.state[p]["meta"]["factor"].item(), 2.0)
        p.grad = -torch.ones_like(p)
        optimizer.state[p]["previous_base_step"] = torch.ones_like(p) * 10000
        optimizer.observe_unclipped_gradients()
        optimizer.step()
        self.assertEqual(optimizer.state[p]["meta"]["factor"].item(), 0.5)

    def test_resume_both_modes(self):
        for method in ("hg_normuon", "hg_normuon_global"):
            p = torch.nn.Parameter(torch.randn(3, 4))
            optimizer = make_optimizer(p, method)
            for _ in range(3):
                p.grad = torch.randn_like(p)
                optimizer.observe_unclipped_gradients()
                optimizer.step()
            q = torch.nn.Parameter(p.detach().clone())
            restored = make_optimizer(q, method)
            restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            for _ in range(4):
                grad = torch.randn_like(p)
                for param, opt in ((p, optimizer), (q, restored)):
                    param.grad = grad.clone()
                    opt.observe_unclipped_gradients()
                    opt.step()
                torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_missing_grad_zero_lr_and_warmup(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        optimizer = make_optimizer(p, warmup_steps=2)
        for _ in range(2):
            p.grad = torch.ones_like(p)
            optimizer.observe_unclipped_gradients()
            optimizer.step()
            self.assertEqual(optimizer.state[p]["meta"]["factor"].item(), 1.0)
            self.assertEqual(optimizer.state[p]["meta"]["sum_sq"].item(), 0.0)
        p.grad = None
        optimizer.observe_unclipped_gradients()
        optimizer.step()
        self.assertNotIn("previous_base_step", optimizer.state[p])
        optimizer.param_groups[0]["lr"] = 0.0
        p.grad = torch.ones_like(p)
        optimizer.observe_unclipped_gradients()
        optimizer.step()
        self.assertNotIn("previous_base_step", optimizer.state[p])

    def test_unobserved_and_nonfinite_fail(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        opt = make_optimizer(p)
        p.grad = torch.ones_like(p)
        with self.assertRaises(RuntimeError):
            opt.step()
        opt.state[p]["previous_base_step"] = torch.ones_like(p)
        p.grad.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            opt.observe_unclipped_gradients()

    def test_global_uses_sum_not_separate_adaptation(self):
        p, q = [torch.nn.Parameter(torch.ones(2, 2)) for _ in range(2)]
        opt = ExperimentOptimizer(
            [{"params": [p, q], "kind": "matrix"}], method="hg_normuon_global"
        )
        for param, sign in ((p, 1.0), (q, -1.0)):
            param.grad = torch.ones_like(param) * sign
            opt.state[param]["previous_base_step"] = torch.ones_like(param)
        opt.observe_unclipped_gradients()
        opt.step()
        self.assertEqual(opt.state[p]["global_meta"]["factor"].item(), 1.0)


class DataTests(unittest.TestCase):
    def test_masking_and_attention(self):
        examples = [{"input_ids": [1, 2, 3, 1, 4], "doc_lens": [3, 2]}]
        batch = PackedCollator(0, 8, varlen=False)(examples)
        self.assertEqual(
            batch["labels"].tolist(), [[-100, 2, 3, -100, 4, -100, -100, -100]]
        )
        self.assertTrue(torch.isneginf(batch["attention_mask"][0, 0, 4, 0]))
        self.assertEqual(batch["attention_mask"][0, 0, 4, 3].item(), 0.0)
        varlen = PackedCollator(0, 8)(examples)
        self.assertEqual(varlen["cu_seq_lens_q"].tolist(), [0, 3, 5, 8])

    def test_duplicates_do_not_cross_eval_splits(self):
        docs = [[1, i + 2] for i in range(20)]
        rows = [{"input_ids": d + d, "doc_lens": [2, 2]} for d in docs]
        dev, confirm, _ = unpack_and_split(rows)
        self.assertFalse(set(map(tuple, dev)) & set(map(tuple, confirm)))
        self.assertEqual(len(dev) + len(confirm), 40)

    def test_budget_order(self):
        counts = np.arange(10, 110)
        order, prefix = budget_order(counts, 42, 1000, 4)
        self.assertEqual(len(order) % 4, 0)
        self.assertLessEqual(prefix[-1], 1000)
        self.assertEqual(prefix[-1], counts[order].sum())
        self.assertTrue(np.array_equal(order, budget_order(counts, 42, 1000, 4)[0]))

    def test_decay_matches_requested_product(self):
        values, _ = schedule(2300, 0.01, 0.8)
        for lr in (1e-4, 1e-2):
            wd = calibrated_decay(lr, values, 1.5)
            actual = -sum(math.log1p(-lr * value * wd) for value in values)
            self.assertAlmostEqual(actual, 1.5, places=12)

    def test_model_size_and_tying(self):
        from transformers import LlamaForCausalLM

        with torch.device("meta"):
            model = LlamaForCausalLM(model_config())
        self.assertEqual(sum(p.numel() for p in model.parameters()), 149_314_048)
        self.assertIs(model.model.embed_tokens.weight, model.lm_head.weight)


if __name__ == "__main__":
    unittest.main()
