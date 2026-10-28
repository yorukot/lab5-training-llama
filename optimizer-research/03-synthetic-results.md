# Synthetic mechanism study — 2026-10-16

Executed on nano4 CPU after six probe tests passed. This is a controlled quadratic study, not a language-model result.

The preregistered study ran 260 trials with 4×4 parameters, 64 optimizer steps, paired initialization/noise, tuning seeds 11/22, and separate evaluation seeds 101/202/303. Every method used the same five learning rates {0.003, 0.01, 0.03, 0.1, 0.3}; only learning rate was selected. All selections were frozen before evaluation seeds ran. Weight decay was zero and there was no learning-rate schedule.

The reported quantity below is mean final population **quadratic loss** across three evaluation seeds; smaller is better. An asterisk means the selected learning rate is at a search boundary, so tuning is unresolved.

| Regime | SGD | Matrix AdamW | Shampoo | Shrunk Shampoo | Soft-polar |
|---|---:|---:|---:|---:|---:|
| abrupt curvature rotation | 0.013270 | 0.151962* | 0.093331 | 0.081102 | 1.176097* |
| aligned noise | 0.019110 | 0.053232 | 0.052019 | 0.052406 | 1.101944* |
| rotated noise | 0.024404 | 0.111710* | 0.062116 | 0.058241 | 1.193676* |
| weak singular signal | 0.782371* | 0.075036* | 0.054144* | 0.095286* | 0.122620* |

## Interpretation

- Shrinkage is not uniformly helpful. Its parent comparison must be checked across regimes rather than inferred from one favorable example.
- abrupt curvature rotation: shrinkage changes mean loss by -13.10% relative to its identical zero-shrinkage parent; selected learning rates are 0.1 and 0.1.
- aligned noise: shrinkage changes mean loss by +0.74% relative to its identical zero-shrinkage parent; selected learning rates are 0.1 and 0.1.
- rotated noise: shrinkage changes mean loss by -6.24% relative to its identical zero-shrinkage parent; selected learning rates are 0.1 and 0.1.
- weak singular signal: shrinkage changes mean loss by +75.99% relative to its identical zero-shrinkage parent; selected learning rates are 0.3 and 0.3.

The weak-signal case deliberately gives small gradient singular directions meaningful remaining loss. The adverse shrinkage result is evidence against interpreting regularization as universally beneficial. Soft-polar also has unresolved learning-rate boundaries, so its current scores do not establish the best attainable performance of its published method family.

There were 248 completed trials and 12 recorded divergences. All failures remain in the trial records; an LR with a failed tuning seed was ineligible. The three rotating/anisotropic cases have maximum Hessian eigenvalue 30, so plain gradient descent has the familiar deterministic stability restriction eta < 2/30; the larger SGD grid values intentionally test that limit. These small problems and fixed defaults do not establish robust optimizer rankings, GPU feasibility, or lower held-out PPL.

The current evidence supports preserving the trace-shrinkage hypothesis for guarded GPU screening while retaining its weak-signal failure case. GPU promotion still requires production-shape numerical checks, distributed CUDA verification, measured cost, and the fixed-token LM development experiment.

## Reproduction and artifacts

```bash
PYTHONPATH=src /home/u9603854/lab5-training-llama/.venv/bin/python \
  -m exp.research_probe --output /path/to/fresh-probe
```

- [Frozen protocol](../outputs/gpt2-research-20261016-v1/synthetic-probe/protocol.json)
- [Locked configurations](../outputs/gpt2-research-20261016-v1/synthetic-probe/locked_configurations.json)
- [Summary and individual evaluation results](../outputs/gpt2-research-20261016-v1/synthetic-probe/summary.json)
- [Probe test log](../outputs/gpt2-research-20261016-v1/logs/probe-tests.log)

GPU cost: **0 H200-hours**. The recorded study-loop time is 1.858 CPU wall seconds, excluding interpreter/Torch startup.
