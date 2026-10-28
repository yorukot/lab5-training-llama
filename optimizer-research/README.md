# Optimizer research execution

This directory contains the research notes. Existing
[`01-foundations.md`](../note/optimizer-research/01-foundations.md) and
[`02-math.md`](../note/optimizer-research/02-math.md) are preserved in their earlier
directory as Llama-focused notes. The current GPT-2
research and its qualifications are in
[02-mathematics-and-hypotheses.md](02-mathematics-and-hypotheses.md).
The executed CPU mechanism study is documented in
[03-synthetic-results.md](03-synthetic-results.md).
The adaptive continuation is specified in
[04-adaptive-shrinkage.md](04-adaptive-shrinkage.md), with executed evidence in
[05-adaptive-results.md](05-adaptive-results.md).

**Latest result:** the adaptive version was not promoted. In 15 equal-token
1,000-step development runs, best tested PPL was 40.19570 for adaptive, 39.31948
for its parent, and 33.65999 for Muon. The selected LR is bracketed for every
method. Total charged cost is 8.46222 of 64 H200 GPU-hours; no reservations or
research jobs remain active. The reserved slice and confirmation seeds were
not used, and the 0.5 PPL improvement goal has not been achieved.

## Objective and boundaries

The current objective is an absolute **0.5 PPL improvement at fixed training
tokens**, not a fixed-time improvement. The main benchmark is the pinned
162,354,816-parameter GPT-2-derived model: 3,250 steps, 524,288 target tokens per
step, 1,703,936,000 total training targets, context 1,024. Architecture, data,
auxiliary AdamW, precision, global batch, and cooldown stay fixed. Runtime,
allocated GPU-hours, failures, and peak memory are reported separately.

The first phase is capped at **64 allocated H200 GPU-hours**. Five new paired
seeds (101, 202, 303, 404, 505) are reserved for a locked comparison if a candidate
earns promotion. The mean paired PPL gain must be at least 0.5 and the lower end
of the prespecified 95% paired t interval must exceed zero. No candidate is
assumed to pass. Llama experiments require a later costed protocol and are not
part of this phase's execution.

The [versioned protocol](../src/exp/gpt2_research_protocol.json) supersedes the
old fixed-cost primary criterion without rewriting historical records. Its
validation offsets index the shifted next-token target stream: input begins at
`offset`, the first target is shard index `offset + 1`. Development uses offset
0; the reserved slice uses offset 80,000,000; both contain 10,485,760 targets.
Training rejects the reserved slice. Standalone reserved evaluation requires a
locked, identity-matched confirmation manifest before a forward pass.

## Implemented methods and evidence

- `muon`: the preserved upstream anchor, with explicit LR/decay controls.
- `matrix_adamw`: hidden-matrix AdamW; the auxiliary optimizer stays unchanged.
- `shampoo`: a declared blocked EMA, bias-corrected, Adam-grafted variant.
- `shampoo_shrink`: the same variant, with trace-preserving factor shrinkage only
  at inverse-root computation. Zero shrinkage must reproduce its parent.
- `shampoo_adaptive`: estimates temporal uncertainty of each factor's trace-free
  component using scalar weighted Welford statistics, and caps the resulting
  shrinkage coefficient at 0.1. It retains the parent's raw factors, refresh
  schedule and Adam graft. Temporal drift can produce harmful shrinkage even
  without stochastic noise; the derivation and experiments retain this failure.
- `soft_polar`: a SoftMuon-family comparison, with a bounded regularized polar
  map and an explicit finite-iteration accuracy guard. This map is published
  prior art, not an original invention. RMS matching separates update direction
  from its overall scale.
- Existing TCD1 and its zero-risk control remain available, but are deprioritized
  after the [historical remote screen](../outputs/gpt2-research-20261016-v1/prior-tcd-screen.json).
  At 1,000 steps/seed 42, the best tested TCD1 PPL was 84.90 versus Muon's 33.66.
  Both TCD and its folding/RMS control performed poorly, and their best LR was
  at the low boundary; these results do not isolate geometry as the only cause.

The [registered initial search](../src/exp/research_screen.json) has 18 cases
and five equal family caps of 5.6 GPU-hours within the 28-hour screen budget.
These are budget caps, not a reason to continue a failed or infeasible method.
Short runs cannot establish full-horizon superiority. Additional search choices
must be recorded before their metrics are inspected.
The [adaptive amendment](../src/exp/research_adaptive_screen.json) registers a
four-method initial comparison and a conditional LR expansion. Fixed and
adaptive shrinkage share the existing 5.6 GPU-hour family cap; the 64-hour total
and 28-hour screening cap are unchanged.

## Reproducible execution on nano4

All verification and GPU work runs through `ssh nano4`, using the existing
Python environment and Slurm setup. The workstation's system SSH config has an
ownership error, so this session uses `ssh -F ~/.ssh/config nano4` to read the
same host entry directly.

Remote study:

```
/work/u9603854/lab5-training-llama/experiments/gpt2-research-20261016-v1
```

The study reuses verified public FineWeb shards through a symlink. Source,
configuration, protocol, and dataset identities are recorded; use a new source
snapshot for changes after a training job has started. Do not modify the
vendored benchmark, overwrite outputs, or restart retired 150M campaigns.

From its source directory on nano4:

```bash
export PYTHONPATH="$PWD/src"
research_python=/home/u9603854/lab5-training-llama/.venv/bin/python
research_root=/work/u9603854/lab5-training-llama/experiments/gpt2-research-20261016-v1

"$research_python" -m exp.research_study show --study "$research_root/budget.json"
"$research_python" -m exp.research_submit submit \
  --study "$research_root" --source "$PWD" \
  --run-id UNIQUE_RUN_ID --stage screening \
  --config "$research_root/configs/CONFIG.json" --minutes 10
"$research_python" -m exp.research_submit reconcile --study "$research_root"
```

The launcher reserves requested GPUs times the full Slurm wall limit **before**
submission. It passes immutable config/protocol snapshots, explicitly requests
one node/eight GPUs, disables requeue, and uses fresh per-rank compiler caches.
Reconciliation charges actual allocation GPU count and elapsed time, including
failed jobs. An ambiguous submission error retains its reservation until
scheduler accounting establishes what happened.

The evaluator exports final model weights, not a resumable optimizer checkpoint.
For confirmation, lock one method configuration per side, the five seeds, data,
source hashes, evaluator runtime/settings, and training budget; then evaluate
the reserved slice with that manifest. `exp.research_report` checks every pair
and refuses a confirmation claim for missing, duplicated, or mismatched runs.

## Verification status

The reviewed suite passed **96 tests on nano4** on 2026-10-16 (90 infrastructure
and optimizer tests plus six synthetic-study tests). It covers
optimizer numerical references and controls, state continuation, data/source
integrity, slice boundaries, checkpoint evaluation, locked confirmation,
statistical criteria, concurrent budget reservations, and submission-timeout
recovery. Final lint and test logs are stored under the remote study's `logs/`.
GPU numerical, synchronization, runtime, and full training evidence are recorded
separately as they complete; unit tests alone are not evidence of better PPL.
Real two-rank CPU/Gloo collectives also passed, and all 16 new/updated Python
files passed Ruff lint and format checks. This was the initial CPU verification;
the later adaptive GPU checks are recorded below.

The [260-trial synthetic study](03-synthetic-results.md) completed on CPU with
zero H200 usage. Factor shrinkage improved two regimes, was slightly worse in
aligned noise, and substantially worse on weak useful singular signal. Learning
rate boundaries and the tiny designed problems limit interpretation; these
results cannot substitute for fixed-token LM comparisons.

### Scheduler interruption and recovery on 2026-10-16

The initial verification submission returned an RPC timeout without a job ID.
A subsequent scheduler-only preflight also timed out; `scontrol ping` reported
primary controller `25a-rms01` DOWN and backup controllers UP. `sacct` returned
no record named `opt-verify-r1`, but live queue verification also timed out.

At the end of that interrupted attempt, the ledger recorded
zero settled research GPU-hours and a conservative **2.6667 GPU-hour accounting
hold** (`verify-r1-accounting-hold`) for the ambiguous submission. This hold is
not evidence that GPUs ran. The source directory used for that attempt is
preserved; the corrected and formatted implementation is in `source-r2`.

The scheduler recovered during the adaptive continuation. Both live queue and
accounting reads showed no `opt-verify-r1` allocation, so the hold was released.
The first retried GPU test exposed a test-reference issue: NCCL and sequential
FP32 sums can differ in rounding. The corrected test checks the SUM reduction
separately, then gives identical reduced gradients to both update paths without
loosening parameter tolerances. Its failed allocation cost was retained.

The frozen adaptive source is `source-adaptive-r1`. Its 98-test suite, real
two-rank CPU/Gloo check, eight-rank GPU/NCCL check, production-shape GPU profile,
two-step model smoke, and standalone checkpoint evaluation passed. Distributed
coverage explicitly refreshes and applies nonzero adaptive shrinkage. The
180-trial/60-stream synthetic study had no numerical failures but mixed quality
results. See [05-adaptive-results.md](05-adaptive-results.md) for the latest LM
outcomes and settled budget; old outage artifacts remain historical evidence.
