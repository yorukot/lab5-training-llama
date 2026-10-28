# Optimizer research experiments

## Active benchmark: fixed-architecture Modded-NanoGPT (2026-10-16)

The user retired the custom 150M Llama experiment and selected the
[Modded-NanoGPT Optimization Benchmark](https://github.com/KellerJordan/modded-nanogpt/tree/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_3_optimization).
It is a GPT-2-derived model with **162,354,816 actual parameters**, not original
GPT-2 124M. The previous HG and LR jobs and their dependent controllers were
cancelled. Their results/checkpoints remain; each remote study has a
STOPPED_BY_USER.json record. **Do not restart either retired campaign.**
The 150M sections below are historical research records, not active instructions.

The active specification is [gpt2_protocol.json](gpt2_protocol.json). The user
wants the highest quality per unit training cost, through either substantially
faster training at comparable quality or substantially better quality at
comparable speed. The specified quality threshold is an **absolute PPL reduction
of at least 0.5** against a well-tuned baseline. Numerical speed/noninferiority
thresholds have not been selected; the suggested 25% speed / 0.1 PPL thresholds
are not recorded as user-approved requirements.

We report the quality/cost frontier, not PPL divided by seconds: both smaller PPL
and smaller time are desirable, so that naive ratio can reward a slower method.
Compare:

- PPL at matched end-to-end GPU-hours, with training-token counts shown.
- GPU-hours and wall time to the same target PPL, using observed evaluation
  points and crossing intervals rather than fabricated interpolation.
- Fixed-token PPL and optimizer/training overhead as secondary diagnostics.
- Total tuning GPU-hours, failures, peak memory, compilation-cache condition,
  and Slurm allocation cost. Report training-only time separately.

Hold tokenizer, validation slice, architecture, batch size, precision policy,
hardware and timing scope fixed across comparisons. A reduction of 0.5 PPL is
not 0.5 cross-entropy: the published reference mean NLL 3.2787 corresponds to
PPL about 26.5412; a 0.5 reduction there means PPL 26.0412 / NLL 3.25968.
These numbers illustrate the threshold, not a score we have achieved.

### Validated anchor and reproduction

Start from the benchmark's **tuned Muon + auxiliary AdamW, result #36**, reported
at mean NLL 3.2787 over 10 trials at 3250 steps. This is a validated anchor, not a
claim of the best available optimizer or globally optimal hyperparameters.
After reproducing it locally, candidate evaluation must include strong applicable
published competitors and comparable tuning budgets. A weak or explicitly
undertuned Adam entry is insufficient evidence of a new optimizer improvement.

The upstream source and license are vendored byte-for-byte at commit
4ea6b937337a4889b8cfe3f38a93d120048d8f71; source_manifest.json records each SHA256.
Full reproduction uses the unmodified upstream script. The wrapper adds seeds,
hash checks, metadata, W&B and elapsed-time accounting. A smoke run changes only
training/evaluation budgets and is explicitly ineligible for comparison.

| Setting | Published reference |
|---|---|
| Model | 12 layers, width 768, vocab 50304, untied head; fixed upstream architecture |
| Data | GPT-2-tokenized FineWeb10B, public pretokenized shards |
| Context / global batch | 1024 / 524,288 target tokens per step |
| Training | 3250 steps = 1,703,936,000 target tokens |
| Validation | First 10,485,760 targets from the public validation shard |
| Matrix optimizer | Muon LR .025, decay .05, momentum .95, 12 NS iterations |
| Auxiliary AdamW | Embed/head/scalar LR .7/.004/.015; betas .8/.95; eps 1e-10; decay .001 |
| Schedule | Stable then linear cooldown over the final 70%; no warmup |
| Gradient convention | Sum, exactly as upstream; do not silently average |

The existing nano4 GPT-2 project uses C4 and currently adds teacher distillation;
its recipe is a different objective and cannot reproduce this FineWeb reference.
The benchmark instead downloads public pretokenized shards, pins the dataset
revision and hashes each shard. There is no retokenization or pretrained-weight
loading. Prepared data is shared across runs.

The local environment is PyTorch 2.14/CUDA 13 on H200, whereas the public recipe
currently recommends PyTorch 2.11. Preserving its hyperparameters does not by
itself prove numerical equivalence across software/hardware. Reproduction must
be measured. Use the same local software for all future comparisons.

### Execution and interpretation

[gpt2_benchmark.py](gpt2_benchmark.py) executes the pinned recipe and saves
metadata.json, effective_source.py, history.json and result.json. W&B project
remains **lab-training-llama**, group gpt2-optimizer-reference-v1.
The wrapper records both upstream training time and elapsed runner time including
Torch import, initialization, compile, data validation/loading, training,
validation and logging. Python startup and Slurm launch are covered separately
by allocation accounting. Shared initial data download is separately amortized.
The Slurm script uses fresh node-local compiler caches, isolated per rank, with
two compiler workers per rank. An unreported warm cache must not manufacture a
speedup; warm-cache measurements can be added separately. The first smoke job
(556810) failed during compiled-kernel loading with CUDA_ERROR_FILE_NOT_FOUND
while ranks shared a cache on /work. Its dependent full job (556819) was cancelled
without running. With cache isolation, smoke job **556838 completed** on eight
H200 GPUs (two steps; 162,354,816 parameters; 93 allocation seconds = 0.2067
GPU-hours). This verifies execution, not quality. The first failed attempt cost
another 0.2133 allocation GPU-hours. The workaround does not change the training
recipe and does not by itself prove the precise root cause.

Full reference job **556842 completed** on nano4, seed 42, 3250 steps,
1,703,936,000 training tokens, with validation NLL **3.27735 / PPL 26.50544**.
The first observed NLL below 3.28 was at step 3200. Training-only time was
486.195 seconds, runner elapsed time 554.794 seconds (1.23287 GPU-hours), and
Slurm allocation time 566 seconds (1.25778 GPU-hours), on eight H200 GPUs.
Including both smoke attempts, the allocation cost was 1.67778 GPU-hours;
initial shared data preparation is excluded. This single seed meets the public
target and is close to its reported mean, but does not establish multi-seed
equivalence or a new optimizer improvement. No candidate comparison has run
on this benchmark yet. Against this seed alone, a 0.5 PPL reduction would require
PPL <= 26.00544 at the matched cost; the final criterion must use tuned baselines
and paired-seed confirmation.
[W&B baseline](https://wandb.ai/cerulean-labs/lab-training-llama/runs/w7ztb32j).
Use Slurm plus result.json to determine terminal status. A rank-zero result
left at `running` is not evidence of a live job when
another distributed rank has failed. The v1 execution_audit.json records this
case for the failed first smoke attempt.

The public validation slice is suitable for recipe reproduction and development,
but is not historically blind. Reserve a distinct evaluation slice before
selecting new candidates. Lock baseline/candidate hyperparameters on development
data, then run paired fresh-seed confirmation. Neither one successful baseline
reproduction nor one favorable seed establishes optimizer superiority.

Frozen first-attempt source/data/logs:
/work/u9603854/lab5-training-llama/experiments/gpt2-opt-20261016-v1.
Cache-isolated reproduction:
/work/u9603854/lab5-training-llama/experiments/gpt2-opt-20261016-v2,
reusing the verified v1 data through a symlink.

~~~bash
PYTHONPATH=src .venv/bin/python -m unittest exp.test_gpt2_benchmark -v
PYTHONPATH=src python -m exp.gpt2_benchmark prepare \
  --data-dir /path/to/data/fineweb10B --shards 20
torchrun --standalone --nproc-per-node=8 -m exp.gpt2_benchmark run \
  --data-dir /path/to/data/fineweb10B --output /path/to/fresh/run --seed 42
~~~

Six checks cover immutable source verification, actual model size and tying,
shard integrity, smoke-patch scope, separate timing clocks, and honest
time-to-target censoring. The full reference has no checkpoint/resume facility;
interrupted reference runs remain incomplete and cannot count as a completed
budget. Preserve frozen source snapshots and use a new output directory per run.

## Active candidate: TCD1 on the fixed GPT-2 benchmark

The candidate is implemented in [gpt2_candidate.py](gpt2_candidate.py). This is
an experimental extension of the TCD1 operator derived later in this file, not
a demonstrated improvement. The adverse symmetric-signal and curvature examples
remain valid warnings; adding momentum and scaling does not resolve them.
There is no Muon, polar decomposition, SVD or Newton--Schulz step in its matrix
update. Embeddings, output head and scalar parameters retain reference AdamW.

### Exact rule and what the theory does (and does not) cover

Eight GPUs each produce the original sum gradient over 65,536 target tokens.
Pair adjacent ranks and divide each pair sum by 131,072 to form four mean
gradients g1,...,g4. Each token is used once, with the same parameter values.
Define a=(g1+g2)/2, b=(g3+g4)/2, eA=(g1-g2)/2, eB=(g3-g4)/2.
With historical momentum M and beta=.95, define

\[
 N_A=\beta^2M+(1-\beta^2)a,\quad
 N_B=\beta^2M+(1-\beta^2)b,\quad
 M^+=\beta M+(1-\beta)(a+b)/2.
\]

This recovers the reference's EMA-Nesterov convention before preconditioning.
Let P_A be the trace-compensated TCD1 operator from (eA,a), P_B from (eB,b),
and s_A=max(RMS(N_A),1e-8), s_B=max(RMS(N_B),1e-8). For fan-in n, use

\[
 D=\frac{1}{2\sqrt n}\left(P_A N_B/s_A+P_B N_A/s_B\right),\qquad
 W^+=(1-\eta\lambda)W-\eta D.
\]

The 1/sqrt(n) nominal entry scale matches the exact polar-Muon shape convention,
but does not make the directions or actual finite-iteration RMS equal. The
other-fold RMS divisor is a new integration choice, not a theorem about optimal
learning rate. In the rho=0 control P_A=P_B=I; it retains this scaling, Nesterov,
gradient collection and decay, isolating the geometric effect of TCD1.

Recall the differentiable quadratic risk and its gradient:

\[
 R_E(D)=\tfrac14(\|E^TD+D^TE\|_F^2+\|ED^T+DE^T\|_F^2),\quad
 K_E(D)=EE^TD+DE^TE+2ED^TE.
\]

TCD1 uses P=(I-alpha K_e/c)/tau, e=E/||E||, alpha=rho*r/(1+rho*r),
r=||E||²/(||E||²+||fold_mean||²),
c=4|| (ee^T)^4 ||_F^(1/4), tau=Tr(I-alpha K_e/c)/(mn).
Transpose rectangular inputs to the smaller Gram side, then transpose back.
For E=0, P=I. The full derivation, trace formula, spectral bounds and explicit
counterexamples appear in the historical TCD section below. In exact arithmetic
P is SPD with eigenvalues in [1/(1+rho),1+rho].

To extend the conditional-mean argument to this integration, suppose that,
given the past, the two groups are independent and have mean mu=grad F(W).
Set A_A=P_A/(sqrt(n)*s_A), A_B=P_B/(sqrt(n)*s_B) and
nu=beta² M+(1-beta²)mu. Independence gives exactly

\[
 \mathbb E[D\mid past]=\tfrac12(\mathbb EA_A+\mathbb EA_B)\nu.
\]

This is why the divisor also comes from the other fold. It removes this specific
numerator/geometry dependence, not inverse bias or all optimization bias. If
s_min <= s_A,s_B <= s_max, then a0 I <= A <= b0 I, where
a0=1/(sqrt(n)*s_max*(1+rho)) and b0=(1+rho)/(sqrt(n)*s_min).
Writing delta=M-mu gives the explicit momentum alignment bound

\[
 \langle\mu,\mathbb ED\rangle
 \ge a_0\|\mu\|^2-\beta^2 b_0\|\mu\|\|\delta\|.
\]

With conditional group noise variance <=v², convexity of squared norm also gives
E||D||² <= b0²[||nu||²+(1-beta²)²v²]. L-smoothness therefore yields
E F(W-eta D) <= F(W)-eta<mu,ED>+L eta² E||D||²/2.
Substitute the alignment bound above directly. For beta=0 this
is a conventional small-step descent bound; for beta>0 stale momentum can make
the lower bound negative. Decoupled decay adds another term and is not included
in that inequality. No global improvement or convergence claim follows for the
actual packed, sequential FineWeb batches, which do not guarantee independent
unbiased folds. FP32 kernel arithmetic is also not an exact spectral certificate.

Cross-fitting already appears in
[Nayak et al.](https://arxiv.org/html/2605.20756v1); it is not claimed as our
invention. The particular Gram-transpose penalty is the experimental hypothesis.
A literature search is insufficient to establish that nobody has tried it.

### Preregistered first screen

The frozen study is gpt2-tcd-20261016-v1 on nano4. Its protocol.json is written
before candidate metrics: Muon, TCD1 rho=3, and rho=0 fold-RMS control each get
LR {.00625,.025,.1}, matrix decay .05, seed 42, 1000 steps, and the same auxiliary
recipe. This is nine exploratory runs, not a complete hyperparameter search.
The integrated rule still fails a concrete curvature example: with
g=[[0,-1],[0,0]], e=diag(2,1), samples=(g+e,g-e,g+e,g-e), beta=0 and eta=.1,
the loss F(W)=.5(W12-1)²+5000W21² goes from .5 to **7.55533** for rho=3,
versus **.36858** for the rho=0 control. This example has correlated folds and
large curvature, so it does not satisfy the small-step independent-fold theorem;
it directly rules out interpreting the normalized update as universally safe.
Their 70% cooldown is compressed equally, so the screen can mis-rank methods at
3250 steps. A boundary optimum requires an expanded range before any superiority
claim. Keep the published full-budget anchor even if a short screen prefers
something else. Additional strong published competitors are required for a
credible new-optimizer claim.

The public first 10,485,760 validation targets remain development data. Targets
80,000,000 through 90,485,759 in the same validation shard are reserved before
candidate selection and are not inspected in this screen. These share the same
source distribution and are not a claim of an independent corpus.
Gradient collection has extra communication; all of it counts in elapsed cost.
Both new and existing tests check fold weights, conditional mean, model/source
preservation and actual eight-rank parameter synchronization before training.

## Archived 150M HG-NorMuon experiment

## LR diagnosis after the first search results (2026-10-16)

The first nine completed configurations per method do **not** establish a new
optimizer improvement. At the shared anchor, NorMuon dev PPL was 29.25319 and
HG-NorMuon was 29.47907 (+0.772%). The HG matrix-factor median decreased from
0.901 at step 580 to 0.849 at 1150 and 0.825 at the end. This motivates a
learning-rate explanation; it does not establish one. A final unweighted median
is neither the time-averaged factor nor a measure of total update energy.

The matrix update is

```
W[t+1,l] = (1 - eta[t,l] * lambda[l]) W[t,l]
           - eta[t,l] * a[t,l] * U[t,l].
```

For a **constant** multiplier `a=c`, substituting `eta'=c*eta` and
`lambda'=lambda/c` yields the same update in real arithmetic, including decay.
Starting with the same weights and deterministic gradient/state computation,
induction gives the same trajectory. BF16 rounding can break bitwise equality.
Thus a constant factor cannot by itself be evidence for a better direction.
If `a[t,l]` varies, the equivalence instead requires a time- and layer-dependent
LR and inverse decay adjustment; one constant LR cannot generally reproduce it.
Even though each matrix direction is unchanged at a fixed state, the relative
matrix factors can change the concatenated model direction.

The configured LR still matters for HG. Holding the gradient/direction history
fixed, scaling base LR by `c>0` scales `h` by `c` and its cumulative squared
sum by `c^2`. The ratio `h/sqrt(sum(h^2))` approximately cancels that scale
(up to epsilon); it does **not** force the multiplier to compensate by `1/c`.
Actual histories also change with the LR, so this is a conditional algebraic
observation, not an invariance theorem for training.

A finite-budget counterexample is `f(w)=k*w^2/2`, `U=k*w`, constant `a`, and
zero decay. Then `f(w[T])=k*w[0]^2*(1-eta*a*k)^(2T)/2`. When
`0<eta*a*k<1`, reducing `a` slows optimization. With independent zero-mean
gradient noise of variance `sigma^2`, the stable recurrence has stationary
variance `eta*a*sigma^2 / (k*(2-eta*a*k))`, provided `0<eta*a*k<2`.
Smaller steps reduce this stationary variance while slowing bias decay. This
tradeoff explains why one-step loss adaptation need not minimize final loss.
It is not a proof of the cause of the observed neural-network result.

Reference starting points are recorded in `note/SOTA.md`, `note/BU.md`, and
`note/BV.md`: BR uses matrix/Adam LR `0.0012/0.0068`; BU uses
`0.0015/0.0085`; BV uses `0.0018/0.0102`. Their 1B-model, 1B-token PPL values
17.061 / 17.027 / 17.058 are **not comparable** to our 150M-model, 300M-token
dev scores. BU confirms the same `prepare-66ea6ae/data/bestfit` source. We
transfer LR hypotheses, momentum .95, beta2 .9873, batch 16, warmup .01, and
linear decay over the final .8; we recalibrate weight decay for the shorter run.

`exp.lr_diagnostic` preregisters **24 full runs**: NorMuon and HG each get the
same twelve pairs of (matrix LR, embedding/norm LR) multipliers relative to BR.
The core is the factorial grid `{.825,1,1.25,1.5,2} x {1,1.25}`, plus
`(.5,1)` and BV's `(1.5,1.5)`. Both original anchors are rerun. The
`(.825,1)` NorMuon point tests fixed shrinkage; `(1.25,1)` HG tests rough
compensation; `(1.25,1.25)` transfers BU. None is an exact trajectory match.

All use seed 42, identical initialization/data order/token budget, HG meta LR
.01, and integrated decays `C_matrix=C_embed=1.5`. Recalibrating decay as LR
changes preserves **every per-step shrinkage factor**, because each schedule
differs only by a constant scale. Changing raw LR with raw wd fixed would
confound optimization with regularization. This is a bounded two-group LR
diagnostic, not fully optimized hyperparameters: inspect boundary minima and
repeat selected configurations across seeds before interpreting a small gap.
The confirmation split is not evaluated or used by this diagnostic. The
original four-method campaign continues separately, including its meta-LR HPO.

The runner uses two independent single-GPU processes in one Slurm allocation,
alternating method-to-GPU assignments across paired points. It can continue
unfinished configurations in at most three 4-hour allocations; completed runs
are skipped, software failures stop for diagnosis. Actual allocation hours and
per-run worker hours are recorded separately. W&B project remains
`lab-training-llama`, group `exp150m-lr-diagnostic-v1`.

```
PYTHONPATH=src python -m exp.lr_diagnostic init --root /path/to/lr-study \
  --data-dir /path/to/existing/bestfit --cache-dir /path/to/existing/assets
PYTHONPATH=src python -m exp.lr_diagnostic submit --root /path/to/lr-study
PYTHONPATH=src python -m exp.lr_diagnostic report --root /path/to/lr-study
```

The diagnostic occupies one dev-partition job slot (two briefly during a
continuation). The original campaign uses at most 16 workers plus two
controllers. Their combined maximum fits the verified `p_dev` limit of 20;
the user's separate 8gpus-partition jobs use `p_8gpus`. No original campaign
code or controller is changed. Other future dev jobs can still exhaust the
shared limit; a submission failure requires inspecting the allocation log.

Related primary work: [Baydin et al., ICLR 2018](https://gbaydin.github.io/assets/pdf/baydin-2018-hypergradient.pdf)
derives online LR hypergradients; [Schmidt et al., 2021](https://arxiv.org/abs/2007.01547)
studies optimizer benchmarking and hyperparameter-search effects. The current
method is a research variant of existing ideas, not a claim of first invention.

Experimental candidate, **not a demonstrated improvement**. It is a bounded,
per-matrix hypergradient adaptation of NorMuon, based on
[Baydin et al., ICLR 2018](https://gbaydin.github.io/assets/pdf/baydin-2018-hypergradient.pdf).
All experiments use `lab-training-llama` on W&B. The original training entrypoint
and existing running jobs are independent of this package.

## Mathematics and limitations

For matrix `l`, let `U[l,t]` be the existing clipped-gradient Nesterov → Polar
Express → NorMuon update, including its existing shape scaling. Use

```
v[l,t] = eta[l,t] * U[l,t]
W[l,t+1] = (1 - eta[l,t] * wd[l]) * W[l,t] - a[l,t] * v[l,t]
```

At step `t`, the previous update and all earlier history are held fixed. Define
`Wbar = (1-eta_prev*wd)*W_prev` and the counterfactual batch objective
`phi_t(a) = loss(B_t, Wbar - a*v_prev)`, holding other updated matrices fixed.
By the chain rule, at the multiplier actually used last step,

```
d phi_t / da = <grad_W loss(B_t, W_t), -v_prev> = h_t
```

The observed gradient must be accumulated, DDP-synchronized, and **unclipped**.
Clipping generally scales this derivative and changes the adaptation. This is
an exact derivative of this one-step counterfactual, not of the full training
trajectory. Fresh conditionally unbiased batch gradients imply an unbiased
conditional derivative of the population counterfactual. Random reshuffling,
distribution drift, and data-dependent batches do not automatically meet that
assumption. No differentiation through momentum or orthogonalization is claimed.

Adapt with projected scalar AdaGrad, starting after warmup:

```
S_t = S_prev + h_t**2
rho_t = gamma / (sqrt(S_t) + 1e-12)
a_t = clip(a_prev - rho_t*h_t, 0.5, 2)
```

Initialize `a=1, S=0`. Warmup records the base displacement but does not adapt or
accumulate `S`. Missing gradients invalidate the previous displacement. A zero
LR leaves no useful displacement. Weight decay is independent of `a`. The global
control sums `h` across all transformer matrices before its single AdaGrad update.

For a fixed quadratic `phi(a)=c-b*a+q*a**2/2`, `q>0`, the constrained minimizer is
`clip(b/q,0.5,2)`, and the sign of `h=q*a-b` points toward it. More generally,
projection optimality gives `h*(a_next-a) <= -(a_next-a)**2/rho`. Combining this
with the smoothness inequality yields

```
phi(a_next) <= phi(a) - (1/rho - L_phi/2)*(a_next-a)**2.
```

This only guarantees descent for a fixed smooth objective with `rho < 2/L_phi`.
The implementation does not know `L_phi`, and applies feedback to a NEW update
and batch. It therefore does **not** guarantee monotone training loss, convergence,
or lower validation perplexity.

An explicit noisy-sign failure: let batch loss be `(w-s)**2/2`, with `s=±1`
equiprobably. At `w=.5, v_prev=.1`, the population hypergradient is `-.05`, but
the batch with `s=1` gives `+.05` and moves the multiplier the wrong way. Short
horizons, cross-matrix Hessian coupling, bad underlying directions, accumulated
AdaGrad state slowing adaptation, and boundary saturation can also hurt. Positive
matrix scaling preserves that matrix's current update direction, but changes the
concatenated model direction. Bounded multipliers do not guarantee stability.

The derivative's central finite-difference check has error approximately `2.4e-12`
in float64. This is a correctness check, not evidence of language-model gains.

## Reproducible protocol

- Llama: 22 layers, hidden 512, FFN 2048, 8 query / 2 KV heads, tied 128256-token
  embedding, context 8192: **149,314,048 parameters**. The added FP32 history for
  83,623,936 transformer matrix parameters costs **319 MiB** per replica.
- Reuse the existing prepared Arrow data and CUDA environment; no retokenization.
  Each seed has a deterministic permutation of complete packed rows, stopping
  at the last complete global batch that fits 300M non-padding tokens. Four
  methods get identical initialization, order, tokens, and steps for that seed.
- Global batch 16, micro batch 4, one H200 by default; FP32 parameters and BF16
  forward/backward, FA3, Liger, gradient norm clipping at 1. FP32 hypergradients.
- Evaluation: unpack existing evaluation documents, group identical token hashes,
  split groups 20%/80% with seed 20261016, and repack. Every 10% of training,
  evaluate only the dev subset. Sum NLL and valid predicted-token counts. The
  final subset is opened only after configurations are locked. These documents
  HAVE been used in historical experiments: this is not a historically blind set.
- Compare AdamW, NorMuon, global HG-NorMuon, and per-matrix HG-NorMuon. Each gets
  one anchor and 15 Sobol configurations, seed 42. Matrix LR `[1e-4,1e-2]`,
  embedding LR `[3e-4,3e-2]`, independently calibrated integrated decay
  `C=-sum(log(1-eta_t*wd))` (zero with probability 1/8, otherwise log-uniform
  `[.05,3]`), Muon momentum `{.9,.95,.98}`, Adam matrix beta1 `{.85,.9,.95}`,
  Adam beta2 `{.95,.9873,.999}`, warmup `{.01,.05}`, final linear decay
  fraction `{.5,.8}`, HG meta LR `[.001,.05]`. Embedding beta1 stays .9.
- Boundary optimum (outer 10% in log space) expands that boundary by 2x. A best
  dev PPL improvement >=.2% in the latest eight trials also triggers expansion.
  If ANY method triggers it, ALL receive eight more complete runs, capped at 32
  each. Shared NorMuon base coordinates/ranges remain matched. No loss-based
  early pruning. Numerical divergence counts as a failed trial.
- Top three configurations per method repeat on seeds 43,44. Select by mean
  dev NLL over 42,43,44; choose the stronger AdamW/NorMuon primary baseline BEFORE
  confirmation. Lock configurations, then run all four on seeds 101..105.
- Initial study: 64 search + 24 selection + 20 confirmation = 108 full runs;
  maximum after two expansions: 172. Four 50-step GPU smoke runs precede them.
  Report all failures, parameter ranges, best-so-far vs trials and GPU-hours.
- Primary success: paired five-seed mean NLL difference has 95% t interval upper
  bound below zero, geometric mean PPL improvement >=.3%, and >=4 seeds improve.
  Secondary intervals are descriptive, not multiplicity-adjusted significance
  claims. If tuning has not saturated at the cap, explicitly retain that caveat.
  No automatic 1B scaling or model publication.

## Run and inspect

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m unittest exp.test_exp exp.test_integration -v
PYTHONPATH=src .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 -m exp.test_distributed
PYTHONPATH=src OMP_NUM_THREADS=2 .venv/bin/python -m exp.train \
  --tiny-smoke --no-wandb --output /tmp/hg150m-smoke

# On nano4, from an isolated snapshot of this repository, using the existing venv:
PYTHONPATH=src /home/$USER/lab5-training-llama/.venv/bin/python -m exp.campaign init \
  --root /work/$USER/lab5-training-llama/experiments/hg150m-20261016-v1/study \
  --data-dir /work/$USER/lab5-training-llama/prepare-66ea6ae/data/bestfit \
  --concurrency 4 --submit

PYTHONPATH=src .venv/bin/python -m exp.campaign status --root /path/to/study
PYTHONPATH=src .venv/bin/python -m exp.report --root /path/to/study --no-wandb
```

`run.sbatch` uses the existing nano4 CUDA environment and single-GPU `dev`
partition. Arrays contain at most 16 tasks (four running concurrently), respecting
the partition's 20 submitted jobs per user limit. Dependent controller jobs advance the study after each array, even
when a worker fails. No repeated authentication, package installation, or data
download is needed. Infrastructure failures can retry the same configuration at
most twice; software errors stop progression for diagnosis. Smoke runs must pass
for all methods; an HG step-time overhead above 25% stops automatic progression.

`--resume auto` restores model, optimizer (including base history, factors and
AdaGrad accumulators), schedule, RNG and data position. It rejects mismatched
source/config/data fingerprints. Keep the deployed source snapshot immutable.
Numerical divergence never silently retries with a lower learning rate.

Each run writes `metadata.json`, atomic `result.json`, one rolling full checkpoint,
and final model weights. `study.json`, `locked_finalists.json`, `trials.csv`, and
`report.json` make selection decisions and failed runs reviewable. Final reports
upload tables and tuning curves to W&B. Worker hours include Python/model/data
initialization; Slurm `sacct` is authoritative for total allocation billing,
including controller and launch overhead.

## TCD research notebook — 2026-10-16

**Status: an implemented, falsifiable candidate; not a demonstrated improvement
on language-model training.** [tcd.py](tcd.py) replaces the matrix direction rule
itself. It does not invoke Muon, Adam, a polar factor, or Newton–Schulz. Its exact
solver uses an SVD of a *noise residual*; its first-order variant uses matrix
products. The existing 150M HG and LR campaigns continue from their immutable
snapshots. TCD is not yet connected to their Trainer.

### What existing optimizers actually assume

These are mechanisms, not a ranking. Limitations in the last column are our
mathematical/engineering analysis, except where a published counterexample is
cited. No finite review covers every optimizer, and no search proves world-first
novelty. For an SPD metric \(H\), a useful reference problem is

\[
 d=\arg\min_x\{-\langle g,x\rangle+\tfrac12\langle x,Hx\rangle\}
 =H^{-1}g,\qquad \theta^+=\theta-\eta d.
\]

Gradient estimation, metric choice, norm constraint, temporal memory, and step
size are separate decisions. A second moment is not automatically a Hessian.

| Family / primary source | Mechanism and reason it can help | Where the mechanism can fail |
|---|---|---|
| [SGD + momentum](https://proceedings.mlr.press/v28/sutskever13.html) | Average gradients over time; on suitable quadratics momentum accelerates slow modes and filters noise. | Stale directions and oscillation; one scalar LR cannot correct arbitrary anisotropic curvature. Deterministic acceleration is not a universal stochastic guarantee. |
| [AdaGrad](https://jmlr.org/papers/v12/duchi11a.html), RMSProp-style EMA | In the diagonal case divide by accumulated or averaged coordinatewise gradient energy; useful for unequal scales and sparse features. Full-matrix AdaGrad also exists. | Diagonal geometry misses rotated interactions; cumulative energy can make steps too small; finite-memory estimates can lag. |
| [Adam](https://arxiv.org/abs/1412.6980), [AdamW](https://arxiv.org/abs/1711.05101) | First-moment direction divided by square-root second moment; AdamW separates decay from this transformation. | Moments mix signal and noise; same-sample dependence matters. Adam has published convex failure examples; [AMSGrad](https://arxiv.org/abs/1904.09237) addresses a particular long-memory issue. |
| [Adafactor](https://proceedings.mlr.press/v80/shazeer18a.html) | Reconstruct approximate matrix second moments from row/column summaries; saves memory. | Factorization loses joint structure. Clipping and relative-step rules also affect performance. |
| [AdaBelief](https://arxiv.org/abs/2010.07468) | Use squared deviations from momentum to reduce damping for predictable coordinates. | Temporal residuals contain parameter drift and lag as well as stochastic noise; they are not generally fixed-parameter covariance estimates. |
| [Shampoo](https://arxiv.org/abs/1802.09568) | Matrix inverse roots, schematically \(L^{-1/4}GR^{-1/4}\), retain more geometry than diagonal scaling. | Factors omit general vectorized covariance; roots, damping, and stale statistics cost time or distort geometry. |
| [SOAP](https://arxiv.org/html/2409.11321v2) | Track Adam statistics in a basis built from Shampoo factors to handle rotated coordinate structure. | Basis changes, refresh frequency and matrix-factor memory matter. This is not an exact Hessian inverse. |
| [Muon](https://kellerjordan.github.io/posts/muon/) | For momentum \(M=U\Sigma V^\top\), a spectral-norm constrained linear problem admits \(UV^\top\) as an optimizer. Polynomial polar approximations redistribute singular-direction gains. | Flattening singular values can elevate weak/noisy modes; momentum and approximation error matter. It orthogonalizes an update, not weights, and estimates no loss Hessian. |
| [NorMuon](https://arxiv.org/html/2510.05491v1) | Output-row normalization and norm restoration modify a Muon update. | Row statistics can lag or be noisy; preserving the norm does not prove the direction is better. Its preferred LR need not match another rule's. |
| [K-FAC](https://arxiv.org/abs/1503.05671), [PSGD](https://arxiv.org/abs/1512.04202) | K-FAC approximates Fisher blocks with Kronecker factors. PSGD fits preconditioning using stochastic perturbation/gradient information. | Fisher, empirical gradient covariance and Hessian differ in general; approximations, damping, estimation noise and extra work limit benefit. These are distinct methods. |
| [Sophia](https://arxiv.org/html/2305.14342v4) | Diagonal curvature estimates and clipped curvature-scaled steps. | Off-diagonal curvature is omitted; stale estimates and clipping change behavior. Unclipped Newton intuition does not directly describe the update. |
| [Lion](https://arxiv.org/abs/2302.06675) | Momentum-informed signs give low-state-memory updates. | Magnitude information is discarded; LR and decay require retuning. |
| [SAM](https://arxiv.org/abs/2010.01412) | Gradients near adversarial weight perturbations optimize a neighborhood-loss objective. | Usually adds backward work; norm and parameterization matter. It changes the objective. |
| [MARS](https://arxiv.org/abs/2411.10438) | Recursive variance-reduced gradient estimation combined with adaptive optimization. | Reference-gradient cost, approximations and nonstationarity determine whether variance reduction pays off. |
| [ADOPT](https://arxiv.org/html/2411.02853v1), [cross-fitted preconditioning](https://arxiv.org/html/2605.20756v1) | Change which samples or time indices determine normalization, enabling different dependence arguments. | Independence-based guarantees have assumptions; lower bias need not improve finite-budget loss. Inverse-estimator bias remains after sample splitting. |

The [preconditioned-norm framework](https://arxiv.org/html/2510.10777v1) already
unifies many metric/norm choices. Mixing Adam and Muon, modifying spectral shaping,
or adding cross-fitting is insufficient novelty by itself. Targeted checks also
included [PRISM](https://arxiv.org/abs/2602.03096),
[Softsign](https://arxiv.org/abs/2605.31371), and
[Dead-Direction Conditioners](https://arxiv.org/abs/2606.29176).

### Candidate question and novelty boundary

Can *same-parameter gradient disagreement* define a useful matrix metric whose
action couples a direction to its transpose?

**Transpose-Coupled Descent (TCD)** uses the differential of the two Gram maps of
a noise residual to penalize a proposed update, solves that direction problem,
and cross-fits the resulting operator. The candidate contribution is this
particular noise-derived metric and update. Gram differentials, quadratic
minimization, SVD, trace normalization and sample splitting are existing tools.
[Nayak et al. (2026)](https://arxiv.org/html/2605.20756v1) already use independent
microbatch groups for preconditioning. Their inverse-bias correction is a separate
ingredient and is **not** implemented here.

Searches included “transpose-coupled optimizer”, “gradient noise Sylvester”,
“optimizer Kronecker sum”, “gradient disagreement Gram”, “noise pullback Gram”,
“symmetric antisymmetric optimizer”, and “cross-fitted preconditioner”.
No identical rule was identified in those checks. This is a limited novelty
assessment, not a world-first claim or exhaustive literature clearance.

### 1. Distinguish signal, noise, and curvature

Fix \(W\in\mathbb R^{m\times n}\) and condition on the complete past. Let
\(\mu=\nabla F(W)\). Four equally weighted independent microbatch **mean**
gradients are \(G_i=\mu+\xi_i\), with zero-mean IID \(\xi_i\).

\[
 S_A=(G_1+G_2)/2,\quad E_A=(G_1-G_2)/2,\qquad
 S_B=(G_3+G_4)/2,\quad E_B=(G_3-G_4)/2.
\]

Then \(\mathbb E S_A=\mu\), \(\mathbb E E_A=0\), and

\[
 \mathbb E[E_AE_A^\top]
 =\tfrac12\mathbb E[\xi_1\xi_1^\top]
 =\mathbb E[(S_A-\mu)(S_A-\mu)^\top].
\]

The analogous right- and vectorized-covariance identities follow by expanding
products and cancelling independent cross terms. No Gaussian assumption is
needed. \(S_A\) and \(E_A\) need not be independent; the whole A group is independent
of the whole B group. This estimates noise at the current parameter, unlike
\(G_t-G_{t-1}\), which includes changes in the true gradient. Noise covariance
still is **not** \(\nabla^2F(W)\).

### 2. A concrete direction objective

For a residual \(E\) and proposed direction \(D\), define

\[
 R_E(D)=\tfrac14\left(
 \|E^\top D+D^\top E\|_F^2+\|ED^\top+DE^\top\|_F^2\right).
\]

These are the two components of \(J_E[D]\), the differential of
\(\Phi(X)=(X^\top X,XX^\top)\). Thus \(R_E(D)=\frac14\|J_E[D]\|^2\) and

\[
 \nabla_D R_E(D)=K_E(D)
 =EE^\top D+DE^\top E+\boxed{2ED^\top E}.
\]

Expanding and differentiating the squared norms gives the three terms;
equivalently \(K_E=\frac12J_E^*J_E\). Therefore it is self-adjoint and PSD, with
\(\langle D,K_E(D)\rangle=2R_E(D)\). The boxed term couples transposed coordinates;
dropping it is an explicit ablation. This Gram pullback metric is not the loss
Hessian.

Both Gram differentials have trace \(2\langle E,D\rangle\).
Applying \(\|A\|_F^2\ge(\operatorname{tr}A)^2/\dim(A)\) to each gives

\[
 R_E(D)\ge(1/m+1/n)\langle E,D\rangle^2.
\]

For fixed \(D\) independent of \(E\), expectation bounds the variance of the
linearized noisy gain. **It does not prove that minimizing this empirical penalty
produces a population variance-optimal direction:** the chosen direction depends
on the sampled geometry. The penalty can also suppress noiseless modes.

Triangle and submultiplicativity inequalities give
\(R_E(D)\le2\|E\|_{\rm op}^2\|D\|_F^2\), hence

\[
 0\preceq K_E\preceq4\|E\|_{\rm op}^2I.
\]

### 3. Fully specified update

Choose \(\rho\ge0\). For each group define

\[
 r_A=\frac{\|E_A\|_F^2}{\|E_A\|_F^2+\|S_A\|_F^2},\quad
 \gamma_A=\frac{\rho r_A}{4\|E_A\|_{\rm op}^2},\quad
 H_A=I+\gamma_AK_{E_A}.
\]

Set \(r_A=0\) when both norms vanish and \(\gamma_A=0\) when \(E_A=0\).
The bounded gate is a heuristic, **not** an unbiased signal-to-noise ratio.
The unique unnormalized solution is

\[
 \widetilde D_A
 =\arg\min_D\{\tfrac12\|D-S_B\|_F^2+\gamma_A R_{E_A}(D)\}
 =H_A^{-1}S_B.
\]

The quadratic Hessian is SPD; setting the gradient
\(D-S_B+\gamma_AK_{E_A}(D)\) to zero proves the equality. Finally,

\[
 \tau_A=\operatorname{Tr}(H_A^{-1})/(mn),\quad
 P_A=H_A^{-1}/\tau_A,\qquad
 D=\tfrac12(P_AS_B+P_BS_A),\quad W^+=W-\eta D.
\]

Here \(\operatorname{Tr}\) is the trace of the \(mn\)-dimensional operator, not of
the parameter matrix. Normalizing it makes the mean eigenvalue one, removing
uniform attenuation. It does **not** preserve actual update norm or energy and
does **not** eliminate LR tuning. For \(\rho=0\), TCD is exactly the average of
the four gradients.

### 4. What can actually be proved

Let \(\kappa=1+\rho\). The operator bound implies

\[
 I\preceq H_A\preceq\kappa I,\quad
 \kappa^{-1}I\preceq H_A^{-1}\preceq I,\quad
 \kappa^{-1}\le\tau_A\le1,\quad
 \kappa^{-1}I\preceq P_A\preceq\kappa I.
\]

Since \(P_A\) uses group A only and \(S_B\) is independent of A,

\[
 \mathbb E[D\mid\text{past}]
 =\tfrac12(\mathbb E[P_A\mid\text{past}]
          +\mathbb E[P_B\mid\text{past}])\mu,\qquad
 \mathbb E\langle\mu,D\rangle\ge\kappa^{-1}\|\mu\|_F^2.
\]

The last step applies the pointwise SPD lower bound to \(\mu\), then averages.
It needs no independence between the two update terms. Also,
\(\mathbb E[H^{-1}]\ne(\mathbb EH)^{-1}\) in general: cross-fitting removes one
dependence, not inverse-estimator bias.

For a smooth nonconvex bound, assume \(F\) is \(L\)-smooth, bounded below by \(F_*\),
and each group mean has conditional noise variance at most \(\nu^2\).
Convexity of squared norm and the operator upper bound imply

\[
 \mathbb E\|D\|_F^2
 \le\tfrac12\mathbb E(\|P_AS_B\|_F^2+\|P_BS_A\|_F^2)
 \le\kappa^2(\|\mu\|_F^2+\nu^2).
\]

Insert both bounds into smoothness:

\[
 \mathbb E[F(W-\eta D)\mid\text{past}]
 \le F(W)-\frac{\eta}{\kappa}\|\mu\|_F^2
       +\frac{L\eta^2\kappa^2}{2}(\|\mu\|_F^2+\nu^2).
\]

For constant \(0<\eta\le1/(L\kappa^3)\), the gradient coefficient is at most
\(-\eta/(2\kappa)\). Sum over \(T\) steps, use the tower property and
\(\mathbb EF(W_T)\ge F_*\), obtaining

\[
 \frac1T\sum_{t=0}^{T-1}\mathbb E\|\nabla F(W_t)\|_F^2
 \le\frac{2\kappa(F(W_0)-F_*)}{\eta T}+L\eta\kappa^3\nu^2.
\]

A sufficiently small \(\eta\propto T^{-1/2}\) gives an \(O(T^{-1/2})\) stationarity
bound. This proves neither a global minimum nor superiority over SGD; the
conservative constants get **worse** with \(\rho\). Multiple blocks admit the
same proof using their block-diagonal operator and full-gradient assumptions.

The proof covers no momentum, no decay, exact arithmetic, unbiased equally
weighted independent samples, and fixed finite \(\rho\). Optional momentum is an
implementation feature with no theorem here. Current-batch global clipping,
random reshuffling, unequal token weights, correlated packed sequences and
finite precision need separate treatment. Calling this class in the current
150M Trainer would not automatically satisfy these assumptions.

### 5. Exact solver without a gigantic vectorized matrix

Transpose if necessary so \(m\le n\). With thin SVD
\(E=U\operatorname{diag}(s)V^\top\), write \(B=U^\top GV\). On the core,

\[
 \widetilde B_{ij}
 =\frac{(B_{ij}+B_{ji})/2}{1+\gamma(s_i+s_j)^2}
  +\frac{(B_{ij}-B_{ji})/2}{1+\gamma(s_i-s_j)^2}.
\]

Indeed, the operator on a pair is
\((s_i^2+s_j^2)B_{ij}+2s_is_jB_{ji}\). Symmetric and antisymmetric pairs have
eigenvalues \((s_i+s_j)^2\) and \((s_i-s_j)^2\); diagonal entries have \(4s_i^2\).
The right-nullspace component \(U^\top G-BV^\top\) has row eigenvalues \(s_i^2\),
so divide row \(i\) by \(1+\gamma s_i^2\), then recombine with \(U\).

For \(a_{ij}=[1+\gamma(s_i+s_j)^2]^{-1}\) and
\(b_{ij}=[1+\gamma(s_i-s_j)^2]^{-1}\), the exact trace is

\[
 mn\tau=\sum_i a_{ii}+\sum_{i<j}(a_{ij}+b_{ij})
 +(n-m)\sum_i(1+\gamma s_i^2)^{-1}.
\]

This handles rectangular and rank-deficient cases without a full \(V\). Tests
compare it against explicit vectorized linear systems and exact inverse traces.

### 6. SVD-free variant TCD1: a distinct approximation

For nonzero \(E\), let \(e=E/\|E\|_F\), \(A=ee^\top\), transposing as above, and
\(c=4\|A^4\|_F^{1/4}\). For PSD \(A\),
\(\|A^4\|_F^{1/4}=(\sum_i\lambda_i^8)^{1/8}\ge\lambda_{\max}(A)\), hence
\(K_e\preceq cI\). Set

\[
 \alpha=\frac{\rho r}{1+\rho r},\quad Q=I-\frac{\alpha}{c}K_e,\quad
 \tau_1=\frac{\operatorname{Tr}Q}{mn}
 =1-\frac{\alpha(m+n+2)\|e\|_F^2}{c\,mn},\quad P_1=Q/\tau_1.
\]

Use \(P_1\) in the same cross-fitted rule, with \(P_1=I\) for \(E=0\).
The trace follows from \(\operatorname{Tr}K_e=(m+n+2)\|e\|_F^2\).
After compensation its spectrum lies in \([\kappa^{-1},\kappa]\), so the same
expectation and stationarity arguments apply in exact arithmetic.

This is one gradient step from \(G\) on
\(J(D)=\frac12\|D-G\|^2+(\rho r/c)R_e(D)\), with step \(1/(1+\rho r)\).
The Hessian norm is at most \(1+\rho r\), so **unnormalized** \(QG\) decreases this
direction objective. Trace compensation afterward need not do so. TCD1 is not
equal to the exact inverse or claimed to converge to it in one step.

Compute \(T=Ge^\top\), then \(K_e(G)=AG+(T+2T^\top)e\). Including \(A,A^2,A^4\),
this requires six matrix products per group and two groups. FP32 rounding can
perturb spectral bounds; the proof is not a floating-point certificate.

### 7. A genuine direction change and explicit failure

For \(E=\operatorname{diag}(2,1)\), \(G=\begin{pmatrix}0&-1\\0&0\end{pmatrix}\),
group mean zero and \(\rho=3\), the exact compensated map gives

\[
 P_EG\approx\begin{pmatrix}0&-1.1929478\\0.4617862&0\end{pmatrix}.
\]

Rank-one gradient becomes a rank-two direction: a scalar LR cannot create the
missing entry. The repository's zero-history Muon polynomial preserves its
input's singular row/column subspaces, unlike this example. This is not a claim
about every momentum history or arbitrary SVD nullspace completion.

Now consider the real loss

\[
 F(W)=\tfrac12(W_{12}-1)^2+5000W_{21}^2,\qquad W=0.
\]

Its gradient is the stated \(G\). Despite positive alignment
\(\langle G,P_EG\rangle\approx1.19295\), a step at \(\eta=.1\) increases loss
from **0.5 to 11.0501476**; SGD gives **0.405**. TCD creates a component along
unobserved high curvature. This refutes universal per-step superiority, not the
theorem with its much smaller \(L\)-dependent step bound.

If \(E=zI\), then \(K_E(D)=2z^2(D+D^\top)\): symmetric modes are damped and skew
modes survive. The uncoupled ablation is just \(2z^2D\), cancelled by trace
compensation. This predicts benefit for skew signal and identity-mode noise.
It also exposes a flaw: **traceless symmetric signal is noiseless in this
idealization but is still damped**. Gram risk is a stronger, potentially
misdirected surrogate for noise variance, not the actual rank-one covariance.

Other failures: one residual pair estimates geometry noisily; domain differences
may be treated as nuisance noise; rare useful directions can be suppressed;
trace compensation amplifies other modes; isotropic noise offers little useful
directional information; noise and curvature can disagree; and separate gradient
capture plus geometry construction costs memory and time.

### 8. Experiments and evidence limits

[tcd_probe.py](tcd_probe.py) uses \(4\times4\) quadratic objectives, 200
constant-LR steps, no decay, and four independent gradient samples per step for
**every** method. These are not language-model results. Evaluation uses the true
quadratic objective. Distinct tuning/confirmation seeds share a problem family;
these are mechanism probes, not an unbiased benchmark of all tasks.

Each method gets **24 configurations**: LRs \(\{.001,.003,.01,.03,.1,.3,1,3\}\)
times three secondary settings. SGD/NorMuon search momentum \(\{0,.9,.95\}\);
Adam fixes beta1=.9 and searches beta2 \(\{.9,.99,.999\}\); TCD/ablations search
\(\rho\in\{.3,3,30\}\) without momentum. Three tuning seeds select a config;
eight new seeds evaluate it. All methods in a case are locked before any
confirmation run in that case. Shared initial states and noise draws give paired
comparisons. Failed configurations remain failures, with non-finite loss stored
as null.

Equal configuration counts do not establish global hyperparameter optimality.
Several secondary optima hit boundaries, including \(\rho=30\) in the favorable
case; ranges are not exhausted. TCD has no momentum search here, relevant when
comparing to momentum SGD. Selected LRs are interior, but a coarse interior
optimum does not prove sufficient tuning. NorMuon's polar iteration uses BF16;
the rest of this synthetic simulation, including its row equalization, uses
float64. This differs from the production FP32 row statistics.

Round 1 (seeds 11–13 and 101–108) compared SGD, Adam, NorMuon, exact TCD,
no coupling, same-fold geometry and no trace compensation. Round 2 tested TCD1
with fresh seeds offset by 1000. Round 3 added a Sherman–Morrison rank-one noise
covariance inverse and compared both solvers on the **same** seeds, offset by
2000. The control uses the same split, gate and trace compensation, with core
\(Q=I-\alpha\,\operatorname{vec}(E)\operatorname{vec}(E)^\top/\|E\|^2\).
It tests whether a simpler noise filter already explains the gain.

Round 3 confirmation means (lower is better):

| Method | Identity-mode noise + skew signal | Isotropic noise | Rotated curvature |
|---|---:|---:|---:|
| SGD, momentum tuned | 0.0695253 | 0.00919589 | **0.00135595** |
| Adam | 0.0135803 | 0.0292980 | 0.00299512 |
| NorMuon | 0.0525300 | 0.0282318 | 0.0726615 |
| Exact TCD | **0.0116537** | 0.00926665 | 0.00630063 |
| TCD1 | 0.0187505 | 0.00926600 | 0.00630060 |
| Rank-one noise control | 0.0608751 | 0.00920539 | 0.00629443 |
| TCD without transpose coupling | 0.0992464 | 0.00922052 | 0.00629486 |
| TCD same-fold ablation | 0.0341823 | 0.00922211 | 0.00630742 |
| TCD without trace compensation | 0.0602636 | **0.00765288** | 0.00691641 |

These are descriptive means, not significance claims. The favorable problem was
deliberately constructed around the mechanism: skew initial signal benefits
from trace compensation's skew-subspace amplification.

In round 1 exact TCD's favorable mean was .0147934 versus Adam's .0297953.
In round 2 TCD1 gave .0174293 versus Adam's .0173252; its same-fold ablation was
better still (.0119793). Neither the first win nor cross-fitting's cleaner proof
establishes a general advantage. The paired round 3 shows that the cheaper
approximation can lose useful behavior.

Configs, individual seeds, failures, histories and source snapshots are saved
in repository-local ignored artifacts:

- outputs/tcd-probe-20261016-v1/: exact reference, seven methods.
- outputs/tcd-probe-20261016-v2-first-order/: fresh-seed approximation check.
- outputs/tcd-probe-20261016-v3-controls/: paired solvers and stronger control.
- outputs/tcd-probe-20261016-v4-symmetric/: symmetric-signal falsification.

The first source snapshot was captured after its run, explicitly recorded in its
manifest. Later rounds save source hashes and protocol before evaluation. Fresh
seeds do not erase that later experiments were designed after seeing earlier ones.

Round 4 keeps identity-mode noise but replaces the skew initial signal with a
symmetric traceless matrix (new seeds offset by 3000). Confirmation means:
SGD **.0376431**, Adam **.0296217**, NorMuon **.0374104**, exact TCD **.0490604**,
TCD1 **.0491521**, rank-one control **.0497466**, no coupling **.0503739**,
same-fold **.0436446**, no trace compensation **.0475560**. Exact TCD selects the
lowest tested rho (.3) and still loses to all three conventional baselines.
Together with the analytic example, this weakens the interpretation that the
Gram penalty identifies noise alone. Its preferred subspace matters; the first
round's favorable signal was unusually aligned with it. These finite-budget
results do not imply the simpler covariance control is intrinsically worse.

### 9. H200 cost and the gate before 150M

Slurm job **555883** on nano4 completed. Medians of 20 CUDA-event timings after
five warmups, eager execution, one NVIDIA H200:

| Matrix shape | Existing NorMuon | Exact TCD | TCD1 |
|---|---:|---:|---:|
| 128 × 512 | 0.482 ms | 8.528 ms | 0.652 ms |
| 512 × 512 | 0.475 ms | 38.460 ms | 0.706 ms |
| 2048 × 512 | 0.488 ms | 39.490 ms | 0.720 ms |

TCD uses FP32 with highest matmul precision; NorMuon uses native BF16 Polar
Express. Timing includes averaging and direction construction but excludes
backward, gradient capture, parameter mutation, compilation and communication.
These are not model-throughput measurements. There is one allocation, with no
cross-device uncertainty estimate. Exact TCD is currently impractical; TCD1's
kernel costs roughly 1.35–1.49 times NorMuon's eager kernel on these shapes.
Raw results: outputs/tcd-probe-20261016-v3-controls/gpu-timings.json.
Remote snapshot:
/work/u9603854/lab5-training-llama/experiments/tcd-kernel-20261016-v1.

For the 150M model's 83,623,936 optimized transformer matrix entries, four FP32
samples alone occupy about **1.25 GiB** per replica. The reference clones samples
and holds all solved directions before mutation, so peak memory is higher.
Streaming folds or recomputation would need additional implementation/validation.

Before treating this as a 150M contender, address symmetric-signal/curvature
failure, study momentum with matched tuning effort, and preserve token weighting
when capturing four same-parameter gradients. Then compare fixed tokens **and**
GPU-hours, independently tune matrix/embedding LR, match integrated decay, lock
choices on dev, and use the existing confirmation protocol. A kernel smoke test
and small-matrix win do not establish a stronger language-model optimizer.
Current decision: retain TCD as a research candidate, not a training default.

### Reproduction and API

TCD.set_samples requires one tensor of shape (4,m,n) for each matrix parameter,
matching dtype/device and containing finite values. step consumes it once;
an accumulated .grad is not silently reinterpreted as four independent samples.
The default solver is "exact"; "first_order" chooses TCD1. Optional bias-corrected
momentum and decoupled weight decay are outside the theorem.

~~~bash
PYTHONPATH=src .venv/bin/python -m unittest exp.test_tcd -v
PYTHONPATH=src .venv/bin/python -m exp.tcd_probe \
  --output /tmp/tcd-new-probe --solver exact \
  --seed-offset 2000 --extended-controls
PYTHONPATH=src .venv/bin/python -m exp.tcd_probe \
  --output /tmp/tcd-new-symmetric-probe --solver exact \
  --seed-offset 3000 --extended-controls --cases common_mode_symmetric_signal
# On CUDA with the existing environment:
PYTHONPATH=src python -m exp.tcd_benchmark --output /tmp/tcd-new-timings.json
~~~

Twelve new tests check the derivative, risk bound, explicit inverse and trace,
SPD bounds, orthogonal/transpose equivariance, rank-changing failure, enumerated
independent-sample expectations, zero-noise behavior, resume, first-order
objective descent, and the rank-one control against a full covariance solve.
Together with existing experiment/integration/LR checks, **38 tests passed**.
These check identities and implementation behavior, not empirical superiority.
