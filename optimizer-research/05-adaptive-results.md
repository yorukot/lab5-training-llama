# Adaptive shrinkage: results — 2026-10-16

**This candidate did not improve language-model PPL in the development screen.**
Across 15 equal-token runs on nano4, its best tested PPL was **40.19570**, versus
**39.31948** for its unshrunk Shampoo parent and **33.65999** for Muon. Every
method's selected LR is bracketed by worse tested rates. These are 1,000-step,
single-seed development results, not a full-horizon or confirmation result.
The version is not promoted. Total charged cost, including verification and
the failed verification attempt, is **8.46222 of 64 H200 GPU-hours**.

**The adaptive rule preserves a constant weak factor, but it does not uniformly
improve optimization.** On the deterministic weak-signal optimization problem,
it remains worse than the unshrunk parent, although it is less harmful than
fixed shrinkage. Its temporal statistic responds to genuine gradient drift as
well as stochastic variability. The synthetic findings below help diagnose
the hypothesis; they do not establish the cause of the LM result.

The rule and its assumptions are derived in
[04-adaptive-shrinkage.md](04-adaptive-shrinkage.md). This experiment freezes
the adaptive coefficient cap at **0.1**. The separately labeled cap-1 factor
diagnostic does not change the candidate's default.

## Protocol and execution

The study completed on nano4 CPU: **180 optimization trials and 60 factor
streams, with zero recorded failures**. Its recorded study-loop wall time was
**5.804 seconds**, excluding interpreter and Torch startup. GPU cost for this
synthetic study was **0 H200-hours**. The centralized verification run passed
98 tests and the separate two-rank CPU synchronization check.

Optimization used 4×4 parameters, float64 arithmetic, one CPU thread, and
128 steps. Every method received the same LR grid
**{0.003, 0.01, 0.03, 0.1, 0.3, 1.0}**, tuning seeds **11/22**, and fresh test
seeds **1001/2002/3003**. Each method's LR was selected by mean final population
quadratic loss, and every choice was locked before any test trial or factor
diagnostic. Initialization and noise were paired within each problem/seed;
the stationary low/high-noise cases additionally shared their Hessian,
initialization, and standardized noise draws.

All three methods retained the same blocked EMA Shampoo parent: beta1 0.9,
beta2 0.95, block size 128, inverse fourth powers, root refresh every ten steps,
relative ridge, and tensor-level Adam graft. Weight decay was zero and the LR
was constant. Only the shrinkage rule differed: zero, fixed 0.1, or adaptive
in [0, 0.1]. Equal LR grids do not establish that all methods are fully tuned.

## Final quadratic losses

Each entry is the arithmetic mean final population loss across the three test
seeds. Smaller is better. The selected LR happened to be the same for all
three methods within each regime.

| Regime | Parent Shampoo | Fixed 0.1 | Adaptive, cap 0.1 | Selected LR, all three |
|---|---:|---:|---:|---:|
| Stationary low noise | 0.00348499 | 0.00404013 | 0.00369942 | 0.1 |
| Stationary high noise | 0.06532258 | 0.06352591 | 0.06459438 | 0.03 |
| Deterministic weak signal | 5.87153e-6 | 1.04196e-4 | 4.39867e-5 | **1.0*** |
| Abrupt curvature drift | 0.01480627 | 0.01425311 | 0.01408507 | 0.03 |

*All three weak-signal choices are at the upper LR boundary. Their tuning is
unresolved; the experiment did not expand the range after seeing test results.*

- **Low noise:** adaptive loss is 6.15% above the parent. Two of the three
  paired seeds are worse. Fixed shrinkage is 15.93% above the parent.
- **High noise:** adaptive loss is 1.11% below the parent, with all three paired
  seeds improving. Fixed shrinkage is 2.75% below the parent, so adaptive is
  **1.68% worse than fixed** on this comparison.
- **Weak signal:** adaptive loss is **7.49× the parent loss**, while fixed
  shrinkage is 17.75× the parent loss. Adaptive reduces the fixed method's loss
  by 57.78%; that is not an improvement over the parent. Absolute losses are
  small, and all three LRs remain at the boundary. The sign-symmetric
  deterministic seeds produce identical losses; their zero sample standard
  deviation is not evidence that general training uncertainty vanishes.
- **Curvature drift:** adaptive mean loss is 4.87% below the parent, but one
  of three paired seeds is worse. Fixed shrinkage is 3.74% below the parent.

These small designed cases and three test seeds do not support a general
optimizer ranking or a significance claim. The complete paired losses and
selection grids remain in the artifacts.

## Factor-estimation diagnostics

The separate streams hold parameters fixed and replay identical gradients
through each optimizer at LR zero. For
\(G=M+\sigma L^{1/2}ZR^{1/2}\), with independent standard-normal entries in
\(Z\), the exact population factors are

\[
C_L=MM^\top+\sigma^2\operatorname{tr}(R)L,\qquad
C_R=M^\top M+\sigma^2\operatorname{tr}(L)R.
\]

This supplies known factor truth without identifying it with a loss Hessian.
Both the current population factor and the bias-corrected EMA of population
factors are recorded. They coincide in stationary streams and differ after
drift.

The table reports **squared Frobenius factor error at the final scheduled root
refresh, step 121**, before ridge and inverse powers. Each value weights the
left and right 4×4 factors equally and weights all three seeds equally: six
factor errors per entry. No favorable side is selected, and errors are not
pooled across regimes with different scales.

The oracle is the best scalar coefficient for each observed factor given its
exact current truth, restricted to [0, 0.1]. It uses information unavailable to
the optimizer. It is a same-dimension factor-error reference, not an oracle for
final optimization loss.

| Fixed-parameter stream | Parent error | Fixed 0.1 error | Adaptive, cap 0.1 error | Oracle, cap 0.1 error |
|---|---:|---:|---:|---:|
| Constant weak signal | 7.38868e-31 | 0.00705206 | 7.38868e-31 | 5.26542e-31 |
| Anisotropic low noise | 4.03327e-6 | 0.00707126 | 4.03513e-6 | 4.03265e-6 |
| Anisotropic high noise | 0.19985635 | 0.20355949 | 0.19360935 | 0.17162985 |
| Isotropic noise | 0.12228365 | 0.10256226 | 0.10256226 | 0.10256226 |
| Deterministic drift | 0.00307654 | 0.01488714 | 0.00322298 | 0.00307654 |

The constant weak-factor case behaves as intended: adaptive coefficients stay
below approximately 2.71e-32 over every recorded refresh and the factor remains
unchanged to numerical precision. Fixed shrinkage introduces a visible bias.
In anisotropic low noise, the adaptive left/right coefficients average
1.02e-5 and 3.76e-6 at the final refresh. This largely avoids fixed shrinkage's
distortion, but its average factor error is still **0.046% above the parent**.

High-noise results differ by side:

| Anisotropic high-noise factor | Parent error | Adaptive error | Change from parent |
|---|---:|---:|---:|
| Left | 0.18783691 | 0.20899957 | **+11.27%** |
| Right | 0.21187578 | 0.17821913 | **−15.89%** |
| Equal mean of both sides | 0.19985635 | 0.19360935 | −3.13% |

Thus the aggregate improvement does not mean both factors improved. The
sample-dependent uncertainty ratio is not an oracle coefficient.

On purely isotropic noise, both adaptive coefficients hit the 0.1 cap and
match fixed shrinkage at the final refresh. The diagnostic with cap 1 reaches
mean error 0.02447388, compared with the cap-1 oracle's 0.01848686. This is an
expectedly favorable setting for removing anisotropy. It does not justify
raising the production cap or transferring the improvement to an anisotropic
training trajectory.

## Why constant signal and deterministic optimization disagree

A constant gradient stream and a noiseless optimization trajectory are
different experiments. As the optimizer moves, its deterministic gradient
changes. The temporal scatter measures that change even when sampling noise
is exactly zero.

In every weak-signal optimization test seed, adaptive shrinkage starts at zero
on step 1 and then hits **0.1 on both factors at every recorded refresh from
step 11 onward**. It therefore does not retain the near-zero coefficients seen
in the constant weak-factor stream. This rules out interpreting the
constant-stream success as evidence that useful weak directions will always
be preserved during training.

The deterministic drift stream makes the confounding explicit. At the first
recorded refresh after the abrupt change, step 71, the mean adaptive
coefficient is 0.01647 despite zero stochastic noise. The raw factor already
matches the expected EMA factor up to roundoff; adaptive shrinkage introduces
mean error 0.00013367 against that expectation. Against the *current* population
factor it happens to improve error slightly at this refresh,
0.545774 → 0.539997, illustrating that the two reference targets answer
different questions.

By the final refresh, adaptive current-factor error is 0.00322298 versus
0.00307654 for the parent, **4.76% worse**. Against expected EMA truth, the
parent's error is approximately 8.83e-33 and adaptive's is 3.98e-6. Temporal
uncertainty cannot be described as a measurement that separates noise from
drift. Reduced factor error also does not, by itself, establish a better
inverse root, grafted direction, or terminal loss.

The evidence supports keeping adaptive shrinkage as a falsifiable candidate
with a bounded coefficient. It does not establish superiority over the parent
or fixed shrinkage. The weak-signal and drift failures remain required controls;
the coefficient cap and test conditions were not changed to remove them.

## Reproduction and evidence

~~~bash
PYTHONPATH=src python -m exp.adaptive_shrink_probe \
  --output /path/to/fresh-directory --steps 128 --seeds 1001 2002 3003
~~~

- [Frozen protocol and source hashes](../outputs/gpt2-research-20261016-v1/adaptive/synthetic-r1/protocol.json)
- [Configurations locked before test evaluation](../outputs/gpt2-research-20261016-v1/adaptive/synthetic-r1/locked_configurations.json)
- [Summary, selection grids, and individual test losses](../outputs/gpt2-research-20261016-v1/adaptive/synthetic-r1/summary.json)
- [Every optimization trial and applied-coefficient history](../outputs/gpt2-research-20261016-v1/adaptive/synthetic-r1/trials.json)
- [Both factor sides, every seed and refresh, current/EMA truths, and oracle diagnostics](../outputs/gpt2-research-20261016-v1/adaptive/synthetic-r1/factor_diagnostics.json)

## GPU correctness and execution

The scheduler recovered, and the previous ambiguous submission was absent from
both the live queue and accounting. Its conservative hold was released. Job
557755 then exposed an issue in the distributed test reference: NCCL and a
sequential FP32 sum can differ in rounding, which a nearly singular Shampoo
factor can amplify. The corrected test checks the summed gradient independently
and then supplies the exact same reduced gradient to both optimizer paths.
Parameter tolerances were not loosened.

Job **557855** passed all-method eight-rank NCCL synchronization, state
continuation, and production-shape GPU profiling, followed by a two-step adaptive
model smoke test and standalone checkpoint evaluation. The distributed test
uses a two-step root refresh so it actually applies nonzero adaptive shrinkage.
The standalone evaluation uses the public development slice; it does not open
the reserved confirmation slice. The two smoke PPL values differ slightly
between compiled training and eager standalone evaluation, so bit-identical
evaluation is not claimed.

The three-matrix profile showed roughly 1.8 ms for warmed non-refresh adaptive
steps versus roughly 1.1 ms for its parent. This is an optimizer kernel
diagnostic, not an end-to-end training comparison. Peak memory values include
other live profiling tensors and must not be read as optimizer-state memory.
Actual allocated GPU time, including failures and startup, is charged separately.

- [Final 98-test suite](../outputs/gpt2-research-20261016-v1/adaptive/verification/logs/adaptive-unit-final.log)
- [Two-rank CPU synchronization](../outputs/gpt2-research-20261016-v1/adaptive/verification/logs/adaptive-distributed-cpu.log)
- [Passed GPU verification and profile](../outputs/gpt2-research-20261016-v1/adaptive/verification/logs/adaptive-verify-r1-557855.log)
- [Checkpoint evaluation](../outputs/gpt2-research-20261016-v1/adaptive/verification/runs/adaptive-verify-r1/development-eval.json)

## Language-model development screen

The [registered comparison](../src/exp/research_adaptive_screen.json) uses the
pinned 162,354,816-parameter GPT-2-derived model, seed **42**, **1,000 steps**,
and **524,288,000 training targets** per run. All runs share the architecture,
data order, auxiliary AdamW, precision, global batch, 70% cooldown, and public
10,485,760-target development validation slice. Each run starts from a fresh
compiler cache. The report checks that their common training identities match.

The initial adaptive result was unfavorable. The remaining two learning rates
per method were then run from the already registered grids, with the diagnostic
decision recorded before those runs. This did not promote the candidate to
full-horizon development. The complete original grid produced:

| Method | Matrix LR | Development PPL |
|---|---:|---:|
| Muon | 0.0125 | 34.27444 |
| Muon | 0.025 | **33.65999** |
| Muon | 0.05 | 33.77497 |
| Parent Shampoo variant | 0.001 | 39.31948 |
| Parent Shampoo variant | 0.003 | 52.15395 |
| Parent Shampoo variant | 0.009 | 65.51113 |
| Fixed shrinkage 0.1 | 0.001 | 40.45175 |
| Fixed shrinkage 0.1 | 0.003 | 91.22053 |
| Fixed shrinkage 0.1 | 0.009 | 196.14418 |
| Adaptive shrinkage, cap 0.1 | 0.001 | 40.19570 |
| Adaptive shrinkage, cap 0.1 | 0.003 | 102.89201 |
| Adaptive shrinkage, cap 0.1 | 0.009 | 118.43212 |

All three Shampoo variants selected the lower LR boundary. One additional
geometric step, **0.001 / 3**, was therefore registered for each before seeing
its result, within the same family caps. All three additional runs completed:

| Method | Additional matrix LR | Development PPL |
|---|---:|---:|
| Parent Shampoo variant | 0.0003333333 | 43.69484 |
| Fixed shrinkage 0.1 | 0.0003333333 | 44.61723 |
| Adaptive shrinkage, cap 0.1 | 0.0003333333 | 44.52185 |

The best tested LR is therefore 0.001 for all three variants, with worse tested
rates on either side. Muon's best tested LR is 0.025, also bracketed. This only
resolves the boundary issue for these LR grids; other hyperparameters and
longer horizons were not exhaustively tuned. At their best tested settings,
adaptive is **0.87622 PPL worse than its parent** and **6.53571 worse than Muon**.
It is 0.25605 better than fixed shrinkage, which is insufficient for promotion.

The existing result already rejects promotion of the default adaptive setting:
it is worse than its identical parent at every shared LR in the original grid.
This is not a claim that every possible adaptive coefficient, Shampoo variant,
or training horizon must fail. This short, single-seed screen does not establish
a statistical comparison or the final 3,250-step ranking.

At LR 0.001, the final applied adaptive coefficients across all **10,368**
left/right factor tiles have mean **0.05783**; **38.03%** are at the 0.1 cap.
These are coefficients applied at the last root refresh, **step 991**, not a
training-time average. The rule is active and varies between factors; these
diagnostics do not establish that it successfully separates noise from drift.

The reserved validation slice and five confirmation seeds remain unopened.
The **0.5 absolute held-out PPL improvement target has not been achieved**.

## Budget, disposition, and retained artifacts

All **15 LM runs completed** and all research allocations are settled. The
first failed verification cost 0.03778 GPU-hours; successful GPU verification
cost 0.33556. Screening cost 8.08889, split into 1.64667 for Muon, 2.13778 for
the parent, and 4.30444 shared by fixed/adaptive shrinkage. Every family remains
below its 5.6-hour cap. Total charged cost is **8.46222 H200 GPU-hours**, with
**zero reserved** and **55.53778 remaining** in the authorized study. No study
jobs or background research controller are left running.

This adaptive version is **not promoted** to full-horizon development,
confirmation, or Llama transfer. Its code and negative results remain available
for reproduction. A future uncertainty proposal must address the demonstrated
noise/drift confounding and the mismatch between factor Frobenius error and
useful inverse-root updates; reducing estimated uncertainty alone is not an
adequate success criterion.

- [Complete 15-run summary, per-run cost, common-identity check, and final coefficients](../outputs/gpt2-research-20261016-v1/adaptive/final/adaptive/lm-screen-summary-expanded.json)
- [Original 12-run grid record](../outputs/gpt2-research-20261016-v1/adaptive/final/adaptive/lm-screen-summary.json)
- [Decision recorded before completing the original LR grid](../outputs/gpt2-research-20261016-v1/adaptive/final/adaptive/lr-grid-decision.json)
- [Decision recorded before the lower-boundary tests](../outputs/gpt2-research-20261016-v1/adaptive/final/adaptive/lower-boundary-decision.json)
- [Final allocation ledger](../outputs/gpt2-research-20261016-v1/adaptive/final/budget.json)
- [Exact source hashes](../outputs/gpt2-research-20261016-v1/adaptive/final/adaptive/source-manifest-r1.json)

The final artifact directory also contains every run's metadata, validation
history, result, Slurm log, and applicable rank-local diagnostics, plus the exact
source snapshot. Weight checkpoints remain on nano4 under the remote study's
`runs/`; they were not duplicated into the local notes directory.
