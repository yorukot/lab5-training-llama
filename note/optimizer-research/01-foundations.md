# 01 — What an optimizer does

Phase 1 of the optimizer research plan. Goal: a shared vocabulary for every later candidate, and a
map of what our 1B experiments already say about each part of the optimizer.

## The job

Each step the trainer computes one minibatch gradient `g_t = ∇L_B(θ_t) = μ_t + ξ_t`: the true
(population) gradient `μ_t` plus sampling noise `ξ_t` whose covariance shrinks as `Σ/B`. The
optimizer turns the history `g_1..g_t` into a parameter change. Its only goal here is the **final**
eval loss after a fixed token budget (1B tokens = 7,632 steps of 131k tokens at 1B; 6B for the
final run), at one forward/backward per step and a small memory and compute overhead.

Every optimizer we use or read about fits one template:

```
m_t      = estimator(g_1..g_t)                        # which signal to follow
u_t      = geometry(m_t)                              # how the signal becomes a direction
θ_{t+1}  = (1 − η_t λ) θ_t − η_t u_t                  # step size η_t, decoupled weight decay λ
θ_eval   = output(θ_1..θ_T)                           # which weights we evaluate
```

## Five components

| Component | Question it answers | Examples | Ours (BU) |
|---|---|---|---|
| Estimator | which signal to follow | EMA momentum, Nesterov, two EMAs (AdEMAMix), variance reduction (MARS) | Nesterov EMA, β = 0.95 |
| Geometry | how a signal becomes a direction | identity (SGD), diagonal (Adam, Sophia), Kronecker (Shampoo, SOAP), spectral (Muon) | Polar Express + NorMuon row norm on block matrices; AdamW on the embedding and norms |
| Step size | how far, per group, over time | warmup, WSD, cosine, per-layer lr, μP | WSD: 1% warmup, flat, linear to 0 over the last 80%; two lr groups; every matrix update scaled to RMS 0.2 |
| Regularization | where the weights settle | decoupled wd, cautious wd, norm constraints | decoupled wd, τ = 1/(η·λ·T) = 0.39 for every group |
| Output | which weights we evaluate | last iterate, EMA / Polyak averages | last iterate |

### Estimator

`src/muon.py:muon_update`: `m_t = β m_{t−1} + (1 − β) g_t`, update input
`ũ_t = (1 − β) g_t + β m_t` (Nesterov). With independent noise of variance σ², the Nesterov
input has variance about `0.030 σ²` at β = 0.95, i.e. the noise of ~33 averaged batches (~4.3M
tokens at 131k). The cost is lag: `m_t` points where the gradient was ~20 steps ago.

### Geometry

- AdamW (`src/adamw.py`): `u = m̂ / (√v̂ + ε)`, entry by entry; close to `sign(m)` when noise is
  small. This is steepest descent in the max-norm (ℓ∞).
- Muon: for `ũ = U Σ Vᵀ`, `u = U Vᵀ`: every singular value set to 1. This is steepest descent in
  the spectral norm (see 02-math.md). Computed without an SVD by 5 bf16 Polar Express quintic
  iterations (`zeropower_via_newtonschulz5`).
- NorMuon (`normuon_equalize`): divide each output row of `U Vᵀ` by the root of an EMA (β₂ 0.95)
  of its mean square, then restore the Frobenius norm. A diagonal left preconditioner on top of
  the spectral direction.
- Scale: `u × 0.2·√max(m, n)`, so every matrix's update has entry RMS 0.2 (Moonlight), the same as
  AdamW's typical update, letting the two share lr and wd conventions.

### Step size

`η_t = η · s(t)`, WSD shape. Muon lr 1.5e-3, AdamW lr 8.5e-3 (embedding and norms). Inside the
Muon group every matrix gets the same entry RMS: there is no per-layer-type step allocation.

### Regularization

`θ ← (1 − η_t λ) θ` before the step. With updates of fixed size and roughly orthogonal to `W`, the
norm settles where shrinkage balances growth: `‖W‖² ≈ η‖u‖² / (2λ)`, so each step changes `W` by a
relative `≈ √(2ηλ)` (≈ 2.6% at BU's peak Muon lr). τ = 1/(η λ T) is the fraction of the run over
which the weights forget old updates; 0.39 at every scale so far.

### Output

The last iterate after the linear decay to 0. No weight averaging has been tried.

## Sensitivity map: what our experiments say about each component

PPL changes are the final 1B in-training eval vs the run each note compares against.

| Component | Change | Run | Effect |
|---|---|---|---|
| Geometry | AdamW → Muon (same lr) | AF vs AC | Muon −8.3% |
| Geometry | Muon → NorMuon + Polar Express | AH / AJ / AM | −1.7% (wd 0.1), −1.35% (wd 0.4), −0.76% (wd 0.8) |
| Geometry | block matrices on Sophia (diagonal curvature) | BF, BG, BJ, BK | diverged, or +11.2% at best |
| Estimator | Muon momentum 0.95 → 0.98 | BD | +0.88% (lags) |
| Estimator | Muon momentum 0.95 → 0.9 | BX | +0.09% (ahead early, behind in the decay) |
| Estimator | NorMuon row EMA 0.95 → 0.9873 | BZ | pending |
| Step size | lr ladder 6e-4 → 1.2e-3 / 2.4e-3 | AD / AE | −6.9% / −5.1% |
| Step size | AdamW / Muon lr ratio up (1 → 5.7) | AW, AX, AY, AZ | −1.6% in total |
| Step size | both lrs ×1.25 / ×1.5 | BQ / BV | −0.31% / tie; best ×1.23 |
| Step size | warmup 5% → 1% / 0.25% | BR / BW | −0.52% / +0.14% |
| Step size | decay over last 50% → 80% / 95% | BE / BL | −0.70% / +0.37% vs 80% |
| Step size | linear → 1−sqrt decay | AQ | +0.60% |
| Step size | norms' lr halved | CA | pending |
| Noise (batch) | 524k → 262k → 131k tokens a step (τ fixed) | AT, AU | −2.96%, −3.89% |
| Noise (batch) | 131k → 65k (lr not retuned) | AV | +17.7% |
| Noise (batch) | 131k → 262k on the tuned recipe | BO | +1.53% |
| Regularization | τ 4.4 → 1.09 → 0.55 → 0.27 | AG, AK, AN | −2.7%, −2.0%, then flat |
| Regularization | embedding wd 0 | AR | +0.11% (noise) |
| Regularization | cautious wd | BA | −5% mid-run, +0.04% at the end |
| Init | zero-init output projections | BB | −4.4% early, +0.47% at the end |
| Geometry + step (150M) | hypergradient lr per matrix (HG-NorMuon) | src/exp | +0.77% |
| Geometry (probe) | TCD1 noise-Gram metric | src/exp | wins only on a constructed probe |

Seed noise at 1B: 0.02% (BE vs BM, seeded); gaps of 0.1% or more are real.

## What the map says

1. **The large wins came from three components**: geometry (AdamW → Muon ~8%, NorMuon ~1%),
   noise / step count (524k → 131k batch ~4%), and the regularization timescale (~5% in total).
   Schedule tuning gave 0.3–0.7% per change and is now settled.
2. **NorMuon's gain shrinks as wd grows** (−1.7% → −0.76%). Both act on the same thing: the
   relative size of updates across rows and the norm the weights settle at. A new geometry has to
   be judged at the tuned wd, not at a weak one.
3. **We are step-limited and noise matters**: more, noisier steps helped down to 131k tokens; at
   65k the noise wins (at an untuned lr). A rule that gets more out of each noisy step should help;
   a longer momentum that only averages more (BD) loses to lag.
4. **The decay erases mid-run leads** (BA, BB, BX, AQ, BL). Only the loss after the full decay
   counts; early leads usually mean "a lower effective lr", not a better optimizer.
5. **Entry-by-entry curvature (Sophia) loses to matrix geometry** at this scale.
6. **Untested so far**: the output iterate (no averaging), per-layer-type step sizes inside the
   Muon group, noise-aware treatment of singular directions, lag-free variance reduction of the
   estimator, the embedding's geometry (AdamW only), per-head geometry for q/k/v.
