# 05 — Results

Runs of the optimizer research, newest round last. 150M runs: `src/exp/screen.sbatch` with a config
from `src/exp/screen/`, code snapshot and outputs in
`/work/u9603854/lab5-training-llama/experiments/screen150m-v1/`. Metric: final dev PPL (the 20% dev
split of the eval documents, token-weighted), 300M tokens, seed 42 unless stated.

## Round 1 — tune the 150M baseline, first look at the candidates

Setup shared by every 150M run unless stated: NorMuon + Polar Express on the block matrices, AdamW
(β2 0.9873) on the tied embedding and norms at lr 6.8e-3, integrated decay `C = Σ −log(1 − η_t λ)`
= 1.5 for both groups (≈ τ 0.39, so wd follows lr), momentum 0.95, 1% warmup, linear decay over
the last 80%, 131k tokens a step (16 × 8192), 2,289 steps. `src/exp/train.py` now builds
`LabOptimizer`; on CPU it matches the old 150M NorMuon bit for bit (`exp.test_candidates`), so the
earlier lr150m results carry over.

Known points (lr150m study, same code path, data and split), Muon lr → dev PPL at AdamW 6.8e-3:
0.99e-3 → 29.54, 1.2e-3 → 29.28, 1.5e-3 → 28.93, 1.8e-3 → **28.74** (top of that sweep; AdamW
1.02e-2 there gave 28.80, so the AdamW lr is flat around 6.8e-3 and stays fixed).

Baseline sweep, why: 28.74 sat at the edge of the sweep and each ×1.2 had gained 0.4–0.9%, so the
optimum is higher; a candidate compared against an under-tuned baseline can win by lr alone. All
three carry `"probe": true` (checkpoints every 10% plus pairs 20 and 100 steps after the 20/50/80%
saves, a weight-norm log, tail EMAs over the last 1% and 4% read out at the end). base-lr1.8e-3
reruns the known point in the new code (a reproducibility check and one sample of run-to-run noise).
Expected: best near 2.5e-3–3.6e-3, about 28.4–28.6; if 3.6e-3 is still best, 5.0e-3 next.

- **base-lr1.8e-3**: results: dev PPL 28.685; the same config in the old code gave 28.741, so run-to-run noise at a fixed seed is ~0.2% (one pair; GPU nondeterminism). Tail EMAs: 1% window 28.703 (+0.06%), 4% 28.750 (+0.23%): no gain after a decay to 0, as predicted; tail averaging is dropped.
- **base-lr2.5e-3** (×1.4): results: 28.776 (+0.32%), ahead only in the first 20%.
- **base-lr3.6e-3** (×2): results: 28.977 (+1.0%). With the old sweep (1.5e-3 → 28.93) the best Muon lr is ~1.8–2.2e-3: the 1.8e-3 baseline is already about tuned, against the expectation. The Sobol point at 28.54 (lr 3.47e-3) differed in momentum (0.9), AdamW lr (3.8e-3) and decay settings, so it is not a plain lr effect.

Candidates, a first look at the untuned baseline's lr (1.8e-3), as the user asked: a candidate
that loses here is not dropped on that alone, since its best lr may differ (head damping is
predicted to want a higher lr, EMA-Nesterov a lower one); the curve and the predicted lr shift
decide whether it gets its own lr points at the tuned baseline's lr in round 2. Each compares with
base-lr1.8e-3 (code: `src/ema_nesterov.py`, `src/head_damping.py`, variants in `src/muon.py`,
per-type groups in `src/optimizer.py`; tests in `src/exp/test_candidates.py`).

- **ema-lr1.8e-3**: EMA-Nesterov on every parameter, γ 0.99, β = 0.6·η_t/η_peak over 15–74% of
  the run (Track 3 #39's setting). Expected −0.3% (0 to −0.8%), ahead from ~20% on. results: **28.536 (−0.52%; −0.62% vs the mean of the two 1.8e-3 baselines)**; ahead from 30% on (−2.1% at 30%, −1.7% at 50%, −2.0% at 80%, −1.2% at 90%), the lead shrinking in the decay but holding. Not an lr effect: raising the baseline's lr hurt (2.5e-3: +0.32%), while the lookahead, which lengthens only persistent motion, helped. Best candidate; ~3σ of the noise above.
- **long-lr1.8e-3**: NorMuon per column on wide matrices (k, v, down). Expected −0.1%.
  results: 28.641 (−0.15%), ahead all the way (−1.0% at 30%, −0.6% at 50%, −0.15% at the end); within noise, needs seeds.
- **damped-lr1.8e-3**: head damping γ 5 after NorMuon, cosine ramp over the first 30%. Expected
  −0.15%, possibly behind at this lr. results: 28.736 (+0.18%), ahead mid-run (−0.9% at 30%), behind after 80%; within noise. Predicted to want a higher lr; one point at 2.5e-3 decides.
- **mlp1.4-lr1.8e-3**, **attn0.7-lr1.8e-3**: per-type rotation (lr and wd ×1.4 on MLP matrices;
  ×0.7 on attention). Expected within ±0.3%; at an untuned global lr, mlp ×1.4 is partly a global
  lr rise, so it is read against the base sweep. results: mlp1.4 28.736 (+0.18%; +2.6% behind at 50%, caught up in the decay); attn0.7 **29.263 (+2.0%)**, behind all the way. Attention wants more rotation, not less: the opposite arms (attn ×1.4, mlp ×0.7) are worth one run each.
- **pe-lr1.8e-3**: Polar Express without NorMuon's row scaling: splits NorMuon's gain into its
  two parts (and gates long-lr). results: **29.474 (+2.75%)**: NorMuon's per-unit scaling (at 150M it acts mainly on gate/up's rows, the FFN units) is worth 2.75% here, far more than Polar Express alone. So per-unit normalization matters, which supports extending it (long-lr).

## Round 2 — confirm EMA-Nesterov, give the others a fair second look

From here the EMA-Nesterov switch is its largest β (`"ema_nesterov": 0.6`; round 1's `true` ran
with β 0.6 hardcoded). Seed noise at a fixed seed is ~0.2%, so a single-seed gain under ~0.4% is not
yet a result; seeds 43 and 44 change both the init and the data order and are compared in pairs.

- **base / ema / long at seeds 43, 44** (6 runs): is EMA-Nesterov's −0.5% (and long's −0.15%)
  real? Expected: ema ahead at every seed by 0.3–0.7%; long within ±0.2%.
- **ema-lr1.5e-3**: the lookahead lengthens persistent steps, so its best lr should sit lower than
  the baseline's. Expected: about level with ema at 1.8e-3, better would mean the gain grows.
- **ema0.3-lr1.8e-3**: Track 3 went from β 0.6 (#39) to 0.3 (#40) on another base. Expected: a
  smaller gain than 0.6.
- **damped-lr2.5e-3**: head damping should move the best lr up; compare with base-lr2.5e-3
  (28.776) and base-lr1.8e-3. Expected: within ±0.2% of base-lr1.8e-3; worse ends the card.
- **attn1.4-lr1.8e-3, mlp0.7-lr1.8e-3**: round 1 showed attention wants more rotation (attn ×0.7:
  +2.0%), so try the opposite arms. Expected: attn ×1.4 −0.1 to −0.5%; mlp ×0.7 within ±0.3%.
- **ema-long-lr1.8e-3**: do the two winners add? Expected: about ema's gain plus long's.

Round 2 results (dev PPL; % vs the baseline at the same seed and lr unless stated):

| Run | s42 | s43 | s44 | mean |
|---|---|---|---|---|
| base-lr1.8e-3 | 28.685 | 28.683 | 28.833 | 28.734 |
| ema (β 0.6) | 28.536 (−0.52%) | 28.606 (−0.27%) | 28.676 (−0.55%) | 28.606 (**−0.45%**) |
| long | 28.641 (−0.15%) | 28.802 (+0.41%) | 28.875 (+0.15%) | 28.773 (+0.14%) |

- **EMA-Nesterov is confirmed**: better at all three seeds, paired mean −0.45% (sd 0.15%, t = 5.1
  with 2 df, p ≈ 0.04), ahead from 30% on at every seed, the lead shrinking in the decay. Seed-to-
  seed spread of the baseline itself is 0.3%, so paired seeds are what make this visible.
- **β 0.3 ties β 0.6** (ema0.3: 28.537 vs 28.536): the gain does not grow with the lookahead
  between 1.43× and 2.5× amplification.
- **A lower lr hurts it** (ema-lr1.5e-3: 28.734, +0.69% vs ema at 1.8e-3), against the prediction
  that the lookahead moves the best lr down; its best lr is at 1.8e-3 or above.
- **ema + long** (s42 only): 28.468, −0.76% vs base, −0.24% vs ema alone; long alone is +0.14% over
  three seeds, so this extra is within noise and needs seeds before it means anything.
- **long-side NorMuon: no gain** (+0.14% over three seeds). Dropped.
- **head damping at 2.5e-3**: 28.698, −0.27% vs base-lr2.5e-3 (28.776) and +0.05% vs the tuned
  base-lr1.8e-3: its best lr does move up, as predicted, but it only matches the tuned baseline.
  Parked.
- **per-type rotation: both types already at their best**: attn ×0.7 +2.0%, attn ×1.4 +0.05%,
  mlp ×0.7 **29.400 (+2.5%)**, mlp ×1.4 +0.18%. Less rotation hurts either type a lot, more is
  neutral. Dropped.

Budget so far: 21 runs × ~0.47 = ~10 H200-h.

## 1B confirmation — EMA-Nesterov on the Lab 5 ladder (note/CF.md, note/CG.md)

BU + EMA-Nesterov (β_max 0.6, window 15–74%), two EMA lengths:

| Run | γ | dev PPL | vs BU (17.027) | curve |
|---|---|---|---|---|
| CF | 0.99 (~100 steps) | **16.921** | **−0.62%** | +6.0% at 20% (switch-on jump), level by 55%, −1.9% at 80% |
| CG | 0.997 (~333 steps) | 16.936 | −0.53% | smooth, −2.9% at 30%, −1.5% at 75% |

The 150M gain (−0.45%, three seeds) carried to 1B and grew. CF is the new best 1B run. Its jump
when β switches from 0 to 0.6 in one step is the obvious thing to fix next (a ramp). Budget so far:
~10 H200-h at 150M + ~19 H200-h for CF and CG.
