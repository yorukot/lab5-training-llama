# 03 — Candidates

Phase 3 of the optimizer research plan. Five scouts (S1 random-matrix theory, S2 manifold / norm
geometry, S3 second-order methods, S4 filters and averaging, S5 training dynamics) proposed 15
cards; a red team checked the key papers and our own data against them; a feasibility review read
our code and costed each card. Full scout reports: `/tmp/.../scratchpad/cards/S1–S5.md` (session
scratch; the essentials are below).

## What the search changed in our picture

1. **Muon's flat spectrum is close to optimal in the stable phase.** About 95% of the Nesterov
   input's directions lie below the noise edge, yet they carry 10–64% of the signed alignment with
   a reference gradient (BulkBoost, arXiv 2610.07497, 576 cases, verified); random or inverted
   spectra match Muon (arXiv 2605.11181). Whole-run shrinkage of the "noise" directions is
   predicted to lose (+31% in S1's toy), consistent with TCD1's failure (modded-nanogpt, 1000 steps:
   best TCD1 PPL 84.9 vs Muon 33.7). See 02-math §3.
2. **NorMuon's row normalization barely acts on 5 of our 7 matrix types.** For out ≤ in (q, k, v,
   o, down) the polar factor has orthonormal rows, so the row EMA is constant up to Polar Express
   error. CPU check at our shapes: row-norm CV ≤ 0.7% for noise-like or column-scaled input, but
   5–31% when the momentum is ill-conditioned (5 Polar Express steps leave 11–37% of the singular
   values of square matrices below 0.7). So the NorMuon gains of AH/AJ/AM (−1.7% → −0.76%) are a
   bundle no run has separated: row scaling on gate/up, Polar Express itself (+14% update RMS at an
   under-tuned lr), and its approximation error.
3. **Weight norms and per-layer scale** (02-math §2, §4, corrected): consecutive updates are
   correlated, so the relative step is ~0.45%, not 2.6%; under decoupled wd a per-type scale
   constant sets the weight norm, not the step. A real per-type step is lr and wd scaled together.
4. **The fairest evidence is the modded-nanogpt Track 3 leaderboard** (copy in
   `src/exp/vendor/nanogpt_optimization/UPSTREAM_README.md`): tuned baselines, 8–20 seeds, linear
   decay to 0, τ ≈ 0.3. There NorMuon ≈ tuned Muon (3,250 steps each); Frobenius hyperball ties;
   one-sided Shampoo and Newton-Muon give no gain; EMA-Nesterov needs 3,125 steps vs 3,250 (−3.8%).
5. **Most published gains are against weak baselines**: wd = 0 (SAMuon, EMA-Nesterov), untuned
   wd, or cosine decay to 10%. Our recipe already has tuned τ and a decay to 0, which absorbs part
   of what these methods fix. Large geometry wins are unlikely; −0.1% to −0.3% per surviving card
   is the honest expectation.
6. **New 1B results**: BZ (NorMuon row EMA 0.95 → 0.9873) +0.12%; CA (norms' lr halved) +0.68%,
   so the norms want the high AdamW lr.
7. **The polar step is ~19–21% of the 1B step time** (from AF vs AD and BF; not measured
   directly). Splitting it across the 8 ranks would save ~16% of wall time (~9 GPU-h on the 6B
   run): a speed item, not a PPL one.

## Verdicts

| Card | Verdict | Reason |
|---|---|---|
| S4-A EMA-Nesterov (slow momentum in weight space) | **keep #1** | Only card with a gain against a tuned Muon at decay to 0 (Track 3 #39: −3.8% steps, n = 20). Frequency-selective, so no schedule reproduces it. |
| S3-2 long-side NorMuon (column normalization on wide matrices) | **keep #2** | Follows from finding 2; ~4 lines, zero FLOPs, free pre-run check. Modest prior. |
| S5-A head damping (SAMuon-lite), absorbs S1-B | **keep #3** | Numbers verified (1B −1.4% to −1.9% PPL), but wd = 0 and one seed; wd also caps the top singular value, so expect much less. |
| S5-B per-type rotation (lr and wd scaled together per type) | **keep #4** | Legitimate per-layer lr; the one per-group lr we tuned paid (−1.6%), the Muon group was never split. |
| S1-A late-decay soft polar | **conditional #5** | Theory + toy only; ≤ −0.1%. Run as a cheap tail branch only if the 90% checkpoint is noise-dominated. |
| S4-B tail EMA of weights | **free ride-along** | Expected tie after a decay to 0; measured on every run at no training cost. |
| S2-1 spectral sphere | park | Fair comparison only −0.39% at 240M against a Muon at its lr-grid edge; adds two knobs; we have no QK-norm. Revisit if q/k σ₁ grows (probe P3). |
| S3-3 Mousse (one-sided Shampoo) | park | A smooth form of head damping; Track 3 shows no gain; +7–10% step time at our batch. Only if #3 wins. |
| S1-C whole-run shrinkage | drop | Finding 1. |
| S2-2 Frobenius hyperball | drop | Ties in three fair comparisons. |
| S2-3 per-GQA-group polar for q | drop | Negative evidence; at 150M q splits into only 2 blocks. |
| S3-1 K-FAC input whitening | drop | Gain shrinks to "marginal" by 455M; Track 3 no gain; +5–8% step time, ~40 lines. |
| S4-C two-rail momentum | drop | No result against tuned NorMuon; BD/BX show the single-EMA trade-off is flat. |
| S5-C momentum rising in the decay | drop | BD lost more during the decay; Track 3 lowers momentum at the end. |

Extra zero-cost ideas from the red team: (a) a separate, shorter decay for the AdamW groups
(Track 3 #24: AdamW over the last 40%, Muon over 80%; CA shows the AdamW side is sensitive);
(b) a Polar-Express-only ablation at 150M, to split NorMuon's bundle and gate #2.

## Shortlist, gates and protocol

| # | Candidate | Rule (off = NorMuon) | Pre-run gate (kill if) | lr protocol at 150M | Honest 1B effect |
|---|---|---|---|---|---|
| 1 | EMA-Nesterov | `m ← (γ+(1−γ)β_t)m + (1−γ)Δ_t`, `p ← p + Δ_t + β_{t+1}m`; γ 0.99 (0.995 at 1B), β_t = 0.5·η_t/η_peak over 30–80% of the run, 0 elsewhere; wd kept out of `m` | path coherence C₁₀₀ < 2 in most matrices (P5) | ×{0.7, 1} at β 0.5, then β 0.3 at the better | −0.3% (0 to −0.8%) |
| 2 | Long-side NorMuon | normalize per column instead of per row when out < in | column CV of polar(ũ) < 0.1 on down, k, v (P1), or the Polar-Express-only ablation shows gate/up row scaling ≈ 0 | ×{1, 1.4}, ×0.7 only if ×1.4 loses | −0.1% |
| 3 | Head damping | `O ← O − (1−1/γ_t)(u₁ᵀOv₁)u₁v₁ᵀ` after NorMuon, γ 5, ramped over the first 30%; rank-r arm only if rank-1 wins | MLP loss-optimal step on bulk bands < 1.5× that on the head (P11) | ×{0.7, 1, 1.4} | −0.15% |
| 4 | Per-type rotation | `a_X·η` and `a_X·λ` for X ∈ {attn, mlp}; arms a_mlp 1.4, a_attn 0.7 | per-type edge ratios within 1.3× of each other (P11) | arms at the tuned lr, winner at global ×{0.8, 1.25} | −0.1% |
| 5 | Late-decay soft polar | stacked-matrix soft map `PE([ũ, εI])`, ε = 0.5·ê, norm-matched, only below 12.5% of peak lr | SNR > 2 or ℛ < 1.03 at the 90% checkpoint (P7) | tail branches from the 90% checkpoint at ×{0.7, 1, 1.4} | ≤ −0.1% |
| — | Tail EMA readout | EMAs over the last 1% and 4%, norm-restored, evaluated at the end | — | ride-along | tie |

Pass rule at 150M: the candidate at its best lr beats the baseline at its best lr by at least
max(0.15%, 2× seed σ); then 2 more seeds. Winners go to 1B against BU at the 150M lr ratio, then one
run with all winners combined.

## Step 0: what must exist before any candidate run

1. Point `src/exp/train.py` at `LabOptimizer` (`src/optimizer.py:build_optimizer`) instead of the
   HG-specific `src/exp/optimizer.py` (~15 lines), so 150M and 1B share one implementation.
2. Schedule access inside the optimizer: `group["lr"] / group["initial_lr"]` is η_t/η_peak, plus a
   `peaked` flag so warmup does not trigger the decay-phase gates (#1, #5), and `total_steps` for
   ramps (#3).
3. `src/exp/diagnose.py` (~150–200 lines): a `Probe` callback for the baseline run (weight norms
   every 10 steps, gradient autocorrelation ring buffer for one layer, tail EMAs, checkpoints every
   10% plus pairs at +20 / +100 steps around 20/50/80%) and a CLI that turns a checkpoint into the
   probe values P1–P13 (spectrum vs noise edge, SNR, ℛ, row/column CVs, Γ, stable rank, C₁₀₀,
   finite-difference curvature per direction band, λ_max). ~1.4 GPU-h for all probes.
4. Tuned 150M baseline: NorMuon at Muon lr ×{1, 1.4, 2} from 1.8e-3 (AdamW/Muon ratio, τ and
   momentum 0.95 as at 1B), all three with `Probe`; then 2 more seeds at the best; plus the
   Polar-Express-only ablation.

## Budget

| Item | Runs | H200-h |
|---|---|---|
| Step 0 baseline sweep, seeds, ablation, probes | 6 + probes | ~4.5 |
| Candidates #1–#4 at 150M | ~12 | ~6 |
| #5 tail branches | 3 short | ~0.5 |
| Confirmation seeds, combinations | ~9 | ~4.5 |
| 1B: winners at 2 lrs + one combined run | ~5 | ~45 |
| BU rerun with checkpoints (only if #5 or 1B probes need it) | 1 | ~9.5 |
| Reserve | | ~30 |
