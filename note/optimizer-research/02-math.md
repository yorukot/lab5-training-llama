# 02 — The mathematics of "better"

Phase 2 of the optimizer research plan. One framework every candidate must be argued in, so that
"why it should lower PPL" is a claim about a specific term, and a measurable one.

Notation: `θ` parameters, `L` population loss, `g = μ + ξ` minibatch gradient (true gradient `μ`,
noise `ξ` with covariance `Σ/B`), `H = ∇²L`, `u` the update direction, `η` the lr.

## 1. One step: signal, curvature, noise

For a small step, a second-order expansion and the expectation over the noise give

```
E[L(θ − ηu)] − L(θ) ≈ −η ⟨μ, E u⟩  +  (η²/2) (E u)ᵀ H (E u)  +  (η²/2) tr(H · Cov u)
                       signal term     curvature of the mean      noise term
```

An optimizer can lower the loss per step in three ways only: point `E u` more along `μ` (signal),
shape it so it does not climb steep directions (curvature), or put less variance into steep
directions (noise). Newton's `u = H⁻¹μ` is optimal for the first two and ignores the third.

## 2. Geometry: steepest descent under a norm

Choose a norm `‖·‖` and take the unit step that descends fastest:

```
u = argmax_{‖u‖ ≤ 1} ⟨m, u⟩
```

| Norm | Direction | Descent `⟨m, u⟩` |
|---|---|---|
| ℓ2 (Euclidean) | `m / ‖m‖₂` (SGD) | `‖m‖₂` |
| ℓ∞ (max entry) | `sign(m)` (≈ Adam, Lion) | `‖m‖₁` |
| spectral `‖·‖_op` | `U Vᵀ` for `m = U Σ Vᵀ` (Muon) | `‖m‖_*` (sum of singular values) |

The right norm is the one in which the loss is smooth: `L(θ + Δ) ≤ L(θ) + ⟨∇L, Δ⟩ + (K/2)‖Δ‖²`.
For a linear layer `y = W x`, `‖ΔW x‖_rms ≤ ‖ΔW‖_{rms→rms} ‖x‖_rms` with
`‖A‖_{rms→rms} = √(n_in / n_out) · ‖A‖_op`: the spectral norm bounds how much an update can move the
layer's output, which is why Muon fits matrix weights and diagonal rules (Adam, Sophia) do not.

### Per-layer scale is a choice, not a consequence

The polar factor `U Vᵀ` has spectral norm 1; the scale applied after it decides how far each layer
type moves. Three conventions, normalized to the square `q_proj` (1B shapes, `out × in`):

| Matrix | Shape | Ours (entry RMS 0.2, Moonlight) | Keller's Muon (`√max(1, out/in)`) | Spectral / μP (`√(out/in)`) |
|---|---|---:|---:|---:|
| q, o | 2048 × 2048 | 1 | 1 | 1 |
| k, v | 512 × 2048 | 1 | 1 | 0.5 |
| gate, up | 8192 × 2048 | 2 | 2 | 2 |
| down | 2048 × 8192 | 2 | 1 | 0.5 |

Our convention gives `down_proj` 4× and `k/v` 2× the spectral convention's step. No experiment has
tested this.

Correction (scout S2): under decoupled wd this is **not** a per-layer step. For a matrix the network
is scale-invariant in, `W ← (1 − ηλ)W − η·c·u` with Muon's scale-free `u` makes `W/c` follow the
`c = 1` run exactly, so `c` sets the matrix's equilibrium norm (∝ `c`) and the early transient, not
its relative step. Moonlight (arXiv 2502.16982, Table 1) shows it: doubling the MLP update scale
doubled the MLP weight RMS and left `q`'s unchanged. Since no Llama matrix is exactly
scale-invariant, changing `c` tests the **function scale** (residual mix of o/down, attention
temperature through q·k) plus the transient. A per-layer step is `η·λ` per layer (L3/L4), or the
angular step on a norm sphere.

## 3. Noise

### Noise floor of a linear rule (quadratic model)

On `L = ½ θᵀHθ` with update `θ ← θ − ηP(Hθ + ξ)`, the stationary excess loss for small `η` is

```
E[L] − L* ≈ (η/4) · tr(P Σ) / B
```

It depends on the noise measured in the preconditioner's geometry, not on `H`, and it falls with
`η` and with the batch `B`. EMA momentum does not change it to first order; it only changes the lag.

### Normalized rules (Muon) never stop moving

Muon's `u` has a fixed size however small `μ` gets. Near a valley floor the step is a direction of
fixed length, mostly noise, so the iterate jitters with an amplitude set by `η` alone. Only the lr
decay removes it, which is why the decay length and shape mattered so much (BE −0.70%, BL, AQ).

### What the polar map does to noise

Write the momentum matrix as signal plus noise, `M = S + N`, with `N`'s entries of variance `s²`
(m × n, m ≤ n). Random-matrix theory (Marchenko–Pastur) puts the noise singular values in
`[s(√n − √m), s(√n + √m)]`. A signal direction of strength `θ` separates from that bulk only when
`θ > s·(mn)^{1/4}` (BBP threshold); weaker ones are lost.

Muon sets **every** singular value to 1. If `S` has rank `r`, a share `(m − r)/m` of the update's
squared norm goes to noise directions, at the same gain as the signal directions. Muon still beats
Adam by 8%, so one of these must hold, and which one is an open question (diagnostic D1):

- the momentum's useful signal is not low-rank: most directions carry some signal;
- noise directions are cheap: they change sign from step to step and average out (a random walk
  grows like `√t`, a drift like `t`), while the equal gain helps weak signal directions;
- the flat spectrum acts as a useful regularizer.

If D1 shows a large share of directions inside the noise bulk at 50–90% of training, a spectral
shrinkage `u = U f(Σ) Vᵀ` (f ≈ 1 above the bulk edge, small inside) is a direct L2 candidate,
computable with odd matrix polynomials as Polar Express is, without an SVD.

Answer from the literature (scout S1): the first two hold. About 95% of the Nesterov input's
directions sit below the noise edge, yet they carry 10–64% of the signed alignment with a reference
gradient (BulkBoost, arXiv 2610.07497), and random or inverted spectra match Muon (arXiv
2605.11181). With gains `f_i` on directions of alignment `a_i`, the one-step loss is
`−η Σ f_i a_i + (η²/2) Q(f)`: a Frobenius-like curvature cost gives `f_i ∝ a_i` (optimal shrinkage),
a spectral-like one gives `f_i = 1` wherever `a_i > 0` (Muon). Muon's −8% says we are on the spectral
side in the stable phase. Shrinkage can only help where `a_i ≈ 0`, at the noise-dominated end of
the decay; a whole-run shrinkage is predicted to lose.

## 4. Weight decay sets the norm, and the norm sets the effective lr

With `W ← (1 − ηλ) W − ηu` and `u` roughly orthogonal to `W`, the norm settles where shrinkage
balances growth:

```
‖W‖² ≈ η ‖u‖² / (2λ)        relative step  ‖ηu‖ / ‖W‖ ≈ √(2ηλ)        reached in ~1/(2ηλ) steps
```

At BU's peak Muon lr: a 2.6% relative step, equilibrium reached in ~1,500 steps (~20% of the run).

Correction (scout S2): consecutive updates are **not** independent. Nesterov momentum at β = 0.95
correlates them; for white-noise gradients the lag-k correlation is about `0.88·β^{k+1}`, a sum
`C ≈ 16`. Then `‖W‖² ≈ η‖u‖²(1 + 2C)/(2λ)` and the relative step is `√(2ηλ/(1 + 2C))` ≈ 0.45%, and
`1/(2ηλ)` is only the e-folding time of `‖W‖²`, while the decay starts at 20% of the run. A
simulation of the norm recursion gives entry RMS rising from 0.02 (init) to ~0.055 by 20% of the
run, with the relative step falling 1.4% → 0.55% during the "stable" phase. So BU's angular step
is not `s(t) × constant`; diagnostic D4 measures which picture holds.
For a weight whose scale the network ignores (one followed by a normalization), only the direction
matters, the angular step is `√(2ηλ)`, and lr and wd act only through `ηλ`, i.e. through τ. That is
why τ, not wd alone, transferred across our runs. In Llama no matrix is exactly scale-free (`W_q`'s
scale is the attention temperature; `W_v W_o` scales a block's residual contribution), so the norm
itself is also a lever (L4): a direct norm constraint is an alternative to tuning τ.

## 5. The schedule as annealing

In the quadratic picture a direction of curvature `h` relaxes in about `1/(η h)` steps, and its
noise floor follows `η` only if `η` changes slower than that. The common picture of LLM loss
surfaces (a flat "river" along which progress is made, steep "valley" walls across it) explains our
schedule results: progress along the river needs high-lr steps (so the stable phase matters, BL),
and the decay removes the jitter across the valley (so a long decay matters, BE). A better
optimizer must improve one of the two at the end of the decay, not the loss mid-run.

## 6. Averaging

An average of the last `K` iterates cuts the across-valley noise roughly by `K` (for uncorrelated
jitter) but lags along the river. With a decay to 0 already in place most of this is used up; the
rest is cheap (one extra copy of the weights, no extra compute) and untested here.

## 7. Five levers

| Lever | Term it lowers | Done in our runs | Open |
|---|---|---|---|
| L1 direction | signal, curvature of the mean | Muon (−8%), NorMuon (−0.8% at tuned wd) | precondition-then-polar (Shampoo / SOAP style); per-head blocks for q/k/v |
| L2 noise | noise term, noise floor | batch 131k; momentum tuned (0.95) | spectral shrinkage; lag-free variance reduction (MARS-like); second slow EMA |
| L3 step allocation | curvature term per layer | two lr groups; uniform RMS 0.2 | per-layer-type scale (table in §2); curvature-based per-layer lr |
| L4 norm equilibrium | effective lr over time | τ = 0.39 for every group | per-layer τ; hypersphere / norm-constrained weights instead of wd |
| L5 averaging | noise floor at the end | last iterate only | EMA of weights over the decay |

## 8. What does not count as a new optimizer

If a candidate's update is the old one times a constant `c` per matrix, then lr `cη` and wd `λ/c`
reproduce it exactly (same trajectory in exact arithmetic). So:

- a constant rescaling is an lr change, not a new direction;
- a time-varying or per-layer rescaling is a schedule or a per-layer lr (L3), legitimate but to be
  compared against that, not against the untouched baseline;
- only a change in the **relative** gains inside a matrix (singular directions, rows, entries) or
  across time in a way no schedule reproduces is new geometry.

Every candidate is therefore run at lr ×{0.7, 1, 1.4} with τ held fixed, and compared at its best
lr against the baseline at its best lr.

## 9. Size of a real gain

Our ladder gives `ppl ∝ tokens^−0.20` (BN → BP). An optimizer worth `k`× the tokens lowers PPL by
a factor `k^−0.20`: 1.05× tokens ≈ −1.0% PPL, 1.10× ≈ −1.9%. Careful benchmarks report matrix
optimizers' advantage over tuned AdamW shrinking toward ~1.1× near 1B parameters, and NorMuon is
already a strong matrix optimizer. A realistic target is **−0.3% to −1.5% PPL** (worth 1.5–8% more
tokens); 0.3% is still 15× our seeded noise at 1B.

## 10. What every candidate card must state

1. The lever (L1–L5), the component it changes, and the term in §1 it lowers.
2. The update rule in equations, and how it reduces to NorMuon when switched off.
3. Evidence: paper, model size, gain, and whether the baseline was a tuned Muon.
4. Cost: FLOPs, memory, lines changed in `src/muon.py` / `src/optimizer.py`.
5. Failure modes, including the ones our map already shows (lag, mid-run-only gains).
6. A falsifiable prediction checkable with the diagnostics (D1–D4) before any training run.
7. How it transfers from 1B to the 6B run.
