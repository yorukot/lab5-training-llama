# Adaptive trace shrinkage from temporal Gram uncertainty

Status: implemented mathematical mechanism and prior-art note, 2026-10-16.
The formula was frozen before the adaptive experiments. Executed evidence is
reported separately in [05-adaptive-results.md](05-adaptive-results.md); this
derivation does not establish a PPL improvement. All verification and
experiments run through `ssh nano4`.

The fixed-shrinkage study in [03-synthetic-results.md](03-synthetic-results.md)
was mixed: the existing coefficient 0.1 helped two designed regimes and hurt
the weak useful signal case. We therefore test whether an estimated
uncertainty-dependent coefficient can retain useful anisotropy when the factor
is reliable. The first adaptive coefficient is capped at **0.1**. A cap of 1
is a separately labeled diagnostic, not an automatically selected alternative.

The primary language-model objective remains at least 0.5 absolute held-out PPL
improvement at fixed training tokens. The GPT-2 phase remains capped at 64
allocated H200 GPU-hours; Llama execution is outside the current budget.

## 1. Exact proposal and connection to the current parent

Keep the blocked EMA, bias-corrected, Adam-grafted Shampoo parent described in
[02-mathematics-and-hypotheses.md](02-mathematics-and-hypotheses.md). Retain its
block size, momentum, inverse fourth powers, ten-step root refresh, ridge,
tensor-level Adam graft, learning-rate schedule and decay. This changes only the
scalar trace-shrinkage coefficient for each left/right block factor. It does not
shrink SOAP's Gram eigenvalues, which would leave an exact SOAP eigenbasis
unchanged.

For either factor let its current observation be
\(X_t=G_tG_t^\top\) or \(X_t=G_t^\top G_t\), of size \(d\times d\).
One observation means one globally aggregated training-step gradient Gram;
matrix columns, tokens and distributed ranks are not counted as independent
observations. The raw factor is

\[
C_t^{\rm raw}=\beta C_{t-1}^{\rm raw}+(1-\beta)X_t,
\qquad \widehat C_t=C_t^{\rm raw}/(1-\beta^t),\qquad \beta=0.95.
\]

Define the orthogonal trace-free projection

\[
\mathcal T(X)=X-\frac{\operatorname{tr}X}{d}I,\qquad
B_t=\mathcal T(X_t),\qquad \overline B_t=\mathcal T(\widehat C_t).
\]

Do not temporally center the gradients before building their Gram matrices.
Trace removal here isolates factor anisotropy; it does not turn a gradient
second moment into gradient-noise covariance.

For \(t\ge2\), set

\[
w_t=\frac{1-\beta}{1-\beta^t},\qquad
\Delta_t=B_t-\overline B_{t-1},
\]
\[
V_t=(1-w_t)\left(V_{t-1}+w_t\|\Delta_t\|_F^2\right),
\qquad
q_t=(1-w_t)^2q_{t-1}+w_t^2.
\]

Initialize \(V_1=0,q_1=1\), and use coefficient zero on the first observation.
The corrected mean already satisfies
\(\overline B_t=(1-w_t)\overline B_{t-1}+w_tB_t\).
For \(q_t<1\), define

\[
\widehat N_t=\frac{q_t V_t}{1-q_t},\qquad
A_t=\|\overline B_t\|_F^2,\qquad
\alpha_t=\operatorname{clip}\left(\frac{\widehat N_t}{A_t},0,0.1\right).
\]

The symbol \(A_t\) in this subsection is a scalar anisotropy energy, not the
left Gram factor. Use `anisotropy_energy` for the diagnostic to avoid that
ambiguity in code. The numerator is called **temporal uncertainty**, not a
measurement that separates stochastic noise from parameter drift.

Update the scalar scatter every step. At a scheduled root refresh apply

\[
\widehat C_t^{\rm shrunk}
=(1-\alpha_t)\widehat C_t
 +\alpha_t\frac{\operatorname{tr}\widehat C_t}{d}I.
\]

Then add the parent's existing ridge and compute its inverse fourth root. Cache
that root until the next scheduled refresh. Never overwrite the raw factor with
the shrunk factor. Distinguish the latest estimated coefficient from the
coefficient that produced the currently cached root.

Degenerate cases are explicit: coefficient zero for \(t=1\), \(\beta=0\),
\(d=1\), or zero temporal scatter. If current anisotropy energy is zero but
uncertainty is positive, use the cap; an exactly isotropic factor is unaffected
by this choice. If both are zero, use zero. Non-finite observations or states are
recorded failures, not silent resets to a different optimizer.

## 2. What is exact under a stationary model

The following calculation is our elementary Hilbert-space derivation of the
proposed statistic. It is not a claim to have invented covariance shrinkage.
Assume independent, identically distributed symmetric observations with
\(\mathbb E X_t=C\), and finite
\(v=\mathbb E\|\mathcal T(X_t)-\mathcal T(C)\|_F^2\).
For gradient Gram observations, this requires the relevant fourth gradient
moments to exist. Let \(\mu=\mathcal T(C)\).

Normalized EMA weights are

\[
a_{t,i}=\frac{(1-\beta)\beta^{t-i}}{1-\beta^t},\qquad
\sum_{i=1}^t a_{t,i}=1,
\qquad \overline B_t=\sum_i a_{t,i}B_i.
\]

The scalar \(q_t=\sum_i a_{t,i}^2\) has the exact closed form

\[
q_t=\frac{1-\beta}{1+\beta}
        \frac{1+\beta^t}{1-\beta^t},\qquad
n_{\rm eff}(t)=\frac1{q_t}.
\]

For \(\beta=0.95\), the limiting independent-observation effective sample
size is 39, not 20. This is a variance-equivalent observation count, not the
number of independent tokens in a batch or a guarantee of local stationarity.

### Weighted Welford identity

The recurrence maintains exactly

\[
V_t=\sum_i a_{t,i}\|B_i-\overline B_t\|_F^2.
\]

To see this, rescale all old weights by \(1-w_t\) and add the new weight
\(w_t\). The old observations' displacement from their old mean has weighted
sum zero. Since
\(\overline B_t-\overline B_{t-1}=w_t\Delta_t\), expanding the squared
distances gives

\[
V_t=(1-w_t)V_{t-1}
 +(1-w_t)w_t^2\|\Delta_t\|_F^2
 +w_t(1-w_t)^2\|\Delta_t\|_F^2,
\]

which is the recurrence in Section 1. No extra matrix mean is required: the
existing corrected Gram factor supplies it.

Independence and stationarity then give

\[
\mathbb E\|\overline B_t-\mu\|_F^2=q_tv,
\qquad
\mathbb EV_t=(1-q_t)v,
\qquad
\mathbb E\widehat N_t=q_tv.
\]

Thus the numerator estimates the squared error of the **mean anisotropy**, not
the total sample variance, per-token gradient variance, or curvature error.

### The oracle and the actual plug-in estimator are different

For a fixed, non-random scalar \(\alpha\), use the empirical trace as the
shrinkage target. The trace and trace-free matrix subspaces are orthogonal, so
the trace-estimation error is independent of the choice of coefficient in the
risk expression, even if the random trace and anisotropy are statistically
correlated. The Frobenius risk is

\[
\mathbb E\|\widehat C_t^{\rm shrunk}-C\|_F^2
=R_{\rm trace}+(1-\alpha)^2q_tv+\alpha^2\|\mu\|_F^2.
\]

The best constant coefficient in \([0,1]\) is

\[
\alpha_t^*=\frac{q_tv}{\|\mu\|_F^2+q_tv}.
\]

Also \(\mathbb E A_t=\|\mu\|_F^2+q_tv\). Replacing numerator and
denominator by their observed estimates motivates the proposed ratio. The
ratio, clipping and use of the same random history make the actual coefficient
sample-dependent: the constant-coefficient risk formula does **not** prove that
this plug-in rule beats raw or fixed-shrinkage estimates. The cap 0.1 is an
experimental restriction, not a consequence of the oracle derivation.

A separate per-realization diagnostic can use known synthetic truth:

\[
\alpha_{\rm realized}
=\operatorname{clip}\left(
\frac{\langle\widehat C-C,\widehat C-T\rangle_F}
     {\|\widehat C-T\|_F^2},0,\alpha_{\max}\right),\quad
T=\operatorname{tr}(\widehat C)I/d.
\]

That is a clairvoyant samplewise lower-envelope comparator, not an available
optimizer statistic, and differs from the best constant expected-risk oracle.

## 3. Noise, drift, dependence, and useful weak directions

**Drift is inseparable without additional information.** Write
\(B_i=\mu_i+\varepsilon_i\), where the noises are independent with zero mean
and common squared norm expectation \(v\), while \(\mu_i\) is deterministic.
Then

\[
\mathbb EV_t
=(1-q_t)v+\sum_i a_{t,i}\|\mu_i-\overline\mu_t\|_F^2,
\qquad \overline\mu_t=\sum_i a_{t,i}\mu_i.
\]

The estimator calls the second term uncertainty too. Even zero-noise changing
Gram matrices produce positive shrinkage. It cannot identify whether a recent
rotation, growing useful signal, minibatch noise or changing data caused the
variation. A single temporal stream does not identify these components without
a model or same-parameter independent replicates.

A concrete algebraic counterexample is a noiseless two-step PSD stream
\(X_1=sI\), \(X_2=sI+H\), with symmetric trace-free \(H\ne0\) small enough
to preserve PSD. Then \(\overline B_2=w_2H\) and
\(V_2=w_2(1-w_2)\|H\|_F^2>0\). The rule selects positive shrinkage, yet
the error against current truth \(X_2\) increases from
\((1-w_2)^2\|H\|_F^2\) to
\([1-(1-\alpha_2)w_2]^2\|H\|_F^2\). No stochastic noise exists. This is a
calculation, not a newly executed numerical result.

For a changing truth the constant-coefficient oracle's numerator would instead
include
\(q_tv+\langle\overline\mu_t,\overline\mu_t-\mu_t\rangle_F\), which can
be negative. Shrinking toward isotropy cannot repair every tracking error.
Synthetic reporting must therefore separate error against the instantaneous
population factor from error against its correctly weighted historical mean.

**Serial dependence changes variance.** In general,

\[
\mathbb E\|\overline\varepsilon_t\|_F^2
=\sum_{i,j}a_{t,i}a_{t,j}
  \mathbb E\langle\varepsilon_i,\varepsilon_j\rangle_F,
\]

which need not equal \(q_tv\). For an infinite stationary EMA with normalized
Frobenius autocorrelations \(r_k\), it becomes
\(qv[1+2\sum_{k\ge1}\beta^k r_k]\), when the series exists. Positive serial
correlation can make the nominal ESS optimistic. Packed sequential language
data and changing parameters do not satisfy the simple iid model by default.
No estimated autocorrelation correction is part of this first implementation.

**Useful weak signal remains a failure case.** Frobenius risk emphasizes large
matrix entries and aggregate eigenvalue error. It does not measure loss decrease,
inverse-root error, or the importance of a particular small eigenvalue. Shrinkage
raises small eigenvalues and can suppress the inverse-root gain that a useful
weak direction needs. A scalar uncertainty coefficient cannot distinguish this
from unreliable anisotropy. Even an oracle improvement in factor Frobenius risk
does not guarantee a better optimizer or held-out PPL.

Other required failure cases include heavy-tailed Gram observations without the
needed moments; sudden noise-scale changes; rank deficiency and repeated
eigenvalues; very early estimates; a one-dimensional block; and stale cached
roots. The inherited ridge controls numerical inversion, not these statistical
or optimization risks.

## 4. Prior art and why not substitute OAS blindly

- [Ledoit and Wolf, 2004](https://doi.org/10.1016/S0047-259X(03)00096-4)
  establish linear covariance shrinkage toward a scaled identity under
  high-dimensional assumptions and quadratic estimation loss. Our scalar
  variance-to-anisotropy ratio belongs to this established risk-balancing idea.
  The observations here are temporal Gram matrices, with EMA weights and
  training drift. Do not label this implementation the Ledoit-Wolf estimator
  or transfer its asymptotic theorem to transformer optimization.
- [Chen, Wiesel, Eldar and Hero, OAS/RBLW](https://arxiv.org/abs/0907.4698)
  study Gaussian covariance estimation. For iid zero-mean Gaussian vectors in
  dimension \(d>1\), equal-weight sample covariance \(S\), and sample count
  \(n\), their OAS coefficient is
  \[
  \min\!\left(1,
  \frac{(1-2/d)\operatorname{tr}(S^2)+\operatorname{tr}(S)^2}
  {(n+1-2/d)[\operatorname{tr}(S^2)-\operatorname{tr}(S)^2/d]}
  \right).
  \]
  Inserting \(1/q_t\) for \(n\) does not reproduce its Wishart model:
  weighted temporal Gram observations are not independent Gaussian vector
  outer products, and columns of a minibatch gradient are not independent
  samples. OAS with such a substitution would be a separate heuristic.
- [Oriol, weighted covariance shrinkage](https://arxiv.org/abs/2410.14420)
  explicitly studies weighted sample covariances, including exponential
  weights, through asymptotic nonlinear shrinkage. Weighted shrinkage is prior
  art; its full random-matrix model is not implied by a scalar ESS adjustment.
- [Tan and Zohren, covariance cross-validation](https://arxiv.org/abs/2012.05757)
  combine exponential weighting with nonlinear shrinkage selected by
  cross-validation in changing covariance settings. Their financial task and
  loss differ from optimizer design, but they establish relevant prior art for
  checking shrinkage out of sample instead of trusting an in-sample spectrum.
- [How Bregman Divergences Shape Shampoo, October 2026](https://arxiv.org/abs/2610.08534)
  connects structured approximation and finite-sample spectral estimation with
  Shampoo behavior. This proposal retains its existing parent factor estimates;
  it does not implement a new divergence-based Kronecker objective.
- [Minimax Optimality of Shampoo for Nonstationary Stochastic Optimization,
  October 2026](https://arxiv.org/abs/2610.16923) treats nonstationary tracking
  under specified assumptions. It motivates explicit drift tests; its theory
  does not certify this shrinkage rule or unconstrained language-model PPL.

These are sources for mechanisms and limitations. None establishes novelty or
effectiveness of the particular integration proposed here.

## 5. Controls, diagnostics, and economical implementation

Compare raw Shampoo, fixed shrinkage 0.1, and adaptive shrinkage capped at 0.1
with identical parent configuration, tuning budget, and paired randomness.
Keep cap 1 and oracle coefficients as labeled diagnostics. A lagged-coefficient
control uses the previous step's estimated scalar at a scheduled refresh; it
only removes that scalar's direct dependence on the current sample. Current
Gram factors and momentum can still depend on the current gradient, so this
control is **not** cross-fitting of the update.

A full lagged-factor control must form the entire geometry from previous-step
statistics. Under a conditional sampling model this avoids current-sample
geometry dependence but introduces staleness. A genuine same-parameter
cross-fit requires independent, equal-size fold gradients and applies geometry
from one fold to the other. Fold Gram expectations differ from full-batch Gram
expectations because gradient-noise scale depends on batch size; naive folding
changes the estimator target and is not a free equivalent control. These are
separate follow-up experiments, not silent additions to the current candidate.
Cross-fitting in preconditioned optimizers is existing work.
[Correcting Stochastic Update Bias](https://arxiv.org/abs/2605.20756)

The requested synthetic assessment separates two questions: factor-estimation
error for streams with known truth, and population objective progress for
optimization problems. Include stationary low/high noise, deterministic useful
weak signal, and abrupt/gradual drift; tune on development seeds and freeze
choices before independent evaluation seeds. Keep boundary learning-rate optima
and failed trials visible. A better factor estimate alone does not earn language
model promotion.

For each factor log `temporal_scatter`, `temporal_uncertainty`,
`anisotropy_energy`, `q`, `effective_samples`, unclipped coefficient,
estimated/applied coefficient, cap-hit status, and warmup/degenerate status.
Synthetic diagnostics additionally report instantaneous truth error,
EMA-target error, and both constant-risk and realized-oracle controls where
available. Training logs may aggregate factors to avoid prohibitive logging.

Persistent extra state is one scatter scalar per left/right block factor;
\(q_t\) can be computed from the shared step and \(\beta\), or maintained as
a shared scalar recurrence. Optional cached coefficient diagnostics are scalar
state. There is no extra persistent \(d\times d\) matrix or backward pass.
Per-factor work adds matrix subtraction, trace removal and squared-norm
reductions, \(O(d^2)\) operations and temporary storage, to already required
Gram construction. This is an operation-count claim; GPU launch, synchronization
and memory costs still require measurement.

Compute normalized weights and \(q\) with double-precision scalar arithmetic;
use a stable expression for \(1-\beta^t\) near \(\beta=1\). Compute the
trace-free difference explicitly before squaring rather than subtracting nearly
equal quantities \(\|X\|_F^2-\operatorname{tr}(X)^2/d\). Guard zero
denominators before tensor selection so unused branches do not create NaNs.
Preserve the parent's matrix work precision. Float arithmetic is not an exact
statistical or PSD certificate. State serialization must include the new scalar
statistics so continuation does not silently reset the estimator.

The next decision is empirical: does this cap-limited temporal rule improve the
parent on fresh synthetic cases without hiding its drift/weak-signal failures,
and then deliver enough fixed-token validation improvement to justify its
measured GPU cost? No such result is asserted in this note.
