# 02 — Optimizer mathematics, hypotheses, and evidence

Research date: **2026-10-16**. This document explains the mechanisms under test;
it does not claim that a new optimizer has already lowered language-model PPL.
Links were checked against primary papers, author implementations, and local
artifacts on that date. The literature review is targeted rather than exhaustive.
Recent preprints are hypotheses and reported evidence, not independently
reproduced results.

This note supplements the existing [foundations](../note/optimizer-research/01-foundations.md) and
[mathematical discussion](../note/optimizer-research/02-math.md), which are preserved unchanged.
Their BU/NorMuon experiments concern the historical Llama recipe; numerical
comparisons from that recipe do not transfer to the current GPT-2 study.
Several statements there need qualifications before they become research
assumptions:

- AdamW is generally a history-dependent diagonal preconditioner, not exact
  steepest descent in the max norm. That constrained direction is
  \(\operatorname{sign}(m)\); Adam's ratio approaches it only under additional
  assumptions about its first and second moments.
- A repeated-run difference of 0.02% cannot establish that every 0.1% gap is
  real. Paired fresh-seed uncertainty, configuration selection, and evaluation
  reuse matter. The historical relative-PPL target does not replace the approved
  absolute 0.5-PPL target below.
- A covariance reduction of \(1/B\) assumes independent samples. A quadratic
  expansion is an approximation unless the loss is quadratic; a Hessian can be
  indefinite, and a Newton direction is not always a local minimizer.
- A Marchenko–Pastur edge or a spiked-model separation threshold requires its
  noise and asymptotic assumptions. Empirical modes below such an edge are not
  proven useless, and finite-sample singular vectors are generally mixtures of
  signal and noise. Neither the number of small modes nor an assumed signal
  rank alone gives the fraction of useless optimizer energy.
- Momentum noise formulas require a stationary signal/noise model. Weight-norm
  equilibrium and iterate-averaging arguments require their stated independence,
  alignment, and timescale assumptions; they are not exact properties of every
  transformer layer or its training trajectory.

The approved objective is **at least 0.5 absolute held-out PPL improvement at
the same training-token budget**, with a total initial research cap of
**64 H200 GPU-hours**. Development uses the pinned 162,354,816-parameter
GPT-2-derived FineWeb benchmark: 3,250 steps and 1,703,936,000 target tokens.
Cost, stability, and memory remain required measurements. Llama 1B transfer
requires a separate costed study after confirmation. Historical 150M campaigns
remain retired.

## 1. What an optimizer does

Training changes model parameters so that observed next tokens become more
probable. For valid target tokens \(x_1,\ldots,x_N\), evaluation is

\[
L_{\rm eval}(\theta)=
  -\frac1N\sum_{i=1}^N \log p_\theta(x_i\mid x_{<i}),\qquad
\operatorname{PPL}(\theta)=\exp(L_{\rm eval}(\theta)).
\]

Accumulate the sum of negative log probabilities and the number of valid
targets before dividing. Averaging batch PPLs gives a different statistic.
For a fixed tokenizer and evaluation set, lower NLL and lower PPL rank models
identically, but a difference of 0.5 PPL is not a difference of 0.5 NLL.
At the reproduced PPL 26.50544, a reduction to 26.00544 corresponds to roughly
0.01904 nats/token. That single-seed number is an anchor, not the final baseline.

Backpropagation produces a stochastic gradient \(g_t\), indicating how the
current batch loss changes locally. The optimizer converts it and historical
state into a direction and a step:

\[
(s_{t+1},d_t)=\mathcal A(s_t,g_t),\qquad
\theta_{t+1}=(1-\eta_t\lambda)\theta_t-\eta_t d_t.
\]

The learning rate \(\eta_t\) controls displacement. Momentum averages recent
directions. Preconditioning changes relative displacement across coordinates.
Normalization changes update scale, and sometimes relative scales across
layers. Decoupled weight decay \(\lambda\) contracts parameters separately.
The schedule changes these choices over the finite training horizon.

These choices interact, but they are distinct interventions. A larger step
may learn faster initially and oscillate later; a lower training loss need not
produce lower held-out loss. Architecture, data, tokenizer, training objective,
and target-token accounting must remain fixed to attribute a measured change
to optimization.

### Geometry gives a concrete design language

For a symmetric positive-definite metric \(H\),

\[
d=\arg\min_z\left\{-\langle g,z\rangle+\tfrac12 z^\top H z\right\}
=H^{-1}g.
\]

Setting the derivative to zero proves this identity. Euclidean geometry gives
SGD; choosing \(H\) changes which displacements are expensive. A norm-constrained
alternative maximizes \(\langle g,d\rangle\) over \(\|d\|\le1\). Its maximum is
the dual norm \(\|g\|_*\). For a matrix \(M=U\Sigma V^\top\), the spectral-norm
constraint admits \(UV^\top\) as a maximizing direction, explaining the
idealized polar direction in Muon. Temporal averaging and numerical
approximation are additional operations.
[Bernstein and Newhouse, 2024](https://arxiv.org/abs/2409.20325)

Mirror descent instead chooses

\[
\theta^+=\arg\min_z\{\eta\langle g,z\rangle+D_\psi(z,\theta)\},\qquad
D_\psi(z,\theta)=\psi(z)-\psi(\theta)
 -\langle\nabla\psi(\theta),z-\theta\rangle .
\]

On an unconstrained domain its first-order equation is
\(\nabla\psi(\theta^+)=\nabla\psi(\theta)-\eta g\).
The quadratic choice \(\psi(z)=z^\top H z/2\) recovers the metric update.
A step-space regularizer, a parameter-space mirror map, and a learned
preconditioner are related design tools; they are not automatically the same
algorithm. Existing work already combines preconditioning and matrix norms.
[Preconditioned Norms, v3, 2026](https://arxiv.org/abs/2510.10777v3)

## 2. Derive the baseline update rules

All elementwise powers and divisions below are explicitly distinct from matrix
powers. Write \(g_t\) for a vector gradient and \(G_t\) for a matrix gradient.
The benchmark preserves upstream **sum** gradients. Mean gradients used in
the statistical derivations require the corresponding explicit conversion.

| Method | State and update | Mechanism and limitation |
|---|---|---|
| SGD | \(d_t=g_t\). | Cheap Euclidean descent; one scalar LR cannot correct arbitrary anisotropic curvature. |
| EMA momentum | \(m_t=\beta m_{t-1}+(1-\beta)g_t,\ d_t=m_t\). | A temporal filter; its lag can hurt when gradients turn. This EMA convention differs by a constant factor from an unnormalized momentum buffer. |
| AdamW | \(m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\ v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2}\); \(d_t=\widehat m_t/(\sqrt{\widehat v_t}+\epsilon)\), with \(\widehat m_t=m_t/(1-\beta_1^t)\) and similarly for \(v_t\). Apply decay separately. | A diagonal metric formed from uncentered gradient energy. It cannot represent arbitrary rotated interactions. |
| Muon, idealized | \(M_t=\beta M_{t-1}+(1-\beta)G_t\), \(N_t=\beta M_t+(1-\beta)G_t\); \(d_t=c_{m,n}\operatorname{polar}(N_t)\). | A spectral-norm direction using EMA-Nesterov input. Practical polynomial iterations do not produce an exact polar factor. |
| Matrix Shampoo | \(A_t=\epsilon I+\sum_{s\le t}G_sG_s^\top\), \(B_t=\epsilon I+\sum_{s\le t}G_s^\top G_s\); \(d_t=A_t^{-1/4}G_tB_t^{-1/4}\). | Two-sided structured adaptive scaling. The original cumulative rule differs from blocked, EMA, momentum, or grafted variants. |
| SOAP, schematic | Obtain bases \(Q_L,Q_R\) from Shampoo factors, project \(Z_t=Q_L^\top G_tQ_R\), form Adam moments in this basis, then unproject \(d_t=Q_L(\widehat m_t/(\sqrt{\widehat v_t}+\epsilon))Q_R^\top\). | Adam in a changing basis. State transport and refresh order are essential parts of a real implementation. |

Sources: [momentum](https://proceedings.mlr.press/v28/sutskever13.html),
[Adam](https://arxiv.org/abs/1412.6980),
[AdamW](https://arxiv.org/abs/1711.05101),
[Muon author implementation notes](https://kellerjordan.github.io/posts/muon/),
[Shampoo](https://proceedings.mlr.press/v80/gupta18a.html),
[SOAP](https://arxiv.org/abs/2409.11321).

For Shampoo, columnwise vectorization gives
\[
\operatorname{vec}(A^{-1/4}GB^{-1/4})
 =(B^{-1/4}\otimes A^{-1/4})\operatorname{vec}(G).
\]
This is an SPD preconditioner on the vectorized matrix. It keeps more structure
than a diagonal metric without storing an \(mn\)-by-\(mn\) matrix; it still
cannot express a general full-matrix covariance. Different published Shampoo
variants use different powers. Holding the parent variant's powers fixed is
necessary when testing a change in factor estimation.

The exact polar map sets each positive singular value to one and leaves the
zero subspace at zero under the canonical partial-isometry convention.
Practical Muon uses a finite polynomial with its own small-singular-value
behavior. In this repository the shape multiplier is
\(c_{m,n}=\sqrt{\max(1,m/n)}\); the exact full-rank scaled direction has
Frobenius norm \(\sqrt m\). Orthogonalizing an update does not orthogonalize
the model weights or estimate a Hessian.

### Curvature, noise, and gradient energy are different objects

At a fixed parameter let \(\mu=\mathbb E[g]\). Then

\[
C=\mathbb E[(g-\mu)(g-\mu)^\top],\qquad
S=\mathbb E[gg^\top]=C+\mu\mu^\top,\qquad
H_{\rm loss}=\nabla^2 L.
\]

The covariance \(C\) describes sampling noise. The uncentered moment \(S\)
contains signal as well. The Hessian describes local curvature and can be
indefinite. The model Fisher averages score outer products over labels sampled
from the model; an empirical Fisher substitutes observed labels. Under suitable
regularity, the model Fisher equals a model-distribution expected NLL Hessian.
This does not establish equality with the observed training-loss Hessian,
minibatch covariance, or Adam's moment state.
[Kunstner, Balles, and Hennig](https://arxiv.org/abs/1905.12558)

For independent examples, a mean over \(B\) examples has second moment
\(\mu\mu^\top+C_{\rm example}/B\); its outer product is not the mean of
per-example outer products. Packed language-model targets and sequential batches
need not be independent. Consequently, observing a small singular value or a
large gradient variance does not identify a low-curvature or useless direction.

### What would mathematically constitute an improvement?

Condition on past state and assume a \(K\)-smooth population objective \(L\).
Taylor's smoothness bound, followed by conditional expectation, gives

\[
\mathbb E[L(\theta-\eta d)]
\le L(\theta)-\eta\langle\nabla L,\mathbb E[d]\rangle
 +\frac{K\eta^2}{2}\mathbb E\|d\|^2 .
\]

Define \(a=\langle\nabla L,\mathbb E[d]\rangle\) and
\(b=\mathbb E\|d\|^2\). For \(a>0,b>0\), the bound's best scalar step is
\(\eta=a/(Kb)\), giving predicted decrease \(a^2/(2Kb)\). Thus meaningful
targets include better alignment, lower harmful variance, or a tighter
direction-dependent curvature bound. Simply multiplying all directions by a
constant changes the best LR, not this optimally rescaled bound.

Neither \(K\) nor the population gradient is available during LM training.
Data-dependent preconditioning requires analyzing \(\mathbb E[P(g)g]\), which
generally differs from \(\mathbb E[P(g)]\mu\). Stale momentum can have negative
alignment. SPD preconditioning guarantees a positive inner product with its
own input, but does not guarantee a safe finite step. For example,
\(L(w)=kw^2/2\) and \(d=pkw\), \(p>0\), yield loss multiplication by
\((1-\eta pk)^2\); the loss grows when \(\eta pk>2\).

Decoupled decay is not included in that displayed bound. Its actual
displacement uses \(d+\lambda\theta\), and the bound must be applied to that
whole direction when decay is present. With
\(W^+=(1-\eta\lambda)W-\eta cD\) and constant \(c>0\), the substitutions
\(\eta'=c\eta,\lambda'=\lambda/c\) reproduce the same real-arithmetic update.
For varying \(c_{t,l}\), this requires varying LR and decay per step and layer.
Record integrated contraction
\(-\sum_t\log(1-\eta_t\lambda)\) where each factor is positive.
These identities motivate LR/decay controls; they do not prove a generalization
benefit.

## 3. Current mathematical research and what it changes here

The papers below were checked on **2026-10-16**. The table distinguishes the
authors' research direction from the inference used in this project. Numerical
claims on other models or corpora are not transferred to our benchmark.

| Primary source | Relevant mathematics / evidence | Consequence for this study |
|---|---|---|
| [Optimal Shrinkage of Singular Values, 2014](https://arxiv.org/abs/1405.7511) and [OptShrink, 2013](https://arxiv.org/abs/1306.6042) | Random matrix theory supplies denoising rules under specified signal/noise models and matrix losses. | Denoising a matrix and choosing a useful optimization direction are different objectives. Check weak signal and correlated noise before importing a shrinker. |
| [PRISM, February 2026](https://arxiv.org/abs/2602.03096) | Proposes anisotropic spectral shaping using an innovation-augmented polar construction. | Spectral and noise-aware changes have direct prior art; gradient innovations also include temporal drift. |
| [Freon / Kaon, May 2026](https://arxiv.org/abs/2605.11181) | Tests Schatten-family and randomized spectra and emphasizes local alignment and descent potential. | Fine spectral detail cannot be assumed essential; evaluate scale-matched directions and simpler controls. |
| [SoftSignum / SoftMuon, May 2026, §2.4](https://arxiv.org/html/2605.31371v1#S2.SS4) | Smooth spectral maps use an algebraic soft sign and inverse-root computation, with a temperature schedule. | Our ridge soft-polar map belongs to this existing family. Fixed relative temperature is an experimental policy, not a new spectral invention. |
| [Stochastic update bias correction, May 2026](https://arxiv.org/abs/2605.20756) | Separates gradient–preconditioner coupling from bias caused by nonlinear inversion; proposes cross-fitting and variance correction. | Cross-fitting alone addresses only the first dependence. TCD does not implement the paper's inverse-bias correction. |
| [Dead-Direction Conditioners, June 2026](https://arxiv.org/abs/2606.29176) | Explores gauge-equivariant preconditioning and architecture symmetries. | A useful later direction, but verify exact symmetries of this architecture first. It is outside the initial three-mechanism implementation. |
| [BulkBoost, October 5, 2026](https://arxiv.org/abs/2610.07497) | Reports that many modes below an estimated noise edge still collectively align with a reference gradient; studies two-band reweighting. | This directly challenges indiscriminate suppression of small modes. Include low-amplitude useful signal, not only easy noisy-tail examples. |
| [How Bregman Divergences Shape Shampoo, October 6, 2026](https://arxiv.org/abs/2610.08534) | Studies how divergence choices interact with Kronecker approximation and finite-sample spectral error. | Factor shrinkage tests one estimation mechanism. It is not an implementation of all divergence-based variants in that paper. |
| [Nonstationary Shampoo analysis, October 12, 2026](https://arxiv.org/html/2610.16923v1) | Analyzes tracking, momentum/drift tradeoffs, and stabilization. Its tracking setup includes a compact convex domain, noise assumptions, and strong monotonicity. | Test changing gradients and stabilization sensitivity. Its minimax tracking statement is not a guarantee for unconstrained nonconvex transformer PPL. |

Mathematics does not establish priority by itself. Combining standard operations,
using a new name, or finding no identical search result is insufficient for a
novelty claim. The implementation can be useful even when its strongest method
turns out to be a replication or a minor variant of existing work.

## 4. Hypothesis register and exact mechanisms

Implementations live in [research_optimizers.py](../../src/exp/research_optimizers.py)
and the existing [gpt2_candidate.py](../../src/exp/gpt2_candidate.py).
The initial matrix methods
leave embeddings, output head, and scalar parameter groups on the reference
auxiliary AdamW recipe. All candidate hyperparameters must be in run metadata.
Method controls are baselines and ablations, not additional new mechanisms.

### H1 — Bounded soft-polar updates

**Question:** does retaining relative magnitude in weak singular modes improve
finite-token validation loss without merely shrinking the entire update?

Use the reference EMA-Nesterov input
\[
M^+=\beta M+(1-\beta)G,\qquad
N=\beta M^++(1-\beta)G,\qquad \beta=0.95 .
\]
For \(N\ne0\), set \(X=N/\|N\|_F\). With
\(X=U\operatorname{diag}(\sigma_i)V^\top\), define
\[
D_0=U\operatorname{diag}
 \left(\frac{\sigma_i}{\sqrt{\sigma_i^2+\tau^2}}\right)V^\top ,
\qquad D=c_{m,n}D_0,\qquad \tau=0.1\ \text{by default}.
\]
For \(N=0\), return zero. The map has singular gains in \([0,1)\);
it preserves singular vectors and aligns with \(N\). These facts do not imply
alignment with the population gradient. Frobenius normalization makes
\(\tau\) relative to the whole matrix, not an estimate of a statistical noise
edge. A fixed \(\tau\) can behave differently across shapes and effective ranks.

This is a **SoftMuon-family variant**: the published algebraic map is identical
under a reciprocal temperature and a rescaling of the input. The plan's initial
reference \(h(\sigma)=\sigma/(\sigma+\tau)\) has been replaced by the ridge
inverse-root map above to permit a simple matrix-product implementation.
It must not be labeled as implementing the rational filter.

For a wide \(X\), compute
\[
A=(XX^\top+\tau^2 I)/(1+\tau^2),\quad Y_0=A,\quad Z_0=I,
\]
\[
T_k=(3I-Z_kY_k)/2,\quad
Y_{k+1}=Y_kT_k,\quad Z_{k+1}=T_kZ_k,\quad
D_0\approx Z_{12}X/\sqrt{1+\tau^2}.
\]
Transpose tall inputs before the calculation and transpose back afterward.
The eigenvalues of \(A\) are positive and at most one, giving the standard
convergent inverse-root iteration in exact arithmetic. Twelve iterations are a
numerical approximation whose error must be measured at every admitted
temperature; finite-precision arithmetic is not a spectral certificate.
The implementation uses FP32 working matrices for normal training. Training
does not use SVD. Exact SVD is a small-matrix diagnostic.

For an eigenvalue \(a\), the relative inverse-root gain starts at
\(r_0=\sqrt a\) and follows \(r_{k+1}=(3r_k-r_k^3)/2\).
This proves \(0<r_k\le1\) and reveals the finite-iteration limitation:
at the smallest possible eigenvalue for \(\tau=0.01\), twelve iterations still
underestimate the inverse-root gain by approximately **10.67%**. The exact
zero singular direction contributes no update, but nearby nonzero directions
inherit this relative error. At \(\tau=0.1\), twelve iterations have negligible
exact-arithmetic error. Temperature and iteration count therefore both define
the implemented spectral profile. Production validation requires this worst-case
scalar relative error to be at most \(10^{-3}\): \(\tau=0.01\) with twelve
iterations is rejected and needs additional iterations. Admitted temperatures
are \([0.01,1]\), with at most twenty iterations.

An optional norm-matched control rescales the final nonzero \(D\) to Frobenius
norm \(\sqrt m\). This separates spectral shape from the nominal full-rank
polar update scale. The pre-rescaling spectral bound no longer applies after
this operation. The unmodified pinned Muon remains the parent control:
the exact \(\tau\to0\) polar limit does not equate two different finite
Newton–Schulz implementations.

**State/cost:** one momentum matrix, plus temporary matrices on the smaller
Gram side. For \(r=\min(m,n)\), \(q=\max(m,n)\), cost is
\(O(r^2q+kr^3)\), \(k=12\), with \(O(mn+r^2)\) working storage.

**Prediction and disconfirmation:** noisy small singular directions should
contribute less update energy; norm-matched validation improvement should
survive LR tuning. Failure on weak but useful low-curvature signal, decreasing
alignment with an independent reference gradient, or gains disappearing after
norm matching falsifies the intended explanation. Large \(\tau\) approaches an
RMS-like normalized momentum direction, which is a useful limiting control.

### H2 — Trace-preserving shrinkage before inverse matrix powers

**Question:** does reducing unreliable factor anisotropy improve a fixed
Shampoo variant at the same token budget?

The implemented parent is explicitly a **blocked EMA, Adam-grafted Shampoo
variant**, not a byte-for-byte reproduction of original cumulative Shampoo.
Blocking and grafting have existing scalable-Shampoo precedent.
[Scalable Second Order Optimization](https://arxiv.org/abs/2002.09018)
Partition each matrix into blocks of at most \(128\times128\). For each block
use unshrunk running factors
\[
A_t=\beta_2A_{t-1}+(1-\beta_2)G_tG_t^\top,\qquad
B_t=\beta_2B_{t-1}+(1-\beta_2)G_t^\top G_t,
\]
with \(\beta_2=0.95\), and bias-correct them by \(1-\beta_2^t\).
Maintain full matrix Adam moments with \(\beta_1=0.9,\beta_2=0.95\).
At a root refresh define, for a factor \(C\in\mathbb R^{d\times d}\),
\[
\bar c=\operatorname{tr}(C)/d,\quad
C_\rho=(1-\rho)C+\rho\bar c I,\quad
\widetilde C=C_\rho+
 10^{-6}\max(\bar c,10^{-12})I ,
\]
and form its inverse fourth root. Refresh every ten steps, beginning on the
first step; cache the roots between refreshes. Apply
\[
\widetilde D_b=\widetilde A_b^{-1/4}\widehat M_b
                    \widetilde B_b^{-1/4}.
\]
Concatenate block directions, then use the parent's common tensor-level Adam
graft:
\[
D_{\rm Adam}=\widehat M/(\sqrt{\widehat V}+10^{-8}),\qquad
D=\widetilde D\,\|D_{\rm Adam}\|_F/\|\widetilde D\|_F
\]
when the denominator is nonzero; a zero direction remains zero.
Finally apply the same decoupled weight decay and schedule. The parent uses
\(\rho=0\); the first shrinkage point is \(\rho=0.1\).
Shrunk factors must never overwrite the stored EMA statistics.

For PSD \(C\), shrinkage preserves trace and eigenvectors while changing
eigenvalues to \((1-\rho)c_i+\rho\bar c\). Positive \(\rho\) raises small
eigenvalues when \(\bar c>0\), reducing inverse-root sensitivity at those modes.
For a scalar eigenvalue \(x>0\),
\(|d(x^{-1/4})/dx|=\tfrac14x^{-5/4}\), illustrating why erroneous small
eigenvalues can matter. This is an estimation-versus-anisotropy tradeoff:
shrinking a well-estimated useful metric can hurt.

**State/cost:** moments plus two factors and two cached roots per block;
factor updates and applications are \(O(mnb)\) for block size \(b\), with
root refresh work of the same aggregate order for square blocks. Actual
eigendecomposition constants, block loops, memory, and distributed replication
must be profiled. The mathematical operation count does not establish acceptable
GPU overhead.

**Controls:** \(\rho=0\) must recover this parent exactly; keep the exponent,
blocking, factor cadence, graft, decay, and schedule fixed. Test repeated
eigenvalues, rank deficiency, and near-zero factors. Compare useful accurately
known anisotropy against noisy estimated anisotropy and include unmodified SOAP
as a published comparator when its implementation and cost are verified.
If all benefit is explained by a different tensor update norm, the geometric
explanation fails.

**Why not shrink SOAP's Gram spectrum instead?** If \(C=Q\Lambda Q^\top\),
then \(C_\rho=Q[(1-\rho)\Lambda+\rho\bar cI]Q^\top\).
For \(0\le\rho<1\), an exact eigensolver has the same eigenspaces and ordering;
if SOAP uses only those eigenvectors, post-hoc shrinkage changes no update.
At \(\rho=1\), the basis is arbitrary. The official implementation's later
refreshes use a power step followed by QR, so
\(\operatorname{qr}(C_\rho Q_{\rm old})\) can differ from
\(\operatorname{qr}(CQ_{\rm old})\). That changes finite-iteration basis
tracking, not the target eigenspace.
[Author SOAP implementation](https://raw.githubusercontent.com/nikhilvyas/SOAP/main/soap.py)

### H3 — Cross-fitted stochastic geometry, retaining TCD as a negative control

**Question:** can separating geometry estimation from its input gradient improve
the bias/variance balance? Cross-fitting is established prior art; the
repository's transpose-coupled operator is one experimental geometry.

Condition on past state. Four equal-token mean gradients \(g_1,\ldots,g_4\)
define
\[
a=(g_1+g_2)/2,\ b=(g_3+g_4)/2,\quad
e_A=(g_1-g_2)/2,\ e_B=(g_3-g_4)/2 .
\]
Under independent identically distributed groups, both means estimate
\(\mu=\nabla L\); each difference estimates the noise of its group mean.
This needs no Gaussian assumption. A group's mean and difference need not be
independent; independence is used **between** A and B.

The existing TCD1 operator is
\[
K_e(D)=ee^\top D+De^\top e+2eD^\top e,\qquad e=E/\|E\|_F ,
\]
\[
r=\|E\|_F^2/(\|E\|_F^2+\|S\|_F^2),\quad
\alpha=\rho r/(1+\rho r),\quad
c=4\|(ee^\top)^4\|_F^{1/4},
\]
\[
P=(I-\alpha K_e/c)/\tau,\qquad
\tau=\operatorname{Tr}(I-\alpha K_e/c)/(mn).
\]
Use the smaller Gram side after transposing as necessary; set \(P=I\) for
zero noise. Here the capital trace acts on the \(mn\)-dimensional linear
operator. The complete derivation and exact-solver distinction are preserved
in [the TCD notebook](../../src/exp/README.md#tcd-research-notebook--2026-10-16)
and [tcd.py](../../src/exp/tcd.py).

For the GPT-2 integration, with old momentum \(M\),
\[
N_A=\beta^2M+(1-\beta^2)a,\quad
N_B=\beta^2M+(1-\beta^2)b,\quad
M^+=\beta M+(1-\beta)(a+b)/2 ,
\]
\[
s_A=\max(\operatorname{RMS}(N_A),10^{-8}),\quad
s_B=\max(\operatorname{RMS}(N_B),10^{-8}),
\]
\[
D=\frac1{2\sqrt n}
\left(P_A N_B/s_A+P_B N_A/s_B\right).
\]
The divisor comes from the same fold as the geometry, opposite to the numerator.
The \(\rho=0\) control retains collection, Nesterov, fold scaling, and decay,
while setting \(P_A=P_B=I\).

Let \(A_A=P_A/(\sqrt n\,s_A)\) and similarly for B. Conditional independence
gives the exact identity
\[
\mathbb E[D\mid{\rm past}]
=\tfrac12(\mathbb E[A_A]+\mathbb E[A_B])
  [\beta^2M+(1-\beta^2)\mu].
\]
It removes a particular numerator–geometry dependence. It does not imply
\(\mathbb E[H^{-1}]=(\mathbb E[H])^{-1}\), remove momentum lag, establish
independent FineWeb batches, or guarantee lower PPL. Lagged geometry provides
another dependence control but introduces staleness. A same-fold rule provides
a coupling control. They must use the same total target tokens; extra
communication and gradient work still count in GPU-hours.

**Negative evidence is part of the hypothesis record.**

- The existing symmetric-signal synthetic artifact reports mean loss
  **0.02962 for Adam, 0.04906 for exact TCD, and 0.04915 for TCD1**.
  Several search coordinates end at a boundary; this is evidence against a
  universal benefit, not a final optimized ranking.
  [Recorded summary](../../outputs/tcd-probe-20261016-v4-symmetric/summary.json)
- The retained curvature counterexample
  \(L(W)=\tfrac12(W_{12}-1)^2+5000W_{21}^2\), with
  \(g=\left[\begin{smallmatrix}0&-1\\0&0\end{smallmatrix}\right]\),
  \(e=\operatorname{diag}(2,1)\), samples \((g+e,g-e,g+e,g-e)\),
  \(\beta=0,\eta=0.1,\rho=3\), increases loss from **0.5 to 7.55533**;
  the \(\rho=0\) control reaches **0.36858**. Correlated folds and the large
  step violate assumptions needed for the conditional small-step argument.
- The audited completed GPT-2 screen used seed 42 and 1,000 steps. Muon PPL
  was **35.5205 / 33.65865** at LR **0.00625 / 0.025**; LR 0.1 failed
  with an NCCL infrastructure error. TCD1 \(\rho=3\) produced PPL
  **84.89795 / 364.02407 / 334.21684**, and the identity-geometry control
  produced **48.51728 / 292.33488 / 298.50301**, over the same three LRs.
  Both TCD and its control fail, so these results do not isolate geometry from
  shared scaling/integration choices. The best LR is at the lowest boundary.
  The screen is already completed and must not be duplicated as new evidence.
  [Audited prior screen](../../outputs/gpt2-research-20261016-v1/prior-tcd-screen.json)

TCD is therefore deprioritized under the 64-hour cap. Revisiting it requires
much lower LR/scaling diagnostics and a reason that addresses these failures;
favorable hand-selected synthetic examples are insufficient. Its four-fold
collector also adds communication not present in the ordinary summed-gradient
baseline.

## 5. Evidence needed before claiming success

Correctness comes before language-model ranking. Required mechanism checks
include exact small-matrix references; zero/rank-deficient gradients;
ill-conditioned and rotated quadratics; useful weak signal; noise aligned and
misaligned with curvature; changing directions; mixed precision; state restore;
and distributed token weighting and parameter synchronization. A test that
only mirrors an implementation does not validate its mathematical claim.

Use the public first 10,485,760 validation targets for development. Preserve
the designated reserved target interval beginning at 80,000,000 and containing
10,485,760 targets for locked confirmation, after verifying shard integrity and
disjointness. A separate slice from the same shard is not a separate corpus.
Short runs screen methods; final selection requires the full 3,250-step
horizon. Comparable tuning budgets and LR boundary expansion matter because
optimizer rankings depend on tuning.
[Choi et al.](https://arxiv.org/abs/1910.05446)

Freeze one candidate, the strongest eligible development baseline,
hyperparameters, and evaluation before running fresh paired seeds
**101, 202, 303, 404, 505**. Match initialization and data order within each
pair. For \(\Delta_i=\operatorname{PPL}_{b,i}-\operatorname{PPL}_{c,i}\),
report every pair, the arithmetic mean \(\bar\Delta\), and the prespecified
paired 95% Student interval
\[
\bar\Delta\ \pm\ t_{0.975,4}\,s_\Delta/\sqrt5,\qquad t_{0.975,4}\approx2.776.
\]
Its normality/independent-seed assumptions and the small sample size must be
stated. Evaluation tokens do not substitute for independent training seeds.
Do not replace mean per-seed PPL with exponentiated mean NLL silently.

The point-estimate target requires \(\bar\Delta\ge0.5\) and the interval's lower
bound above zero, identical completed token budgets, and no unresolved
correctness failures. Claiming the underlying improvement is at least 0.5
requires the lower bound itself to exceed 0.5. Missing confirmation, a
cross-zero interval, or unfinished tuning gives an exploratory/inconclusive
outcome. Never add seeds after looking at the result merely to reach significance.

Reserve confirmation compute from measured allocation cost before screening.
Charge successful runs, failed runs, profiling, compilation, evaluation, and
idle allocated GPUs to the 64 H200-hour ledger. Preserve the historical
reference result and source hashes. Report slower optimizers' overhead
explicitly, even when they win on the primary fixed-token criterion. A negative
result with a correct implementation, a falsified mechanism, and reproducible
artifacts is a legitimate research result; it is not a lowered-PPL optimizer.
