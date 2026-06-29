# GP-EFE: Gaussian-Process auto-regressive active-inference agent

Derivation notes for extending the MARX-EFE agent (linear multivariate
auto-regressive eXogenous generative model) to a GP-AR agent (nonlinear
auto-regressive with a Gaussian-process transition density).

The driver (`run_episode_maxrefe`, `evaluate_candidate`,
`marxefe_optimize_cpg`) is unchanged. Only the generative model inside
`MARXAgent` — the predictive distribution, the Bayesian update, and the
expected-free-energy terms — needs to be replaced.

---

## 1. Motivation

The MARX likelihood

$$
p(\mathbf{y}_k \mid \mathbf{x}_k, M, \Sigma) = \mathcal{N}(\mathbf{y}_k \mid M^{\top}\mathbf{x}_k, \Sigma)
$$

is *linear* in the regressor $\mathbf{x}_k$ and pools information across all
input locations through a single shared matrix $M$ with precision $\Lambda$.
Two consequences:

1. With $N$ observations the posterior precision grows as $\Lambda_N = \Lambda_0 + \sum_k \mathbf{x}_k \mathbf{x}_k^{\top}$, i.e. the per-state predictive variance
   $$
   \mathrm{Var}(\mathbf{y}^{*} \mid \mathbf{x}^{*}, \mathcal{D}) \propto 1 + \mathbf{x}^{*\top} \Lambda_N^{-1} \mathbf{x}^{*}
   $$
   shrinks *uniformly over $\mathbf{x}^{*}$* once $N \gg D_x$ — every state looks
   well-explored to the agent, so the epistemic term flattens and the EFE
   gradient vanishes.
2. Nonlinear CPG/Laikago dynamics are approximated by a single linear map,
   so residuals are correlated and systematically under-modelled, biasing
   $M$ to whatever regime the first trial visited.

A Gaussian process indexed by $\mathbf{x}$ replaces the global linear map
with a *local* one whose predictive variance depends on the distance from
$\mathbf{x}^{*}$ to observed inputs. Regions never visited keep their prior
variance and the epistemic gradient stays informative across trials.

---

## 2. Notation

Recap (unchanged from MARX):

| symbol | meaning |
|---|---|
| $\mathbf{y}_k \in \mathbb{R}^{D_y}$ | observation at time $k$ (here $[y, x, \mathrm{pitch}, \mathrm{roll}]$, $D_y = 4$) |
| $\mathbf{u}_k \in \mathbb{R}^{D_u}$ | control at time $k$ ($D_u = 8$ CPG params) |
| $M_u, M_y$ | input / output lag orders |
| $\mathbf{x}_k = [\mathbf{u}_k, \dots, \mathbf{u}_{k-M_u}, \mathbf{y}_{k-1}, \dots, \mathbf{y}_{k-M_y}] \in \mathbb{R}^{D_x}$ | regressor, $D_x = D_u(M_u + 1) + D_y M_y$ |
| $\mathcal{D}_{k} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^{k}$ | training set up to time $k$ |
| $m^{\star}, S^{\star}$ | goal-prior mean and covariance over $\mathbf{y}$ |
| $\boldsymbol{\mu}, \Upsilon$ | control-prior mean and precision |
| $T$ | EFE planning horizon (here 5) |

---

## 3. MARX baseline (for reference)

Conjugate matrix-normal inverse-Wishart prior on $(M, \Sigma)$ gives a
Student-$t$ posterior predictive

$$
p(\mathbf{y}^{*} \mid \mathbf{x}^{*}, \mathcal{D}) = t_{\eta}\bigl(M_N^{\top}\mathbf{x}^{*},\; \Psi(\mathbf{x}^{*})\bigr)
$$

with

$$
\eta = \nu_N - D_y + 1, \qquad
\Psi(\mathbf{x}^{*}) = \frac{\eta\, \Omega_N^{-1}}{1 + \mathbf{x}^{*\top}\Lambda_N^{-1}\mathbf{x}^{*}}.
$$

EFE per roll-out step:

$$
G_t = \underbrace{\log|\Psi(\mathbf{x}_t)|}_{\text{epistemic: }H[p(\mathbf{y}\mid\mathbf{x})]\text{ (up to const.)}}
    + \underbrace{\tfrac{1}{2}\bigl[\tfrac{\eta}{\eta-2}\,\mathrm{tr}(S^{\star-1}\Psi^{-1}) + (\mu_t - m^{\star})^{\top}S^{\star-1}(\mu_t - m^{\star})\bigr]}_{\text{pragmatic}}
    + \underbrace{\tfrac{1}{2}(\mathbf{u}_t - \boldsymbol{\mu})^{\top}\Upsilon(\mathbf{u}_t - \boldsymbol{\mu})}_{\text{control prior}}.
$$

---

## 4. GP-AR generative model

Replace the linear map $M^{\top}\mathbf{x}$ with a vector-valued latent
function $\mathbf{f}(\mathbf{x}) = [f_1(\mathbf{x}), \dots, f_{D_y}(\mathbf{x})]$, model each output
dimension with an **independent Gaussian process**:

$$
f_d \sim \mathcal{GP}\bigl(0,\; k_d(\mathbf{x}, \mathbf{x}')\bigr), \qquad d = 1, \dots, D_y,
$$

and add homoscedastic observation noise:

$$
y_{k,d} = f_d(\mathbf{x}_k) + \varepsilon_{k,d}, \qquad \varepsilon_{k,d} \sim \mathcal{N}(0, \sigma_{n,d}^{2}).
$$

Take an **ARD** squared-exponential kernel,

$$
k_d(\mathbf{x}, \mathbf{x}') = \sigma_{f,d}^{2}\,\exp\!\left(-\tfrac{1}{2}\sum_{j=1}^{D_x} \frac{(x_j - x_j')^{2}}{\ell_{d,j}^{2}}\right),
$$

with per-output hyperparameters $\boldsymbol{\theta}_d = \{\sigma_{f,d}^{2}, \boldsymbol{\ell}_d, \sigma_{n,d}^{2}\}$ fit once per trial (or once total) by log-marginal-likelihood.

The **independent-output** assumption makes the joint predictive density
diagonal, so all derivations below hold per $d$; we drop the subscript where
unambiguous.

---

## 5. Posterior

Given $\mathcal{D}_N = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ for one output, stack
$\mathbf{y} = (y_1, \dots, y_N)^{\top}$, form the Gram matrix
$K_N \in \mathbb{R}^{N \times N}$ with $[K_N]_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$, and define

$$
A_N = K_N + \sigma_n^{2} I_N, \qquad L_N L_N^{\top} = A_N \quad (\text{Cholesky}).
$$

The standard GP regression posterior at a test input $\mathbf{x}^{*}$ is

$$
f(\mathbf{x}^{*}) \mid \mathcal{D}_N \sim \mathcal{N}\!\bigl(m_N(\mathbf{x}^{*}),\; v_N(\mathbf{x}^{*})\bigr),
$$

with

$$
m_N(\mathbf{x}^{*}) = \mathbf{k}_{*}^{\top} A_N^{-1} \mathbf{y}, \qquad
v_N(\mathbf{x}^{*}) = k(\mathbf{x}^{*}, \mathbf{x}^{*}) - \mathbf{k}_{*}^{\top} A_N^{-1} \mathbf{k}_{*},
$$

where $[\mathbf{k}_{*}]_i = k(\mathbf{x}_i, \mathbf{x}^{*})$.

### Online update

Appending one point $(\mathbf{x}_{N+1}, y_{N+1})$ updates the Cholesky via a
rank-1 extension in $\mathcal{O}(N^{2})$:

$$
L_{N+1} = \begin{pmatrix} L_N & \mathbf{0} \\ \boldsymbol{\ell}_{N+1}^{\top} & \ell_{N+1,N+1} \end{pmatrix},
$$

with $\boldsymbol{\ell}_{N+1} = L_N^{-1} \mathbf{k}_{N+1}$ (triangular solve) and
$\ell_{N+1,N+1}^{2} = k(\mathbf{x}_{N+1}, \mathbf{x}_{N+1}) + \sigma_n^{2} - \|\boldsymbol{\ell}_{N+1}\|^{2}$.

A matching rank-1 update of the pre-solved vector $\boldsymbol{\alpha}_N = A_N^{-1}\mathbf{y}$ is also available but is simpler to just recompute via one triangular solve after each step.

For long runs (many trials × hundreds of steps) a **sliding window** of the
most recent $N_w$ points, or a **sparse variational** approximation with $M$
inducing points (SVGP), keeps the cost bounded.

---

## 6. Posterior predictive

Adding the noise $\varepsilon$ to the latent value gives the one-step-ahead
predictive density that the agent uses in planning:

$$
y^{*} \mid \mathbf{x}^{*}, \mathcal{D}_N \sim \mathcal{N}\!\bigl(m_N(\mathbf{x}^{*}),\; s_N^{2}(\mathbf{x}^{*})\bigr), \qquad
s_N^{2}(\mathbf{x}^{*}) = v_N(\mathbf{x}^{*}) + \sigma_n^{2}.
$$

Joint predictive over the $D_y$ output dimensions (independent GPs):

$$
\mathbf{y}^{*} \mid \mathbf{x}^{*}, \mathcal{D}_N \sim \mathcal{N}\!\bigl(\mathbf{m}_N(\mathbf{x}^{*}),\; \Sigma_N(\mathbf{x}^{*})\bigr),
$$

with

$$
\mathbf{m}_N(\mathbf{x}^{*}) = [m_{N,1}(\mathbf{x}^{*}), \dots, m_{N,D_y}(\mathbf{x}^{*})]^{\top}, \quad
\Sigma_N(\mathbf{x}^{*}) = \mathrm{diag}(s_{N,1}^{2}(\mathbf{x}^{*}), \dots, s_{N,D_y}^{2}(\mathbf{x}^{*})).
$$

(If cross-output correlations are ever needed, swap in a
linear-model-of-coregionalization kernel; the rest of the agent is unchanged.)

---

## 7. Expected free energy under the GP

The EFE decomposes into the same three terms as in MARX:

$$
G_t = H[p(\mathbf{y} \mid \mathbf{x}_t, \mathcal{D})] + \mathrm{CE}(\mathbf{x}_t) + \tfrac{1}{2}(\mathbf{u}_t - \boldsymbol{\mu})^{\top}\Upsilon(\mathbf{u}_t - \boldsymbol{\mu}),
$$

but the *functional form* of the first two terms changes.

### 7.1 Epistemic (entropy of the predictive)

For a Gaussian predictive $\mathcal{N}(\mathbf{m}_N, \Sigma_N)$ with diagonal $\Sigma_N$:

$$
H[p(\mathbf{y}\mid\mathbf{x}_t, \mathcal{D})] = \tfrac{1}{2}\log|2\pi e\, \Sigma_N(\mathbf{x}_t)|
 = \tfrac{1}{2}\sum_{d=1}^{D_y} \log\!\bigl(2\pi e\, s_{N,d}^{2}(\mathbf{x}_t)\bigr).
$$

Dropping constants (they don't affect the argmin over $\mathbf{u}$):

$$
\boxed{\;\mathrm{MI}(\mathbf{x}_t) \;=\; \tfrac{1}{2}\sum_{d=1}^{D_y} \log s_{N,d}^{2}(\mathbf{x}_t)\;}
$$

Minimising EFE drives the planner towards $\mathbf{x}_t$ with *large*
$s_{N,d}^{2}$, i.e. regions still uncertain under the GP — this is the
pointwise epistemic drive that MARX lacked.

This replaces the MARX `mutualinfo(x) = slogdet(Ψ_t)[1]` (a *global*
scaling by $\Lambda_N$) with a *local* GP-variance term that remains
non-trivial no matter how many observations the agent accumulates, as long
as $\mathbf{x}_t$ is far from any training point.

### 7.2 Pragmatic (cross-entropy with goal prior)

With goal prior $p^{\star}(\mathbf{y}) = \mathcal{N}(m^{\star}, S^{\star})$ and
Gaussian predictive $q(\mathbf{y}) = \mathcal{N}(\mathbf{m}_N(\mathbf{x}_t), \Sigma_N(\mathbf{x}_t))$,

$$
\mathrm{CE}(\mathbf{x}_t) = H(q, p^{\star}) = \tfrac{1}{2}\Bigl[D_y \log(2\pi) + \log|S^{\star}| + \mathrm{tr}\bigl(S^{\star-1}\Sigma_N(\mathbf{x}_t)\bigr) + (\mathbf{m}_N(\mathbf{x}_t) - m^{\star})^{\top}S^{\star-1}(\mathbf{m}_N(\mathbf{x}_t) - m^{\star})\Bigr].
$$

Dropping the $\mathbf{x}_t$-independent constants:

$$
\boxed{\;\mathrm{CE}(\mathbf{x}_t) \;=\; \tfrac{1}{2}\,\mathrm{tr}\bigl(S^{\star-1}\Sigma_N(\mathbf{x}_t)\bigr) + \tfrac{1}{2}(\mathbf{m}_N(\mathbf{x}_t) - m^{\star})^{\top}S^{\star-1}(\mathbf{m}_N(\mathbf{x}_t) - m^{\star}).\;}
$$

This is structurally identical to the MARX pragmatic term with
$\mathbf{m}_N, \Sigma_N$ taking the role of $\mu_t, \Psi_t^{-1}$.

### 7.3 Full objective

$$
G(\mathbf{u}_{0:T-1}) = \sum_{t=0}^{T-1} \Bigl[\mathrm{MI}(\mathbf{x}_t) + \mathrm{CE}(\mathbf{x}_t) + \tfrac{1}{2}(\mathbf{u}_t - \boldsymbol{\mu})^{\top}\Upsilon(\mathbf{u}_t - \boldsymbol{\mu})\Bigr],
$$

where the planning-horizon regressor $\mathbf{x}_t$ is constructed by rolling
the input and (predicted) output buffers forward:

$$
\begin{aligned}
\mathbf{x}_t &= [\mathbf{u}_t, \mathbf{u}_{t-1}, \dots, \mathbf{u}_{t-M_u}, \hat{\mathbf{y}}_{t-1}, \dots, \hat{\mathbf{y}}_{t-M_y}], \\
\hat{\mathbf{y}}_{\tau} &= \mathbf{m}_N(\mathbf{x}_{\tau}) \quad \text{for } \tau \ge 0.
\end{aligned}
$$

(i.e. the planner pushes predicted means through the regressor, identical
to MARX).

---

## 8. Implementation notes

### 8.1 CasADi symbolic graph

For each IPOPT evaluation the planner currently rolls the buffers forward
$T$ steps and evaluates `MI + CE + control_prior`. With a GP, each of those
$T$ evaluations requires

$$
\mathbf{k}_{*} \in \mathbb{R}^{N}, \quad
\alpha = A_N^{-1}\mathbf{y} \in \mathbb{R}^{N}, \quad
v = L_N^{-1}\mathbf{k}_{*} \in \mathbb{R}^{N}, \quad
m_N = \mathbf{k}_{*}^{\top}\alpha, \quad
s_N^{2} = k(\mathbf{x}^{*}, \mathbf{x}^{*}) + \sigma_n^{2} - v^{\top}v,
$$

per output $d$.

- $\alpha_d, L_d$ are precomputed once per IPOPT call (constant w.r.t.
  $\mathbf{u}$) and passed in as `ca.DM`.
- $\mathbf{k}_{*}$ is a symbolic vector in $\mathbf{x}^{*}$, hence in $\mathbf{u}$.
- The triangular solve $L_d\, v_d = \mathbf{k}_{*}$ can be written symbolically
  via `ca.solve(L_d, k_star)` (CasADi handles sparse triangular solves
  efficiently) or unrolled as an explicit loop.

The symbolic-graph size scales as $\mathcal{O}(T \cdot N \cdot D_x \cdot D_y)$;
for $T = 5$, $D_x = 40$, $D_y = 4$ this is usable up to $N \sim 10^{3}$
training points. Beyond that, go sparse.

### 8.2 Training-set management

Three sensible strategies:

1. **Full buffer across trials** — keep everything; refit hyperparameters
   at the start of each trial by L-BFGS on log marginal likelihood.
   Simplest, highest accuracy, $\mathcal{O}(N^{3})$ per refit.
2. **Sliding window of size $N_w$** — keep only the most recent $N_w$
   points (say 500). Constant per-step cost, gracefully forgets stale
   dynamics if the control distribution shifts.
3. **SVGP with $M$ inducing points** — fully non-parametric Bayesian cost
   $\mathcal{O}(NM^{2})$ for training and $\mathcal{O}(M^{2})$ for
   prediction. Heavier bookkeeping but scales indefinitely.

I recommend **(2)** as the first implementation: it's a one-line change
vs. (1) and keeps the EFE gradient informative as the agent visits
new regions.

### 8.3 Hyperparameter fitting

- Default kernel: ARD squared-exponential, per-output.
- Fit by L-BFGS on the log marginal likelihood
  $\log p(\mathbf{y}_d \mid X) = -\tfrac{1}{2}\mathbf{y}_d^{\top} A_d^{-1}\mathbf{y}_d - \tfrac{1}{2}\log|A_d| - \tfrac{N}{2}\log 2\pi$,
  either once at the start (using trial-1 data) or once per trial.
- Good off-the-shelf implementations: `gpytorch`, `scikit-learn`
  `GaussianProcessRegressor`. GPyTorch is the cleaner bet if we want
  GPU or sparse extensions later.

### 8.4 Input normalisation

GP kernels are highly scale-sensitive. Standardise the regressor
coordinates before fitting:

$$
\tilde{\mathbf{x}}_k = (\mathbf{x}_k - \bar{\mathbf{x}}) / \mathbf{s}_{\mathbf{x}},
$$

with $\bar{\mathbf{x}}, \mathbf{s}_{\mathbf{x}}$ estimated from the first trial. Keep
the same transformation for all subsequent trials to keep the learned
length-scales interpretable.

### 8.5 API sketch (proposed)

```python
class GPAgent:
    def __init__(self, Dy, Du, delay_inp, delay_out, time_horizon,
                 control_prior_mean, control_prior_precision,
                 goal_prior, window_size=500, kernel="ard_rbf"):
        ...

    def update(self, y_k, u_k):
        """Append (x_k, y_k) to the training set, optionally refit."""

    def posterior_predictive(self, x_t):
        """Returns (m_N, s2_N) per output."""

    def EFE(self, controls):
        """Symbolic-ready rollout matching MARX EFE interface."""

    def minimizeEFE(self, u_0, control_lims, lambda_energy, max_iter, tol):
        """CasADi/IPOPT solve (unchanged driver)."""
```

The existing `run_episode_maxrefe` driver only calls
`agent.update`, `agent.posterior_predictive`, `agent.EFE`,
`agent.minimizeEFE`, `agent.reset_buffer`; it does not touch the model
internals. So the replacement is a drop-in provided these five methods
keep the same signatures.

---

## 9. What to expect

If the diagnosis from the cps-sweep is right (posterior locks in because
the linear model pools variance globally), replacing MARX with a GP-AR
should restore across-trial exploration: regions that trial 1 did *not*
visit keep their prior variance $\sigma_f^{2} + \sigma_n^{2}$, so
$\mathrm{MI}(\mathbf{x}_t)$ continues to pull IPOPT into unexplored
corners of the regressor space across all subsequent trials.

Concretely, the `per-param std across trials` table we currently read as
$\{0, 0, 0\}$ for $\omega_{\text{swing}}, \omega_{\text{stance}}, F_{\text{FAST}}$
should recover non-trivial spread once the GP posterior variance is high
outside the single visited corner.
