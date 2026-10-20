# Methods and experiment protocol

## Scope and units

This implementation evaluates 2-D handwriting-style demonstrations as movement trajectories. It does not implement image recognition, reinforcement learning, a robot simulator or a physical robot interface. Raw positions retain the dataset's units; these are not asserted to be meters.

The default time grid has 200 samples from 0 to 0.995 seconds, matching a 0.005-second interval. Demonstrations are resampled to this declared duration. Derivatives and LQR dynamics use the resulting actual grid, not an unrelated hard-coded step. Original `.mat` velocity/acceleration fields are not reused after resampling and scaling.

## Fold-local data pipeline

1. Load original positions from `{letter}.mat`. Retain the original zero-based demo index.
2. Split complete demonstrations with shuffled five-fold cross-validation, seed 42, separately within each letter. All methods and tasks reuse the saved indices. This measures within-letter generalization, not transfer to unseen letters.
3. Resample each training demonstration. Apply an 11-point, degree-three Savitzky–Golay filter.
4. Estimate one per-coordinate mean and standard deviation from the training demonstrations only; use this common scaler for every method and held-out observation in the fold.
5. Compute training velocities on the actual time grid. Refit each model and the GMM/GMR reference within this fold. Do not load `ref*.mat`.

Evaluation targets are resampled, unsmoothed held-out positions. Start and goal positions are supplied as declared task conditions to every method. Held-out trajectory interiors are not exposed in the reproduction or obstacle task. In missing completion, only explicitly visible positions are supplied.

## State and kernels

Time-major state order is `[px, py, vx, vy]`. An RBF position kernel is

`k(a,b) = exp(-(a-b)^2 / (2*l^2))`.

The state kernel uses its analytic derivatives: position/velocity blocks are `d_b k`, velocity/position blocks `d_a k`, and velocity/velocity blocks `d_a d_b k`, separately for each coordinate. This preserves position/velocity consistency and cross-time covariance. Numerical tests compare the derivatives with finite differences and check positive semidefiniteness.

Symmetric positive-definite systems use Cholesky solves, not explicit inverses. Bounded scale-relative jitter handles roundoff. Nonconvergent GMM fits and material negative conditional variances are reported as failures.

## DMP

The discrete transformation system is

`tau^2 * y_ddot = kp*(g-y) - tau*kv*y_dot + f(s)`, with `s = exp(-alpha*t/tau)`.

The training target is consequently `tau^2*y_ddot - kp*(g-y) + tau*kv*y_dot`. Normalized phase RBF features, multiplied by phase, fit the forcing function by pooled least squares across all training demonstrations. Rollout uses RK45 with recorded tolerances. New start/goal conditions are supported. There is no obstacle-coupling term and no mid-trajectory conditioning; unsupported adaptation is explicitly marked in the metrics.

## ProMP

Fit basis weights separately to all training demonstrations, then estimate their mean and full sample covariance. Position and analytic velocity features share weights. Velocity training rows are weighted by `0.1*duration` to balance units; this choice is explicit rather than a hidden change of metric. A small weight covariance regularizer permits conditioning beyond the finite sample rank.

## GP

Use an RBF GP with independent output coordinates and configured IID observation noise. For H demonstrations sharing identical inputs, all repeated observations are exactly equivalent, for posterior mean and covariance, to their pointwise average with noise variance divided by H. This avoids factoring an unnecessary `(H*T)`-square system. A numerical test checks equivalence against the uncompressed calculation. Hyperparameters remain fixed by configuration; no held-out optimization is performed.

## GMM/GMR reference and KMP

Fit a full-covariance GMM to the training-fold joint variable `[phase, px, py, vx, vy]`. Standardize this joint variable using training-only statistics for mixture fitting. Gaussian mixture regression conditions on phase and applies the law of total covariance, including between-component mean dispersion. The returned 4-by-4 reference covariance retains correlations between position, velocity and coordinates.

For stacked reference mean `mu`, block-diagonal reference covariance `Sigma`, and kernel matrix `K`, compute

`mean(t) = K_tR (K + lambda_mean*Sigma)^(-1) mu`

`Cov(t,s) = (N/lambda_cov) [K_ts - K_tR (K + lambda_cov*Sigma)^(-1) K_Rs]`.

These are KMP mean/covariance equations from Huang et al. and Silvério et al. Default reference count and `lambda_cov` are both 200, so their ratio is one. The state derivative kernel is the time-driven position/velocity extension. The full joint trajectory distribution is retained internally; only marginal 4-by-4 blocks are saved per prediction.

### Conditioning choice

Given any fitted Gaussian trajectory `(m,C)` and selected position/state observations `y` with observation noise `R`, the new implementation uses

`m_post = m + C_:o (C_oo + R)^(-1) (y - m_o)`

`C_post = C - C_:o (C_oo + R)^(-1) C_o:`.

This handles partial state observations without inventing unavailable velocities. It is Gaussian posterior conditioning of the learned trajectory distribution. It differs from the historical KMP notebook's replace/append-reference-point-and-refit implementation and is documented as an implementation choice, not asserted to reproduce its numerical output exactly. The mean/covariance hyperparameters remain separate, as in the KMP formulation. Duplicate observations of the same state index are replaced, not double-counted.

## U-KMP and separate control validation

Ordinary KMP already predicts covariance. U-KMP in the cited research exploits that information for control; it need not produce a different planned mean. Consequently this benchmark does **not** fabricate planning improvements by renaming identity-noise kernel ridge regression as KMP or by assigning different arbitrary regularizers to U-KMP.

The five-method planning table contains KMP and U-KMP with identical planned distributions under equal parameters. The uncertainty-aware behavior is evaluated separately using a 2-D unit-mass double integrator:

`x_(t+1) = A*x_t + B*u_t`, where `A=[[I,dt*I],[0,I]]` and `B=[[0.5*dt^2*I],[dt*I]]`.

Finite-horizon affine LQR tracks the same predicted position/velocity sequence. Its adaptive state cost is `Q_t = (Cov_t + 0.001*I)^(-1)` and input cost is `R=0.001*I`. The fixed-Q ablation uses `diag(1,1,0.1,0.1)` with the same R, reference, dynamics and disturbance. Both solve the Riccati tracking recursion, including the affine reference term. No arbitrary target acceleration is added to only one controller.

An external normalized acceleration `[4,-2]` is applied from 45% to 55% of the trajectory duration. Record tracking error in normalized coordinates, command RMS, gain norms and rollout time. Examples are evaluated for the first held-out demo in each fold/task and saved separately from planning metrics. Tests additionally verify lower gains at higher predicted covariance, and increasing KMP variance far outside training support.

This fixed-Q comparison is an explicit baseline, not a parameter-optimized proof of superiority. Neither this numerical experiment nor a variance band establishes physical robot safety. Posterior spread can reflect both demonstration variability and missing support; this implementation does not claim a uniquely identifiable aleatoric/epistemic decomposition.

## Three tasks

### Reproduction

Generate a complete trajectory with supplied held-out start/goal positions. Compare with that held-out demonstration on the common time grid. This is endpoint-conditioned, within-letter generalization.

### Missing completion

Hide `[40%,60%)` of the resampled held-out positions. Apply filtering independently within each visible contiguous segment **after masking**. Select 20 evenly spaced visible samples plus exact endpoint conditions; all Gaussian methods receive the same positions. No missing position, derivative, full-test normalization statistic, or smoothing stencil across the gap is used. Report both full-path and gap-only RMSE. The DMP endpoint-only baseline does not use intermediate visible samples.

### Obstacle adaptation

Place a circular obstacle on the training mean trajectory in its middle phase, away from endpoints. Radius defaults to 7% of the training trajectory bounding-box diagonal, capped by endpoint clearance; requested margin is 2% of that span. All methods in a fold use this same obstacle. Gaussian methods share a deterministic geometric via-point policy and up to eight residual repair steps, using generated candidate paths rather than hidden test interiors. DMP receives no synthetic avoidance mechanism.

Check the distance from the circle center to **every entire path segment**, including zero-length segments. Contact counts as collision. Successful obstacle trials require both margin satisfaction and endpoint error at most 3% of the training span. The geometric policy has no feasibility guarantee; failed avoidance, boundary conflicts and repair-budget exhaustion remain in the results.

## Metrics, aggregation and timing

- RMSE: `sqrt(mean((prediction-truth)^2))`, averaged over time and both coordinates.
- End error: Euclidean position error at the final sample.
- Jerk: mean Euclidean norm of the third numerical time derivative, using second-order boundary stencils. Units depend on the recorded dataset coordinate and assumed duration.
- Gap RMSE: the same coordinate RMSE, restricted to hidden samples.
- Coverage95: fraction of coordinate/time targets within `mean +/- 1.96*sqrt(marginal variance)`. It is empirical coverage of nominal bands, not a guaranteed calibrated confidence level.
- Clearance: minimum segment-to-circle boundary distance in original dataset units. Collision and requested-margin satisfaction are recorded separately.
- Train time: actual reference/model fit and initial distribution computation once per fold/method. Repeated on trial rows for that fit; do not sum repeated entries.
- Adapt time: observation construction and obstacle constraint/repair search.
- Infer time: one untimed warmup, then median conditional-prediction runtime across three repeats. BLAS is limited to one thread. This includes final conditioning but not the preceding geometric search. It is not a real-time hardware guarantee.

`summary.csv` reports trial means, sample standard deviations and metric-specific nonmissing counts. `fold_metrics.csv` supplies within-letter fold means. These standard deviations are descriptive, not independent-sample confidence intervals. No IQR filter, hard cap, winner preference or removal of poor valid results is used. Failures have explicit status and make the CLI exit nonzero; total and valid counts are both reported.

## References

- Huang et al. (2019), *Kernelized Movement Primitives*, IJRR 38(7), 833–852. https://arxiv.org/abs/1708.08638
- Silvério et al. (2019), *Uncertainty-Aware Imitation Learning using Kernelized Movement Primitives*, IROS, 90–97. https://arxiv.org/abs/1903.02114
- Authors' reference implementation: https://github.com/yanlongtu/robInfLib-matlab
