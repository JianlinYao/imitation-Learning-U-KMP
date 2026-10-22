# Research and workflow improvements (v0.5)

The original dissertation, notebooks, A-H source demonstrations and archived
v0.2-v0.4 result files are retained. v0.5 outputs are new experiments and are
not replacements for the dissertation's numerical tables.

## Run a complete demonstration on Windows

Double-click `run_demo.bat` in the extracted project root. Alternatively:

```powershell
python -m pip install -e ".[simulation]"
python -u -m imitation_learning demo
```

Open `outputs/demo/robot/dashboard.html`. It contains all eight letters and all
five methods after a successful default run. The default demo is one held-out
fold with 80 samples; it is clearly marked as partial cross-validation. The
full research configuration is `configs/research.json` (five folds, 200 samples).

Rerunning `demo` resumes completed learning folds and verified simulation cases.
A learning checkpoint is atomically replaced only after an entire letter/fold
finishes. An interrupted, incomplete fold is recalculated. A simulation case
is moved from a staging directory only after all of its outputs are complete.
The pipeline rejects incompatible input/configuration/source/environment
changes rather than mixing results. Select a new `--output` after such changes.
Tracking failures remain visible in the final page and make the command exit
nonzero; an unsuccessful task is distinct from an execution error.

## Reference KMP adaptation

`ModelConfig.kmp_adaptation` selects `reference` or `gaussian`.

- `reference` replaces a nearby reference site or appends a new site, then
  solves the KMP mean and covariance systems with separate regularizers. This
  follows the reference-database adaptation in Huang et al., Section 3.1.
- Full-state constraints match the independently evaluated KMP matrix
  equations. Partial-state constraints retain only the observed components at
  the replaced site; missing velocities are not invented. The analytic
  position/velocity kernel and training-fold GMM/GMR are retained, so this is
  not a claim of bit-for-bit equivalence with the MATLAB demo's GMM and finite
  difference kernel implementation.
- Changing `lambda_cov` leaves the reference-adapted mean unchanged. A
  regression test covers the case where v0.4's Gaussian conditioning coupled
  the adapted mean to that parameter.
- `gaussian` retains the historical posterior-conditioning variant. The old
  `benchmark.json` and `quick.json` explicitly select it for comparisons.
  `research.json` and `demo.json` select `reference`.

## Predictive uncertainty and calibration

GP's state covariance remains a latent-function covariance for control. Scoring
a new position observation adds the configured per-demonstration noise, not
the noise divided by the number of training demonstrations. Other probabilistic
models provide their learned trajectory variability. These semantics are
explicit rather than treating every variance as the same object.

New outputs include raw coordinate coverage, a bivariate 95% position ellipse
coverage, average interval width, per-time bivariate position NLPD and separate
calibrated coverage/width. NLPD is a marginal position score, not a joint
whole-trajectory likelihood. DMP has no fabricated probability distribution.

When `calibration_fraction > 0`, reserve complete demonstrations within each
outer-training fold. Neither the model nor the shared spatial scaler sees
these calibration demonstrations. Fit all methods on the same remaining
demonstrations. For each method and reproduction/missing scenario, obtain
absolute standardized residuals on the reserved demonstrations, then use their
empirical 95th percentile to inflate the marginal bands. Missing calibration
uses hidden positions only. Record reserved IDs, quantile, count and inflation
in `calibration.json`; outer-test residuals are never used to fit inflation.

This is **empirical marginal calibration**. Time samples and coordinates are
dependent and the dataset is small; no finite-sample conformal or whole-path
95% guarantee is asserted. Inflation never shrinks a band. Its achieved
held-out coverage and width are reported even when poor. A deliberate obstacle
detour does not use the unmodified demonstration as a calibration target.
The full state covariance used by LQR is not replaced by these scalar
marginal-band adjustments.

## Task criteria

`status=ok` means the computation produced valid output. `goal_reached` checks
only the endpoint. `task_success` additionally requires:

- reproduction: coordinate RMSE / training-reference bounding-box diagonal
  at most `reproduction_rmse_tolerance_fraction` (default 0.10);
- missing: gap RMSE / the same training span at most
  `missing_gap_tolerance_fraction` (default 0.05);
- obstacle: continuous path-segment clearance meets the requested margin.

All tasks also require the endpoint criterion. Thresholds are declared in the
configuration before evaluation. No poor or failed case is removed.

## Uncertainty-aware physics control

The teaching arm supports `joint_pd`, `fixed_lqr` and `covariance_lqr`.
`auto` chooses fixed LQR for ordinary planning methods and covariance LQR for
U-KMP. Thus KMP and U-KMP may share their path while exhibiting different
feedback behavior. This choice is recorded per case, not hidden in the label.

Both task-space LQR modes use the same affine finite-horizon Riccati solution,
input penalty, known-model inverse dynamics and torque limit. They read the
actual integrated TCP position and velocity. Task-space acceleration is mapped
to joint acceleration using the analytic Jacobian and its convective term;
MuJoCo's inertia/bias/passive forces yield requested actuator torque.
Torque is clipped before every `mj_step`; states are never prescribed after
initialization. The adaptive cost is the inverse of the saved covariance plus
declared position/velocity floors. Fixed LQR uses the inverse of those floors.

Spatial mapping and time warping transform covariance as `J C J.T`, with
`J=diag(s,s,s*source_time_rate,s*source_time_rate)`. Covariance interpolation is
linear, preserving positive semidefiniteness. Actual target velocity is the
derivative of the same PCHIP/time-warp position reference.

The new grid uses one common feasible motion duration for all methods and
controllers within each letter. Separate letters can have different durations.
Original single-case commands retain their per-trajectory timing and fixed-PD
default. Full state covariance is required for adaptive LQR; older saved
examples remain usable with fixed PD or fixed LQR.

To isolate controller effects on the **same** learned KMP path:

```bash
python -m imitation_learning simulate outputs/demo/learning --all-letters --method KMP --controllers fixed_lqr covariance_lqr --disturbance-force 0.02 0 --output outputs/controller-study
```

This adds a real simulated TCP force pulse from 45% to 55% of each motion.
Outputs include actual states, applied force, torque RMS, absolute mechanical
work estimate, gain norms, deflection and recovery below the declared tracking
threshold. More compliance can mean more deflection and less torque; it is not
automatically better tracking. Read all metrics and retained failure states.
The gallery can switch controllers as well as letters and methods.

## Remaining research scope

The robot is an ideal known two-link teaching model with no hardware, contact
writing, sensor/communication delay or environment obstacle geometry. The
existing circle benchmark concerns a 2-D planned path, not complete-arm
collision avoidance. Cross-letter transfer, repeated-seed confidence intervals,
training-support-gap experiments, task-parameterized 3-D pose learning,
multi-controller fusion, EKMP and NS-KMP remain separate research extensions.

Relevant primary references:

- Huang et al., *Kernelized Movement Primitives*, IJRR 2019:
  https://arxiv.org/abs/1708.08638
- Silvério et al., *Uncertainty-Aware Imitation Learning using Kernelized
  Movement Primitives*, IROS 2019: https://arxiv.org/abs/1903.02114
- Authors' reference implementation:
  https://github.com/yanlongtu/robInfLib-matlab
- Huang, *EKMP: Generalized Imitation Learning with Adaptation, Nonlinear Hard
  Constraints and Obstacle Avoidance*, 2021 preprint:
  https://arxiv.org/abs/2103.00452
- Silvério and Huang, *A Non-parametric Skill Representation with Soft Null
  Space Projectors for Fast Generalization*, ICRA 2023:
  https://arxiv.org/abs/2209.08522
