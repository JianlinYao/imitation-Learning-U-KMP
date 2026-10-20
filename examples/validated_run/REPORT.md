# Recomputed experiment report

These are new measurements from the recorded configuration, not recovered dissertation tables.

- Status: **completed**
- Trials: 1575 valid / 1575 expected; 0 errors.
- Partial cross-validation: False
- Planned trajectories and closed-loop control are evaluated separately.
- KMP and U-KMP have the same planned mean/covariance; their planning errors may be identical.
- Intervals are nominal marginal Gaussian bands; empirical coverage is reported without a calibration guarantee.
- No outlier filtering or preferred-method selection is applied.

## Reproduction

| Method | Valid / total | RMSE | End error | Jerk | Gap RMSE | Collision rate | Inference ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| DMP | 105 / 105 | 1.375 ± 0.593 | 0.00066 ± 0.000265 | 1.724e+04 ± 7.57e+03 | — | — | 34.28 ± 3.26 |
| ProMP | 105 / 105 | 1.757 ± 0.865 | 0.002945 ± 0.00385 | 3.626e+04 ± 2.04e+04 | — | — | 0.1997 ± 0.0429 |
| GP | 105 / 105 | 1.364 ± 0.589 | 0.008197 ± 0.00511 | 1.401e+04 ± 6.18e+03 | — | — | 0.2084 ± 0.0547 |
| KMP | 105 / 105 | 1.378 ± 0.566 | 0.000168 ± 0.000175 | 1.04e+04 ± 4.02e+03 | — | — | 0.2099 ± 0.0503 |
| U-KMP | 105 / 105 | 1.378 ± 0.566 | 0.000168 ± 0.000175 | 1.04e+04 ± 4.02e+03 | — | — | 0.2045 ± 0.0439 |

## Missing

| Method | Valid / total | RMSE | End error | Jerk | Gap RMSE | Collision rate | Inference ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| DMP | 105 / 105 | 1.375 ± 0.593 | 0.00066 ± 0.000265 | 1.724e+04 ± 7.57e+03 | 1.699 ± 0.905 | — | 34.43 ± 3.89 |
| ProMP | 105 / 105 | 0.6157 ± 0.395 | 0.2536 ± 0.176 | 4.453e+04 ± 2.11e+04 | 1.142 ± 0.866 | — | 0.8588 ± 0.14 |
| GP | 105 / 105 | 0.6868 ± 0.388 | 0.01154 ± 0.00698 | 2.587e+04 ± 9.63e+03 | 1.515 ± 0.87 | — | 0.8475 ± 0.119 |
| KMP | 105 / 105 | 1.241 ± 1.03 | 0.00318 ± 0.00264 | 2.812e+04 ± 1.31e+04 | 2.769 ± 2.31 | — | 0.869 ± 0.123 |
| U-KMP | 105 / 105 | 1.241 ± 1.03 | 0.00318 ± 0.00264 | 2.812e+04 ± 1.31e+04 | 2.769 ± 2.31 | — | 0.8445 ± 0.122 |

## Obstacle

| Method | Valid / total | RMSE | End error | Jerk | Gap RMSE | Collision rate | Inference ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| DMP | 105 / 105 | 1.375 ± 0.593 | 0.00066 ± 0.000265 | 1.724e+04 ± 7.57e+03 | — | 1 ± 0 | 34.45 ± 4.23 |
| ProMP | 105 / 105 | 3.026 ± 1.86 | 0.6818 ± 0.439 | 1.156e+05 ± 9.32e+04 | — | 0 ± 0 | 0.4322 ± 0.128 |
| GP | 105 / 105 | 1.578 ± 0.521 | 0.008272 ± 0.0052 | 3.393e+04 ± 2.78e+04 | — | 0 ± 0 | 0.3984 ± 0.0974 |
| KMP | 105 / 105 | 4.569 ± 4.77 | 0.0002204 ± 0.000215 | 4.218e+04 ± 3.69e+04 | — | 0 ± 0 | 0.4072 ± 0.0974 |
| U-KMP | 105 / 105 | 4.569 ± 4.77 | 0.0002204 ± 0.000215 | 4.218e+04 ± 3.69e+04 | — | 0 ± 0 | 0.3956 ± 0.0877 |

## Interpretation

Obstacle RMSE measures deviation from the original demonstration, which can increase for a valid detour. Read it alongside continuous clearance, goal reach and collision rate. DMP is an endpoint-only baseline with no obstacle coupling or gap conditioning.

Training is performed once per method/fold and its measured cost is repeated on the corresponding trial rows; do not sum these repeated costs. Inference uses one warmup followed by the median of the configured repeats. Adaptation includes constraint construction and obstacle repair. Timing depends on the recorded runtime and hardware.

Control ablation uses the same predicted reference, unit-mass system, input cost and acceleration disturbance for fixed-Q and covariance-aware LQR. It is a separate numerical experiment, not physical robot validation.

See `manifest.json`, `splits.json`, `metrics.csv`, `fold_metrics.csv`, `control_ablation.csv` and the offline `dashboard.html` for provenance and details.
