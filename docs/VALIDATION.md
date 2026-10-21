# Validation record

The checked-in `examples/validated_run/` contains a newly computed run, not the original dissertation numbers.

## Completed checks

- 16 standard-library unittest cases passed, including two repeated end-to-end runs. Tests cover numerical derivatives, DMP forcing, Gaussian conditioning, GP replicate compression, KMP/uncertainty-aware planning consistency, extrapolation covariance, LQR gains, whole-demo partitions, masked-value noninterference, continuous collision geometry, local detours on revisited paths, failure/outlier accounting and output consistency.
- Full five-fold benchmark: A–H, 105 demonstrations, five methods, three tasks, **1,575 recorded trials, zero execution errors**.
- Independent audit of all 1,575 saved NPZ predictions against their corresponding CSV keys: direct coordinate RMSE and endpoint-error calculations agree with recorded metrics.
- Every method/task group contains the same 105 held-out demonstrations. No duplicate trial keys were found.
- Source SHA-256 values in the final run manifest match the implementation used for the numeric results and regenerated reports. Original data files are retained.
- Editable package installation and `python -m imitation_learning inspect` succeeded.
- Generated dashboard JSON parses and its JavaScript passes Node syntax validation. Generated numerical figures were visually reviewed. A real-browser interaction/screenshot test could not run in this environment because a browser binary was unavailable; the attempted browser download did not provide a valid archive. This limitation is separate from the successful Python pipeline tests.

## Reproduce

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m imitation_learning run --config configs/benchmark.json --output outputs/recheck
```

Versions used for the local run are listed in `requirements-tested.txt` and the recorded manifest. Hardware-dependent timing is not expected to reproduce exactly. Numerical reproducibility is assessed under the same versions, configuration, data, seed and one-thread BLAS limit.

GitHub Actions additionally runs the suite and the quick experiment on Python 3.10 and 3.12. Remote CI status is authoritative for those environments; a local pass alone is not a claim that remote checks have completed.

## Optional MuJoCo simulation (v0.3)

- With simulation dependencies installed, the combined suite passed **24 tests** (16 learning/protocol cases and 8 robot cases).
- A separate dynamics sweep of the 40 representative A-H / method reproduction examples passed the declared maximum tracking error, contact, joint-position and actual velocity/acceleration criteria. This is not full robot cross-validation and is not hardware validation. Recorded summaries and configuration are in `examples/robot_validation/`.
- A deliberately near-zero motor-torque limit produces a large tracking error, a failed criterion and preserved logs. Actual joint states are integrated by MuJoCo, not reset to the reference at each step.
- Full-rate NPZ and CSV logs reproduce the reported Euclidean TCP errors. Independent analytic forward kinematics agrees with MuJoCo site positions.
- A built wheel contains the XML model and both HTML templates. The generated simulation page's JSON parses and its JavaScript passes Node syntax checking. A real browser interaction test remains outside this recorded local check.
- New GitHub Actions jobs install simulation dependencies and run the robot checks plus a saved G-case simulation on **Ubuntu and Windows with Python 3.12**. Check their actual run status for remote results.

The existing full learning benchmark was computed under the source snapshot in its original manifest. v0.3 adds the simulation stage; it does not relabel the prior 1,575 planned-path records as robot dynamics or hardware results.
