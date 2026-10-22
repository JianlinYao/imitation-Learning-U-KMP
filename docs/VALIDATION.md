# Validation record

## Local experiment backend (v0.6)

The local combined suite ran **52 tests: 50 passed and two opt-in Chrome cases
skipped**. The eight backend cases cover durable SQLite records and atomic queue
claims, exclusive workspace ownership, HTTP validation and filtering, cancellation
and retry, process failures, graceful restart, bounded logs, metric isolation and
registered-file access. Two cases execute the real scientific CLI through the
HTTP API: a learning evaluation and a MuJoCo KMP/U-KMP tracking demo with both
physical controllers. Lifecycle and browser fixtures are explicitly synthetic
and are not additional research results.

The v0.6 wheel built successfully and contains the task-manager template. The
server CLI help resolves correctly. Dedicated CI checks the new page in Chrome,
including submission, report links, retry, validation and mobile overflow, and
tests the real backend on Linux and Windows. Windows also checks installation
and argument handling through `run_server.bat --help`. Check the publishing PR's
actual CI result before treating these remote checks as passed. This backend
does not replace or relabel the archived scientific benchmarks below.

## Research and complete workflow (v0.5)

The new [v0.5 snapshot](../examples/research_validation/README.md) records a
separate full research run: **1,575 records, zero execution errors**, with
reference KMP updates and reserved-demo empirical calibration. All predictions'
RMSE, endpoint error and gap error were independently rechecked, and all 40
fit/calibration/test partitions were verified disjoint. The source/input hashes
match the files used for this run. The original archived records below are
retained.

The default 80-sample, one-fold `demo` workflow generated 345 learning records
and **40/40** passing actual MuJoCo tracking cases; each letter's five methods
shared one duration. The corresponding KMP fixed/covariance-LQR force study
generated **16/16** passing tracking cases, with all control-effort and
deflection results retained. The covariance affects actual gains and plant
response; it is not only a display label.

The local suite ran **43 tests: 42 passed, one opt-in browser case skipped**.
The dedicated Chrome CI job additionally checks controller selection, playback
reset and file links. Linux/Windows CI exercises the complete restartable CLI
workflow; Windows also executes `run_demo.bat` with browser opening and pausing
disabled. The publishing PR's actual CI results are authoritative.

The Windows CI exposed a legacy-console encoding failure when a gallery path
contained Chinese characters. Progress output now escapes only unsupported
characters, without changing stdout's encoding or numerical calculations.
Gallery generation and the full demo/restart test both exercise Chinese output
paths against a strict cp1252 stream, including on non-Windows hosts.
Source hash keys also use forward slashes consistently across metadata creation
and report regeneration. This prevents Windows from treating an unchanged
completed run as modified when resuming; source-content checks remain enforced.

Empirical calibration is scored honestly: in the new full run, KMP hidden
coordinate coverage was 93.23% raw / 95.04% calibrated; ProMP and GP remained
below 95%. The protocol differs from v0.4, so historical changes cannot be
attributed solely to calibration. These are numerical and teaching-model
physics experiments, not physical hardware validation.

## Archived v0.2-v0.4 records

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

## Letter/method selection (v0.4)

- Local learning, physics and gallery checks passed **29 tests**; the separate opt-in real-browser case requires a Chrome/Chromium executable. This environment could not download a valid browser archive, so browser interaction runs in its dedicated GitHub Actions job using the runner's installed Chrome.
- A-H/KMP simulations were generated with the new batch command and combined into one page. All eight retained their declared tracking pass state. Existing simulation logs can also be combined without invoking MuJoCo.
- Merge tests verify case-specific metrics and relative file links, preserve original source files, retain failed tracking cases and reject duplicate or corrupted logs. UTF-8 paths are exercised explicitly.
- A real A/B learning run is used for the batch simulation integration test. UI fixtures in the browser test are explicitly synthetic and are not additional robot experiment results.
- Browser CI checks actual dropdown interactions, playback reset, changing method availability, failed-to-passed status changes, download/report targets, JavaScript errors and mobile horizontal overflow. Check the corresponding workflow result before treating the browser check as passed.
