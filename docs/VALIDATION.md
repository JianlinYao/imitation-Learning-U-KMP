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
