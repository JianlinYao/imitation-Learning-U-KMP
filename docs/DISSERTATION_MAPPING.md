# Dissertation-to-code implementation map

The dissertation title, research questions, five method labels and three task categories are retained. This code completion does not edit the PDF and does not claim that newly measured values are the original dissertation results.

| Dissertation content | Executable implementation | Verification / output |
|---|---|---|
| Chapter 3: preprocessing | `data.py`: MATLAB loading, resampling, segmented smoothing, training-only scaler | masking and scaler tests; `splits.json` |
| Chapter 3: five-fold comparison | `experiments.py`: whole-demo, per-letter common folds | disjoint split/coverage tests; all raw trial rows |
| Chapter 4: DMP | `models.py`: pooled forcing fit and RK45 rollout | correct-sign synthetic regression test |
| Chapter 4: ProMP | `models.py`: joint position/velocity basis and Gaussian weights | constraint and covariance tests |
| Chapter 4: GP | `models.py`: RBF mean/covariance using all demos | exact repeated-observation equivalence test |
| Chapter 4: KMP | `reference.py`, `kernels.py`, `models.py`: GMM/GMR and derivative-kernel KMP | reference refit provenance; kernel/PSD tests |
| Uncertainty-aware extension | `control.py`: covariance-dependent finite-horizon tracking | gain and extrapolation-variance tests; `control_ablation.csv` |
| Reproduction | common endpoint-conditioned held-out protocol | `metrics.csv`, trajectory figures |
| Missing completion | mask middle segment, process only visible runs, condition Gaussian models | hidden-value perturbation invariance; gap-only metrics |
| Obstacle adaptation | shared circle/waypoints, bounded repair, continuous clearance | cross-segment/inside/zero-length collision tests |
| Chapter 5: evaluation | real RMSE/EPE/Jerk/timing/coverage; no outlier filtering | saved predictions, recomputed metrics, trial accounting |
| Appendix C: running demo | CLI and generated offline viewer | README commands; end-to-end and CI smoke run |

## Points to reconcile with the dissertation

1. **Result tables:** rerun outputs are new measurements. Do not tune the implementation to reproduce a preferred ranking or copy old numbers into generated CSV files.
2. **Planning versus control:** KMP already supports predictive covariance. The cited uncertainty-aware work uses covariance for control; attaching the name U-KMP to a different arbitrary regularizer does not establish that extension. In the completed code, KMP/U-KMP planning can be identical; control outcomes are reported separately.
3. **Missing task:** some historical notebooks evaluate prefix extrapolation (40%/60% observed). The new default instead implements the dissertation's middle-gap completion. Prefix completion is not silently pooled with middle-gap results.
4. **Cross-validation:** historical experiments include leave-one-out. The new default implements the stated five-fold protocol. Test trajectories do not fit scalers or reference models.
5. **Different prose conclusions:** Chapter 4 describes U-KMP as best for missing completion, while Chapter 5 describes GP as best. Reconcile this against measured, clearly identified experiment settings rather than assuming both describe one identical result.
6. **Dataset scope:** this repository provides the A–H two-dimensional letter data. Human–robot handover/painting data and physical robot results are not added or implied.
7. **Safety and dynamic environments:** the implemented obstacle benchmark is a declared static circular-obstacle numerical task. Moving obstacles, robot sensor/actuator effects and physical safety remain outside its validated scope. They require their own explicit experiment, not a relabeling of this task.
8. **Conditioning:** the new implementation conditions a learned joint Gaussian trajectory. Historical KMP notebooks insert/replace reference points. This implementation difference is explicit in `METHODOLOGY.md` and can change numeric results.

The original notebook calculations remain available for historical comparison. New training, evaluation and reports should use `python -m imitation_learning`, not legacy placeholder/filtered summary cells.

## v0.5 supplemental experiments

Reference KMP adaptation, training-only empirical calibration, explicit path/gap success criteria and actual MuJoCo task-space LQR control are described in [Research improvements](RESEARCH_IMPROVEMENTS.md). They produce new versioned results; the dissertation PDF and previously recorded tables are retained. The original `benchmark.json` now explicitly chooses historical Gaussian adaptation, while `research.json` chooses reference-point updates.
