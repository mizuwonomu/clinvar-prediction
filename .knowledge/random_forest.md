# Random Forest result snapshot

- Configuration: `configs/random_forest.yaml`; selected parameters are stored under `final`.
- Validation: `PredefinedSplit(variant_fold - 1)`, 5 fixed folds; the test labels are scored once after choosing the threshold from train out-of-fold probabilities.
- CV logs: `results/random_forest/cv_folds.csv` and `cv_summary.csv`.
- Test log: `results/random_forest/test_metrics.csv`.
- The baseline was retained unless a candidate cleared both the 0.002 mean AP gain and 4-of-5-fold win rules. The selected config is the simplest member of the best-score tolerance band.
- The test result is a single-use holdout result. Do not retune against it; any later correction must record a new run and preserve this result for comparison.
- Run history: the initial test evaluation from the non-sequential search was AP 0.505299 at threshold 0.483469 (accuracy 0.697500, precision 0.439824, recall 0.730453, F1 0.549051, ROC-AUC 0.776055; confusion matrix TN 6693, FP 3058, FN 886, TP 2401). The search-order defect was then fixed and a second evaluation was performed as the contract directs when correcting an issue after opening test. The corrected sequential run is the current row in `test_metrics.csv`: CV AP 0.505900; threshold 0.512584; test AP 0.510065 (accuracy 0.721123, precision 0.463471, recall 0.673563, F1 0.549107, ROC-AUC 0.779366; TN 7188, FP 2563, FN 1073, TP 2214).
