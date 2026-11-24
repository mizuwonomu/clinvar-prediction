# Random Forest result snapshot

- Configuration: `configs/random_forest.yaml`; selected parameters are stored under `final`.
- Validation: `PredefinedSplit(variant_fold - 1)`, 5 fixed folds; the test labels are scored once after choosing the threshold from train out-of-fold probabilities.
- CV logs: `results/random_forest/cv_folds.csv` and `cv_summary.csv`.
- Test log: `results/random_forest/test_metrics.csv`.
- The baseline was retained unless a candidate cleared both the 0.002 mean AP gain and 4-of-5-fold win rules. The selected config is the simplest member of the best-score tolerance band.
- The test result is a single-use holdout result. Do not retune against it; any later correction must record a new run and preserve this result for comparison.
