"""Five-fold solver diagnostic for the locked ClinVar Logistic Regression inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import PredefinedSplit
from threadpoolctl import threadpool_limits

from src.paths import CONFIGS, FEATURE_RETAINED, RESULTS, ROOT
from study_retained import configuration, read_model_inputs


MODEL = "LogisticRegression"
FOLD_COUNT = 5
TRAIN_ROWS = 52_150
FOLD_ROWS = 10_430
FOLD_COLUMNS = [
    "config_id", "params", "solver", "fold", "n_train", "n_valid", "positive_valid", "train_ap",
    "valid_ap", "valid_logloss", "n_iter", "converged", "fit_seconds",
    "convergence_warnings", "git_commit", "seed",
]
SUMMARY_COLUMNS = [
    "record_type", "solver", "fold", "mean_valid_ap", "std_valid_ap",
    "se_valid_ap", "mean_valid_logloss", "converged_folds",
    "delta_ap_liblinear_minus_lbfgs",
]


def load_config(path: Path | None = None) -> dict:
    """Read and validate the model config without evaluating any code in YAML."""
    config_path = path or CONFIGS / "logistic_regression.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {
        "model", "input_set", "fixed", "baseline", "rounds", "simplicity_order",
        "selection", "final",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"Config keys must be exactly {sorted(required)}.")
    if config["model"] != MODEL or config["fixed"] != {
        "solver": "liblinear", "max_iter": 3000, "random_state": 42,
    }:
        raise ValueError("Official LR solver/fixed parameters differ from the agreed config.")
    if config["input_set"] != "af_log+af_zero+consequence_rare":
        raise ValueError("LR input set does not match the locked feature selection.")
    diagnostic_round = next((r for r in config["rounds"] if r["type"] == "diagnostic"), None)
    if diagnostic_round is None or diagnostic_round["solvers"] != ["lbfgs", "liblinear"]:
        raise ValueError("Solver diagnostic must compare lbfgs and liblinear in this order.")
    expected_params = {
        "C": 1.0, "penalty": "l2", "class_weight": None,
        "max_iter": 3000, "random_state": 42,
    }
    if diagnostic_round["params"] != expected_params:
        raise ValueError("Solver diagnostic parameters must match exactly.")
    if config["baseline"] != {
        "C": 1.0, "penalty": "l2", "class_weight": None, "solver": "liblinear",
    }:
        raise ValueError("Official baseline must use C=1, L2, no class weighting, and liblinear.")
    return config


def load_training_data(feature_dir: Path = FEATURE_RETAINED):
    """Load the three contract train inputs and align them by source_row."""
    X = read_model_inputs(feature_dir, MODEL, "train")
    y_frame = pd.read_csv(feature_dir / "y_train.csv", index_col="source_row")
    folds = pd.read_csv(feature_dir / "fold_assignments.csv", index_col="source_row")
    if X.index.has_duplicates or y_frame.index.has_duplicates or folds.index.has_duplicates:
        raise ValueError("Duplicate source_row found in model inputs, labels, or folds.")
    pd.testing.assert_index_equal(X.index, y_frame.index, check_names=True)
    pd.testing.assert_index_equal(X.index, folds.index, check_names=True)
    if y_frame.columns.tolist() != ["CLASS"] or "CLASS" not in folds:
        raise ValueError("Expected CLASS in y_train.csv and fold_assignments.csv.")
    y = y_frame["CLASS"].astype(np.int8)
    pd.testing.assert_series_equal(folds["CLASS"].astype(y.dtype), y)
    if set(y.unique()) != {0, 1}:
        raise ValueError("CLASS must contain exactly labels 0 and 1.")
    if "variant_fold" not in folds:
        raise ValueError("fold_assignments.csv is missing variant_fold.")
    variant_fold = folds["variant_fold"].astype(int)
    if len(X) != TRAIN_ROWS or set(variant_fold.unique()) != set(range(1, FOLD_COUNT + 1)):
        raise ValueError("Train row count or variant_fold values differ from the data contract.")
    sizes = variant_fold.value_counts().sort_index()
    if not sizes.eq(FOLD_ROWS).all():
        raise ValueError("Each existing variant_fold must contain 10,430 rows.")
    return X, y, variant_fold


def make_predefined_splits(variant_fold: pd.Series):
    """Materialize and validate the existing 1-based variant assignments."""
    values = np.asarray(variant_fold, dtype=int)
    if set(np.unique(values)) != set(range(1, FOLD_COUNT + 1)):
        raise ValueError("variant_fold must contain exactly 1 through 5.")
    splitter = PredefinedSplit(test_fold=values - 1)
    splits = list(splitter.split())
    if len(splits) != FOLD_COUNT:
        raise ValueError("PredefinedSplit did not produce exactly five folds.")
    validation_seen = np.zeros(len(values), dtype=np.int8)
    for fold_number, (train_idx, valid_idx) in enumerate(splits, start=1):
        if np.intersect1d(train_idx, valid_idx).size:
            raise ValueError(f"Train/validation overlap in fold {fold_number}.")
        validation_seen[valid_idx] += 1
    if not np.all(validation_seen == 1):
        raise ValueError("Every row must be validation exactly once.")
    return splits


def _new_pipeline(solver: str, params: dict):
    """Build the locked preprocessing pipeline fresh for each solver/fold fit."""
    selected_path = FEATURE_RETAINED / "selected.json"
    input_columns_path = FEATURE_RETAINED / MODEL / "input_columns.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))[MODEL]
    columns = json.loads(input_columns_path.read_text(encoding="utf-8"))
    if selected["configuration"] != "af_log+af_zero+consequence_rare" or selected["input_columns"] != 27:
        raise ValueError("Locked selected.json Logistic Regression input has changed.")
    # The established feature-study builder includes fold-fit imputation, scaling,
    # encoding, and RareConsequence; assert it still builds exactly the locked X.
    lock_path = ROOT / "clinvar" / "feature_study" / "locked.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    preprocessor, built_columns = configuration(MODEL, selected["configuration"], lock)
    if built_columns != columns:
        raise ValueError("Preprocessing builder columns differ from input_columns.json.")
    estimator = LogisticRegression(
        solver=solver,
        C=params["C"],
        penalty=params["penalty"],
        class_weight=params["class_weight"],
        max_iter=params["max_iter"],
        random_state=params["random_state"],
    )
    preprocessor.steps.append(("model", estimator))
    return preprocessor


def run_solver_comparison(X, y, variant_fold, config: dict, git_commit: str):
    """Run only the two fixed solver diagnostics on the same five predefined folds."""
    if not X.index.equals(y.index) or not X.index.equals(variant_fold.index):
        raise ValueError("X, y, and variant_fold indexes are not identical.")
    diagnostic = next(r for r in config["rounds"] if r["type"] == "diagnostic")
    params = diagnostic["params"]
    splits = make_predefined_splits(variant_fold)
    rows = []
    with threadpool_limits(limits=1):
        for solver in diagnostic["solvers"]:
            for fold_number, (train_idx, valid_idx) in enumerate(splits, start=1):
                pipeline = _new_pipeline(solver, params)
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    start = time.perf_counter()
                    pipeline.fit(X_train, y_train)
                    fit_seconds = time.perf_counter() - start
                convergence = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
                train_probability = pipeline.predict_proba(X_train)[:, 1]
                valid_probability = pipeline.predict_proba(X_valid)[:, 1]
                estimator = pipeline.named_steps["model"]
                config_id = f"solver_diagnostic:{solver}"
                rows.append({
                    "config_id": config_id,
                    "params": json.dumps({**params, "solver": solver}, sort_keys=True),
                    "solver": solver,
                    "fold": fold_number,
                    "n_train": len(train_idx),
                    "n_valid": len(valid_idx),
                    "positive_valid": int(y_valid.sum()),
                    "train_ap": float(average_precision_score(y_train, train_probability, pos_label=1)),
                    "valid_ap": float(average_precision_score(y_valid, valid_probability, pos_label=1)),
                    "valid_logloss": float(log_loss(y_valid, valid_probability, labels=[0, 1])),
                    "n_iter": int(np.max(estimator.n_iter_)),
                    "converged": not convergence,
                    "fit_seconds": fit_seconds,
                    "convergence_warnings": " | ".join(str(w.message) for w in convergence),
                    "git_commit": git_commit,
                    "seed": params["random_state"],
                })
    result = pd.DataFrame(rows, columns=FOLD_COLUMNS)
    if len(result) != len(diagnostic["solvers"]) * FOLD_COUNT:
        raise ValueError("Solver comparison did not produce both solvers on all folds.")
    return result


def summarize_comparison(folds: pd.DataFrame) -> pd.DataFrame:
    """Summarize solver metrics and paired per-fold AP differences without rounding."""
    expected_solvers = {"lbfgs", "liblinear"}
    if set(folds["solver"]) != expected_solvers:
        raise ValueError("Expected both lbfgs and liblinear results.")
    pivot = folds.pivot(index="fold", columns="solver", values="valid_ap")
    rows = []
    for solver, group in folds.groupby("solver", sort=False):
        values = group["valid_ap"].to_numpy(dtype=float)
        rows.append({
            "record_type": "solver_summary",
            "solver": solver,
            "fold": np.nan,
            "mean_valid_ap": float(np.mean(values)),
            "std_valid_ap": float(np.std(values, ddof=1)),
            "se_valid_ap": float(np.std(values, ddof=1) / np.sqrt(len(values))),
            "mean_valid_logloss": float(group["valid_logloss"].mean()),
            "converged_folds": int(group["converged"].sum()),
            "delta_ap_liblinear_minus_lbfgs": np.nan,
        })
    for fold_number, difference in (pivot["liblinear"] - pivot["lbfgs"]).items():
        rows.append({
            "record_type": "paired_fold_delta",
            "solver": "liblinear_minus_lbfgs",
            "fold": int(fold_number),
            "mean_valid_ap": np.nan,
            "std_valid_ap": np.nan,
            "se_valid_ap": np.nan,
            "mean_valid_logloss": np.nan,
            "converged_folds": np.nan,
            "delta_ap_liblinear_minus_lbfgs": float(difference),
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def current_git_commit(root: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True,
    ).strip()


def save_comparison(folds: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
    """Write new diagnostic artifacts only; refuse to overwrite any existing result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "solver_comparison.csv"
    summary_path = output_dir / "solver_comparison_summary.csv"
    existing = [path.name for path in (fold_path, summary_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing solver diagnostic files: {existing}")
    folds.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    return fold_path, summary_path


def run(output_dir: Path = RESULTS / "logistic_regression", config_path: Path | None = None):
    config = load_config(config_path)
    X, y, variant_fold = load_training_data()
    folds = run_solver_comparison(X, y, variant_fold, config, current_git_commit())
    summary = summarize_comparison(folds)
    return save_comparison(folds, summary, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS / "logistic_regression.yaml")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "logistic_regression")
    args = parser.parse_args()
    fold_path, summary_path = run(args.output_dir, args.config)
    print(f"Saved solver diagnostic folds: {fold_path.relative_to(ROOT)}")
    print(f"Saved solver diagnostic summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
