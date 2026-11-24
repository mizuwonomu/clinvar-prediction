import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import PredefinedSplit

from src.logistic_regression import (
    FOLD_COLUMNS,
    SUMMARY_COLUMNS,
    load_config,
    load_training_data,
    make_predefined_splits,
    run_solver_comparison,
    save_comparison,
    summarize_comparison,
)


ROOT = Path(__file__).resolve().parents[1]


class LogisticRegressionSolverDiagnosticTests(unittest.TestCase):
    def test_config_keeps_solver_diagnostic_separate_from_official_solver(self):
        config = load_config(ROOT / "configs" / "logistic_regression.yaml")
        diagnostic = next(r for r in config["rounds"] if r["type"] == "diagnostic")
        self.assertEqual(diagnostic["solvers"], ["lbfgs", "liblinear"])
        self.assertEqual(diagnostic["params"], {
            "C": 1.0, "penalty": "l2", "class_weight": None,
            "max_iter": 3000, "random_state": 42,
        })
        self.assertEqual(config["fixed"]["solver"], "liblinear")
        self.assertTrue(all(
            round_config.get("parameter") != "solver"
            for round_config in config["rounds"]
            if round_config["type"] == "tuning"
        ))

    def test_config_uses_safe_yaml_loader_and_expected_schema(self):
        config_path = ROOT / "configs" / "logistic_regression.yaml"
        with config_path.open(encoding="utf-8") as stream:
            parsed = yaml.safe_load(stream)
        self.assertEqual(set(parsed), {
            "model", "input_set", "fixed", "baseline", "rounds", "simplicity_order",
            "selection", "final",
        })
        self.assertEqual(load_config(config_path), parsed)

    def test_predefined_split_uses_existing_fold_ids_and_has_disjoint_rows(self):
        fold_values = pd.Series(np.repeat([1, 2, 3, 4, 5], 10))
        splits = make_predefined_splits(fold_values)
        reference = list(PredefinedSplit(fold_values.to_numpy() - 1).split())
        self.assertEqual(len(splits), 5)
        for (train, valid), (expected_train, expected_valid) in zip(splits, reference):
            np.testing.assert_array_equal(train, expected_train)
            np.testing.assert_array_equal(valid, expected_valid)
            self.assertFalse(np.intersect1d(train, valid).size)

    def test_real_train_inputs_align_by_source_row_and_keep_all_variant_folds(self):
        X, y, variant_fold = load_training_data()
        self.assertEqual(len(X), 52_150)
        self.assertEqual(len(y), 52_150)
        self.assertTrue(X.index.equals(y.index))
        self.assertTrue(X.index.equals(variant_fold.index))
        self.assertFalse(X.index.has_duplicates)
        self.assertEqual(variant_fold.value_counts().sort_index().to_dict(),
                         {fold: 10_430 for fold in range(1, 6)})

    def test_metrics_and_paired_fold_deltas_are_calculated_without_rounding(self):
        labels = np.array([0, 1, 0, 1])
        probabilities = np.array([0.1, 0.8, 0.4, 0.6])
        self.assertEqual(
            average_precision_score(labels, probabilities, pos_label=1), 1.0,
        )
        self.assertAlmostEqual(log_loss(labels, probabilities, labels=[0, 1]),
                               -np.mean(np.log([0.9, 0.8, 0.6, 0.6])))

        rows = []
        for fold in range(1, 6):
            for solver, ap in (("lbfgs", 0.50 + fold / 100), ("liblinear", 0.51 + fold / 100)):
                rows.append({
                    "solver": solver, "fold": fold, "valid_ap": ap,
                    "valid_logloss": 0.4, "converged": True,
                })
        summary = summarize_comparison(pd.DataFrame(rows))
        self.assertEqual(summary.columns.tolist(), SUMMARY_COLUMNS)
        deltas = summary.query("record_type == 'paired_fold_delta'")
        self.assertEqual(len(deltas), 5)
        np.testing.assert_allclose(deltas.delta_ap_liblinear_minus_lbfgs, 0.01)

    def test_both_solvers_run_on_identical_data_folds_and_pipeline(self):
        columns = json.loads((ROOT / "clinvar" / "feature_retained" / "LogisticRegression"
                              / "input_columns.json").read_text(encoding="utf-8"))
        n_rows = 50
        index = pd.Index(np.arange(1001, 1001 + n_rows), name="source_row")
        numeric_columns = {
            "log_AF_ESP", "log_AF_EXAC", "log_AF_TGP", "CADD_RAW", "LoFtool",
            "EXON_index", "EXON_total", "EXON_fraction", "INTRON_index", "INTRON_total",
            "INTRON_fraction", "CDS_position_start", "CDS_position_span",
            "Protein_position_start", "Protein_position_span", "has_LoFtool", "has_SIFT",
            "has_PolyPhen", "has_BLOSUM62", "zero_AF_ESP", "zero_AF_EXAC", "zero_AF_TGP",
        }
        data = {}
        for column in columns:
            if column in numeric_columns:
                data[column] = np.linspace(0.0, 1.0, n_rows)
            else:
                data[column] = np.resize(
                    np.array(["single_nucleotide_variant", "missense_variant", "MODERATE", "protein_coding"]),
                    n_rows,
                )
        X = pd.DataFrame(data, index=index)[columns]
        y = pd.Series(np.tile([0, 1], n_rows // 2), index=index, name="CLASS")
        variant_fold = pd.Series(np.repeat([1, 2, 3, 4, 5], n_rows // 5), index=index)
        result = run_solver_comparison(X, y, variant_fold, load_config(), "test-commit")
        self.assertEqual(len(result), 10)
        self.assertEqual(set(result.solver), {"lbfgs", "liblinear"})
        for fold in range(1, 6):
            pair = result[result.fold.eq(fold)].set_index("solver")
            self.assertEqual(pair.loc["lbfgs", "n_train"], pair.loc["liblinear", "n_train"])
            self.assertEqual(pair.loc["lbfgs", "n_valid"], pair.loc["liblinear", "n_valid"])
            self.assertEqual(pair.loc["lbfgs", "positive_valid"], pair.loc["liblinear", "positive_valid"])
            lbfgs_params = json.loads(pair.loc["lbfgs", "params"])
            liblinear_params = json.loads(pair.loc["liblinear", "params"])
            self.assertEqual({k: v for k, v in lbfgs_params.items() if k != "solver"},
                             {k: v for k, v in liblinear_params.items() if k != "solver"})
            self.assertEqual(lbfgs_params["solver"], "lbfgs")
            self.assertEqual(liblinear_params["solver"], "liblinear")

    def test_result_writer_uses_separate_files_and_refuses_overwrite(self):
        folds = pd.DataFrame(columns=FOLD_COLUMNS)
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            fold_path, summary_path = save_comparison(folds, summary, output_dir)
            self.assertTrue(fold_path.exists())
            self.assertTrue(summary_path.exists())
            with self.assertRaises(FileExistsError):
                save_comparison(folds, summary, output_dir)


if __name__ == "__main__":
    unittest.main()
