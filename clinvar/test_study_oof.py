import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from study_endpoints import evaluate_threshold
from study_oof import MODELS, errors_by_subgroup, validate_folds, validate_predictions


class OutOfFoldTests(unittest.TestCase):
    def setUp(self):
        self.assignments = pd.DataFrame({"source_row": [4, 9], "CLASS": [1, 0], "outer_fold": [1, 2]})
        self.thresholds = pd.DataFrame(
            [(model, fold, threshold) for model in MODELS for fold, threshold in [(1, .4), (2, .7)]],
            columns=["model", "fold", "threshold"],
        )
        self.rows = pd.concat([
            self.assignments[["source_row", "CLASS"]].assign(
                model=model, outer_fold=[1, 2], probability=[.4, .6], fold_threshold=[.4, .7], prediction=[1, 0])
            for model in MODELS
        ], ignore_index=True)

    def test_rejects_duplicate_source_row_and_model(self):
        validate_predictions(self.rows, self.assignments, self.thresholds)
        duplicate = self.rows.copy()
        duplicate.iloc[1] = duplicate.iloc[0]
        with self.assertRaisesRegex(ValueError, r"trùng source_row \+ model"):
            validate_predictions(duplicate, self.assignments, self.thresholds)

    def test_rejects_global_threshold_in_outer_folds(self):
        incorrect = self.rows.assign(fold_threshold=.5)
        incorrect["prediction"] = incorrect.probability.ge(.5).astype(int)
        with self.assertRaisesRegex(ValueError, "ngưỡng từng fold"):
            validate_predictions(incorrect, self.assignments, self.thresholds)

    def test_rejects_site_leakage_between_folds(self):
        frame = pd.DataFrame({
            "source_row": range(10), "CLASS": [0, 1] * 5,
            "site_id": [f"1:{fold}" for fold in range(5) for _ in range(2)],
            "outer_fold": np.repeat(np.arange(1, 6), 2),
        })
        validate_folds(frame, "site_id")
        frame.loc[2, "site_id"] = frame.loc[0, "site_id"]
        with self.assertRaisesRegex(ValueError, "nhiều fold"):
            validate_folds(frame, "site_id")

    def test_nested_fit_excludes_both_held_groups(self):
        fold_ids = np.repeat(np.arange(1, 6), 2)
        y = pd.Series([0, 1] * 5)

        def mock_fit(model, selected, lock, arm, X, labels, train, valid):
            self.assertEqual(len(set(fold_ids[valid])), 2)
            self.assertFalse(set(fold_ids[train]) & set(fold_ids[valid]))
            return np.where(labels.iloc[valid] == 1, .8, .2)

        with patch("study_endpoints.fit_probs", side_effect=mock_fit) as fit:
            rows, summary, thresholds = evaluate_threshold(
                MODELS[0], "example", {}, "A", pd.DataFrame(index=y.index),
                y, fold_ids, np.where(y == 1, .8, .2),
            )
        self.assertEqual(fit.call_count, 10)
        self.assertEqual(len(rows), 5)
        self.assertEqual(summary["f1"], 1)
        np.testing.assert_array_equal(thresholds, [.8] * 10)

    def test_subgroup_uses_positives_as_denominator_and_token_matching(self):
        clean = pd.DataFrame({
            "source_row": [4, 9, 11, 15],
            "Consequence_terms": ["splice&missense_variant", "missense_variant", "not_missense_variant", None],
            "transcript_region": ["exon", "exon", "intron", "unresolved"],
            "CLNVC": ["snp", "snp", "deletion", None],
            "AF_ESP": [0., 0., 0.1, np.nan], "AF_EXAC": [0., 0., np.nan, 0.],
            "AF_TGP": [0., 0., 0., 0.], "has_SIFT": [0, 0, 1, 0],
            "has_PolyPhen": [0, 1, 0, 0], "has_BLOSUM62": [0, 0, 0, 0],
        })
        predictions = pd.DataFrame({
            "source_row": [4, 9, 11, 15], "CLASS": [1, 0, 0, 1],
            "prediction": [0, 0, 0, 1], "model": MODELS[0],
        })
        report = errors_by_subgroup(predictions, clean).set_index(["grouping", "group"])
        exon = report.loc["transcript_region", "exon"]
        self.assertEqual(exon.FN, 1)
        self.assertEqual(exon.miss_rate, 1)
        self.assertTrue(np.isnan(exon.precision))
        protein = report.loc["missense_protein"]
        self.assertEqual(protein.n.sum(), 2)
        self.assertTrue(np.isnan(protein.loc["present", "miss_rate"]))
        self.assertEqual(report.loc[("allele_frequency", "unclassified"), "n"], 1)


if __name__ == "__main__":
    unittest.main()
