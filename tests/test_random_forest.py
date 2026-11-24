import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
import random_forest as rf


class RandomForestContractTests(unittest.TestCase):
    def test_selection_uses_simplicity_inside_tolerance(self):
        cfg = rf.load_config()
        rows = []
        for cid, params, score, se in [
            ("complex", {"n_estimators": 200, "min_samples_leaf": 5, "max_features": "sqrt", "max_depth": 16}, .812, .003),
            ("simple", {"n_estimators": 100, "min_samples_leaf": 10, "max_features": "sqrt", "max_depth": 12}, .811, .003),
            ("baseline", cfg["baseline"], .800, .002),
        ]:
            rows.append({"config_id": cid, "params": json.dumps(params), "valid_ap_mean": score,
                         "valid_ap_se": se, "folds": 5})
        frame = pd.DataFrame(rows)
        folds = pd.DataFrame([
            {"config_id": cid, "fold": f, "valid_ap": val}
            for cid, vals in [("complex", [.82]*5), ("simple", [.81]*5), ("baseline", [.80]*5)]
            for f, val in enumerate(vals, 1)])
        frame.attrs["folds_frame"] = folds
        chosen = rf.select_configuration(frame, "baseline", cfg)
        self.assertEqual(chosen.config_id, "simple")

    def test_predefined_split_matches_variant_fold_contract(self):
        assignments = np.array([1, 2, 1, 2, 1, 2])
        split = PredefinedSplit(assignments - 1)
        pairs = list(split.split())
        self.assertEqual(len(pairs), 2)
        self.assertTrue(np.array_equal(pairs[0][1], [0, 2, 4]))

    def test_train_reader_does_not_read_test_labels(self):
        import study_retained
        original = study_retained.read_model_inputs
        calls = []
        def spy(out, model, split):
            calls.append(split)
            return original(out, model, split)
        study_retained.read_model_inputs = spy
        try:
            rf.load_train()
        finally:
            study_retained.read_model_inputs = original
        self.assertEqual(calls, ["train"])

    def test_cv_csv_columns_are_contract_columns(self):
        path = ROOT / "results" / "random_forest" / "cv_folds.csv"
        self.assertTrue(path.exists(), "Run pipeline to create the contracted CV output")
        columns = pd.read_csv(path, nrows=1).columns.tolist()
        self.assertEqual(columns, rf.FOLD_COLUMNS)

    def test_preprocessing_is_inside_each_model_pipeline(self):
        X = pd.DataFrame({"numeric": [1.0, np.nan, 3.0], "category": ["a", "b", "a"]})
        cfg = rf.load_config()
        pipe = rf.build_pipeline(X, cfg, {"n_estimators": 2, "max_depth": 2,
                                          "min_samples_leaf": 1, "max_features": "sqrt"})
        self.assertIn("preprocess", pipe.named_steps)
        prep = pipe.named_steps["preprocess"]
        self.assertFalse(hasattr(prep, "transformers_"))
        prep.fit(X.iloc[:2])
        learned_median = prep.named_transformers_["numeric"].statistics_[0]
        self.assertEqual(learned_median, 1.0)
        self.assertNotEqual(learned_median, X.numeric.median())


if __name__ == "__main__":
    unittest.main()
