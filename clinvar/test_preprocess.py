"""Kiểm thử hồi quy cho bộ nạp và hai cách mã hóa Consequence."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline

from preprocess import FEATURES, NUM, load_matrix, load_split, make_preprocessor


def feature_frame(consequences, index=None):
    return pd.DataFrame({
        "AF_ESP": 0.0, "AF_EXAC": 0.01, "AF_TGP": 0.0, "CADD_PHRED": 10.0,
        "CLNVC": "SNV", "Consequence": consequences, "IMPACT": "LOW", "BIOTYPE": "protein_coding",
    }, index=index, columns=FEATURES)


class TestLoaders(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)
        self.index = pd.Index([2, 4, 8], name="source_row")
        self.X = feature_frame(["intron_variant", "missense_variant", None], self.index)
        self.X.loc[2, "CADD_PHRED"] = np.nan
        self.y = pd.DataFrame({"CLASS": [0, 1, 0]}, index=self.index)
        self.ids = pd.DataFrame({
            "CHROM": "1", "POS": ["100", "101", "102"], "REF": "A", "ALT": "G",
            "variant_id": ["1:100:A:G", "1:101:A:G", "1:102:A:G"],
        }, index=self.index)
        self.matrix = sparse.csr_matrix([[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
        self.X.to_csv(self.directory / "X_test.csv")
        self.y.to_csv(self.directory / "y_test.csv")
        self.ids.to_csv(self.directory / "ids_test.csv")
        self.write_matrix(self.matrix)
        self.write_rows(self.index.to_numpy())
        self.write_names(["first", "second"])

    def write_matrix(self, matrix):
        sparse.save_npz(self.directory / "scaled_test.npz", sparse.csr_matrix(matrix))

    def write_rows(self, ids):
        np.save(self.directory / "source_rows_test.npy", ids, allow_pickle=False)

    def write_names(self, names):
        (self.directory / "feature_names_scaled.json").write_text(json.dumps(names), encoding="utf-8")

    def assert_invalid(self, loader, name, message):
        with self.assertRaises(ValueError) as error:
            loader(self.directory, "test")
        for fragment in ["[test]", name, message]:
            self.assertIn(fragment, str(error.exception))

    def test_valid_split_and_missing_values_are_imputed(self):
        X, y, ids = load_split(self.directory, "test")
        self.assertEqual(y.dtype, np.dtype("int8"))
        self.assertTrue(X.index.equals(ids.index) and X.index.equals(y.index))
        self.assertTrue(X[NUM].dtypes.eq("float64").all())
        self.assertTrue(np.isnan(X.loc[2, "CADD_PHRED"]))
        self.assertTrue(pd.isna(X.loc[8, "Consequence"]))
        for mode in ["onehot", "multihot"]:
            with self.subTest(mode=mode):
                prep = make_preprocessor(False, mode)
                matrix = sparse.csr_matrix(prep.fit_transform(X)).toarray()
                self.assertEqual(matrix[0, NUM.index("CADD_PHRED")], 10.0)
                self.assertTrue(np.isfinite(matrix).all())
        self.assertIsInstance(load_matrix(self.directory, "test")[0], sparse.csr_matrix)

    def test_labels_validated_before_int8(self):
        for label in ["256", "257", "-256", "1.5", "1.0000000000000000001", "", "abc"]:
            with self.subTest(label=label):
                bad = self.y.astype(str)
                bad.loc[2, "CLASS"] = label
                bad.to_csv(self.directory / "y_test.csv")
                self.assert_invalid(load_split, "y_test.csv", "CLASS chỉ được chứa 0 hoặc 1")

    def test_missing_unexpected_and_duplicate_columns(self):
        for bad in [self.X.drop(columns="BIOTYPE"), self.X.assign(CLASS=0),
                    self.X.rename(columns={"BIOTYPE": "IMPACT"})]:
            with self.subTest(columns=bad.columns.tolist()):
                bad.to_csv(self.directory / "X_test.csv")
                self.assert_invalid(load_split, "X_test.csv", "Cột phải đúng thứ tự")

    def test_csv_row_extra_or_missing_field(self):
        path = self.directory / "y_test.csv"
        for row in ["2,0,3", "2"]:
            with self.subTest(row=row):
                path.write_text(f"source_row,CLASS\n{row}\n", encoding="utf-8")
                self.assert_invalid(load_split, "y_test.csv", "Bản ghi 1 phải có 2 trường")

    def test_duplicate_source_row_in_each_csv(self):
        files = [("X_test.csv", self.X), ("y_test.csv", self.y), ("ids_test.csv", self.ids)]
        for filename, frame in files:
            with self.subTest(filename=filename):
                frame.set_axis(pd.Index([2, 2, 8], name="source_row")).to_csv(self.directory / filename)
                self.assert_invalid(load_split, filename, "source_row bị trùng")
                frame.to_csv(self.directory / filename)

    def test_invalid_source_row(self):
        for row in ["0", "-2", "2.5", "", "9223372036854775808"]:
            with self.subTest(row=row):
                bad = self.X.set_axis(pd.Index([row, "4", "8"], name="source_row"))
                bad.to_csv(self.directory / "X_test.csv")
                self.assert_invalid(load_split, "X_test.csv", "source_row")

    def test_order_of_labels_or_ids_must_match(self):
        for filename, frame in [("y_test.csv", self.y), ("ids_test.csv", self.ids)]:
            with self.subTest(filename=filename):
                frame.iloc[::-1].to_csv(self.directory / filename)
                self.assert_invalid(load_split, filename, "source_row không khớp")
                self.assert_invalid(load_matrix, filename, "source_row không khớp")
                frame.to_csv(self.directory / filename)

    def test_variant_id_must_match_key(self):
        bad = self.ids.copy()
        bad.loc[2, "variant_id"] = "1:999:A:G"
        bad.to_csv(self.directory / "ids_test.csv")
        self.assert_invalid(load_split, "ids_test.csv", "variant_id không khớp")

    def test_bad_numeric_or_infinite_csv_value(self):
        for value in ["not_numeric", "inf", "-inf"]:
            with self.subTest(value=value):
                bad = self.X.astype({"CADD_PHRED": object})
                bad.loc[2, "CADD_PHRED"] = value
                bad.to_csv(self.directory / "X_test.csv")
                expected = "not_numeric" if value == "not_numeric" else "giá trị vô hạn"
                self.assert_invalid(load_split, "X_test.csv", expected)

    def test_wrong_matrix_shape(self):
        for shape, message in [((2, 2), "Số hàng"), ((3, 1), "Số cột")]:
            with self.subTest(shape=shape):
                self.write_matrix(sparse.csr_matrix(shape))
                self.assert_invalid(load_matrix, "scaled_test.npz", message)

    def test_duplicate_or_invalid_feature_names(self):
        for names, message in [(["same", "same"], "bị trùng"), (["first", ""], "chuỗi khác rỗng")]:
            with self.subTest(names=names):
                self.write_names(names)
                self.assert_invalid(load_matrix, "feature_names_scaled.json", message)

    def test_duplicate_invalid_or_reordered_matrix_ids(self):
        for rows, message in [([2, 2, 8], "bị trùng"), ([2.0, 4.0, 8.0], "số nguyên một chiều"),
                              ([8, 4, 2], "không khớp"), ([0, 4, 8], "số nguyên dương")]:
            with self.subTest(rows=rows):
                self.write_rows(rows)
                self.assert_invalid(load_matrix, "source_rows_test.npy", message)

    def test_nonfinite_matrix(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value):
                bad = self.matrix.copy()
                bad.data[0] = value
                self.write_matrix(bad)
                self.assert_invalid(load_matrix, "scaled_test.npz", "NaN hoặc giá trị vô hạn")


class TestEncoding(unittest.TestCase):
    def test_multihot_terms_missing_and_unseen(self):
        train = feature_frame(["missense_variant", "splice_region_variant & intron_variant"])
        test = feature_frame(["missense_variant", "splice_region_variant & intron_variant & intron_variant ",
                              np.nan, " & ", "unseen_variant", "missense_variant & unseen_variant"])
        prep = make_preprocessor(consequence_encoding="multihot").fit(train)
        names = prep.get_feature_names_out().tolist()
        positions = [i for i, name in enumerate(names) if name.startswith("consequence__")]
        self.assertEqual([names[i] for i in positions], [
            "consequence__Consequence_term_intron_variant",
            "consequence__Consequence_term_missense_variant",
            "consequence__Consequence_term_splice_region_variant",
            "consequence__Consequence_missing",
        ])
        encoded = sparse.csr_matrix(prep.transform(test))[:, positions].toarray()
        np.testing.assert_array_equal(encoded, [
            [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1],
            [0, 0, 0, 0], [0, 1, 0, 0],
        ])

    def test_default_keeps_combination_and_matches_explicit_onehot(self):
        X = feature_frame(["missense_variant", "intron_variant&splice_region_variant"])
        default = make_preprocessor().fit(X)
        explicit = make_preprocessor(consequence_encoding="onehot").fit(X)
        self.assertIn("categorical__Consequence_intron_variant&splice_region_variant",
                      default.get_feature_names_out())
        np.testing.assert_array_equal(default.get_feature_names_out(), explicit.get_feature_names_out())
        self.assertEqual((sparse.csr_matrix(default.transform(X)) !=
                          sparse.csr_matrix(explicit.transform(X))).nnz, 0)

    def test_joblib_loads_in_fresh_process(self):
        X = feature_frame(["intron_variant&splice_region_variant", "missense_variant", None])
        for mode in ["onehot", "multihot"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                prep = make_preprocessor(consequence_encoding=mode).fit(X)
                joblib.dump(prep, directory / "pipeline.joblib")
                X.to_pickle(directory / "X.pkl")
                sparse.save_npz(directory / "expected.npz", sparse.csr_matrix(prep.transform(X)))
                command = (
                    "from pathlib import Path; import sys, joblib, pandas as pd; from scipy import sparse; "
                    "d = Path(sys.argv[1]); p = joblib.load(d/'pipeline.joblib'); "
                    "actual = sparse.csr_matrix(p.transform(pd.read_pickle(d/'X.pkl'))); "
                    "assert (actual != sparse.load_npz(d/'expected.npz')).nnz == 0"
                )
                result = subprocess.run([sys.executable, "-B", "-c", command, tmp], capture_output=True,
                                        cwd=Path(__file__).resolve().parent, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_each_cv_fold_fits_its_own_vocabulary(self):
        terms = np.array([f"term_{i:02d}" for i in range(24)])
        X = feature_frame(terms)
        y = np.arange(len(X)) % 2
        groups = np.arange(len(X))
        folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X, y, groups))
        for mode, prefix in [("onehot", "categorical__Consequence_"),
                             ("multihot", "consequence__Consequence_term_")]:
            with self.subTest(mode=mode):
                pipe = Pipeline([("prep", make_preprocessor(consequence_encoding=mode)),
                                 ("model", LogisticRegression())])
                result = cross_validate(pipe, X, y, cv=folds, return_estimator=True, error_score="raise")
                self.assertFalse(hasattr(pipe.named_steps["prep"].named_steps["columns"], "transformers_"))
                for (train, validation), estimator in zip(folds, result["estimator"]):
                    names = estimator.named_steps["prep"].get_feature_names_out()
                    got = {name.removeprefix(prefix) for name in names if name.startswith(prefix)}
                    self.assertEqual(got, set(terms[train]))
                    self.assertTrue(got.isdisjoint(terms[validation]))


if __name__ == "__main__":
    unittest.main()
