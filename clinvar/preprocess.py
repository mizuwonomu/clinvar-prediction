"""Chuẩn bị bộ ClinVar Conflicting và kiểm tra các tệp đã xuất."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
KEY = ["CHROM", "POS", "REF", "ALT"]
AF = ["AF_ESP", "AF_EXAC", "AF_TGP"]
NUM = AF + ["CADD_PHRED"]
CAT = ["CLNVC", "Consequence", "IMPACT", "BIOTYPE"]
FEATURES = NUM + CAT
MISSING = {"", ".", "-", "NA", "N/A", "NAN", "NULL", "NONE", "?"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def class_summary(labels):
    return {
        str(label): {"count": int((labels == label).sum()), "share": float((labels == label).mean())}
        for label in (0, 1)
    }


def clean(raw):
    """Các phép đổi dưới đây áp dụng độc lập trên từng ô hoặc từng dòng."""
    require(set(KEY + FEATURES + ["CLASS"]) <= set(raw), "CSV thiếu cột bắt buộc.")
    fields = raw.apply(lambda col: col.str.strip())
    missing = fields.apply(lambda col: col.str.upper().isin(MISSING))
    fields = fields.mask(missing)
    labels = pd.to_numeric(fields["CLASS"], errors="coerce")
    ids = fields[KEY].copy()
    ids["CHROM"] = ids["CHROM"].str.upper().str.replace(r"^CHR", "", regex=True).replace({"M": "MT"})
    ids["POS"] = ids["POS"].str.replace(r"^0+", "", regex=True)
    for column in ["REF", "ALT"]:
        ids[column] = ids[column].str.upper()
    key_ok = (
        ids["CHROM"].str.fullmatch(r"(?:[1-9]|1[0-9]|2[0-2]|X|Y|MT)", na=False)
        & ids["POS"].str.fullmatch(r"[1-9][0-9]*", na=False)
        & ids["REF"].str.fullmatch(r"[ACGTN]+", na=False)
        & ids["ALT"].str.fullmatch(r"[ACGTN]+(?:,[ACGTN]+)*", na=False)
    )
    label_ok = labels.isin([0, 1])
    exact_dup = raw.duplicated(keep="first")
    ids["variant_id"] = ids[KEY].fillna("").agg(":".join, axis=1).where(key_ok)
    ids.index.name = "source_row"
    keep = key_ok & label_ok & ~exact_dup
    rows = raw.index[keep]
    exclusions = raw[KEY + ["CLASS"]].assign(
        invalid_key=~key_ok, invalid_label=~label_ok, exact_duplicate=exact_dup,
    ).loc[~keep]
    require(len(raw) == len(rows) + len(exclusions), "Sai số dòng bị loại.")

    X = pd.DataFrame(index=rows)
    invalid_numeric = {}
    for column in NUM:
        original = fields.loc[rows, column]
        values = pd.to_numeric(original, errors="coerce").astype("float64")
        valid = np.isfinite(values) & values.ge(0)
        if column in AF:
            valid &= values.le(1)
        invalid_numeric[column] = int((original.notna() & ~valid).sum())
        X[column] = values.where(valid)
    for column in CAT:
        X[column] = fields.loc[rows, column].astype(object).where(fields.loc[rows, column].notna(), np.nan)

    numeric_dtypes = {"float64", "int64"}
    inferred_dtypes = {}
    for column in raw:
        present = fields[column].dropna()
        parsed = pd.to_numeric(present, errors="coerce")
        kind = str(parsed.dtype)
        if len(present) and parsed.notna().all() and kind in numeric_dtypes:
            inferred_dtypes[column] = "float64" if missing[column].any() or kind == "float64" else "int64"
        else:
            inferred_dtypes[column] = "string"
    raw_groups = ids.loc[key_ok, ["variant_id"]].assign(CLASS=labels.where(label_ok))
    stats = raw_groups.groupby("variant_id").CLASS.agg(["size", "nunique"])
    repeated = stats["size"].gt(1)
    report = {
        "raw_shape": list(raw.shape),
        "raw_exact_duplicate_rows_after_first": int(exact_dup.sum()),
        "raw_keys": {
            "distinct_valid": len(stats),
            "repeated_keys": int(repeated.sum()),
            "rows_in_repeated_keys": int(stats.loc[repeated, "size"].sum()),
            "keys_with_both_classes": int(stats["nunique"].gt(1).sum()),
        },
        "raw_class_valid_labels": class_summary(labels[label_ok]),
        "invalid_key_rows": int((~key_ok).sum()),
        "invalid_label_rows": int((~label_ok).sum()),
        "rows_excluded": len(exclusions),
        "clean_shape": list(X.shape),
        "invalid_numeric_to_missing_after_exclusions": invalid_numeric,
        "raw_columns": [
            {"column": col, "dtype_inferred": inferred_dtypes[col],
             "missing": int(missing[col].sum()), "distinct_non_missing": int(fields[col].nunique())}
            for col in raw
        ],
    }
    return X, labels.loc[rows].astype("int8"), ids.loc[rows], exclusions, report


def split_rows(X, y, groups):
    """Phân tầng theo CLASS, đồng thời giữ nguyên nhóm biến thể."""
    require(set(y) == {0, 1}, "Cần có cả hai lớp sau khi làm sạch.")
    if groups.is_unique:
        train, test = train_test_split(X.index.to_numpy(), test_size=0.20, stratify=y, random_state=SEED)
        method = "train_test_split stratified (unique variant keys)"
    else:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        target_share = y.mean()
        candidates = []
        for train_pos, test_pos in cv.split(X, y, groups):
            if y.iloc[train_pos].nunique() == 2 and y.iloc[test_pos].nunique() == 2:
                error = abs(len(test_pos) / len(X) - 0.20) + abs(y.iloc[test_pos].mean() - target_share)
                candidates.append((error, train_pos, test_pos))
        require(candidates, "Không tìm được cách chia nhóm 5 fold mà cả hai tập đều có đủ hai lớp.")
        _, train_pos, test_pos = min(candidates, key=lambda candidate: candidate[0])
        train, test = X.index[train_pos].to_numpy(), X.index[test_pos].to_numpy()
        method = "StratifiedGroupKFold; best of 5 by test size and positive share errors"
    return {"train": np.sort(train), "test": np.sort(test)}, method


def make_preprocessor(scale_numeric=True):
    """Tạo pipeline chưa fit, nhận các đặc trưng từ X_train.csv hoặc X_test.csv."""
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", dtype=np.float64)),
    ])
    return Pipeline([("columns", ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUM),
        ("categorical", categorical, CAT),
    ], sparse_threshold=1.0))])


def load_split(directory, split):
    """Nạp X, y, IDs theo cùng source_row. source_row là index, không phải đặc trưng."""
    directory = Path(directory)
    X = pd.read_csv(directory / f"X_{split}.csv", index_col="source_row", keep_default_na=False,
                    na_values=[""], dtype={col: str for col in CAT}, float_precision="round_trip")
    y = pd.read_csv(directory / f"y_{split}.csv", index_col="source_row")["CLASS"].astype("int8")
    ids = pd.read_csv(directory / f"ids_{split}.csv", index_col="source_row", dtype=str,
                      keep_default_na=False)
    require(X.index.equals(y.index) and X.index.equals(ids.index), f"Lệch source_row ở {split}.")
    return X, y, ids


def load_matrix(directory, split, scaled=True):
    directory = Path(directory)
    variant = "scaled" if scaled else "unscaled"
    matrix = sparse.load_npz(directory / f"{variant}_{split}.npz")
    row_ids = np.load(directory / f"source_rows_{split}.npy", allow_pickle=False)
    return matrix, row_ids


def export_and_check(out, X, y, ids, splits, report):
    train, test = splits["train"], splits["test"]
    require(np.array_equal(np.sort(np.concatenate([train, test])), X.index), "Sai độ phủ train/test.")
    overlap = set(ids.loc[train, "variant_id"]) & set(ids.loc[test, "variant_id"])
    require(not overlap, "Trùng biến thể giữa train và test.")
    require(list(X.columns) == FEATURES, "X chứa cột ngoài danh sách cho phép.")

    report["splits"] = {}
    for split, rows in splits.items():
        X.loc[rows].to_csv(out / f"X_{split}.csv")
        y.loc[rows].to_csv(out / f"y_{split}.csv")
        ids.loc[rows].to_csv(out / f"ids_{split}.csv")
        np.save(out / f"source_rows_{split}.npy", rows, allow_pickle=False)
        loaded_X, loaded_y, loaded_ids = load_split(out, split)
        pd.testing.assert_frame_equal(loaded_X, X.loc[rows], check_dtype=False, check_exact=True)
        pd.testing.assert_series_equal(loaded_y, y.loc[rows])
        pd.testing.assert_frame_equal(loaded_ids, ids.loc[rows])
        report["splits"][split] = {
            "shape_before_transform": list(loaded_X.shape),
            "share_of_rows": len(rows) / len(X),
            "classes": class_summary(loaded_y),
            "distinct_variants": loaded_ids["variant_id"].nunique(),
            "missing_before_transform": {col: int(loaded_X[col].isna().sum()) for col in FEATURES},
        }

    report["preprocessors"] = {}
    for name, scaled in [("scaled", True), ("unscaled", False)]:
        pipe = make_preprocessor(scaled).fit(X.loc[train])
        names = pipe.get_feature_names_out().tolist()
        require(len(names) == len(set(names)), "Trùng tên cột sau biến đổi.")
        require(names[:len(NUM)] == [f"numeric__{col}" for col in NUM], "Sai tên cột số.")
        require(all(any(name.startswith(f"categorical__{col}_") for col in CAT)
                    for name in names[len(NUM):]), "Sai tên cột phân loại.")
        joblib.dump(pipe, out / f"preprocessor_{name}.joblib", compress=3)
        saved_pipe = joblib.load(out / f"preprocessor_{name}.joblib")
        require(names == saved_pipe.get_feature_names_out().tolist(), "Sai tên cột sau khi nạp pipeline.")
        write_json(out / f"feature_names_{name}.json", names)
        shapes = {}
        for split, rows in splits.items():
            matrix = sparse.csr_matrix(pipe.transform(X.loc[rows]))
            sparse.save_npz(out / f"{name}_{split}.npz", matrix, compressed=True)
            saved, row_ids = load_matrix(out, split, scaled)
            loaded_X, _, _ = load_split(out, split)
            repro = sparse.csr_matrix(saved_pipe.transform(loaded_X))
            require(np.array_equal(row_ids, loaded_X.index), f"Sai thứ tự ma trận {name}/{split}.")
            require(saved.shape == (len(rows), len(names)), f"Sai kích thước ma trận {name}/{split}.")
            require(np.isfinite(saved.data).all(), f"Ma trận {name}/{split} có NaN hoặc Inf.")
            require((saved != repro).nnz == 0, f"Pipeline nạp lại cho kết quả khác ở {name}/{split}.")
            shapes[split] = list(saved.shape)
        unseen = X.loc[test[:1]].copy()
        for col in CAT:
            unseen[col] = "__CHECK_UNSEEN_CATEGORY__"
        probe = sparse.csr_matrix(saved_pipe.transform(unseen))
        require(probe.shape == (1, len(names)) and np.isfinite(probe.data).all()
                and probe[:, len(NUM):].nnz == 0,
                f"Không xử lý được mức phân loại mới ở {name}.")
        numeric = pipe.named_steps["columns"].named_transformers_["numeric"]
        medians = dict(zip(NUM, numeric.named_steps["impute"].statistics_.tolist()))
        report["preprocessors"][name] = {
            "matrix_shapes": shapes, "train_imputation_values": medians, "unseen_category_check": "pass",
        }


def run(source, out):
    source = Path(source)
    out = Path(out)
    require(source.resolve().parent != out.resolve(), "Đặt CSV gốc ngoài thư mục kết quả.")
    raw = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    raw.index = pd.RangeIndex(1, len(raw) + 1, name="source_row")
    X, y, ids, exclusions, report = clean(raw)
    splits, method = split_rows(X, y, ids["variant_id"])
    repeated, _ = split_rows(X, y, ids["variant_id"])
    require(all(np.array_equal(splits[part], repeated[part]) for part in splits), "Chia tập thiếu tái lập.")
    out.mkdir(parents=True, exist_ok=True)
    exclusions.to_csv(out / "excluded_rows.csv")
    pd.DataFrame(report.pop("raw_columns")).to_csv(out / "raw_column_profile.csv", index=False)
    report.update({
        "input_file": source.name,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "features": FEATURES,
        "split_method": method,
        "random_state": SEED,
        "variant_overlap": 0,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
    })
    export_and_check(out, X, y, ids, splits, report)
    report["checks"] = {
        "split_reproducible": "pass", "both_classes_in_each_split": "pass",
        "all_clean_rows_partitioned": "pass", "variant_group_overlap_zero": "pass",
        "feature_allowlist": "pass", "row_alignment_on_reload": "pass",
        "finite_transformed_values": "pass", "same_transformed_schema": "pass",
        "reloaded_pipeline_matches_saved_matrices": "pass",
    }
    require(all(y.loc[rows].nunique() == 2 for rows in splits.values()), "Một tập bị mất lớp.")
    write_json(out / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Đường dẫn CSV gốc")
    parser.add_argument("--output", type=Path, required=True, help="Thư mục ghi kết quả")
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(f"{result['raw_shape']} -> {result['clean_shape']}; "
          f"train={result['splits']['train']['shape_before_transform'][0]}, "
          f"test={result['splits']['test']['shape_before_transform'][0]}")
