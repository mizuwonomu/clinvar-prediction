"""Xuất OOF đã khóa, phân tích lỗi và kiểm tra tách vị trí trên train."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from feature_study import MODELS, SEED, estimator
from preprocess import require
from study_endpoints import evaluate_threshold, fit_probs
from study_retained import LOCK_CONFIG, NEW, configuration, read_model_inputs


FOLDS = set(range(1, 6))
PROTEIN_FLAGS = ["has_SIFT", "has_PolyPhen", "has_BLOSUM62"]
AF = ["AF_ESP", "AF_EXAC", "AF_TGP"]
METRICS = ["ap", "precision", "recall", "f1"]
PREDICTION_COLUMNS = ["source_row", "CLASS", "model", "outer_fold", "probability", "fold_threshold", "prediction"]


def validate_folds(assignments, group_key):
    require(assignments.source_row.is_unique, "Trùng source_row trong phân fold.")
    require(assignments.outer_fold.notna().all() and set(assignments.outer_fold) == FOLDS, "Thiếu hoặc sai fold.")
    require(assignments[group_key].notna().all() and assignments.groupby(group_key).outer_fold.nunique().eq(1).all(),
            "Khóa nhóm xuất hiện ở nhiều fold.")
    require(assignments.groupby("outer_fold").CLASS.nunique().eq(2).all()
            and set(assignments.CLASS) == {0, 1}, "Fold thiếu lớp nhãn.")


def site_assignments(labels, ids):
    require(labels.source_row.equals(ids.source_row), "Khóa biến thể và nhãn lệch thứ tự.")
    site_id = ids.CHROM.str.cat(ids.POS, sep=":")
    outer = np.zeros(len(labels), dtype=int)
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (_, valid) in enumerate(splitter.split(labels, labels.CLASS, site_id), 1):
        require(not np.any(outer[valid]), "Dòng đã được gán fold.")
        outer[valid] = fold
    assignments = labels.assign(site_id=site_id, outer_fold=outer)
    validate_folds(assignments, "site_id")
    return assignments


def validate_predictions(rows, assignments, threshold_folds):
    require(list(rows) == PREDICTION_COLUMNS, "Schema OOF không đúng.")
    require(len(rows) == len(assignments) * len(MODELS) and set(rows.model) == set(MODELS)
            and not rows.duplicated(["source_row", "model"]).any(), "OOF thiếu dòng hoặc trùng source_row + model.")
    matched = rows.merge(assignments[["source_row", "CLASS", "outer_fold"]], on="source_row",
                         suffixes=("", "_expected"), how="left", validate="many_to_one")
    require(matched.CLASS.eq(matched.CLASS_expected).all() and matched.outer_fold.eq(matched.outer_fold_expected).all(),
            "Nhãn hoặc fold OOF lệch bảng phân fold.")
    matched = rows.merge(threshold_folds[["model", "fold", "threshold"]],
                         left_on=["model", "outer_fold"], right_on=["model", "fold"],
                         how="left", validate="many_to_one")
    require(rows.probability.between(0, 1).all() and rows.fold_threshold.between(0, 1).all(),
            "Xác suất hoặc ngưỡng không hợp lệ.")
    require(matched.fold_threshold.eq(matched.threshold).all(), "OOF không dùng đúng ngưỡng từng fold ngoài.")
    require(rows.prediction.eq(rows.probability.ge(rows.fold_threshold).astype(int)).all(),
            "Nhãn dự đoán khác probability >= fold_threshold.")


def confusion_metrics(y, pred):
    tp = int((y.eq(1) & pred.eq(1)).sum())
    fp = int((y.eq(0) & pred.eq(1)).sum())
    fn = int((y.eq(1) & pred.eq(0)).sum())
    tn = int((y.eq(0) & pred.eq(0)).sum())
    positive = tp + fn
    return dict(n=len(y), positive=positive, TP=tp, FP=fp, FN=fn, TN=tn,
                precision=tp / (tp + fp) if tp + fp else np.nan,
                recall=tp / positive if positive else np.nan,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
                miss_rate=fn / positive if positive else np.nan)


def errors_by_subgroup(rows, clean):
    merged = rows.merge(clean, on="source_row", how="left", validate="many_to_one", indicator=True)
    require(len(merged) == len(rows) and merged._merge.eq("both").all(), "Ghép OOF và bảng sạch lệch dòng.")
    missense = merged.Consequence_terms.fillna("").str.split("&").map(lambda terms: "missense_variant" in terms)
    require(merged[PROTEIN_FLAGS].isin([0, 1]).all().all(), "Cờ chú giải không hợp lệ.")
    protein = merged[PROTEIN_FLAGS].eq(1).any(axis=1)
    af = merged[AF]
    af_groups = pd.Series(np.select([af.eq(0).all(axis=1), af.gt(0).any(axis=1)],
                                   ["all_zero", "any_positive"], default="unclassified"), index=merged.index)
    region = merged.transcript_region.fillna("unresolved")
    require(region.isin(["exon", "intron", "exon_and_intron", "unresolved"]).all(), "Vùng transcript không hợp lệ.")
    definitions = {
        "transcript_region": (merged, region),
        "missense_protein": (merged.loc[missense], protein.loc[missense].map({True: "present", False: "missing"})),
        "allele_frequency": (merged, af_groups),
        "variant_type": (merged, merged.CLNVC.fillna("missing")),
    }
    result = []
    for name, (scope, groups) in definitions.items():
        section = []
        for (model, group), block in scope.assign(group=groups).groupby(["model", "group"], sort=True):
            section.append(dict(grouping=name, group=group, model=model,
                                **confusion_metrics(block.CLASS, block.prediction)))
        totals = ["n", "positive", "TP", "FP", "FN", "TN"]
        expected = {model: confusion_metrics(block.CLASS, block.prediction)
                    for model, block in scope.groupby("model")}
        if section:
            observed = pd.DataFrame(section).groupby("model")[totals].sum()
            pd.testing.assert_frame_equal(observed, pd.DataFrame.from_dict(expected, orient="index")[totals]
                                          .rename_axis("model").sort_index(), check_dtype=False)
        result.extend(section)
    return pd.DataFrame(result)


def summary_from_predictions(rows):
    per_fold, pooled = [], []
    for model, subset in rows.groupby("model"):
        records = []
        for fold, block in subset.groupby("outer_fold"):
            records.append(dict(model=model, outer_fold=fold,
                                ap=average_precision_score(block.CLASS, block.probability),
                                **confusion_metrics(block.CLASS, block.prediction)))
        fold_table = pd.DataFrame(records)
        per_fold.extend(records)
        dispersion = {f"{name}_std": fold_table[name].std(ddof=1) for name in METRICS}
        pooled.append(dict(model=model, ap=fold_table.ap.mean(),
                           **confusion_metrics(subset.CLASS, subset.prediction), **dispersion))
    return pd.DataFrame(per_fold), pd.DataFrame(pooled)


def run_split(root, assignments, models, lock, scheme):
    rows, fit_folds, operating = [], [], []
    fold_ids = assignments.outer_fold.to_numpy()
    y = assignments.set_index("source_row").CLASS
    for model, name in models.items():
        X = read_model_inputs(root, model, "train")
        pd.testing.assert_index_equal(X.index, y.index)
        _, fields = configuration(model, name, lock)
        require(X.columns.tolist() == fields, "Đầu vào lệch cấu hình đã khóa.")
        oof = np.full(len(y), np.nan)
        print(f"{scheme} {model}: outer folds", flush=True)
        for fold in sorted(FOLDS):
            train, valid = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
            oof[valid] = fit_probs(model, name, lock, "A", X, y, train, valid)
        print(f"{scheme} {model}: nested thresholds", flush=True)
        nested, summary, thresholds = evaluate_threshold(model, name, lock, "A", X, y, fold_ids, oof)
        fit_folds.extend(nested)
        operating.append(summary)
        rows.append(assignments[["source_row", "CLASS"]].assign(
            model=model, outer_fold=fold_ids, probability=oof, fold_threshold=thresholds,
            prediction=(oof >= thresholds).astype(int),
        ))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fit_folds), pd.DataFrame(operating)


def match_metrics(recomputed, expected, keys, values):
    one = recomputed.set_index(keys)[values].sort_index()
    two = expected.set_index(keys)[values].sort_index()
    pd.testing.assert_frame_equal(one, two, rtol=0, atol=1e-12, check_dtype=False)


def run(root):
    root = Path(root)
    load = lambda name, **options: pd.read_csv(root / name, float_precision="round_trip", **options)
    labels = load("y_train.csv")
    ids = load("ids_train.csv", dtype={"CHROM": str, "POS": str})
    variant = load("fold_assignments.csv").rename(columns={"variant_fold": "outer_fold"})
    snapshot = variant[["source_row", "CLASS"]]
    pd.testing.assert_frame_equal(snapshot, labels)
    require(len(labels) == 52150, "Tập train lệch mốc 52.150 dòng.")
    pd.testing.assert_series_equal(ids.source_row, labels.source_row)
    require(ids.CHROM.str.fullmatch(r"(?:[1-9]|1\d|2[0-2]|X|Y|MT)").all()
            and ids.POS.str.fullmatch(r"[1-9]\d*").all(), "CHROM/POS chưa chuẩn hóa.")
    variant = variant.assign(variant_id=ids.variant_id)
    validate_folds(variant, "variant_id")
    site = site_assignments(labels, ids)
    clean = load("clean_train.csv", usecols=["source_row", "transcript_region", "Consequence_terms", "CLNVC",
                                            *AF, *PROTEIN_FLAGS], keep_default_na=False, na_values=[""])
    pd.testing.assert_series_equal(clean.source_row, labels.source_row)
    lock = json.loads(LOCK_CONFIG.read_text(encoding="utf-8"))
    require(lock["seed"] == SEED and lock["models"] == {m: estimator(m).get_params() for m in MODELS},
            "Tham số mô hình khác cấu hình hiện hành.")
    models = {model: row["configuration"] for model, row in
              json.loads((root / "selected.json").read_text(encoding="utf-8")).items()}
    require(set(models) == set(MODELS), "Thiếu cấu hình đã khóa.")
    review = root / "endpoint_review"
    hist_fold = load("endpoint_review/threshold_folds.csv")
    hist_operating = load("endpoint_review/operating_metrics.csv")
    require(dict(zip(hist_operating.model, hist_operating.config)) == models and hist_operating.arm.eq("A").all(),
            "Bộ mốc ngưỡng trước không khớp cấu hình đã khóa.")
    with TemporaryDirectory(dir=review) as staging:
        temp = Path(staging)
        tables = []
        for scheme, assignments, filename in [
            ("variant", variant, "oof_predictions.csv"), ("site", site, "site_oof_predictions.csv"),
        ]:
            rows, by_fold, operating = run_split(root, assignments, models, lock, scheme)
            if scheme == "variant":
                match_metrics(by_fold, hist_fold, ["model", "fold"], ["n", "threshold", "inner_tuning_f1", *METRICS])
                match_metrics(operating, hist_operating, ["model"],
                              ["n", "positives", "ap_mean", "final_threshold", "tuning_f1", "precision", "recall", "f1"])
                old_threshold = hist_fold.set_index(["model", "fold"]).threshold
                new_threshold = by_fold.set_index(["model", "fold"]).threshold
                print(f"Historical fold thresholds: largest absolute difference {(old_threshold-new_threshold).abs().max():.3g}",
                      flush=True)
            rows.to_csv(temp / filename, index=False, float_format="%.17g")
            rows = pd.read_csv(temp / filename, float_precision="round_trip")
            validate_predictions(rows, assignments, by_fold)
            fold_output, summary = summary_from_predictions(rows)
            ref = hist_fold if scheme == "variant" else by_fold
            match_metrics(fold_output, ref.rename(columns={"fold": "outer_fold"}), ["model", "outer_fold"], ["n", *METRICS])
            match_metrics(summary, (hist_operating if scheme == "variant" else operating).rename(columns={"ap_mean": "ap"}),
                          ["model"], METRICS)
            tables.append(summary.assign(scheme=scheme))
            if scheme == "variant":
                errors_by_subgroup(rows, clean).to_csv(temp / "error_by_subgroup.csv", index=False, na_rep="NA")
                continue
            folds_path = temp / "site_fold_assignments.csv"
            site.to_csv(folds_path, index=False)
            pd.testing.assert_frame_equal(pd.read_csv(folds_path), site)
            stats = []
            for fold in sorted(FOLDS):
                valid = site.outer_fold.eq(fold)
                fit_sites, val_sites = set(site.loc[~valid, "site_id"]), set(site.loc[valid, "site_id"])
                shared = len(fit_sites & val_sites)
                require(shared == 0, "Rò rỉ vị trí giữa fit và validation.")
                stats.append(dict(outer_fold=fold, fit_sites=len(fit_sites), validation_sites=len(val_sites),
                                  shared_sites=shared))
            fold_output.merge(by_fold[["model", "fold", "threshold", "inner_tuning_f1"]].rename(
                columns={"fold": "outer_fold", "threshold": "fold_threshold"},
            ), on=["model", "outer_fold"], validate="one_to_one").merge(
                pd.DataFrame(stats), on="outer_fold", validate="many_to_one",
            ).to_csv(temp / "site_cv_folds.csv", index=False)
            summary.assign(sites=site.site_id.nunique()).to_csv(temp / "site_cv_summary.csv", index=False)
        comparison = pd.concat(tables, ignore_index=True)
        comparison.to_csv(temp / "cv_by_split_comparison.csv", index=False)
        for path in temp.iterdir():
            path.replace(review / path.name)
    print(comparison[["model", "scheme", "n", "positive", "ap", "f1", "ap_std", "f1_std"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=NEW)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)
