"""So sánh cách xử lý mút tọa độ và đánh giá ngưỡng F1 trên train."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from feature_study import MODELS, estimator, scores
from preprocess import require, write_json
from study_retained import LOCK_CONFIG, NEW, configuration, model_inputs, pair, qualified, summarize


SOURCES = ("CDS_position", "Protein_position")
ARMS = ("A", "B", "C")


def inputs(clean, columns, name, arm):
    X = model_inputs(clean, columns, name)
    for source in SOURCES:
        partial = clean[f"{source}_status"].eq("partial")
        start = clean[f"{source}_start"]
        end = clean[f"{source}_end"]
        if arm != "A":
            X[f"{source}_start"] = start
        if arm == "C":
            X[f"{source}_partial_end"] = end.where(partial)
            X[f"{source}_partial_endpoint"] = np.select(
                [partial & start.notna(), partial & end.notna()],
                ["start_only", "end_only"], default="other",
            )
    return X


def make_pipeline(model, selected, lock, arm):
    pipe, columns = configuration(model, selected, lock)
    if arm == "C":
        transforms = pipe.named_steps["columns"].transformers
        transforms[0][2].extend(f"{source}_partial_end" for source in SOURCES)
        transforms[1][2].extend(f"{source}_partial_endpoint" for source in SOURCES)
    pipe.steps.append(("model", estimator(model)))
    return pipe, columns


def threshold_for(y, probabilities):
    """Tối đa hóa F1 lớp 1, lấy ngưỡng cao nhất khi F1 bằng nhau."""
    require(np.isfinite(probabilities).all() and len(set(y)) == 2, "OOF thiếu xác suất hoặc lớp.")
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    total = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], total, out=np.zeros_like(total), where=total > 0)
    idx = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[idx]), float(f1[idx])


def fit_probs(model, selected, lock, arm, X, y, train, valid):
    pipe, _ = make_pipeline(model, selected, lock, arm)
    pipe.fit(X.iloc[train], y.iloc[train])
    return pipe.predict_proba(X.iloc[valid])[:, 1]


def metrics(y, labels):
    p, r, f1, _ = precision_recall_fscore_support(y, labels, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f1}


def evaluate_threshold(model, selected, lock, arm, X, y, fold_ids, oof):
    final_threshold, tuning_f1 = threshold_for(y, oof)
    thresholds = np.full(len(y), np.nan)
    fold_numbers = sorted(set(fold_ids))
    rows = []
    inner_oof = {outer: np.full(len(y), np.nan) for outer in fold_numbers}
    # Một cặp fold bị loại có cùng phần fit khi hoán đổi vòng trong và vòng ngoài.
    for first, second in combinations(fold_numbers, 2):
        held = (fold_ids == first) | (fold_ids == second)
        train, valid = np.flatnonzero(~held), np.flatnonzero(held)
        prob = fit_probs(model, selected, lock, arm, X, y, train, valid)
        for outer, inner in ((first, second), (second, first)):
            use = fold_ids[valid] == inner
            inner_oof[outer][valid[use]] = prob[use]
    for outer in fold_numbers:
        outer_fit = fold_ids != outer
        threshold, inner_f1 = threshold_for(y[outer_fit], inner_oof[outer][outer_fit])
        valid = fold_ids == outer
        thresholds[valid] = threshold
        rows.append(dict(model=model, arm=arm, fold=outer, n=int(valid.sum()), threshold=threshold,
                         inner_tuning_f1=inner_f1, ap=average_precision_score(y[valid], oof[valid]),
                         **metrics(y[valid], oof[valid] >= threshold)))
    pred = oof >= thresholds
    return rows, dict(model=model, config=selected, arm=arm, n=len(y), positives=int(y.sum()),
                      ap_mean=float(np.mean([x["ap"] for x in rows])),
                      final_threshold=final_threshold, tuning_f1=tuning_f1, **metrics(y, pred)), thresholds


def run(root):
    requested = {"source_row", "Consequence_terms",
                 *(f"{source}_{part}" for source in SOURCES for part in ("raw", "status", "start", "end"))}
    schemas = {m: json.loads((root / m / "input_columns.json").read_text(encoding="utf-8")) for m in MODELS}
    for columns in schemas.values():
        requested.update(columns)
    clean = pd.read_csv(root / "clean_train.csv", usecols=requested, index_col="source_row",
                        keep_default_na=False, na_values=[""], float_precision="round_trip")
    folds = pd.read_csv(root / "fold_assignments.csv", index_col="source_row")
    y = pd.read_csv(root / "y_train.csv", index_col="source_row").CLASS
    pd.testing.assert_index_equal(clean.index, y.index)
    pd.testing.assert_index_equal(folds.index, y.index)
    pd.testing.assert_series_equal(folds.CLASS, y)
    require(y.index.is_unique, "Trùng source_row.")
    lock = json.loads(LOCK_CONFIG.read_text(encoding="utf-8"))
    selected = json.loads((root / "selected.json").read_text(encoding="utf-8"))
    subset = clean.CDS_position_status.eq("partial")
    pd.testing.assert_series_equal(subset, clean.Protein_position_status.eq("partial"), check_names=False)
    for source in SOURCES:
        require(int(clean.loc[subset, f"{source}_start"].notna().sum()) == 25
                and int(clean.loc[subset, f"{source}_end"].notna().sum()) == 10,
                "Số mút đã biết lệch mốc đối chiếu.")
    require(int(subset.sum()) == 35 and int(y[subset].sum()) == 8, "Nhóm partial lệch mốc đối chiếu.")
    missense = clean.Consequence_terms.fillna("").str.split("&").map(lambda parts: "missense_variant" in parts)
    masks = {"all": np.ones(len(y), dtype=bool), "missense": missense.to_numpy()}
    records, detail, threshold_rows, operating = [], [], [], []
    decisions = {}
    historical = pd.read_csv(root / "cv_folds.csv")
    for model in MODELS:
        config = selected[model]["configuration"]
        _, columns = configuration(model, config, lock)
        require(columns == schemas[model],
                "Mốc A lệch schema đã khóa.")
        frames = {arm: inputs(clean, columns, config, arm) for arm in ARMS}
        partial = clean.loc[subset, [f"{s}_raw" for s in SOURCES]].assign(CLASS=y[subset])
        variant_oof = {}
        for scheme in ("variant", "gene"):
            fold_ids = folds[f"{scheme}_fold"].to_numpy()
            require(set(fold_ids) == set(range(1, 6)), "Fold không đầy đủ.")
            partial[f"{scheme}_fold"] = fold_ids[subset]
            for arm in ARMS:
                probabilities = np.full(len(y), np.nan)
                print(f"{model} {scheme} {arm}", flush=True)
                for fold in range(1, 6):
                    train = np.flatnonzero(fold_ids != fold)
                    valid = np.flatnonzero(fold_ids == fold)
                    prob = fit_probs(model, config, lock, arm, frames[arm], y, train, valid)
                    probabilities[valid] = prob
                    for result in scores(y.iloc[valid], prob, {g: m[valid] for g, m in masks.items()}):
                        records.append(dict(config=arm, model=model, scheme=scheme, fold=fold, **result))
                partial[f"{scheme}_{arm}"] = probabilities[subset]
                if scheme == "variant":
                    variant_oof[arm] = probabilities
            partial[f"{scheme}_delta_B_A"] = partial[f"{scheme}_B"] - partial[f"{scheme}_A"]
            partial[f"{scheme}_delta_C_B"] = partial[f"{scheme}_C"] - partial[f"{scheme}_B"]
        compared = pair(pd.DataFrame(records).query("model == @model"), "A")
        chosen = next(iter(qualified(summarize(compared), model)), "A")
        actual = compared.query("config == 'A'").set_index(["scheme", "fold", "subgroup"])
        expected = historical.query("model == @model and config == @config and subgroup in ['all','missense']").set_index(actual.index.names)
        np.testing.assert_allclose(actual.ap, expected.loc[actual.index, "ap"], rtol=0, atol=1e-12)
        decisions[model] = {"baseline": config, "selected_arm": chosen}
        detail.append(partial.assign(model=model))
        print(f"{model} selected {chosen}, nested threshold CV", flush=True)
        by_fold, pooled, _ = evaluate_threshold(model, config, lock, chosen, frames[chosen], y,
                                               folds.variant_fold.to_numpy(), variant_oof[chosen])
        threshold_rows.extend(by_fold)
        operating.append(pooled)
    out = root / "endpoint_review"
    out.mkdir(exist_ok=True)
    paired = pair(pd.DataFrame(records), "A")
    paired.to_csv(out / "cv_folds.csv", index=False)
    summarize(paired).to_csv(out / "cv_summary.csv", index=False)
    pd.concat(detail).to_csv(out / "partial_oof.csv")
    pd.DataFrame(threshold_rows).to_csv(out / "threshold_folds.csv", index=False)
    pd.DataFrame(operating).to_csv(out / "operating_metrics.csv", index=False)
    write_json(out / "selection.json", decisions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=NEW)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)
