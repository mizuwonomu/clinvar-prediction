"""Xuất bảng ClinVar sạch và chọn đầu vào theo mô hình bằng CV trên train."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from consequence_encoder import RareConsequence
from feature_study import (
    MODELS, OUTPUT, ROOT, SEED, estimator, load_rows, preprocessor, read_source, scores, subgroups,
)
from preprocess import KEY, require, write_json
from retained_features import EXTRA_POSITIONS, PROTEIN_VALUES, build_clean


LOCKED = "locked_24"
HISTORY = OUTPUT / "vy_thao"
NEW = ROOT / "feature_retained"
MIN_GAIN = 0.002
MAX_LOSS = 0.002
ALLOW_HISTORICAL = {"af_zero", "af_log", "af_log+af_zero", "cadd_present"}
NEW_ARMS = ("protein_confidence", "positions", "consequence_rare", "impact_order", "chrom")
RECORD = "scheme model config fold subgroup n positive ap precision recall f1".split()
MATCH = "scheme model fold subgroup n positive".split()
LOCK_CONFIG = OUTPUT / "locked.json"


def configuration(model, name, lock):
    changes = [] if name == LOCKED else name.split("+")
    historical = [arm for arm in changes if arm.startswith("af_") or arm == "cadd_present"]
    require(set(changes).issubset(set(NEW_ARMS) | set(historical)), "Nhánh đặc trưng chưa được định nghĩa.")
    pipe, _ = preprocessor(model, [*lock["arms"], *historical], lock["encoding"])
    transformers = pipe.named_steps["columns"].transformers
    numbers, categories = transformers[0][2], transformers[1][2]
    if "protein_confidence" in changes:
        numbers.extend(["SIFT_low_confidence", "BLOSUM62_score"])
        categories.extend(col for col in PROTEIN_VALUES if col not in numbers)
    if "positions" in changes:
        numbers.extend(col for col in EXTRA_POSITIONS if not col.endswith("_status"))
        categories.extend(col for col in EXTRA_POSITIONS if col.endswith("_status"))
    if "consequence_rare" in changes:
        categories.remove("Consequence")
        transformers.append(("terms", RareConsequence(), ["Consequence"]))
    if "impact_order" in changes:
        categories.remove("IMPACT")
        numbers.append("IMPACT_rank")
    if "chrom" in changes:
        categories.append("CHROM")
    columns = [col for _, _, inputs in transformers for col in inputs]
    forbidden = set(KEY[1:]) | {"source_row", "variant_id", "CLASS", "SYMBOL", "CLNSIGINCL", "ORIGIN",
                              "CLNDN", "CLNDN_status", "CLNDN_unknown_only"}
    require(len(columns) == len(set(columns)) and not forbidden.intersection(columns),
            "Cột trùng hoặc cột truy vết/hồ sơ nằm trong ma trận.")
    return pipe, columns


def model_inputs(clean, columns, name):
    # Hai đầu vào CDS/protein của bộ mốc dùng quy tắc cũ, kể cả khoảng thiếu một đầu.
    X = clean[columns].copy()
    if "positions" not in name.split("+"):
        for source in ("CDS_position", "Protein_position"):
            col = f"{source}_start"
            if col in X:
                X[col] = X[col].mask(clean[f"{source}_status"].eq("partial"))
    return X


def read_model_inputs(out, model, split):
    """Nạp đầu vào đúng cấu hình đã xuất từ bảng sạch và từ điển cột."""
    require(split in {"train", "test"} and model in MODELS, "Model hoặc split không hợp lệ.")
    out = Path(out)
    selected = json.loads((out / "selected.json").read_text(encoding="utf-8"))[model]["configuration"]
    columns = json.loads((out / model / "input_columns.json").read_text(encoding="utf-8"))
    dictionary = pd.read_csv(out / "column_dictionary.csv", index_col="column")
    extra = ["CDS_position_status", "Protein_position_status"]
    requested = ["source_row", *dict.fromkeys(columns + extra)]
    strings = dict.fromkeys(dictionary.loc[requested].query("type == 'object'").index, object)
    clean = pd.read_csv(out / f"clean_{split}.csv", usecols=requested, dtype=strings, index_col="source_row",
                        keep_default_na=False, na_values=[""], float_precision="round_trip")
    return model_inputs(clean, columns, selected)


def get_folds(base, y, ids):
    assignments = pd.read_csv(OUTPUT / "fold_assignments.csv", index_col="source_row")
    pd.testing.assert_index_equal(assignments.index, y.index)
    pd.testing.assert_series_equal(assignments.CLASS.astype(y.dtype), y)
    require(set(assignments.fold) == set(range(1, 6)), "Mã fold theo biến thể không đủ.")
    variant = [(np.flatnonzero(assignments.fold != i), np.flatnonzero(assignments.fold == i))
               for i in range(1, 6)]
    genes = base.SYMBOL.where(base.SYMBOL.ne("__MISSING__"), "unknown:" + ids.variant_id)
    gene = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(base, y, genes))
    assignments = assignments.rename(columns={"fold": "variant_fold"})
    assignments["gene_fold"] = 0
    for scheme, pairs in [("variant", variant), ("gene", gene)]:
        for fold, (train, valid) in enumerate(pairs, 1):
            require(y.iloc[train].nunique() == y.iloc[valid].nunique() == 2, "Fold thiếu lớp.")
            require(set(ids.variant_id.iloc[train]).isdisjoint(ids.variant_id.iloc[valid]), "Trùng biến thể.")
            if scheme == "gene":
                require(set(genes.iloc[train]).isdisjoint(genes.iloc[valid]), "Trùng gene.")
                require(assignments.iloc[valid].gene_fold.eq(0).all(), "Dòng trùng fold gene.")
                assignments.iloc[valid, assignments.columns.get_loc("gene_fold")] = fold
    require(assignments.gene_fold.gt(0).all(), "Dòng thiếu fold gene.")
    return {"variant": variant, "gene": gene}, assignments


def reuse_metrics(folds, masks, y, lock):
    current = lock["selected"]
    baseline_v = pd.read_csv(OUTPUT / "cv_fold_metrics.csv").query("config == @current")
    baseline_g = pd.read_csv(OUTPUT / "gene_fold_metrics.csv").query("scheme == 'gene' and config == 'selected'")
    frames = [
        baseline_v.assign(scheme="variant", config=LOCKED), baseline_g.assign(config=LOCKED),
        pd.read_csv(HISTORY / "cv_folds.csv").assign(scheme="variant"),
        pd.read_csv(HISTORY / "gene_folds.csv").assign(scheme="gene"),
    ]
    frame = pd.concat(frames, ignore_index=True)[RECORD]
    require(set(frame.config).issubset(ALLOW_HISTORICAL | {LOCKED}), "Có cấu hình lịch sử chưa được xác minh.")
    expected = []
    for scheme, pairs in folds.items():
        for fold, (_, valid) in enumerate(pairs, 1):
            for subgroup, mask in masks.items():
                chosen = valid[mask[valid]]
                expected.append((scheme, fold, subgroup, len(chosen), int(y.iloc[chosen].sum())))
    keys = [col for col in MATCH if col != "model"]
    checked = frame.merge(pd.DataFrame(expected, columns=keys), on=keys, how="left",
                          indicator=True, validate="many_to_one")
    require(checked._merge.eq("both").all() and not frame.duplicated(["config", *MATCH]).any(),
            "Không thể ghép chỉ số lịch sử với danh sách fold hiện tại.")
    complete = frame.groupby(["scheme", "model", "config"]).size()
    require(complete.eq(5 * len(masks)).all(), "Kết quả lịch sử thiếu nhóm hoặc fold.")
    return frame


def evaluate(name, model, scheme, clean, y, masks, folds, lock):
    print(f"{scheme} {model} {name}", flush=True)
    result = []
    for fold, (train, valid) in enumerate(folds, 1):
        pipe, columns = configuration(model, name, lock)
        X = model_inputs(clean, columns, name)
        pipe.steps.append(("model", estimator(model)))
        pipe.fit(X.iloc[train], y.iloc[train])
        prob = pipe.predict_proba(X.iloc[valid])[:, 1]
        for stats in scores(y.iloc[valid], prob, {g: mask[valid] for g, mask in masks.items()}):
            result.append(dict(config=name, model=model, scheme=scheme, fold=fold, **stats))
    return pd.DataFrame(result)


def pair(frame, reference=LOCKED):
    ref = frame.loc[frame.config.eq(reference), MATCH + ["ap", "recall"]]
    merged = frame.merge(ref, on=MATCH, suffixes=("", "_reference"), validate="many_to_one")
    require(len(merged) == len(frame), "Thiếu fold tham chiếu.")
    merged["delta_ap"] = merged.ap - merged.ap_reference
    merged["delta_recall"] = merged.recall - merged.recall_reference
    return merged.assign(reference=reference)


def summarize(frame):
    return frame.groupby(["config", "model", "scheme", "subgroup"], sort=False).agg(
        ap=("ap", "mean"), delta_ap=("delta_ap", "mean"), recall=("recall", "mean"),
        folds_up=("delta_ap", lambda values: int(values.gt(0).sum())),
        folds_valid=("delta_ap", "count"),
    ).reset_index()


def qualified(frame, model):
    """Ưu tiên toàn tập, với giới hạn suy giảm trên gene và trên missense."""
    main = frame.query("model == @model and subgroup in ['all', 'missense']")
    outcome = {}
    for config, group in main.groupby("config"):
        if len(group) != 4 or group.folds_valid.ne(5).any() or group.delta_ap.min() < -MAX_LOSS:
            continue
        variant = group.query("scheme == 'variant' and subgroup == 'all'").iloc[0]
        if variant.delta_ap >= MIN_GAIN and variant.folds_up >= 4:
            outcome[config] = variant.delta_ap
    return sorted(outcome, key=lambda name: (-outcome[name], name))


def plan_combinations(singles):
    """Thử ghép tín hiệu đã qua sàng lọc ở ít nhất một mô hình."""
    winners = {model: qualified(singles, model) for model in MODELS}
    best_af = {
        af for names in winners.values() if (af := next((name for name in names if name.startswith("af_")), None))
    }
    robust = [name for name in NEW_ARMS if any(name in wins for wins in winners.values())]
    planned = {f"{af}+{arm}" for af in best_af for arm in robust}
    if len(robust) >= 2:
        planned.update("+".join([*([af] if af else []), *robust]) for af in (best_af or {None}))
    return sorted(planned)


def matrix_export(out, selected, lock, train, test):
    (X, y, ids), (Xt, yt, sid) = train, test
    require(X.index.intersection(Xt.index).empty, "Trùng source_row giữa hai tập.")
    require(set(ids.variant_id).isdisjoint(sid.variant_id), "Trùng khóa biến thể giữa hai tập.")
    manifest = {}
    for model, name in selected.items():
        folder = out / model
        folder.mkdir(exist_ok=True)
        pipe, columns = configuration(model, name, lock)
        fit = model_inputs(X, columns, name)
        pipe.fit(fit)
        schema = pipe.get_feature_names_out().tolist()
        require(len(schema) == len(set(schema)), "Tên đặc trưng sau mã hóa bị trùng.")
        saved = folder / "preprocessor.joblib"
        joblib.dump(pipe, saved, compress=3)
        loaded = joblib.load(saved)
        require(loaded.get_feature_names_out().tolist() == schema, "Schema thay đổi sau khi nạp.")
        write_json(folder / "input_columns.json", columns)
        write_json(folder / "feature_names.json", schema)
        manifest[model] = {"configuration": name, "input_columns": len(columns), "encoded_columns": len(schema)}
        for split, clean, labels in [("train", X, y), ("test", Xt, yt)]:
            frame = model_inputs(clean, columns, name)
            pd.testing.assert_index_equal(frame.index, labels.index)
            matrix = sparse.csr_matrix(pipe.transform(frame))
            require(matrix.shape == (len(labels), len(schema)) and np.isfinite(matrix.data).all(),
                    "Kích thước ma trận hoặc giá trị không hợp lệ.")
            if name == LOCKED:
                old_tag = "scaled" if model == "LogisticRegression" else "unscaled"
                old_matrix = sparse.load_npz(OUTPUT / "pipeline" / f"{old_tag}_{split}.npz")
                require((matrix != old_matrix).nnz == 0, "Bộ mốc không tái tạo đúng từng phần tử.")
            path = folder / f"X_{split}.npz"
            sparse.save_npz(path, matrix)
            restored = sparse.csr_matrix(loaded.transform(frame))
            require((sparse.load_npz(path) != restored).nnz == 0, "Ma trận không khớp sau khi nạp lại.")
            row_path = folder / f"source_rows_{split}.npy"
            np.save(row_path, frame.index.to_numpy(), allow_pickle=False)
            require(np.array_equal(np.load(row_path, allow_pickle=False), labels.index), "Mã dòng không khớp.")
    return manifest


def verify_written_inputs(out):
    """Đọc lại CSV đã xuất bằng đúng đường nạp công bố cho người dùng."""
    for split in ("train", "test"):
        labels = pd.read_csv(out / f"y_{split}.csv", index_col="source_row").CLASS
        ids = pd.read_csv(out / f"ids_{split}.csv", index_col="source_row")
        pd.testing.assert_index_equal(labels.index, ids.index)
        require(labels.index.is_unique and labels.isin([0, 1]).all(), "Mã dòng hoặc nhãn không hợp lệ.")
        for model in MODELS:
            X = read_model_inputs(out, model, split)
            folder = out / model
            pipe = joblib.load(folder / "preprocessor.joblib")
            saved = sparse.load_npz(folder / f"X_{split}.npz")
            pd.testing.assert_index_equal(X.index, labels.index)
            require(np.array_equal(X.index, np.load(folder / f"source_rows_{split}.npy", allow_pickle=False)),
                    "CSV và ma trận khác thứ tự source_row.")
            replay = sparse.csr_matrix(pipe.transform(X))
            require(saved.shape == replay.shape and (saved != replay).nnz == 0,
                    "Đọc lại CSV rồi biến đổi cho ma trận khác.")


def publish(out, source, lock, clean, y, ids, results):
    from retained_publication import write_report
    from feature_documentation import decision_table

    paired = pair(results)
    summary = summarize(paired)
    selected = {m: next(iter(qualified(summary, m)), LOCKED) for m in MODELS}
    paired.to_csv(out / "cv_folds.csv", index=False)
    summary.to_csv(out / "cv_summary.csv", index=False)
    raw_test, baseline_test, yt, idt = load_rows(source, "test")
    clean_test, _ = build_clean(source, raw_test, baseline_test, idt)
    clean_test.to_csv(out / "clean_test.csv")
    yt.to_csv(out / "y_test.csv")
    idt.to_csv(out / "ids_test.csv")
    manifest = matrix_export(out, selected, lock, (clean, y, ids), (clean_test, yt, idt))
    write_json(out / "selected.json", manifest)
    verify_written_inputs(out)
    decision_table(source.columns, out, {m: configuration(m, name, lock)[1] for m, name in selected.items()})
    write_report(out, summary, manifest)


def run(out):
    from feature_documentation import audit, decision_table, dictionary

    out.mkdir(parents=True, exist_ok=True)
    source = read_source()
    lock = json.loads(LOCK_CONFIG.read_text(encoding="utf-8"))
    require(lock["seed"] == SEED and lock["models"] == {m: estimator(m).get_params() for m in MODELS},
            "Cấu hình mô hình khác bộ mốc đã khóa.")
    decision_table(source.columns, out)
    raw, baseline, y, ids = load_rows(source, "train")
    clean, base = build_clean(source, raw, baseline, ids)
    audit(raw, clean, out)
    dictionary(clean, out)
    clean.to_csv(out / "clean_train.csv")
    y.to_csv(out / "y_train.csv")
    ids.to_csv(out / "ids_train.csv")
    models_config = {m: configuration(m, LOCKED, lock)[1] for m in MODELS}
    for model, columns in models_config.items():
        expected = json.loads((OUTPUT / "pipeline" / "input_columns.json").read_text(encoding="utf-8"))
        require(columns == expected, "Thứ tự bộ mốc khác tài liệu.")
        pd.testing.assert_frame_equal(model_inputs(clean, columns, LOCKED), base[columns],
                                      check_dtype=False, check_exact=True)
        af_cols = [col for col in clean if col.startswith(("log_AF_", "zero_AF_"))]
        pd.testing.assert_frame_equal(clean[af_cols], base[af_cols], check_exact=True)
        existing = joblib.load(OUTPUT / "pipeline" /
                               f"preprocessor_{'scaled' if model == 'LogisticRegression' else 'unscaled'}.joblib")
        encoded = sparse.csr_matrix(existing.transform(base[columns]))
        archived = sparse.load_npz(OUTPUT / "pipeline" /
                                   f"{'scaled' if model == 'LogisticRegression' else 'unscaled'}_train.npz")
        require((encoded != archived).nnz == 0, "Bộ mốc hiện hành không khớp ma trận đã lưu.")
    folds, assignments = get_folds(base, y, ids)
    assignments.to_csv(out / "fold_assignments.csv")
    masks = subgroups(base)
    results = [reuse_metrics(folds, masks, y, lock).assign(source="reused")]

    def trial(name, model, scheme):
        results.append(evaluate(name, model, scheme, clean, y, masks, folds[scheme], lock)
                       .assign(source="new"))

    for scheme in folds:
        for model in MODELS:
            for name in (*NEW_ARMS, *(("af_zero",) if scheme == "gene" else ())):
                trial(name, model, scheme)
    singles = summarize(pair(pd.concat(results, ignore_index=True)))
    for name in plan_combinations(singles):
        for model in MODELS:
            for scheme in folds:
                trial(name, model, scheme)
    publish(out, source, lock, clean, y, ids, pd.concat(results, ignore_index=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=NEW)
    with threadpool_limits(limits=1):
        run(parser.parse_args().out)
