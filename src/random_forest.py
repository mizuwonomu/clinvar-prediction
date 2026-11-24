"""Contract-compliant Random Forest CV, selection, final fit, and one-time test score."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from paths import FEATURES, ROOT

NAME = "random_forest"
FOLD_COLUMNS = ["model", "config_id", "params", "round", "scheme", "fold", "n_train", "n_valid",
                "positive_valid", "train_ap", "valid_ap", "gap", "valid_logloss", "valid_roc_auc",
                "n_iter", "converged", "max_abs_coef", "oob_ap", "fit_seconds", "git_commit", "seed"]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "random_forest.yaml"
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if cfg["model"] != "RandomForest" or cfg["input_set"] != "selected":
        raise ValueError("Contract expects RandomForest and input_set=selected")
    return cfg


def build_pipeline(X, cfg, params):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                   ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ], remainder="drop")
    options = dict(cfg["fixed"])
    options.update(params)
    return Pipeline([("preprocess", prep), ("model", RandomForestClassifier(**options))])


def select_configuration(summary, baseline_id, cfg):
    """Choose simplest eligible candidate in the best-score tolerance band, else baseline."""
    valid = summary.loc[summary["folds"] == 5].copy()
    best = valid["valid_ap_mean"].max()
    best_row = valid.loc[valid.valid_ap_mean.idxmax()]
    tolerance = max(float(best_row.valid_ap_se), float(cfg["selection"]["tolerance_floor"]))
    eligible = valid.loc[valid.valid_ap_mean >= best - tolerance].copy()
    def simplicity_key(row):
        decoded = json.loads(row.params)
        ranks = []
        for key in ("n_estimators", "min_samples_leaf", "max_features", "max_depth"):
            order = cfg["simplicity_order"][key]
            def same(a, b): return a == b
            ranks.append(next((i for i, value in enumerate(order) if same(value, decoded[key])), len(order)))
        return tuple(ranks)
    chosen = min((row for _, row in eligible.iterrows()), key=simplicity_key)
    base = valid.loc[valid.config_id == baseline_id].iloc[0]
    if chosen.config_id != baseline_id:
        cand = folds_for(summary.attrs.get("folds_frame"), chosen.config_id)
        basf = folds_for(summary.attrs.get("folds_frame"), baseline_id)
        wins = int((cand.valid_ap.to_numpy() > basf.valid_ap.to_numpy()).sum())
        gain = float(chosen.valid_ap_mean - base.valid_ap_mean)
        if gain < float(cfg["selection"]["min_gain_over_baseline"]) or wins < int(cfg["selection"]["min_folds_up_vs_baseline"]):
            chosen = base
    return chosen


def folds_for(frame, config_id):
    return frame.loc[frame.config_id == config_id].sort_values("fold")


def _commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_train():
    # Required API supplies feature columns and its own typed, source_row-indexed table.
    from study_retained import read_model_inputs
    X = read_model_inputs(FEATURES, "RandomForest", "train")
    y = pd.read_csv(FEATURES / "y_train.csv", index_col="source_row")["CLASS"]
    assignments = pd.read_csv(FEATURES / "fold_assignments.csv", index_col="source_row")
    joined = y.to_frame().join(assignments[["variant_fold", "gene_fold"]], how="inner", validate="one_to_one")
    if not X.index.equals(joined.index) or len(X) != 52150:
        raise ValueError("Train inputs must align by source_row and contain 52,150 rows")
    if set(joined.variant_fold.unique()) != {1, 2, 3, 4, 5}:
        raise ValueError("variant_fold must be exactly 1..5")
    return X, joined.CLASS.astype(int), joined


def _id(params):
    return ",".join(f"{k}={json.dumps(v)}" for k, v in params.items())


def _evaluate(X, y, folds, cfg, configs):
    split = PredefinedSplit(folds.variant_fold.to_numpy() - 1)
    rows, probabilities = [], {}
    commit = _commit()
    for config_id, params, round_name in configs:
        pred = np.full(len(y), np.nan)
        for fold, (tr, va) in enumerate(split.split(), start=1):
            pipe = build_pipeline(X, cfg, params)
            start = time.perf_counter()
            pipe.fit(X.iloc[tr], y.iloc[tr])
            elapsed = time.perf_counter() - start
            pva = pipe.predict_proba(X.iloc[va])[:, 1]
            ptr = pipe.predict_proba(X.iloc[tr])[:, 1]
            model = pipe.named_steps["model"]
            aptr = average_precision_score(y.iloc[tr], ptr)
            apva = average_precision_score(y.iloc[va], pva)
            oob = model.oob_decision_function_
            oob_mask = np.isfinite(oob).all(axis=1)
            oob_ap = average_precision_score(y.iloc[tr].iloc[oob_mask], oob[oob_mask, 1]) if oob_mask.any() else np.nan
            rows.append(dict(model="RandomForest", config_id=config_id, params=json.dumps(params, sort_keys=True),
                round=round_name, scheme="variant", fold=fold, n_train=len(tr), n_valid=len(va),
                positive_valid=int(y.iloc[va].sum()), train_ap=aptr, valid_ap=apva, gap=aptr-apva,
                valid_logloss=log_loss(y.iloc[va], pva, labels=[0, 1]), valid_roc_auc=roc_auc_score(y.iloc[va], pva),
                n_iter=np.nan, converged=np.nan, max_abs_coef=np.nan, oob_ap=oob_ap,
                fit_seconds=elapsed, git_commit=commit, seed=cfg["fixed"]["random_state"]))
            pred[va] = pva
        probabilities[config_id] = pred
    frame = pd.DataFrame(rows, columns=FOLD_COLUMNS)
    return frame, probabilities


def _summary(frame):
    metrics = ["train_ap", "valid_ap", "gap", "valid_logloss", "valid_roc_auc", "oob_ap", "fit_seconds"]
    agg = frame.groupby(["model", "config_id", "params", "round", "scheme"], dropna=False)
    out = agg.size().rename("folds").to_frame()
    for metric in metrics:
        out[f"{metric}_mean"] = agg[metric].mean()
        out[f"{metric}_std"] = agg[metric].std(ddof=1)
        out[f"{metric}_se"] = out[f"{metric}_std"] / np.sqrt(out.folds)
    return out.reset_index()


def _make_configs(cfg):
    baseline = dict(cfg["baseline"])
    configs = [(_id(baseline), baseline, "baseline")]
    seen = {_id(baseline)}
    current = dict(baseline)
    for rnd in cfg["rounds"]:
        param = rnd["param"]
        for value in rnd["values"]:
            params = dict(current); params[param] = value
            if _id(params) not in seen:
                configs.append((_id(params), params, rnd["name"]))
                seen.add(_id(params))
    if len(configs) > 60:
        raise ValueError("Configuration budget exceeds contract limit of 60")
    return configs


def run_cv(output_dir=None, config_path=None):
    cfg = load_config(config_path)
    X, y, folds = load_train()
    baseline = dict(cfg["baseline"])
    baseline_id = _id(baseline)
    frame, probs = _evaluate(X, y, folds, cfg, [(baseline_id, baseline, "baseline")])
    current = dict(baseline)
    current_id = baseline_id
    for rnd in cfg["rounds"]:
        param = rnd["param"]
        candidates = []
        for value in rnd["values"]:
            params = dict(current); params[param] = value
            candidate_id = f"{_id(params)} [{rnd['name']}]"
            candidates.append((candidate_id, params, rnd["name"]))
        round_frame, round_probs = _evaluate(X, y, folds, cfg, candidates)
        frame = pd.concat([frame, round_frame], ignore_index=True)
        probs.update(round_probs)
        stats = _summary(round_frame)
        best_mean = float(stats.valid_ap_mean.max())
        best_row = stats.loc[stats.valid_ap_mean.idxmax()]
        tolerance = max(float(best_row.valid_ap_se), float(cfg["selection"]["tolerance_floor"]))
        eligible = stats.loc[stats.valid_ap_mean >= best_mean - tolerance]
        order = cfg["simplicity_order"][param]
        def rank_value(raw):
            value = json.loads(raw)[param]
            return next((i for i, option in enumerate(order) if option == value), len(order))
        if param == "n_estimators":
            # RF trees reduce variance; use the first point before the next gain flattens.
            means = stats.set_index(stats.params.map(lambda raw: json.loads(raw)[param])).valid_ap_mean.to_dict()
            chosen_value = order[-1]
            values = rnd["values"]
            for index in range(len(values) - 1):
                if means[values[index + 1]] - means[values[index]] < float(cfg["selection"]["tolerance_floor"]):
                    chosen_value = values[index]
                    break
            if chosen_value not in [json.loads(raw)[param] for raw in eligible.params]:
                chosen_value = json.loads(min(eligible.params, key=rank_value))[param]
        else:
            chosen_value = json.loads(min(eligible.params, key=rank_value))[param]
        current[param] = chosen_value
        current_id = next(cid for cid, candidate, _ in candidates if candidate[param] == chosen_value)
    out = Path(output_dir) if output_dir else ROOT / "results" / NAME
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "cv_folds.csv", index=False)
    summary = _summary(frame)
    summary.attrs["folds_frame"] = frame
    baseline_id = _id(cfg["baseline"])
    selected_id = current_id
    baseline_fold = folds_for(frame, baseline_id)
    chosen_fold = folds_for(frame, selected_id)
    gain = float(chosen_fold.valid_ap.mean() - baseline_fold.valid_ap.mean())
    wins = int((chosen_fold.valid_ap.to_numpy() > baseline_fold.valid_ap.to_numpy()).sum())
    if selected_id != baseline_id and (gain < float(cfg["selection"]["min_gain_over_baseline"])
                                       or wins < int(cfg["selection"]["min_folds_up_vs_baseline"])):
        selected_id = baseline_id
    selected = summary.loc[summary.config_id == selected_id].iloc[0]
    summary["folds_up_vs_baseline"] = summary.config_id.map(lambda c: int((folds_for(frame, c).valid_ap.to_numpy() > folds_for(frame, baseline_id).valid_ap.to_numpy()).sum()))
    summary.to_csv(out / "cv_summary.csv", index=False)
    return cfg, X, y, folds, frame, summary, selected, probs, out


def _oof_for_selected(X, y, folds, cfg, params):
    splitter = PredefinedSplit(folds.variant_fold.to_numpy() - 1)
    oof = np.full(len(y), np.nan)
    for tr, va in splitter.split():
        pipe = build_pipeline(X, cfg, params).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, oof)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best = np.flatnonzero(f1 == np.nanmax(f1))
    return float(thresholds[best[0]]), oof


def train_and_score(output_dir=None, config_path=None):
    cfg, X, y, folds, frame, summary, selected, probs, out = run_cv(output_dir, config_path)
    params = json.loads(selected.params)
    threshold, oof = _oof_for_selected(X, y, folds, cfg, params)
    cfg["final"] = {"params": params, "threshold": threshold, "selection_config_id": selected.config_id,
                    "cv_ap": float(selected.valid_ap_mean), "cv_ap_se": float(selected.valid_ap_se)}
    config_file = Path(config_path) if config_path else ROOT / "configs" / "random_forest.yaml"
    config_file.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    model = build_pipeline(X, cfg, params).fit(X, y)
    checkpoint = ROOT / "checkpoints" / NAME
    checkpoint.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, checkpoint / "rf_model.joblib", compress=3)

    # First and only read of test labels/features; this happens after threshold is fixed from OOF.
    from study_retained import read_model_inputs
    Xt = read_model_inputs(FEATURES, "RandomForest", "test")
    yt = pd.read_csv(FEATURES / "y_test.csv", index_col="source_row")["CLASS"].astype(int)
    if not Xt.index.equals(yt.index):
        raise ValueError("Test X and y do not align by source_row")
    p = model.predict_proba(Xt)[:, 1]
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    metrics = {"model": "RandomForest", "config_id": selected.config_id, "params": json.dumps(params, sort_keys=True),
        "threshold": threshold, "n_test": len(yt), "accuracy": accuracy_score(yt, pred),
        "precision": precision_score(yt, pred, zero_division=0), "recall": recall_score(yt, pred, zero_division=0),
        "f1": f1_score(yt, pred, zero_division=0), "ap": average_precision_score(yt, p),
        "roc_auc": roc_auc_score(yt, p), "log_loss": log_loss(yt, p), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "git_commit": _commit(), "seed": cfg["fixed"]["random_state"]}
    pd.DataFrame([metrics]).to_csv(out / "test_metrics.csv", index=False)
    _generate_docs(cfg, frame, summary, selected, metrics, p, yt, threshold)
    print(f"XONG! Mô hình đạt AP CV là {selected.valid_ap_mean:.6f}. Hãy mở file document/AGENT_HANDOFF.txt và document/BAO_CAO_XAY_DUNG_MO_HINH.docx để xem kết quả.")


def _generate_docs(cfg, folds_frame, summary, selected, test, test_prob, y_test, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay
    from docx import Document
    from docx.shared import Inches
    from sklearn.metrics import ConfusionMatrixDisplay
    docdir = ROOT / "document"; figdir = docdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    params = json.loads(selected.params)
    chosen_folds = folds_for(folds_frame, selected.config_id)
    baseline_id = _id(cfg["baseline"])
    base_folds = folds_for(folds_frame, baseline_id)
    curves = []
    for rnd in cfg["rounds"]:
        param = rnd["param"]
        subset = folds_frame.loc[folds_frame["round"] == rnd["name"]]
        grouped = subset.groupby("params").agg(train=("train_ap", "mean"), valid=("valid_ap", "mean"),
                                                    std=("valid_ap", "std"))
        grouped["value"] = grouped.index.map(lambda s: json.loads(s)[param])
        grouped = grouped.sort_values("value", key=lambda x: x.map(lambda v: (v is None, str(v))))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(len(grouped)), grouped.train, "o-", label="Train AP")
        ax.plot(range(len(grouped)), grouped.valid, "o-", label="Validation AP")
        ax.fill_between(range(len(grouped)), grouped.valid-grouped["std"], grouped.valid+grouped["std"], alpha=.15)
        ax.set_xticks(range(len(grouped)), [str(v) for v in grouped.value], rotation=25)
        ax.set(title=f"Validation curve: {param}", xlabel=param, ylabel="Average Precision"); ax.grid(alpha=.2); ax.legend()
        fig.tight_layout(); path = figdir / f"validation_{param}.png"; fig.savefig(path, dpi=300); plt.close(fig)
        curves.append((path, param))
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, test_prob >= threshold), display_labels=["CLASS 0", "CLASS 1"]).plot(ax=ax, colorbar=False)
    fig.tight_layout(); cm_path = figdir / "confusion_matrix.png"; fig.savefig(cm_path, dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_test, test_prob, ax=ax, name="Random Forest")
    fig.tight_layout(); pr_path = figdir / "precision_recall_curve.png"; fig.savefig(pr_path, dpi=300); plt.close(fig)
    feature_names = json.loads((FEATURES / "RandomForest" / "input_columns.json").read_text(encoding="utf-8"))
    cv_table = pd.DataFrame({"fold": range(1, 6), "Baseline AP": base_folds.valid_ap.values,
                             "Final AP": chosen_folds.valid_ap.values})
    tn, fp, fn, tp = test["tn"], test["fp"], test["fn"], test["tp"]
    searched = folds_frame.config_id.nunique()
    (docdir / "AGENT_HANDOFF.txt").write_text(
        "BÀN GIAO RANDOM FOREST CLINVAR\n\n"
        "Mã nguồn: src/random_forest.py; mô hình fit cuối: checkpoints/random_forest/rf_model.joblib\n"
        "Thiết kế: PredefinedSplit(variant_fold - 1) với 5 variant_fold từ fold_assignments.csv. Train data: "
        f"{len(folds_frame) and int(chosen_folds.n_train.iloc[0] + chosen_folds.n_valid.iloc[0]):,} dòng, ghép qua source_row.\n"
        f"Đặc trưng: {', '.join(feature_names)}\n"
        f"Tìm kiếm: {searched} cấu hình, AP làm tiêu chí. Cấu hình tốt nhất: {json.dumps(params, ensure_ascii=False)}. AP tốt nhất CV={selected.valid_ap_mean:.6f}.\n"
        f"Hold-out test tại ngưỡng tối ưu {threshold:.6f}: Accuracy={test['accuracy']:.6f}; Precision={test['precision']:.6f}; "
        f"Recall={test['recall']:.6f}; F1={test['f1']:.6f}; AP={test['ap']:.6f}; ROC-AUC={test['roc_auc']:.6f}. "
        f"Ma trận nhầm lẫn (TN, FP; FN, TP)=({tn}, {fp}; {fn}, {tp}).\n"
        "Diễn giải: Cấu hình được chọn tuân thủ luật dung sai 0.002 và thay baseline chỉ khi tăng AP ít nhất 0.002, thắng ít nhất 4/5 fold.\n",
        encoding="utf-8")
    (docdir / "GIAI_THICH_CHO_NGUOI_HOC.txt").write_text(
        "RANDOM FOREST CLINVAR: GIẢI THÍCH CHO NGƯỜI HỌC\n\n"
        "1. Vì sao dùng PredefinedSplit? Fold variant_fold đã được team cố định. PredefinedSplit(variant_fold - 1) tái sử dụng chính xác các nhóm này, "
        "giúp so sánh mô hình công bằng. train_test_split tạo một phép chia mới, còn StratifiedKFold tự chia lại; cả hai có thể làm kết quả lệch khỏi contract.\n\n"
        "2. Tránh rò rỉ: Mỗi fold chỉ fit imputer, one-hot encoder và Random Forest trên phần train. Fold validation chỉ dùng để chấm điểm. Đặc biệt, "
        "ngưỡng phân lớp được chọn từ dự đoán OOF trên train; nhãn test chỉ được mở ở lần đánh giá cuối.\n\n"
        "3. AP và khoảng cách train-valid: Average Precision đánh giá khả năng xếp hạng lớp dương qua nhiều ngưỡng. AP train cao hơn AP validation tạo ra gap; "
        "gap lớn có thể là dấu hiệu cây ghi nhớ chi tiết của tập train. Cây rất sâu thường đạt AP train gần 1, nên cần nhìn đồng thời AP validation, OOB AP và độ ổn định giữa folds.\n\n"
        "4. Cắt tỉa độ phức tạp: max_depth giới hạn số tầng của cây. Giảm max_depth làm quy tắc quyết định ngắn hơn. min_samples_leaf yêu cầu mỗi lá có đủ mẫu; "
        "tăng giá trị này ngăn lá chỉ đại diện cho vài biến thể. Nếu AP train cao nhưng AP validation thấp và gap rộng, tăng min_samples_leaf hoặc giảm max_depth "
        "thường giúp giảm overfitting. Cần xác nhận bằng đường cong validation, không áp dụng máy móc.\n\n"
        "5. Baseline và chọn cấu hình: chỉ chấp nhận cấu hình mới khi AP trung bình tăng tối thiểu 0.002, thắng baseline ít nhất 4/5 fold; trong các ứng viên gần điểm tốt nhất "
        "theo dung sai SE, ưu tiên cấu hình đơn giản hơn. Test chỉ dùng một lần sau khi khóa cấu hình và threshold.\n", encoding="utf-8")
    doc = Document(); doc.add_heading("Báo cáo xây dựng mô hình Random Forest", 0)
    doc.add_paragraph(f"AP CV được chọn: {selected.valid_ap_mean:.4f} (SE {selected.valid_ap_se:.4f}); cấu hình {json.dumps(params)}.")
    doc.add_heading("Thiết kế và đặc trưng", level=1)
    doc.add_paragraph("Dùng PredefinedSplit từ variant_fold, với tiền xử lý bên trong Pipeline. Đặc trưng: " + ", ".join(feature_names) + ".")
    doc.add_heading("Baseline và mô hình cuối theo fold", level=1)
    table = doc.add_table(rows=1, cols=3); table.style = "Light Shading Accent 1"
    for cell, value in zip(table.rows[0].cells, cv_table.columns): cell.text = value
    for _, row in cv_table.iterrows():
        cells = table.add_row().cells
        for cell, value in zip(cells, row): cell.text = f"{value:.5f}" if isinstance(value, (float, np.floating)) else str(value)
    doc.add_heading("Biểu đồ chẩn đoán", level=1)
    for path, param in curves:
        doc.add_paragraph(f"Validation curve cho {param}: AP train/validation trung bình, dải ± độ lệch chuẩn qua folds.")
        doc.add_picture(str(path), width=Inches(6.0))
    doc.add_paragraph("Confusion Matrix tại threshold chọn từ OOF train."); doc.add_picture(str(cm_path), width=Inches(4.8))
    doc.add_paragraph("Precision–Recall curve trên test."); doc.add_picture(str(pr_path), width=Inches(5.8))
    doc.add_heading("Kết quả test một lần", level=1)
    doc.add_paragraph(f"Threshold={threshold:.5f}; Accuracy={test['accuracy']:.4f}; Precision={test['precision']:.4f}; Recall={test['recall']:.4f}; "
                      f"F1={test['f1']:.4f}; AP={test['ap']:.4f}; ROC-AUC={test['roc_auc']:.4f}. Confusion matrix (TN, FP; FN, TP)=({tn}, {fp}; {fn}, {tp}).")
    doc.save(docdir / "BAO_CAO_XAY_DUNG_MO_HINH.docx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    train_and_score(args.output_dir, args.config)


if __name__ == "__main__":
    main()
