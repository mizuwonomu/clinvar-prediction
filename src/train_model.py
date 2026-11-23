"""Huấn luyện và đánh giá Random Forest trên dữ liệu ClinVar Conflicting."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             ConfusionMatrixDisplay, RocCurveDisplay,
                             PrecisionRecallDisplay)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, train_test_split,
                                     learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "clinvar/source/clinvar_conflicting.csv"
OUT = ROOT / "document"
FIG = OUT / "figures"
NUMERIC = ["AF_ESP", "AF_EXAC", "AF_TGP", "CADD_PHRED"]
CATEGORICAL = ["CLNVC", "Consequence", "IMPACT", "BIOTYPE"]
FEATURES = NUMERIC + CATEGORICAL


def main() -> None:
    """Tách stratified 80/20, tìm siêu tham số bằng CV trên train và báo cáo test."""
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, na_values=["nan", "NaN", "", "."])
    X = df[FEATURES].copy()
    for col in NUMERIC:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = pd.to_numeric(df["CLASS"], errors="raise").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC), ("categorical", categorical_pipe, CATEGORICAL)
    ])
    pipe = Pipeline([
        ("prep", prep),
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced_subsample",
                                           n_jobs=-1, bootstrap=True)),
    ])
    param_distributions = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [8, 12, 18, None],
        "model__min_samples_leaf": [2, 5, 10, 20],
        "model__min_samples_split": [5, 10, 20],
        "model__max_features": ["sqrt", 0.5],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(pipe, param_distributions, n_iter=12,
                                scoring="average_precision", cv=cv,
                                random_state=42, n_jobs=-1, refit=True, verbose=1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    pred_train = (p_train >= 0.5).astype(int)
    pred_test = (p_test >= 0.5).astype(int)

    def metrics(y_true, pred, prob):
        return {"accuracy": float(accuracy_score(y_true, pred)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "average_precision": float(average_precision_score(y_true, prob)),
                "roc_auc": float(roc_auc_score(y_true, prob))}

    # Đường cong học được tính trong train bằng CV, không truy cập tập test.
    sizes, train_scores, cv_scores = learning_curve(
        model, X_train, y_train, cv=cv, scoring="average_precision",
        train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1, shuffle=True, random_state=42,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_scores.mean(axis=1), "o-", label="Train AP")
    ax.fill_between(sizes, train_scores.mean(axis=1)-train_scores.std(axis=1),
                    train_scores.mean(axis=1)+train_scores.std(axis=1), alpha=.15)
    ax.plot(sizes, cv_scores.mean(axis=1), "o-", label="Validation AP (5-fold)")
    ax.fill_between(sizes, cv_scores.mean(axis=1)-cv_scores.std(axis=1),
                    cv_scores.mean(axis=1)+cv_scores.std(axis=1), alpha=.15)
    ax.set(title="Learning curve (train only)", xlabel="Số mẫu huấn luyện", ylabel="Average Precision")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIG/"learning_curve.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC curve - hold-out test"); fig.tight_layout(); fig.savefig(FIG/"roc_curve.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, p_test, ax=ax, name="Random Forest")
    ax.set_title("Precision-Recall - hold-out test"); fig.tight_layout(); fig.savefig(FIG/"precision_recall_curve.png", dpi=300); plt.close(fig)
    cm = confusion_matrix(y_test, pred_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Không mâu thuẫn", "Mâu thuẫn"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion matrix - hold-out test"); fig.tight_layout(); fig.savefig(FIG/"confusion_matrix.png", dpi=300); plt.close(fig)

    names = model.named_steps["prep"].get_feature_names_out()
    importance = model.named_steps["model"].feature_importances_
    imp = pd.Series(importance, index=names).sort_values(ascending=False).head(20).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    imp.plot.barh(ax=ax, color="#3478a5"); ax.set(title="Top 20 feature importances", xlabel="Importance", ylabel="")
    fig.tight_layout(); fig.savefig(FIG/"feature_importance.png", dpi=300); plt.close(fig)
    results = {
        "source": str(DATA.relative_to(ROOT)), "target": "CLASS",
        "features": FEATURES, "excluded_columns": [c for c in df.columns if c not in FEATURES+["CLASS"]],
        "split": {"method": "train_test_split(stratify=y)", "test_size": 0.2, "random_state": 42,
                  "train_rows": len(y_train), "test_rows": len(y_test),
                  "train_positive_rate": float(y_train.mean()), "test_positive_rate": float(y_test.mean())},
        "cv": {"method": "StratifiedKFold", "folds": 5, "best_average_precision": float(search.best_score_)},
        "best_params": search.best_params_, "train_metrics": metrics(y_train, pred_train, p_train),
        "test_metrics": metrics(y_test, pred_test, p_test), "confusion_matrix": cm.tolist(),
        "threshold": 0.5,
        "learning_curve": {"train_sizes": sizes.tolist(), "train_ap_mean": train_scores.mean(axis=1).tolist(),
                           "validation_ap_mean": cv_scores.mean(axis=1).tolist()},
        "top_features": [{"name": str(k), "importance": float(v)} for k,v in imp.sort_values(ascending=False).items()],
    }
    (OUT/"results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    import joblib
    joblib.dump(model, OUT/"random_forest_model.joblib")
    # Sinh bàn giao, tài liệu học viên và báo cáo Word từ chính kết quả vừa lưu.
    from generate_documents import main as generate_documents
    generate_documents()
    print(f"Hoàn tất huấn luyện ClinVar. Test AP = {results['test_metrics']['average_precision']:.4f}. Mở document/GIAI_THICH_CHO_NGUOI_HOC.txt để đọc đầu tiên.")


if __name__ == "__main__":
    main()
