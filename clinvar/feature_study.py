"""Thí nghiệm đặc trưng ClinVar với các lệnh audit, cv và holdout."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from consequence_encoder import ConsequenceMultiHot, _terms
from preprocess import (
    AF, CAT, FEATURES, MISSING, NUM, SEED, load_split, make_preprocessor,
    require, write_json,
)


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "source" / "clinvar_conflicting.csv"
OUTPUT = ROOT / "feature_study"
PROTEIN = ("SIFT", "PolyPhen", "BLOSUM62")
SCORES = ("LoFtool", *PROTEIN)
POSITIONS = ("CDS_position", "Protein_position")
COUNTS = ("EXON", "INTRON")
MOTIFS = ("MOTIF_NAME", "MOTIF_POS", "HIGH_INF_POS", "MOTIF_SCORE_CHANGE")
ALLELE = ("ref_length", "alt_length", "alt_minus_ref", "multiple_alt")
TRANSCRIPT = tuple(
    [f"{col}_{part}" for col in COUNTS for part in ("index", "total", "fraction")]
    + [f"{col}_{part}" for col in POSITIONS for part in ("start", "span")]
)
MODELS = ("LogisticRegression", "RandomForest")
BASE = "baseline_onehot"
THRESHOLD = 0.5
GROUPS = {
    "lof_flags": ("LoFtool",),
    "lof_values": ("LoFtool",),
    "protein_flags": PROTEIN,
    "protein_values": PROTEIN,
    "protein_recovered": PROTEIN,
    "all_flags": SCORES,
    "all_values": SCORES,
}
ENCODINGS = {
    "lof": ("lof_flags", "lof_values"),
    "protein": ("protein_flags", "protein_values", "protein_recovered"),
    "allele": ("allele",),
    "transcript": ("transcript",),
    "cadd": ("cadd_raw",),
    "mc": ("mc",),
}
# Canonical amino-acid pairs with a zero score in the NCBI BLOSUM62 matrix.
# https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM62
ZERO_BLOSUM = {
    "AC", "AG", "AT", "AV", "NR", "ER", "HR", "NQ", "EN", "GN", "KN", "NT",
    "DQ", "DS", "HQ", "MQ", "QS", "EH", "ES", "GS", "FI", "FL", "FM",
    "KS", "TV",
}


class GeneFrequency(TransformerMixin, BaseEstimator):
    """Tần suất gene trong dữ liệu fit, với 0 cho gene chưa gặp."""

    def fit(self, X, y=None):
        self.frequency_ = X.iloc[:, 0].value_counts(normalize=True)
        return self

    def transform(self, X):
        return X.iloc[:, 0].map(self.frequency_).fillna(0).to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["symbol_frequency"], dtype=object)


def read_source():
    ref = json.loads((ROOT / "data" / "report.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    require(digest == ref["input_sha256"], "CSV gốc không khớp bộ train/test.")
    raw = pd.read_csv(INPUT, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    raw.index = pd.RangeIndex(1, len(raw) + 1, name="source_row")
    return raw


def load_rows(raw, split):
    X, y, ids = load_split(ROOT / "data", split)
    files = [f"{prefix}_{split}.csv" for prefix in ("X", "y", "ids")]
    files.append(f"source_rows_{split}.npy")
    for file in files:
        require((ROOT / "data" / file).read_bytes() == (ROOT / "data_multihot" / file).read_bytes(),
                f"Hai baseline lệch ở {file}.")
    rows = np.load(ROOT / "data" / files[-1], allow_pickle=False)
    require(np.array_equal(rows, X.index), f"Mã dòng {split} không khớp.")
    raw = raw.loc[rows].apply(lambda col: col.str.strip())
    missing = raw.apply(lambda col: col.str.upper().isin(MISSING))
    missing["Allele"] &= raw.Allele.ne("-")  # VEP dùng dấu này cho alen bị xóa.
    raw = raw.mask(missing)
    require(raw.CLASS.astype("int8").equals(y), f"Nhãn gốc {split} không khớp.")
    for col in ("CHROM", "POS", "REF", "ALT"):
        require(raw[col].eq(ids[col]).all(), f"Khóa gốc {col} không khớp.")
    for col in CAT:
        require(raw[col].fillna("").eq(X[col].fillna("")).all(), f"Cột gốc {col} không khớp.")
    for col in NUM:
        parsed = number(raw[col])
        valid = parsed.ge(0) & (parsed.le(1) if col in AF else True)
        pd.testing.assert_series_equal(parsed.where(valid), X[col], check_exact=True)
    return raw, X, y, ids


def number(values):
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    return numeric.where(np.isfinite(numeric))


def position_features(raw):
    derived = pd.DataFrame(index=raw.index)
    valid = {}
    for col in COUNTS:
        tokens = raw[col].str.extract(r"^([1-9]\d*)/([1-9]\d*)$").astype(float)
        first, total = tokens[0], tokens[1]
        good = first.le(total)
        derived[f"{col}_index"] = first.where(good)
        derived[f"{col}_total"] = total.where(good)
        derived[f"{col}_fraction"] = (first / total).where(good)
        valid[col] = good
    for col in POSITIONS:
        tokens = raw[col].str.extract(r"^([1-9]\d*)(?:-([1-9]\d*))?$").astype(float)
        first, last = tokens[0], tokens[1].fillna(tokens[0])
        good = last.ge(first)
        derived[f"{col}_start"] = first.where(good)
        derived[f"{col}_span"] = (last - first + 1).where(good)
        valid[col] = good
    return derived, valid


def recover_blosum(raw, observed):
    pair = raw.Amino_acids.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]/[ACDEFGHIKLMNPQRSTVWY]", na=False)
    pair_codes = raw.Amino_acids[pair].map(lambda value: "".join(sorted(value.split("/"))))
    score_is_zero = pair_codes.isin(ZERO_BLOSUM).reindex(raw.index, fill_value=False)
    missense = raw.Consequence.map(lambda value: "missense_variant" in _terms(value))
    return observed.isna() & score_is_zero & missense


def prepare_features(raw, baseline, ids):
    features = baseline.copy()
    for col in ("LoFtool", "BLOSUM62", "CADD_RAW"):
        features[col] = number(raw[col])
    for col in ("SIFT", "PolyPhen"):
        features[col] = raw[col].astype(object).where(raw[col].notna(), np.nan)
    for col in SCORES:
        features[f"has_{col}"] = features[col].notna().astype("int8")
    for col in AF:
        # AF=0 trong snapshot có thể gồm cả giá trị mà nguồn đã điền.
        features[f"zero_{col}"] = features[col].eq(0).astype("int8")
        features[f"log_{col}"] = np.log1p(features[col] * 1_000_000) / np.log(10)
    features["has_CADD_RAW"] = features.CADD_RAW.notna().astype("int8")
    features["BLOSUM62_recovered"] = features.BLOSUM62.mask(recover_blosum(raw, features.BLOSUM62), 0)
    features["SYMBOL"] = raw.SYMBOL.fillna("__MISSING__")
    single = ~ids.ALT.str.contains(",", regex=False)
    features["ref_length"] = ids.REF.str.len().astype(float)
    features["alt_length"] = ids.ALT.str.len().astype(float).where(single)
    features["alt_minus_ref"] = features.alt_length - features.ref_length
    features["multiple_alt"] = (~single).astype("int8")
    positions, valid = position_features(raw)
    features["transcript_region"] = np.select(
        [valid["EXON"] & valid["INTRON"], valid["EXON"], valid["INTRON"]],
        ["exon_and_intron", "exon", "intron"], default="unresolved",
    )
    features["MC"] = raw.MC.map(
        lambda v: "&".join(part.split("|", 1)[-1] for part in v.split(",")) if pd.notna(v) else np.nan
    )
    return features.join(positions)


def preprocessor(model, arms, encoding="onehot"):
    pipe = make_preprocessor(model == "LogisticRegression", encoding)
    columns = pipe.named_steps["columns"]
    transformers = columns.transformers
    numeric = list(NUM)
    categorical = [col for col in CAT if encoding == "onehot" or col != "Consequence"]
    flags = []
    for arm in arms:
        if arm in GROUPS:
            flags.extend(f"has_{col}" for col in GROUPS[arm])
            if arm.endswith(("_values", "_recovered")):
                numeric.extend("BLOSUM62_recovered" if arm == "protein_recovered" else col
                               for col in GROUPS[arm] if col in {"LoFtool", "BLOSUM62"})
                categorical.extend(col for col in GROUPS[arm] if col in {"SIFT", "PolyPhen"})
        elif arm == "allele":
            numeric.extend(ALLELE[:-1])
            flags.append(ALLELE[-1])
        elif arm == "transcript":
            numeric.extend(TRANSCRIPT)
            categorical.append("transcript_region")
        elif arm == "cadd_raw":
            numeric[numeric.index("CADD_PHRED")] = "CADD_RAW"
        elif arm == "mc":
            transformers.append(("mc", ConsequenceMultiHot(), ["MC"]))
        elif arm == "symbol_frequency":
            transformers.append(("gene", GeneFrequency(), ["SYMBOL"]))
        elif arm == "af_log":
            numeric = [f"log_{col}" if col in AF else col for col in numeric]
        elif arm == "af_zero":
            flags.extend(f"zero_{col}" for col in AF)
        elif arm == "cadd_present":
            flags.append("has_CADD_RAW")
        else:
            raise ValueError(f"Cấu hình không hợp lệ: {arm}")
    transformers[0] = ("numeric", transformers[0][1], numeric)
    transformers[1] = ("categorical", transformers[1][1], categorical)
    if flags:
        transformers.append(("flags", "passthrough", flags))
    names = list(dict.fromkeys(col for _, _, cols in transformers for col in cols))
    return pipe, names


def estimator(model):
    if model == "LogisticRegression":
        return LogisticRegression(max_iter=3000, random_state=SEED)
    return RandomForestClassifier(
        n_estimators=80, max_depth=16, min_samples_leaf=5,
        max_features=6, random_state=SEED, n_jobs=6,
    )


def subgroups(X):
    missense = X.Consequence.map(lambda value: "missense_variant" in _terms(value))
    has_protein = X[[f"has_{col}" for col in PROTEIN]].any(axis=1)
    return {
        "all": np.ones(len(X), dtype=bool),
        "missense": missense.to_numpy(),
        "non_missense": (~missense).to_numpy(),
        "protein_present": has_protein.to_numpy(),
        "protein_missing": (~has_protein).to_numpy(),
        "missense_protein_present": (missense & has_protein).to_numpy(),
        "missense_protein_missing": (missense & ~has_protein).to_numpy(),
    }


def scores(y, probabilities, masks, threshold=THRESHOLD):
    for subgroup, mask in masks.items():
        true, prob = y[mask], probabilities[mask]
        n_positive = int(true.sum())
        ap = average_precision_score(true, prob) if 0 < n_positive < len(true) else np.nan
        precision, recall, f1, _ = precision_recall_fscore_support(
            true, prob >= threshold, average="binary", zero_division=0,
        )
        yield dict(subgroup=subgroup, n=len(true), positive=n_positive, ap=ap,
                   precision=precision, recall=recall, f1=f1)


def fit_config(model, arms, encoding, X, y):
    pipe, columns = preprocessor(model, arms, encoding)
    pipe.steps.append(("model", estimator(model)))
    pipe.fit(X[columns], y)
    return pipe, columns


def run_config(name, config, X, y, folds, oof=None):
    encoding, arms = config
    records = []
    print(name, flush=True)
    for model, (fold, (train, valid)) in product(MODELS, enumerate(folds, start=1)):
        fit, columns = fit_config(model, arms, encoding, X.iloc[train], y.iloc[train])
        prob = fit.predict_proba(X.iloc[valid][columns])[:, 1]
        if oof is not None:
            oof[model][valid] = prob
        for metrics in scores(y.iloc[valid], prob, subgroups(X.iloc[valid])):
            records.append(dict(config=name, model=model, fold=fold, **metrics))
    return pd.DataFrame(records)


def comparisons(results, reference=BASE):
    keys = ["model", "fold", "subgroup"]
    base = results.loc[results.config.eq(reference), keys + ["ap", "recall"]]
    paired = results.merge(base, on=keys, suffixes=("", "_baseline"), validate="many_to_one")
    paired["delta_ap"] = paired.ap - paired.ap_baseline
    paired["delta_recall"] = paired.recall - paired.recall_baseline
    return paired


def summarize(paired):
    summary = paired.groupby(["config", "model", "subgroup"], sort=False).agg(
        ap=("ap", "mean"), ap_sd=("ap", "std"), recall=("recall", "mean"),
        precision=("precision", "mean"), f1=("f1", "mean"),
        delta_ap=("delta_ap", "mean"), delta_ap_sd=("delta_ap", "std"),
        folds_up=("delta_ap", lambda d: d.gt(0).sum()), folds_valid=("delta_ap", "count"),
        min_positive=("positive", "min"),
    ).reset_index()
    return summary


def qualified(summary, subgroup, cutoff):
    frame = summary[summary.subgroup.eq(subgroup)]
    return frame.loc[
        frame.delta_ap.ge(cutoff) & frame.folds_up.ge(4) & frame.folds_valid.eq(5)
        & frame.delta_ap.gt(frame.delta_ap_sd / np.sqrt(5)) & frame.min_positive.ge(50)
    ]


def gene_check(X, y, ids, standard_folds, baseline, selected, out):
    genes = X.SYMBOL.where(X.SYMBOL.ne("__MISSING__"), "unknown:" + ids.variant_id)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    gene_folds = list(splitter.split(X, y, genes))
    coverage = []
    for scheme, folds in (("variant", standard_folds), ("gene", gene_folds)):
        for fold, (train, valid) in enumerate(folds, start=1):
            overlap = set(genes.iloc[train]) & set(genes.iloc[valid])
            if scheme == "gene":
                require(not overlap, "Gene xuất hiện ở cả phần fit và validation.")
                require(not set(ids.variant_id.iloc[train]) & set(ids.variant_id.iloc[valid]),
                        "Biến thể xuất hiện ở cả phần fit và validation theo gene.")
            known = X.SYMBOL.iloc[valid].ne("__MISSING__")
            require(y.iloc[train].nunique() == y.iloc[valid].nunique() == 2, "Fold theo gene mất lớp.")
            coverage.append(dict(scheme=scheme, fold=fold, n=len(valid), positive=int(y.iloc[valid].sum()),
                                 genes_fit=X.SYMBOL.iloc[train].replace("__MISSING__", np.nan).nunique(),
                                 genes_validation=X.SYMBOL.iloc[valid][known].nunique(),
                                 genes_shared=len(overlap), missing_gene=int((~known).sum())))
    pd.DataFrame(coverage).to_csv(out / "gene_fold_coverage.csv", index=False)
    result = []
    for scheme, folds in (("variant", standard_folds), ("gene", gene_folds)):
        base = baseline if scheme == "variant" else run_config(BASE, ("onehot", ()), X, y, folds)
        symbol = run_config("symbol_frequency", ("onehot", ("symbol_frequency",)), X, y, folds)
        trials = [base, symbol]
        if scheme == "gene":
            trials.append(run_config("selected", selected, X, y, folds))
        result.append(comparisons(pd.concat(trials)).assign(scheme=scheme))
    pd.concat(result).to_csv(out / "gene_fold_metrics.csv", index=False)


def operating_thresholds(X, y, folds, selected):
    result = {}
    for name, config in {BASE: ("onehot", ()), "selected": selected}.items():
        oof = {model: np.full(len(y), np.nan) for model in MODELS}
        run_config(f"{name} (ngưỡng OOF)", config, X, y, folds, oof=oof)
        thresholds_by_model = {}
        for model, probabilities in oof.items():
            require(np.isfinite(probabilities).all(), "Xác suất OOF thiếu hoặc không hữu hạn.")
            precision, recall, thresholds = precision_recall_curve(y, probabilities)
            total = precision[:-1] + recall[:-1]
            f1 = np.divide(2 * precision[:-1] * recall[:-1], total, out=np.zeros_like(total), where=total > 0)
            index = len(f1) - 1 - np.argmax(f1[::-1])
            thresholds_by_model[model] = float(thresholds[index])
        result[name] = thresholds_by_model
    return result


def cv(raw, X, y, ids, out):
    if (out / "holdout_metrics.csv").exists():
        raise ValueError("Holdout đã có kết quả. Giữ cấu hình và báo cáo đã khóa.")
    X = prepare_features(raw, X, ids)
    groups = ids["variant_id"]
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups))
    assignments = pd.Series(0, index=X.index, name="fold", dtype=int)
    for fold, (train, valid) in enumerate(folds, start=1):
        require(not set(groups.iloc[train]) & set(groups.iloc[valid]), "Trùng biến thể giữa các phần fold.")
        require(y.iloc[train].nunique() == y.iloc[valid].nunique() == 2, "Fold mất lớp.")
        require(assignments.iloc[valid].eq(0).all(), "Dòng xuất hiện nhiều lần trong validation.")
        assignments.iloc[valid] = fold
    require(assignments.gt(0).all(), "Có dòng không thuộc validation.")
    pd.concat([assignments, y], axis=1).to_csv(out / "fold_assignments.csv")
    coverage = raw[["DISTANCE", *MOTIFS]].notna().join(assignments).groupby("fold").sum()
    coverage.insert(0, "n", assignments.value_counts().sort_index())
    coverage.to_csv(out / "sparse_coverage_by_fold.csv")
    configs = {BASE: ("onehot", ()), "baseline_multihot": ("multihot", ())}
    for arm in (*GROUPS, "allele", "transcript", "cadd_raw", "mc"):
        configs[arm] = ("onehot", (arm,))
    trials = [run_config(name, config, X, y, folds) for name, config in configs.items()]
    first_summary = summarize(comparisons(pd.concat(trials, ignore_index=True)))
    global_qualified = qualified(first_summary, "all", 0.002)
    missense_qualified = qualified(first_summary, "missense", 0.005)
    promising = pd.concat([global_qualified, missense_qualified])
    components = []
    for arms in ENCODINGS.values():
        ranking = promising[promising.config.isin(arms)].sort_values(
            ["delta_ap", "config"], ascending=[False, True],
        )
        if not ranking.empty:
            components.append(ranking.iloc[0].config)
    if len(components) > 1:
        combos = {"combined": tuple(components)}
        combos.update({f"combined_without_{arm}": tuple(c for c in components if c != arm)
                       for arm in components})
        for name, arms in combos.items():
            configs[name] = ("onehot", arms)
            existing = next((trial for trial in trials
                             if configs[trial.config.iloc[0]] == configs[name]), None)
            trials.append(run_config(name, configs[name], X, y, folds) if existing is None
                          else existing.assign(config=name))
    paired = comparisons(pd.concat(trials, ignore_index=True))
    summary = summarize(paired)
    paired.to_csv(out / "cv_fold_metrics.csv", index=False)
    summary.to_csv(out / "cv_summary.csv", index=False)
    values_comparisons = []
    for arm in ("lof", "protein", "all"):
        values_comparisons.append((f"{arm}_values", f"{arm}_flags"))
    values_comparisons.extend([("protein_recovered", "protein_flags"), ("protein_recovered", "protein_values")])
    if "combined" in configs:
        values_comparisons.extend(("combined", f"combined_without_{arm}") for arm in components)
    pd.concat([
        comparisons(paired.drop(columns=["ap_baseline", "recall_baseline", "delta_ap", "delta_recall"])
                    .query("config in [@variant, @reference]"), reference).query("config == @variant")
        .assign(reference=reference)
        for variant, reference in values_comparisons
    ]).to_csv(out / "cv_ablation.csv", index=False)
    # Chọn một bộ dùng chung theo AP toàn tập, trung bình đều hai mô hình.
    eligible = qualified(summary, "all", 0.002).query("config != 'baseline_multihot'")
    overall = summary.query("subgroup == 'all'")
    ranking = overall[overall.config.isin(eligible.config)].groupby("config").delta_ap.agg(["mean", "min"])
    ranking = ranking[ranking["min"].ge(-0.002) & ranking["mean"].gt(0)].sort_index().sort_values(
        "mean", ascending=False, kind="stable",
    )
    selected = ranking.index[0] if len(ranking) else BASE
    encoding, arms = configs[selected]
    gene_check(X, y, ids, folds, trials[0], (encoding, arms), out)
    thresholds = operating_thresholds(X, y, folds, (encoding, arms))
    write_json(out / "locked.json", {
        "selected": selected, "encoding": encoding, "arms": list(arms),
        "components_screened_in_combination": components, "threshold": THRESHOLD,
        "thresholds_oof_f1": thresholds,
        "seed": SEED, "models": {model: estimator(model).get_params() for model in MODELS},
        "selection": "AP toàn tập; ít nhất một mô hình ΔAP ≥ 0.002, 4/5 fold tăng, ΔAP vượt SE, "
                     "và cả hai mô hình ΔAP ≥ -0.002; lấy ΔAP trung bình hai mô hình cao nhất.",
    })
    print(f"locked: {selected} ({arms})", flush=True)


def holdout(source, train, out):
    if (out / "holdout_metrics.csv").exists():
        raise ValueError("Holdout đã được đánh giá.")
    config = json.loads((out / "locked.json").read_text(encoding="utf-8"))
    require(config["threshold"] == THRESHOLD and config["seed"] == SEED
            and config["models"] == {model: estimator(model).get_params() for model in MODELS},
            "Thiết lập hiện tại khác cấu hình đã khóa.")
    thresholds = config["thresholds_oof_f1"]
    raw, X, y, ids = train
    X = prepare_features(raw, X, ids)
    fitted = {}
    for name, arms in {BASE: (), "selected": config["arms"]}.items():
        for model in MODELS:
            fitted[name, model] = fit_config(model, arms, config["encoding"], X, y)
    raw_test, X_test, y_test, ids_test = load_rows(source, "test")
    require(not X.index.intersection(X_test.index).size, "Train/test trùng mã dòng.")
    require(not set(ids.variant_id) & set(ids_test.variant_id), "Train/test trùng biến thể.")
    X_test = prepare_features(raw_test, X_test, ids_test)
    rows = []
    operating_rows = []
    export_dir = out / "pipeline"
    export_dir.mkdir(exist_ok=True)
    schemas = []
    for (name, model), (pipe, columns) in fitted.items():
        if name == "selected":
            tag = "scaled" if model == "LogisticRegression" else "unscaled"
            joblib.dump(pipe, export_dir / f"model_{model}.joblib", compress=3)
            pre = pipe[:-1]
            joblib.dump(pre, export_dir / f"preprocessor_{tag}.joblib", compress=3)
            restored = joblib.load(export_dir / f"preprocessor_{tag}.joblib")
            schema = pre.get_feature_names_out().tolist()
            require(len(set(schema)) == len(schema) and schema == restored.get_feature_names_out().tolist(),
                    "Tên cột không hợp lệ hoặc lệch sau khi nạp.")
            schemas.append(schema)
            write_json(export_dir / "feature_names.json", schema)
            for split, data, labels in (("train", X, y), ("test", X_test, y_test)):
                prepared = data[columns]
                matrix = sparse.csr_matrix(pre.transform(prepared))
                require(matrix.shape == (len(labels), len(schema)) and np.isfinite(matrix.data).all(),
                        "Ma trận có kích thước hoặc giá trị không hợp lệ.")
                require(data.index.equals(labels.index), "Ma trận lệch thứ tự nhãn.")
                path = export_dir / f"{tag}_{split}.npz"
                sparse.save_npz(path, matrix)
                require((sparse.load_npz(path) != sparse.csr_matrix(restored.transform(prepared))).nnz == 0,
                        "Pipeline nạp lại cho ma trận khác.")
                prepared.to_csv(export_dir / f"X_{split}.csv")
                labels.to_csv(export_dir / f"y_{split}.csv")
                np.save(export_dir / f"source_rows_{split}.npy", data.index.to_numpy(), allow_pickle=False)
            write_json(export_dir / "input_columns.json", columns)
        prob = pipe.predict_proba(X_test[columns])[:, 1]
        masks = subgroups(X_test)
        rows.extend(dict(config=name, model=model, fold=0, **metrics)
                    for metrics in scores(y_test, prob, masks))
        threshold = thresholds[name][model]
        operating_rows.extend(dict(config=name, model=model, fold=0, threshold=threshold, **metrics)
                              for metrics in scores(y_test, prob, masks, threshold))
    require(schemas[0] == schemas[1], "Schema chuẩn hóa và giữ thang gốc khác nhau.")
    comparisons(pd.DataFrame(operating_rows)).drop(columns="fold").to_csv(
        out / "holdout_threshold_metrics.csv", index=False,
    )
    comparisons(pd.DataFrame(rows)).drop(columns="fold").to_csv(out / "holdout_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "cv", "holdout"))
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = read_source()
    train = load_rows(source, "train")
    with threadpool_limits(limits=1):
        if args.phase == "audit":
            from audit_feature_coverage import audit
            audit(*train, args.out)
        elif args.phase == "cv":
            require((args.out / "spearman.png").is_file() and (args.out / "coverage.png").is_file(),
                    "Chạy audit để khảo sát hai heatmap trước.")
            cv(*train, args.out)
        else:
            holdout(source, train, args.out)


if __name__ == "__main__":
    main()
