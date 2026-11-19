"""Đối chiếu Week 1 bằng train và các fold đã công bố."""

import argparse
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.model_selection import StratifiedGroupKFold

from feature_study import AF, OUTPUT, PROTEIN, SEED, load_rows, prepare_features, read_source, run_config, subgroups


def main(gene=False):
    raw, baseline, y, ids = load_rows(read_source(), "train")
    X = prepare_features(raw, baseline, ids)
    saved = pd.read_csv(OUTPUT / "fold_assignments.csv", index_col="source_row")
    pd.testing.assert_index_equal(saved.index, y.index)
    pd.testing.assert_series_equal(saved.CLASS.astype(y.dtype), y)
    assert set(saved.fold) == set(range(1, 6))
    folds = [(np.flatnonzero(saved.fold != i), np.flatnonzero(saved.fold == i)) for i in range(1, 6)]
    config = json.loads((OUTPUT / "locked.json").read_text(encoding="utf-8"))
    out = OUTPUT / "vy_thao"
    out.mkdir(exist_ok=True)
    if gene:
        groups = X.SYMBOL.where(X.SYMBOL.ne("__MISSING__"), "unknown:" + ids.variant_id)
        folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, groups))
        assert all(set(groups.iloc[t]).isdisjoint(groups.iloc[v]) for t, v in folds)
        existing = pd.read_csv(OUTPUT / "gene_fold_metrics.csv").query("scheme == 'gene' and config == 'selected'")
        with threadpool_limits(limits=1):
            trial = run_config("af_log+af_zero", (config["encoding"], [*config["arms"], "af_log", "af_zero"]), X, y, folds)
        keys = ["model", "fold", "subgroup", "n", "positive"]
        compared = trial.merge(existing[keys + ["ap"]].rename(columns={"ap": "ap_locked"}),
                               on=keys, validate="one_to_one")
        assert len(compared) == len(trial) == len(existing)
        compared["delta_ap"] = compared.ap - compared.ap_locked
        compared.to_csv(out / "gene_folds.csv", index=False)
        return
    missense = subgroups(X)["missense"]
    coverage = []
    for group, rows in (("missense", missense), ("non_missense", ~missense)):
        for col in PROTEIN:
            count = int(raw.loc[rows, col].isna().sum())
            coverage.append(dict(group=group, column=col, n=int(rows.sum()), missing=count,
                                 share_missing=count / rows.sum()))
    pd.DataFrame(coverage).to_csv(out / "protein_coverage.csv", index=False)
    coding = ["Amino_acids", "Codons", "Protein_position", "CDS_position"]
    states = pd.DataFrame({
        "transcript_region": X.transcript_region,
        "coding_fields_missing": raw[coding].isna().sum(axis=1),
        "missense": missense,
    })
    states.value_counts().rename("n").to_csv(out / "coding_coverage.csv")
    freq = pd.DataFrame({"n": len(X), "zeros": X[AF].eq(0).sum(),
                         "missing": X[AF].isna().sum(), "share_zero": X[AF].eq(0).mean()})
    freq.rename_axis("column").to_csv(out / "af_coverage.csv")
    existing = pd.read_csv(OUTPUT / "cv_fold_metrics.csv")
    keys = ["model", "fold", "subgroup"]
    reference = existing.loc[existing.config.eq(config["selected"]), keys + ["ap"]].rename(
        columns={"ap": "ap_locked"})
    variants = [("af_zero",), ("af_log",), ("af_log", "af_zero"), ("cadd_present",)]
    results = []
    with threadpool_limits(limits=1):
        for arms in variants:
            trial = run_config("+".join(arms), (config["encoding"], [*config["arms"], *arms]), X, y, folds)
            compared = trial.merge(reference, on=keys, how="left", validate="one_to_one", indicator=True)
            assert compared["_merge"].eq("both").all() and len(compared) == len(reference)
            compared["delta_ap"] = compared.ap - compared.ap_locked
            results.append(compared.drop(columns="_merge"))
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(out / "cv_folds.csv", index=False)
    combined.groupby(["config", "model", "subgroup"]).agg(
        ap=("ap", "mean"), delta_ap=("delta_ap", "mean"),
        folds_up=("delta_ap", lambda values: values.gt(0).sum()),
        folds_valid=("delta_ap", "count"),
    ).to_csv(out / "cv_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", action="store_true", help="Kiểm tra cấu hình log AF và cờ 0 trên gene chưa gặp.")
    main(parser.parse_args().gene)
