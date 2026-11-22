"""Build a portable K-means dataset from the existing variant split."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from preprocess import load_split, make_preprocessor


ROOT = Path(__file__).resolve().parent


def main():
    train, _, train_ids = load_split(ROOT / "data_multihot", "train")
    test, _, test_ids = load_split(ROOT / "data_multihot", "test")
    folds = pd.read_csv(ROOT / "feature_retained/fold_assignments.csv", index_col="source_row")
    pd.testing.assert_index_equal(train.index, folds.index)
    assert set(folds.variant_fold) == set(range(1, 6))
    dev = folds.variant_fold.eq(1)
    frames = {"train": train.loc[~dev], "dev": train.loc[dev], "test": test}
    keys = [set(train_ids.loc[~dev, "variant_id"]), set(train_ids.loc[dev, "variant_id"]), set(test_ids.variant_id)]
    assert sum(map(len, keys)) == len(set.union(*keys)) == len(train) + len(test)
    prep = make_preprocessor(scale_numeric=True, consequence_encoding="multihot")
    prep.fit(frames["train"])
    names = prep.get_feature_names_out().tolist()
    with TemporaryDirectory(dir=ROOT) as temp:
        out = Path(temp)
        for split, frame in frames.items():
            matrix = sparse.csr_matrix(prep.transform(frame))
            assert matrix.shape == (len(frame), len(names)) and np.isfinite(matrix.data).all()
            sparse.save_npz(out / f"X_{split}.npz", matrix)
            np.save(out / f"source_rows_{split}.npy", frame.index.to_numpy())
        (out / "feature_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
        joblib.dump(prep, out / "preprocessor.joblib")
        archive = out / "clinvar_kmeans.zip"
        files = sorted(out.iterdir())
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
            for path in [*files, ROOT / "consequence_encoder.py", ROOT / "requirements.txt"]:
                bundle.write(path, path.name)
        archive.replace(ROOT / archive.name)


if __name__ == "__main__":
    main()
