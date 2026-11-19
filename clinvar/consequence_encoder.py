"""Mã hóa các hệ quả VEP phân cách bằng dấu &."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import check_is_fitted


def _terms(value):
    if pd.isna(value):
        return set()
    return {term for part in value.split("&") if (term := part.strip())}


class ConsequenceMultiHot(TransformerMixin, BaseEstimator):
    """Học các hệ quả từ train và dành một cột cố định cho ô thiếu."""

    def fit(self, X, y=None):
        self.encoder_ = MultiLabelBinarizer(sparse_output=True)
        self.encoder_.fit(_terms(value) for value in X.iloc[:, 0])
        return self

    def transform(self, X):
        check_is_fitted(self, "encoder_")
        parsed = [_terms(value) for value in X.iloc[:, 0]]
        known = set(self.encoder_.classes_)
        encoded = self.encoder_.transform(terms & known for terms in parsed)
        missing = sparse.csr_matrix(np.array([not terms for terms in parsed], dtype=float).reshape(-1, 1))
        return sparse.hstack([encoded, missing], format="csr")

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "encoder_")
        return np.array(
            [f"Consequence_term_{term}" for term in self.encoder_.classes_] + ["Consequence_missing"],
            dtype=object,
        )


class RareConsequence(TransformerMixin, BaseEstimator):
    """Gộp hệ quả hiếm hoặc chưa gặp thành một cột và tách riêng ô thiếu."""

    def __init__(self, min_count=50):
        self.min_count = min_count

    def fit(self, X, y=None):
        counts = X.iloc[:, 0].map(_terms).explode().value_counts()
        self.terms_ = sorted(counts[counts.ge(self.min_count)].index)
        return self

    def transform(self, X):
        check_is_fitted(self, "terms_")
        lookup = {term: i for i, term in enumerate(self.terms_)}
        rare = len(lookup)
        columns, rows = [], []
        for row, value in enumerate(X.iloc[:, 0]):
            terms = _terms(value)
            positions = {lookup.get(term, rare) for term in terms} if terms else {rare + 1}
            rows.extend([row] * len(positions))
            columns.extend(positions)
        return sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(X), rare + 2))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "terms_")
        return np.array([f"Consequence_term_{term}" for term in self.terms_]
                        + ["Consequence_rare_or_unseen", "Consequence_missing"], dtype=object)
