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
