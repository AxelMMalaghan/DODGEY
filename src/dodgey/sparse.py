from __future__ import annotations

import numpy as np
from scipy import sparse


def is_sparse(x: object) -> bool:
    return sparse.issparse(x)


def log1p_mean(x: sparse.spmatrix, axis: int) -> np.ndarray:
    if not sparse.issparse(x):
        raise TypeError("log1p_mean expects a scipy sparse matrix")
    log_x = x.copy()
    log_x.data = np.log1p(log_x.data)
    summed = np.asarray(log_x.sum(axis=axis)).ravel()
    n = x.shape[axis]
    return summed / float(n)


def mean_expr(x: sparse.spmatrix, axis: int) -> np.ndarray:
    if not sparse.issparse(x):
        raise TypeError("mean_expr expects a scipy sparse matrix")
    summed = np.asarray(x.sum(axis=axis)).ravel()
    n = x.shape[axis]
    return summed / float(n)


def nnz_count(x: sparse.spmatrix, axis: int) -> np.ndarray:
    if not sparse.issparse(x):
        raise TypeError("nnz_count expects a scipy sparse matrix")
    return x.getnnz(axis=axis)
