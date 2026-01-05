from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests


def lowess_fit(x: np.ndarray, y: np.ndarray, span: float) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    fitted_sorted = lowess(y_sorted, x_sorted, frac=span, return_sorted=False)
    fitted = np.empty_like(fitted_sorted)
    fitted[order] = fitted_sorted
    return fitted


def normal_pvals(ds: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute one-sided p-values assuming ds follows a normal distribution.

    Uses sample mean and standard deviation to parameterize the null distribution.
    This tests whether each residual is significantly larger than expected under
    the empirical distribution of all residuals.
    """
    mean = float(np.mean(ds))
    sd = float(np.std(ds))
    if sd < eps:
        sd = eps
    return 1.0 - stats.norm.cdf(ds, loc=mean, scale=sd)


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    return qvals
