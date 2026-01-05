from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from . import sparse as sparse_ops
from .stats import fdr_bh, lowess_fit, normal_pvals
from .typing import AxisOption, LabelsLike, MatrixLike


@dataclass
class Result:
    """Container for detailed scoring results.

    Attributes:
        rogue: The ROGUE-inspired score in [0, 1]. Higher means more homogeneous.
        sig_value: Sum of absolute deviations for significant genes.
        n_sig_genes: Number of genes passing significance thresholds.
        n_cells: Number of cells in the scored matrix.
        n_genes: Number of genes after filtering.
        params: Dictionary of parameters used for scoring.
        ds: Residuals (fitted - observed entropy) per gene.
        pvals: Raw p-values per gene.
        qvals: FDR-adjusted q-values per gene.
        mean_expr_proxy: Log mean expression proxy per gene.
        entropy_proxy: Entropy proxy per gene.
        fitted_entropy: LOWESS-fitted entropy values.
        gene_mask: Boolean mask of genes passing filters.
    """

    rogue: float
    sig_value: float
    n_sig_genes: int
    n_cells: int
    n_genes: int
    params: Dict[str, Any]
    ds: Optional[np.ndarray] = None
    pvals: Optional[np.ndarray] = None
    qvals: Optional[np.ndarray] = None
    mean_expr_proxy: Optional[np.ndarray] = None
    entropy_proxy: Optional[np.ndarray] = None
    fitted_entropy: Optional[np.ndarray] = None
    gene_mask: Optional[np.ndarray] = None


def _infer_axis(shape: Tuple[int, int]) -> str:
    n_rows, n_cols = shape
    if n_rows < n_cols:
        return "cells_by_genes"
    return "genes_by_cells"


def _axis_from_labels(shape: Tuple[int, int], labels: LabelsLike) -> str:
    n_rows, n_cols = shape
    n_labels = len(labels)
    if n_labels == n_rows:
        return "cells_by_genes"
    if n_labels == n_cols:
        return "genes_by_cells"
    raise ValueError("Labels length does not match any matrix axis")


def _validate_non_negative(x: MatrixLike) -> None:
    if sparse_ops.is_sparse(x):
        if x.data.size and np.min(x.data) < 0:
            raise ValueError("Input contains negative values")
    else:
        arr = np.asarray(x)
        if arr.size and np.min(arr) < 0:
            raise ValueError("Input contains negative values")


def _warn_if_loglike(x: MatrixLike) -> None:
    if sparse_ops.is_sparse(x):
        data = x.data
    else:
        data = np.asarray(x).ravel()
    if data.size == 0:
        return
    max_val = float(np.max(data))
    frac_nonint = float(np.mean(np.abs(data - np.round(data)) > 1e-6))
    if max_val <= 20.0 and frac_nonint > 0.5:
        warnings.warn(
            "Input appears log-transformed or normalized. dodgey expects count-like values.",
            UserWarning,
            stacklevel=2,
        )


def _validate_params(
    *,
    k: float,
    cutoff: float,
    span: float,
    r: float,
    min_cells_per_gene: int,
    min_mean_expr: Optional[float],
    min_cells_per_cluster: int,
) -> None:
    if k <= 0:
        raise ValueError("k must be > 0")
    if r <= 0:
        raise ValueError("r must be > 0")
    if span <= 0 or span > 1:
        raise ValueError("span must be in (0, 1]")
    if cutoff <= 0 or cutoff >= 1:
        raise ValueError("cutoff must be in (0, 1)")
    if min_cells_per_gene < 1:
        raise ValueError("min_cells_per_gene must be >= 1")
    if min_cells_per_cluster < 1:
        raise ValueError("min_cells_per_cluster must be >= 1")
    if min_mean_expr is not None and min_mean_expr < 0:
        raise ValueError("min_mean_expr must be >= 0 when provided")


def _to_cells_by_genes(x: MatrixLike, axis: AxisOption) -> Tuple[MatrixLike, str]:
    if axis == "auto":
        axis = _infer_axis(x.shape)
    if axis not in {"cells_by_genes", "genes_by_cells"}:
        raise ValueError("axis must be 'auto', 'cells_by_genes', or 'genes_by_cells'")
    if axis == "genes_by_cells":
        return (x.T if sparse_ops.is_sparse(x) else np.asarray(x).T), axis
    return (x if sparse_ops.is_sparse(x) else np.asarray(x)), axis


def _gene_filters(
    x: MatrixLike,
    min_cells_per_gene: int,
    min_mean_expr: Optional[float],
) -> np.ndarray:
    if sparse_ops.is_sparse(x):
        nnz = sparse_ops.nnz_count(x, axis=0)
        mean_expr = sparse_ops.mean_expr(x, axis=0)
    else:
        nnz = np.count_nonzero(x > 0, axis=0)
        mean_expr = np.mean(x, axis=0)
    mask = nnz >= min_cells_per_gene
    if min_mean_expr is not None:
        mask &= mean_expr >= min_mean_expr
    return mask


def _compute_proxies(x: MatrixLike, r: float) -> Tuple[np.ndarray, np.ndarray]:
    if sparse_ops.is_sparse(x):
        mean_expr = sparse_ops.mean_expr(x, axis=0)
        entropy = sparse_ops.log1p_mean(x, axis=0)
    else:
        mean_expr = np.mean(x, axis=0)
        entropy = np.log1p(x).mean(axis=0)
    mean_expr_proxy = np.log(mean_expr + r)
    return mean_expr_proxy, entropy


def _score_matrix_internal(
    x: MatrixLike,
    *,
    k: float,
    cutoff: float,
    span: float,
    r: float,
    axis: AxisOption,
    min_cells_per_gene: int,
    min_mean_expr: Optional[float],
    min_cells_per_cluster: int,
    return_details: bool,
) -> Result:
    _validate_params(
        k=k,
        cutoff=cutoff,
        span=span,
        r=r,
        min_cells_per_gene=min_cells_per_gene,
        min_mean_expr=min_mean_expr,
        min_cells_per_cluster=min_cells_per_cluster,
    )
    _validate_non_negative(x)
    _warn_if_loglike(x)
    x_cb, resolved_axis = _to_cells_by_genes(x, axis)
    n_cells = x_cb.shape[0]
    if n_cells < min_cells_per_cluster:
        warnings.warn(
            f"Cluster has too few cells (n={n_cells}); returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return Result(
            rogue=float("nan"),
            sig_value=float("nan"),
            n_sig_genes=0,
            n_cells=n_cells,
            n_genes=0,
            params={
                "k": k,
                "cutoff": cutoff,
                "span": span,
                "r": r,
                "axis": resolved_axis,
                "min_cells_per_gene": min_cells_per_gene,
                "min_mean_expr": min_mean_expr,
                "min_cells_per_cluster": min_cells_per_cluster,
            },
        )

    gene_mask = _gene_filters(x_cb, min_cells_per_gene, min_mean_expr)
    if not np.any(gene_mask):
        warnings.warn("No genes passed filters; returning NaN.", UserWarning, stacklevel=2)
        return Result(
            rogue=float("nan"),
            sig_value=float("nan"),
            n_sig_genes=0,
            n_cells=n_cells,
            n_genes=0,
            params={
                "k": k,
                "cutoff": cutoff,
                "span": span,
                "r": r,
                "axis": resolved_axis,
                "min_cells_per_gene": min_cells_per_gene,
                "min_mean_expr": min_mean_expr,
                "min_cells_per_cluster": min_cells_per_cluster,
            },
            gene_mask=gene_mask if return_details else None,
        )

    x_filtered = x_cb[:, gene_mask]
    mean_expr_proxy, entropy_proxy = _compute_proxies(x_filtered, r)
    if mean_expr_proxy.size < 3:
        warnings.warn("Too few genes to fit LOWESS; returning NaN.", UserWarning, stacklevel=2)
        return Result(
            rogue=float("nan"),
            sig_value=float("nan"),
            n_sig_genes=0,
            n_cells=n_cells,
            n_genes=int(mean_expr_proxy.size),
            params={
                "k": k,
                "cutoff": cutoff,
                "span": span,
                "r": r,
                "axis": resolved_axis,
                "min_cells_per_gene": min_cells_per_gene,
                "min_mean_expr": min_mean_expr,
                "min_cells_per_cluster": min_cells_per_cluster,
            },
            mean_expr_proxy=mean_expr_proxy if return_details else None,
            entropy_proxy=entropy_proxy if return_details else None,
            gene_mask=gene_mask if return_details else None,
        )

    fitted_entropy = lowess_fit(mean_expr_proxy, entropy_proxy, span)
    ds = fitted_entropy - entropy_proxy
    pvals = normal_pvals(ds)
    qvals = fdr_bh(pvals)

    sig_mask = (pvals < cutoff) & (qvals < cutoff)
    sig_value = float(np.sum(np.abs(ds[sig_mask]))) if np.any(sig_mask) else 0.0
    n_sig_genes = int(np.sum(sig_mask))

    rogue = 1.0 - sig_value / (sig_value + float(k))

    return Result(
        rogue=float(rogue),
        sig_value=float(sig_value),
        n_sig_genes=n_sig_genes,
        n_cells=n_cells,
        n_genes=int(mean_expr_proxy.size),
        params={
            "k": k,
            "cutoff": cutoff,
            "span": span,
            "r": r,
            "axis": resolved_axis,
            "min_cells_per_gene": min_cells_per_gene,
            "min_mean_expr": min_mean_expr,
            "min_cells_per_cluster": min_cells_per_cluster,
        },
        ds=ds if return_details else None,
        pvals=pvals if return_details else None,
        qvals=qvals if return_details else None,
        mean_expr_proxy=mean_expr_proxy if return_details else None,
        entropy_proxy=entropy_proxy if return_details else None,
        fitted_entropy=fitted_entropy if return_details else None,
        gene_mask=gene_mask if return_details else None,
    )


def score_matrix(
    x: MatrixLike,
    *,
    k: float = 45.0,
    cutoff: float = 0.05,
    span: float = 0.5,
    r: float = 1.0,
    axis: AxisOption = "auto",
    return_details: bool = False,
    min_cells_per_gene: int = 1,
    min_mean_expr: Optional[float] = None,
    min_cells_per_cluster: int = 10,
) -> float | Result:
    """Compute a ROGUE-inspired purity score for a count matrix.

    Args:
        x: Count matrix (cells x genes or genes x cells).
        k: Scale parameter. Use 45.0 for droplet/UMI data, 500.0 for full-length.
        cutoff: P-value and FDR threshold for significance.
        span: LOWESS smoothing span (frac parameter).
        r: Offset added inside log for mean expression.
        axis: Matrix orientation. "auto" infers from shape.
        return_details: If True, return a Result object with full details.
        min_cells_per_gene: Minimum cells expressing a gene to include it.
        min_mean_expr: Minimum mean expression to include a gene.
        min_cells_per_cluster: Minimum cells required; returns NaN if fewer.

    Returns:
        If return_details is False, returns the score as a float.
        If return_details is True, returns a Result object.
    """
    _validate_params(
        k=k,
        cutoff=cutoff,
        span=span,
        r=r,
        min_cells_per_gene=min_cells_per_gene,
        min_mean_expr=min_mean_expr,
        min_cells_per_cluster=min_cells_per_cluster,
    )
    result = _score_matrix_internal(
        x,
        k=k,
        cutoff=cutoff,
        span=span,
        r=r,
        axis=axis,
        min_cells_per_gene=min_cells_per_gene,
        min_mean_expr=min_mean_expr,
        min_cells_per_cluster=min_cells_per_cluster,
        return_details=return_details,
    )
    return result if return_details else result.rogue


def score_labels(
    x: MatrixLike,
    labels: LabelsLike,
    *,
    k: float = 45.0,
    cutoff: float = 0.05,
    span: float = 0.5,
    r: float = 1.0,
    axis: AxisOption = "auto",
    return_details: bool = False,
    min_cells_per_gene: int = 1,
    min_mean_expr: Optional[float] = None,
    min_cells_per_cluster: int = 10,
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """Compute ROGUE-inspired purity scores for each cluster defined by labels.

    Args:
        x: Count matrix (cells x genes or genes x cells).
        labels: Cluster labels, one per cell.
        k: Scale parameter. Use 45.0 for droplet/UMI data, 500.0 for full-length.
        cutoff: P-value and FDR threshold for significance.
        span: LOWESS smoothing span (frac parameter).
        r: Offset added inside log for mean expression.
        axis: Matrix orientation. "auto" infers from labels length.
        return_details: If True, include a Result object per cluster.
        min_cells_per_gene: Minimum cells expressing a gene to include it.
        min_mean_expr: Minimum mean expression to include a gene.
        min_cells_per_cluster: Minimum cells required per cluster.

    Returns:
        DataFrame with columns: cluster, rogue, n_cells, n_genes, n_sig_genes,
        sig_value, k, cutoff, span, r. Falls back to list of dicts if pandas
        is not available.
    """
    labels = np.asarray(labels)
    _validate_params(
        k=k,
        cutoff=cutoff,
        span=span,
        r=r,
        min_cells_per_gene=min_cells_per_gene,
        min_mean_expr=min_mean_expr,
        min_cells_per_cluster=min_cells_per_cluster,
    )
    if axis == "auto":
        axis = _axis_from_labels(x.shape, labels)
    if axis not in {"cells_by_genes", "genes_by_cells"}:
        raise ValueError("axis must be 'auto', 'cells_by_genes', or 'genes_by_cells'")

    unique_labels, first_idx = np.unique(labels, return_index=True)
    clusters = unique_labels[np.argsort(first_idx)]
    rows = []
    for cluster in clusters:
        mask = labels == cluster
        if axis == "cells_by_genes":
            x_cluster = x[mask, :]
        else:
            x_cluster = x[:, mask]
        result = _score_matrix_internal(
            x_cluster,
            k=k,
            cutoff=cutoff,
            span=span,
            r=r,
            axis=axis,
            min_cells_per_gene=min_cells_per_gene,
            min_mean_expr=min_mean_expr,
            min_cells_per_cluster=min_cells_per_cluster,
            return_details=return_details,
        )
        row = {
            "cluster": cluster,
            "rogue": result.rogue,
            "n_cells": result.n_cells,
            "n_genes": result.n_genes,
            "n_sig_genes": result.n_sig_genes,
            "sig_value": result.sig_value,
            "k": k,
            "cutoff": cutoff,
            "span": span,
            "r": r,
        }
        if return_details:
            row["details"] = result
        rows.append(row)

    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        return rows
