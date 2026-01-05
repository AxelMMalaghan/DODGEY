from __future__ import annotations

from typing import Optional

import numpy as np

from .core import score_labels


def score_adata(
    adata,
    cluster_key: str,
    *,
    sample_key: Optional[str] = None,
    layer: str = "counts",
    **kwargs,
):
    """Compute ROGUE-inspired purity scores for clusters in an AnnData object.

    Args:
        adata: AnnData object with count data.
        cluster_key: Column in adata.obs containing cluster labels.
        sample_key: Optional column in adata.obs for sample-wise scoring.
            If provided, scores are computed per sample.
        layer: Layer containing count data. Use None for adata.X.
        **kwargs: Additional arguments passed to score_labels.

    Returns:
        DataFrame with cluster scores. If sample_key is provided, includes
        a 'sample' column.

    Raises:
        ImportError: If pandas is not installed.
        KeyError: If cluster_key, sample_key, or layer is not found.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("dodgey[scanpy] requires pandas and anndata") from exc

    if layer is None:
        x = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        x = adata.layers[layer]

    obs = adata.obs
    if cluster_key not in obs:
        raise KeyError(f"Cluster key '{cluster_key}' not found in adata.obs")

    rows = []
    if sample_key is None:
        labels = np.asarray(obs[cluster_key])
        df = score_labels(x, labels, axis="cells_by_genes", **kwargs)
        if isinstance(df, list):
            return pd.DataFrame(df)
        return df

    if sample_key not in obs:
        raise KeyError(f"Sample key '{sample_key}' not found in adata.obs")

    for sample in obs[sample_key].unique():
        mask = obs[sample_key] == sample
        labels = np.asarray(obs.loc[mask, cluster_key])
        x_sub = x[mask, :]
        df = score_labels(x_sub, labels, axis="cells_by_genes", **kwargs)
        if isinstance(df, list):
            df = pd.DataFrame(df)
        df.insert(0, "sample", sample)
        rows.append(df)

    return pd.concat(rows, ignore_index=True)
