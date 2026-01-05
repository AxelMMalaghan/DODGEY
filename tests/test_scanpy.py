import numpy as np
import pytest

pytest.importorskip("anndata")
pytest.importorskip("pandas")

import anndata
import pandas as pd

import dodgey as sn


@pytest.fixture
def simple_adata():
    """Create a simple AnnData object for testing."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 60, 40
    counts = rng.poisson(2.0, size=(n_cells, n_genes))
    adata = anndata.AnnData(X=counts.astype(float))
    adata.layers["counts"] = counts
    adata.obs["cluster"] = ["A"] * 30 + ["B"] * 30
    adata.obs["sample"] = ["S1"] * 20 + ["S2"] * 20 + ["S1"] * 10 + ["S2"] * 10
    return adata


def test_score_adata_basic(simple_adata):
    """Test basic score_adata functionality."""
    result = sn.score_adata(simple_adata, cluster_key="cluster", layer="counts")
    assert isinstance(result, pd.DataFrame)
    assert "cluster" in result.columns
    assert "rogue" in result.columns
    assert len(result) == 2  # Two clusters: A and B


def test_score_adata_with_sample_key(simple_adata):
    """Test score_adata with sample stratification."""
    result = sn.score_adata(
        simple_adata, cluster_key="cluster", sample_key="sample", layer="counts"
    )
    assert isinstance(result, pd.DataFrame)
    assert "sample" in result.columns
    assert "cluster" in result.columns
    # 2 samples x 2 clusters = 4 rows
    assert len(result) == 4


def test_score_adata_missing_cluster_key(simple_adata):
    """Test that missing cluster key raises KeyError."""
    with pytest.raises(KeyError, match="not found in adata.obs"):
        sn.score_adata(simple_adata, cluster_key="nonexistent", layer="counts")


def test_score_adata_missing_layer(simple_adata):
    """Test that missing layer raises KeyError."""
    with pytest.raises(KeyError, match="not found in adata.layers"):
        sn.score_adata(simple_adata, cluster_key="cluster", layer="nonexistent")


def test_score_adata_missing_sample_key(simple_adata):
    """Test that missing sample key raises KeyError."""
    with pytest.raises(KeyError, match="not found in adata.obs"):
        sn.score_adata(
            simple_adata,
            cluster_key="cluster",
            sample_key="nonexistent",
            layer="counts",
        )


def test_score_adata_uses_x_when_layer_none(simple_adata):
    """Test that layer=None uses adata.X."""
    result = sn.score_adata(simple_adata, cluster_key="cluster", layer=None)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_score_adata_passes_kwargs(simple_adata):
    """Test that kwargs are passed to score_labels."""
    result = sn.score_adata(
        simple_adata, cluster_key="cluster", layer="counts", k=100.0, cutoff=0.1
    )
    assert isinstance(result, pd.DataFrame)
    assert result["k"].iloc[0] == 100.0
    assert result["cutoff"].iloc[0] == 0.1
