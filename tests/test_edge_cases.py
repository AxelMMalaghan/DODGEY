import numpy as np
import pytest
from scipy import sparse

import dodgey as sn


def test_empty_matrix():
    """Test that empty matrix returns NaN with warning."""
    x = np.zeros((0, 10))
    with pytest.warns(UserWarning, match="too few cells"):
        score = sn.score_matrix(x, axis="cells_by_genes")
    assert np.isnan(score)


def test_all_zeros_matrix():
    """Test matrix with all zeros."""
    x = np.zeros((20, 30))
    with pytest.warns(UserWarning, match="No genes passed filters"):
        score = sn.score_matrix(x, axis="cells_by_genes")
    assert np.isnan(score)


def test_single_cell():
    """Test with single cell returns NaN."""
    rng = np.random.default_rng(123)
    x = rng.poisson(1.0, size=(1, 50))
    with pytest.warns(UserWarning, match="too few cells"):
        score = sn.score_matrix(x, axis="cells_by_genes")
    assert np.isnan(score)


def test_few_cells_below_threshold():
    """Test that clusters below min_cells_per_cluster return NaN."""
    rng = np.random.default_rng(456)
    x = rng.poisson(1.0, size=(5, 50))
    with pytest.warns(UserWarning, match="too few cells"):
        score = sn.score_matrix(x, axis="cells_by_genes", min_cells_per_cluster=10)
    assert np.isnan(score)


def test_single_gene():
    """Test with single gene returns NaN (too few for LOWESS)."""
    rng = np.random.default_rng(789)
    x = rng.poisson(1.0, size=(30, 1))
    with pytest.warns(UserWarning, match="Too few genes"):
        score = sn.score_matrix(x, axis="cells_by_genes")
    assert np.isnan(score)


def test_two_genes():
    """Test with two genes returns NaN (too few for LOWESS)."""
    rng = np.random.default_rng(101)
    x = rng.poisson(1.0, size=(30, 2))
    with pytest.warns(UserWarning, match="Too few genes"):
        score = sn.score_matrix(x, axis="cells_by_genes")
    assert np.isnan(score)


def test_log_transformed_warning():
    """Test that log-transformed data triggers a warning."""
    rng = np.random.default_rng(202)
    # Create data that looks log-transformed (small values, mostly non-integer)
    x = rng.uniform(0.0, 5.0, size=(30, 50))
    with pytest.warns(UserWarning, match="log-transformed"):
        sn.score_matrix(x, axis="cells_by_genes")


def test_sparse_empty_matrix():
    """Test sparse matrix with no data."""
    x = sparse.csr_matrix((20, 30))
    with pytest.warns(UserWarning, match="No genes passed filters"):
        score = sn.score_matrix(x, axis="cells_by_genes")
    assert np.isnan(score)


def test_return_details_on_nan_result():
    """Test that return_details works even when result is NaN."""
    x = np.zeros((20, 30))
    with pytest.warns(UserWarning):
        result = sn.score_matrix(x, axis="cells_by_genes", return_details=True)
    assert isinstance(result, sn.Result)
    assert np.isnan(result.rogue)


def test_score_labels_with_small_cluster():
    """Test score_labels when one cluster is too small."""
    rng = np.random.default_rng(303)
    x = rng.poisson(1.0, size=(35, 40))
    # Cluster A has 30 cells, cluster B has only 5
    labels = ["A"] * 30 + ["B"] * 5

    with pytest.warns(UserWarning, match="too few cells"):
        result = sn.score_labels(x, labels, axis="cells_by_genes", min_cells_per_cluster=10)

    # Check that small cluster got NaN
    if hasattr(result, "loc"):
        b_row = result.loc[result["cluster"] == "B"]
        assert np.isnan(b_row["rogue"].values[0])
    else:
        b_row = [r for r in result if r["cluster"] == "B"][0]
        assert np.isnan(b_row["rogue"])


def test_homogeneous_cluster_high_score():
    """Test that a homogeneous cluster gets a high score."""
    rng = np.random.default_rng(404)
    # Create very uniform data
    x = np.ones((50, 100)) + rng.poisson(0.1, size=(50, 100))
    score = sn.score_matrix(x, axis="cells_by_genes")
    assert score > 0.8  # Should be high for homogeneous data


def test_genes_by_cells_orientation():
    """Test that genes_by_cells orientation works correctly."""
    rng = np.random.default_rng(505)
    x = rng.poisson(1.0, size=(50, 30))  # genes x cells
    score = sn.score_matrix(x, axis="genes_by_cells")
    assert 0.0 <= score <= 1.0


def test_sparse_csc_format():
    """Test that CSC sparse format works."""
    rng = np.random.default_rng(606)
    x_dense = rng.poisson(1.5, size=(40, 50))
    x_csc = sparse.csc_matrix(x_dense)

    dense_score = sn.score_matrix(x_dense, axis="cells_by_genes")
    sparse_score = sn.score_matrix(x_csc, axis="cells_by_genes")

    assert np.isclose(dense_score, sparse_score, rtol=1e-6)
