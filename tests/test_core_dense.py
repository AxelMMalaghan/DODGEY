import numpy as np
import pytest

import dodgey as sn


def test_determinism_dense():
    rng = np.random.default_rng(42)
    x = rng.poisson(1.0, size=(50, 80))
    score1 = sn.score_matrix(x, axis="cells_by_genes")
    score2 = sn.score_matrix(x, axis="cells_by_genes")
    assert np.isclose(score1, score2)


def test_negative_values_error():
    x = np.array([[1.0, -1.0], [0.0, 2.0]])
    with pytest.raises(ValueError):
        sn.score_matrix(x, axis="cells_by_genes")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": -1},
        {"r": 0},
        {"r": -1},
        {"span": 0},
        {"span": 1.5},
        {"cutoff": 0},
        {"cutoff": 1},
        {"min_cells_per_gene": 0},
        {"min_cells_per_cluster": 0},
        {"min_mean_expr": -0.1},
    ],
)
def test_param_validation(kwargs):
    x = np.ones((10, 12))
    with pytest.raises(ValueError):
        sn.score_matrix(x, axis="cells_by_genes", **kwargs)


def test_mixed_cluster_scores_lower():
    rng = np.random.default_rng(0)
    n_cells = 100
    n_genes = 50
    x = rng.poisson(1.0, size=(n_cells, n_genes))
    # Create strong differential signal between two halves
    x[:50, 0:10] += rng.poisson(8.0, size=(50, 10))
    x[50:, 10:20] += rng.poisson(8.0, size=(50, 10))

    pure_score = sn.score_matrix(x[:50, :], axis="cells_by_genes")
    mixed_score = sn.score_matrix(x, axis="cells_by_genes")
    assert mixed_score < pure_score
