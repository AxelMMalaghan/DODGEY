import numpy as np

import dodgey as sn


def test_axis_auto_cells_by_genes():
    rng = np.random.default_rng(2)
    x = rng.poisson(1.0, size=(20, 60))
    auto_score = sn.score_matrix(x, axis="auto")
    explicit_score = sn.score_matrix(x, axis="cells_by_genes")
    assert np.isclose(auto_score, explicit_score)


def test_axis_auto_genes_by_cells():
    rng = np.random.default_rng(3)
    x = rng.poisson(1.0, size=(60, 20))
    auto_score = sn.score_matrix(x, axis="auto")
    explicit_score = sn.score_matrix(x, axis="genes_by_cells")
    assert np.isclose(auto_score, explicit_score)
