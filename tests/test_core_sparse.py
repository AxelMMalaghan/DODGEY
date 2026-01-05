import numpy as np
from scipy import sparse

import dodgey as sn


def test_dense_sparse_equivalence():
    rng = np.random.default_rng(1)
    x = rng.poisson(1.5, size=(40, 50))
    x_sparse = sparse.csr_matrix(x)
    dense_score = sn.score_matrix(x, axis="cells_by_genes")
    sparse_score = sn.score_matrix(x_sparse, axis="cells_by_genes")
    assert np.isclose(dense_score, sparse_score, rtol=1e-6, atol=1e-6)
