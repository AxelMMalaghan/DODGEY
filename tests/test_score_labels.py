import numpy as np

import dodgey as sn


def test_score_labels_output_fields():
    rng = np.random.default_rng(4)
    x = rng.poisson(1.0, size=(30, 40))
    labels = np.array(["a"] * 15 + ["b"] * 15)
    result = sn.score_labels(x, labels, axis="cells_by_genes")

    if hasattr(result, "columns"):
        columns = set(result.columns)
        assert {"cluster", "rogue", "n_cells", "n_genes", "n_sig_genes", "sig_value"}.issubset(
            columns
        )
    else:
        assert isinstance(result, list)
        keys = set(result[0].keys())
        assert {"cluster", "rogue", "n_cells", "n_genes", "n_sig_genes", "sig_value"}.issubset(
            keys
        )


def test_score_labels_preserves_order():
    rng = np.random.default_rng(5)
    x = rng.poisson(1.0, size=(12, 8))
    labels = np.array(["b", "a", "b", "c", "a", "c", "b", "b", "c", "a", "a", "c"])
    result = sn.score_labels(x, labels, axis="cells_by_genes")
    if hasattr(result, "cluster"):
        clusters = list(result["cluster"])
    else:
        clusters = [row["cluster"] for row in result]
    assert clusters == ["b", "a", "c"]


def test_score_labels_return_details():
    rng = np.random.default_rng(6)
    x = rng.poisson(1.0, size=(20, 10))
    labels = np.array(["a"] * 10 + ["b"] * 10)
    result = sn.score_labels(x, labels, axis="cells_by_genes", return_details=True)
    if hasattr(result, "columns"):
        assert "details" in result.columns
        assert all(hasattr(details, "rogue") for details in result["details"])
    else:
        assert "details" in result[0]
        assert hasattr(result[0]["details"], "rogue")
