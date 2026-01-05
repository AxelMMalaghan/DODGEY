# dodgey

ROGUE-inspired cluster purity scoring for count matrices.

`dodgey` provides a lightweight, NumPy-like API for computing a heterogeneity
score inspired by the ROGUE method used in scRNA-seq cluster QC. This is **not**
a verbatim port of ROGUE; it is a deterministic, documented implementation that
captures the core idea while keeping dependencies minimal.

## Installation

```bash
pip install dodgey
```

Optional integrations:

```bash
pip install dodgey[scanpy]
```

## Quickstart

### Score a matrix

```python
import numpy as np
import dodgey as dg

X = np.random.poisson(1.0, size=(200, 300))  # cells x genes
score = dg.score_matrix(X, k=45.0, axis="cells_by_genes")
print(score)
```

### Score clusters by labels

```python
import numpy as np
import dodgey as dg

X = np.random.poisson(1.0, size=(200, 300))
labels = np.repeat(["A", "B"], 100)
result = dg.score_labels(X, labels)
print(result)
```

### Score an AnnData object (optional)

```python
import dodgey as dg

scores = dg.score_adata(adata, cluster_key="leiden", layer="counts")
print(scores.head())
```

## What the metric means

Given a count matrix, `dodgey` computes per-gene summary statistics over cells,
fits a smooth expected relationship between those statistics, and measures how
much each gene deviates from expectation. Genes with significant deviations
contribute to a `sig_value`, and the final score is:

```
score = 1 - sig_value / (sig_value + k)
```

Higher scores indicate more homogeneous clusters (lower heterogeneity), and
lower scores indicate more heterogeneous clusters. The input is expected to be
raw or lightly processed counts (not already log-transformed).

## Scoring details

The method is **ROGUE-inspired** and intentionally lightweight:

1. Compute per-gene proxies across cells:
   - `entropy_proxy = mean(log(expr + 1))`
   - `mean_expr_proxy = log(mean(expr) + r)`
2. Fit a smooth expected relationship `entropy_proxy ~ mean_expr_proxy` using
   LOWESS/LOESS (statsmodels). The LOWESS `span` maps to the `frac` parameter.
3. Compute residuals: `ds = fitted_entropy - observed_entropy`.
4. Estimate p-values using a normal approximation on `ds`:
   - `p = 1 - NormalCDF(ds; mean(ds), sd(ds))`
   - `sd` is guarded with a small epsilon to avoid division by zero.
5. Apply Benjamini–Hochberg FDR correction.
6. Significant genes satisfy `p < cutoff` and `q < cutoff`.
7. `sig_value = sum(abs(ds))` over significant genes.
8. Final score: `score = 1 - sig_value / (sig_value + k)` in `[0, 1]`.

Notes:
- This implementation is deterministic: LOWESS is fitted on sorted mean
  expression, then mapped back to the original gene order.
- Sparse matrices are processed without densifying.
- If the input appears already log-transformed, a warning is emitted.

### Formal definition

Let \(X \in \mathbb{R}^{n \times g}\) be a non-negative count matrix with
cells as rows and genes as columns. For gene \(j\):

```
H_j = (1/n) * sum_i log(X_ij + 1)
M_j = log((1/n) * sum_i X_ij + r)
```

Fit a LOWESS curve \(\hat{H}(M)\) and compute residuals:

```
ds_j = \\hat{H}(M_j) - H_j
```

Let \(\mu\) and \(\sigma\) be the mean and standard deviation of `ds`. Then:

```
p_j = 1 - Phi((ds_j - mu) / sigma)
```

Apply Benjamini–Hochberg FDR to obtain `q_j`. Significant genes satisfy
`p_j < cutoff` and `q_j < cutoff`.

```
sig_value = sum_j |ds_j| for significant genes
score = 1 - sig_value / (sig_value + k)
```

### Worked example (tiny)

```
X = [[2, 0, 1],
     [3, 0, 0],
     [0, 1, 0],
     [1, 0, 0]]
```

For gene 1 (index 0) with r = 1.0:

```
mean = (2 + 3 + 0 + 1) / 4 = 1.5
M_1 = log(1.5 + 1.0) = log(2.5)
H_1 = mean(log([3, 4, 1, 2])) / 4
```

Repeat for all genes, fit LOWESS over `M_j` vs `H_j`, compute `ds_j`, then the
final score. In real use, larger gene sets are needed for stable LOWESS fits.

## Key parameters

- `k`: scale parameter (default `45.0` for droplet/UMI; use `500.0` for full-length)
- `cutoff`: p-value and FDR threshold (default `0.05`)
- `span`: LOWESS span (`frac`) controlling smoothing (default `0.5`)
- `r`: small offset added inside the log for mean expression (default `1.0`)
- `axis`: `"auto"`, `"cells_by_genes"`, or `"genes_by_cells"` (auto uses a simple
  heuristic: if rows < columns, assume cells by genes; otherwise genes by cells)

## Acknowledgement

This package is **ROGUE-inspired** and is not intended as a verbatim
reimplementation of the original method.

## Citations

If you use `dodgey` in your research, please consider citing:

> McKenna, A. (2025). dodgey: ROGUE-inspired cluster purity scoring for count matrices. GitHub. https://github.com/AxelMMalaghan/DODGEY

## License

MIT
