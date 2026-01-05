from __future__ import annotations

from typing import Literal, Sequence, Union

import numpy as np
import numpy.typing as npt
from scipy import sparse

ArrayLike = Union[npt.NDArray[np.floating], npt.NDArray[np.integer]]
SparseMatrix = Union[sparse.csr_matrix, sparse.csc_matrix]
MatrixLike = Union[ArrayLike, SparseMatrix]
AxisOption = Literal["auto", "genes_by_cells", "cells_by_genes"]

LabelsLike = Union[Sequence[str], Sequence[int], npt.NDArray[np.str_], npt.NDArray[np.integer]]

__all__ = [
    "ArrayLike",
    "SparseMatrix",
    "MatrixLike",
    "AxisOption",
    "LabelsLike",
]
