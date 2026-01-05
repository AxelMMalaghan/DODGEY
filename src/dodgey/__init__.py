from __future__ import annotations

from ._version import __version__
from .core import Result, score_labels, score_matrix

try:
    from .scanpy import score_adata  # type: ignore
except ImportError:

    def score_adata(*args, **kwargs):  # type: ignore
        raise ImportError(
            "score_adata requires optional dependencies. Install with: pip install dodgey[scanpy]"
        )


__all__ = ["Result", "score_matrix", "score_labels", "score_adata", "__version__"]
