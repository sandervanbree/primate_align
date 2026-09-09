"""pysrf: Similarity-based Representation Factorization."""

from __future__ import annotations

import logging

logging.getLogger("pysrf").addHandler(logging.NullHandler())

from .consensus import AlignedConsensus, EnsembleFit
from .cross_validation import CVResult, cross_val_score
from .model import SRF

try:
    from importlib.metadata import version

    __version__ = version("pysrf")
except Exception:
    __version__ = "0.1.0"


__all__: list[str] = [
    "SRF",
    "CVResult",
    "cross_val_score",
    "EnsembleFit",
    "AlignedConsensus",
]
