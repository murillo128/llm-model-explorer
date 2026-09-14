"""Static metadata-only architecture graph core. Family descriptions register explicitly."""

from .core import (
    AnalysisInput,
    AnalysisResult,
    Description,
    DescriptionRegistry,
    GraphBuilder,
    Producer,
    parse_graph,
    serialize_graph,
)
from .dense import register_dense_descriptions
from .validation import BindingContext, GraphError, NumericTensor

__all__ = [
    "AnalysisInput",
    "AnalysisResult",
    "BindingContext",
    "Description",
    "DescriptionRegistry",
    "GraphBuilder",
    "GraphError",
    "NumericTensor",
    "Producer",
    "parse_graph",
    "serialize_graph",
    "register_dense_descriptions",
]
