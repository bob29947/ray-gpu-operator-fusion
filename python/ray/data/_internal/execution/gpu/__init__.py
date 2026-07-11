"""Declarative contracts for Ray Data GPU physical operators.

This package deliberately does not expose actor-pool implementation details.  A
backend adapter turns these immutable contracts into a concrete physical operator.
"""

from ray.data._internal.execution.gpu.composition import (
    GPUCompositionError,
    GPUCompositionErrorReason,
    compose_gpu_operator_specs,
)
from ray.data._internal.execution.gpu.operator import (
    GPUOperator,
    GPUOperatorCreationOptions,
    GPUOperatorFactory,
    create_gpu_operator,
    register_gpu_operator_factory,
)
from ray.data._internal.execution.gpu.operator_spec import (
    GPUBoundaryPolicy,
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUOperatorSpec,
    GPUTransformSpec,
    PayloadKind,
)
from ray.data._internal.execution.gpu.properties import (
    Evidence,
    KeysColocated,
    LocalOrdering,
    OrderedBy,
    PropertyFact,
    RangePartitioning,
    RowsOwnedByRange,
    SortKey,
    UnknownOrdering,
    UnknownPartitioning,
)
from ray.data._internal.execution.gpu.region_compiler import compile_gpu_region

__all__ = [
    "Evidence",
    "GPUBoundaryPolicy",
    "GPUCompositionError",
    "GPUCompositionErrorReason",
    "GPUExecutionCompatibilityKey",
    "GPUExecutionRequirements",
    "GPUOperator",
    "GPUOperatorCreationOptions",
    "GPUOperatorFactory",
    "GPUOperatorSpec",
    "GPUTransformSpec",
    "KeysColocated",
    "LocalOrdering",
    "OrderedBy",
    "PayloadKind",
    "PropertyFact",
    "RangePartitioning",
    "RowsOwnedByRange",
    "SortKey",
    "UnknownOrdering",
    "UnknownPartitioning",
    "compile_gpu_region",
    "compose_gpu_operator_specs",
    "create_gpu_operator",
    "register_gpu_operator_factory",
]
