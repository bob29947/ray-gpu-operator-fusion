"""Declarative contracts for Ray Data GPU physical operators.

This package deliberately does not expose actor-pool implementation details.  A
backend adapter turns these immutable contracts into a concrete physical operator.
"""

from ray.data._internal.execution.gpu.composition import (
    GPUCompositionError,
    GPUCompositionErrorReason,
    compose_gpu_operator_specs,
)
from ray.data._internal.execution.gpu.boundary_adapters import (
    GPUBoundaryAdapter,
    GPUBoundaryDirection,
    get_gpu_boundary_adapter,
    register_gpu_boundary_adapter,
    unregister_gpu_boundary_adapter,
)
from ray.data._internal.execution.gpu.operator import (
    GPURequiredFusionError,
    GPUOperator,
    GPUOperatorCreationOptions,
    GPUOperatorFactory,
    PlannedGPUOperator,
    create_gpu_operator,
    register_gpu_operator_factory,
)
from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUOperatorSpec,
    GPUStageRetryPolicy,
    GPUTransformSpec,
    PayloadKind,
    RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    RANGE_DATA_TERMINAL_ADAPTER_KEY,
    RANGE_WORK_ADAPTER_KEY,
)
from ray.data._internal.execution.gpu.payloads import (
    GroupedRangeData,
    PartitionIdentity,
    RangeData,
    RangePartitionIdentity,
)
from ray.data._internal.execution.gpu.properties import (
    CompletePartitionBundles,
    Evidence,
    KeysColocated,
    PartitioningProperty,
    PartitioningValidation,
    PropertyFact,
    PropertyRequirement,
    PropertyValue,
    RangePartitioning,
    RangePartitioningRequired,
    RowsOwnedByRange,
    UnknownPartitioning,
    property_satisfies,
    requirements_satisfied,
    ValidationObligation,
)
from ray.data._internal.execution.gpu.region_compiler import compile_gpu_region
from ray.data._internal.execution.gpu.runtime import (
    register_gpu_transform_runtime,
    supports_gpu_transform_runtime,
    unregister_gpu_transform_runtime,
)

__all__ = [
    "Evidence",
    "CompletePartitionBundles",
    "GPUBoundaryAdapter",
    "GPUBoundaryDirection",
    "GPUCompositionError",
    "GPUCompositionErrorReason",
    "GPUExecutionCompatibilityKey",
    "GPUExecutionRequirements",
    "GPUOperator",
    "GPUOperatorCreationOptions",
    "GPUOperatorFactory",
    "GPURequiredFusionError",
    "GPUOperatorSpec",
    "GPUStageRetryPolicy",
    "GPUTransformSpec",
    "GroupedRangeData",
    "KeysColocated",
    "PartitionIdentity",
    "PartitioningProperty",
    "PartitioningValidation",
    "PayloadKind",
    "PlannedGPUOperator",
    "PropertyFact",
    "PropertyRequirement",
    "PropertyValue",
    "RANGE_DATA_BOUNDARY_ADAPTER_KEY",
    "RANGE_DATA_TERMINAL_ADAPTER_KEY",
    "RANGE_WORK_ADAPTER_KEY",
    "RangeData",
    "RangePartitionIdentity",
    "RangePartitioning",
    "RangePartitioningRequired",
    "RowsOwnedByRange",
    "UnknownPartitioning",
    "ValidationObligation",
    "compile_gpu_region",
    "compose_gpu_operator_specs",
    "create_gpu_operator",
    "get_gpu_boundary_adapter",
    "property_satisfies",
    "register_gpu_operator_factory",
    "register_gpu_transform_runtime",
    "register_gpu_boundary_adapter",
    "requirements_satisfied",
    "supports_gpu_transform_runtime",
    "unregister_gpu_boundary_adapter",
    "unregister_gpu_transform_runtime",
]
