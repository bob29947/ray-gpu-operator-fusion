"""Region fold shared by the physical fusion rule and contract tests."""

from functools import reduce
from typing import Iterable

from ray.data._internal.execution.gpu.composition import (
    GPUCompositionError,
    GPUCompositionErrorReason,
    compose_gpu_operator_specs,
    unresolved_validation_obligations,
)
from ray.data._internal.execution.gpu.operator_spec import GPUOperatorSpec


def compile_gpu_region(
    specs: Iterable[GPUOperatorSpec],
    *,
    require_resolved_output: bool = True,
) -> GPUOperatorSpec:
    """Fold adjacent specs and optionally require a sealed-stage output.

    Gate 2 uses ``require_resolved_output=False`` while probing a partial
    component prefix because a later operator may discharge an obligation.
    Final stage construction and physical fusion keep the default and reject
    any obligation that would escape an executable stage.
    """

    specs = tuple(specs)
    if not specs:
        raise ValueError("cannot compile an empty GPU region")
    transforms = tuple(transform for spec in specs for transform in spec.transforms)
    transform_ids = tuple(transform.transform_id for transform in transforms)
    if len(transform_ids) != len(set(transform_ids)):
        duplicates = sorted(
            transform_id
            for transform_id in set(transform_ids)
            if transform_ids.count(transform_id) > 1
        )
        raise GPUCompositionError(
            GPUCompositionErrorReason.DUPLICATE_TRANSFORM_ID,
            f"duplicate transform ids: {duplicates!r}",
        )
    pending = unresolved_validation_obligations(transforms)
    if require_resolved_output and pending:
        raise GPUCompositionError(
            GPUCompositionErrorReason.UNRESOLVED_VALIDATION_OBLIGATION,
            f"compiled stage has unresolved obligations: {pending!r}",
        )
    return reduce(compose_gpu_operator_specs, specs)
