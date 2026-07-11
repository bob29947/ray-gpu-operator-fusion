"""Pure pairwise composition for declarative GPU operator specifications."""

from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Tuple

from ray.data._internal.execution.gpu.operator_spec import (
    GPUBoundaryPolicy,
    GPUOperatorSpec,
    GPUTransformSpec,
)
from ray.data._internal.execution.gpu.properties import (
    ValidationObligation,
    requirements_satisfied,
)


class GPUCompositionErrorReason(str, Enum):
    EXECUTION_INCOMPATIBLE = "EXECUTION_INCOMPATIBLE"
    INPUT_REQUIREMENTS_UNSATISFIED = "INPUT_REQUIREMENTS_UNSATISFIED"
    PAYLOAD_INCOMPATIBLE = "PAYLOAD_INCOMPATIBLE"
    PHYSICAL_BOUNDARY_REQUIRED = "PHYSICAL_BOUNDARY_REQUIRED"
    UNRESOLVED_VALIDATION_OBLIGATION = "UNRESOLVED_VALIDATION_OBLIGATION"
    DUPLICATE_TRANSFORM_ID = "DUPLICATE_TRANSFORM_ID"


class GPUCompositionError(ValueError):
    def __init__(self, reason: GPUCompositionErrorReason, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}")


def unresolved_validation_obligations(
    transforms: Iterable[GPUTransformSpec],
) -> Tuple[ValidationObligation, ...]:
    """Validate obligation ordering and return obligations left at region output."""

    pending: List[ValidationObligation] = []
    for transform in transforms:
        if transform.requires_resolved_obligations and pending:
            raise GPUCompositionError(
                GPUCompositionErrorReason.UNRESOLVED_VALIDATION_OBLIGATION,
                f"{transform.transform_id} requires validation before use; "
                f"pending={pending!r}",
            )
        pending.extend(transform.creates_obligations)
        for obligation in transform.discharges_obligations:
            if obligation in pending:
                pending.remove(obligation)
    return tuple(pending)


def compose_gpu_operator_specs(
    upstream: GPUOperatorSpec,
    downstream: GPUOperatorSpec,
) -> GPUOperatorSpec:
    """Compose two adjacent specifications without consulting Ray runtime state."""

    if (
        upstream.execution_requirements.backend_name
        != downstream.execution_requirements.backend_name
        or upstream.compatibility_key != downstream.compatibility_key
    ):
        raise GPUCompositionError(
            GPUCompositionErrorReason.EXECUTION_INCOMPATIBLE,
            "GPU execution compatibility keys differ",
        )
    if (
        upstream.region_id is not None
        and downstream.region_id is not None
        and upstream.region_id != downstream.region_id
    ):
        raise GPUCompositionError(
            GPUCompositionErrorReason.EXECUTION_INCOMPATIBLE,
            f"region ids differ: {upstream.region_id!r} != {downstream.region_id!r}",
        )
    expected_upstream = upstream.required_logical_provenance
    expected_downstream = downstream.required_logical_provenance
    if (
        expected_upstream
        and expected_downstream
        and expected_upstream != expected_downstream
    ):
        raise GPUCompositionError(
            GPUCompositionErrorReason.EXECUTION_INCOMPATIBLE,
            "required logical provenance differs",
        )

    if upstream.output_payload != downstream.input_payload:
        raise GPUCompositionError(
            GPUCompositionErrorReason.PAYLOAD_INCOMPATIBLE,
            f"{upstream.output_payload.value} cannot feed "
            f"{downstream.input_payload.value}",
        )

    if not requirements_satisfied(
        upstream.output_properties, downstream.input_requirements
    ):
        raise GPUCompositionError(
            GPUCompositionErrorReason.INPUT_REQUIREMENTS_UNSATISFIED,
            f"facts={upstream.output_properties!r}, "
            f"requirements={downstream.input_requirements!r}",
        )

    if (
        upstream.output_boundary is GPUBoundaryPolicy.REQUIRED
        or downstream.input_boundary is GPUBoundaryPolicy.REQUIRED
    ):
        raise GPUCompositionError(
            GPUCompositionErrorReason.PHYSICAL_BOUNDARY_REQUIRED,
            "an intermediate physical boundary is required",
        )

    transforms = upstream.transforms + downstream.transforms
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

    # This scans the complete ordered transform list so an obligation created in
    # one node must be discharged before a dependent transform in a later node.
    unresolved_validation_obligations(transforms)

    return GPUOperatorSpec(
        transforms=transforms,
        input_payload=upstream.input_payload,
        output_payload=downstream.output_payload,
        execution_requirements=upstream.execution_requirements,
        input_requirements=upstream.input_requirements,
        output_properties=downstream.output_properties,
        input_boundary=upstream.input_boundary,
        output_boundary=downstream.output_boundary,
        logical_provenance=(
            upstream.logical_provenance + downstream.logical_provenance
        ),
        required_logical_provenance=(
            expected_upstream or expected_downstream
        ),
        region_id=upstream.region_id or downstream.region_id,
        requires_complete_fusion=(
            upstream.requires_complete_fusion
            or downstream.requires_complete_fusion
        ),
    )
