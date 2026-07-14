"""Pure composition for declarative GPU operator specifications."""

from __future__ import annotations

from enum import Enum

from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionRequirementsConflict,
    GPUOperatorSpec,
    merge_gpu_execution_requirements,
)
from ray.data._internal.execution.gpu.properties import requirements_satisfied


class GPUCompositionErrorReason(str, Enum):
    EXECUTION_INCOMPATIBLE = "EXECUTION_INCOMPATIBLE"
    INPUT_REQUIREMENTS_UNSATISFIED = "INPUT_REQUIREMENTS_UNSATISFIED"
    PAYLOAD_INCOMPATIBLE = "PAYLOAD_INCOMPATIBLE"
    MATERIALIZABLE_BOUNDARY = "MATERIALIZABLE_BOUNDARY"


class GPUCompositionError(ValueError):
    def __init__(self, reason: GPUCompositionErrorReason, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}")


def foldable_edge(upstream: GPUOperatorSpec, downstream: GPUOperatorSpec) -> bool:
    payload = upstream.output_payload
    return payload == downstream.input_payload and not payload.materializable_boundary


def fuse_gpu_operator_specs(
    upstream: GPUOperatorSpec,
    downstream: GPUOperatorSpec,
) -> GPUOperatorSpec:
    if upstream.output_payload != downstream.input_payload:
        raise GPUCompositionError(
            GPUCompositionErrorReason.PAYLOAD_INCOMPATIBLE,
            f"{upstream.output_payload.key} cannot feed "
            f"{downstream.input_payload.key}",
        )
    if upstream.output_payload.materializable_boundary:
        raise GPUCompositionError(
            GPUCompositionErrorReason.MATERIALIZABLE_BOUNDARY,
            f"cannot fold across materializable payload "
            f"{upstream.output_payload.key!r}",
        )
    if not requirements_satisfied(
        upstream.output_facts,
        downstream.input_requirements,
    ):
        raise GPUCompositionError(
            GPUCompositionErrorReason.INPUT_REQUIREMENTS_UNSATISFIED,
            f"facts={upstream.output_facts!r}, "
            f"requirements={downstream.input_requirements!r}",
        )
    try:
        execution = merge_gpu_execution_requirements(
            upstream.execution_requirements,
            downstream.execution_requirements,
        )
    except GPUExecutionRequirementsConflict as error:
        raise GPUCompositionError(
            GPUCompositionErrorReason.EXECUTION_INCOMPATIBLE,
            str(error),
        ) from error
    return GPUOperatorSpec(
        transforms=upstream.transforms + downstream.transforms,
        execution_requirements=execution,
        input_requirements=upstream.input_requirements,
        output_facts=downstream.output_facts,
    )
