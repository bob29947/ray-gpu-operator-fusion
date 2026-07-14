"""Pure pairwise composition for declarative GPU operator specifications."""

from __future__ import annotations

from enum import Enum
from typing import Iterable, List, Tuple

from ray.data._internal.execution.gpu.operator_spec import (
    GPUOperatorSpec,
    GPUTransformSpec,
)
from ray.data._internal.execution.gpu.properties import (
    ValidationObligation,
    requirements_satisfied,
)


class GPUCompositionErrorReason(str, Enum):
    EXECUTION_INCOMPATIBLE = "EXECUTION_INCOMPATIBLE"
    FUSION_GROUP_INCOMPATIBLE = "FUSION_GROUP_INCOMPATIBLE"
    INPUT_REQUIREMENTS_UNSATISFIED = "INPUT_REQUIREMENTS_UNSATISFIED"
    PAYLOAD_INCOMPATIBLE = "PAYLOAD_INCOMPATIBLE"
    ADAPTER_INCOMPATIBLE = "ADAPTER_INCOMPATIBLE"
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
    """Validate obligation ordering and return obligations left at stage output."""

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


def _composed_spec_instance_id(transforms: Tuple[GPUTransformSpec, ...]) -> str:
    # Derived only from the flattened ordered transform sequence, so pairwise
    # composition produces the same identifier under either association.
    return (
        "gpu-stage["
        + ",".join(transform.transform_id for transform in transforms)
        + "]"
    )


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
            "GPU execution or retry compatibility keys differ",
        )

    if upstream.fusion_group_id != downstream.fusion_group_id:
        raise GPUCompositionError(
            GPUCompositionErrorReason.FUSION_GROUP_INCOMPATIBLE,
            "GPU fusion group identifiers differ: "
            f"{upstream.fusion_group_id!r} != {downstream.fusion_group_id!r}",
        )
    if upstream.requires_complete_fusion != downstream.requires_complete_fusion:
        raise GPUCompositionError(
            GPUCompositionErrorReason.FUSION_GROUP_INCOMPATIBLE,
            "required-fusion declarations differ across one fusion group",
        )
    if (
        upstream.required_logical_provenance
        != downstream.required_logical_provenance
    ):
        raise GPUCompositionError(
            GPUCompositionErrorReason.FUSION_GROUP_INCOMPATIBLE,
            "required logical provenance differs across one fusion group",
        )

    if upstream.native_output_payload != downstream.native_input_payload:
        raise GPUCompositionError(
            GPUCompositionErrorReason.PAYLOAD_INCOMPATIBLE,
            f"{upstream.native_output_payload.value} cannot feed "
            f"{downstream.native_input_payload.value}",
        )

    if (
        upstream.external_output_payload != downstream.external_input_payload
        or upstream.output_adapter_key != downstream.input_adapter_key
    ):
        raise GPUCompositionError(
            GPUCompositionErrorReason.ADAPTER_INCOMPATIBLE,
            "declared intermediate boundary adapters differ: "
            f"{upstream.external_output_payload.value}/"
            f"{upstream.output_adapter_key!r} -> "
            f"{downstream.external_input_payload.value}/"
            f"{downstream.input_adapter_key!r}",
        )
    if upstream.uses_terminal_output_adapter:
        raise GPUCompositionError(
            GPUCompositionErrorReason.ADAPTER_INCOMPATIBLE,
            "a terminal GPU egress cannot feed another planned GPU operator",
        )

    if not requirements_satisfied(
        upstream.output_properties, downstream.input_requirements
    ):
        raise GPUCompositionError(
            GPUCompositionErrorReason.INPUT_REQUIREMENTS_UNSATISFIED,
            f"facts={upstream.output_properties!r}, "
            f"requirements={downstream.input_requirements!r}",
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

    # Scan the complete ordered list so an obligation created in one spec must
    # be discharged before a dependent transform in a later spec.
    unresolved_validation_obligations(transforms)

    return GPUOperatorSpec(
        spec_instance_id=_composed_spec_instance_id(transforms),
        transforms=transforms,
        external_input_payload=upstream.external_input_payload,
        external_output_payload=downstream.external_output_payload,
        native_input_payload=upstream.native_input_payload,
        native_output_payload=downstream.native_output_payload,
        input_adapter_key=upstream.input_adapter_key,
        # Composition compatibility is intentionally based on the preserving
        # boundary declaration. Terminal materialization is selected only for
        # the final node in a required physical-fusion group.
        output_adapter_key=downstream.output_adapter_key,
        terminal_output_adapter_key=downstream.terminal_output_adapter_key,
        execution_requirements=upstream.execution_requirements,
        input_requirements=upstream.input_requirements,
        # Property transfer is always explicit.  Composition never smuggles an
        # upstream fact through a downstream operator that did not advertise it.
        output_properties=downstream.output_properties,
        logical_provenance=(
            upstream.logical_provenance + downstream.logical_provenance
        ),
        fusion_group_id=upstream.fusion_group_id,
        requires_complete_fusion=upstream.requires_complete_fusion,
        required_logical_provenance=upstream.required_logical_provenance,
        # Input parallelism belongs to the composed region's external input.
        owns_input_parallelism=upstream.owns_input_parallelism,
        uses_terminal_output_adapter=downstream.uses_terminal_output_adapter,
    )
