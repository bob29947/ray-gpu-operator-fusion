"""Immutable declarative specifications for modular GPU physical operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple

from ray.data._internal.execution.gpu.properties import (
    PropertyFact,
    PropertyRequirement,
    ValidationObligation,
)


class PayloadKind(str, Enum):
    RANGE_WORK = "range_work"
    RAY_BLOCK_STREAM = "ray_block_stream"
    CUDF_BATCH_STREAM = "cudf_batch_stream"
    CUDF_PARTITION = "cudf_partition"
    GROUPED_CUDF_PARTITION = "grouped_cudf_partition"
    OUTPUT_BATCH_STREAM = "output_batch_stream"


class GPUBoundaryPolicy(str, Enum):
    ELIDABLE = "elidable"
    REQUIRED = "required"


@dataclass(frozen=True)
class GPUExecutionCompatibilityKey:
    """Canonical fields which must match for actor-local execution to compose."""

    runtime_id: str = "ray-data-gpu-v1"
    worker_lifecycle: str = "actor_pool"
    min_workers: int = 1
    initial_workers: int = 1
    max_workers: int = 1
    max_tasks_in_flight_per_worker: int = 1
    max_udf_concurrency: int = 1
    num_cpus: float = 0
    num_gpus: float = 1
    memory: Optional[float] = None
    custom_resources: Tuple[Tuple[str, float], ...] = ()
    accelerator_type: Optional[str] = None
    scheduling_strategy: Optional[str] = None
    label_selector: Tuple[Tuple[str, str], ...] = ()
    runtime_env_serialized: Optional[str] = None
    retry_policy: str = "none"
    actor_bootstrap: Tuple[Tuple[str, str], ...] = ()
    external_adapter_abi: str = "ray-block-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "custom_resources", tuple(self.custom_resources))
        object.__setattr__(self, "label_selector", tuple(self.label_selector))
        object.__setattr__(self, "actor_bootstrap", tuple(self.actor_bootstrap))
        if self.min_workers < 1:
            raise ValueError("min_workers must be at least 1")
        if not self.min_workers <= self.initial_workers <= self.max_workers:
            raise ValueError(
                "initial_workers must be within [min_workers, max_workers]"
            )
        if self.max_tasks_in_flight_per_worker != 1:
            raise ValueError("GPU V1 requires exactly one task in flight per worker")
        if self.max_udf_concurrency != 1:
            raise ValueError("GPU V1 requires exactly one concurrent UDF per worker")
        if self.retry_policy != "none":
            raise ValueError("GPU V1 requires the no-retry execution policy")


@dataclass(frozen=True)
class GPUExecutionRequirements:
    backend_name: str = "actor_pool"
    compatibility_key: GPUExecutionCompatibilityKey = field(
        default_factory=GPUExecutionCompatibilityKey
    )


@dataclass(frozen=True, eq=False)
class GPUTransformSpec:
    """One actor-local transform.

    ``config`` is deliberately opaque to the fusion core.  It may contain a UDF,
    so contract comparison never calls its equality operator.
    """

    transform_id: str
    kind: str
    input_payload: PayloadKind
    output_payload: PayloadKind
    config: Any = field(default=None, compare=False, repr=False)
    creates_obligations: Tuple[ValidationObligation, ...] = ()
    discharges_obligations: Tuple[ValidationObligation, ...] = ()
    requires_resolved_obligations: bool = False

    def __post_init__(self) -> None:
        if not self.transform_id:
            raise ValueError("transform_id must be nonempty")
        if not self.kind:
            raise ValueError("kind must be nonempty")
        object.__setattr__(self, "creates_obligations", tuple(self.creates_obligations))
        object.__setattr__(
            self, "discharges_obligations", tuple(self.discharges_obligations)
        )

    def canonical_contract(self) -> Tuple[Any, ...]:
        return (
            self.transform_id,
            self.kind,
            self.input_payload,
            self.output_payload,
            self.creates_obligations,
            self.discharges_obligations,
            self.requires_resolved_obligations,
        )


@dataclass(frozen=True, eq=False)
class GPUOperatorSpec:
    transforms: Tuple[GPUTransformSpec, ...]
    input_payload: PayloadKind
    output_payload: PayloadKind
    execution_requirements: GPUExecutionRequirements
    input_requirements: Tuple[PropertyRequirement, ...] = ()
    output_properties: Tuple[PropertyFact, ...] = ()
    input_boundary: GPUBoundaryPolicy = GPUBoundaryPolicy.ELIDABLE
    output_boundary: GPUBoundaryPolicy = GPUBoundaryPolicy.ELIDABLE
    logical_provenance: Tuple[str, ...] = ()
    required_logical_provenance: Tuple[str, ...] = ()
    region_id: Optional[str] = None
    requires_complete_fusion: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "transforms", tuple(self.transforms))
        object.__setattr__(self, "input_requirements", tuple(self.input_requirements))
        object.__setattr__(self, "output_properties", tuple(self.output_properties))
        object.__setattr__(self, "logical_provenance", tuple(self.logical_provenance))
        object.__setattr__(
            self,
            "required_logical_provenance",
            tuple(self.required_logical_provenance),
        )
        if not self.transforms:
            raise ValueError("GPUOperatorSpec requires at least one transform")
        if self.transforms[0].input_payload != self.input_payload:
            raise ValueError(
                "first transform input does not match operator input payload"
            )
        if self.transforms[-1].output_payload != self.output_payload:
            raise ValueError(
                "last transform output does not match operator output payload"
            )
        for upstream, downstream in zip(self.transforms, self.transforms[1:]):
            if upstream.output_payload != downstream.input_payload:
                raise ValueError(
                    "adjacent transform payloads do not match: "
                    f"{upstream.transform_id} -> {downstream.transform_id}"
                )
        if self.requires_complete_fusion and not self.region_id:
            raise ValueError("a complete-fusion requirement needs a region_id")

    @property
    def compatibility_key(self) -> GPUExecutionCompatibilityKey:
        return self.execution_requirements.compatibility_key

    @property
    def transform_ids(self) -> Tuple[str, ...]:
        return tuple(transform.transform_id for transform in self.transforms)

    def canonical_contract(self) -> Tuple[Any, ...]:
        """Return a deterministic comparison view that excludes opaque configs."""

        return (
            tuple(transform.canonical_contract() for transform in self.transforms),
            self.input_payload,
            self.output_payload,
            self.execution_requirements,
            self.input_requirements,
            self.output_properties,
            self.input_boundary,
            self.output_boundary,
            self.logical_provenance,
            self.required_logical_provenance,
            self.region_id,
            self.requires_complete_fusion,
        )
