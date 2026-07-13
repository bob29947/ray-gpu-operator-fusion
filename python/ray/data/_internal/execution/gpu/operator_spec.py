"""Immutable declarative specifications for modular GPU physical operators."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Optional, Tuple, Union

from ray.data._internal.execution.gpu.properties import (
    PropertyFact,
    PropertyRequirement,
    ValidationObligation,
)


class PayloadKind(str, Enum):
    # Opaque provider-owned descriptor for one range.  ParquetRangeWork is the
    # only V1 value, but a registered source may use its own descriptor behind
    # its own boundary adapter without extending this vocabulary.
    RANGE_WORK = "range_work"
    # Distribution-neutral RangeData envelope containing a frame stream.
    RANGE_DATA = "range_data"
    # The same envelope after materializing the range to one frame.
    RANGE_PARTITION_DATA = "range_partition_data"
    # GroupedRangeData envelope carrying the materialized frame and boundaries.
    GROUPED_RANGE_DATA = "grouped_range_data"
    RAY_BLOCK_STREAM = "ray_block_stream"
    # Raw device payloads remain available for future operators that do not
    # carry a range envelope; V1 range transforms must not use these in place
    # of RANGE_DATA/RANGE_PARTITION_DATA/GROUPED_RANGE_DATA.
    CUDF_BATCH_STREAM = "cudf_batch_stream"
    CUDF_PARTITION = "cudf_partition"
    GROUPED_CUDF_PARTITION = "grouped_cudf_partition"
    OUTPUT_BATCH_STREAM = "output_batch_stream"


# These are declarations, not V1 standalone adapter implementations.  Gate 2
# elides compatible internal boundaries and the final composed stage owns egress.
RANGE_WORK_ADAPTER_KEY = "ray-data-gpu-range-work-v1"
RANGE_DATA_BOUNDARY_ADAPTER_KEY = "ray-data-gpu-range-data-boundary-v1"
RANGE_DATA_TERMINAL_ADAPTER_KEY = "ray-data-gpu-range-data-terminal-v1"


ActorTaskRetryOnErrors = Union[bool, Tuple[type[BaseException], ...]]
RetriedMapErrors = Union[bool, Tuple[str, ...]]


def _validate_retry_count(name: str, value: int, *, allow_infinite: bool) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    minimum = -1 if allow_infinite else 0
    if value < minimum:
        qualifier = "-1 or nonnegative" if allow_infinite else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")


@dataclass(frozen=True)
class GPUStageRetryPolicy:
    """Canonical effective retry scopes that must match before composition.

    Values are the post-default Ray settings, rather than the presence or
    absence of user arguments.  Lists are normalized to immutable tuples.  Two
    policies that can retry different work therefore compare unequal.
    """

    actor_max_restarts: int = -1
    actor_max_task_retries: int = -1
    actor_task_retry_on_errors: ActorTaskRetryOnErrors = False
    actor_init_retry_on_errors: bool = False
    actor_init_max_retries: int = 3
    retried_map_errors: RetriedMapErrors = False
    max_map_retries: int = 3
    retried_io_errors: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.actor_init_retry_on_errors, bool):
            raise TypeError("actor_init_retry_on_errors must be a bool")
        _validate_retry_count(
            "actor_max_restarts", self.actor_max_restarts, allow_infinite=True
        )
        _validate_retry_count(
            "actor_max_task_retries",
            self.actor_max_task_retries,
            allow_infinite=True,
        )
        _validate_retry_count(
            "actor_init_max_retries",
            self.actor_init_max_retries,
            allow_infinite=True,
        )
        _validate_retry_count(
            "max_map_retries", self.max_map_retries, allow_infinite=False
        )

        actor_errors = self.actor_task_retry_on_errors
        if not isinstance(actor_errors, bool):
            actor_errors = tuple(actor_errors)
            if not all(
                isinstance(error, type) and issubclass(error, BaseException)
                for error in actor_errors
            ):
                raise TypeError(
                    "actor_task_retry_on_errors must be a bool or exception classes"
                )
            object.__setattr__(
                self,
                "actor_task_retry_on_errors",
                actor_errors if actor_errors else False,
            )

        map_errors = self.retried_map_errors
        if not isinstance(map_errors, bool):
            if isinstance(map_errors, str):
                raise TypeError("retried_map_errors must be a bool or string patterns")
            map_errors = tuple(map_errors)
            if not all(isinstance(pattern, str) for pattern in map_errors):
                raise TypeError("retried_map_errors must be a bool or string patterns")
            object.__setattr__(
                self,
                "retried_map_errors",
                map_errors if map_errors else False,
            )

        if isinstance(self.retried_io_errors, str):
            raise TypeError("retried_io_errors must contain string patterns")
        io_errors = tuple(self.retried_io_errors)
        if not all(isinstance(pattern, str) for pattern in io_errors):
            raise TypeError("retried_io_errors must contain string patterns")
        object.__setattr__(self, "retried_io_errors", io_errors)

        # These limits are not observed by Ray when their scope is disabled.
        # Normalize them so semantically identical policies compare equal.
        if not self.actor_init_retry_on_errors:
            object.__setattr__(self, "actor_init_max_retries", 0)
        if self.retried_map_errors is False:
            object.__setattr__(self, "max_map_retries", 0)


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
    retry_policy: GPUStageRetryPolicy = field(default_factory=GPUStageRetryPolicy)
    actor_bootstrap: Tuple[Tuple[str, str], ...] = ()
    external_adapter_abi: str = "ray-block-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "custom_resources", tuple(self.custom_resources))
        object.__setattr__(self, "label_selector", tuple(self.label_selector))
        object.__setattr__(self, "actor_bootstrap", tuple(self.actor_bootstrap))
        if not isinstance(self.retry_policy, GPUStageRetryPolicy):
            raise TypeError("retry_policy must be a GPUStageRetryPolicy")
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
    """One independently declared operator or an ordered composed GPU stage."""

    spec_instance_id: str
    transforms: Tuple[GPUTransformSpec, ...]
    external_input_payload: PayloadKind
    external_output_payload: PayloadKind
    native_input_payload: PayloadKind
    native_output_payload: PayloadKind
    input_adapter_key: str
    # Used for an unfused edge between GPU stages.  This adapter must preserve
    # the PartitionIdentity carried by RANGE_DATA.
    output_adapter_key: str
    # Used only at component egress, where partition identity is no longer an
    # internal execution obligation and ordinary Ray blocks are produced.
    terminal_output_adapter_key: str
    execution_requirements: GPUExecutionRequirements
    input_requirements: Tuple[PropertyRequirement, ...] = ()
    output_properties: Tuple[PropertyFact, ...] = ()
    logical_provenance: Tuple[str, ...] = ()
    gate2_composed: bool = False
    specialized_read_parallelism_resolved: bool = False
    uses_terminal_output_adapter: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "transforms", tuple(self.transforms))
        object.__setattr__(self, "input_requirements", tuple(self.input_requirements))
        object.__setattr__(self, "output_properties", tuple(self.output_properties))
        object.__setattr__(self, "logical_provenance", tuple(self.logical_provenance))
        if not self.spec_instance_id:
            raise ValueError("spec_instance_id must be nonempty")
        if (
            not self.input_adapter_key
            or not self.output_adapter_key
            or not self.terminal_output_adapter_key
        ):
            raise ValueError("GPU boundary adapter keys must be nonempty")
        if not self.transforms:
            raise ValueError("GPUOperatorSpec requires at least one transform")
        transform_ids = tuple(transform.transform_id for transform in self.transforms)
        if len(transform_ids) != len(set(transform_ids)):
            raise ValueError("GPUOperatorSpec transform_ids must be unique")
        if self.transforms[0].input_payload != self.native_input_payload:
            raise ValueError(
                "first transform input does not match native operator input"
            )
        if self.transforms[-1].output_payload != self.native_output_payload:
            raise ValueError(
                "last transform output does not match native operator output"
            )
        for upstream, downstream in zip(self.transforms, self.transforms[1:]):
            if upstream.output_payload != downstream.input_payload:
                raise ValueError(
                    "adjacent transform payloads do not match: "
                    f"{upstream.transform_id} -> {downstream.transform_id}"
                )
        if self.specialized_read_parallelism_resolved and not self.gate2_composed:
            raise ValueError(
                "specialized Read parallelism may be marked resolved only on a "
                "gate-2-composed stage"
            )

    @property
    def compatibility_key(self) -> GPUExecutionCompatibilityKey:
        return self.execution_requirements.compatibility_key

    @property
    def transform_ids(self) -> Tuple[str, ...]:
        return tuple(transform.transform_id for transform in self.transforms)

    @property
    def active_output_adapter_key(self) -> str:
        return (
            self.terminal_output_adapter_key
            if self.uses_terminal_output_adapter
            else self.output_adapter_key
        )

    def for_gate2_stage(
        self,
        *,
        specialized_read_parallelism_resolved: bool,
        uses_terminal_output_adapter: bool,
    ) -> "GPUOperatorSpec":
        """Seal a planner-composed stage against later physical re-composition."""

        return replace(
            self,
            gate2_composed=True,
            specialized_read_parallelism_resolved=(
                specialized_read_parallelism_resolved
            ),
            uses_terminal_output_adapter=uses_terminal_output_adapter,
        )

    def canonical_contract(self) -> Tuple[Any, ...]:
        """Return a deterministic comparison view that excludes opaque configs."""

        return (
            self.spec_instance_id,
            tuple(transform.canonical_contract() for transform in self.transforms),
            self.external_input_payload,
            self.external_output_payload,
            self.native_input_payload,
            self.native_output_payload,
            self.input_adapter_key,
            self.output_adapter_key,
            self.terminal_output_adapter_key,
            self.execution_requirements,
            self.input_requirements,
            self.output_properties,
            self.logical_provenance,
            self.gate2_composed,
            self.specialized_read_parallelism_resolved,
            self.uses_terminal_output_adapter,
        )
