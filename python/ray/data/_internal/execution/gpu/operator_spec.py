"""Immutable, backend-neutral contracts for modular GPU operators."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, FrozenSet, Optional, Sequence, Tuple, Union

from ray.data._internal.execution.gpu.properties import (
    PropertyFact,
    PropertyRequirement,
    RangePartitioningRequired,
    range_partitioning_fact,
)


@dataclass(frozen=True)
class PayloadKind:
    key: str
    materializable_boundary: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("payload key must be a nonempty string")
        if not isinstance(self.materializable_boundary, bool):
            raise TypeError("materializable_boundary must be a bool")
        expected = {
            "ray_block_stream": True,
            "device_handle": True,
        }.get(self.key)
        if expected is not None and self.materializable_boundary is not expected:
            raise ValueError(
                f"payload {self.key!r} requires materializable_boundary={expected}"
            )

    @property
    def value(self) -> str:
        return self.key


RAY_BLOCK_STREAM = PayloadKind("ray_block_stream", True)
# Reserved only. A future device boundary must carry an opaque bulk-data handle
# plus a small host sidecar; CPU control-plane transforms may touch only the
# sidecar and must forward the handle unchanged. No V1 runtime implements this.
DEVICE_HANDLE = PayloadKind("device_handle", True)
RANGE_WORK = PayloadKind("range_work")
RANGE_DATA = PayloadKind("range_data")
RANGE_PARTITION_DATA = PayloadKind("range_partition_data")
GROUPED_RANGE_DATA = PayloadKind("grouped_range_data")


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
        for name, allow_infinite in (
            ("actor_max_restarts", True),
            ("actor_max_task_retries", True),
            ("actor_init_max_retries", True),
            ("max_map_retries", False),
        ):
            _validate_retry_count(
                name, getattr(self, name), allow_infinite=allow_infinite
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
        if not self.actor_init_retry_on_errors:
            object.__setattr__(self, "actor_init_max_retries", 0)
        if self.retried_map_errors is False:
            object.__setattr__(self, "max_map_retries", 0)


@dataclass(frozen=True)
class GPUExecutionCompatibilityKey:
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

    def __post_init__(self) -> None:
        for name in ("custom_resources", "label_selector", "actor_bootstrap"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.runtime_id or not self.worker_lifecycle:
            raise ValueError("runtime_id and worker_lifecycle must be nonempty")
        if not isinstance(self.retry_policy, GPUStageRetryPolicy):
            raise TypeError("retry_policy must be a GPUStageRetryPolicy")
        if not all(
            isinstance(key, str) and key and isinstance(value, str)
            for key, value in self.actor_bootstrap
        ):
            raise TypeError("actor_bootstrap must contain nonempty string keys")
        if len(dict(self.actor_bootstrap)) != len(self.actor_bootstrap):
            raise ValueError("actor_bootstrap contains duplicate keys")
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


class GPUExecutionRequirementsConflict(ValueError):
    pass


@dataclass(frozen=True)
class GPUExecutionRequirements:
    backend_keys: FrozenSet[str] = frozenset(("actor_pool",))
    compatibility_key: Optional[GPUExecutionCompatibilityKey] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_keys", frozenset(self.backend_keys))
        if not self.backend_keys or not all(
            isinstance(key, str) and key for key in self.backend_keys
        ):
            raise ValueError("GPU execution requires nonempty backend keys")
        if self.compatibility_key is not None and not isinstance(
            self.compatibility_key, GPUExecutionCompatibilityKey
        ):
            raise TypeError(
                "compatibility_key must be a GPUExecutionCompatibilityKey or None"
            )

    @property
    def backend_name(self) -> str:
        if len(self.backend_keys) != 1:
            raise ValueError(
                "execution requirements do not select one backend: "
                f"{self.backend_keys!r}"
            )
        return next(iter(self.backend_keys))

    @property
    def is_resolved(self) -> bool:
        return self.compatibility_key is not None and len(self.backend_keys) == 1


def merge_gpu_execution_requirements(
    upstream: GPUExecutionRequirements,
    downstream: GPUExecutionRequirements,
) -> GPUExecutionRequirements:
    backend_keys = upstream.backend_keys & downstream.backend_keys
    if not backend_keys:
        raise GPUExecutionRequirementsConflict(
            "GPU operators have no compatible execution backend"
        )
    left = upstream.compatibility_key
    right = downstream.compatibility_key
    if left is not None and right is not None:
        compatibility_key = _merge_compatibility_keys(left, right)
    else:
        compatibility_key = left if left is not None else right
    return GPUExecutionRequirements(backend_keys, compatibility_key)


def _merge_compatibility_keys(
    upstream: GPUExecutionCompatibilityKey,
    downstream: GPUExecutionCompatibilityKey,
) -> GPUExecutionCompatibilityKey:
    fixed_fields = (
        "runtime_id",
        "worker_lifecycle",
        "min_workers",
        "initial_workers",
        "max_workers",
        "max_tasks_in_flight_per_worker",
        "max_udf_concurrency",
        "num_cpus",
        "num_gpus",
        "retry_policy",
    )
    for name in fixed_fields:
        left = getattr(upstream, name)
        right = getattr(downstream, name)
        if left != right:
            raise GPUExecutionRequirementsConflict(
                f"GPU execution requirement {name!r} differs: " f"{left!r} != {right!r}"
            )
    bootstrap = dict(upstream.actor_bootstrap)
    for key, value in downstream.actor_bootstrap:
        existing = bootstrap.get(key)
        if existing is not None and existing != value:
            raise GPUExecutionRequirementsConflict(
                f"GPU actor bootstrap requirement {key!r} differs"
            )
        bootstrap[key] = value
    optional_fields = (
        ("memory", "memory", False),
        ("custom_resources", "custom_resources", True),
        ("accelerator_type", "accelerator_type", False),
        ("scheduling_strategy", "scheduling_strategy", False),
        ("label_selector", "label_selector", True),
        ("runtime_env_serialized", "runtime_env", False),
    )
    merged = {
        field: _merge_optional_constraint(
            label,
            getattr(upstream, field),
            getattr(downstream, field),
            empty_is_unset=empty_is_unset,
        )
        for field, label, empty_is_unset in optional_fields
    }
    return replace(upstream, **merged, actor_bootstrap=tuple(sorted(bootstrap.items())))


def _merge_optional_constraint(
    name: str,
    upstream: Any,
    downstream: Any,
    *,
    empty_is_unset: bool = False,
) -> Any:
    upstream_unset = upstream is None or (empty_is_unset and not upstream)
    downstream_unset = downstream is None or (empty_is_unset and not downstream)
    if upstream_unset:
        return downstream
    if downstream_unset:
        return upstream
    if upstream != downstream:
        raise GPUExecutionRequirementsConflict(
            f"GPU execution requirement {name!r} differs: "
            f"{upstream!r} != {downstream!r}"
        )
    return upstream


@dataclass(frozen=True)
class RuntimeKey:
    runtime_id: str
    kind: str
    input_payload: PayloadKind
    output_payload: PayloadKind

    def __post_init__(self) -> None:
        if not self.runtime_id or not self.kind:
            raise ValueError("runtime_id and transform kind must be nonempty")


@dataclass(frozen=True, eq=False)
class GPUTransformSpec:
    kind: str
    input_payload: PayloadKind
    output_payload: PayloadKind
    config: Any = field(default=None, compare=False, repr=False)
    runtime_id: str = "ray-data-gpu-v1"

    def __post_init__(self) -> None:
        if not self.kind or not self.runtime_id:
            raise ValueError("transform kind and runtime_id must be nonempty")
        if not isinstance(self.input_payload, PayloadKind) or not isinstance(
            self.output_payload, PayloadKind
        ):
            raise TypeError("transform payloads must be PayloadKind instances")

    @property
    def runtime_key(self) -> RuntimeKey:
        return RuntimeKey(
            self.runtime_id,
            self.kind,
            self.input_payload,
            self.output_payload,
        )

    def canonical_contract(self) -> Tuple[Any, ...]:
        return (
            self.runtime_id,
            self.kind,
            self.input_payload,
            self.output_payload,
        )


@dataclass(frozen=True, eq=False)
class GPUOperatorSpec:
    transforms: Tuple[GPUTransformSpec, ...]
    execution_requirements: GPUExecutionRequirements
    input_requirements: Tuple[PropertyRequirement, ...] = ()
    output_facts: Tuple[PropertyFact, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transforms", tuple(self.transforms))
        object.__setattr__(self, "input_requirements", tuple(self.input_requirements))
        object.__setattr__(self, "output_facts", tuple(self.output_facts))
        if not self.transforms:
            raise ValueError("GPUOperatorSpec requires at least one transform")
        if not isinstance(self.execution_requirements, GPUExecutionRequirements):
            raise TypeError("execution_requirements must be GPUExecutionRequirements")
        if not all(
            isinstance(requirement, PropertyRequirement)
            for requirement in self.input_requirements
        ):
            raise TypeError("input_requirements must contain PropertyRequirement")
        if not all(isinstance(fact, PropertyFact) for fact in self.output_facts):
            raise TypeError("output_facts must contain PropertyFact")
        for upstream, downstream in zip(self.transforms, self.transforms[1:]):
            if upstream.output_payload != downstream.input_payload:
                raise ValueError(
                    "adjacent transform payloads do not match: "
                    f"{upstream.kind} -> {downstream.kind}"
                )
            if upstream.output_payload.materializable_boundary:
                raise ValueError(
                    "materializable payload cannot occur inside one operator spec: "
                    f"{upstream.output_payload.key}"
                )

    @property
    def input_payload(self) -> PayloadKind:
        return self.transforms[0].input_payload

    @property
    def output_payload(self) -> PayloadKind:
        return self.transforms[-1].output_payload

    @property
    def compatibility_key(self) -> GPUExecutionCompatibilityKey:
        key = self.execution_requirements.compatibility_key
        if key is None:
            raise ValueError("GPU execution requirements are not resolved")
        return key

    def canonical_contract(self) -> Tuple[Any, ...]:
        return (
            tuple(transform.canonical_contract() for transform in self.transforms),
            self.execution_requirements,
            self.input_requirements,
            self.output_facts,
        )


@dataclass(frozen=True)
class RangePreservation:
    input_requirements: Tuple[PropertyRequirement, ...]
    output_facts: Tuple[PropertyFact, ...]
    validation_transforms: Tuple[GPUTransformSpec, ...]


def preserve_range(
    facts: Sequence[PropertyFact],
    payload: PayloadKind,
    validation_config: Any,
) -> RangePreservation:
    fact = range_partitioning_fact(facts)
    if fact is None:
        return RangePreservation((), (), ())
    validation = GPUTransformSpec(
        kind="validate_range_ownership",
        input_payload=payload,
        output_payload=payload,
        config=validation_config,
    )
    return RangePreservation(
        (RangePartitioningRequired(fact.keys, fact.scheme_id),),
        (fact,),
        (validation,),
    )
