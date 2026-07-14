"""Canonical Ray ActorPool execution profile for required GPU fusion."""

from __future__ import annotations

import numbers
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional

from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUStageRetryPolicy,
)
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)
from ray.data.context import DataContext


_UNSUPPORTED = GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
_CONFLICTING = GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS


def _unsupported(consumer: str, detail: str) -> GPUPlanningError:
    return GPUPlanningError(_UNSUPPORTED, f"consumer={consumer}; {detail}")


def _valid_mapping(value: Any, valid_value: Callable[[Any], bool]) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str) and valid_value(item) for key, item in value.items()
    )


def canonical_gpu_execution_requirements(
    *,
    compute: Any,
    resources: Mapping[str, Any],
    data_context: DataContext,
    consumer: str,
) -> GPUExecutionRequirements:
    """Return the effective immutable profile for one GPU unary operator."""

    def unsupported(detail: str) -> None:
        raise _unsupported(consumer, detail)

    if not isinstance(compute, ActorPoolStrategy):
        unsupported("GPU execution requires ActorPoolStrategy")
    if compute.max_tasks_in_flight_per_actor not in (None, 1):
        unsupported("max_tasks_in_flight_per_actor must be 1")
    if compute.enable_true_multi_threading:
        unsupported("enable_true_multi_threading=True is unsupported")
    minimum = int(compute.min_size)
    initial = int(compute.initial_size)
    if compute.max_size == float("inf"):
        from ray.data._internal.execution.operators.hash_shuffle import (
            _get_total_cluster_resources,
        )

        maximum = int(_get_total_cluster_resources().gpu)
        if maximum <= 0:
            unsupported("no GPU workers are available")
        minimum = min(minimum, maximum)
        initial = min(initial, maximum)
    elif (
        not isinstance(compute.max_size, numbers.Integral)
        or isinstance(compute.max_size, bool)
        or compute.max_size <= 0
    ):
        unsupported("ActorPoolStrategy.max_size must be a positive integer")
    else:
        maximum = int(compute.max_size)

    remaining = dict(resources)
    max_concurrency = remaining.pop("max_concurrency", 1)
    num_gpus = remaining.pop("num_gpus", None)
    num_cpus = remaining.pop("num_cpus", 1)
    memory = remaining.pop("memory", None)
    custom = remaining.pop("resources", None) or {}
    accelerator_type = remaining.pop("accelerator_type", None)
    runtime_env = remaining.pop("runtime_env", None)
    scheduling = remaining.pop("scheduling_strategy", None)
    labels = remaining.pop("label_selector", None) or {}
    max_restarts = remaining.pop("max_restarts", -1)
    max_task_retries = remaining.pop("max_task_retries", -1 if max_restarts != 0 else 0)
    if max_concurrency != 1:
        unsupported("max_concurrency must be 1")
    if (
        not isinstance(num_gpus, numbers.Real)
        or isinstance(num_gpus, bool)
        or float(num_gpus) != 1.0
    ):
        unsupported("GPU MapBatches must request exactly num_gpus=1")
    if remaining:
        unsupported(f"unsupported GPU actor options: {sorted(remaining)!r}")

    if scheduling is not None and not isinstance(scheduling, str):
        unsupported("only canonical string scheduling strategies are supported")
    if not _valid_mapping(labels, lambda value: isinstance(value, str)):
        unsupported("label_selector must map strings to strings")
    if not _valid_mapping(
        custom,
        lambda value: isinstance(value, numbers.Real) and not isinstance(value, bool),
    ):
        unsupported("custom resources must map strings to numeric values")
    if memory is not None and (
        not isinstance(memory, numbers.Real) or isinstance(memory, bool)
    ):
        unsupported("memory must be numeric")

    actor_errors = data_context.actor_task_retry_on_errors
    actor_errors = (
        actor_errors if isinstance(actor_errors, bool) else tuple(actor_errors)
    )
    map_errors = data_context.retried_map_errors
    map_errors = map_errors if isinstance(map_errors, bool) else tuple(map_errors)
    try:
        retry_policy = GPUStageRetryPolicy(
            actor_max_restarts=max_restarts,
            actor_max_task_retries=max_task_retries,
            actor_task_retry_on_errors=actor_errors,
            actor_init_retry_on_errors=data_context.actor_init_retry_on_errors,
            actor_init_max_retries=data_context.actor_init_max_retries,
            retried_map_errors=map_errors,
            max_map_retries=data_context.max_map_retries,
            retried_io_errors=tuple(data_context.retried_io_errors),
        )
    except (TypeError, ValueError) as error:
        unsupported(f"invalid GPU retry policy: {error}")

    return GPUExecutionRequirements(
        backend_keys=frozenset(("actor_pool",)),
        compatibility_key=GPUExecutionCompatibilityKey(
            min_workers=minimum,
            initial_workers=initial,
            max_workers=maximum,
            max_tasks_in_flight_per_worker=1,
            max_udf_concurrency=1,
            num_cpus=float(num_cpus),
            num_gpus=1.0,
            memory=float(memory) if memory is not None else None,
            custom_resources=tuple(
                sorted((key, float(value)) for key, value in custom.items())
            ),
            accelerator_type=accelerator_type,
            scheduling_strategy=scheduling,
            label_selector=tuple(sorted(labels.items())),
            runtime_env_serialized=_runtime_env_serialized(runtime_env),
            retry_policy=retry_policy,
            actor_bootstrap=(("allocator", "rmm-pool-v1"),),
        ),
    )


def _reconcile_read_constraint(
    name: str,
    read_value: Any,
    map_value: Any,
    consumer: str,
) -> Any:
    if read_value is not None and map_value is not None and read_value != map_value:
        raise GPUPlanningError(
            _CONFLICTING,
            f"consumer={consumer}; Read and GPU MapBatches specify conflicting "
            f"{name} values: {read_value!r} != {map_value!r}",
        )
    return map_value if read_value is None else read_value


def reconcile_read_remote_args(
    requirements: GPUExecutionRequirements,
    read_remote_args: Mapping[str, Any],
    *,
    consumer: str,
) -> GPUExecutionRequirements:
    """Fold specialized Read placement constraints into the component profile."""

    remaining = dict(read_remote_args)
    scheduling = remaining.pop("scheduling_strategy", None)
    raw_labels = remaining.pop("label_selector", None)
    if remaining:
        raise _unsupported(
            consumer,
            f"unsupported specialized Read remote args: {sorted(remaining)!r}",
        )
    key = requirements.compatibility_key
    if key is None:
        raise _unsupported(
            consumer,
            "specialized Read requires a concrete GPU execution profile",
        )

    if scheduling is not None and not isinstance(scheduling, str):
        raise _unsupported(consumer, "Read scheduling_strategy must be a string")
    scheduling = _reconcile_read_constraint(
        "scheduling_strategy",
        scheduling,
        key.scheduling_strategy,
        consumer,
    )

    if raw_labels is not None and not _valid_mapping(
        raw_labels, lambda value: isinstance(value, str)
    ):
        raise _unsupported(
            consumer,
            "Read label_selector must map strings to strings",
        )
    read_labels = tuple(sorted((raw_labels or {}).items())) or None
    labels = _reconcile_read_constraint(
        "label_selector",
        read_labels,
        key.label_selector or None,
        consumer,
    )
    return replace(
        requirements,
        compatibility_key=replace(
            key,
            scheduling_strategy=scheduling,
            label_selector=labels or (),
        ),
    )


def _runtime_env_serialized(value: Any) -> Optional[str]:
    if value is None:
        return None
    from ray.runtime_env import RuntimeEnv

    if isinstance(value, RuntimeEnv):
        runtime_env = value
    elif isinstance(value, dict):
        runtime_env = RuntimeEnv(**value)
    else:
        raise GPUPlanningError(
            GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
            "runtime_env must be a dict or RuntimeEnv for GPU execution",
        )
    return runtime_env.serialize()


__all__ = ["canonical_gpu_execution_requirements", "reconcile_read_remote_args"]
