"""Canonical Ray ActorPool execution profiles for gate-2 GPU planning."""

from __future__ import annotations

import numbers
from dataclasses import replace
from typing import Any, Mapping, Optional

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


def canonical_gpu_execution_requirements(
    *,
    compute: Any,
    resources: Mapping[str, Any],
    data_context: DataContext,
    consumer: str,
) -> GPUExecutionRequirements:
    """Return the effective immutable profile for one GPU unary operator."""

    def unsupported(detail: str) -> None:
        raise GPUPlanningError(
            GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
            f"consumer={consumer}; {detail}",
        )

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

    remote = dict(resources)
    if remote.get("max_concurrency", 1) != 1:
        unsupported("max_concurrency must be 1")
    num_gpus = remote.get("num_gpus")
    if (
        not isinstance(num_gpus, numbers.Real)
        or isinstance(num_gpus, bool)
        or float(num_gpus) != 1.0
    ):
        unsupported("GPU MapBatches must request exactly num_gpus=1")

    supported_options = {
        "num_cpus",
        "num_gpus",
        "memory",
        "resources",
        "accelerator_type",
        "runtime_env",
        "scheduling_strategy",
        "label_selector",
        "max_concurrency",
        "max_restarts",
        "max_task_retries",
    }
    unknown = set(remote) - supported_options
    if unknown:
        unsupported(f"unsupported GPU actor options: {sorted(unknown)!r}")

    scheduling = remote.get("scheduling_strategy")
    if scheduling is not None and not isinstance(scheduling, str):
        unsupported("only canonical string scheduling strategies are supported")
    labels = remote.get("label_selector") or {}
    if not (
        isinstance(labels, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in labels.items())
    ):
        unsupported("label_selector must map strings to strings")
    custom = remote.get("resources") or {}
    if not (
        isinstance(custom, dict)
        and all(
            isinstance(k, str)
            and isinstance(v, numbers.Real)
            and not isinstance(v, bool)
            for k, v in custom.items()
        )
    ):
        unsupported("custom resources must map strings to numeric values")
    memory = remote.get("memory")
    if memory is not None and (
        not isinstance(memory, numbers.Real) or isinstance(memory, bool)
    ):
        unsupported("memory must be numeric")

    max_restarts = remote.get("max_restarts", -1)
    max_task_retries = remote.get("max_task_retries", -1 if max_restarts != 0 else 0)
    actor_errors = data_context.actor_task_retry_on_errors
    if not isinstance(actor_errors, bool):
        actor_errors = tuple(actor_errors)
    map_errors = data_context.retried_map_errors
    if not isinstance(map_errors, bool):
        map_errors = tuple(map_errors)
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
        raise AssertionError("unreachable")

    return GPUExecutionRequirements(
        backend_name="actor_pool",
        compatibility_key=GPUExecutionCompatibilityKey(
            min_workers=minimum,
            initial_workers=initial,
            max_workers=maximum,
            max_tasks_in_flight_per_worker=1,
            max_udf_concurrency=1,
            num_cpus=float(remote.get("num_cpus", 1)),
            num_gpus=1.0,
            memory=float(memory) if memory is not None else None,
            custom_resources=tuple(
                sorted((key, float(value)) for key, value in custom.items())
            ),
            accelerator_type=remote.get("accelerator_type"),
            scheduling_strategy=scheduling,
            label_selector=tuple(sorted(labels.items())),
            runtime_env_serialized=_runtime_env_serialized(remote.get("runtime_env")),
            retry_policy=retry_policy,
            actor_bootstrap=(("allocator", "rmm-pool-v1"),),
        ),
    )


def default_gpu_execution_requirements(
    data_context: DataContext,
) -> GPUExecutionRequirements:
    """Profile for the valid Read -> MGP case with no MapBatches profile."""

    return canonical_gpu_execution_requirements(
        compute=ActorPoolStrategy(),
        resources={"num_cpus": 1, "num_gpus": 1},
        data_context=data_context,
        consumer="Read->MapGroupPartitions",
    )


def reconcile_read_remote_args(
    requirements: GPUExecutionRequirements,
    read_remote_args: Mapping[str, Any],
    *,
    consumer: str,
) -> GPUExecutionRequirements:
    """Fold specialized Read placement constraints into the component profile."""

    unknown = set(read_remote_args) - {"scheduling_strategy", "label_selector"}
    if unknown:
        raise GPUPlanningError(
            GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
            f"consumer={consumer}; unsupported specialized Read remote args: "
            f"{sorted(unknown)!r}",
        )
    key = requirements.compatibility_key
    scheduling = read_remote_args.get("scheduling_strategy")
    if scheduling is not None and not isinstance(scheduling, str):
        raise GPUPlanningError(
            GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
            f"consumer={consumer}; Read scheduling_strategy must be a string",
        )
    if (
        scheduling is not None
        and key.scheduling_strategy is not None
        and scheduling != key.scheduling_strategy
    ):
        raise GPUPlanningError(
            GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS,
            f"consumer={consumer}; Read and GPU MapBatches specify conflicting "
            f"scheduling_strategy values: {scheduling!r} != "
            f"{key.scheduling_strategy!r}",
        )

    raw_labels = read_remote_args.get("label_selector")
    if raw_labels is not None and not (
        isinstance(raw_labels, dict)
        and all(
            isinstance(label, str) and isinstance(value, str)
            for label, value in raw_labels.items()
        )
    ):
        raise GPUPlanningError(
            GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
            f"consumer={consumer}; Read label_selector must map strings to strings",
        )
    read_labels = tuple(sorted((raw_labels or {}).items()))
    if read_labels and key.label_selector and read_labels != key.label_selector:
        raise GPUPlanningError(
            GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS,
            f"consumer={consumer}; Read and GPU MapBatches specify conflicting "
            f"label_selector values: {read_labels!r} != {key.label_selector!r}",
        )

    return replace(
        requirements,
        compatibility_key=replace(
            key,
            scheduling_strategy=(
                key.scheduling_strategy if scheduling is None else scheduling
            ),
            label_selector=key.label_selector or read_labels,
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


__all__ = [
    "canonical_gpu_execution_requirements",
    "default_gpu_execution_requirements",
    "reconcile_read_remote_args",
]
