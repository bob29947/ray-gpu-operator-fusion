"""Anchored provider for the Parquet/cuDF map_group_partitions GPU region."""

from __future__ import annotations

import functools
import inspect
import logging
import numbers
from typing import Any, Mapping

from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.execution.gpu import (
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    create_gpu_operator,
)
from ray.data._internal.execution.gpu.transforms.boundary import descriptor_bundles
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators import (
    MapBatches,
    MapGroupPartitions,
    Read,
)
from ray.data._internal.planner.gpu_lowering import (
    GPULogicalLowererRegistry,
    GPULoweringContext,
)
from ray.data._internal.planner.gpu_lowering.map_batches_lowerer import (
    GPUMapBatchesLowerer,
)
from ray.data._internal.planner.gpu_lowering.map_group_partitions_lowerer import (
    GPUMapGroupPartitionsLowerer,
)
from ray.data._internal.planner.gpu_lowering.parquet_read_lowerer import (
    ParquetRangeReadLowerer,
)
from ray.data._internal.planner.gpu_plan_provider import (
    GPUPlanFragment,
    GPUProviderAccepted,
    GPUProviderNotApplicable,
    GPUProviderRejected,
    GPUProviderResult,
)
from ray.data._internal.planner.parquet_cudf_range_planner import (
    ParquetCudfPlanningError,
    plan_parquet_ranges,
)
from ray.data._internal.planner.parquet_cudf_read_recognizer import (
    recognize_parquet_cudf_read,
)
from ray.data.context import DataContext

logger = logging.getLogger(__name__)


class GPUProviderPlanningError(ValueError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise GPUProviderPlanningError(reason)


def _is_synchronous(callable_: Any) -> bool:
    if not callable(callable_):
        return False
    if inspect.isroutine(callable_) or isinstance(callable_, functools.partial):
        function = callable_
    else:
        function = callable_.__call__
    return not inspect.iscoroutinefunction(function) and not inspect.isasyncgenfunction(
        function
    )


def _group_keys(op: MapGroupPartitions) -> tuple[str, ...]:
    keys = (op.key,) if isinstance(op.key, str) else tuple(op.key or ())
    _require(
        bool(keys)
        and len(keys) == len(set(keys))
        and all(isinstance(key, str) and key for key in keys),
        "map_group_partitions requires one or more unique nonempty string group keys",
    )
    return keys


def _discover_tfm_region(
    anchor: MapGroupPartitions,
    consumer_counts: Mapping[LogicalOperator, int],
    registry: GPULogicalLowererRegistry,
) -> tuple[Read, MapBatches, MapGroupPartitions]:
    _require(
        consumer_counts.get(anchor, 0) <= 1,
        "GPU V1 cannot consume map_group_partitions shared by multiple branches",
    )
    discovered: list[LogicalOperator] = []
    cursor: LogicalOperator = anchor
    while registry.get(cursor) is not None:
        discovered.append(cursor)
        if type(cursor) is Read:
            break
        _require(
            len(cursor.input_dependencies) == 1,
            f"{cursor.name} must have exactly one upstream operator",
        )
        child = cursor.input_dependencies[0]
        _require(
            consumer_counts.get(child, 0) == 1,
            "GPU V1 cannot consume a logical operator shared by multiple branches",
        )
        cursor = child

    _require(
        len(discovered) == 3
        and type(discovered[0]) is MapGroupPartitions
        and type(discovered[1]) is MapBatches
        and type(discovered[2]) is Read,
        "map_group_partitions V1 requires Read -> MapBatches -> "
        "MapGroupPartitions with no intervening frontier",
    )
    return discovered[2], discovered[1], discovered[0]  # type: ignore[return-value]


def _pool_and_range_count(
    compute: Any,
    requested_partitions: Any,
) -> tuple[int, int, int, int]:
    _require(
        isinstance(compute, ActorPoolStrategy),
        "the GPU map_batches operation must use ActorPoolStrategy",
    )
    _require(
        compute.max_tasks_in_flight_per_actor in (None, 1),
        "GPU V1 requires max_tasks_in_flight_per_actor=1",
    )
    _require(
        not compute.enable_true_multi_threading,
        "GPU V1 does not support enable_true_multi_threading=True",
    )
    minimum = int(compute.min_size)
    initial = int(compute.initial_size)
    if compute.max_size == float("inf"):
        from ray.data._internal.execution.operators.hash_shuffle import (
            _get_total_cluster_resources,
        )

        maximum = int(_get_total_cluster_resources().gpu)
        _require(maximum >= minimum and maximum > 0, "no GPU workers are available")
        minimum = min(minimum, maximum)
        initial = min(initial, maximum)
    else:
        _require(
            isinstance(compute.max_size, numbers.Integral)
            and not isinstance(compute.max_size, bool)
            and compute.max_size > 0,
            "ActorPoolStrategy.max_size must be a positive integer",
        )
        maximum = int(compute.max_size)

    if requested_partitions is None:
        target_ranges = 2 * maximum
    else:
        _require(
            isinstance(requested_partitions, numbers.Integral)
            and not isinstance(requested_partitions, bool)
            and requested_partitions > 0,
            "groupby num_partitions must be a positive integer",
        )
        target_ranges = int(requested_partitions)
    return minimum, initial, maximum, target_ranges


def _runtime_env_serialized(value: Any) -> str | None:
    if value is None:
        return None
    from ray.runtime_env import RuntimeEnv

    if isinstance(value, RuntimeEnv):
        runtime_env = value
    elif isinstance(value, dict):
        runtime_env = RuntimeEnv(**value)
    else:
        raise GPUProviderPlanningError(
            "runtime_env must be a dict or RuntimeEnv for GPU fusion"
        )
    return runtime_env.serialize()


def _execution_requirements(
    map_op: MapBatches,
    read_op: Read,
    data_context: DataContext,
    *,
    source_kind: str,
    region: str | None,
    min_workers: int,
    initial_workers: int,
    max_workers: int,
) -> GPUExecutionRequirements:
    resources = dict(map_op.ray_remote_args)
    retry_keys = {
        "max_restarts",
        "max_task_retries",
        "max_retries",
        "retry_exceptions",
    }
    _require(
        not retry_keys.intersection(resources),
        "custom map_batches retry options are unsupported",
    )
    lifecycle_keys = {"name", "namespace", "lifetime", "get_if_exists"}
    _require(
        not lifecycle_keys.intersection(resources),
        "named or detached GPU actor lifecycle options are unsupported",
    )
    _require(resources.get("max_concurrency", 1) == 1, "max_concurrency must be 1")
    num_gpus = resources.get("num_gpus")
    _require(
        isinstance(num_gpus, numbers.Real)
        and not isinstance(num_gpus, bool)
        and float(num_gpus) == 1.0,
        "the GPU map_batches operation must request exactly num_gpus=1",
    )

    for option in ("scheduling_strategy", "label_selector"):
        if option not in read_op.ray_remote_args:
            continue
        _require(
            option not in resources
            or resources[option] == read_op.ray_remote_args[option],
            f"read_parquet and map_batches specify conflicting {option}",
        )
        resources[option] = read_op.ray_remote_args[option]

    supported = {
        "num_cpus",
        "num_gpus",
        "memory",
        "resources",
        "accelerator_type",
        "runtime_env",
        "scheduling_strategy",
        "label_selector",
        "max_concurrency",
    }
    unknown = set(resources) - supported
    _require(not unknown, f"unsupported GPU actor options: {sorted(unknown)!r}")

    scheduling = resources.get("scheduling_strategy")
    _require(
        scheduling is None or isinstance(scheduling, str),
        "GPU V1 supports only canonical string scheduling strategies",
    )
    labels = resources.get("label_selector") or {}
    _require(
        isinstance(labels, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in labels.items()),
        "label_selector must map strings to strings",
    )
    custom = resources.get("resources") or {}
    _require(
        isinstance(custom, dict)
        and all(
            isinstance(k, str)
            and isinstance(v, numbers.Real)
            and not isinstance(v, bool)
            for k, v in custom.items()
        ),
        "custom resources must map strings to numeric values",
    )

    return GPUExecutionRequirements(
        backend_name="actor_pool",
        compatibility_key=GPUExecutionCompatibilityKey(
            min_workers=min_workers,
            initial_workers=initial_workers,
            max_workers=max_workers,
            max_tasks_in_flight_per_worker=1,
            max_udf_concurrency=1,
            num_cpus=float(resources.get("num_cpus", 1)),
            num_gpus=1.0,
            memory=(
                float(resources["memory"])
                if resources.get("memory") is not None
                else None
            ),
            custom_resources=tuple(sorted((k, float(v)) for k, v in custom.items())),
            accelerator_type=resources.get("accelerator_type"),
            scheduling_strategy=scheduling,
            label_selector=tuple(sorted(labels.items())),
            runtime_env_serialized=_runtime_env_serialized(
                resources.get("runtime_env")
            ),
            retry_policy="none",
            actor_bootstrap=tuple(
                sorted(
                    {
                        "allocator": "rmm-pool-v1",
                        "source_kind": source_kind,
                        "region": region or "",
                    }.items()
                )
            ),
        ),
    )


def _validate_context(data_context: DataContext) -> None:
    _require(
        not data_context.use_datasource_v2,
        "GPU V1 requires use_datasource_v2=False",
    )
    _require(data_context.checkpoint_config is None, "checkpointing is unsupported")
    _require(data_context.max_errored_blocks == 0, "max_errored_blocks must be 0")
    _require(not data_context.retried_map_errors, "retried_map_errors must be disabled")
    _require(
        not data_context.actor_task_retry_on_errors,
        "actor_task_retry_on_errors must be disabled",
    )
    _require(
        not data_context.actor_init_retry_on_errors,
        "actor_init_retry_on_errors must be disabled",
    )
    _require(
        data_context.max_tasks_in_flight_per_actor in (None, 1),
        "DataContext.max_tasks_in_flight_per_actor must be None or 1",
    )


class ParquetCudfMapGroupPartitionsProvider:
    def __init__(self) -> None:
        registry = GPULogicalLowererRegistry()
        registry.register(Read, ParquetRangeReadLowerer())
        registry.register(MapBatches, GPUMapBatchesLowerer())
        registry.register(MapGroupPartitions, GPUMapGroupPartitionsLowerer())
        self._registry = registry

    def plan(
        self,
        logical_op: LogicalOperator,
        data_context: DataContext,
        consumer_counts: Mapping[LogicalOperator, int],
    ) -> GPUProviderResult:
        if type(logical_op) is not MapGroupPartitions:
            return GPUProviderNotApplicable()
        try:
            return self._plan(logical_op, data_context, consumer_counts)
        except (GPUProviderPlanningError, ParquetCudfPlanningError) as error:
            return GPUProviderRejected(str(error))

    def _plan(
        self,
        anchor: MapGroupPartitions,
        data_context: DataContext,
        consumer_counts: Mapping[LogicalOperator, int],
    ) -> GPUProviderAccepted:
        _validate_context(data_context)
        read_op, map_op, anchor = _discover_tfm_region(
            anchor, consumer_counts, self._registry
        )
        group_keys = _group_keys(anchor)
        _require(
            not inspect.isclass(anchor.fn) and _is_synchronous(anchor.fn),
            "map_group_partitions requires a synchronous function, not a class",
        )
        properties = map_op.udf_properties
        _require(
            properties is not None and properties.preserves_partitioning,
            "map_group_partitions requires the upstream map_batches to pass "
            "UDFProperties(preserves_partitioning=True)",
        )
        _require(
            inspect.isclass(map_op.fn)
            and _is_synchronous(map_op.fn)
            and map_op.batch_format == "cudf"
            and isinstance(map_op.batch_size, numbers.Integral)
            and not isinstance(map_op.batch_size, bool)
            and map_op.batch_size > 0
            and map_op.per_block_limit is None
            and map_op.ray_remote_args_fn is None,
            "the upstream GPU map_batches must use a synchronous callable class, "
            "batch_format='cudf', a positive integer batch_size, and no dynamic "
            "remote arguments or per-block limit",
        )

        recognition = recognize_parquet_cudf_read(read_op, group_keys)
        _require(
            recognition.accepted,
            "read_parquet is not eligible for direct cuDF execution "
            f"({recognition.reason})",
        )
        descriptor = recognition.descriptor
        assert descriptor is not None
        minimum, initial, maximum, target_ranges = _pool_and_range_count(
            map_op.compute, anchor.num_partitions
        )
        target_bytes = data_context.target_max_block_size
        _require(
            isinstance(target_bytes, numbers.Integral)
            and not isinstance(target_bytes, bool)
            and target_bytes > 0,
            "DataContext.target_max_block_size must be positive",
        )
        retried_io_errors = tuple(data_context.retried_io_errors)
        _require(
            all(isinstance(pattern, str) for pattern in retried_io_errors),
            "DataContext.retried_io_errors must contain strings",
        )
        requirements = _execution_requirements(
            map_op,
            read_op,
            data_context,
            source_kind=descriptor.source_kind,
            region=descriptor.region,
            min_workers=minimum,
            initial_workers=initial,
            max_workers=maximum,
        )
        range_plan = plan_parquet_ranges(
            descriptor,
            range_key=group_keys[0],
            target_ranges=target_ranges,
            num_workers=maximum,
        )
        region_id = "parquet-cudf-mgp:" + range_plan.scheme_id
        lowering_context = GPULoweringContext(
            region_id=region_id,
            group_keys=group_keys,
            read_descriptor=descriptor,
            range_plan=range_plan,
            execution_requirements=requirements,
            target_max_block_size=int(target_bytes),
            retried_io_errors=retried_io_errors,
        )

        input_op = InputDataBuffer(
            data_context,
            input_data_factory=lambda _target_size: descriptor_bundles(range_plan.work),
        )
        read_lowered = self._registry.lower(read_op, lowering_context)
        read_gpu_op = create_gpu_operator(
            read_lowered.spec,
            input_op,
            data_context,
            read_lowered.creation_options,
        )
        map_lowered = self._registry.lower(map_op, lowering_context)
        map_gpu_op = create_gpu_operator(
            map_lowered.spec,
            read_gpu_op,
            data_context,
            map_lowered.creation_options,
        )
        grouped_lowered = self._registry.lower(anchor, lowering_context)
        grouped_gpu_op = create_gpu_operator(
            grouped_lowered.spec,
            map_gpu_op,
            data_context,
            grouped_lowered.creation_options,
        )

        input_op.set_logical_operators(read_op)
        read_gpu_op.set_logical_operators(read_op)
        map_gpu_op.set_logical_operators(map_op)
        grouped_gpu_op.set_logical_operators(anchor)
        new_operators = (input_op, read_gpu_op, map_gpu_op, grouped_gpu_op)
        op_map = {
            input_op: read_op,
            read_gpu_op: read_op,
            map_gpu_op: map_op,
            grouped_gpu_op: anchor,
        }
        logger.info(
            "Planned modular GPU region %s with %d ranges and %d workers",
            region_id,
            len(range_plan.work),
            maximum,
        )
        return GPUProviderAccepted(
            GPUPlanFragment(
                root=grouped_gpu_op,
                new_operators=new_operators,
                consumed_logical_operators=(read_op, map_op, anchor),
                op_map=op_map,
                region_id=region_id,
            )
        )
