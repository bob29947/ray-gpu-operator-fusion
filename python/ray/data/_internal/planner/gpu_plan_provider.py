"""Planning for the required-fusion Parquet/cuDF GPU pipeline.

The planner deliberately supports one shape in V1::

    Read -> MapBatches -> MapGroupPartitions

It lowers each logical operator to an independent, non-executable physical
operator.  It never composes specifications and never creates an execution
backend.  The physical GPU fusion rule owns both of those operations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Mapping, Optional, Tuple

from ray.data._internal.execution.gpu.operator import PlannedGPUOperator
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators import MapBatches, MapGroupPartitions, Read
from ray.data._internal.planner.gpu_execution_profile import (
    reconcile_read_remote_args,
)
from ray.data._internal.planner.gpu_lowering.interfaces import GPULoweringContext
from ray.data._internal.planner.gpu_lowering.map_batches_lowerer import (
    GPUMapBatchesLowerer,
)
from ray.data._internal.planner.gpu_lowering.map_group_partitions_lowerer import (
    GPUMapGroupPartitionsLowerer,
)
from ray.data._internal.planner.gpu_lowering.parquet_read_lowerer import GPUReadLowerer
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)
from ray.data._internal.planner.gpu_read_source_provider import (
    GPUReadSourceDemand,
    GPUReadSourceProvider,
    ParquetRangeGPUReadProvider,
)
from ray.data.context import DataContext


@dataclass(frozen=True)
class GPUPlannedStage:
    root: PhysicalOperator
    op_map: Mapping[PhysicalOperator, LogicalOperator]


def create_gpu_required_fusion_planner(
    root: LogicalOperator,
    data_context: DataContext,
    *,
    source_provider: Optional[GPUReadSourceProvider] = None,
) -> "GPURequiredFusionPlanner":
    """Create a plan-local exact-shape GPU planner.

    ``data_context`` is accepted to preserve the existing planner hook.  No
    footer reads, range planning, spec composition, or backend construction
    occurs until :meth:`GPURequiredFusionPlanner.try_plan` sees an MGP anchor.
    """

    del data_context
    return GPURequiredFusionPlanner(
        root,
        source_provider=source_provider or ParquetRangeGPUReadProvider(),
    )


class GPURequiredFusionPlanner:
    """Lower exactly one Read/MapBatches/MGP component to deferred GPU nodes."""

    def __init__(
        self,
        root: LogicalOperator,
        *,
        source_provider: GPUReadSourceProvider,
    ) -> None:
        operators = tuple(_post_order_logical(root))
        self._consumer_counts = _logical_consumer_counts(operators)
        self._logical_node_ids = {
            op: f"logical-{ordinal:04d}-{type(op).__name__}"
            for ordinal, op in enumerate(operators)
        }
        self._source_provider = source_provider
        self._read_lowerer = GPUReadLowerer()
        self._map_lowerer = GPUMapBatchesLowerer()
        self._mgp_lowerer = GPUMapGroupPartitionsLowerer()

    def try_plan(
        self,
        logical_op: LogicalOperator,
        data_context: DataContext,
    ) -> Optional[GPUPlannedStage]:
        if type(logical_op) is not MapGroupPartitions:
            return None
        return self._plan_exact_component(logical_op, data_context)

    def _plan_exact_component(
        self,
        anchor: MapGroupPartitions,
        data_context: DataContext,
    ) -> GPUPlannedStage:
        if data_context.checkpoint_config is not None:
            _unsupported(anchor, "checkpointing is not supported")

        read_op, map_op = self._discover_exact_shape(anchor)
        read_analysis = self._read_lowerer.analyze(read_op, data_context)
        map_analysis = self._map_lowerer.analyze(map_op, data_context)
        mgp_analysis = self._mgp_lowerer.analyze(anchor, data_context)
        for op, analysis in (
            (read_op, read_analysis),
            (map_op, map_analysis),
            (anchor, mgp_analysis),
        ):
            if not analysis.accepted:
                _unsupported(
                    anchor,
                    f"operator={_op_label(op)}; capability={analysis.reason}",
                )

        if not map_analysis.preserves_partitioning:
            _unsupported(
                anchor,
                "the cuDF MapBatches UDF must declare "
                "UDFProperties(preserves_partitioning=True)",
            )
        requirements = map_analysis.execution_requirements
        if requirements is None:
            raise AssertionError("GPU MapBatches analysis must declare execution")
        requirements = reconcile_read_remote_args(
            requirements,
            read_op.ray_remote_args,
            consumer=_op_label(anchor),
        )

        group_keys = mgp_analysis.partition_keys
        target_ranges = (
            2 * requirements.compatibility_key.max_workers
            if mgp_analysis.requested_partitions is None
            else mgp_analysis.requested_partitions
        )
        assert target_ranges is not None
        demand = GPUReadSourceDemand(
            group_keys=group_keys,
            target_ranges=target_ranges,
            num_workers=requirements.compatibility_key.max_workers,
        )
        support = self._source_provider.supports(read_op, demand)
        if not support.accepted:
            _unsupported(
                anchor,
                "the Read is not supported by the required Parquet range source "
                f"({support.reason})",
            )

        target_bytes = data_context.target_max_block_size
        if (
            not isinstance(target_bytes, int)
            or isinstance(target_bytes, bool)
            or target_bytes <= 0
        ):
            _unsupported(anchor, "DataContext.target_max_block_size must be positive")

        logical_ops: Tuple[LogicalOperator, ...] = (read_op, map_op, anchor)
        provenance = tuple(self._logical_node_ids[op] for op in logical_ops)
        fusion_group_id = f"gpu-required:{provenance[0]}->{provenance[-1]}"
        context = GPULoweringContext(
            stage_id=fusion_group_id,
            logical_node_id=provenance[0],
            group_keys=group_keys,
            source_provider=self._source_provider,
            execution_requirements=requirements,
            target_max_block_size=int(target_bytes),
            retried_io_errors=tuple(data_context.retried_io_errors),
            target_ranges=target_ranges,
            num_workers=requirements.compatibility_key.max_workers,
        )

        read_lowered = self._read_lowerer.lower(read_op, context)
        read_spec = _bind_required_region(
            read_lowered.spec,
            fusion_group_id=fusion_group_id,
            required_provenance=provenance,
            owns_input_parallelism=True,
        )
        map_context = read_lowered.downstream_context.for_operator(
            logical_node_id=provenance[1],
        )
        map_lowered = self._map_lowerer.lower(map_op, map_context)
        map_spec = _bind_required_region(
            map_lowered.spec,
            fusion_group_id=fusion_group_id,
            required_provenance=provenance,
        )
        mgp_context = map_lowered.downstream_context.for_operator(
            logical_node_id=provenance[2],
            group_keys=group_keys,
        )
        mgp_lowered = self._mgp_lowerer.lower(anchor, mgp_context)
        mgp_spec = _bind_required_region(
            mgp_lowered.spec,
            fusion_group_id=fusion_group_id,
            required_provenance=provenance,
            uses_terminal_output_adapter=True,
        )

        artifacts = read_lowered.source_artifacts
        if artifacts is None:
            _unsupported(anchor, "GPU Read lowering returned no source work")
        source_input = InputDataBuffer(
            data_context,
            input_data_factory=artifacts.input_data_factory,
        )
        read_node = PlannedGPUOperator(
            read_spec,
            source_input,
            data_context,
            read_lowered.creation_options,
        )
        map_node = PlannedGPUOperator(
            map_spec,
            read_node,
            data_context,
            map_lowered.creation_options,
        )
        mgp_node = PlannedGPUOperator(
            mgp_spec,
            map_node,
            data_context,
            mgp_lowered.creation_options,
        )

        source_input.set_logical_operators(read_op)
        for physical, logical in (
            (read_node, read_op),
            (map_node, map_op),
            (mgp_node, anchor),
        ):
            physical.set_logical_operators(logical)
        return GPUPlannedStage(
            root=mgp_node,
            op_map={
                source_input: read_op,
                read_node: read_op,
                map_node: map_op,
                mgp_node: anchor,
            },
        )

    def _discover_exact_shape(
        self,
        anchor: MapGroupPartitions,
    ) -> Tuple[Read, MapBatches]:
        if len(anchor.input_dependencies) != 1:
            _unsupported(anchor, "MapGroupPartitions must have exactly one input")
        map_op = anchor.input_dependencies[0]
        if type(map_op) is not MapBatches or len(map_op.input_dependencies) != 1:
            _unsupported(
                anchor,
                "V1 requires exactly Read -> one cuDF MapBatches -> "
                "MapGroupPartitions",
            )
        read_op = map_op.input_dependencies[0]
        if type(read_op) is not Read:
            _unsupported(
                anchor,
                "V1 requires exactly Read -> one cuDF MapBatches -> "
                "MapGroupPartitions",
            )
        for op in (read_op, map_op):
            if self._consumer_counts.get(op, 0) != 1:
                _unsupported(
                    anchor,
                    f"{_op_label(op)} is shared by multiple logical consumers",
                )
        if self._consumer_counts.get(anchor, 0) > 1:
            _unsupported(anchor, "MapGroupPartitions is shared by multiple consumers")
        return read_op, map_op


def _bind_required_region(
    spec,
    *,
    fusion_group_id: str,
    required_provenance: Tuple[str, ...],
    owns_input_parallelism: bool = False,
    uses_terminal_output_adapter: bool = False,
):
    return replace(
        spec,
        fusion_group_id=fusion_group_id,
        required_logical_provenance=required_provenance,
        requires_complete_fusion=True,
        owns_input_parallelism=owns_input_parallelism,
        uses_terminal_output_adapter=uses_terminal_output_adapter,
    )


def _post_order_logical(root: LogicalOperator) -> Iterable[LogicalOperator]:
    visited: set[LogicalOperator] = set()

    def visit(op: LogicalOperator):
        if op in visited:
            return
        visited.add(op)
        for child in op.input_dependencies:
            yield from visit(child)
        yield op

    yield from visit(root)


def _logical_consumer_counts(
    operators: Tuple[LogicalOperator, ...],
) -> Dict[LogicalOperator, int]:
    counts = {op: 0 for op in operators}
    for op in operators:
        for child in op.input_dependencies:
            counts[child] += 1
    return counts


def _unsupported(anchor: MapGroupPartitions, detail: str) -> None:
    raise GPUPlanningError(
        GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
        f"consumer={_op_label(anchor)}; {detail}",
    )


def _op_label(op: LogicalOperator) -> str:
    return f"{type(op).__name__}({op.name})"


def raise_mgp_without_required_gpu_fusion(
    logical_op: MapGroupPartitions,
    *,
    gpu_enabled: bool,
) -> None:
    detail = (
        "MapGroupPartitions has no CPU/stock fallback and requires "
        "DataContext.enable_gpu_operator_fusion=True"
        if not gpu_enabled
        else "MapGroupPartitions was not claimed by the exact required-fusion "
        "GPU planner and has no CPU/stock fallback"
    )
    raise GPUPlanningError(
        GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
        f"consumer={logical_op.name}; {detail}",
    )


__all__ = [
    "GPURequiredFusionPlanner",
    "GPUPlannedStage",
    "GPUPlanningError",
    "GPUPlanningErrorReason",
    "create_gpu_required_fusion_planner",
    "raise_mgp_without_required_gpu_fusion",
]
