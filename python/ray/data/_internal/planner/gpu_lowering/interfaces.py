"""Exact-type logical lowerer registry used inside an activated GPU region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

from ray.data._internal.execution.gpu.operator import GPUOperatorCreationOptions
from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionRequirements,
    GPUOperatorSpec,
)
from ray.data._internal.execution.gpu.parquet_types import ParquetRangePlan
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.planner.parquet_cudf_read_recognizer import (
    ParquetCudfReadDescriptor,
)


@dataclass(frozen=True)
class GPULoweringContext:
    region_id: str
    group_keys: tuple[str, ...]
    read_descriptor: ParquetCudfReadDescriptor
    range_plan: ParquetRangePlan
    execution_requirements: GPUExecutionRequirements
    target_max_block_size: int
    retried_io_errors: tuple[str, ...]

    @property
    def required_logical_provenance(self) -> tuple[str, ...]:
        return ("Read", "MapBatches", "MapGroupPartitions")


@dataclass(frozen=True)
class GPULoweredOperator:
    spec: GPUOperatorSpec
    creation_options: GPUOperatorCreationOptions


class GPULogicalLowerer(Protocol):
    def lower(
        self, logical_op: LogicalOperator, context: GPULoweringContext
    ) -> GPULoweredOperator: ...


class GPULogicalLowererRegistry:
    """Private exact-class registry; it has no public plugin surface in V1."""

    def __init__(self) -> None:
        self._lowerers: Dict[type[LogicalOperator], GPULogicalLowerer] = {}

    def register(
        self,
        logical_type: type[LogicalOperator],
        lowerer: GPULogicalLowerer,
    ) -> None:
        if logical_type in self._lowerers:
            raise ValueError(
                f"GPU lowerer already registered for {logical_type.__name__}"
            )
        self._lowerers[logical_type] = lowerer

    def get(self, logical_op: LogicalOperator) -> Optional[GPULogicalLowerer]:
        return self._lowerers.get(type(logical_op))

    def lower(
        self, logical_op: LogicalOperator, context: GPULoweringContext
    ) -> GPULoweredOperator:
        lowerer = self.get(logical_op)
        if lowerer is None:
            raise ValueError(
                f"No GPU lowerer for exact logical type {type(logical_op).__name__}"
            )
        return lowerer.lower(logical_op, context)

