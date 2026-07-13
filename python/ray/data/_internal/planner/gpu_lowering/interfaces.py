"""Exact-type logical lowerer registry used by the gate-2 GPU planner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Protocol

from ray.data._internal.execution.gpu.operator import GPUOperatorCreationOptions
from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionRequirements,
    GPUOperatorSpec,
)
from ray.data._internal.execution.gpu.properties import PropertyFact
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.planner.gpu_read_source_provider import (
    GPUReadSourceArtifacts,
    GPUReadSourceProvider,
)

if TYPE_CHECKING:
    from ray.data.context import DataContext


class GPUComponentRole(str, Enum):
    SOURCE = "source"
    UNARY = "unary"
    PARTITION_CONSUMER = "partition_consumer"


@dataclass(frozen=True)
class GPULogicalAnalysis:
    accepted: bool
    reason: Optional[str]
    execution_requirements: Optional[GPUExecutionRequirements] = None
    preserves_partitioning: bool = False
    partition_keys: tuple[str, ...] = ()
    requested_partitions: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "partition_keys", tuple(self.partition_keys))
        if self.accepted != (self.reason is None):
            raise ValueError("accepted must be true exactly when reason is None")
        if self.requested_partitions is not None:
            if (
                not isinstance(self.requested_partitions, int)
                or isinstance(self.requested_partitions, bool)
                or self.requested_partitions <= 0
            ):
                raise ValueError("requested_partitions must be positive")


@dataclass(frozen=True)
class GPULoweringContext:
    """Per-occurrence inputs and carried facts for one independent lowering."""

    stage_id: str
    logical_node_id: str
    group_keys: tuple[str, ...]
    source_provider: GPUReadSourceProvider
    execution_requirements: GPUExecutionRequirements
    target_max_block_size: int
    retried_io_errors: tuple[str, ...]
    target_ranges: int
    num_workers: int
    input_properties: tuple[PropertyFact, ...] = ()
    source_artifacts: Optional[GPUReadSourceArtifacts] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_keys", tuple(self.group_keys))
        object.__setattr__(self, "retried_io_errors", tuple(self.retried_io_errors))
        object.__setattr__(self, "input_properties", tuple(self.input_properties))
        if not self.stage_id:
            raise ValueError("stage_id must be nonempty")
        if not self.logical_node_id:
            raise ValueError("logical_node_id must be nonempty")
        if not self.group_keys or any(
            not isinstance(key, str) or not key for key in self.group_keys
        ):
            raise ValueError("group_keys must contain nonempty strings")
        if self.target_max_block_size <= 0:
            raise ValueError("target_max_block_size must be positive")
        if self.target_ranges <= 0:
            raise ValueError("target_ranges must be positive")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if not all(isinstance(pattern, str) for pattern in self.retried_io_errors):
            raise TypeError("retried_io_errors must contain strings")

    def transform_id(self, ordinal: int, kind: str) -> str:
        if ordinal < 0:
            raise ValueError("transform ordinal must be nonnegative")
        if not kind:
            raise ValueError("transform kind must be nonempty")
        return f"{self.logical_node_id}/{ordinal}/{kind}"

    def for_operator(
        self,
        *,
        logical_node_id: str,
        group_keys: Optional[tuple[str, ...]] = None,
    ) -> "GPULoweringContext":
        """Select the next logical occurrence while retaining carried facts."""

        return replace(
            self,
            logical_node_id=logical_node_id,
            group_keys=self.group_keys if group_keys is None else tuple(group_keys),
        )

    def for_lowered_output(
        self,
        spec: GPUOperatorSpec,
        *,
        source_artifacts: Optional[GPUReadSourceArtifacts] = None,
    ) -> "GPULoweringContext":
        """Carry only the facts explicitly re-advertised by this operator."""

        artifacts = (
            self.source_artifacts if source_artifacts is None else source_artifacts
        )
        return replace(
            self,
            execution_requirements=spec.execution_requirements,
            input_properties=spec.output_properties,
            source_artifacts=artifacts,
        )


@dataclass(frozen=True)
class GPULoweredOperator:
    spec: GPUOperatorSpec
    creation_options: GPUOperatorCreationOptions
    downstream_context: GPULoweringContext
    source_artifacts: Optional[GPUReadSourceArtifacts] = None

    def __post_init__(self) -> None:
        if self.downstream_context.input_properties != self.spec.output_properties:
            raise ValueError(
                "downstream context must carry exactly the lowered output facts"
            )
        if (
            self.downstream_context.execution_requirements
            != self.spec.execution_requirements
        ):
            raise ValueError(
                "downstream context and spec execution requirements must match"
            )
        if self.source_artifacts is not self.downstream_context.source_artifacts:
            raise ValueError(
                "lowered source artifacts and downstream context must match"
            )


class GPULogicalLowerer(Protocol):
    component_role: GPUComponentRole

    def analyze(
        self, logical_op: LogicalOperator, data_context: "DataContext"
    ) -> GPULogicalAnalysis: ...

    def lower(
        self, logical_op: LogicalOperator, context: GPULoweringContext
    ) -> GPULoweredOperator: ...


class GPULogicalLowererRegistry:
    """Exact-class registry used by one plan-local gate-2 analysis."""

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

    def unregister(
        self,
        logical_type: type[LogicalOperator],
        *,
        expected: Optional[GPULogicalLowerer] = None,
    ) -> GPULogicalLowerer:
        """Remove one lowerer, optionally guarding against stale cleanup."""

        try:
            lowerer = self._lowerers[logical_type]
        except KeyError as exc:
            raise KeyError(
                f"No GPU lowerer is registered for {logical_type.__name__}"
            ) from exc
        if expected is not None and lowerer is not expected:
            raise ValueError(
                f"GPU lowerer for {logical_type.__name__} is not the expected "
                "instance"
            )
        del self._lowerers[logical_type]
        return lowerer

    def copy(self) -> "GPULogicalLowererRegistry":
        """Return a plan-local snapshot of the registered lowerers."""

        registry = GPULogicalLowererRegistry()
        registry._lowerers.update(self._lowerers)
        return registry

    def get(self, logical_op: LogicalOperator) -> Optional[GPULogicalLowerer]:
        return self._lowerers.get(type(logical_op))

    def role_for(self, logical_op: LogicalOperator) -> Optional[GPUComponentRole]:
        lowerer = self.get(logical_op)
        return None if lowerer is None else lowerer.component_role

    def analyze(
        self, logical_op: LogicalOperator, data_context: "DataContext"
    ) -> GPULogicalAnalysis:
        lowerer = self.get(logical_op)
        if lowerer is None:
            raise ValueError(
                f"No GPU lowerer for exact logical type {type(logical_op).__name__}"
            )
        return lowerer.analyze(logical_op, data_context)

    def lower(
        self, logical_op: LogicalOperator, context: GPULoweringContext
    ) -> GPULoweredOperator:
        lowerer = self.get(logical_op)
        if lowerer is None:
            raise ValueError(
                f"No GPU lowerer for exact logical type {type(logical_op).__name__}"
            )
        return lowerer.lower(logical_op, context)
