"""Immutable inputs and results shared by the three V1 GPU lowerers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ray.data._internal.execution.gpu.operator import GPUOperatorCreationOptions
from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionRequirements,
    GPUOperatorSpec,
)
from ray.data._internal.execution.gpu.properties import PropertyFact
from ray.data._internal.planner.gpu_read_source_provider import (
    GPUReadSourceArtifacts,
    GPUReadSourceProvider,
)


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
