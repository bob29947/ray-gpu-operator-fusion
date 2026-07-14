"""Plan-local contracts for independently lowered GPU logical operators."""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Any, Hashable, Optional, Protocol, Tuple, Union

from ray.data._internal.execution.gpu.operator import GPUOperatorCreationOptions
from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionRequirements,
    GPUOperatorSpec,
)
from ray.data._internal.execution.gpu.properties import PropertyFact
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)


class GPURequirementDemand:
    """Mergeable typed requirement propagated toward logical inputs."""

    @property
    def demand_key(self) -> Hashable:
        return type(self)

    def merge(self, other: "GPURequirementDemand") -> "GPURequirementDemand":
        if type(self) is type(other) and self == other:
            return self
        raise GPUPlanningError(
            GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS,
            f"one logical edge received incompatible {type(self).__name__} demands",
        )


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_synchronous_callable(value: Any) -> bool:
    if not callable(value):
        return False
    function = (
        value
        if inspect.isroutine(value) or isinstance(value, functools.partial)
        else value.__call__
    )
    return not inspect.iscoroutinefunction(function) and not inspect.isasyncgenfunction(
        function
    )


@dataclass(frozen=True)
class GPULogicalAnalysis:
    accepted: bool
    reason: Optional[str] = None
    execution_requirements: Optional[GPUExecutionRequirements] = None

    def __post_init__(self) -> None:
        if self.accepted != (self.reason is None):
            raise ValueError("accepted must be true exactly when reason is None")


@dataclass(frozen=True)
class GPULoweringContext:
    """Immutable facts and requirements available to one logical occurrence."""

    target_max_block_size: int
    input_facts: Tuple[PropertyFact, ...] = ()
    execution_requirements: Optional[GPUExecutionRequirements] = None
    source_provider: Any = None
    source_analysis: Any = None
    retried_io_errors: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_facts", tuple(self.input_facts))
        object.__setattr__(self, "retried_io_errors", tuple(self.retried_io_errors))
        if self.target_max_block_size <= 0:
            raise ValueError("target_max_block_size must be positive")
        if not all(isinstance(pattern, str) for pattern in self.retried_io_errors):
            raise TypeError("retried_io_errors must contain strings")


@dataclass(frozen=True)
class GPULoweredOperator:
    spec: GPUOperatorSpec
    creation_options: GPUOperatorCreationOptions
    source_artifacts: Any = None


@dataclass(frozen=True)
class GPUPlan:
    lowered: GPULoweredOperator


@dataclass(frozen=True)
class GPUDecline:
    reason: GPUPlanningErrorReason
    detail: str


@dataclass(frozen=True)
class GPUInvalid:
    reason: GPUPlanningErrorReason
    detail: str


GPUDecision = Union[GPUPlan, GPUDecline, GPUInvalid]


class GPULogicalLowerer(Protocol):
    def analyze(
        self, logical_op: LogicalOperator, data_context: Any
    ) -> GPULogicalAnalysis: ...

    def transfer_demands(
        self,
        logical_op: LogicalOperator,
        downstream_demands: Tuple[GPURequirementDemand, ...],
        analysis: GPULogicalAnalysis,
    ) -> Tuple[Tuple[GPURequirementDemand, ...], ...]: ...

    def lower(
        self, logical_op: LogicalOperator, context: GPULoweringContext
    ) -> GPULoweredOperator: ...
