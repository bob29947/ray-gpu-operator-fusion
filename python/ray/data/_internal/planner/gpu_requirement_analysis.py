"""No-I/O typed demand analysis for plan-local GPU lowering."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionRequirements,
    merge_gpu_execution_requirements,
)
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.planner.gpu_lowering.interfaces import (
    GPULogicalAnalysis,
    GPULogicalLowerer,
    GPURequirementDemand,
    is_positive_int,
)
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)


@dataclass(frozen=True)
class RangePartitionDemand(GPURequirementDemand):
    keys: Tuple[str, ...]
    requested_partitions: Optional[int] = None
    execution_requirements: Optional[GPUExecutionRequirements] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))
        if not self.keys or any(
            not isinstance(key, str) or not key for key in self.keys
        ):
            raise ValueError("range partition demand requires nonempty string keys")
        if self.requested_partitions is not None and not is_positive_int(
            self.requested_partitions
        ):
            raise ValueError("requested_partitions must be positive")

    def with_execution(
        self, requirements: GPUExecutionRequirements
    ) -> "RangePartitionDemand":
        detail = (
            "incompatible GPU execution requirements while propagating "
            f"range demand for keys={self.keys!r}"
        )
        return replace(
            self,
            execution_requirements=_merge_execution(
                self.execution_requirements, requirements, detail
            ),
        )

    def merge(self, other: "RangePartitionDemand") -> "RangePartitionDemand":
        if self.keys != other.keys:
            raise GPUPlanningError(
                GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS,
                f"one logical edge received incompatible range keys: "
                f"{self.keys!r} != {other.keys!r}",
            )
        if (
            self.requested_partitions is not None
            and other.requested_partitions is not None
            and self.requested_partitions != other.requested_partitions
        ):
            raise GPUPlanningError(
                GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS,
                f"one logical edge received incompatible partition counts: "
                f"{self.requested_partitions} != {other.requested_partitions}",
            )
        requested = self.requested_partitions or other.requested_partitions
        execution = _merge_execution(
            self.execution_requirements,
            other.execution_requirements,
            "one logical edge received incompatible GPU execution requirements",
        )
        return RangePartitionDemand(self.keys, requested, execution)

    def resolved_counts(self) -> Tuple[int, int]:
        requirements = self.execution_requirements
        if requirements is None or requirements.compatibility_key is None:
            raise GPUPlanningError(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                "range demand has no concrete GPU execution profile; bare "
                "Read -> MapGroupPartitions is not supported",
            )
        workers = requirements.compatibility_key.max_workers
        ranges = self.requested_partitions or 2 * workers
        return ranges, workers


@dataclass(frozen=True)
class GPURequirementAnalysis:
    logical_analyses: Mapping[LogicalOperator, Optional[GPULogicalAnalysis]]
    downstream_demands: Mapping[LogicalOperator, Tuple[GPURequirementDemand, ...]]

    def logical_analysis(
        self, logical_op: LogicalOperator
    ) -> Optional[GPULogicalAnalysis]:
        return self.logical_analyses.get(logical_op)

    def demands_for(
        self, logical_op: LogicalOperator
    ) -> Tuple[GPURequirementDemand, ...]:
        return self.downstream_demands.get(logical_op, ())


LowererResolver = Callable[[LogicalOperator], Optional[GPULogicalLowerer]]


def _merge_execution(
    left: Optional[GPUExecutionRequirements],
    right: Optional[GPUExecutionRequirements],
    detail: str,
) -> Optional[GPUExecutionRequirements]:
    if left is None or right is None:
        return right if left is None else left
    try:
        return merge_gpu_execution_requirements(left, right)
    except ValueError as error:
        raise GPUPlanningError(
            GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS,
            f"{detail}: {error}",
        ) from error


def analyze_gpu_requirements(
    root: LogicalOperator,
    data_context,
    resolve_lowerer: LowererResolver,
) -> GPURequirementAnalysis:
    """Propagate typed demands from consumers to inputs without doing source I/O."""

    post_order = tuple(_post_order(root))
    lowerers = {op: resolve_lowerer(op) for op in post_order}
    analyses: Dict[LogicalOperator, Optional[GPULogicalAnalysis]] = {
        op: None if lowerer is None else lowerer.analyze(op, data_context)
        for op, lowerer in lowerers.items()
    }

    incoming: Dict[LogicalOperator, list[GPURequirementDemand]] = {
        op: [] for op in post_order
    }
    normalized: Dict[LogicalOperator, Tuple[GPURequirementDemand, ...]] = {}
    for op in reversed(post_order):
        demands = _normalize_demands(incoming[op])
        normalized[op] = demands
        lowerer = lowerers[op]
        analysis = analyses[op]
        if lowerer is None or analysis is None or not analysis.accepted:
            transfers = tuple(() for _ in op.input_dependencies)
        else:
            transfers = lowerer.transfer_demands(op, demands, analysis)
        if len(transfers) != len(op.input_dependencies):
            raise ValueError(
                f"GPU lowerer for {type(op).__name__} returned {len(transfers)} "
                f"demand edges for {len(op.input_dependencies)} inputs"
            )
        for child, child_demands in zip(op.input_dependencies, transfers):
            incoming[child].extend(child_demands)
    return GPURequirementAnalysis(analyses, normalized)


def _normalize_demands(
    demands: Iterable[GPURequirementDemand],
) -> Tuple[GPURequirementDemand, ...]:
    merged = {}
    order = []
    for demand in demands:
        if not isinstance(demand, GPURequirementDemand):
            raise TypeError("GPU demand transfers must contain GPURequirementDemand")
        key = demand.demand_key
        try:
            previous = merged.get(key)
        except TypeError as error:
            raise TypeError("GPU demand keys must be hashable") from error
        if previous is None:
            merged[key] = demand
            order.append(key)
        else:
            merged[key] = previous.merge(demand)
    return tuple(merged[key] for key in order)


def _post_order(root: LogicalOperator) -> Iterable[LogicalOperator]:
    visited: set[LogicalOperator] = set()

    def visit(op: LogicalOperator):
        if op in visited:
            return
        visited.add(op)
        for child in op.input_dependencies:
            yield from visit(child)
        yield op

    yield from visit(root)
