"""Plan-local state for fail-closed GPU physical planning."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

from ray.data._internal.planner.gpu_planning_errors import GPUPlanningErrorReason

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data._internal.logical.interfaces import LogicalOperator


@dataclass(frozen=True)
class GPUPlanningDiagnostic:
    reason: GPUPlanningErrorReason
    detail: str
    logical_ops: Tuple[Any, ...] = ()
    physical_ops: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class MaterializedGPURegion:
    operator: "PhysicalOperator"
    compiled_spec: Any
    backend_key: str
    input_payload: Any
    output_payload: Any
    internal_payloads: Tuple[Any, ...]
    attested: bool


def _unique_identity(values: Iterable[Any]) -> Tuple[Any, ...]:
    result, seen = [], set()
    for value in values:
        if id(value) not in seen:
            seen.add(id(value))
            result.append(value)
    return tuple(result)


def _reason(value: Any) -> GPUPlanningErrorReason:
    if isinstance(value, GPUPlanningErrorReason):
        return value
    try:
        return GPUPlanningErrorReason(value)
    except (TypeError, ValueError):
        return GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY


def _payload(
    explicit: Any, spec: Any, transforms: Tuple[Any, ...], attribute: str, index: int
) -> Any:
    if explicit is not None:
        return explicit
    value = getattr(spec, attribute, None)
    if value is not None or not transforms:
        return value
    return getattr(transforms[index], attribute, None)


class GPUFusionPlanState:
    """Diagnostics and trusted materializations belonging to one physical plan."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._logical_order: Tuple["LogicalOperator", ...] = ()
        self._terminal_logical: set["LogicalOperator"] = set()
        self._infrastructure_physical: set["PhysicalOperator"] = set()
        self._terminal_physical: set["PhysicalOperator"] = set()
        self._materialized_regions: dict["PhysicalOperator", MaterializedGPURegion] = {}
        self._stock_fallbacks: dict["LogicalOperator", GPUPlanningDiagnostic] = {}
        self._fusion_refusals: list[GPUPlanningDiagnostic] = []

    @property
    def logical_order(self) -> Tuple["LogicalOperator", ...]:
        return self._logical_order

    @property
    def required_logical(self) -> frozenset["LogicalOperator"]:
        return frozenset(
            op for op in self._logical_order if op not in self._terminal_logical
        )

    @property
    def terminal_logical(self) -> frozenset["LogicalOperator"]:
        return frozenset(self._terminal_logical)

    @property
    def infrastructure_physical(self) -> frozenset["PhysicalOperator"]:
        return frozenset(self._infrastructure_physical)

    @property
    def terminal_physical(self) -> frozenset["PhysicalOperator"]:
        return frozenset(self._terminal_physical)

    @property
    def materialized_regions(
        self,
    ) -> Mapping["PhysicalOperator", MaterializedGPURegion]:
        return MappingProxyType(self._materialized_regions)

    @property
    def stock_fallbacks(
        self,
    ) -> Mapping["LogicalOperator", GPUPlanningDiagnostic]:
        return MappingProxyType(self._stock_fallbacks)

    @property
    def fusion_refusals(self) -> Tuple[GPUPlanningDiagnostic, ...]:
        return tuple(self._fusion_refusals)

    def initialize_logical_universe(
        self,
        root: "LogicalOperator",
        *,
        terminal_logical: Sequence["LogicalOperator"] = (),
    ) -> None:
        order = _unique_identity(root.post_order_iter())
        terminals = set(terminal_logical)
        if not terminals.issubset(order):
            raise ValueError("terminal logical operators must belong to the plan")
        if self._logical_order and self._logical_order != order:
            raise ValueError("GPU fusion state cannot be reused across logical plans")
        self._logical_order = order
        self._terminal_logical.update(terminals)

    def register_stock_fallback(
        self,
        logical_op: "LogicalOperator",
        reason: Any,
        detail: str,
        physical_ops: Sequence["PhysicalOperator"] = (),
    ) -> None:
        self._stock_fallbacks[logical_op] = GPUPlanningDiagnostic(
            _reason(reason), detail, (logical_op,), tuple(physical_ops)
        )

    def register_fusion_refusal(
        self,
        physical_ops: Sequence["PhysicalOperator"],
        reason: Any,
        detail: str,
    ) -> None:
        self._fusion_refusals.append(
            GPUPlanningDiagnostic(_reason(reason), detail, (), tuple(physical_ops))
        )

    def register_infrastructure(self, *physical_ops: "PhysicalOperator") -> None:
        self._infrastructure_physical.update(physical_ops)

    def register_terminal(
        self,
        logical_op: Optional["LogicalOperator"] = None,
        physical_op: Optional["PhysicalOperator"] = None,
    ) -> None:
        if logical_op is not None:
            self._terminal_logical.add(logical_op)
        if physical_op is not None:
            self._terminal_physical.add(physical_op)

    def register_materialized_region(
        self,
        operator: "PhysicalOperator",
        compiled_spec: Any,
        backend_key: str,
        *,
        input_payload: Any = None,
        output_payload: Any = None,
        internal_payloads: Sequence[Any] = (),
        attested: bool = True,
    ) -> MaterializedGPURegion:
        transforms = tuple(getattr(compiled_spec, "transforms", ()))
        if not internal_payloads and len(transforms) > 1:
            internal_payloads = tuple(
                getattr(transform, "output_payload", None)
                for transform in transforms[:-1]
            )
        record = MaterializedGPURegion(
            operator,
            compiled_spec,
            backend_key,
            _payload(input_payload, compiled_spec, transforms, "input_payload", 0),
            _payload(output_payload, compiled_spec, transforms, "output_payload", -1),
            tuple(internal_payloads),
            bool(attested),
        )
        existing = self._materialized_regions.get(operator)
        if existing is not None and existing != record:
            raise ValueError("physical operator has conflicting GPU region records")
        self._materialized_regions[operator] = record
        return record

    def replace_physical(
        self,
        old_ops: Sequence["PhysicalOperator"],
        new_op: "PhysicalOperator",
    ) -> None:
        old = set(old_ops)
        for roles in (self._infrastructure_physical, self._terminal_physical):
            if old & roles:
                roles.difference_update(old)
                roles.add(new_op)
        for operator in old:
            self._materialized_regions.pop(operator, None)

    def synchronize_physical_roles(self, root: "PhysicalOperator") -> None:
        from ray.data._internal.execution.operators.input_data_buffer import (
            InputDataBuffer,
        )

        reachable = _unique_identity(root.post_order_iter())
        live = set(reachable)
        self._infrastructure_physical.intersection_update(live)
        self._terminal_physical.intersection_update(live)
        self._materialized_regions = {
            operator: record
            for operator, record in self._materialized_regions.items()
            if operator in live
        }
        self._infrastructure_physical.update(
            operator for operator in reachable if isinstance(operator, InputDataBuffer)
        )
        for operator in reachable:
            if (
                set(getattr(operator, "_logical_operators", ()))
                & self._terminal_logical
            ):
                self._terminal_physical.add(operator)
