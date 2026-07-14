"""Type-independent execution-cover validation for strict GPU fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, FrozenSet, Iterable, Mapping, Optional, Tuple

from ray.data._internal.execution.gpu.planning_state import (
    GPUFusionPlanState,
    GPUPlanningDiagnostic,
    MaterializedGPURegion,
)
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)


DATA_REGION_ROLE = "data_region"
TERMINAL_OUTPUT_ROLE = "terminal_output"
UNSANCTIONED_ROLE = "unsanctioned"


@dataclass(frozen=True)
class CoverMember:
    member_id: str
    role: str
    logical_lineage: Tuple[Any, ...]
    physical_members: FrozenSet[Any]
    executable: bool
    input_payload: Any = None
    output_payload: Any = None
    internal_payloads: Tuple[Any, ...] = ()
    attested: bool = False

    @property
    def logical_members(self) -> FrozenSet[Any]:
        return frozenset(self.logical_lineage)


@dataclass(frozen=True)
class CoverRule:
    role: str
    minimum: int
    maximum: int
    predicate: Callable[[CoverMember], bool] = field(compare=False, repr=False)


CoverViolation = Tuple[str, FrozenSet[Any], FrozenSet[Any]]
TopologyValidator = Callable[
    [Mapping[str, Tuple[CoverMember, ...]], "CoverSnapshot"],
    Optional[CoverViolation],
]


@dataclass(frozen=True)
class CoverPolicy:
    rules: Tuple[CoverRule, ...]
    validate_topology: TopologyValidator = field(compare=False, repr=False)
    classify_unowned: Optional[Callable[[Any, GPUFusionPlanState], CoverMember]] = (
        field(default=None, compare=False, repr=False)
    )


@dataclass(frozen=True)
class CoverSnapshot:
    required_logical: FrozenSet[Any]
    required_physical: FrozenSet[Any]
    expected_logical_order: Tuple[Any, ...]
    members: Tuple[CoverMember, ...]
    member_edges: FrozenSet[Tuple[str, str]]
    diagnostics: Tuple[GPUPlanningDiagnostic, ...] = ()


def _payload_key(payload: Any) -> str:
    if payload is None:
        return ""
    return str(getattr(payload, "key", getattr(payload, "value", payload))).lower()


def _materializable(payload: Any) -> bool:
    declared = getattr(payload, "materializable_boundary", None)
    return (
        bool(declared)
        if declared is not None
        else _payload_key(payload) in {"ray_block_stream", "device_handle"}
    )


def _label(operator: Any) -> str:
    name = getattr(operator, "name", type(operator).__name__)
    return f"{type(operator).__name__}[{name}]"


def _valid_lineage(member: CoverMember) -> bool:
    return bool(member.logical_lineage) and len(member.logical_lineage) == len(
        member.logical_members
    )


def _strict_region(member: CoverMember) -> bool:
    return (
        member.executable
        and member.attested
        and _valid_lineage(member)
        and _payload_key(member.input_payload) == "ray_block_stream"
        and _payload_key(member.output_payload) == "ray_block_stream"
        and not any(map(_materializable, member.internal_payloads))
    )


def _strict_terminal(member: CoverMember) -> bool:
    return member.executable and member.attested and _valid_lineage(member)


def _validate_strict_topology(
    accepted: Mapping[str, Tuple[CoverMember, ...]], snapshot: CoverSnapshot
) -> Optional[CoverViolation]:
    region = accepted[DATA_REGION_ROLE][0]
    terminals = accepted[TERMINAL_OUTPUT_ROLE]
    terminal_logical = {
        logical for terminal in terminals for logical in terminal.logical_members
    }
    expected_lineage = tuple(
        logical
        for logical in snapshot.expected_logical_order
        if logical in snapshot.required_logical and logical not in terminal_logical
    )
    if region.logical_lineage != expected_lineage:
        return (
            "the materialized GPU region lineage is missing, foreign, or reordered",
            frozenset(region.logical_lineage) | frozenset(expected_lineage),
            region.physical_members,
        )
    expected_edges = (
        frozenset()
        if not terminals
        else frozenset(((region.member_id, terminals[0].member_id),))
    )
    if snapshot.member_edges != expected_edges:
        return (
            "strict GPU fusion cover is not one linear data region followed by "
            f"an optional terminal output: edges={sorted(snapshot.member_edges)!r}",
            snapshot.required_logical,
            snapshot.required_physical,
        )
    return None


STRICT_FULL_FUSION_POLICY = CoverPolicy(
    (
        CoverRule(DATA_REGION_ROLE, 1, 1, _strict_region),
        CoverRule(TERMINAL_OUTPUT_ROLE, 0, 1, _strict_terminal),
    ),
    _validate_strict_topology,
)


def _reachable(root: Any) -> Tuple[Any, ...]:
    result, visited, stack = [], set(), [root]
    while stack:
        operator = stack.pop()
        if operator not in visited:
            visited.add(operator)
            result.append(operator)
            stack.extend(reversed(operator.input_dependencies))
    return tuple(result)


def _lineage(operator: Any) -> Tuple[Any, ...]:
    return tuple(getattr(operator, "_logical_operators", ()))


def _member_id(role: str, operator: Any) -> str:
    return f"{role}:{getattr(operator, 'id', id(operator))}"


def _region_member(record: MaterializedGPURegion) -> CoverMember:
    operator = record.operator
    attested = (
        record.attested
        and getattr(operator, "gpu_operator_spec", None) is record.compiled_spec
        and getattr(operator, "execution_backend_name", None) == record.backend_key
    )
    return CoverMember(
        _member_id("region", operator),
        DATA_REGION_ROLE,
        _lineage(operator),
        frozenset((operator,)),
        True,
        record.input_payload,
        record.output_payload,
        record.internal_payloads,
        attested,
    )


def _terminal_member(operator: Any, terminal_logical: FrozenSet[Any]) -> CoverMember:
    lineage = _lineage(operator)
    return CoverMember(
        _member_id("terminal", operator),
        TERMINAL_OUTPUT_ROLE,
        lineage,
        frozenset((operator,)),
        True,
        attested=bool(lineage) and set(lineage).issubset(terminal_logical),
    )


def _unsanctioned_member(operator: Any) -> CoverMember:
    return CoverMember(
        _member_id("unsanctioned", operator),
        UNSANCTIONED_ROLE,
        _lineage(operator),
        frozenset((operator,)),
        False,
    )


def build_execution_cover(
    root: Any,
    state: GPUFusionPlanState,
    policy: CoverPolicy = STRICT_FULL_FUSION_POLICY,
) -> CoverSnapshot:
    state.synchronize_physical_roles(root)
    reachable = _reachable(root)
    physical = frozenset(
        operator
        for operator in reachable
        if operator not in state.infrastructure_physical
    )
    members, owned = [], set()
    for operator, record in state.materialized_regions.items():
        if operator in physical:
            members.append(_region_member(record))
            owned.add(operator)
    for operator in state.terminal_physical:
        if operator in physical and operator not in owned:
            members.append(_terminal_member(operator, state.terminal_logical))
            owned.add(operator)
    for operator in reachable:
        if operator not in physical or operator in owned:
            continue
        members.append(
            policy.classify_unowned(operator, state)
            if policy.classify_unowned is not None
            else _unsanctioned_member(operator)
        )
        owned.add(operator)
    owners = {
        operator: member.member_id
        for member in members
        for operator in member.physical_members
    }
    edges = frozenset(
        (owners[input_op], owners[operator])
        for operator in physical
        for input_op in operator.input_dependencies
        if input_op in owners
        and operator in owners
        and owners[input_op] != owners[operator]
    )
    return CoverSnapshot(
        state.required_logical | state.terminal_logical,
        physical,
        state.logical_order,
        tuple(members),
        edges,
        tuple(state.stock_fallbacks.values()) + state.fusion_refusals,
    )


def _partition(
    universe: FrozenSet[Any], members: Iterable[FrozenSet[Any]]
) -> Tuple[FrozenSet[Any], FrozenSet[Any], FrozenSet[Any]]:
    counts = dict.fromkeys(universe, 0)
    extras = set()
    for values in members:
        for value in values:
            if value in counts:
                counts[value] += 1
            else:
                extras.add(value)
    return (
        frozenset(item for item, count in counts.items() if count == 0),
        frozenset(item for item, count in counts.items() if count > 1),
        frozenset(extras),
    )


def _diagnostic_reason(
    diagnostics: Tuple[GPUPlanningDiagnostic, ...],
    logical: FrozenSet[Any],
    physical: FrozenSet[Any],
) -> Tuple[GPUPlanningErrorReason, str]:
    for diagnostic in reversed(diagnostics):
        if not (
            set(diagnostic.logical_ops) & logical
            or set(diagnostic.physical_ops) & physical
        ):
            continue
        if diagnostic.reason in {
            GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
            GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS,
        }:
            return diagnostic.reason, diagnostic.detail
    return GPUPlanningErrorReason.INCOMPLETE_FUSION, ""


def _raise_cover_error(
    snapshot: CoverSnapshot,
    detail: str,
    *,
    logical: Iterable[Any] = (),
    physical: Iterable[Any] = (),
) -> None:
    logical, physical = frozenset(logical), frozenset(physical)
    reason, cause = _diagnostic_reason(snapshot.diagnostics, logical, physical)
    if cause:
        detail = f"{detail}; cause={cause}"
    raise GPUPlanningError(
        reason,
        detail,
        logical_ops=tuple(map(_label, logical)),
        physical_ops=tuple(map(_label, physical)),
    )


def validate_execution_cover(
    snapshot: CoverSnapshot,
    policy: CoverPolicy = STRICT_FULL_FUSION_POLICY,
) -> None:
    rules = {rule.role: rule for rule in policy.rules}
    accepted = {role: [] for role in rules}
    unsupported = []
    for member in snapshot.members:
        rule = rules.get(member.role)
        if rule is not None and rule.predicate(member):
            accepted[member.role].append(member)
        else:
            unsupported.append(member)
    if unsupported:
        _raise_cover_error(
            snapshot,
            "strict GPU fusion found unsanctioned, unattested, or invalid-lineage "
            f"physical members: {[member.member_id for member in unsupported]!r}",
            logical=(item for member in unsupported for item in member.logical_members),
            physical=(
                item for member in unsupported for item in member.physical_members
            ),
        )
    for role, rule in rules.items():
        if not rule.minimum <= len(accepted[role]) <= rule.maximum:
            _raise_cover_error(
                snapshot,
                f"strict GPU fusion requires {rule.minimum}..{rule.maximum} "
                f"{role} members, found {len(accepted[role])}",
                logical=snapshot.required_logical,
                physical=snapshot.required_physical,
            )
    logical_partition = _partition(
        snapshot.required_logical,
        (member.logical_members for member in snapshot.members),
    )
    physical_partition = _partition(
        snapshot.required_physical,
        (member.physical_members for member in snapshot.members),
    )
    if any(logical_partition) or any(physical_partition):
        _raise_cover_error(
            snapshot,
            "strict GPU fusion cover is not an exact logical/physical partition: "
            f"logical={tuple(map(len, logical_partition))!r}, "
            f"physical={tuple(map(len, physical_partition))!r}",
            logical=set().union(*logical_partition),
            physical=set().union(*physical_partition),
        )
    accepted_view = {role: tuple(members) for role, members in accepted.items()}
    violation = policy.validate_topology(accepted_view, snapshot)
    if violation is not None:
        detail, logical, physical = violation
        _raise_cover_error(
            snapshot,
            detail,
            logical=logical,
            physical=physical,
        )


def validate_gpu_execution_plan(
    physical_plan: Any,
    policy: CoverPolicy = STRICT_FULL_FUSION_POLICY,
) -> None:
    if not getattr(physical_plan.context, "enable_gpu_operator_fusion", False):
        return
    state = getattr(physical_plan, "gpu_fusion_state", None)
    if state is None or not state.enabled:
        raise GPUPlanningError(
            GPUPlanningErrorReason.INCOMPLETE_FUSION,
            "strict GPU fusion has no plan-local fusion state",
        )
    validate_execution_cover(
        build_execution_cover(physical_plan.dag, state, policy), policy
    )
