"""Physical-adjacent requirement analysis for explicit GPU Data operations.

This module deliberately does not build a physical operator or inspect Parquet
footers.  It annotates the frozen logical DAG with the facts gate 2 needs to
lower an MGP-bearing component.  Concrete source recognition and range planning
belong to Read lowering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from ray.data._internal.execution.gpu.operator_spec import GPUExecutionRequirements
from ray.data._internal.execution.gpu.properties import (
    CompletePartitionBundles,
    Evidence,
    KeysColocated,
    PartitioningPreservationError,
    PropertyFact,
    RangePartitioning,
    UnknownPartitioning,
    readvertise_preserved_properties,
    requirements_satisfied,
    usable_partitioning_fact,
)
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators import MapBatches, MapGroupPartitions, Read
from ray.data._internal.planner.gpu_execution_profile import (
    default_gpu_execution_requirements,
    reconcile_read_remote_args,
)
from ray.data._internal.planner.gpu_lowering import (
    GPUComponentRole,
    GPULogicalAnalysis,
    GPULogicalLowerer,
    GPULogicalLowererRegistry,
)
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)
from ray.data._internal.planner.gpu_read_source_provider import (
    GPUReadSourceDemand,
    GPUReadSourceProvider,
    GPUReadSourceProviderRegistry,
    create_default_gpu_source_provider_registry,
)
from ray.data.context import DataContext


@dataclass(frozen=True)
class GPUPropertyState:
    """Planner facts on one edge before a concrete range scheme exists."""

    facts: Tuple[PropertyFact, ...]
    partition_count: Optional[int]
    first_invalidator: Optional[LogicalOperator] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "facts", tuple(self.facts))

    @classmethod
    def unknown(
        cls,
        *,
        partition_count: Optional[int],
        complete_partition_bundles: bool,
        first_invalidator: LogicalOperator,
    ) -> "GPUPropertyState":
        facts = [
            PropertyFact(UnknownPartitioning(), Evidence.DERIVED),
        ]
        if complete_partition_bundles:
            facts.append(PropertyFact(CompletePartitionBundles(), Evidence.DERIVED))
        return cls(
            facts=tuple(facts),
            partition_count=partition_count,
            first_invalidator=first_invalidator,
        )


@dataclass(frozen=True)
class GPUEnforcementRequest:
    consumer: LogicalOperator
    required_keys: Tuple[str, ...]
    required_partition_count: int
    available: GPUPropertyState
    upstream_execution_requirements: GPUExecutionRequirements
    downstream_execution_requirements: GPUExecutionRequirements
    detail: str


@dataclass(frozen=True)
class GPUEnforcerPlan:
    """Physical boundary and facts established by one registered enforcer."""

    physical_input: Any
    output_state: GPUPropertyState
    op_map: Mapping[Any, LogicalOperator]

    def __post_init__(self) -> None:
        object.__setattr__(self, "op_map", MappingProxyType(dict(self.op_map)))


class GPUPropertyEnforcer(Protocol):
    def supports(self, request: GPUEnforcementRequest) -> bool: ...

    def established_state(self, request: GPUEnforcementRequest) -> GPUPropertyState: ...

    def plan(
        self,
        request: GPUEnforcementRequest,
        *,
        upstream: Any,
        data_context: DataContext,
    ) -> GPUEnforcerPlan: ...


class GPUEdgeResolutionKind(str, Enum):
    SAME_STAGE = "same_stage"
    ENFORCER = "enforcer"


@dataclass(frozen=True)
class GPUEdgeResolution:
    kind: GPUEdgeResolutionKind
    request: Optional[GPUEnforcementRequest] = None
    enforcer: Optional[GPUPropertyEnforcer] = None
    established_state: Optional[GPUPropertyState] = None

    def __post_init__(self) -> None:
        enforced = self.kind is GPUEdgeResolutionKind.ENFORCER
        if enforced != (
            self.request is not None
            and self.enforcer is not None
            and self.established_state is not None
        ):
            raise ValueError(
                "enforcer resolution requires both a request and an enforcer"
            )

    @classmethod
    def same_stage(cls) -> "GPUEdgeResolution":
        return cls(GPUEdgeResolutionKind.SAME_STAGE)


class GPUEnforcerRegistry:
    """Registry consulted by the one unsatisfied-requirement branch."""

    def __init__(self) -> None:
        self._enforcers: list[GPUPropertyEnforcer] = []

    def register(self, enforcer: GPUPropertyEnforcer) -> None:
        if any(registered is enforcer for registered in self._enforcers):
            raise ValueError("GPU property enforcer instance is already registered")
        self._enforcers.append(enforcer)

    def unregister(self, enforcer: GPUPropertyEnforcer) -> None:
        for index, registered in enumerate(self._enforcers):
            if registered is enforcer:
                del self._enforcers[index]
                return
        raise KeyError("GPU property enforcer instance is not registered")

    def copy(self) -> "GPUEnforcerRegistry":
        """Return a plan-local snapshot preserving enforcer priority order."""

        registry = GPUEnforcerRegistry()
        registry._enforcers.extend(self._enforcers)
        return registry

    def select(self, request: GPUEnforcementRequest) -> Optional[GPUPropertyEnforcer]:
        return next(
            (enforcer for enforcer in self._enforcers if enforcer.supports(request)),
            None,
        )

    def __len__(self) -> int:
        return len(self._enforcers)


_REGISTERED_GPU_ENFORCERS = GPUEnforcerRegistry()


def register_gpu_property_enforcer(enforcer: GPUPropertyEnforcer) -> None:
    """Register an enforcer used by every subsequently planned logical plan."""

    _REGISTERED_GPU_ENFORCERS.register(enforcer)


def unregister_gpu_property_enforcer(enforcer: GPUPropertyEnforcer) -> None:
    """Unregister an enforcer for plugin teardown or test isolation."""

    _REGISTERED_GPU_ENFORCERS.unregister(enforcer)


def create_default_gpu_enforcer_registry() -> GPUEnforcerRegistry:
    """Snapshot the globally registered enforcers for one planning pass."""

    return _REGISTERED_GPU_ENFORCERS.copy()


class GPURequirementResolver:
    """Resolve a same-stage proof or route through the enforcer registry."""

    def __init__(self, enforcers: GPUEnforcerRegistry):
        self._enforcers = enforcers

    def require_partitioned_edge(
        self,
        *,
        consumer: LogicalOperator,
        required_keys: Tuple[str, ...],
        required_partition_count: int,
        available: GPUPropertyState,
        execution_requirements: GPUExecutionRequirements,
        detail: str,
    ) -> GPUEdgeResolution:
        if self._state_satisfies(
            available,
            required_keys=required_keys,
            required_partition_count=required_partition_count,
        ):
            return GPUEdgeResolution.same_stage()
        return self._resolve_unsatisfied(
            GPUEnforcementRequest(
                consumer=consumer,
                required_keys=required_keys,
                required_partition_count=required_partition_count,
                available=available,
                upstream_execution_requirements=execution_requirements,
                downstream_execution_requirements=execution_requirements,
                detail=detail,
            )
        )

    def require_same_stage(
        self,
        *,
        consumer: LogicalOperator,
        required_keys: Tuple[str, ...],
        required_partition_count: int,
        available: GPUPropertyState,
        upstream_execution_requirements: GPUExecutionRequirements,
        downstream_execution_requirements: GPUExecutionRequirements,
        detail: str,
    ) -> GPUEdgeResolution:
        """Resolve a failed same-stage attempt through an explicit enforcer."""

        return self._resolve_unsatisfied(
            GPUEnforcementRequest(
                consumer=consumer,
                required_keys=required_keys,
                required_partition_count=required_partition_count,
                available=available,
                upstream_execution_requirements=upstream_execution_requirements,
                downstream_execution_requirements=downstream_execution_requirements,
                detail=detail,
            )
        )

    def _resolve_unsatisfied(self, request: GPUEnforcementRequest) -> GPUEdgeResolution:
        # Keep this branch structurally stable.  A future exchange adds an
        # enforcer; it does not add another special case to the planner.
        enforcer = self._enforcers.select(request)
        if enforcer is not None:
            established_state = enforcer.established_state(request)
            self.validate_enforcer_state(request, established_state)
            return GPUEdgeResolution(
                GPUEdgeResolutionKind.ENFORCER,
                request=request,
                enforcer=enforcer,
                established_state=established_state,
            )
        invalidator = request.available.first_invalidator
        invalidator_detail = (
            _op_label(invalidator) if invalidator is not None else "none"
        )
        raise GPUPlanningError(
            GPUPlanningErrorReason.EXCHANGE_REQUIRED,
            f"consumer={_op_label(request.consumer)}; "
            f"required_keys={request.required_keys!r}; "
            f"required_count={request.required_partition_count}; "
            f"available={_property_label(request.available)}; "
            f"first_invalidator={invalidator_detail}; {request.detail}; "
            "no GPU exchange is registered",
        )

    def validate_enforcer_state(
        self,
        request: GPUEnforcementRequest,
        state: GPUPropertyState,
    ) -> None:
        if self._state_satisfies(
            state,
            required_keys=request.required_keys,
            required_partition_count=request.required_partition_count,
        ):
            return
        raise GPUPlanningError(
            GPUPlanningErrorReason.EXCHANGE_REQUIRED,
            f"consumer={_op_label(request.consumer)}; selected GPU enforcer "
            f"did not establish required_keys={request.required_keys!r}, "
            f"required_count={request.required_partition_count}; "
            f"enforced_state={_property_label(state)}",
        )

    @staticmethod
    def _state_satisfies(
        state: GPUPropertyState,
        *,
        required_keys: Tuple[str, ...],
        required_partition_count: int,
    ) -> bool:
        requirements = (
            KeysColocated(required_keys),
            CompletePartitionBundles(),
        )
        return (
            state.partition_count == required_partition_count
            and requirements_satisfied(state.facts, requirements)
        )


class GPUPropertyTransfer(Protocol):
    def transfer(
        self,
        logical_op: LogicalOperator,
        input_state: GPUPropertyState,
    ) -> GPUPropertyState: ...


class GPUPropertyTransferRegistry:
    """Exact-logical-type registry for explicit property transfer."""

    def __init__(self) -> None:
        self._transfers: Dict[type[LogicalOperator], GPUPropertyTransfer] = {}

    def register(
        self,
        logical_type: type[LogicalOperator],
        transfer: GPUPropertyTransfer,
    ) -> None:
        if logical_type in self._transfers:
            raise ValueError(
                f"GPU property transfer already registered for {logical_type.__name__}"
            )
        self._transfers[logical_type] = transfer

    def unregister(
        self,
        logical_type: type[LogicalOperator],
        *,
        expected: Optional[GPUPropertyTransfer] = None,
    ) -> GPUPropertyTransfer:
        try:
            transfer = self._transfers[logical_type]
        except KeyError as exc:
            raise KeyError(
                f"No GPU property transfer is registered for "
                f"{logical_type.__name__}"
            ) from exc
        if expected is not None and transfer is not expected:
            raise ValueError(
                f"GPU property transfer for {logical_type.__name__} is not the "
                "expected instance"
            )
        del self._transfers[logical_type]
        return transfer

    def copy(self) -> "GPUPropertyTransferRegistry":
        """Return a plan-local snapshot of the registered transfer functions."""

        registry = GPUPropertyTransferRegistry()
        registry._transfers.update(self._transfers)
        return registry

    def transfer(
        self,
        logical_op: LogicalOperator,
        input_state: GPUPropertyState,
    ) -> GPUPropertyState:
        transfer = self._transfers.get(type(logical_op))
        if transfer is None:
            return GPUPropertyState.unknown(
                partition_count=input_state.partition_count,
                complete_partition_bundles=False,
                first_invalidator=input_state.first_invalidator or logical_op,
            )
        return transfer.transfer(logical_op, input_state)


class _MapBatchesPropertyTransfer:
    def transfer(
        self, logical_op: LogicalOperator, input_state: GPUPropertyState
    ) -> GPUPropertyState:
        assert type(logical_op) is MapBatches
        properties = logical_op.udf_properties
        return _readvertised_property_state(
            logical_op,
            input_state,
            preserves_partitioning=bool(
                properties is not None and properties.preserves_partitioning
            ),
            # MapBatches retains the incoming planner unit independently of
            # whether its UDF retains semantic partitioning.
            preserves_complete_partition_bundles=True,
        )


class _MapGroupPartitionsPropertyTransfer:
    def transfer(
        self, logical_op: LogicalOperator, input_state: GPUPropertyState
    ) -> GPUPropertyState:
        assert type(logical_op) is MapGroupPartitions
        properties = logical_op.udf_properties
        preserves = bool(properties is not None and properties.preserves_partitioning)
        return _readvertised_property_state(
            logical_op,
            input_state,
            preserves_partitioning=preserves,
            # Once MGP changes partition identity, bundle completeness has no
            # semantic partition to describe.
            preserves_complete_partition_bundles=preserves,
        )


def _readvertised_property_state(
    logical_op: LogicalOperator,
    input_state: GPUPropertyState,
    *,
    preserves_partitioning: bool,
    preserves_complete_partition_bundles: bool,
) -> GPUPropertyState:
    """Use the same explicit re-advertisement model as physical lowering."""

    try:
        readvertisement = readvertise_preserved_properties(
            input_state.facts,
            preserves_partitioning=preserves_partitioning,
            preserves_complete_partition_bundles=(preserves_complete_partition_bundles),
            # Analysis confirms that an implementation exists; construction of
            # its concrete validation specs belongs to per-operator lowering.
            plan_validations=False,
        )
        output_partitioning = usable_partitioning_fact(
            readvertisement.output_properties
        )
    except PartitioningPreservationError as error:
        raise GPUPlanningError(
            GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
            f"consumer={_op_label(logical_op)}; {error}",
        ) from error

    first_invalidator = input_state.first_invalidator
    if output_partitioning is None:
        first_invalidator = first_invalidator or logical_op
    return GPUPropertyState(
        facts=readvertisement.output_properties,
        partition_count=input_state.partition_count,
        first_invalidator=first_invalidator,
    )


_REGISTERED_GPU_PROPERTY_TRANSFERS = GPUPropertyTransferRegistry()
_REGISTERED_GPU_PROPERTY_TRANSFERS.register(MapBatches, _MapBatchesPropertyTransfer())
_REGISTERED_GPU_PROPERTY_TRANSFERS.register(
    MapGroupPartitions, _MapGroupPartitionsPropertyTransfer()
)


def register_gpu_property_transfer(
    logical_type: type[LogicalOperator],
    transfer: GPUPropertyTransfer,
) -> None:
    """Register an explicit property transfer for future planning passes."""

    _REGISTERED_GPU_PROPERTY_TRANSFERS.register(logical_type, transfer)


def unregister_gpu_property_transfer(
    logical_type: type[LogicalOperator],
    *,
    expected: Optional[GPUPropertyTransfer] = None,
) -> GPUPropertyTransfer:
    """Unregister a non-built-in transfer for plugin teardown or tests."""

    if logical_type in (MapBatches, MapGroupPartitions):
        raise ValueError("Built-in GPU property transfers cannot be removed")
    return _REGISTERED_GPU_PROPERTY_TRANSFERS.unregister(
        logical_type,
        expected=expected,
    )


def create_default_property_transfer_registry() -> GPUPropertyTransferRegistry:
    """Snapshot all built-in and globally registered property transfers."""

    return _REGISTERED_GPU_PROPERTY_TRANSFERS.copy()


@dataclass(frozen=True)
class GPUComponentMembership:
    component_id: str
    ordinal: int
    input_state: GPUPropertyState
    output_state: GPUPropertyState
    input_resolution: GPUEdgeResolution
    execution_requirements: GPUExecutionRequirements


@dataclass(frozen=True)
class GPUOperatorAnnotation:
    logical_node_id: str
    memberships: Tuple[GPUComponentMembership, ...]


@dataclass(frozen=True)
class GPUSourceDemand:
    source: LogicalOperator
    range_key: str
    partition_count: int
    execution_requirements: GPUExecutionRequirements
    provider: GPUReadSourceProvider


@dataclass(frozen=True)
class MGPRequirement:
    operator: MapGroupPartitions
    keys: Tuple[str, ...]
    partition_count: int
    input_resolution: GPUEdgeResolution


@dataclass(frozen=True)
class MGPGPUComponent:
    component_id: str
    source: LogicalOperator
    operators: Tuple[LogicalOperator, ...]
    anchor: MapGroupPartitions
    source_demand: GPUSourceDemand
    mgp_requirements: Tuple[MGPRequirement, ...]
    execution_requirements: GPUExecutionRequirements


@dataclass(frozen=True)
class GPURequirementAnalysis:
    annotations: Mapping[LogicalOperator, GPUOperatorAnnotation]
    components_by_anchor: Mapping[MapGroupPartitions, MGPGPUComponent]
    logical_node_ids: Mapping[LogicalOperator, str]

    @classmethod
    def empty(cls) -> "GPURequirementAnalysis":
        empty = MappingProxyType({})
        return cls(empty, empty, empty)


def analyze_gpu_requirements(
    root: LogicalOperator,
    data_context: DataContext,
    *,
    lowerers: Optional[GPULogicalLowererRegistry] = None,
    source_providers: Optional[GPUReadSourceProviderRegistry] = None,
    property_transfers: Optional[GPUPropertyTransferRegistry] = None,
    enforcers: Optional[GPUEnforcerRegistry] = None,
) -> GPURequirementAnalysis:
    """Analyze all explicit MGP anchors without mutating or lowering the DAG."""

    if property_transfers is None:
        property_transfers = create_default_property_transfer_registry()
    if lowerers is None:
        lowerers = create_default_gpu_lowerer_registry()
    if source_providers is None:
        source_providers = create_default_gpu_source_provider_registry()
    if enforcers is None:
        enforcers = create_default_gpu_enforcer_registry()
    resolver = GPURequirementResolver(enforcers)
    operators = tuple(_post_order_logical(root))
    consumers = _logical_consumers(operators)
    logical_node_ids = {
        op: f"logical-{ordinal:04d}-{type(op).__name__}"
        for ordinal, op in enumerate(operators)
    }

    memberships: Dict[LogicalOperator, list[GPUComponentMembership]] = {
        op: [] for op in operators
    }
    components: Dict[MapGroupPartitions, MGPGPUComponent] = {}
    anchors = tuple(
        op
        for op in operators
        if type(op) is MapGroupPartitions
        and _is_component_anchor(op, consumers, lowerers)
    )
    if anchors and data_context.checkpoint_config is not None:
        raise GPUPlanningError(
            GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
            "consumer=MapGroupPartitions; checkpointing is not supported by "
            "the gate-2 GPU stage",
        )
    for anchor in anchors:
        component, component_memberships = _analyze_anchor(
            anchor,
            data_context,
            logical_node_ids,
            lowerers,
            source_providers,
            property_transfers,
            resolver,
        )
        components[anchor] = component
        for op, membership in component_memberships:
            memberships[op].append(membership)

    _validate_source_demands(components.values(), logical_node_ids)
    annotations = {
        op: GPUOperatorAnnotation(
            logical_node_id=logical_node_ids[op],
            memberships=tuple(memberships[op]),
        )
        for op in operators
    }
    return GPURequirementAnalysis(
        annotations=MappingProxyType(annotations),
        components_by_anchor=MappingProxyType(components),
        logical_node_ids=MappingProxyType(logical_node_ids),
    )


def _analyze_anchor(
    anchor: MapGroupPartitions,
    data_context: DataContext,
    logical_node_ids: Mapping[LogicalOperator, str],
    lowerers: GPULogicalLowererRegistry,
    source_providers: GPUReadSourceProviderRegistry,
    property_transfers: GPUPropertyTransferRegistry,
    resolver: GPURequirementResolver,
) -> Tuple[
    MGPGPUComponent,
    Tuple[Tuple[LogicalOperator, GPUComponentMembership], ...],
]:
    ordered = _component_prefix(anchor, lowerers)
    source = ordered[0]
    if lowerers.role_for(source) is not GPUComponentRole.SOURCE:
        raise AssertionError("a GPU component prefix must begin at a SOURCE lowerer")
    analyses: Dict[LogicalOperator, GPULogicalAnalysis] = {}
    for op in ordered:
        analysis = lowerers.analyze(op, data_context)
        if not analysis.accepted:
            raise GPUPlanningError(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"consumer={_op_label(anchor)}; operator={_op_label(op)}; "
                f"capability={analysis.reason}",
            )
        analyses[op] = analysis

    execution_profiles = _effective_execution_profiles(
        ordered,
        analyses,
        lowerers,
        source,
        anchor,
        data_context,
    )
    source_requirements = execution_profiles[source]

    partition_ops = tuple(
        op
        for op in ordered
        if lowerers.role_for(op) is GPUComponentRole.PARTITION_CONSUMER
    )
    first_analysis = analyses[partition_ops[0]]
    first_keys = first_analysis.partition_keys
    first_profile = execution_profiles[partition_ops[0]]
    first_count = _partition_count(
        first_analysis,
        first_profile.compatibility_key.max_workers,
    )
    provider = source_providers.select(
        source,
        GPUReadSourceDemand(
            group_keys=first_keys,
            target_ranges=first_count,
            num_workers=source_requirements.compatibility_key.max_workers,
        ),
    )
    initial_state = GPUPropertyState(
        facts=(
            PropertyFact(
                RangePartitioning(
                    (first_keys[0],),
                    "analysis:" + logical_node_ids[source],
                ),
                Evidence.OPERATOR_GUARANTEE,
            ),
            PropertyFact(
                CompletePartitionBundles(),
                Evidence.OPERATOR_GUARANTEE,
            ),
        ),
        partition_count=first_count,
    )

    component_id = (
        "gpu-component:" f"{logical_node_ids[source]}->{logical_node_ids[anchor]}"
    )
    state = initial_state
    component_memberships: list[Tuple[LogicalOperator, GPUComponentMembership]] = []
    mgp_requirements: list[MGPRequirement] = []
    for ordinal, op in enumerate(ordered):
        input_state = state
        execution_requirements = execution_profiles[op]
        input_resolution = GPUEdgeResolution.same_stage()
        if ordinal:
            upstream_requirements = execution_profiles[ordered[ordinal - 1]]
            if execution_requirements != upstream_requirements:
                next_partition = _next_partition_consumer(
                    ordered,
                    ordinal,
                    lowerers,
                )
                next_analysis = analyses[next_partition]
                next_profile = execution_profiles[next_partition]
                input_resolution = resolver.require_same_stage(
                    consumer=op,
                    required_keys=next_analysis.partition_keys,
                    required_partition_count=_partition_count(
                        next_analysis,
                        next_profile.compatibility_key.max_workers,
                    ),
                    available=input_state,
                    upstream_execution_requirements=upstream_requirements,
                    downstream_execution_requirements=execution_requirements,
                    detail=(
                        "partition-dependent component has mismatched execution "
                        "profiles; "
                        f"upstream_profile={upstream_requirements!r}; "
                        f"boundary_operator={_op_label(op)}; "
                        f"downstream_profile={execution_requirements!r}"
                    ),
                )
        if input_resolution.kind is GPUEdgeResolutionKind.ENFORCER:
            assert input_resolution.established_state is not None
            input_state = input_resolution.established_state
        role = lowerers.role_for(op)
        if role is GPUComponentRole.SOURCE:
            output_state = state
        elif role is GPUComponentRole.UNARY:
            output_state = property_transfers.transfer(op, input_state)
        else:
            assert role is GPUComponentRole.PARTITION_CONSUMER
            op_analysis = analyses[op]
            keys = op_analysis.partition_keys
            count = _partition_count(
                op_analysis,
                execution_requirements.compatibility_key.max_workers,
            )
            partition_resolution = resolver.require_partitioned_edge(
                consumer=op,
                required_keys=keys,
                required_partition_count=count,
                available=input_state,
                execution_requirements=execution_requirements,
                detail="partition-dependent input cannot remain actor-local",
            )
            if partition_resolution.kind is GPUEdgeResolutionKind.ENFORCER:
                input_resolution = partition_resolution
                assert partition_resolution.established_state is not None
                input_state = partition_resolution.established_state
            mgp_requirements.append(MGPRequirement(op, keys, count, input_resolution))
            output_state = property_transfers.transfer(op, input_state)
        component_memberships.append(
            (
                op,
                GPUComponentMembership(
                    component_id=component_id,
                    ordinal=ordinal,
                    input_state=input_state,
                    output_state=output_state,
                    input_resolution=input_resolution,
                    execution_requirements=execution_requirements,
                ),
            )
        )
        state = output_state

    return (
        MGPGPUComponent(
            component_id=component_id,
            source=source,
            operators=ordered,
            anchor=anchor,
            source_demand=GPUSourceDemand(
                source=source,
                range_key=first_keys[0],
                partition_count=first_count,
                execution_requirements=source_requirements,
                provider=provider,
            ),
            mgp_requirements=tuple(mgp_requirements),
            execution_requirements=source_requirements,
        ),
        tuple(component_memberships),
    )


def _effective_execution_profiles(
    ordered: Tuple[LogicalOperator, ...],
    analyses: Mapping[LogicalOperator, GPULogicalAnalysis],
    lowerers: GPULogicalLowererRegistry,
    source: LogicalOperator,
    anchor: MapGroupPartitions,
    data_context: DataContext,
) -> Mapping[LogicalOperator, GPUExecutionRequirements]:
    """Assign the exact actor profile used by every logical occurrence.

    A source may declare its own profile.  Otherwise it adopts the first
    segment's unary profile when one is available, or the source-to-MGP default.
    Built-in Read remote arguments are reconciled with that inherited profile.
    Partition consumers inherit the profile feeding them.  Every later unary
    profile change therefore appears as an explicit adjacent segment boundary.
    """

    unary_profiles: Dict[LogicalOperator, GPUExecutionRequirements] = {}
    for op in ordered:
        if lowerers.role_for(op) is not GPUComponentRole.UNARY:
            continue
        requirements = analyses[op].execution_requirements
        if requirements is None:
            raise GPUPlanningError(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"consumer={_op_label(anchor)}; registered unary lowerer did "
                "not declare execution requirements for "
                f"{_op_label(op)}",
            )
        unary_profiles[op] = requirements

    source_declared = analyses[source].execution_requirements
    first_after_source = ordered[1]
    if lowerers.role_for(first_after_source) is GPUComponentRole.UNARY:
        initial_declared = unary_profiles[first_after_source]
    elif type(source) is not Read and source_declared is not None:
        # A custom source with no leading unary already pins the first segment;
        # do not consult live cluster defaults that cannot affect this plan.
        initial_declared = source_declared
    else:
        initial_declared = default_gpu_execution_requirements(data_context)
    if type(source) is Read:
        initial_effective = reconcile_read_remote_args(
            initial_declared,
            source.ray_remote_args,
            consumer=_op_label(anchor),
        )
        coalesce_initial_unary = True
    elif source_declared is not None:
        initial_effective = source_declared
        coalesce_initial_unary = False
    else:
        initial_effective = initial_declared
        coalesce_initial_unary = True

    profiles: Dict[LogicalOperator, GPUExecutionRequirements] = {
        source: initial_effective
    }
    current = initial_effective
    in_initial_segment = coalesce_initial_unary
    for op in ordered[1:]:
        role = lowerers.role_for(op)
        if role is GPUComponentRole.UNARY:
            declared = unary_profiles[op]
            effective = (
                initial_effective
                if in_initial_segment and declared == initial_declared
                else declared
            )
            if effective != current:
                in_initial_segment = False
            current = effective
        else:
            assert role is GPUComponentRole.PARTITION_CONSUMER
        profiles[op] = current
    return profiles


def _next_partition_consumer(
    ordered: Tuple[LogicalOperator, ...],
    start: int,
    lowerers: GPULogicalLowererRegistry,
) -> LogicalOperator:
    for op in ordered[start:]:
        if lowerers.role_for(op) is GPUComponentRole.PARTITION_CONSUMER:
            return op
    raise AssertionError("a profile boundary must precede the component anchor")


def _component_prefix(
    anchor: MapGroupPartitions,
    lowerers: GPULogicalLowererRegistry,
) -> Tuple[LogicalOperator, ...]:
    reverse: list[LogicalOperator] = []
    cursor: LogicalOperator = anchor
    while True:
        reverse.append(cursor)
        role = lowerers.role_for(cursor)
        if role is GPUComponentRole.SOURCE:
            break
        if role not in (
            GPUComponentRole.UNARY,
            GPUComponentRole.PARTITION_CONSUMER,
        ):
            raise GPUPlanningError(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"consumer={_op_label(anchor)}; unsupported logical operator "
                f"on partition-dependent path: {_op_label(cursor)}",
            )
        if len(cursor.input_dependencies) != 1:
            raise GPUPlanningError(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"consumer={_op_label(anchor)}; {_op_label(cursor)} requires "
                "exactly one input for GPU lowering",
            )
        cursor = cursor.input_dependencies[0]
    reverse.reverse()
    return tuple(reverse)


def _partition_count(analysis: GPULogicalAnalysis, max_workers: int) -> int:
    return (
        2 * max_workers
        if analysis.requested_partitions is None
        else analysis.requested_partitions
    )


def _validate_source_demands(
    components: Iterable[MGPGPUComponent],
    logical_node_ids: Mapping[LogicalOperator, str],
) -> None:
    by_source: Dict[LogicalOperator, set[Tuple[str, int]]] = {}
    for component in components:
        demand = component.source_demand
        by_source.setdefault(demand.source, set()).add(
            (demand.range_key, demand.partition_count)
        )
    for source, demands in by_source.items():
        if len(demands) <= 1:
            continue
        ordered = tuple(sorted(demands))
        raise GPUPlanningError(
            GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS,
            f"source={logical_node_ids[source]}; branch source demands disagree; "
            f"required_key_count_pairs={ordered!r}",
        )


def _post_order_logical(root: LogicalOperator) -> Iterable[LogicalOperator]:
    visited: set[LogicalOperator] = set()

    def visit(op: LogicalOperator) -> Iterable[LogicalOperator]:
        if op in visited:
            return
        visited.add(op)
        for child in op.input_dependencies:
            yield from visit(child)
        yield op

    yield from visit(root)


def _logical_consumers(
    operators: Sequence[LogicalOperator],
) -> Mapping[LogicalOperator, Tuple[LogicalOperator, ...]]:
    consumers: Dict[LogicalOperator, list[LogicalOperator]] = {
        op: [] for op in operators
    }
    for op in operators:
        for child in op.input_dependencies:
            consumers[child].append(op)
    return {op: tuple(op_consumers) for op, op_consumers in consumers.items()}


def _is_component_anchor(
    op: MapGroupPartitions,
    consumers: Mapping[LogicalOperator, Tuple[LogicalOperator, ...]],
    lowerers: GPULogicalLowererRegistry,
) -> bool:
    """Return whether some consumer branch terminates at this MGP.

    A linear preserving ``MGP -> MGP`` run has only its final MGP as an
    anchor.  An earlier MGP is also an anchor when a branch leaves that run and
    must consume its ordinary Ray-block output independently.
    """

    downstream = consumers.get(op, ())
    return not downstream or not all(
        _all_paths_reach_next_mgp(consumer, consumers, lowerers)
        for consumer in downstream
    )


def _all_paths_reach_next_mgp(
    op: LogicalOperator,
    consumers: Mapping[LogicalOperator, Tuple[LogicalOperator, ...]],
    lowerers: GPULogicalLowererRegistry,
) -> bool:
    role = lowerers.role_for(op)
    if role is GPUComponentRole.PARTITION_CONSUMER:
        return True
    if role is not GPUComponentRole.UNARY:
        return False
    downstream = consumers.get(op, ())
    return bool(downstream) and all(
        _all_paths_reach_next_mgp(consumer, consumers, lowerers)
        for consumer in downstream
    )


_REGISTERED_GPU_LOWERERS: Optional[GPULogicalLowererRegistry] = None


def _registered_gpu_lowerers() -> GPULogicalLowererRegistry:
    """Lazily initialize the canonical built-in plus extension registry."""

    global _REGISTERED_GPU_LOWERERS
    if _REGISTERED_GPU_LOWERERS is not None:
        return _REGISTERED_GPU_LOWERERS

    from ray.data._internal.planner.gpu_lowering.map_batches_lowerer import (
        GPUMapBatchesLowerer,
    )
    from ray.data._internal.planner.gpu_lowering.map_group_partitions_lowerer import (
        GPUMapGroupPartitionsLowerer,
    )
    from ray.data._internal.planner.gpu_lowering.parquet_read_lowerer import (
        ParquetRangeReadLowerer,
    )

    registry = GPULogicalLowererRegistry()
    registry.register(Read, ParquetRangeReadLowerer())
    registry.register(MapBatches, GPUMapBatchesLowerer())
    registry.register(MapGroupPartitions, GPUMapGroupPartitionsLowerer())
    _REGISTERED_GPU_LOWERERS = registry
    return registry


def register_gpu_logical_lowerer(
    logical_type: type[LogicalOperator],
    lowerer: GPULogicalLowerer,
) -> None:
    """Register a logical lowerer used by all subsequent Planner.plan calls."""

    _registered_gpu_lowerers().register(logical_type, lowerer)


def unregister_gpu_logical_lowerer(
    logical_type: type[LogicalOperator],
    *,
    expected: Optional[GPULogicalLowerer] = None,
) -> GPULogicalLowerer:
    """Unregister a non-built-in lowerer for plugin teardown or tests."""

    if logical_type in (Read, MapBatches, MapGroupPartitions):
        raise ValueError("Built-in GPU logical lowerers cannot be removed")
    return _registered_gpu_lowerers().unregister(
        logical_type,
        expected=expected,
    )


def create_default_gpu_lowerer_registry() -> GPULogicalLowererRegistry:
    """Snapshot all built-in and globally registered logical lowerers."""

    return _registered_gpu_lowerers().copy()


def _op_label(op: Optional[LogicalOperator]) -> str:
    if op is None:
        return "none"
    return f"{type(op).__name__}({op.name})"


def _property_label(state: GPUPropertyState) -> str:
    return f"facts={state.facts!r}, count={state.partition_count}"


__all__ = [
    "GPUComponentMembership",
    "GPUEdgeResolution",
    "GPUEdgeResolutionKind",
    "GPUEnforcementRequest",
    "GPUEnforcerPlan",
    "GPUEnforcerRegistry",
    "GPUOperatorAnnotation",
    "GPUPlanningError",
    "GPUPlanningErrorReason",
    "GPUPropertyEnforcer",
    "GPUPropertyState",
    "GPUPropertyTransferRegistry",
    "GPURequirementAnalysis",
    "GPURequirementResolver",
    "MGPGPUComponent",
    "MGPRequirement",
    "GPUSourceDemand",
    "analyze_gpu_requirements",
    "create_default_gpu_enforcer_registry",
    "create_default_gpu_lowerer_registry",
    "create_default_property_transfer_registry",
    "register_gpu_logical_lowerer",
    "register_gpu_property_enforcer",
    "register_gpu_property_transfer",
    "unregister_gpu_logical_lowerer",
    "unregister_gpu_property_enforcer",
    "unregister_gpu_property_transfer",
]
