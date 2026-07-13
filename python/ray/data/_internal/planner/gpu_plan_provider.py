"""Gate-2 construction for analyzed MGP-bearing GPU components."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Mapping, Optional, Tuple

from ray.data._internal.execution.gpu import (
    GPUCompositionError,
    GPUOperatorCreationOptions,
    create_gpu_operator,
)
from ray.data._internal.execution.gpu.region_compiler import compile_gpu_region
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators import MapGroupPartitions
from ray.data._internal.planner.gpu_lowering import (
    GPUComponentRole,
    GPULogicalLowererRegistry,
    GPULoweringContext,
)
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)
from ray.data._internal.planner.gpu_read_source_provider import (
    GPUReadSourceProviderRegistry,
    create_default_gpu_source_provider_registry,
)
from ray.data._internal.planner.gpu_requirement_analysis import (
    GPUComponentMembership,
    GPUEdgeResolution,
    GPUEdgeResolutionKind,
    GPUEnforcerRegistry,
    GPUPropertyTransferRegistry,
    GPUPropertyState,
    GPURequirementAnalysis,
    GPURequirementResolver,
    MGPGPUComponent,
    MGPRequirement,
    analyze_gpu_requirements,
    create_default_gpu_enforcer_registry,
    create_default_gpu_lowerer_registry,
    create_default_property_transfer_registry,
)
from ray.data.context import DataContext


@dataclass(frozen=True)
class GPUPlannedStage:
    root: PhysicalOperator
    op_map: Mapping[PhysicalOperator, LogicalOperator]


def create_gpu_gate2_planner(
    root: LogicalOperator,
    data_context: DataContext,
    *,
    lowerers: Optional[GPULogicalLowererRegistry] = None,
    source_providers: Optional[GPUReadSourceProviderRegistry] = None,
    property_transfers: Optional[GPUPropertyTransferRegistry] = None,
    enforcers: Optional[GPUEnforcerRegistry] = None,
) -> "GPUGate2Planner":
    if lowerers is None:
        lowerers = create_default_gpu_lowerer_registry()
    if source_providers is None:
        source_providers = create_default_gpu_source_provider_registry()
    if property_transfers is None:
        property_transfers = create_default_property_transfer_registry()
    else:
        property_transfers = property_transfers.copy()
    if enforcers is None:
        enforcers = create_default_gpu_enforcer_registry()
    analysis = analyze_gpu_requirements(
        root,
        data_context,
        lowerers=lowerers,
        source_providers=source_providers,
        property_transfers=property_transfers,
        enforcers=enforcers,
    )
    return GPUGate2Planner(
        analysis,
        lowerers=lowerers,
        property_transfers=property_transfers,
        resolver=GPURequirementResolver(enforcers),
    )


class GPUGate2Planner:
    """Lower, resolve, and compose one already-analyzed component at a time."""

    def __init__(
        self,
        analysis: GPURequirementAnalysis,
        *,
        lowerers: GPULogicalLowererRegistry,
        property_transfers: GPUPropertyTransferRegistry,
        resolver: GPURequirementResolver,
    ) -> None:
        self.analysis = analysis
        self._lowerers = lowerers
        self._property_transfers = property_transfers
        self._resolver = resolver

    def try_plan(
        self,
        logical_op: LogicalOperator,
        data_context: DataContext,
    ) -> Optional[GPUPlannedStage]:
        if type(logical_op) is not MapGroupPartitions:
            return None
        component = self.analysis.components_by_anchor.get(logical_op)
        if component is None:
            # A non-anchor MGP is consumed by a larger maximal component.  It
            # should never be reached recursively unless the analyzed DAG and
            # planner traversal disagree.
            return None
        return self._plan_component(component, data_context)

    def _plan_component(
        self,
        component: MGPGPUComponent,
        data_context: DataContext,
    ) -> GPUPlannedStage:
        target_bytes = data_context.target_max_block_size
        if (
            not isinstance(target_bytes, int)
            or isinstance(target_bytes, bool)
            or target_bytes <= 0
        ):
            raise GPUPlanningError(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"consumer={component.anchor.name}; "
                "DataContext.target_max_block_size must be positive",
            )
        demand = component.source_demand
        first_group_keys = component.mgp_requirements[0].keys
        context = GPULoweringContext(
            stage_id=component.component_id,
            logical_node_id=self.analysis.logical_node_ids[component.source],
            group_keys=first_group_keys,
            source_provider=demand.provider,
            execution_requirements=component.execution_requirements,
            target_max_block_size=int(target_bytes),
            retried_io_errors=tuple(data_context.retried_io_errors),
            target_ranges=demand.partition_count,
            num_workers=(
                component.execution_requirements.compatibility_key.max_workers
            ),
        )

        op_map: Dict[PhysicalOperator, LogicalOperator] = {}
        physical_input: Optional[PhysicalOperator] = None
        pending_specs = []
        pending_options = []
        pending_logical_ops = []
        pending_input: Optional[PhysicalOperator] = None
        previous_annotated_profile = None
        concrete_state: Optional[GPUPropertyState] = None

        def finalize_pending(*, terminal: bool) -> PhysicalOperator:
            nonlocal pending_specs, pending_options, pending_logical_ops, pending_input
            if not pending_specs or pending_input is None:
                raise AssertionError("cannot finalize an empty GPU stage")
            try:
                compiled = compile_gpu_region(pending_specs).for_gate2_stage(
                    specialized_read_parallelism_resolved=any(
                        self._lowerers.role_for(op) is GPUComponentRole.SOURCE
                        for op in pending_logical_ops
                    ),
                    uses_terminal_output_adapter=terminal,
                )
            except GPUPlanningError:
                raise
            except ValueError as error:
                raise GPUPlanningError(
                    GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                    f"consumer={component.anchor.name}; gate-2 final GPU "
                    f"stage composition refused: {error}",
                ) from error

            try:
                options = _composed_creation_options(pending_options, compiled)
                physical = create_gpu_operator(
                    compiled,
                    pending_input,
                    data_context,
                    options,
                )
            except GPUPlanningError:
                raise
            except ValueError as error:
                raise GPUPlanningError(
                    GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                    f"consumer={component.anchor.name}; gate-2 GPU operator "
                    f"factory or boundary adapter refused the compiled stage: {error}",
                ) from error
            physical.set_logical_operators(*pending_logical_ops)
            op_map[physical] = pending_logical_ops[-1]
            pending_specs = []
            pending_options = []
            pending_logical_ops = []
            pending_input = None
            return physical

        for ordinal, op in enumerate(component.operators):
            membership = _membership_for(self.analysis, op, component.component_id)
            group_keys = _group_keys_for(component, ordinal)
            profile_changed = (
                previous_annotated_profile is not None
                and membership.execution_requirements != previous_annotated_profile
            )
            if profile_changed and (
                membership.input_resolution.kind is not GPUEdgeResolutionKind.ENFORCER
            ):
                raise AssertionError(
                    "an analyzed execution-profile boundary requires an enforcer"
                )

            if membership.input_resolution.kind is GPUEdgeResolutionKind.ENFORCER:
                upstream = finalize_pending(terminal=False)
                physical_input, enforced_state = _apply_enforcer(
                    membership.input_resolution,
                    upstream,
                    data_context,
                    op_map,
                    self._resolver,
                )
                context = replace(
                    context,
                    input_properties=enforced_state.facts,
                    execution_requirements=membership.execution_requirements,
                )
                concrete_state = enforced_state

            op_context = context.for_operator(
                logical_node_id=self.analysis.logical_node_ids[op],
                group_keys=group_keys,
            )
            role = self._lowerers.role_for(op)
            operator_input_state = concrete_state
            if role is GPUComponentRole.SOURCE:
                lowered = self._lowerers.lower(op, op_context)
                operator_output_state = GPUPropertyState(
                    facts=lowered.spec.output_properties,
                    partition_count=demand.partition_count,
                )
            else:
                if operator_input_state is None:
                    raise AssertionError(
                        "a non-source GPU operator requires a concrete input state"
                    )
                lowered, operator_output_state = self._lower_with_property_validation(
                    op,
                    op_context,
                    operator_input_state,
                    component,
                )

            if role is GPUComponentRole.SOURCE:
                artifacts = lowered.source_artifacts
                if artifacts is None:
                    raise GPUPlanningError(
                        GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                        f"consumer={component.anchor.name}; GPU Read lowerer "
                        "returned no concrete source work",
                    )
                source_input = InputDataBuffer(
                    data_context,
                    input_data_factory=artifacts.input_data_factory,
                )
                source_input.set_logical_operators(op)
                op_map[source_input] = op
                physical_input = source_input

            if physical_input is None:
                raise AssertionError("GPU component has no physical input")
            if pending_input is None:
                pending_input = physical_input

            if pending_specs:
                try:
                    # Use the exact pure compiler shared with physical fusion.
                    compile_gpu_region(
                        (*pending_specs, lowered.spec),
                        require_resolved_output=False,
                    )
                except GPUCompositionError as error:
                    required = _next_partition_requirement(component, ordinal)
                    resolution = self._resolver.require_same_stage(
                        consumer=op,
                        required_keys=required.keys,
                        required_partition_count=required.partition_count,
                        available=operator_input_state,
                        upstream_execution_requirements=(
                            pending_specs[-1].execution_requirements
                        ),
                        downstream_execution_requirements=(
                            membership.execution_requirements
                        ),
                        detail=f"same-stage composition refused: {error}",
                    )
                    upstream = finalize_pending(terminal=False)
                    physical_input, enforced_state = _apply_enforcer(
                        resolution,
                        upstream,
                        data_context,
                        op_map,
                        self._resolver,
                    )
                    context = replace(
                        context,
                        input_properties=enforced_state.facts,
                        execution_requirements=membership.execution_requirements,
                    )
                    concrete_state = enforced_state
                    operator_input_state = enforced_state
                    op_context = context.for_operator(
                        logical_node_id=self.analysis.logical_node_ids[op],
                        group_keys=group_keys,
                    )
                    lowered, operator_output_state = (
                        self._lower_with_property_validation(
                            op,
                            op_context,
                            operator_input_state,
                            component,
                        )
                    )
                    pending_input = physical_input

            pending_specs.append(lowered.spec)
            pending_options.append(lowered.creation_options)
            pending_logical_ops.append(op)
            context = lowered.downstream_context
            concrete_state = operator_output_state
            previous_annotated_profile = membership.execution_requirements

        root = finalize_pending(terminal=True)
        return GPUPlannedStage(root=root, op_map=op_map)

    def _lower_with_property_validation(
        self,
        logical_op: LogicalOperator,
        context: GPULoweringContext,
        input_state: GPUPropertyState,
        component: MGPGPUComponent,
    ):
        """Lower one non-source op and verify its concrete property contract."""

        lowered = self._lowerers.lower(logical_op, context)
        try:
            expected_state = self._property_transfers.transfer(
                logical_op,
                input_state,
            )
            if not isinstance(expected_state, GPUPropertyState):
                raise TypeError(
                    "registered GPU property transfer must return GPUPropertyState"
                )
        except GPUPlanningError:
            raise
        except Exception as error:
            raise GPUPlanningError(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"consumer={component.anchor.name}; registered property transfer "
                f"failed for {type(logical_op).__name__}({logical_op.name}): "
                f"{error}",
            ) from error

        actual_facts = lowered.spec.output_properties
        if actual_facts != expected_state.facts:
            raise GPUPlanningError(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"consumer={component.anchor.name}; GPU lowerer and registered "
                f"property transfer disagree for "
                f"{type(logical_op).__name__}({logical_op.name}); "
                f"expected_output_facts={expected_state.facts!r}; "
                f"actual_output_facts={actual_facts!r}",
            )
        return lowered, expected_state


def _membership_for(
    analysis: GPURequirementAnalysis,
    op: LogicalOperator,
    component_id: str,
) -> GPUComponentMembership:
    return next(
        membership
        for membership in analysis.annotations[op].memberships
        if membership.component_id == component_id
    )


def _group_keys_for(
    component: MGPGPUComponent,
    ordinal: int,
) -> Tuple[str, ...]:
    remaining = set(component.operators[ordinal:])
    for requirement in component.mgp_requirements:
        if requirement.operator in remaining:
            return requirement.keys
    return component.mgp_requirements[-1].keys


def _next_partition_requirement(
    component: MGPGPUComponent,
    ordinal: int,
) -> MGPRequirement:
    remaining = set(component.operators[ordinal:])
    return next(
        requirement
        for requirement in component.mgp_requirements
        if requirement.operator in remaining
    )


def _apply_enforcer(
    resolution: GPUEdgeResolution,
    upstream: PhysicalOperator,
    data_context: DataContext,
    op_map: Dict[PhysicalOperator, LogicalOperator],
    resolver: GPURequirementResolver,
):
    if (
        resolution.kind is not GPUEdgeResolutionKind.ENFORCER
        or resolution.enforcer is None
        or resolution.request is None
    ):
        raise AssertionError("an unsatisfied edge requires a selected enforcer")
    plan = resolution.enforcer.plan(
        resolution.request,
        upstream=upstream,
        data_context=data_context,
    )
    resolver.validate_enforcer_state(resolution.request, plan.output_state)
    if plan.output_state != resolution.established_state:
        raise GPUPlanningError(
            GPUPlanningErrorReason.EXCHANGE_REQUIRED,
            "selected GPU enforcer materialized a property state different "
            "from its analyzed contract",
        )
    op_map.update(plan.op_map)
    return plan.physical_input, plan.output_state


def _composed_creation_options(
    options: list[GPUOperatorCreationOptions],
    spec,
) -> GPUOperatorCreationOptions:
    result = options[0]
    for downstream in options[1:]:
        result = result.for_fused_spec(downstream, spec)
    return result


def raise_mgp_without_gpu_gate2(
    logical_op: MapGroupPartitions,
    *,
    gpu_enabled: bool,
) -> None:
    detail = (
        "MapGroupPartitions has no CPU/stock fallback and requires "
        "DataContext.enable_gpu_operator_fusion=True"
        if not gpu_enabled
        else "MapGroupPartitions was not claimed by an analyzed maximal GPU "
        "component and has no CPU/stock fallback"
    )
    raise GPUPlanningError(
        GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
        f"consumer={logical_op.name}; {detail}",
    )


__all__ = [
    "GPUGate2Planner",
    "GPUPlannedStage",
    "GPUPlanningError",
    "GPUPlanningErrorReason",
    "create_gpu_gate2_planner",
    "raise_mgp_without_gpu_gate2",
]
