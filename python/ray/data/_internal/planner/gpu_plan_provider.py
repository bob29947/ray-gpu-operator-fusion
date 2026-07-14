"""Per-node planning for fail-closed GPU physical fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

from ray.data._internal.execution.gpu.operator import (
    PlannedGPUOperator,
    has_gpu_operator_factory,
)
from ray.data._internal.execution.gpu.runtime import supports_gpu_transform_runtime
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators import MapBatches, MapGroupPartitions, Read
from ray.data._internal.planner.gpu_lowering.interfaces import (
    GPUDecision,
    GPUDecline,
    GPUInvalid,
    GPULogicalLowerer,
    GPULoweredOperator,
    GPULoweringContext,
    GPUPlan,
    is_positive_int,
)
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
    GPUReadSourceProvider,
    ParquetRangeGPUReadProvider,
)
from ray.data._internal.planner.gpu_requirement_analysis import (
    GPURequirementAnalysis,
    analyze_gpu_requirements,
)
from ray.data.context import DataContext


@dataclass(frozen=True)
class GPUPlannedStage:
    root: PhysicalOperator
    op_map: Mapping[PhysicalOperator, LogicalOperator]


_GPU_LOGICAL_LOWERERS: Dict[type, GPULogicalLowerer] = {}
_GPU_SOURCE_PROVIDERS: Dict[str, GPUReadSourceProvider] = {}
_BUILTINS_REGISTERED = False


def register_gpu_logical_lowerer(
    logical_type: type,
    lowerer: GPULogicalLowerer,
    *,
    replace_existing: bool = False,
) -> None:
    if logical_type in _GPU_LOGICAL_LOWERERS and not replace_existing:
        raise ValueError(f"GPU logical lowerer already registered: {logical_type!r}")
    _GPU_LOGICAL_LOWERERS[logical_type] = lowerer


def unregister_gpu_logical_lowerer(
    logical_type: type,
    *,
    expected: Optional[GPULogicalLowerer] = None,
) -> Optional[GPULogicalLowerer]:
    lowerer = _GPU_LOGICAL_LOWERERS.get(logical_type)
    if lowerer is None:
        return None
    if expected is not None and lowerer is not expected:
        raise ValueError("registered GPU logical lowerer is not the expected instance")
    return _GPU_LOGICAL_LOWERERS.pop(logical_type)


def register_gpu_source_provider(
    provider: GPUReadSourceProvider,
    *,
    replace_existing: bool = False,
) -> None:
    name = getattr(provider, "provider_name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("GPU source provider requires a nonempty provider_name")
    if name in _GPU_SOURCE_PROVIDERS and not replace_existing:
        raise ValueError(f"GPU source provider already registered: {name!r}")
    _GPU_SOURCE_PROVIDERS[name] = provider


def unregister_gpu_source_provider(
    provider_name: str,
    *,
    expected: Optional[GPUReadSourceProvider] = None,
) -> Optional[GPUReadSourceProvider]:
    provider = _GPU_SOURCE_PROVIDERS.get(provider_name)
    if provider is None:
        return None
    if expected is not None and provider is not expected:
        raise ValueError("registered GPU source provider is not the expected instance")
    return _GPU_SOURCE_PROVIDERS.pop(provider_name)


def _ensure_builtins_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True
    for logical_type, lowerer in (
        (Read, GPUReadLowerer()),
        (MapBatches, GPUMapBatchesLowerer()),
        (MapGroupPartitions, GPUMapGroupPartitionsLowerer()),
    ):
        _GPU_LOGICAL_LOWERERS.setdefault(logical_type, lowerer)
    provider = ParquetRangeGPUReadProvider()
    _GPU_SOURCE_PROVIDERS.setdefault(provider.provider_name, provider)


def get_gpu_logical_lowerer(
    logical_op: LogicalOperator,
) -> Optional[GPULogicalLowerer]:
    _ensure_builtins_registered()
    return _GPU_LOGICAL_LOWERERS.get(type(logical_op))


def create_gpu_required_fusion_planner(
    root: LogicalOperator,
    data_context: DataContext,
    *,
    source_provider: Optional[GPUReadSourceProvider] = None,
) -> "GPURequiredFusionPlanner":
    _ensure_builtins_registered()
    providers = (
        (source_provider,)
        if source_provider is not None
        else tuple(_GPU_SOURCE_PROVIDERS.values())
    )
    return GPURequiredFusionPlanner(root, data_context, source_providers=providers)


class GPURequiredFusionPlanner:
    """Analyze once, then independently lower every registered logical occurrence."""

    def __init__(
        self,
        root: LogicalOperator,
        data_context: DataContext,
        *,
        source_providers: Sequence[GPUReadSourceProvider],
    ) -> None:
        self._source_providers = tuple(source_providers)
        self._analysis: GPURequirementAnalysis = analyze_gpu_requirements(
            root,
            data_context,
            get_gpu_logical_lowerer,
        )
        self._output_contracts = {}

    @property
    def requirement_analysis(self) -> GPURequirementAnalysis:
        return self._analysis

    def plan_node(
        self,
        logical_op: LogicalOperator,
        data_context: DataContext,
    ) -> GPUDecision:
        lowerer = get_gpu_logical_lowerer(logical_op)
        analysis = self._analysis.logical_analysis(logical_op)
        if lowerer is None or analysis is None:
            return GPUDecline(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"no GPU lowerer for {_op_label(logical_op)}",
            )
        if not analysis.accepted:
            return GPUDecline(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"operator={_op_label(logical_op)}; capability={analysis.reason}",
            )
        if data_context.checkpoint_config is not None:
            return GPUDecline(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"operator={_op_label(logical_op)}; checkpointing is unsupported",
            )
        target_bytes = data_context.target_max_block_size
        if not is_positive_int(target_bytes):
            return GPUInvalid(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                "DataContext.target_max_block_size must be positive",
            )
        input_facts, input_execution = self._input_contract(logical_op)
        execution = analysis.execution_requirements or input_execution
        source_provider = None
        source_analysis = None
        if not logical_op.input_dependencies:
            refusals = []
            for provider in self._source_providers:
                try:
                    support = provider.analyze(
                        logical_op,
                        self._analysis.demands_for(logical_op),
                    )
                except GPUPlanningError as error:
                    if error.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY:
                        refusals.append(f"{provider.provider_name}:{error.detail}")
                        continue
                    return _decision_from_error(error)
                if support.accepted:
                    source_provider = provider
                    source_analysis = support.provider_state
                    execution = support.execution_requirements
                    break
                refusals.append(f"{provider.provider_name}:{support.reason}")
            if source_provider is None:
                return GPUDecline(
                    GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                    f"no GPU source provider accepted {_op_label(logical_op)} "
                    f"({', '.join(refusals)})",
                )
        context = GPULoweringContext(
            target_max_block_size=int(target_bytes),
            input_facts=input_facts,
            execution_requirements=execution,
            source_provider=source_provider,
            source_analysis=source_analysis,
            retried_io_errors=tuple(data_context.retried_io_errors),
        )
        try:
            lowered = lowerer.lower(logical_op, context)
        except GPUPlanningError as error:
            return _decision_from_error(error)
        refusal = _local_implementation_gate(lowered)
        if refusal is not None:
            return refusal
        self._output_contracts[logical_op] = (
            lowered.spec.output_facts,
            lowered.spec.execution_requirements,
        )
        return GPUPlan(lowered)

    def materialize_planned_node(
        self,
        logical_op: LogicalOperator,
        physical_children: Sequence[PhysicalOperator],
        decision: GPUPlan,
        data_context: DataContext,
    ) -> GPUPlannedStage:
        lowered = decision.lowered
        artifacts = lowered.source_artifacts
        op_map: Dict[PhysicalOperator, LogicalOperator] = {}
        if artifacts is not None:
            if physical_children:
                raise ValueError("a GPU source cannot consume physical children")
            source_input = InputDataBuffer(
                data_context,
                input_data_factory=artifacts.input_data_factory,
            )
            source_input.set_logical_operators(logical_op)
            input_op = source_input
            op_map[source_input] = logical_op
            owns_input_parallelism = True
        else:
            if len(physical_children) != 1:
                raise ValueError(
                    "V1 GPU logical lowerers require exactly one physical input"
                )
            input_op = physical_children[0]
            owns_input_parallelism = False
        physical = PlannedGPUOperator(
            lowered.spec,
            input_op,
            data_context,
            lowered.creation_options,
            owns_input_parallelism=owns_input_parallelism,
        )
        physical.set_logical_operators(logical_op)
        op_map[physical] = logical_op
        return GPUPlannedStage(physical, op_map)

    def record_stock_fallback(self, logical_op: LogicalOperator) -> None:
        self._output_contracts[logical_op] = ((), None)

    def _input_contract(self, logical_op: LogicalOperator):
        if len(logical_op.input_dependencies) != 1:
            return (), None
        child = logical_op.input_dependencies[0]
        return self._output_contracts.get(child, ((), None))


def _decision_from_error(error: GPUPlanningError) -> GPUDecision:
    result_type = (
        GPUInvalid
        if error.reason is GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS
        else GPUDecline
    )
    return result_type(error.reason, error.detail)


def _local_implementation_gate(
    lowered: GPULoweredOperator,
) -> Optional[GPUDecline]:
    spec = lowered.spec
    for transform in spec.transforms:
        if not supports_gpu_transform_runtime(transform):
            return GPUDecline(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"no valid GPU transform runtime for {transform.runtime_key!r}",
            )
    if not any(
        has_gpu_operator_factory(key)
        for key in spec.execution_requirements.backend_keys
    ):
        return GPUDecline(
            GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
            "no registered GPU backend factory for "
            f"{sorted(spec.execution_requirements.backend_keys)!r}",
        )
    return None


def _op_label(op: LogicalOperator) -> str:
    return f"{type(op).__name__}({op.name})"
