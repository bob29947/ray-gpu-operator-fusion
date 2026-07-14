"""Tests for per-node GPU planning and the no-I/O demand prepass."""

from dataclasses import dataclass

import pytest

from ray.data import ActorPoolStrategy, UDFProperties
from ray.data._internal.execution.gpu.operator import (
    GPUOperatorCreationOptions,
    PlannedGPUOperator,
)
from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUOperatorSpec,
    GPUTransformSpec,
    RANGE_DATA,
    RANGE_WORK,
    RuntimeKey,
)
from ray.data._internal.execution.gpu.properties import RangePartitioning
from ray.data._internal.execution.gpu.runtime import (
    TransformRuntimeRegistration,
    register_gpu_transform_runtime,
    unregister_gpu_transform_runtime,
)
from ray.data._internal.execution.gpu.cover import validate_gpu_execution_plan
from ray.data._internal.execution.operators.actor_pool_gpu_operator import (
    ActorPoolGPUOperator,
)
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.optimizers import PhysicalOptimizer
from ray.data._internal.logical.operators import (
    InputData,
    MapBatches,
    MapGroupPartitions,
    Read,
    Write,
)
from ray.data._internal.planner import gpu_plan_provider
from ray.data._internal.planner.gpu_lowering.interfaces import (
    GPUDecline,
    GPURequirementDemand,
    GPULogicalAnalysis,
    GPULoweredOperator,
    GPUPlan,
)
from ray.data._internal.planner.gpu_plan_provider import (
    GPURequiredFusionPlanner,
    create_gpu_required_fusion_planner,
    register_gpu_logical_lowerer,
    register_gpu_source_provider,
    unregister_gpu_logical_lowerer,
    unregister_gpu_source_provider,
)
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)
from ray.data._internal.planner.gpu_read_source_provider import (
    GPUReadSourceArtifacts,
    GPUReadSourceAnalysis,
    GPUReadSourceDemand,
    GPUReadSourceLowering,
)
from ray.data._internal.planner.gpu_requirement_analysis import (
    RangePartitionDemand,
    _normalize_demands,
)
from ray.data._internal.planner.gpu_execution_profile import (
    canonical_gpu_execution_requirements,
)
from ray.data._internal.planner.planner import Planner
from ray.data.context import DataContext
from ray.data.datasource.datasource import Datasource
from ray.data.datasource.datasink import Datasink


class _Source(Datasource):
    pass


class _GPUMap:
    def __call__(self, batch):
        return batch


def _partition_fn(frame, context):
    del context
    return frame


def _requirements(size=1):
    return GPUExecutionRequirements(
        compatibility_key=GPUExecutionCompatibilityKey(
            min_workers=size,
            initial_workers=size,
            max_workers=size,
            num_cpus=1,
            num_gpus=1,
        )
    )


class _StubReadProvider:
    provider_name = "test_range"

    def __init__(self):
        self.support_calls = 0
        self.lower_calls = 0

    def analyze(self, logical_op, demands):
        self.support_calls += 1
        if type(logical_op) is not Read or len(demands) != 1:
            return GPUReadSourceAnalysis(False, "logical_read")
        demand = demands[0]
        ranges, workers = demand.resolved_counts()
        return GPUReadSourceAnalysis(
            True,
            provider_state=GPUReadSourceDemand(demand.keys, ranges, workers),
            execution_requirements=demand.execution_requirements,
        )

    def lower(self, logical_op, request):
        self.lower_calls += 1
        demand = request.provider_state
        spec = GPUOperatorSpec(
            transforms=(
                GPUTransformSpec("read_parquet_range", RANGE_WORK, RANGE_DATA),
            ),
            execution_requirements=request.execution_requirements,
            output_facts=(RangePartitioning((demand.group_keys[0],), "test-scheme"),),
        )
        return GPUReadSourceLowering(
            spec,
            GPUOperatorCreationOptions(name="GPURead"),
            GPUReadSourceArtifacts(lambda _target_size: [], "range-work"),
        )


def _read():
    source = _Source()
    return Read(source, source, parallelism=1)


def _map(parent, *, preserves=True, batch_format="cudf", size=1):
    return MapBatches(
        fn=_GPUMap,
        input_dependencies=[parent],
        batch_size=8,
        batch_format=batch_format,
        compute=ActorPoolStrategy(size=size),
        ray_remote_args={"num_cpus": 1, "num_gpus": 1},
        udf_properties=(
            UDFProperties(preserves_partitioning=True) if preserves else None
        ),
    )


def _mgp(parent, *, partitions=2):
    return MapGroupPartitions(
        key=["user", "card"],
        num_partitions=partitions,
        fn=_partition_fn,
        input_dependencies=[parent],
    )


def _context():
    context = DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    context.checkpoint_config = None
    return context


def _planner(root, provider=None):
    context = _context()
    provider = provider or _StubReadProvider()
    return (
        create_gpu_required_fusion_planner(
            root,
            context,
            source_provider=provider,
        ),
        context,
        provider,
    )


def _accept_local_gate(monkeypatch):
    monkeypatch.setattr(
        gpu_plan_provider,
        "_local_implementation_gate",
        lambda lowered: None,
    )


def _planned(root):
    return [
        operator
        for operator in root.post_order_iter()
        if isinstance(operator, PlannedGPUOperator)
    ]


def _range_demand(planner, logical_op):
    return next(
        (
            demand
            for demand in planner.requirement_analysis.demands_for(logical_op)
            if isinstance(demand, RangePartitionDemand)
        ),
        None,
    )


def _materialize_gpu_tree(planner, logical_op, context):
    child_roots = [
        _materialize_gpu_tree(planner, child, context)
        for child in logical_op.input_dependencies
    ]
    decision = planner.plan_node(logical_op, context)
    assert isinstance(decision, GPUPlan)
    return planner.materialize_planned_node(
        logical_op, child_roots, decision, context
    ).root


@dataclass(frozen=True)
class _SyntheticDemand(GPURequirementDemand):
    value: int


def test_demand_normalization_is_generic_and_fail_closed():
    demand = _SyntheticDemand(1)
    assert _normalize_demands((demand, demand)) == (demand,)
    with pytest.raises(GPUPlanningError) as captured:
        _normalize_demands((demand, _SyntheticDemand(2)))
    assert captured.value.reason is GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS


def test_prepass_is_no_io_and_resolves_read_demand_through_preserving_maps():
    read = _read()
    first = _map(read)
    second = _map(first)
    root = _mgp(second, partitions=7)
    planner, _, provider = _planner(root)
    assert provider.support_calls == provider.lower_calls == 0
    demand = _range_demand(planner, read)
    assert demand is not None
    assert demand.keys == ("user", "card")
    assert demand.requested_partitions == 7
    assert demand.execution_requirements is not None
    assert demand.execution_requirements.compatibility_key.max_workers == 1


def test_nonpreserving_map_stops_only_the_range_demand():
    read = _read()
    mapped = _map(read, preserves=False)
    planner, _, _ = _planner(_mgp(mapped))
    assert _range_demand(planner, read) is None
    assert _range_demand(planner, mapped) is not None


def test_conflicting_pre_group_map_profiles_fail_with_stable_reason():
    read = _read()
    root = _mgp(_map(_map(read, size=1), size=2))
    with pytest.raises(GPUPlanningError) as captured:
        _planner(root)
    assert captured.value.reason is GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS


def test_one_three_and_many_nodes_lower_independently(monkeypatch):
    _accept_local_gate(monkeypatch)
    read = _read()
    pre = read
    for _ in range(3):
        pre = _map(pre)
    grouped = _mgp(pre)
    root = grouped
    for _ in range(2):
        root = _map(root)
    planner, context, provider = _planner(root)
    nodes = _planned(_materialize_gpu_tree(planner, root, context))
    assert len(nodes) == 7
    assert [
        tuple(t.kind for t in node.gpu_operator_spec.transforms) for node in nodes
    ] == [
        ("read_parquet_range",),
        ("map_batches", "validate_range_ownership"),
        ("map_batches", "validate_range_ownership"),
        ("map_batches", "validate_range_ownership"),
        ("sort_partition", "build_group_boundaries", "map_group_partitions"),
        ("map_batches",),
        ("map_batches",),
    ]
    assert provider.support_calls == provider.lower_calls == 1


def test_map_batches_plans_without_mgp_or_source_artifacts(monkeypatch):
    _accept_local_gate(monkeypatch)
    read = _read()
    mapped = _map(read)
    planner, context, _ = _planner(mapped)
    read_decision = planner.plan_node(read, context)
    assert isinstance(read_decision, GPUDecline)
    planner.record_stock_fallback(read)
    map_decision = planner.plan_node(mapped, context)
    assert isinstance(map_decision, GPUPlan)
    assert map_decision.lowered.source_artifacts is None
    assert map_decision.lowered.spec.input_requirements == ()
    assert tuple(t.kind for t in map_decision.lowered.spec.transforms) == (
        "map_batches",
    )


def test_source_provider_can_accept_read_without_range_demand(monkeypatch):
    _accept_local_gate(monkeypatch)

    class _ReadOnlyProvider(_StubReadProvider):
        def analyze(self, logical_op, demands):
            self.support_calls += 1
            assert demands == ()
            return GPUReadSourceAnalysis(
                True,
                provider_state=GPUReadSourceDemand(("user",), 1, 1),
                execution_requirements=_requirements(),
            )

    read = _read()
    planner, context, provider = _planner(read, _ReadOnlyProvider())

    decision = planner.plan_node(read, context)

    assert isinstance(decision, GPUPlan)
    assert provider.support_calls == provider.lower_calls == 1


def test_source_provider_dispatch_continues_after_unsupported_and_allows_none_state(
    monkeypatch,
):
    _accept_local_gate(monkeypatch)
    calls = []

    class RejectingProvider:
        provider_name = "rejecting"

        def analyze(self, logical_op, demands):
            del logical_op, demands
            calls.append("reject")
            raise GPUPlanningError(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                "not this source",
            )

    class AcceptingProvider:
        provider_name = "accepting"

        def analyze(self, logical_op, demands):
            del logical_op, demands
            calls.append("accept")
            return GPUReadSourceAnalysis(
                True,
                provider_state=None,
                execution_requirements=_requirements(),
            )

        def lower(self, logical_op, request):
            del logical_op
            assert request.provider_state is None
            calls.append("lower")
            return GPUReadSourceLowering(
                GPUOperatorSpec(
                    (GPUTransformSpec("read", RANGE_WORK, RANGE_DATA),),
                    request.execution_requirements,
                ),
                GPUOperatorCreationOptions(name="GPURead"),
                GPUReadSourceArtifacts(lambda _target_size: []),
            )

    read = _read()
    context = _context()
    planner = GPURequiredFusionPlanner(
        read,
        context,
        source_providers=(RejectingProvider(), AcceptingProvider()),
    )

    assert isinstance(planner.plan_node(read, context), GPUPlan)
    assert calls == ["reject", "accept", "lower"]


def test_main_planner_emits_standalone_planned_map_after_stock_source(monkeypatch):
    _accept_local_gate(monkeypatch)
    source = InputData([])
    mapped = _map(source)
    plan, _ = Planner().plan(LogicalPlan(mapped, _context()))
    operators = tuple(plan.dag.post_order_iter())
    assert len(operators) == 2
    assert isinstance(operators[-1], PlannedGPUOperator)
    assert source in plan.gpu_fusion_state.stock_fallbacks


def test_flag_off_planner_is_fully_stock(monkeypatch):
    monkeypatch.setattr(
        gpu_plan_provider,
        "create_gpu_required_fusion_planner",
        lambda *args, **kwargs: pytest.fail("GPU planner ran with flag disabled"),
    )
    context = _context()
    context.enable_gpu_operator_fusion = False
    plan, _ = Planner().plan(LogicalPlan(_map(InputData([])), context))
    assert not any(
        isinstance(operator, PlannedGPUOperator)
        for operator in plan.dag.post_order_iter()
    )
    assert plan.gpu_fusion_state is None


def test_terminal_write_is_sanctioned_without_gpu_fallback_diagnostic():
    source = InputData([])
    write = Write(Datasink(), input_dependencies=[source])

    plan, _ = Planner().plan(LogicalPlan(write, _context()))

    assert write not in plan.gpu_fusion_state.stock_fallbacks
    assert write in plan.gpu_fusion_state.terminal_logical


def test_gpu_decline_consults_a_future_stock_lowerer(monkeypatch):
    class StockLowererReached(RuntimeError):
        pass

    grouped = MapGroupPartitions(
        key=["user"],
        num_partitions=1,
        fn=_GPUMap,
        input_dependencies=[InputData([])],
    )
    original = Planner.get_plan_fn

    def get_plan_fn(self, logical_op):
        if logical_op is grouped:
            raise StockLowererReached
        return original(self, logical_op)

    monkeypatch.setattr(Planner, "get_plan_fn", get_plan_fn)
    with pytest.raises(StockLowererReached):
        Planner().plan(LogicalPlan(grouped, _context()))


def test_lowerer_programming_errors_propagate(monkeypatch):
    class BrokenUnary(LogicalOperator):
        def __init__(self, parent):
            object.__setattr__(self, "_input_dependencies", [parent])

    class BrokenLowerer:
        def analyze(self, logical_op, data_context):
            del logical_op, data_context
            return GPULogicalAnalysis(True)

        def transfer_demands(self, logical_op, downstream_demands, analysis):
            del logical_op, downstream_demands, analysis
            return ((),)

        def lower(self, logical_op, context):
            del logical_op, context
            raise ValueError("lowerer programming bug")

    lowerer = BrokenLowerer()
    register_gpu_logical_lowerer(BrokenUnary, lowerer)
    try:
        with pytest.raises(ValueError, match="lowerer programming bug"):
            Planner().plan(LogicalPlan(BrokenUnary(InputData([])), _context()))
    finally:
        unregister_gpu_logical_lowerer(BrokenUnary, expected=lowerer)


def test_missing_gpu_and_stock_lowerers_raise_stable_planning_error():
    unknown = LogicalOperator()
    with pytest.raises(GPUPlanningError) as captured:
        Planner().plan(LogicalPlan(unknown, _context()))
    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "no stock physical lowerer exists" in str(captured.value)
    assert "LogicalOperator" in str(captured.value)


def test_ineligible_map_declines_to_stock_without_poisoning_other_nodes(monkeypatch):
    _accept_local_gate(monkeypatch)
    read = _read()
    mapped = _map(read, batch_format="pyarrow")
    planner, context, _ = _planner(mapped)
    planner.record_stock_fallback(read)
    decision = planner.plan_node(mapped, context)
    assert isinstance(decision, GPUDecline)
    assert decision.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "batch_format='cudf'" in decision.detail


def test_mgp_lowers_as_an_unresolved_planned_fragment_without_pre_map(monkeypatch):
    _accept_local_gate(monkeypatch)
    read = _read()
    grouped = _mgp(read)
    planner, context, _ = _planner(grouped)
    assert isinstance(planner.plan_node(read, context), GPUDecline)
    planner.record_stock_fallback(read)
    decision = planner.plan_node(grouped, context)
    assert isinstance(decision, GPUPlan)
    assert decision.lowered.spec.execution_requirements.compatibility_key is None


def test_local_gate_checks_runtime_and_backend_without_constructing(monkeypatch):
    spec = GPUOperatorSpec(
        transforms=(GPUTransformSpec("synthetic", RANGE_DATA, RANGE_DATA),),
        execution_requirements=_requirements(),
    )
    lowered = GPULoweredOperator(spec, GPUOperatorCreationOptions())
    monkeypatch.setattr(
        gpu_plan_provider,
        "supports_gpu_transform_runtime",
        lambda transform: False,
    )
    refusal = gpu_plan_provider._local_implementation_gate(lowered)
    assert refusal is not None and "transform runtime" in refusal.detail
    monkeypatch.setattr(
        gpu_plan_provider,
        "supports_gpu_transform_runtime",
        lambda transform: True,
    )
    monkeypatch.setattr(
        gpu_plan_provider,
        "has_gpu_operator_factory",
        lambda backend: False,
    )
    refusal = gpu_plan_provider._local_implementation_gate(lowered)
    assert refusal is not None and "backend factory" in refusal.detail


def test_registered_unary_extension_fuses_with_read_and_passes_strict_cover():
    runtime_id = "synthetic-unary-extension-v1"
    read_kind = "synthetic_read"
    unary_kind = "synthetic_unary"
    provider_key = "synthetic_read_provider"
    factory_calls = []
    udf_calls = []

    class NeverCalledUDF:
        def __init__(self):
            udf_calls.append("init")

        def __call__(self, batch):
            udf_calls.append("call")
            return batch

    class SyntheticUnary(LogicalOperator):
        def __init__(self, parent):
            object.__setattr__(self, "udf", NeverCalledUDF)
            object.__setattr__(self, "_input_dependencies", [parent])

    class SyntheticUnaryLowerer:
        def analyze(self, logical_op, data_context):
            del data_context
            return GPULogicalAnalysis(type(logical_op) is SyntheticUnary)

        def transfer_demands(self, logical_op, downstream_demands, analysis):
            del logical_op, downstream_demands, analysis
            return ((),)

        def lower(self, logical_op, context):
            assert context.execution_requirements is not None
            return GPULoweredOperator(
                GPUOperatorSpec(
                    transforms=(
                        GPUTransformSpec(
                            unary_kind,
                            RANGE_DATA,
                            RANGE_DATA,
                            logical_op.udf,
                            runtime_id,
                        ),
                    ),
                    execution_requirements=context.execution_requirements,
                ),
                GPUOperatorCreationOptions(name="SyntheticUnaryGPU"),
            )

    context = _context()
    requirements = canonical_gpu_execution_requirements(
        compute=ActorPoolStrategy(size=1),
        resources={"num_cpus": 1, "num_gpus": 1},
        data_context=context,
        consumer="SyntheticUnary",
    )
    read_config = object()

    class SyntheticReadProvider:
        provider_name = provider_key

        def analyze(self, logical_op, demands):
            if type(logical_op) is not Read or demands:
                return GPUReadSourceAnalysis(False, "synthetic_read")
            return GPUReadSourceAnalysis(
                True,
                provider_state=read_config,
                execution_requirements=requirements,
            )

        def lower(self, logical_op, request):
            assert type(logical_op) is Read
            assert request.provider_state is read_config
            return GPUReadSourceLowering(
                GPUOperatorSpec(
                    transforms=(
                        GPUTransformSpec(
                            read_kind,
                            RANGE_WORK,
                            RANGE_DATA,
                            read_config,
                            runtime_id,
                        ),
                    ),
                    execution_requirements=request.execution_requirements,
                ),
                GPUOperatorCreationOptions(name="SyntheticReadGPU"),
                GPUReadSourceArtifacts(lambda _target_size: []),
            )

    def runtime_factory(transform, runtime_context):
        factory_calls.append((transform, runtime_context))
        raise AssertionError("runtime factories must not run during planning")

    read_key = RuntimeKey(runtime_id, read_kind, RANGE_WORK, RANGE_DATA)
    unary_key = RuntimeKey(runtime_id, unary_kind, RANGE_DATA, RANGE_DATA)
    read_runtime = TransformRuntimeRegistration(
        lambda config: config is read_config,
        runtime_factory,
    )
    unary_runtime = TransformRuntimeRegistration(
        lambda config: config is NeverCalledUDF,
        runtime_factory,
    )
    lowerer = SyntheticUnaryLowerer()
    provider = SyntheticReadProvider()
    register_gpu_logical_lowerer(SyntheticUnary, lowerer)
    register_gpu_source_provider(provider)
    register_gpu_transform_runtime(read_key, read_runtime)
    register_gpu_transform_runtime(unary_key, unary_runtime)
    try:
        read = _read()
        root = SyntheticUnary(read)
        plan, _ = Planner().plan(LogicalPlan(root, context))

        assert len(_planned(plan.dag)) == 2
        assert factory_calls == []
        assert udf_calls == []

        optimized = PhysicalOptimizer().optimize(plan)

        assert isinstance(optimized.dag, ActorPoolGPUOperator)
        assert optimized.dag._logical_operators == [read, root]
        regions = tuple(optimized.gpu_fusion_state.materialized_regions.values())
        assert len(regions) == 1
        assert regions[0].operator is optimized.dag
        assert regions[0].attested
        assert tuple(
            transform.kind for transform in optimized.dag.gpu_operator_spec.transforms
        ) == (
            "import_range_work",
            read_kind,
            unary_kind,
            "export_range_data",
        )
        assert factory_calls == []
        assert udf_calls == []
        validate_gpu_execution_plan(optimized)
    finally:
        unregister_gpu_transform_runtime(unary_key, expected=unary_runtime)
        unregister_gpu_transform_runtime(read_key, expected=read_runtime)
        unregister_gpu_source_provider(provider_key, expected=provider)
        unregister_gpu_logical_lowerer(SyntheticUnary, expected=lowerer)
