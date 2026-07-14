from dataclasses import replace
from types import SimpleNamespace

import pytest

import ray
import ray.data
from ray.data.context import DataContext
from ray.data._internal.execution.gpu.operator import (
    ExecutableGPUOperator,
    GPUBackendRefusal,
    GPUOperatorCreationOptions,
    GPURequiredFusionError,
    PlannedGPUOperator,
    register_gpu_operator_factory,
    unregister_gpu_operator_factory,
)
from ray.data._internal.execution.gpu.operator_spec import (
    DEVICE_HANDLE,
    RANGE_DATA,
    RAY_BLOCK_STREAM,
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUOperatorSpec,
    GPUStageRetryPolicy,
    GPUTransformSpec,
)
from ray.data._internal.execution.gpu.properties import KeysColocated
from ray.data._internal.execution.gpu.region_compiler import compile_region
from ray.data._internal.execution.gpu.cover import validate_gpu_execution_plan
from ray.data._internal.execution.gpu.planning_state import GPUFusionPlanState
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import LogicalOperator, PhysicalPlan
from ray.data._internal.logical.optimizers import get_physical_ruleset
from ray.data._internal.logical.rules import FuseGPUOperators
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)


_BACKEND = "fusion-test"


class _Logical(LogicalOperator):
    def __init__(self, name, inputs=()):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_input_dependencies", list(inputs))


class _FakeExecutable(PhysicalOperator, ExecutableGPUOperator):
    def __init__(self, spec, input_op, context, options):
        self._spec = spec
        self._options = options or GPUOperatorCreationOptions()
        super().__init__(self._options.name or "FakeGPU", [input_op], context)

    @property
    def gpu_operator_spec(self):
        return self._spec

    @property
    def execution_backend_name(self):
        return _BACKEND

    @property
    def gpu_creation_options(self):
        return self._options


class _StructuralLookalike(PhysicalOperator):
    def __init__(self, spec, input_op, context, options):
        self._spec = spec
        self._options = options or GPUOperatorCreationOptions()
        super().__init__("StructuralLookalike", [input_op], context)

    @property
    def gpu_operator_spec(self):
        return self._spec

    @property
    def execution_backend_name(self):
        return _BACKEND

    @property
    def gpu_creation_options(self):
        return self._options


class _FakeFactory:
    def __init__(self, *, reject=False):
        self.reject = reject
        self.prepare_calls = []
        self.create_calls = []

    def prepare(self, core):
        self.prepare_calls.append(core)
        execution = core.execution_requirements
        ingress = GPUOperatorSpec(
            (
                GPUTransformSpec(
                    "test_import",
                    RAY_BLOCK_STREAM,
                    core.input_payload,
                ),
            ),
            execution,
        )
        egress = GPUOperatorSpec(
            (
                GPUTransformSpec(
                    "test_export",
                    core.output_payload,
                    RAY_BLOCK_STREAM,
                ),
            ),
            execution,
        )
        try:
            return compile_region(core, ingress, egress)
        except ValueError as error:
            raise GPUBackendRefusal(_BACKEND, str(error)) from error

    def supports(self, compiled):
        return not self.reject

    def create(self, compiled, input_op, context, creation_options=None):
        self.create_calls.append((compiled, input_op, creation_options))
        return _FakeExecutable(compiled, input_op, context, creation_options)

    def attests(self, operator, compiled):
        return (
            isinstance(operator, _FakeExecutable)
            and operator.gpu_operator_spec is compiled
            and operator.execution_backend_name == _BACKEND
        )


@pytest.fixture
def factory():
    unregister_gpu_operator_factory(_BACKEND)
    value = _FakeFactory()
    register_gpu_operator_factory(_BACKEND, value)
    try:
        yield value
    finally:
        unregister_gpu_operator_factory(_BACKEND, expected=value)


def _execution(key=None):
    return GPUExecutionRequirements(
        frozenset((_BACKEND,)),
        key or GPUExecutionCompatibilityKey(num_gpus=0),
    )


def _spec(
    kind,
    input_payload=RANGE_DATA,
    output_payload=RANGE_DATA,
    *,
    key=None,
    input_requirements=(),
    output_facts=(),
):
    return GPUOperatorSpec(
        (GPUTransformSpec(kind, input_payload, output_payload),),
        _execution(key),
        tuple(input_requirements),
        tuple(output_facts),
    )


def _linear_plan(count, *, specs=None, context=None):
    context = context or DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    source = InputDataBuffer(context, input_data=[])
    logical = []
    planned = []
    upstream = source
    for index in range(count):
        logical_op = _Logical(
            f"logical_{index}",
            () if not logical else (logical[-1],),
        )
        logical.append(logical_op)
        spec = specs[index] if specs is not None else _spec(f"core_{index}")
        operator = PlannedGPUOperator(
            spec,
            upstream,
            context,
            GPUOperatorCreationOptions(
                name=f"GPU{index}",
                min_rows_per_bundle=1 if index == 0 else None,
                target_max_block_size_override=128 + index,
            ),
        )
        operator.set_logical_operators(logical_op)
        planned.append(operator)
        upstream = operator
    source.set_logical_operators(logical[0])
    state = GPUFusionPlanState()
    state.initialize_logical_universe(logical[-1])
    op_map = {source: logical[0]}
    op_map.update(zip(planned, logical))
    return (
        PhysicalPlan(upstream, op_map, context, gpu_fusion_state=state),
        source,
        tuple(planned),
        tuple(logical),
    )


@pytest.mark.parametrize("count", (1, 3, 10))
def test_compatible_planned_nodes_compile_to_one_stage(factory, count):
    plan, source, planned, logical = _linear_plan(count)

    result = FuseGPUOperators().apply(plan)

    assert isinstance(result.dag, _FakeExecutable)
    assert result.dag.input_dependencies == [source]
    assert tuple(item.kind for item in result.dag.gpu_operator_spec.transforms) == (
        "test_import",
        *(f"core_{index}" for index in range(count)),
        "test_export",
    )
    assert result.dag._logical_operators == list(logical)
    assert set(result.op_map) == {source, result.dag}
    assert result.op_map[result.dag] is logical[-1]
    assert source.output_dependencies == [result.dag]
    assert len(factory.create_calls) == 1
    assert len(result.gpu_fusion_state.materialized_regions) == 1
    assert not any(
        isinstance(operator, PlannedGPUOperator)
        for operator in result.dag.post_order_iter()
    )
    validate_gpu_execution_plan(result)
    assert FuseGPUOperators().apply(result) is result
    assert len(factory.create_calls) == 1


def test_import_export_are_inserted_only_at_region_ends(factory):
    plan, _, _, _ = _linear_plan(3)

    result = FuseGPUOperators().apply(plan)

    kinds = tuple(
        transform.kind for transform in result.dag.gpu_operator_spec.transforms
    )
    assert kinds == (
        "test_import",
        "core_0",
        "core_1",
        "core_2",
        "test_export",
    )
    assert kinds.count("test_import") == 1
    assert kinds.count("test_export") == 1


@pytest.mark.parametrize(
    "specs",
    (
        (
            _spec("upstream"),
            _spec("downstream", input_requirements=(KeysColocated(("key",)),)),
        ),
        (
            _spec("upstream"),
            _spec(
                "downstream",
                key=GPUExecutionCompatibilityKey(num_gpus=1),
            ),
        ),
        (
            _spec("upstream"),
            _spec(
                "downstream",
                key=GPUExecutionCompatibilityKey(
                    num_gpus=0,
                    retry_policy=replace(GPUStageRetryPolicy(), actor_max_restarts=2),
                ),
            ),
        ),
    ),
    ids=("fact", "resource", "retry"),
)
def test_composition_refusal_is_atomic(factory, specs):
    plan, source, planned, _ = _linear_plan(2, specs=specs)
    original_op_map = plan.op_map.copy()

    result = FuseGPUOperators().apply(plan)

    assert result.dag is planned[-1]
    assert result.dag.input_dependencies == [planned[0]]
    assert planned[0].input_dependencies == [source]
    assert result.op_map == original_op_map
    assert not factory.create_calls
    assert len(result.gpu_fusion_state.fusion_refusals) == 1
    refusal = result.gpu_fusion_state.fusion_refusals[0]
    assert refusal.reason == GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS
    assert refusal.physical_ops == planned


def test_payload_mismatch_splits_without_cross_edge_fusion(factory):
    other = type(RANGE_DATA)("other_native_payload")
    specs = (
        _spec("left", RANGE_DATA, RANGE_DATA),
        _spec("right", other, other),
    )
    plan, _, _, _ = _linear_plan(2, specs=specs)

    result = FuseGPUOperators().apply(plan)

    stages = [
        operator
        for operator in result.dag.post_order_iter()
        if isinstance(operator, _FakeExecutable)
    ]
    assert len(stages) == 2
    assert [
        tuple(transform.kind for transform in stage.gpu_operator_spec.transforms)
        for stage in stages
    ] == [
        ("test_import", "left", "test_export"),
        ("test_import", "right", "test_export"),
    ]


def test_backend_refusal_leaves_complete_chain_and_records_cause():
    factory = _FakeFactory(reject=True)
    unregister_gpu_operator_factory(_BACKEND)
    register_gpu_operator_factory(_BACKEND, factory)
    try:
        plan, source, planned, _ = _linear_plan(3)
        original_op_map = plan.op_map.copy()

        result = FuseGPUOperators().apply(plan)

        assert result.dag is planned[-1]
        assert planned[0].input_dependencies == [source]
        assert result.op_map == original_op_map
        assert not factory.create_calls
        assert len(result.gpu_fusion_state.fusion_refusals) == 1
        refusal = result.gpu_fusion_state.fusion_refusals[0]
        assert refusal.reason == GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
        assert refusal.physical_ops == planned
    finally:
        unregister_gpu_operator_factory(_BACKEND, expected=factory)


def test_structural_lookalike_backend_result_is_refused_atomically():
    class LookalikeFactory(_FakeFactory):
        def create(self, compiled, input_op, context, creation_options=None):
            self.create_calls.append((compiled, input_op, creation_options))
            return _StructuralLookalike(compiled, input_op, context, creation_options)

        def attests(self, operator, compiled):
            raise AssertionError("structural lookalikes must be rejected first")

    factory = LookalikeFactory()
    unregister_gpu_operator_factory(_BACKEND)
    register_gpu_operator_factory(_BACKEND, factory)
    try:
        plan, source, planned, _ = _linear_plan(1)
        original_op_map = plan.op_map.copy()

        result = FuseGPUOperators().apply(plan)

        assert result.dag is planned[0]
        assert result.op_map == original_op_map
        assert source.output_dependencies == [planned[0]]
        assert len(factory.create_calls) == 1
        assert len(result.gpu_fusion_state.fusion_refusals) == 1
        refusal = result.gpu_fusion_state.fusion_refusals[0]
        assert refusal.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
        assert "nominal ExecutableGPUOperator marker" in refusal.detail
    finally:
        unregister_gpu_operator_factory(_BACKEND, expected=factory)


@pytest.mark.parametrize("error_type", (ValueError, RuntimeError))
def test_backend_programming_errors_propagate_without_recording_refusal(error_type):
    class BrokenFactory(_FakeFactory):
        def create(self, compiled, input_op, context, creation_options=None):
            del compiled, input_op, context, creation_options
            raise error_type("backend programming bug")

    factory = BrokenFactory()
    unregister_gpu_operator_factory(_BACKEND)
    register_gpu_operator_factory(_BACKEND, factory)
    try:
        plan, source, planned, _ = _linear_plan(1)
        original_op_map = plan.op_map.copy()

        with pytest.raises(error_type, match="backend programming bug"):
            FuseGPUOperators().apply(plan)

        assert plan.dag is planned[0]
        assert planned[0].input_dependencies == [source]
        assert plan.op_map == original_op_map
        assert plan.gpu_fusion_state.fusion_refusals == ()
    finally:
        unregister_gpu_operator_factory(_BACKEND, expected=factory)


def test_branch_stops_chain_discovery(factory):
    context = DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    source = InputDataBuffer(context, input_data=[])
    logical = tuple(_Logical(name) for name in ("shared", "left", "right"))
    shared = PlannedGPUOperator(_spec("shared"), source, context)
    left = PlannedGPUOperator(_spec("left"), shared, context)
    right = PlannedGPUOperator(_spec("right"), shared, context)
    for operator, logical_op in zip((shared, left, right), logical):
        operator.set_logical_operators(logical_op)
    root = PhysicalOperator("Branch", [left, right], context)
    plan = PhysicalPlan(
        root,
        {shared: logical[0], left: logical[1], right: logical[2]},
        context,
        gpu_fusion_state=GPUFusionPlanState(),
    )

    result = FuseGPUOperators().apply(plan)

    stages = {
        operator
        for operator in result.dag.post_order_iter()
        if isinstance(operator, _FakeExecutable)
    }
    assert len(stages) == 3
    assert all(
        len(
            [
                transform
                for transform in stage.gpu_operator_spec.transforms
                if transform.kind not in {"test_import", "test_export"}
            ]
        )
        == 1
        for stage in stages
    )


def test_multi_input_planned_node_is_not_materialized(factory):
    plan, source, planned, _ = _linear_plan(1)
    second_source = InputDataBuffer(plan.context, input_data=[])
    planned[0]._input_dependencies = [source, second_source]

    result = FuseGPUOperators().apply(plan)

    assert result.dag is planned[0]
    assert not factory.create_calls
    assert len(result.gpu_fusion_state.fusion_refusals) == 1
    assert (
        result.gpu_fusion_state.fusion_refusals[0].reason
        == GPUPlanningErrorReason.INCOMPLETE_FUSION
    )


def test_materializable_boundary_splits_chains_and_strict_cover_rejects(factory):
    specs = (
        _spec("produce_handle", RANGE_DATA, DEVICE_HANDLE),
        _spec("consume_handle", DEVICE_HANDLE, RANGE_DATA),
    )
    plan, _, planned, _ = _linear_plan(2, specs=specs)

    result = FuseGPUOperators().apply(plan)

    assert result.dag is planned[-1]
    assert not factory.create_calls
    assert len(result.gpu_fusion_state.fusion_refusals) == 2
    assert tuple(
        diagnostic.physical_ops
        for diagnostic in result.gpu_fusion_state.fusion_refusals
    ) == ((planned[0],), (planned[1],))
    with pytest.raises(GPUPlanningError, match="strict GPU fusion"):
        validate_gpu_execution_plan(result)


def test_gpu_rule_flag_off_is_a_noop(factory):
    plan, _, _, _ = _linear_plan(1)
    plan.context.enable_gpu_operator_fusion = False

    result = FuseGPUOperators().apply(plan)

    assert result is plan
    assert not factory.create_calls


def test_planned_gpu_operator_defends_against_execution(factory):
    _, _, planned, _ = _linear_plan(1)

    with pytest.raises(
        GPURequiredFusionError,
        match="reached execution before physical fusion",
    ):
        planned[0].start()


def test_gpu_physical_rule_order_is_explicit_and_unique():
    names = [rule.__name__ for rule in get_physical_ruleset()]
    assert names == [
        "InheritTargetMaxBlockSizeRule",
        "SetReadParallelismRule",
        "FuseGPUOperators",
        "FuseOperators",
        "ConfigureMapTaskMemoryUsingOutputSize",
    ]
    assert len(names) == len(set(names))


def _materialized_region(logical_op, input_op, context, factory, name):
    compiled = factory.prepare(_spec(f"core_{name}"))
    region = _FakeExecutable(
        compiled,
        input_op,
        context,
        GPUOperatorCreationOptions(name=name),
    )
    region.set_logical_operators(logical_op)
    return region, compiled


def test_ten_op_pipeline_with_only_three_gpu_nodes_is_rejected(factory):
    context = DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    source = InputDataBuffer(context, input_data=[])
    logical = []
    physical = []
    upstream = source
    for index in range(10):
        logical_op = _Logical(
            f"logical_{index}",
            () if not logical else (logical[-1],),
        )
        logical.append(logical_op)
        if 3 <= index < 6:
            operator = PlannedGPUOperator(
                _spec(f"gpu_{index}"),
                upstream,
                context,
                GPUOperatorCreationOptions(name=f"GPU{index}"),
            )
        else:
            operator = PhysicalOperator(f"Stock{index}", [upstream], context)
        operator.set_logical_operators(logical_op)
        physical.append(operator)
        upstream = operator
    source.set_logical_operators(logical[0])
    state = GPUFusionPlanState()
    state.initialize_logical_universe(logical[-1])
    state.register_infrastructure(source)
    for index in (*range(3), *range(6, 10)):
        state.register_stock_fallback(
            logical[index],
            GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
            f"logical_{index} used the stock data-plane implementation",
            (physical[index],),
        )
    op_map = {source: logical[0]}
    op_map.update(zip(physical, logical))
    plan = PhysicalPlan(upstream, op_map, context, gpu_fusion_state=state)

    optimized = FuseGPUOperators().apply(plan)

    stages = [
        operator
        for operator in optimized.dag.post_order_iter()
        if isinstance(operator, _FakeExecutable)
    ]
    assert len(stages) == 1
    assert stages[0]._logical_operators == list(logical[3:6])
    assert tuple(
        transform.kind for transform in stages[0].gpu_operator_spec.transforms[1:-1]
    ) == ("gpu_3", "gpu_4", "gpu_5")
    with pytest.raises(GPUPlanningError) as captured:
        validate_gpu_execution_plan(optimized)
    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "Stock" in str(captured.value)


@pytest.mark.parametrize("stock_position", ("before", "after"))
def test_stock_data_plane_before_or_after_complete_region_is_rejected(
    factory, stock_position
):
    context = DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    source = InputDataBuffer(context, input_data=[])
    first = _Logical("first")
    second = _Logical("second", (first,))
    if stock_position == "before":
        stock_logical, gpu_logical = first, second
        stock = PhysicalOperator("StockBefore", [source], context)
        region, compiled = _materialized_region(
            gpu_logical, stock, context, factory, "GPURegion"
        )
        root = region
    else:
        gpu_logical, stock_logical = first, second
        region, compiled = _materialized_region(
            gpu_logical, source, context, factory, "GPURegion"
        )
        stock = PhysicalOperator("StockAfter", [region], context)
        root = stock
    stock.set_logical_operators(stock_logical)
    source.set_logical_operators(first)
    state = GPUFusionPlanState()
    state.initialize_logical_universe(second)
    state.register_infrastructure(source)
    state.register_materialized_region(region, compiled, _BACKEND)
    state.register_stock_fallback(
        stock_logical,
        GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
        f"{stock.name} is stock data-plane work",
        (stock,),
    )
    plan = PhysicalPlan(
        root,
        {region: gpu_logical, stock: stock_logical},
        context,
        gpu_fusion_state=state,
    )

    with pytest.raises(GPUPlanningError) as captured:
        validate_gpu_execution_plan(plan)

    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert stock.name in str(captured.value)


def test_two_attested_gpu_regions_are_rejected_by_strict_cardinality(factory):
    context = DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    source = InputDataBuffer(context, input_data=[])
    first = _Logical("first")
    second = _Logical("second", (first,))
    first_region, first_spec = _materialized_region(
        first, source, context, factory, "GPURegion0"
    )
    second_region, second_spec = _materialized_region(
        second, first_region, context, factory, "GPURegion1"
    )
    state = GPUFusionPlanState()
    state.initialize_logical_universe(second)
    state.register_infrastructure(source)
    state.register_materialized_region(first_region, first_spec, _BACKEND)
    state.register_materialized_region(second_region, second_spec, _BACKEND)
    plan = PhysicalPlan(
        second_region,
        {first_region: first, second_region: second},
        context,
        gpu_fusion_state=state,
    )

    with pytest.raises(GPUPlanningError) as captured:
        validate_gpu_execution_plan(plan)

    assert captured.value.reason is GPUPlanningErrorReason.INCOMPLETE_FUSION
    assert "data_region" in str(captured.value)
    assert "found 2" in str(captured.value)


def test_explain_renders_fresh_source_only_strict_plan_before_execution_rejects(
    monkeypatch, capsys
):
    from ray.data._internal.logical.rules import set_read_parallelism

    original = DataContext.get_current()
    context = original.copy()
    context.enable_gpu_operator_fusion = True
    DataContext._set_current(context)
    monkeypatch.setattr(ray, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": 2})
    monkeypatch.setattr(ray, "available_resources", lambda: {"CPU": 2})
    monkeypatch.setattr(
        set_read_parallelism, "ray_available_resources", lambda: {"CPU": 2}
    )
    monkeypatch.setattr(ray.util, "get_current_placement_group", lambda: None)

    class NeverExecutor:
        def __init__(self):
            self._options = SimpleNamespace(preserve_order=False)
            self.called = False

        def execute(self, *args, **kwargs):
            self.called = True
            raise AssertionError("executor must not start for a rejected strict plan")

    try:
        dataset = ray.data.range(4, override_num_blocks=1)
        dataset.explain()
        explained = capsys.readouterr().out
        assert "Physical Plan (Optimized)" in explained
        assert "TaskPoolMapOperator" in explained

        executor = NeverExecutor()
        with pytest.raises(GPUPlanningError) as captured:
            dataset._execute_dag(executor)

        assert not executor.called
        assert captured.value.reason in {
            GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
            GPUPlanningErrorReason.INCOMPLETE_FUSION,
        }
        assert "data_region" in str(captured.value) or "ReadRange" in str(
            captured.value
        )
    finally:
        DataContext._set_current(original)
