from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from ray.data.context import DataContext
from ray.data._internal.execution.gpu import (
    GPUBoundaryAdapter,
    GPUBoundaryDirection,
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUOperator,
    GPUOperatorCreationOptions,
    GPUOperatorSpec,
    GPURequiredFusionError,
    GPUStageRetryPolicy,
    GPUTransformSpec,
    PayloadKind,
    PlannedGPUOperator,
    RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    RANGE_DATA_TERMINAL_ADAPTER_KEY,
    RANGE_WORK_ADAPTER_KEY,
    compose_gpu_operator_specs,
    create_gpu_operator,
    register_gpu_boundary_adapter,
    register_gpu_operator_factory,
    unregister_gpu_boundary_adapter,
)
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import PhysicalPlan
from ray.data._internal.logical.optimizers import (
    PhysicalOptimizer,
    get_physical_ruleset,
)
from ray.data._internal.logical.rules import FuseGPUOperators
from ray.exceptions import RaySystemError


def _identity(value, _context):
    return value


class _FakeSessionGPUOperator(PhysicalOperator, GPUOperator):
    def __init__(self, spec, input_op, context, options):
        self._spec = spec
        self._options = options or GPUOperatorCreationOptions()
        super().__init__("FakeSessionGPU", [input_op], context)

    @property
    def gpu_operator_spec(self):
        return self._spec

    @property
    def execution_backend_name(self):
        return self._spec.execution_requirements.backend_name

    @property
    def gpu_creation_options(self):
        return self._options


class _FakeSessionFactory:
    def __init__(self, backend_name="fake_session"):
        self.backend_name = backend_name
        self.create_calls = []

    def supports(self, spec):
        return spec.execution_requirements.backend_name == self.backend_name

    def create(self, spec, input_op, data_context, creation_options=None):
        self.create_calls.append((spec, input_op, creation_options))
        return _FakeSessionGPUOperator(
            spec, input_op, data_context, creation_options
        )


class _SingleSpecFactory(_FakeSessionFactory):
    def __init__(self):
        super().__init__("single_spec")

    def supports(self, spec):
        return super().supports(spec) and len(spec.transforms) == 1


def _register_fake_factory(factory=None):
    factory = factory or _FakeSessionFactory()
    register_gpu_operator_factory(
        factory.backend_name,
        factory,
        replace_existing=True,
    )
    return factory


def _spec(
    transform_id,
    native_input,
    native_output,
    *,
    backend="fake_session",
    key=None,
    input_adapter=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    output_adapter=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    terminal_output_adapter=RANGE_DATA_TERMINAL_ADAPTER_KEY,
    fusion_group_id=None,
    required_provenance=(),
    requires_complete_fusion=False,
    owns_input_parallelism=False,
    uses_terminal_output_adapter=False,
):
    if key is None:
        key = GPUExecutionCompatibilityKey(
            worker_lifecycle=(
                "session" if backend == "fake_session" else "actor_pool"
            ),
            num_gpus=0,
        )
    return GPUOperatorSpec(
        spec_instance_id=transform_id,
        transforms=(
            GPUTransformSpec(
                transform_id,
                transform_id.rsplit("/", 1)[-1],
                native_input,
                native_output,
                config=_identity,
            ),
        ),
        external_input_payload=PayloadKind.RAY_BLOCK_STREAM,
        external_output_payload=PayloadKind.RAY_BLOCK_STREAM,
        native_input_payload=native_input,
        native_output_payload=native_output,
        input_adapter_key=input_adapter,
        output_adapter_key=output_adapter,
        terminal_output_adapter_key=terminal_output_adapter,
        execution_requirements=GPUExecutionRequirements(
            backend_name=backend,
            compatibility_key=key,
        ),
        logical_provenance=(transform_id,),
        fusion_group_id=fusion_group_id,
        requires_complete_fusion=requires_complete_fusion,
        required_logical_provenance=required_provenance,
        owns_input_parallelism=owns_input_parallelism,
        uses_terminal_output_adapter=uses_terminal_output_adapter,
    )


def _three_node_plan(
    *,
    backend="fake_session",
    factory=None,
    map_key=None,
    required_provenance=None,
):
    factory = _register_fake_factory(
        factory or _FakeSessionFactory(backend)
    )
    context = DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    source = InputDataBuffer(context, input_data=[])
    transform_ids = (
        "read/0/read",
        "map/0/map",
        "group/0/group",
    )
    required_provenance = required_provenance or transform_ids
    common = dict(
        backend=backend,
        fusion_group_id="required-region",
        required_provenance=required_provenance,
        requires_complete_fusion=True,
    )
    read = PlannedGPUOperator(
        _spec(
            transform_ids[0],
            PayloadKind.RANGE_WORK,
            PayloadKind.CUDF_BATCH_STREAM,
            input_adapter=RANGE_WORK_ADAPTER_KEY,
            owns_input_parallelism=True,
            **common,
        ),
        source,
        context,
        GPUOperatorCreationOptions(
            name="GPURead",
            min_rows_per_bundle=1,
            target_max_block_size_override=128,
        ),
    )
    mapped = PlannedGPUOperator(
        _spec(
            transform_ids[1],
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.CUDF_BATCH_STREAM,
            key=map_key,
            **common,
        ),
        read,
        context,
        GPUOperatorCreationOptions(name="GPUMap"),
    )
    grouped = PlannedGPUOperator(
        _spec(
            transform_ids[2],
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.CUDF_BATCH_STREAM,
            uses_terminal_output_adapter=True,
            **common,
        ),
        mapped,
        context,
        GPUOperatorCreationOptions(
            name="GPUGroup",
            target_max_block_size_override=256,
        ),
    )
    logical = [MagicMock(name=name) for name in ("read", "map", "group")]
    source.set_logical_operators(logical[0])
    for physical, logical_op in zip((read, mapped, grouped), logical):
        physical.set_logical_operators(logical_op)
    plan = PhysicalPlan(
        grouped,
        {
            source: logical[0],
            read: logical[0],
            mapped: logical[1],
            grouped: logical[2],
        },
        context,
    )
    return plan, factory, (source, read, mapped, grouped), logical


def test_required_chain_materializes_once_and_rewrites_op_map_idempotently():
    plan, factory, (source, read, mapped, grouped), logical = _three_node_plan()
    assert len(factory.create_calls) == 0

    result = FuseGPUOperators().apply(plan)

    assert isinstance(result.dag, _FakeSessionGPUOperator)
    assert not isinstance(result.dag, PlannedGPUOperator)
    assert result.dag.gpu_operator_spec.transform_ids == (
        "read/0/read",
        "map/0/map",
        "group/0/group",
    )
    assert (
        result.dag.gpu_operator_spec.logical_provenance
        == result.dag.gpu_operator_spec.required_logical_provenance
    )
    assert result.dag.gpu_operator_spec.owns_input_parallelism
    assert result.dag.gpu_operator_spec.uses_terminal_output_adapter
    assert result.dag.gpu_creation_options.name == "GPU[read,map,group]"
    assert result.dag.gpu_creation_options.min_rows_per_bundle == 1
    assert result.dag.gpu_creation_options.target_max_block_size_override == 256
    assert result.dag.input_dependencies == [source]
    assert result.dag._logical_operators == logical
    assert set(result.op_map) == {source, result.dag}
    assert result.op_map[result.dag] is logical[-1]
    assert source.output_dependencies == [result.dag]
    assert len(factory.create_calls) == 1
    assert not any(
        isinstance(op, PlannedGPUOperator)
        for op in result.dag.post_order_iter()
    )

    again = FuseGPUOperators().apply(result)
    assert again is result
    assert len(factory.create_calls) == 1


def test_incompatible_required_chain_is_fatal_without_partial_creation():
    map_key = GPUExecutionCompatibilityKey(
        runtime_id="different",
        worker_lifecycle="session",
        num_gpus=0,
    )
    plan, factory, _, _ = _three_node_plan(map_key=map_key)

    with pytest.raises(
        GPURequiredFusionError,
        match="could not be composed.*EXECUTION_INCOMPATIBLE",
    ):
        FuseGPUOperators().apply(plan)

    assert len(factory.create_calls) == 0
    assert sum(
        isinstance(op, PlannedGPUOperator)
        for op in plan.dag.post_order_iter()
    ) == 3


def test_backend_refusal_is_fatal_without_partial_creation():
    factory = _SingleSpecFactory()
    plan, factory, _, _ = _three_node_plan(
        backend="single_spec",
        factory=factory,
    )

    with pytest.raises(
        GPURequiredFusionError,
        match="backend materialization failed.*does not support",
    ):
        FuseGPUOperators().apply(plan)

    assert len(factory.create_calls) == 0


def test_branched_required_chain_is_fatal_without_creation():
    plan, factory, (_, _, mapped, grouped), _ = _three_node_plan()
    root = PhysicalOperator("Branched", [grouped, mapped], plan.context)

    with pytest.raises(GPURequiredFusionError, match="is branched"):
        FuseGPUOperators().apply(
            PhysicalPlan(root, {**plan.op_map, root: MagicMock()}, plan.context)
        )

    assert len(factory.create_calls) == 0


def test_incomplete_required_provenance_is_fatal_without_creation():
    plan, factory, _, _ = _three_node_plan(
        required_provenance=(
            "read/0/read",
            "map/0/map",
            "missing/0/transform",
            "group/0/group",
        )
    )

    with pytest.raises(
        GPURequiredFusionError,
        match="incomplete or reordered logical provenance",
    ):
        FuseGPUOperators().apply(plan)

    assert len(factory.create_calls) == 0


def _single_node_region(context, factory, name):
    source = InputDataBuffer(context, input_data=[])
    spec = _spec(
        name,
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
        fusion_group_id=f"region-{name}",
        required_provenance=(name,),
        requires_complete_fusion=True,
        uses_terminal_output_adapter=True,
    )
    node = PlannedGPUOperator(spec, source, context)
    logical = MagicMock(name=name)
    node.set_logical_operators(logical)
    return source, node, logical


def test_multiple_independent_required_regions_materialize_once_each():
    factory = _register_fake_factory()
    context = DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    left_source, left, left_logical = _single_node_region(
        context, factory, "left"
    )
    right_source, right, right_logical = _single_node_region(
        context, factory, "right"
    )
    root = PhysicalOperator("TwoGPURegions", [left, right], context)
    plan = PhysicalPlan(
        root,
        {
            left_source: left_logical,
            left: left_logical,
            right_source: right_logical,
            right: right_logical,
            root: MagicMock(),
        },
        context,
    )

    result = FuseGPUOperators().apply(plan)

    assert result.dag is root
    assert all(
        isinstance(op, _FakeSessionGPUOperator)
        for op in result.dag.input_dependencies
    )
    assert len(factory.create_calls) == 2
    assert not any(
        isinstance(op, PlannedGPUOperator)
        for op in result.dag.post_order_iter()
    )


def test_fused_creation_options_follow_external_boundaries():
    read_spec = _spec(
        "read",
        PayloadKind.RANGE_WORK,
        PayloadKind.CUDF_BATCH_STREAM,
        input_adapter=RANGE_WORK_ADAPTER_KEY,
    )
    group_spec = _spec(
        "group",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
    )
    fused_spec = compose_gpu_operator_specs(read_spec, group_spec)
    upstream = GPUOperatorCreationOptions(
        name="GPURead",
        min_rows_per_bundle=1,
        target_max_block_size_override=128,
    )
    downstream = GPUOperatorCreationOptions(
        name="GPUGroup",
        min_rows_per_bundle=999,
        target_max_block_size_override=256,
    )

    fused = upstream.for_fused_spec(downstream, fused_spec)

    assert fused.name == "GPU[read,group]"
    assert fused.min_rows_per_bundle == 1
    assert fused.target_max_block_size_override == 256
    assert (
        upstream.for_fused_spec(
            replace(downstream, target_max_block_size_override=None),
            fused_spec,
        ).target_max_block_size_override
        == 128
    )


def test_gpu_rule_flag_off_rejects_deferred_region():
    plan, factory, _, _ = _three_node_plan()
    plan.context.enable_gpu_operator_fusion = False

    with pytest.raises(GPURequiredFusionError, match="require.*enabled"):
        FuseGPUOperators().apply(plan)

    assert len(factory.create_calls) == 0


def test_planned_gpu_operator_defends_against_execution():
    plan, factory, (_, read, _, _), _ = _three_node_plan()

    with pytest.raises(
        GPURequiredFusionError,
        match="reached execution before required physical fusion",
    ):
        read.start()

    assert len(factory.create_calls) == 0
    assert plan.dag is not None


def test_physical_optimizer_postcheck_rejects_omitted_fusion_rule():
    class _NoRulesPhysicalOptimizer(PhysicalOptimizer):
        @property
        def rules(self):
            return []

    plan, factory, _, _ = _three_node_plan()

    with pytest.raises(
        GPURequiredFusionError,
        match="physical optimization left non-executable planned GPU operators",
    ):
        _NoRulesPhysicalOptimizer().optimize(plan)

    assert len(factory.create_calls) == 0


def _retry_policy_for_context(context):
    return GPUStageRetryPolicy(
        actor_task_retry_on_errors=context.actor_task_retry_on_errors,
        actor_init_retry_on_errors=context.actor_init_retry_on_errors,
        actor_init_max_retries=context.actor_init_max_retries,
        retried_map_errors=context.retried_map_errors,
        max_map_retries=context.max_map_retries,
        retried_io_errors=context.retried_io_errors,
    )


def _actor_spec(context, *, retry_policy=None):
    key = GPUExecutionCompatibilityKey(
        num_gpus=0,
        retry_policy=retry_policy or _retry_policy_for_context(context),
    )
    return replace(
        _spec(
            "actor-stage",
            PayloadKind.RANGE_WORK,
            PayloadKind.RANGE_DATA,
            backend="actor_pool",
            key=key,
            input_adapter=RANGE_WORK_ADAPTER_KEY,
            output_adapter=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
        ),
        uses_terminal_output_adapter=True,
    )


def test_actor_pool_gpu_operator_uses_effective_stock_retry_defaults():
    context = DataContext.get_current().copy()
    source = InputDataBuffer(context, input_data=[])
    operator = create_gpu_operator(_actor_spec(context), source, context)

    assert operator._ray_remote_args["max_restarts"] == -1
    assert operator._ray_remote_args["max_task_retries"] == -1
    assert RaySystemError in operator._ray_actor_task_remote_args["retry_exceptions"]


def test_actor_pool_gpu_operator_honors_actor_task_retry_context():
    class RetryableUserError(RuntimeError):
        pass

    context = DataContext.get_current().copy()
    context.actor_task_retry_on_errors = [RetryableUserError]
    source = InputDataBuffer(context, input_data=[])
    operator = create_gpu_operator(_actor_spec(context), source, context)

    retry_exceptions = operator._ray_actor_task_remote_args["retry_exceptions"]
    assert RetryableUserError in retry_exceptions
    assert RaySystemError in retry_exceptions


def test_actor_pool_gpu_operator_honors_actor_retry_limits_from_key():
    context = DataContext.get_current().copy()
    source = InputDataBuffer(context, input_data=[])
    policy = replace(
        _retry_policy_for_context(context),
        actor_max_restarts=2,
        actor_max_task_retries=4,
    )
    operator = create_gpu_operator(
        _actor_spec(context, retry_policy=policy), source, context
    )

    assert operator._ray_remote_args["max_restarts"] == 2
    assert operator._ray_remote_args["max_task_retries"] == 4


def test_actor_pool_factory_rejects_context_retry_policy_mismatch():
    context = DataContext.get_current().copy()
    spec = _actor_spec(context)
    context.retried_map_errors = True
    source = InputDataBuffer(context, input_data=[])

    with pytest.raises(ValueError, match="does not match the execution DataContext"):
        create_gpu_operator(spec, source, context)


def test_map_spec_becomes_executable_by_registering_its_missing_import():
    def passthrough(value, _context):
        return value

    context = DataContext.get_current().copy()
    source = InputDataBuffer(context, input_data=[])
    map_spec = _spec(
        "standalone-map",
        PayloadKind.RANGE_DATA,
        PayloadKind.RANGE_DATA,
        backend="actor_pool",
        key=GPUExecutionCompatibilityKey(
            num_gpus=0,
            retry_policy=_retry_policy_for_context(context),
        ),
    )

    with pytest.raises(ValueError, match="does not support"):
        create_gpu_operator(map_spec, source, context)

    boundary_adapters = tuple(
        GPUBoundaryAdapter(
            key=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
            direction=direction,
            external_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_payload=PayloadKind.RANGE_DATA,
            adapt=passthrough,
        )
        for direction in (
            GPUBoundaryDirection.IMPORT,
            GPUBoundaryDirection.EXPORT,
        )
    )
    for adapter in boundary_adapters:
        register_gpu_boundary_adapter(adapter)
    try:
        operator = create_gpu_operator(map_spec, source, context)
        assert operator.gpu_operator_spec is map_spec
    finally:
        for adapter in boundary_adapters:
            unregister_gpu_boundary_adapter(
                key=adapter.key,
                direction=adapter.direction,
                external_payload=adapter.external_payload,
                native_payload=adapter.native_payload,
            )


def test_actor_pool_factory_accepts_new_registered_ports_without_edits():
    def passthrough(value, _context):
        return value

    adapter_key = "test-future-stage-adapter-v1"
    for direction in (
        GPUBoundaryDirection.IMPORT,
        GPUBoundaryDirection.EXPORT,
    ):
        register_gpu_boundary_adapter(
            GPUBoundaryAdapter(
                key=adapter_key,
                direction=direction,
                external_payload=PayloadKind.RAY_BLOCK_STREAM,
                native_payload=PayloadKind.CUDF_PARTITION,
                adapt=passthrough,
            ),
            replace_existing=True,
        )
    context = DataContext.get_current().copy()
    source = InputDataBuffer(context, input_data=[])
    spec = _spec(
        "future-stage",
        PayloadKind.CUDF_PARTITION,
        PayloadKind.CUDF_PARTITION,
        backend="actor_pool",
        key=GPUExecutionCompatibilityKey(
            num_gpus=0,
            retry_policy=_retry_policy_for_context(context),
        ),
        input_adapter=adapter_key,
        output_adapter=adapter_key,
        terminal_output_adapter=adapter_key,
    )

    try:
        operator = create_gpu_operator(spec, source, context)
        assert operator.gpu_operator_spec is spec
    finally:
        for direction in (
            GPUBoundaryDirection.IMPORT,
            GPUBoundaryDirection.EXPORT,
        ):
            unregister_gpu_boundary_adapter(
                key=adapter_key,
                direction=direction,
                external_payload=PayloadKind.RAY_BLOCK_STREAM,
                native_payload=PayloadKind.CUDF_PARTITION,
            )


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


def test_backend_factory_can_be_replaced_without_actor_pool_dependencies():
    factory = _register_fake_factory()
    context = DataContext.get_current().copy()
    source = InputDataBuffer(context, input_data=[])
    spec = _spec(
        "session-transform",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
    )

    operator = create_gpu_operator(spec, source, context)

    assert isinstance(operator, _FakeSessionGPUOperator)
    assert operator.gpu_operator_spec is spec
    assert len(factory.create_calls) == 1
