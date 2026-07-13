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
    GPUStageRetryPolicy,
    GPUTransformSpec,
    PayloadKind,
    RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    RANGE_DATA_TERMINAL_ADAPTER_KEY,
    RANGE_WORK_ADAPTER_KEY,
    compile_gpu_region,
    compose_gpu_operator_specs,
    create_gpu_operator,
    register_gpu_operator_factory,
    register_gpu_boundary_adapter,
    unregister_gpu_boundary_adapter,
)
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import PhysicalPlan
from ray.data._internal.logical.optimizers import get_physical_ruleset
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
        return "fake_session"

    @property
    def gpu_creation_options(self):
        return self._options


class _FakeSessionFactory:
    def supports(self, spec):
        return spec.execution_requirements.backend_name == "fake_session"

    def create(self, spec, input_op, data_context, creation_options=None):
        return _FakeSessionGPUOperator(spec, input_op, data_context, creation_options)


class _SingleSpecFactory(_FakeSessionFactory):
    def supports(self, spec):
        return (
            spec.execution_requirements.backend_name == "single_spec"
            and len(spec.transforms) == 1
        )


def _register_fake_factory():
    register_gpu_operator_factory(
        "fake_session", _FakeSessionFactory(), replace_existing=True
    )


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
):
    if key is None:
        key = GPUExecutionCompatibilityKey(
            worker_lifecycle=("session" if backend == "fake_session" else "actor_pool"),
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
    )


def _three_node_plan():
    _register_fake_factory()
    context = DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    source = InputDataBuffer(context, input_data=[])
    read = create_gpu_operator(
        _spec(
            "read/0/read",
            PayloadKind.RANGE_WORK,
            PayloadKind.CUDF_BATCH_STREAM,
            input_adapter=RANGE_WORK_ADAPTER_KEY,
        ),
        source,
        context,
        GPUOperatorCreationOptions(min_rows_per_bundle=1),
    )
    mapped = create_gpu_operator(
        _spec(
            "map/0/map",
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.CUDF_BATCH_STREAM,
        ),
        read,
        context,
    )
    grouped = create_gpu_operator(
        _spec(
            "group/0/group",
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.CUDF_BATCH_STREAM,
        ),
        mapped,
        context,
    )
    logical = [MagicMock(name=name) for name in ("read", "map", "group")]
    read.set_logical_operators(logical[0])
    mapped.set_logical_operators(logical[1])
    grouped.set_logical_operators(logical[2])
    return PhysicalPlan(
        grouped,
        {source: logical[0], read: logical[0], mapped: logical[1], grouped: logical[2]},
        context,
    )


def test_gpu_rule_fuses_optional_chain_and_rewrites_op_map_idempotently():
    plan = _three_node_plan()
    result = FuseGPUOperators().apply(plan)

    assert isinstance(result.dag, GPUOperator)
    assert result.dag.gpu_operator_spec.transform_ids == (
        "read/0/read",
        "map/0/map",
        "group/0/group",
    )
    assert result.dag.gpu_creation_options.name == "GPU[read,map,group]"
    assert len(result.dag.input_dependencies) == 1
    assert isinstance(result.dag.input_dependencies[0], InputDataBuffer)
    assert set(result.op_map) == {result.dag.input_dependencies[0], result.dag}
    assert result.dag.input_dependencies[0].output_dependencies == [result.dag]

    again = FuseGPUOperators().apply(result)
    assert again.dag is result.dag
    assert again.dag.gpu_operator_spec.canonical_contract() == (
        result.dag.gpu_operator_spec.canonical_contract()
    )


def test_gate2_composed_stage_is_atomic_on_both_edges():
    _register_fake_factory()
    context = DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    source = InputDataBuffer(context, input_data=[])
    read = _spec(
        "read/0/read",
        PayloadKind.RANGE_WORK,
        PayloadKind.CUDF_BATCH_STREAM,
        input_adapter=RANGE_WORK_ADAPTER_KEY,
    )
    group = _spec(
        "group/0/group",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
    )
    sealed_spec = compile_gpu_region((read, group)).for_gate2_stage(
        specialized_read_parallelism_resolved=True,
        uses_terminal_output_adapter=True,
    )
    sealed = create_gpu_operator(sealed_spec, source, context)
    adjacent = create_gpu_operator(
        _spec(
            "adjacent/0/map",
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.CUDF_BATCH_STREAM,
        ),
        sealed,
        context,
    )

    downstream_result = FuseGPUOperators().apply(PhysicalPlan(adjacent, {}, context))
    assert downstream_result.dag is adjacent
    assert downstream_result.dag.input_dependencies[0] is sealed

    upstream = create_gpu_operator(read, source, context)
    sealed_downstream = create_gpu_operator(
        _spec(
            "sealed-downstream",
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.CUDF_BATCH_STREAM,
        ).for_gate2_stage(
            specialized_read_parallelism_resolved=False,
            uses_terminal_output_adapter=True,
        ),
        upstream,
        context,
    )
    upstream_result = FuseGPUOperators().apply(
        PhysicalPlan(sealed_downstream, {}, context)
    )
    assert upstream_result.dag is sealed_downstream
    assert upstream_result.dag.input_dependencies[0] is upstream


def test_fusion_refusal_is_a_silent_noop():
    _register_fake_factory()
    context = DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    source = InputDataBuffer(context, input_data=[])
    upstream = create_gpu_operator(
        _spec(
            "upstream",
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.CUDF_BATCH_STREAM,
        ),
        source,
        context,
    )
    incompatible = create_gpu_operator(
        _spec(
            "incompatible",
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.CUDF_BATCH_STREAM,
            key=GPUExecutionCompatibilityKey(
                runtime_id="different",
                worker_lifecycle="session",
                num_gpus=0,
            ),
        ),
        upstream,
        context,
    )

    result = FuseGPUOperators().apply(PhysicalPlan(incompatible, {}, context))
    assert result.dag is incompatible
    assert result.dag.input_dependencies[0] is upstream


def test_backend_refusal_of_combined_spec_is_a_silent_noop():
    register_gpu_operator_factory(
        "single_spec", _SingleSpecFactory(), replace_existing=True
    )
    context = DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    source = InputDataBuffer(context, input_data=[])
    upstream = create_gpu_operator(
        _spec(
            "upstream",
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.CUDF_BATCH_STREAM,
            backend="single_spec",
        ),
        source,
        context,
    )
    downstream = create_gpu_operator(
        _spec(
            "downstream",
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.CUDF_BATCH_STREAM,
            backend="single_spec",
        ),
        upstream,
        context,
    )

    result = FuseGPUOperators().apply(PhysicalPlan(downstream, {}, context))
    assert result.dag is downstream
    assert result.dag.input_dependencies[0] is upstream


def test_multiple_gpu_regions_are_allowed_and_optimized_independently():
    _register_fake_factory()
    context = DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    left_source = InputDataBuffer(context, input_data=[])
    right_source = InputDataBuffer(context, input_data=[])
    left = create_gpu_operator(
        _spec(
            "left",
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.CUDF_BATCH_STREAM,
        ),
        left_source,
        context,
    )
    right = create_gpu_operator(
        _spec(
            "right",
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.CUDF_BATCH_STREAM,
        ),
        right_source,
        context,
    )
    root = PhysicalOperator("TwoGPURegions", [left, right], context)

    result = FuseGPUOperators().apply(PhysicalPlan(root, {}, context))
    assert result.dag is root
    assert result.dag.input_dependencies == [left, right]


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
            replace(downstream, target_max_block_size_override=None), fused_spec
        ).target_max_block_size_override
        == 128
    )


def test_gpu_rule_flag_off_is_exact_noop():
    plan = _three_node_plan()
    plan.context.enable_gpu_operator_fusion = False
    assert FuseGPUOperators().apply(plan) is plan


def _retry_policy_for_context(context):
    return GPUStageRetryPolicy(
        actor_task_retry_on_errors=context.actor_task_retry_on_errors,
        actor_init_retry_on_errors=context.actor_init_retry_on_errors,
        actor_init_max_retries=context.actor_init_max_retries,
        retried_map_errors=context.retried_map_errors,
        max_map_retries=context.max_map_retries,
        retried_io_errors=context.retried_io_errors,
    )


def _sealed_actor_spec(context, *, retry_policy=None):
    key = GPUExecutionCompatibilityKey(
        num_gpus=0,
        retry_policy=retry_policy or _retry_policy_for_context(context),
    )
    spec = _spec(
        "actor-stage",
        PayloadKind.RANGE_WORK,
        PayloadKind.RANGE_DATA,
        backend="actor_pool",
        key=key,
        input_adapter=RANGE_WORK_ADAPTER_KEY,
        output_adapter=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    )
    return spec.for_gate2_stage(
        specialized_read_parallelism_resolved=True,
        uses_terminal_output_adapter=True,
    )


def test_actor_pool_gpu_operator_uses_effective_stock_retry_defaults():
    context = DataContext.get_current().copy()
    source = InputDataBuffer(context, input_data=[])
    operator = create_gpu_operator(_sealed_actor_spec(context), source, context)

    assert operator._ray_remote_args["max_restarts"] == -1
    assert operator._ray_remote_args["max_task_retries"] == -1
    assert RaySystemError in operator._ray_actor_task_remote_args["retry_exceptions"]


def test_actor_pool_gpu_operator_honors_actor_task_retry_context():
    class RetryableUserError(RuntimeError):
        pass

    context = DataContext.get_current().copy()
    context.actor_task_retry_on_errors = [RetryableUserError]
    source = InputDataBuffer(context, input_data=[])
    operator = create_gpu_operator(_sealed_actor_spec(context), source, context)

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
        _sealed_actor_spec(context, retry_policy=policy), source, context
    )

    assert operator._ray_remote_args["max_restarts"] == 2
    assert operator._ray_remote_args["max_task_retries"] == 4


def test_actor_pool_factory_rejects_context_retry_policy_mismatch():
    context = DataContext.get_current().copy()
    spec = _sealed_actor_spec(context)
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
    ).for_gate2_stage(
        specialized_read_parallelism_resolved=False,
        uses_terminal_output_adapter=False,
    )

    operator = create_gpu_operator(spec, source, context)
    assert operator.gpu_operator_spec is spec


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
    _register_fake_factory()
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
