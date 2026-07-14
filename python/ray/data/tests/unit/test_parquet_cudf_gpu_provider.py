import copy
from dataclasses import replace
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray.data
import ray.data._internal.execution.gpu as gpu_package
from ray.data import ActorPoolStrategy, UDFProperties
from ray.data._internal.datasource import parquet_datasource
from ray.data._internal.execution.gpu import (
    GPUCompositionError,
    GPUCompositionErrorReason,
    GPUOperator,
    GPURequiredFusionError,
    GPUBoundaryDirection,
    PayloadKind,
    PlannedGPUOperator,
    RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    RANGE_DATA_TERMINAL_ADAPTER_KEY,
    RANGE_WORK_ADAPTER_KEY,
    RowsOwnedByRange,
    get_gpu_boundary_adapter,
    register_gpu_boundary_adapter,
    unregister_gpu_boundary_adapter,
)
from ray.data._internal.execution.gpu import operator as gpu_operator_module
from ray.data._internal.execution.gpu import region_compiler
from ray.data._internal.execution.operators import hash_shuffle
from ray.data._internal.execution.operators.actor_pool_gpu_operator import (
    ActorPoolGPUOperator,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.operators import MapBatches, Read
from ray.data._internal.logical.optimizers import PhysicalOptimizer
from ray.data._internal.logical.rules import FuseGPUOperators
from ray.data._internal.logical.rules import gpu_operator_fusion
from ray.data._internal.logical.rules.set_read_parallelism import (
    SetReadParallelismRule,
)
from ray.data._internal.planner.gpu_lowering.map_group_partitions_lowerer import (
    GPUMapGroupPartitionsLowerer,
)
from ray.data._internal.planner.gpu_plan_provider import (
    create_gpu_required_fusion_planner,
)
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)
from ray.data._internal.planner.parquet_cudf_range_planner import (
    ParquetCudfPlanningError,
)
from ray.data._internal.planner.planner import Planner
from ray.data.context import DataContext


class _Tokenizer:
    def __call__(self, batch):
        return batch


def _partition_fn(frame, context):
    del context
    return frame


@pytest.fixture
def fusion_context(monkeypatch):
    context = DataContext.get_current()
    original = copy.deepcopy(context)
    monkeypatch.setattr(ray, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(ray, "put", lambda value: value)
    monkeypatch.setattr(ray, "get", lambda value: value)
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": 8, "GPU": 2})
    monkeypatch.setattr(ray, "available_resources", lambda: {"CPU": 8, "GPU": 2})
    monkeypatch.setattr(ray.util, "get_current_placement_group", lambda: None)
    monkeypatch.setattr(
        hash_shuffle,
        "_get_total_cluster_resources",
        lambda: SimpleNamespace(gpu=2),
    )
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "local-test-node"),
    )
    monkeypatch.setattr(
        parquet_datasource,
        "_fetch_file_infos",
        lambda sampled_fragments, *, columns, schema, local_scheduling: [
            parquet_datasource._fetch_parquet_file_info(
                fragment, columns=columns, schema=schema
            )
            for fragment in sampled_fragments
        ],
    )
    context.use_datasource_v2 = False
    context.enable_gpu_operator_fusion = True
    context.checkpoint_config = None
    context.actor_task_retry_on_errors = False
    context.actor_init_retry_on_errors = False
    context.retried_map_errors = False
    yield context
    DataContext._set_current(original)


def _write_parquet(tmp_path):
    path = tmp_path / "input.parquet"
    table = pa.table(
        {
            "User": pa.array([1, 1, 2, 3], type=pa.int64()),
            "Card": pa.array([10, 10, 20, 30], type=pa.int64()),
            "Merchant": pa.array([7, 8, 7, 9], type=pa.int64()),
            "Value": pa.array([1, 2, 3, 4], type=pa.int64()),
        }
    )
    pq.write_table(table, path, row_group_size=2, write_statistics=True)
    return str(path)


def _read(path, *, ray_remote_args=None):
    return ray.data.read_parquet(
        path,
        columns=["User", "Card", "Merchant", "Value"],
        ray_remote_args=ray_remote_args,
    )


def _map(dataset, *, preserves=True, scheduling_strategy=None):
    remote_args = {"num_cpus": 1, "num_gpus": 1}
    if scheduling_strategy is not None:
        remote_args["scheduling_strategy"] = scheduling_strategy
    return dataset.map_batches(
        _Tokenizer,
        batch_size=2,
        batch_format="cudf",
        compute=ActorPoolStrategy(size=1),
        udf_properties=(
            UDFProperties(preserves_partitioning=True) if preserves else None
        ),
        **remote_args,
    )


def _mgp(dataset, *, keys=("User", "Card"), count=2):
    return dataset.groupby(list(keys), num_partitions=count).map_group_partitions(
        _partition_fn
    )


def _pipeline(path):
    return _mgp(_map(_read(path)))


def _gpu_ops(plan):
    return [op for op in plan.dag.post_order_iter() if isinstance(op, GPUOperator)]


def _planned_ops(plan):
    return [
        op for op in plan.dag.post_order_iter() if isinstance(op, PlannedGPUOperator)
    ]


def _actor_ops(plan):
    return [
        op
        for op in plan.dag.post_order_iter()
        if isinstance(op, ActorPoolGPUOperator)
    ]


def _kinds(gpu_op):
    return tuple(transform.kind for transform in gpu_op.gpu_operator_spec.transforms)


def test_planner_emits_three_deferred_specs_then_physical_optimizer_fuses(
    tmp_path, fusion_context
):
    dataset = _pipeline(_write_parquet(tmp_path))
    grouped = dataset._logical_plan.dag
    mapped = grouped.input_dependencies[0]
    read = mapped.input_dependencies[0]

    plan, _ = Planner().plan(dataset._logical_plan)

    operators = list(plan.dag.post_order_iter())
    planned = _planned_ops(plan)
    assert len(planned) == 3
    assert _actor_ops(plan) == []
    assert isinstance(operators[0], InputDataBuffer)
    assert [_kinds(op) for op in planned] == [
        ("read_parquet_range",),
        ("map_batches", "validate_range_ownership"),
        ("sort_partition", "build_group_boundaries", "map_group_partitions"),
    ]
    assert planned[0].input_dependencies == [operators[0]]
    assert planned[1].input_dependencies == [planned[0]]
    assert planned[2].input_dependencies == [planned[1]]

    specs = [op.gpu_operator_spec for op in planned]
    assert len({spec.fusion_group_id for spec in specs}) == 1
    required_provenance = specs[0].required_logical_provenance
    assert tuple(
        logical_id for spec in specs for logical_id in spec.logical_provenance
    ) == required_provenance
    assert all(
        spec.required_logical_provenance == required_provenance for spec in specs
    )
    assert all(spec.requires_complete_fusion for spec in specs)
    assert [spec.owns_input_parallelism for spec in specs] == [True, False, False]
    assert [spec.uses_terminal_output_adapter for spec in specs] == [False, False, True]
    assert [spec.native_input_payload for spec in specs] == [
        PayloadKind.RANGE_WORK,
        PayloadKind.RANGE_DATA,
        PayloadKind.RANGE_DATA,
    ]
    assert [spec.native_output_payload for spec in specs] == [
        PayloadKind.RANGE_DATA,
        PayloadKind.RANGE_DATA,
        PayloadKind.RANGE_DATA,
    ]
    assert [spec.input_adapter_key for spec in specs] == [
        RANGE_WORK_ADAPTER_KEY,
        RANGE_DATA_BOUNDARY_ADAPTER_KEY,
        RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    ]
    assert [spec.active_output_adapter_key for spec in specs] == [
        RANGE_DATA_BOUNDARY_ADAPTER_KEY,
        RANGE_DATA_BOUNDARY_ADAPTER_KEY,
        RANGE_DATA_TERMINAL_ADAPTER_KEY,
    ]
    transform_ids = tuple(
        transform_id for spec in specs for transform_id in spec.transform_ids
    )
    assert len(transform_ids) == len(set(transform_ids))

    assert set(plan.op_map) == set(operators)
    assert plan.op_map[operators[0]] is read
    assert plan.op_map[planned[0]] is read
    assert plan.op_map[planned[1]] is mapped
    assert plan.op_map[planned[2]] is grouped
    assert operators[0]._logical_operators == [read]
    assert planned[0]._logical_operators == [read]
    assert planned[1]._logical_operators == [mapped]
    assert planned[2]._logical_operators == [grouped]

    optimized = PhysicalOptimizer().optimize(plan)

    assert _planned_ops(optimized) == []
    actors = _actor_ops(optimized)
    assert len(actors) == 1
    actor = actors[0]
    assert _gpu_ops(optimized) == [actor]
    assert _kinds(actor) == (
        "read_parquet_range",
        "map_batches",
        "validate_range_ownership",
        "sort_partition",
        "build_group_boundaries",
        "map_group_partitions",
    )
    assert isinstance(actor.input_dependencies[0], InputDataBuffer)
    assert actor.gpu_operator_spec.logical_provenance == required_provenance
    assert actor.gpu_operator_spec.required_logical_provenance == required_provenance
    assert actor.gpu_operator_spec.requires_complete_fusion
    assert actor.gpu_operator_spec.owns_input_parallelism
    assert actor.gpu_operator_spec.uses_terminal_output_adapter
    assert actor.gpu_creation_options.min_rows_per_bundle == 1
    assert actor._logical_operators == [read, mapped, grouped]
    final_operators = list(optimized.dag.post_order_iter())
    assert set(optimized.op_map) == set(final_operators)
    assert optimized.op_map[actor.input_dependencies[0]] is read
    assert optimized.op_map[actor] is grouped


def test_only_physical_optimizer_composes_and_constructs_backend(
    tmp_path, fusion_context, monkeypatch
):
    compile_calls = []
    factory_calls = []
    original_compile = region_compiler.compile_gpu_region
    original_factory = gpu_operator_module.create_gpu_operator

    def compile_spy(*args, **kwargs):
        compile_calls.append((args, kwargs))
        return original_compile(*args, **kwargs)

    def factory_spy(*args, **kwargs):
        factory_calls.append((args, kwargs))
        return original_factory(*args, **kwargs)

    for module in (gpu_package, region_compiler, gpu_operator_fusion):
        monkeypatch.setattr(module, "compile_gpu_region", compile_spy)
    for module in (gpu_package, gpu_operator_module, gpu_operator_fusion):
        monkeypatch.setattr(module, "create_gpu_operator", factory_spy)

    plan, _ = Planner().plan(_pipeline(_write_parquet(tmp_path))._logical_plan)

    assert len(_planned_ops(plan)) == 3
    assert compile_calls == []
    assert factory_calls == []

    optimized = PhysicalOptimizer().optimize(plan)

    assert len(_actor_ops(optimized)) == 1
    assert len(compile_calls) == 1
    assert len(factory_calls) == 1


class _PhysicalOptimizerWithoutGPUFusion(PhysicalOptimizer):
    @property
    def rules(self):
        return [
            rule for rule in super().rules if not isinstance(rule, FuseGPUOperators)
        ]


def test_missing_physical_fuser_is_fatal(tmp_path, fusion_context):
    plan, _ = Planner().plan(_pipeline(_write_parquet(tmp_path))._logical_plan)

    with pytest.raises(GPURequiredFusionError) as captured:
        _PhysicalOptimizerWithoutGPUFusion().optimize(plan)

    assert "planned gpu" in str(captured.value).lower()


def test_deferred_spec_identity_is_stable_across_replanning(
    tmp_path, fusion_context
):
    dataset = _pipeline(_write_parquet(tmp_path))

    first, _ = Planner().plan(dataset._logical_plan)
    second, _ = Planner().plan(dataset._logical_plan)
    first_specs = [op.gpu_operator_spec for op in _planned_ops(first)]
    second_specs = [op.gpu_operator_spec for op in _planned_ops(second)]

    assert [spec.spec_instance_id for spec in first_specs] == [
        spec.spec_instance_id for spec in second_specs
    ]
    assert [spec.transform_ids for spec in first_specs] == [
        spec.transform_ids for spec in second_specs
    ]
    assert [spec.fusion_group_id for spec in first_specs] == [
        spec.fusion_group_id for spec in second_specs
    ]
    assert [spec.required_logical_provenance for spec in first_specs] == [
        spec.required_logical_provenance for spec in second_specs
    ]


def test_stock_read_parallelism_rule_skips_planned_gpu_source(
    tmp_path, fusion_context, monkeypatch
):
    plan, _ = Planner().plan(_pipeline(_write_parquet(tmp_path))._logical_plan)
    planned = _planned_ops(plan)
    source = planned[0]
    source_input = source.input_dependencies[0]

    def fail_if_stock_read_is_reinterpreted(*args, **kwargs):
        raise AssertionError("stock Read parallelism touched GPU range work")

    monkeypatch.setattr(
        SetReadParallelismRule,
        "_apply",
        fail_if_stock_read_is_reinterpreted,
    )
    result = SetReadParallelismRule().apply(plan)

    assert result is plan
    assert _planned_ops(result) == planned
    assert source.input_dependencies == [source_input]


def test_planner_claims_before_stock_child_planning(
    tmp_path, fusion_context, monkeypatch
):
    def fail_if_stock_child_is_planned(*args, **kwargs):
        raise AssertionError("GPU planner built and discarded a stock child")

    monkeypatch.setitem(Planner._DEFAULT_PLAN_FNS, Read, fail_if_stock_child_is_planned)
    monkeypatch.setitem(
        Planner._DEFAULT_PLAN_FNS, MapBatches, fail_if_stock_child_is_planned
    )

    plan, _ = Planner().plan(_pipeline(_write_parquet(tmp_path))._logical_plan)

    assert len(_planned_ops(plan)) == 3


def test_exact_gpu_fragment_composes_under_stock_parent(tmp_path, fusion_context):
    dataset = _pipeline(_write_parquet(tmp_path)).limit(1)
    parent = dataset._logical_plan.dag
    grouped = parent.input_dependencies[0]

    plan, _ = Planner().plan(dataset._logical_plan)

    assert len(_planned_ops(plan)) == 3
    assert plan.op_map[plan.dag] is parent
    assert plan.op_map[_planned_ops(plan)[-1]] is grouped

    optimized = PhysicalOptimizer().optimize(plan)

    assert len(_actor_ops(optimized)) == 1
    assert optimized.dag is not _actor_ops(optimized)[0]
    assert optimized.op_map[optimized.dag] is parent
    assert optimized.op_map[_actor_ops(optimized)[0]] is grouped


def test_footer_and_range_planning_occurs_only_during_read_lowering(
    tmp_path, fusion_context, monkeypatch
):
    from ray.data._internal.planner import parquet_cudf_range_planner

    calls = []
    original = parquet_cudf_range_planner.plan_parquet_ranges

    def tracked(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(parquet_cudf_range_planner, "plan_parquet_ranges", tracked)
    dataset = _pipeline(_write_parquet(tmp_path))

    required_planner = create_gpu_required_fusion_planner(
        dataset._logical_plan.dag, dataset.context
    )
    assert calls == []

    planned = required_planner.try_plan(dataset._logical_plan.dag, dataset.context)

    assert planned is not None
    assert len(calls) == 1
    assert calls[0][1]["retried_io_errors"] == tuple(
        fusion_context.retried_io_errors
    )
    read = next(
        op
        for op in planned.root.post_order_iter()
        if isinstance(op, PlannedGPUOperator)
        and _kinds(op) == ("read_parquet_range",)
    )
    config = read.gpu_operator_spec.transforms[0].config
    assert config.source_kind == "local"
    assert config.range_key == "User"
    assert config.projection == ("User", "Card", "Merchant", "Value")
    bundles = read.input_dependencies[0]._input_data_factory(
        fusion_context.target_max_block_size
    )
    assert len(bundles) == 2


def test_parquet_range_planning_failure_is_stable(
    tmp_path, fusion_context, monkeypatch
):
    from ray.data._internal.planner import parquet_cudf_range_planner

    def fail_range_planning(*args, **kwargs):
        raise ParquetCudfPlanningError(
            "key_statistics", "test missing range-key statistics"
        )

    monkeypatch.setattr(
        parquet_cudf_range_planner,
        "plan_parquet_ranges",
        fail_range_planning,
    )

    with pytest.raises(GPUPlanningError) as captured:
        Planner().plan(_pipeline(_write_parquet(tmp_path))._logical_plan)

    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "Parquet range planning failed (key_statistics)" in str(captured.value)
    assert isinstance(captured.value.__cause__, ParquetCudfPlanningError)


def test_retry_policy_is_declared_on_every_spec_and_fused_stage(
    tmp_path, fusion_context
):
    fusion_context.actor_task_retry_on_errors = [RuntimeError]
    fusion_context.actor_init_retry_on_errors = True
    fusion_context.actor_init_max_retries = 5
    fusion_context.retried_map_errors = ["transient map"]
    fusion_context.max_map_retries = 7
    fusion_context.retried_io_errors = ["transient io"]

    plan, _ = Planner().plan(_pipeline(_write_parquet(tmp_path))._logical_plan)

    policies = [
        op.gpu_operator_spec.compatibility_key.retry_policy
        for op in _planned_ops(plan)
    ]
    assert len(set(policies)) == 1
    policy = policies[0]
    assert policy.actor_task_retry_on_errors == (RuntimeError,)
    assert policy.actor_init_retry_on_errors
    assert policy.actor_init_max_retries == 5
    assert policy.retried_map_errors == ("transient map",)
    assert policy.max_map_retries == 7
    assert policy.retried_io_errors == ("transient io",)
    read_config = _planned_ops(plan)[0].gpu_operator_spec.transforms[0].config
    assert read_config.retried_io_errors == ("transient io",)

    optimized = PhysicalOptimizer().optimize(plan)

    actor = _actor_ops(optimized)[0]
    assert actor.gpu_operator_spec.compatibility_key.retry_policy == policy


def test_unresolved_final_obligation_fails_during_physical_fusion(
    tmp_path, fusion_context, monkeypatch
):
    original_lower = GPUMapGroupPartitionsLowerer.lower
    pending = RowsOwnedByRange("User", "test-unresolved-output")

    def lower_with_unresolved_output(self, logical_op, context):
        lowered = original_lower(self, logical_op, context)
        transforms = list(lowered.spec.transforms)
        transforms[-1] = replace(
            transforms[-1],
            creates_obligations=transforms[-1].creates_obligations + (pending,),
        )
        return replace(
            lowered,
            spec=replace(lowered.spec, transforms=tuple(transforms)),
        )

    monkeypatch.setattr(
        GPUMapGroupPartitionsLowerer,
        "lower",
        lower_with_unresolved_output,
    )
    plan, _ = Planner().plan(_pipeline(_write_parquet(tmp_path))._logical_plan)
    assert len(_planned_ops(plan)) == 3

    with pytest.raises(GPURequiredFusionError) as captured:
        PhysicalOptimizer().optimize(plan)

    assert isinstance(captured.value.__cause__, GPUCompositionError)
    assert (
        captured.value.__cause__.reason
        is GPUCompositionErrorReason.UNRESOLVED_VALIDATION_OBLIGATION
    )


def test_missing_terminal_adapter_fails_during_physical_materialization(
    tmp_path, fusion_context
):
    adapter = get_gpu_boundary_adapter(
        key=RANGE_DATA_TERMINAL_ADAPTER_KEY,
        direction=GPUBoundaryDirection.EXPORT,
        external_payload=PayloadKind.RAY_BLOCK_STREAM,
        native_payload=PayloadKind.RANGE_DATA,
    )
    assert adapter is not None
    removed = unregister_gpu_boundary_adapter(
        key=RANGE_DATA_TERMINAL_ADAPTER_KEY,
        direction=GPUBoundaryDirection.EXPORT,
        external_payload=PayloadKind.RAY_BLOCK_STREAM,
        native_payload=PayloadKind.RANGE_DATA,
    )
    assert removed is adapter
    try:
        plan, _ = Planner().plan(_pipeline(_write_parquet(tmp_path))._logical_plan)
        assert len(_planned_ops(plan)) == 3
        with pytest.raises(GPURequiredFusionError) as captured:
            PhysicalOptimizer().optimize(plan)
    finally:
        register_gpu_boundary_adapter(adapter)

    assert isinstance(captured.value.__cause__, ValueError)
    assert "backend materialization failed" in str(captured.value)


@pytest.mark.parametrize("factory_mode", ["absent", "refusing"])
def test_absent_or_refusing_gpu_factory_fails_only_during_physical_materialization(
    tmp_path, fusion_context, monkeypatch, factory_mode
):
    class RefusingFactory:
        def __init__(self):
            self.support_calls = 0

        def supports(self, spec):
            del spec
            self.support_calls += 1
            return False

        def create(self, *args, **kwargs):
            raise AssertionError("a refusing factory must not be invoked")

    refusing = RefusingFactory()
    if factory_mode == "absent":
        monkeypatch.delitem(gpu_operator_module._GPU_OPERATOR_FACTORIES, "actor_pool")
    else:
        monkeypatch.setitem(
            gpu_operator_module._GPU_OPERATOR_FACTORIES,
            "actor_pool",
            refusing,
        )

    plan, _ = Planner().plan(_pipeline(_write_parquet(tmp_path))._logical_plan)

    assert len(_planned_ops(plan)) == 3
    assert refusing.support_calls == 0
    with pytest.raises(GPURequiredFusionError) as captured:
        PhysicalOptimizer().optimize(plan)

    assert isinstance(captured.value.__cause__, ValueError)
    assert "backend materialization failed" in str(captured.value)
    if factory_mode == "refusing":
        assert refusing.support_calls == 1


def test_read_placement_is_folded_or_rejected(tmp_path, fusion_context):
    path = _write_parquet(tmp_path)
    inherited = _mgp(
        _map(_read(path, ray_remote_args={"scheduling_strategy": "SPREAD"}))
    )

    plan, _ = Planner().plan(inherited._logical_plan)

    keys = [op.gpu_operator_spec.compatibility_key for op in _planned_ops(plan)]
    assert all(key.scheduling_strategy == "SPREAD" for key in keys)
    optimized = PhysicalOptimizer().optimize(plan)
    assert _actor_ops(optimized)[0].gpu_operator_spec.compatibility_key == keys[0]

    conflict = _mgp(
        _map(
            _read(path, ray_remote_args={"scheduling_strategy": "SPREAD"}),
            scheduling_strategy="DEFAULT",
        )
    )
    with pytest.raises(GPUPlanningError) as captured:
        Planner().plan(conflict._logical_plan)
    assert captured.value.reason is GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS


def test_exact_shape_and_partition_preservation_are_required(
    tmp_path, fusion_context
):
    path = _write_parquet(tmp_path)

    with pytest.raises(GPUPlanningError) as direct_error:
        Planner().plan(_mgp(_read(path))._logical_plan)
    assert direct_error.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "exactly Read -> one cuDF MapBatches" in str(direct_error.value)

    with pytest.raises(GPUPlanningError) as property_error:
        Planner().plan(_mgp(_map(_read(path), preserves=False))._logical_plan)
    assert property_error.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "preserves_partitioning=True" in str(property_error.value)


def test_flag_off_mgp_fails_stably_but_no_mgp_stays_stock(
    tmp_path, fusion_context
):
    path = _write_parquet(tmp_path)
    standalone = _map(_read(path))
    enabled_plan, _ = Planner().plan(standalone._logical_plan)
    assert _gpu_ops(enabled_plan) == []

    dataset = _pipeline(path)
    dataset.context.enable_gpu_operator_fusion = False
    with pytest.raises(GPUPlanningError) as captured:
        Planner().plan(dataset._logical_plan)
    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "no CPU/stock fallback" in str(captured.value)

    standalone = _map(_read(path))
    standalone.context.enable_gpu_operator_fusion = False
    stock_plan, _ = Planner().plan(standalone._logical_plan)
    assert _gpu_ops(stock_plan) == []


def test_checkpointed_exact_gpu_pipeline_is_rejected(tmp_path, fusion_context):
    dataset = _pipeline(_write_parquet(tmp_path))
    dataset.context.checkpoint_config = {
        "id_column": "User",
        "checkpoint_path": str(tmp_path / "checkpoint"),
    }

    with pytest.raises(GPUPlanningError) as captured:
        Planner().plan(dataset._logical_plan)

    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "checkpointing" in str(captured.value)
