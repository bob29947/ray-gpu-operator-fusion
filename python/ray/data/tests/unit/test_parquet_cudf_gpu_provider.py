import copy
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray.data
from ray.data import ActorPoolStrategy, UDFProperties
from ray.data._internal.datasource import parquet_datasource
from ray.data._internal.execution.gpu import (
    CompletePartitionBundles,
    Evidence,
    GPUCompositionError,
    GPUCompositionErrorReason,
    GPUOperator,
    GPUOperatorCreationOptions,
    GPUOperatorSpec,
    GPUTransformSpec,
    PropertyFact,
    RangePartitioning,
    GPUBoundaryAdapter,
    GPUBoundaryDirection,
    PayloadKind,
    RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    RANGE_DATA_TERMINAL_ADAPTER_KEY,
    RANGE_WORK_ADAPTER_KEY,
    RowsOwnedByRange,
    get_gpu_boundary_adapter,
    register_gpu_boundary_adapter,
    register_gpu_transform_runtime,
    unregister_gpu_boundary_adapter,
    unregister_gpu_transform_runtime,
)
from ray.data._internal.execution.operators.actor_pool_gpu_operator import (
    ActorPoolGPUOperator,
)
from ray.data._internal.execution.operators import hash_shuffle
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators import MapBatches, MapGroupPartitions, Read
from ray.data._internal.logical.optimizers import PhysicalOptimizer
from ray.data._internal.logical.rules import FuseGPUOperators
from ray.data._internal.logical.rules.set_read_parallelism import (
    SetReadParallelismRule,
)
from ray.data._internal.planner.gpu_plan_provider import create_gpu_gate2_planner
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)
from ray.data._internal.planner.gpu_read_source_provider import (
    GPUReadSourceArtifacts,
    GPUReadSourceLowering,
    GPUReadSourceSupport,
)
from ray.data._internal.planner.gpu_registrations import (
    register_gpu_logical_lowerer,
    register_gpu_property_enforcer,
    register_gpu_property_transfer,
    register_gpu_read_source_provider,
    unregister_gpu_logical_lowerer,
    unregister_gpu_property_enforcer,
    unregister_gpu_property_transfer,
    unregister_gpu_read_source_provider,
)
from ray.data._internal.planner.gpu_requirement_analysis import (
    GPUEnforcerPlan,
    GPUEnforcerRegistry,
    GPUPropertyState,
)
from ray.data._internal.planner.gpu_lowering.map_batches_lowerer import (
    GPUMapBatchesLowerer,
)
from ray.data._internal.planner.gpu_lowering.map_group_partitions_lowerer import (
    GPUMapGroupPartitionsLowerer,
)
from ray.data._internal.planner.planner import Planner
from ray.data.context import DataContext
from ray.data.datasource.datasource import Datasource


class _Tokenizer:
    def __call__(self, batch):
        return batch


class _Tokenizer2:
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


def _map(
    dataset,
    *,
    fn=_Tokenizer,
    size=1,
    preserves=True,
    scheduling_strategy=None,
):
    remote_args = {"num_cpus": 1, "num_gpus": 1}
    if scheduling_strategy is not None:
        remote_args["scheduling_strategy"] = scheduling_strategy
    return dataset.map_batches(
        fn,
        batch_size=2,
        batch_format="cudf",
        compute=ActorPoolStrategy(size=size),
        udf_properties=(
            UDFProperties(preserves_partitioning=True) if preserves else None
        ),
        **remote_args,
    )


def _mgp(
    dataset,
    *,
    keys=("User", "Card"),
    count=2,
    preserves=False,
):
    return dataset.groupby(list(keys), num_partitions=count).map_group_partitions(
        _partition_fn,
        udf_properties=(
            UDFProperties(preserves_partitioning=True) if preserves else None
        ),
    )


def _pipeline(path):
    return _mgp(_map(_read(path)))


def _gpu_ops(plan):
    return [op for op in plan.dag.post_order_iter() if isinstance(op, GPUOperator)]


def _kinds(gpu_op):
    return tuple(transform.kind for transform in gpu_op.gpu_operator_spec.transforms)


def test_gate2_emits_one_executable_stage_before_physical_fusion(
    tmp_path, fusion_context
):
    dataset = _pipeline(_write_parquet(tmp_path))
    grouped = dataset._logical_plan.dag
    mapped = grouped.input_dependencies[0]
    read = mapped.input_dependencies[0]

    plan, _ = Planner().plan(dataset._logical_plan)

    assert len(_gpu_ops(plan)) == 1
    stage = _gpu_ops(plan)[0]
    assert isinstance(stage.input_dependencies[0], InputDataBuffer)
    assert _kinds(stage) == (
        "read_parquet_range",
        "map_batches",
        "validate_range_ownership",
        "sort_partition",
        "build_group_boundaries",
        "map_group_partitions",
    )
    assert stage.gpu_operator_spec.gate2_composed
    assert stage.gpu_operator_spec.specialized_read_parallelism_resolved
    assert stage.gpu_creation_options.min_rows_per_bundle == 1
    assert stage._min_rows_per_bundle == 1
    assert stage._logical_operators == [read, mapped, grouped]
    assert plan.op_map[stage.input_dependencies[0]] is read
    assert plan.op_map[stage] is grouped
    assert len(stage.gpu_operator_spec.transform_ids) == len(
        set(stage.gpu_operator_spec.transform_ids)
    )
    assert not any(
        "boundary" in kind or kind in ("import", "export") for kind in _kinds(stage)
    )

    optimized = PhysicalOptimizer().optimize(plan)
    assert len(_gpu_ops(optimized)) == 1
    assert _gpu_ops(optimized)[0].gpu_operator_spec.gate2_composed


class _PhysicalOptimizerWithoutGPUFusion(PhysicalOptimizer):
    @property
    def rules(self):
        return [
            rule for rule in super().rules if not isinstance(rule, FuseGPUOperators)
        ]


def test_gate2_stage_is_executable_without_physical_gpu_fusion(
    tmp_path, fusion_context
):
    dataset = _mgp(_read(_write_parquet(tmp_path)))
    plan, _ = Planner().plan(dataset._logical_plan)

    optimized = _PhysicalOptimizerWithoutGPUFusion().optimize(plan)

    assert len(_gpu_ops(optimized)) == 1
    stage = _gpu_ops(optimized)[0]
    assert isinstance(stage, ActorPoolGPUOperator)
    assert stage.gpu_operator_spec.gate2_composed
    assert not stage.supports_fusion()


def test_gate2_identity_is_stable_across_replanning(tmp_path, fusion_context):
    dataset = _pipeline(_write_parquet(tmp_path))

    first, _ = Planner().plan(dataset._logical_plan)
    second, _ = Planner().plan(dataset._logical_plan)
    first_spec = _gpu_ops(first)[0].gpu_operator_spec
    second_spec = _gpu_ops(second)[0].gpu_operator_spec

    assert first_spec.transform_ids == second_spec.transform_ids
    assert first_spec.spec_instance_id == second_spec.spec_instance_id


def test_stock_read_parallelism_rule_skips_specialized_stage(tmp_path, fusion_context):
    plan, _ = Planner().plan(_pipeline(_write_parquet(tmp_path))._logical_plan)
    stage = _gpu_ops(plan)[0]
    stage_input = stage.input_dependencies[0]

    result = SetReadParallelismRule().apply(plan)

    assert result is plan
    assert result.dag is stage
    assert stage.input_dependencies[0] is stage_input


def test_gate2_claims_before_stock_child_planning(
    tmp_path, fusion_context, monkeypatch
):
    def fail_if_stock_child_is_planned(*args, **kwargs):
        raise AssertionError("gate 2 built and discarded a stock child")

    monkeypatch.setitem(Planner._DEFAULT_PLAN_FNS, Read, fail_if_stock_child_is_planned)
    monkeypatch.setitem(
        Planner._DEFAULT_PLAN_FNS, MapBatches, fail_if_stock_child_is_planned
    )

    plan, _ = Planner().plan(_pipeline(_write_parquet(tmp_path))._logical_plan)
    assert len(_gpu_ops(plan)) == 1


class _RegisteredSource(Datasource):
    pass


class _RegisteredSourceProvider:
    provider_name = "test_registered_non_parquet"

    def __init__(
        self,
        *,
        transform_kind="registered_source",
        transform_config=None,
    ):
        self.support_calls = 0
        self.lower_calls = 0
        self.transform_kind = transform_kind
        self.transform_config = (
            (lambda value, _context: value)
            if transform_config is None
            else transform_config
        )

    def supports(self, logical_op, demand):
        del demand
        self.support_calls += 1
        accepted = (
            type(logical_op) is Read
            and type(logical_op.datasource) is _RegisteredSource
        )
        return GPUReadSourceSupport(
            accepted,
            "not_registered_test_source" if not accepted else "",
        )

    def lower(self, logical_op, request):
        assert type(logical_op.datasource) is _RegisteredSource
        self.lower_calls += 1
        range_properties = (
            PropertyFact(
                RangePartitioning((request.group_keys[0],), "registered-source"),
                Evidence.OPERATOR_GUARANTEE,
            ),
            PropertyFact(
                CompletePartitionBundles(),
                Evidence.OPERATOR_GUARANTEE,
            ),
        )
        return GPUReadSourceLowering(
            spec=GPUOperatorSpec(
                spec_instance_id=request.logical_node_id,
                transforms=(
                    GPUTransformSpec(
                        transform_id=request.transform_id(0, self.transform_kind),
                        kind=self.transform_kind,
                        input_payload=PayloadKind.RANGE_WORK,
                        output_payload=PayloadKind.RANGE_DATA,
                        config=self.transform_config,
                    ),
                ),
                external_input_payload=PayloadKind.RAY_BLOCK_STREAM,
                external_output_payload=PayloadKind.RAY_BLOCK_STREAM,
                native_input_payload=PayloadKind.RANGE_WORK,
                native_output_payload=PayloadKind.RANGE_DATA,
                input_adapter_key=RANGE_WORK_ADAPTER_KEY,
                output_adapter_key=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
                terminal_output_adapter_key=RANGE_DATA_TERMINAL_ADAPTER_KEY,
                execution_requirements=request.execution_requirements,
                output_properties=range_properties,
                logical_provenance=(request.logical_node_id,),
            ),
            creation_options=GPUOperatorCreationOptions(
                name="RegisteredGPUSource",
                min_rows_per_bundle=1,
                target_max_block_size_override=request.target_max_block_size,
            ),
            artifacts=GPUReadSourceArtifacts(
                input_data_factory=lambda _target_size: [],
                provider_artifacts="registered-source-artifacts",
            ),
        )


def test_globally_registered_non_parquet_source_is_used_by_planner(fusion_context):
    source = _RegisteredSource()
    read = Read(
        datasource=source,
        datasource_or_legacy_reader=source,
        parallelism=1,
    )
    mapped = MapBatches(
        fn=_Tokenizer,
        input_dependencies=[read],
        batch_size=2,
        batch_format="cudf",
        compute=ActorPoolStrategy(size=1),
        ray_remote_args={"num_cpus": 1, "num_gpus": 1},
        udf_properties=UDFProperties(preserves_partitioning=True),
    )
    grouped = MapGroupPartitions(
        key=["User", "Card"],
        num_partitions=2,
        fn=_partition_fn,
        input_dependencies=[mapped],
    )
    provider = _RegisteredSourceProvider()
    register_gpu_read_source_provider(provider)
    try:
        plan, _ = Planner().plan(LogicalPlan(grouped, fusion_context))
    finally:
        unregister_gpu_read_source_provider(
            provider.provider_name,
            expected=provider,
        )

    assert provider.support_calls == 1
    assert provider.lower_calls == 1
    assert len(_gpu_ops(plan)) == 1
    assert _kinds(_gpu_ops(plan)[0])[0] == "registered_source"


def _registered_source_component(fusion_context):
    source = _RegisteredSource()
    read = Read(
        datasource=source,
        datasource_or_legacy_reader=source,
        parallelism=1,
    )
    mapped = MapBatches(
        fn=_Tokenizer,
        input_dependencies=[read],
        batch_size=2,
        batch_format="cudf",
        compute=ActorPoolStrategy(size=1),
        ray_remote_args={"num_cpus": 1, "num_gpus": 1},
        udf_properties=UDFProperties(preserves_partitioning=True),
    )
    return MapGroupPartitions(
        key=["User", "Card"],
        num_partitions=2,
        fn=_partition_fn,
        input_dependencies=[mapped],
    ), fusion_context


def test_unknown_transform_runtime_is_rejected_by_planner(fusion_context):
    provider = _RegisteredSourceProvider(
        transform_kind="test_unknown_runtime",
        transform_config=object(),
    )
    component, context = _registered_source_component(fusion_context)
    register_gpu_read_source_provider(provider)
    try:
        with pytest.raises(GPUPlanningError) as captured:
            Planner().plan(LogicalPlan(component, context))
    finally:
        unregister_gpu_read_source_provider(
            provider.provider_name,
            expected=provider,
        )

    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "factory or boundary adapter refused" in str(captured.value)


def test_registered_transform_runtime_is_additive_at_planning(fusion_context):
    kind = "test_registered_runtime"
    provider = _RegisteredSourceProvider(
        transform_kind=kind,
        transform_config=object(),
    )
    component, context = _registered_source_component(fusion_context)
    constructed = []

    def factory(transform_spec, runtime_context):
        constructed.append((transform_spec, runtime_context))
        raise AssertionError("planning must not construct a transform runtime")

    register_gpu_transform_runtime(kind, factory)
    register_gpu_read_source_provider(provider)
    try:
        plan, _ = Planner().plan(LogicalPlan(component, context))
    finally:
        unregister_gpu_read_source_provider(
            provider.provider_name,
            expected=provider,
        )
        unregister_gpu_transform_runtime(kind, expected=factory)

    assert len(_gpu_ops(plan)) == 1
    assert constructed == []


class _RegisteredUnary(MapBatches):
    pass


def _as_builtin_map_batches(logical_op):
    return MapBatches(
        fn=logical_op.fn,
        input_dependencies=logical_op.input_dependencies,
        can_modify_num_rows=logical_op.can_modify_num_rows,
        udf_properties=logical_op.udf_properties,
        batch_size=logical_op.batch_size,
        batch_format=logical_op.batch_format,
        zero_copy_batch=logical_op.zero_copy_batch,
        fn_args=logical_op.fn_args,
        fn_kwargs=logical_op.fn_kwargs,
        fn_constructor_args=logical_op.fn_constructor_args,
        fn_constructor_kwargs=logical_op.fn_constructor_kwargs,
        min_rows_per_bundled_input=logical_op.min_rows_per_bundled_input,
        compute=logical_op.compute,
        ray_remote_args_fn=logical_op.ray_remote_args_fn,
        ray_remote_args=logical_op.ray_remote_args,
        per_block_limit=logical_op.per_block_limit,
    )


class _RegisteredUnaryLowerer:
    component_role = GPUMapBatchesLowerer.component_role

    def __init__(self):
        self._delegate = GPUMapBatchesLowerer()
        self.lower_calls = 0

    def analyze(self, logical_op, data_context):
        return self._delegate.analyze(_as_builtin_map_batches(logical_op), data_context)

    def lower(self, logical_op, context):
        self.lower_calls += 1
        return self._delegate.lower(_as_builtin_map_batches(logical_op), context)


class _IncompleteRegisteredUnaryLowerer(_RegisteredUnaryLowerer):
    def lower(self, logical_op, context):
        lowered = super().lower(logical_op, context)
        incomplete_facts = lowered.spec.output_properties[:1]
        return replace(
            lowered,
            spec=replace(
                lowered.spec,
                output_properties=incomplete_facts,
            ),
            downstream_context=replace(
                lowered.downstream_context,
                input_properties=incomplete_facts,
            ),
        )


class _PreservingRegisteredUnaryTransfer:
    def transfer(self, logical_op, input_state):
        assert type(logical_op) is _RegisteredUnary
        return input_state


class _ConcreteFailingRegisteredUnaryTransfer:
    def transfer(self, logical_op, input_state):
        assert type(logical_op) is _RegisteredUnary
        range_fact = next(
            fact
            for fact in input_state.facts
            if isinstance(fact.value, RangePartitioning)
        )
        if range_fact.value.scheme_id.startswith("analysis:"):
            return input_state
        raise RuntimeError("test concrete property transfer failure")


def test_globally_registered_unary_and_transfer_are_used_by_planner(
    tmp_path, fusion_context
):
    read = _read(_write_parquet(tmp_path))._logical_plan.dag
    unary = _RegisteredUnary(
        fn=_Tokenizer,
        input_dependencies=[read],
        batch_size=2,
        batch_format="cudf",
        compute=ActorPoolStrategy(size=1),
        ray_remote_args={"num_cpus": 1, "num_gpus": 1},
        udf_properties=UDFProperties(preserves_partitioning=True),
    )
    grouped = MapGroupPartitions(
        key=["User", "Card"],
        num_partitions=2,
        fn=_partition_fn,
        input_dependencies=[unary],
    )
    lowerer = _RegisteredUnaryLowerer()
    transfer = _PreservingRegisteredUnaryTransfer()
    register_gpu_logical_lowerer(_RegisteredUnary, lowerer)
    try:
        register_gpu_property_transfer(_RegisteredUnary, transfer)
        try:
            plan, _ = Planner().plan(LogicalPlan(grouped, fusion_context))
        finally:
            unregister_gpu_property_transfer(_RegisteredUnary, expected=transfer)
    finally:
        unregister_gpu_logical_lowerer(_RegisteredUnary, expected=lowerer)

    assert lowerer.lower_calls == 1
    assert len(_gpu_ops(plan)) == 1
    assert "map_batches" in _kinds(_gpu_ops(plan)[0])


def test_registered_unary_lowerer_must_match_concrete_property_transfer(
    tmp_path, fusion_context
):
    read = _read(_write_parquet(tmp_path))._logical_plan.dag
    unary = _RegisteredUnary(
        fn=_Tokenizer,
        input_dependencies=[read],
        batch_size=2,
        batch_format="cudf",
        compute=ActorPoolStrategy(size=1),
        ray_remote_args={"num_cpus": 1, "num_gpus": 1},
        udf_properties=UDFProperties(preserves_partitioning=True),
    )
    grouped = MapGroupPartitions(
        key=["User", "Card"],
        num_partitions=2,
        fn=_partition_fn,
        input_dependencies=[unary],
    )
    lowerer = _IncompleteRegisteredUnaryLowerer()
    transfer = _PreservingRegisteredUnaryTransfer()
    register_gpu_logical_lowerer(_RegisteredUnary, lowerer)
    try:
        register_gpu_property_transfer(_RegisteredUnary, transfer)
        try:
            with pytest.raises(GPUPlanningError) as captured:
                Planner().plan(LogicalPlan(grouped, fusion_context))
        finally:
            unregister_gpu_property_transfer(_RegisteredUnary, expected=transfer)
    finally:
        unregister_gpu_logical_lowerer(_RegisteredUnary, expected=lowerer)

    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "GPU lowerer and registered property transfer disagree" in str(
        captured.value
    )
    assert "expected_output_facts=" in str(captured.value)
    assert "actual_output_facts=" in str(captured.value)
    assert "CompletePartitionBundles" in str(captured.value)


def test_concrete_property_transfer_failure_is_chained_planning_error(
    tmp_path, fusion_context
):
    read = _read(_write_parquet(tmp_path))._logical_plan.dag
    unary = _RegisteredUnary(
        fn=_Tokenizer,
        input_dependencies=[read],
        batch_size=2,
        batch_format="cudf",
        compute=ActorPoolStrategy(size=1),
        ray_remote_args={"num_cpus": 1, "num_gpus": 1},
        udf_properties=UDFProperties(preserves_partitioning=True),
    )
    grouped = MapGroupPartitions(
        key=["User", "Card"],
        num_partitions=2,
        fn=_partition_fn,
        input_dependencies=[unary],
    )
    lowerer = _RegisteredUnaryLowerer()
    transfer = _ConcreteFailingRegisteredUnaryTransfer()
    register_gpu_logical_lowerer(_RegisteredUnary, lowerer)
    try:
        register_gpu_property_transfer(_RegisteredUnary, transfer)
        try:
            with pytest.raises(GPUPlanningError) as captured:
                Planner().plan(LogicalPlan(grouped, fusion_context))
        finally:
            unregister_gpu_property_transfer(_RegisteredUnary, expected=transfer)
    finally:
        unregister_gpu_logical_lowerer(_RegisteredUnary, expected=lowerer)

    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "registered property transfer failed" in str(captured.value)
    assert isinstance(captured.value.__cause__, RuntimeError)
    assert "test concrete property transfer failure" in str(captured.value.__cause__)


def test_read_to_mgp_and_repeated_maps(tmp_path, fusion_context):
    path = _write_parquet(tmp_path)
    direct = _mgp(_read(path))
    direct_plan, _ = Planner().plan(direct._logical_plan)
    assert _kinds(_gpu_ops(direct_plan)[0]) == (
        "read_parquet_range",
        "sort_partition",
        "build_group_boundaries",
        "map_group_partitions",
    )

    repeated = _mgp(_map(_map(_read(path)), fn=_Tokenizer2))
    repeated_plan, _ = Planner().plan(repeated._logical_plan)
    kinds = _kinds(_gpu_ops(repeated_plan)[0])
    assert kinds.count("map_batches") == 2
    assert kinds.count("validate_range_ownership") == 2


def test_trailing_map_is_stock(tmp_path, fusion_context):
    dataset = _map(_pipeline(_write_parquet(tmp_path)), fn=_Tokenizer2)
    trailing = dataset._logical_plan.dag
    grouped = trailing.input_dependencies[0]

    plan, _ = Planner().plan(dataset._logical_plan)

    assert len(_gpu_ops(plan)) == 1
    stage = _gpu_ops(plan)[0]
    assert plan.dag is not stage
    assert plan.op_map[plan.dag] is trailing
    assert plan.op_map[stage] is grouped
    assert trailing not in stage._logical_operators


def test_immediate_repeated_mgp_is_one_stage(tmp_path, fusion_context):
    path = _write_parquet(tmp_path)
    first = _mgp(_map(_read(path)), preserves=True)
    dataset = _mgp(first, keys=("User", "Merchant"))

    plan, _ = Planner().plan(dataset._logical_plan)

    assert len(_gpu_ops(plan)) == 1
    kinds = _kinds(_gpu_ops(plan)[0])
    assert kinds.count("map_group_partitions") == 2
    assert kinds.count("sort_partition") == 2
    assert kinds.count("validate_range_ownership") == 2


@pytest.mark.parametrize(
    "second_keys,second_count",
    [
        (("Card", "User"), 2),
        (("User", "Merchant"), 3),
    ],
)
def test_repeated_mgp_incompatible_distribution_requires_exchange(
    tmp_path, fusion_context, second_keys, second_count
):
    first = _mgp(
        _map(_read(_write_parquet(tmp_path))),
        preserves=True,
    )
    dataset = _mgp(first, keys=second_keys, count=second_count)

    with pytest.raises(GPUPlanningError) as captured:
        Planner().plan(dataset._logical_plan)
    assert captured.value.reason is GPUPlanningErrorReason.EXCHANGE_REQUIRED
    assert "required_keys" in str(captured.value)
    assert "required_count" in str(captured.value)
    assert "no GPU exchange is registered" in str(captured.value)


def test_property_invalidation_and_profile_mismatch_require_exchange(
    tmp_path, fusion_context
):
    path = _write_parquet(tmp_path)
    invalidated = _mgp(_map(_map(_read(path)), preserves=False))
    with pytest.raises(GPUPlanningError) as invalidated_error:
        Planner().plan(invalidated._logical_plan)
    assert invalidated_error.value.reason is GPUPlanningErrorReason.EXCHANGE_REQUIRED
    assert "first_invalidator=MapBatches" in str(invalidated_error.value)

    mismatched = _mgp(_map(_map(_read(path), size=1), size=2))
    with pytest.raises(GPUPlanningError) as profile_error:
        Planner().plan(mismatched._logical_plan)
    assert profile_error.value.reason is GPUPlanningErrorReason.EXCHANGE_REQUIRED
    assert "mismatched execution profiles" in str(profile_error.value)


def test_branch_demands_conflict_or_build_independent_stages(tmp_path, fusion_context):
    shared = _map(_read(_write_parquet(tmp_path)))
    first = _mgp(shared, keys=("User", "Card"))
    conflicting = _mgp(shared, keys=("Card", "User"))
    with pytest.raises(GPUPlanningError) as conflict:
        Planner().plan(first.union(conflicting)._logical_plan)
    assert conflict.value.reason is GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS

    compatible = _mgp(shared, keys=("User", "Merchant"))
    plan, _ = Planner().plan(first.union(compatible)._logical_plan)
    assert len(_gpu_ops(plan)) == 2
    assert all(
        isinstance(stage.input_dependencies[0], InputDataBuffer)
        for stage in _gpu_ops(plan)
    )


def test_footer_planning_occurs_only_inside_read_lowering(
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
    gate2 = create_gpu_gate2_planner(dataset._logical_plan.dag, dataset.context)
    assert calls == []

    planned = gate2.try_plan(dataset._logical_plan.dag, dataset.context)
    assert planned is not None
    assert len(calls) == 1


class _FakeRangeEnforcer:
    def __init__(self, *, invalid_materialization=False):
        self.invalid_materialization = invalid_materialization
        self.calls = []

    def supports(self, request):
        del request
        return True

    def established_state(self, request):
        return GPUPropertyState(
            facts=(
                PropertyFact(
                    RangePartitioning((request.required_keys[0],), "fake-exchange"),
                    Evidence.OPERATOR_GUARANTEE,
                ),
                PropertyFact(
                    CompletePartitionBundles(),
                    Evidence.OPERATOR_GUARANTEE,
                ),
            ),
            partition_count=request.required_partition_count,
        )

    def plan(self, request, *, upstream, data_context):
        del data_context
        self.calls.append(request)
        state = self.established_state(request)
        if self.invalid_materialization:
            state = GPUPropertyState(
                facts=state.facts,
                partition_count=request.required_partition_count + 1,
            )
        return GPUEnforcerPlan(
            physical_input=upstream,
            output_state=state,
            op_map={},
        )


@contextmanager
def _registered_fake_range_boundary_adapters():
    """Install the two preserving ports a future enforcer must provide."""

    adapters = tuple(
        GPUBoundaryAdapter(
            key=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
            direction=direction,
            external_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_payload=PayloadKind.RANGE_DATA,
            adapt=lambda value, _task_context: value,
        )
        for direction in (
            GPUBoundaryDirection.IMPORT,
            GPUBoundaryDirection.EXPORT,
        )
    )
    for adapter in adapters:
        register_gpu_boundary_adapter(adapter)
    try:
        yield
    finally:
        for adapter in adapters:
            unregister_gpu_boundary_adapter(
                key=adapter.key,
                direction=adapter.direction,
                external_payload=adapter.external_payload,
                native_payload=adapter.native_payload,
            )


def test_globally_registered_enforcer_is_used_by_planner(tmp_path, fusion_context):
    dataset = _mgp(
        _map(_map(_read(_write_parquet(tmp_path))), preserves=False),
    )
    enforcer = _FakeRangeEnforcer()
    register_gpu_property_enforcer(enforcer)
    try:
        with _registered_fake_range_boundary_adapters():
            plan, _ = Planner().plan(dataset._logical_plan)
    finally:
        unregister_gpu_property_enforcer(enforcer)

    assert len(enforcer.calls) == 1
    assert len(_gpu_ops(plan)) == 2


def test_registered_enforcer_splits_stage_and_carries_validated_state(
    tmp_path, fusion_context
):
    first = _mgp(
        _map(_map(_read(_write_parquet(tmp_path))), preserves=False),
        preserves=True,
    )
    dataset = _mgp(first, keys=("User", "Merchant"))
    enforcer = _FakeRangeEnforcer()
    registry = GPUEnforcerRegistry()
    registry.register(enforcer)
    gate2 = create_gpu_gate2_planner(
        dataset._logical_plan.dag,
        dataset.context,
        enforcers=registry,
    )

    with _registered_fake_range_boundary_adapters():
        planned = gate2.try_plan(dataset._logical_plan.dag, dataset.context)

    assert planned is not None
    assert len(enforcer.calls) == 1
    gpu_ops = [
        op for op in planned.root.post_order_iter() if isinstance(op, GPUOperator)
    ]
    assert len(gpu_ops) == 2
    assert not gpu_ops[0].gpu_operator_spec.uses_terminal_output_adapter
    assert (
        gpu_ops[0].gpu_operator_spec.active_output_adapter_key
        == RANGE_DATA_BOUNDARY_ADAPTER_KEY
    )
    assert gpu_ops[1].gpu_operator_spec.uses_terminal_output_adapter
    assert (
        gpu_ops[1].gpu_operator_spec.active_output_adapter_key
        == RANGE_DATA_TERMINAL_ADAPTER_KEY
    )
    assert _kinds(planned.root).count("map_group_partitions") == 2


def test_profile_boundary_uses_downstream_profile_and_default_partition_count(
    tmp_path, fusion_context
):
    profile_a = _map(_read(_write_parquet(tmp_path)), size=1)
    profile_b = _map(profile_a, fn=_Tokenizer2, size=2)
    dataset = _mgp(profile_b, count=None)
    enforcer = _FakeRangeEnforcer()
    registry = GPUEnforcerRegistry()
    registry.register(enforcer)
    gate2 = create_gpu_gate2_planner(
        dataset._logical_plan.dag,
        dataset.context,
        enforcers=registry,
    )

    component = gate2.analysis.components_by_anchor[dataset._logical_plan.dag]
    assert component.mgp_requirements[0].partition_count == 4
    assert component.source_demand.partition_count == 4

    with _registered_fake_range_boundary_adapters():
        planned = gate2.try_plan(dataset._logical_plan.dag, dataset.context)

    assert planned is not None
    assert len(enforcer.calls) == 1
    request = enforcer.calls[0]
    assert request.required_partition_count == 4
    assert request.upstream_execution_requirements.compatibility_key.max_workers == 1
    assert request.downstream_execution_requirements.compatibility_key.max_workers == 2
    gpu_ops = [
        op for op in planned.root.post_order_iter() if isinstance(op, GPUOperator)
    ]
    assert len(gpu_ops) == 2
    assert [op.gpu_operator_spec.compatibility_key.max_workers for op in gpu_ops] == [
        1,
        2,
    ]
    assert _kinds(gpu_ops[0]) == (
        "read_parquet_range",
        "map_batches",
        "validate_range_ownership",
    )
    assert _kinds(gpu_ops[1]) == (
        "map_batches",
        "validate_range_ownership",
        "sort_partition",
        "build_group_boundaries",
        "map_group_partitions",
    )


def test_enforcer_materialization_must_match_analyzed_property_state(
    tmp_path, fusion_context
):
    dataset = _mgp(_map(_map(_read(_write_parquet(tmp_path))), preserves=False))
    enforcer = _FakeRangeEnforcer(invalid_materialization=True)
    registry = GPUEnforcerRegistry()
    registry.register(enforcer)
    gate2 = create_gpu_gate2_planner(
        dataset._logical_plan.dag,
        dataset.context,
        enforcers=registry,
    )

    with (
        _registered_fake_range_boundary_adapters(),
        pytest.raises(GPUPlanningError) as captured,
    ):
        gate2.try_plan(dataset._logical_plan.dag, dataset.context)
    assert captured.value.reason is GPUPlanningErrorReason.EXCHANGE_REQUIRED
    assert "did not establish" in str(captured.value)


def test_unresolved_final_obligation_is_a_stable_planning_error(
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

    with pytest.raises(GPUPlanningError) as captured:
        dataset = _mgp(_read(_write_parquet(tmp_path)))
        Planner().plan(dataset._logical_plan)

    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "gate-2 final GPU stage composition refused" in str(captured.value)
    assert isinstance(captured.value.__cause__, GPUCompositionError)
    assert (
        captured.value.__cause__.reason
        is GPUCompositionErrorReason.UNRESOLVED_VALIDATION_OBLIGATION
    )


def test_missing_terminal_adapter_is_a_stable_planning_error(tmp_path, fusion_context):
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
        with pytest.raises(GPUPlanningError) as captured:
            dataset = _mgp(_read(_write_parquet(tmp_path)))
            Planner().plan(dataset._logical_plan)
    finally:
        register_gpu_boundary_adapter(adapter)

    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "factory or boundary adapter refused" in str(captured.value)
    assert type(captured.value.__cause__) is ValueError


@pytest.mark.parametrize("factory_mode", ["absent", "refusing"])
def test_absent_or_refusing_gpu_factory_is_a_stable_planning_error(
    tmp_path, fusion_context, monkeypatch, factory_mode
):
    import ray.data._internal.execution.gpu.operator as gpu_operator_module

    class RefusingFactory:
        def supports(self, spec):
            del spec
            return False

        def create(self, *args, **kwargs):
            raise AssertionError("a refusing factory must not be invoked")

    if factory_mode == "absent":
        monkeypatch.delitem(
            gpu_operator_module._GPU_OPERATOR_FACTORIES,
            "actor_pool",
        )
    else:
        monkeypatch.setitem(
            gpu_operator_module._GPU_OPERATOR_FACTORIES,
            "actor_pool",
            RefusingFactory(),
        )

    with pytest.raises(GPUPlanningError) as captured:
        dataset = _mgp(_read(_write_parquet(tmp_path)))
        Planner().plan(dataset._logical_plan)

    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "factory or boundary adapter refused" in str(captured.value)
    assert type(captured.value.__cause__) is ValueError


def test_read_placement_is_folded_or_rejected(tmp_path, fusion_context):
    path = _write_parquet(tmp_path)
    inherited = _mgp(
        _map(_read(path, ray_remote_args={"scheduling_strategy": "SPREAD"}))
    )
    plan, _ = Planner().plan(inherited._logical_plan)
    key = _gpu_ops(plan)[0].gpu_operator_spec.compatibility_key
    assert key.scheduling_strategy == "SPREAD"

    conflict = _mgp(
        _map(
            _read(path, ray_remote_args={"scheduling_strategy": "SPREAD"}),
            scheduling_strategy="DEFAULT",
        )
    )
    with pytest.raises(GPUPlanningError) as captured:
        Planner().plan(conflict._logical_plan)
    assert captured.value.reason is GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS


def test_flag_off_mgp_fails_stably_but_no_mgp_stays_stock(tmp_path, fusion_context):
    path = _write_parquet(tmp_path)
    stock_enabled = _map(_read(path))
    enabled_plan, _ = Planner().plan(stock_enabled._logical_plan)
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


def test_checkpointed_mgp_is_rejected_during_analysis(tmp_path, fusion_context):
    dataset = _pipeline(_write_parquet(tmp_path))
    dataset.context.checkpoint_config = {
        "id_column": "User",
        "checkpoint_path": str(tmp_path / "checkpoint"),
    }
    with pytest.raises(GPUPlanningError) as captured:
        Planner().plan(dataset._logical_plan)
    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "checkpointing" in str(captured.value)
