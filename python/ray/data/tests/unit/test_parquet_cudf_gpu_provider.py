import copy
from datetime import datetime, timezone
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray
import ray.data
from ray.data import ActorPoolStrategy, UDFProperties
from ray.data._internal.datasource import parquet_datasource
from ray.data._internal.execution.gpu.operator import GPUOperator, PlannedGPUOperator
from ray.data._internal.execution.gpu.operator_spec import (
    RANGE_DATA,
    RANGE_WORK,
    RAY_BLOCK_STREAM,
)
from ray.data._internal.execution.gpu.properties import (
    KeysColocated,
    RangePartitioning,
    RangePartitioningRequired,
)
from ray.data._internal.execution.gpu.transforms.boundary import (
    EXPORT_RANGE_DATA_KIND,
    IMPORT_RANGE_WORK_KIND,
)
from ray.data._internal.execution.operators import hash_shuffle
from ray.data._internal.execution.operators.actor_pool_gpu_operator import (
    ActorPoolGPUOperator,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.optimizers import PhysicalOptimizer
from ray.data._internal.logical.rules.set_read_parallelism import (
    SetReadParallelismRule,
)
from ray.data._internal.planner.gpu_lowering.interfaces import GPUDecline
from ray.data._internal.planner.gpu_plan_provider import (
    create_gpu_required_fusion_planner,
)
from ray.data._internal.planner.gpu_planning_errors import GPUPlanningErrorReason
from ray.data._internal.planner.parquet_cudf_range_planner import (
    ParquetCudfPlanningError,
    source_identity,
)
from ray.data._internal.planner.parquet_cudf_read_recognizer import (
    ParquetCudfReadDescriptor,
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


def _planned_ops(plan):
    return [
        op for op in plan.dag.post_order_iter() if isinstance(op, PlannedGPUOperator)
    ]


def _actor_ops(plan):
    return [
        op for op in plan.dag.post_order_iter() if isinstance(op, ActorPoolGPUOperator)
    ]


def _kinds(gpu_op):
    return tuple(transform.kind for transform in gpu_op.gpu_operator_spec.transforms)


def test_provider_lowers_three_independent_core_specs_and_source_artifacts(
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
    specs = [op.gpu_operator_spec for op in planned]
    assert [(spec.input_payload, spec.output_payload) for spec in specs] == [
        (RANGE_WORK, RANGE_DATA),
        (RANGE_DATA, RANGE_DATA),
        (RANGE_DATA, RANGE_DATA),
    ]
    range_fact = specs[0].output_facts[0]
    assert isinstance(range_fact, RangePartitioning)
    assert range_fact.keys == ("User",)
    assert specs[1].input_requirements == (
        RangePartitioningRequired(range_fact.keys, range_fact.scheme_id),
    )
    assert specs[1].output_facts == (range_fact,)
    assert specs[2].input_requirements == (KeysColocated(("User", "Card")),)
    assert specs[2].output_facts == ()
    assert [op.owns_input_parallelism for op in planned] == [True, False, False]
    assert planned[0].input_dependencies == [operators[0]]
    assert planned[1].input_dependencies == [planned[0]]
    assert planned[2].input_dependencies == [planned[1]]
    assert plan.op_map[operators[0]] is read
    assert plan.op_map[planned[0]] is read
    assert plan.op_map[planned[1]] is mapped
    assert plan.op_map[planned[2]] is grouped
    assert planned[0]._logical_operators == [read]
    assert planned[1]._logical_operators == [mapped]
    assert planned[2]._logical_operators == [grouped]

    config = specs[0].transforms[0].config
    assert config.source_kind == "local"
    assert config.range_key == "User"
    assert config.projection == ("User", "Card", "Merchant", "Value")
    assert not hasattr(config, "filesystem")
    assert not hasattr(config, "paths")
    bundles = operators[0]._input_data_factory(fusion_context.target_max_block_size)
    assert len(bundles) == 2


def test_physical_backend_adds_registered_boundary_transforms_only_at_region_ends(
    tmp_path, fusion_context
):
    plan, _ = Planner().plan(_pipeline(_write_parquet(tmp_path))._logical_plan)
    assert all(
        transform.input_payload != RAY_BLOCK_STREAM
        and transform.output_payload != RAY_BLOCK_STREAM
        for op in _planned_ops(plan)
        for transform in op.gpu_operator_spec.transforms
    )

    optimized = PhysicalOptimizer().optimize(plan)

    assert _planned_ops(optimized) == []
    actor = _actor_ops(optimized)[0]
    transforms = actor.gpu_operator_spec.transforms
    assert transforms[0].kind == IMPORT_RANGE_WORK_KIND
    assert transforms[0].input_payload == RAY_BLOCK_STREAM
    assert transforms[0].output_payload == RANGE_WORK
    assert transforms[-1].kind == EXPORT_RANGE_DATA_KIND
    assert transforms[-1].input_payload == RANGE_DATA
    assert transforms[-1].output_payload == RAY_BLOCK_STREAM
    assert _kinds(actor)[1:-1] == (
        "read_parquet_range",
        "map_batches",
        "validate_range_ownership",
        "sort_partition",
        "build_group_boundaries",
        "map_group_partitions",
    )
    assert actor._logical_operators == [
        plan.op_map[_planned_ops(plan)[0]],
        plan.op_map[_planned_ops(plan)[1]],
        plan.op_map[_planned_ops(plan)[2]],
    ]


def test_footer_io_happens_only_when_the_source_node_is_lowered(
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
    root = dataset._logical_plan.dag
    read = root.input_dependencies[0].input_dependencies[0]

    planner = create_gpu_required_fusion_planner(root, dataset.context)
    assert calls == []

    decision = planner.plan_node(read, dataset.context)

    assert not isinstance(decision, GPUDecline)
    assert len(calls) == 1
    assert calls[0][1]["retried_io_errors"] == tuple(fusion_context.retried_io_errors)


def test_source_range_planning_failure_declines_with_stable_reason(
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
    dataset = _pipeline(_write_parquet(tmp_path))
    root = dataset._logical_plan.dag
    read = root.input_dependencies[0].input_dependencies[0]
    planner = create_gpu_required_fusion_planner(root, dataset.context)

    decision = planner.plan_node(read, dataset.context)

    assert isinstance(decision, GPUDecline)
    assert decision.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "Parquet range planning failed (key_statistics)" in decision.detail


def test_source_owned_parallelism_is_not_reinterpreted_by_stock_rule(
    tmp_path, fusion_context, monkeypatch
):
    plan, _ = Planner().plan(_pipeline(_write_parquet(tmp_path))._logical_plan)
    source = _planned_ops(plan)[0]
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
    assert source.input_dependencies == [source_input]


def test_retry_contract_is_carried_by_source_and_complete_region(
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
        op.gpu_operator_spec.compatibility_key.retry_policy for op in _planned_ops(plan)
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

    actor = _actor_ops(PhysicalOptimizer().optimize(plan))[0]
    assert actor.gpu_operator_spec.compatibility_key.retry_policy == policy


def test_source_bootstrap_merges_read_placement_and_rejects_conflict(
    tmp_path, fusion_context
):
    path = _write_parquet(tmp_path)
    inherited = _mgp(
        _map(_read(path, ray_remote_args={"scheduling_strategy": "SPREAD"}))
    )

    plan, _ = Planner().plan(inherited._logical_plan)

    keys = [op.gpu_operator_spec.compatibility_key for op in _planned_ops(plan)]
    assert [key.scheduling_strategy for key in keys] == ["SPREAD", None, None]
    assert dict(keys[0].actor_bootstrap)["source_kind"] == "local"
    actor = _actor_ops(PhysicalOptimizer().optimize(plan))[0]
    assert actor.gpu_operator_spec.compatibility_key.scheduling_strategy == "SPREAD"

    conflict = _mgp(
        _map(
            _read(path, ray_remote_args={"scheduling_strategy": "SPREAD"}),
            scheduling_strategy="DEFAULT",
        )
    )
    with pytest.raises(ValueError, match="CONFLICTING_REQUIREMENTS"):
        Planner().plan(conflict._logical_plan)


def test_local_identity_is_snapshotted_into_range_work(tmp_path, fusion_context):
    path = _write_parquet(tmp_path)
    plan, _ = Planner().plan(_pipeline(path)._logical_plan)
    source = _planned_ops(plan)[0]
    bundles = source.input_dependencies[0]._input_data_factory(
        fusion_context.target_max_block_size
    )
    assert bundles
    from ray.data._internal.execution.gpu.transforms.boundary import (
        decode_one_range_work,
    )

    work = decode_one_range_work((bundles[0].blocks[0].ref,))
    identity = work.files[0].identity
    descriptor = ParquetCudfReadDescriptor(
        filesystem=pa.fs.LocalFileSystem(),
        source_kind="local",
        paths=(path,),
        listed_file_sizes=(identity.size,),
        projection=("User",),
        file_schema=pq.read_schema(path),
        region=None,
        read_ray_remote_args=(),
    )
    assert source_identity(descriptor, path) == identity


def test_s3_identity_prefers_version_and_requires_etag_plus_timestamp():
    descriptor = ParquetCudfReadDescriptor(
        filesystem=object(),
        source_kind="s3",
        paths=("bucket/key",),
        listed_file_sizes=(10,),
        projection=("User",),
        file_schema=None,
        region="us-east-1",
        read_ray_remote_args=(),
    )

    class Client:
        def __init__(self, response):
            self.response = response

        def head_object(self, **kwargs):
            assert kwargs == {"Bucket": "bucket", "Key": "key"}
            return self.response

    versioned = source_identity(
        descriptor,
        "bucket/key",
        Client({"ContentLength": 10, "VersionId": "version-3"}),
    )
    assert versioned.marker == "version:version-3"
    timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)
    unversioned = source_identity(
        descriptor,
        "bucket/key",
        Client(
            {
                "ContentLength": 10,
                "VersionId": "null",
                "ETag": '"etag"',
                "LastModified": timestamp,
            }
        ),
    )
    assert unversioned.marker == f'etag:"etag";last_modified:{timestamp.isoformat()}'
    with pytest.raises(ParquetCudfPlanningError) as captured:
        source_identity(
            descriptor,
            "bucket/key",
            Client({"ContentLength": 10, "ETag": '"etag"'}),
        )
    assert captured.value.reason == "source_identity"


def test_standalone_eligible_map_lowers_without_an_mgp_anchor(tmp_path, fusion_context):
    plan, _ = Planner().plan(_map(_read(_write_parquet(tmp_path)))._logical_plan)

    planned = _planned_ops(plan)
    assert len(planned) == 1
    assert _kinds(planned[0]) == ("map_batches",)
    assert planned[0].gpu_operator_spec.input_payload == RANGE_DATA
    assert planned[0].gpu_operator_spec.output_payload == RANGE_DATA
    assert not planned[0].owns_input_parallelism
    assert any(
        isinstance(op, GPUOperator) is False
        for op in plan.dag.post_order_iter()
        if not isinstance(op, InputDataBuffer)
    )


def test_flag_off_keeps_read_and_map_batches_on_stock_path(tmp_path, fusion_context):
    dataset = _map(_read(_write_parquet(tmp_path)))
    dataset.context.enable_gpu_operator_fusion = False

    plan, _ = Planner().plan(dataset._logical_plan)

    assert _planned_ops(plan) == []
    assert _actor_ops(plan) == []
