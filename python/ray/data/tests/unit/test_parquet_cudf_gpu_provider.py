import copy
from dataclasses import replace
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray.data
from ray.data import ActorPoolStrategy, UDFProperties
from ray.data._internal.datasource import parquet_datasource
from ray.data._internal.execution.gpu import GPUOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.operators import MapBatches, Read
from ray.data._internal.logical.optimizers import PhysicalOptimizer
from ray.data._internal.planner import gpu_plan_provider
from ray.data._internal.planner.gpu_plan_provider import (
    GPUPlanFragment,
    GPUProviderAccepted,
    GPUProviderNotApplicable,
    GPUProviderRejected,
)
from ray.data._internal.planner.gpu_plan_validation import (
    validate_gpu_plan_fragment,
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
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": 4, "GPU": 1})
    monkeypatch.setattr(ray, "available_resources", lambda: {"CPU": 4, "GPU": 1})
    monkeypatch.setattr(ray.util, "get_current_placement_group", lambda: None)
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
    context.max_errored_blocks = 0
    context.retried_map_errors = False
    context.actor_task_retry_on_errors = False
    context.actor_init_retry_on_errors = False
    yield context
    DataContext._set_current(original)


def _pipeline(path):
    return (
        ray.data.read_parquet(path, columns=["User", "Card", "Value"])
        .map_batches(
            _Tokenizer,
            batch_size=2,
            batch_format="cudf",
            compute=ActorPoolStrategy(size=1),
            num_cpus=1,
            num_gpus=1,
            udf_properties=UDFProperties(preserves_partitioning=True),
        )
        .groupby(["User", "Card"], num_partitions=2)
        .map_group_partitions(_partition_fn)
    )


def _write_parquet(tmp_path):
    path = tmp_path / "input.parquet"
    table = pa.table(
        {
            "User": pa.array([1, 1, 2, 3], type=pa.int64()),
            "Card": pa.array([10, 10, 20, 30], type=pa.int64()),
            "Value": pa.array([1, 2, 3, 4], type=pa.int64()),
        }
    )
    pq.write_table(table, path, row_group_size=2, write_statistics=True)
    return str(path)


def test_provider_lowers_three_nodes_then_physical_rule_fuses(
    tmp_path, fusion_context
):
    dataset = _pipeline(_write_parquet(tmp_path))
    grouped_logical = dataset._logical_plan.dag
    mapped_logical = grouped_logical.input_dependencies[0]
    read_logical = mapped_logical.input_dependencies[0]
    plan, _ = Planner().plan(dataset._logical_plan)

    pre_ops = list(plan.dag.post_order_iter())
    gpu_ops = [op for op in pre_ops if isinstance(op, GPUOperator)]
    assert len(gpu_ops) == 3
    assert isinstance(pre_ops[0], InputDataBuffer)
    assert [op.gpu_operator_spec.transform_ids for op in gpu_ops] == [
        ("read_parquet_range",),
        ("map_batches", "validate_range_ownership"),
        ("sort_partition", "build_group_boundaries", "map_group_partitions"),
    ]
    assert set(plan.op_map) == set(pre_ops)
    assert plan.op_map[pre_ops[0]] is read_logical
    assert plan.op_map[gpu_ops[0]] is read_logical
    assert plan.op_map[gpu_ops[1]] is mapped_logical
    assert plan.op_map[gpu_ops[2]] is grouped_logical
    assert pre_ops[0]._logical_operators == [read_logical]
    assert gpu_ops[0]._logical_operators == [read_logical]
    assert gpu_ops[1]._logical_operators == [mapped_logical]
    assert gpu_ops[2]._logical_operators == [grouped_logical]

    optimized = PhysicalOptimizer().optimize(plan)
    final_gpu_ops = [
        op for op in optimized.dag.post_order_iter() if isinstance(op, GPUOperator)
    ]
    assert len(final_gpu_ops) == 1
    assert final_gpu_ops[0].gpu_operator_spec.transform_ids == (
        "read_parquet_range",
        "map_batches",
        "validate_range_ownership",
        "sort_partition",
        "build_group_boundaries",
        "map_group_partitions",
    )
    assert isinstance(final_gpu_ops[0].input_dependencies[0], InputDataBuffer)
    final_ops = list(optimized.dag.post_order_iter())
    assert set(optimized.op_map) == set(final_ops)
    assert optimized.op_map[final_ops[0]] is read_logical
    assert optimized.op_map[final_gpu_ops[0]] is grouped_logical
    assert final_gpu_ops[0]._logical_operators == [
        read_logical,
        mapped_logical,
        grouped_logical,
    ]


def test_provider_claims_before_stock_child_planning(
    tmp_path, fusion_context, monkeypatch
):
    def fail_if_stock_child_is_planned(*args, **kwargs):
        raise AssertionError("provider built and discarded a stock child")

    monkeypatch.setitem(Planner._DEFAULT_PLAN_FNS, Read, fail_if_stock_child_is_planned)
    monkeypatch.setitem(
        Planner._DEFAULT_PLAN_FNS, MapBatches, fail_if_stock_child_is_planned
    )
    dataset = _pipeline(_write_parquet(tmp_path))

    plan, _ = Planner().plan(dataset._logical_plan)

    assert (
        len(
            [
                op
                for op in plan.dag.post_order_iter()
                if isinstance(op, GPUOperator)
            ]
        )
        == 3
    )


def test_provider_fragment_composes_under_stock_parent(tmp_path, fusion_context):
    dataset = _pipeline(_write_parquet(tmp_path)).limit(1)
    parent_logical = dataset._logical_plan.dag
    grouped_logical = parent_logical.input_dependencies[0]

    plan, _ = Planner().plan(dataset._logical_plan)

    ops = list(plan.dag.post_order_iter())
    assert set(plan.op_map) == set(ops)
    assert plan.op_map[plan.dag] is parent_logical
    assert plan.dag._logical_operators == [parent_logical]
    grouped_gpu = next(
        op
        for op in ops
        if isinstance(op, GPUOperator)
        and op.gpu_operator_spec.logical_provenance == ("MapGroupPartitions",)
    )
    assert plan.op_map[grouped_gpu] is grouped_logical
    assert grouped_gpu._logical_operators == [grouped_logical]


def test_shared_anchor_is_rejected_before_duplicate_lowering(
    tmp_path, fusion_context
):
    dataset = _pipeline(_write_parquet(tmp_path))
    branched = dataset.union(dataset)

    with pytest.raises(ValueError, match="shared by multiple branches"):
        Planner().plan(branched._logical_plan)


def test_fragment_validator_rejects_incomplete_or_inconsistent_contracts(
    tmp_path, fusion_context
):
    dataset = _pipeline(_write_parquet(tmp_path))
    grouped_logical = dataset._logical_plan.dag
    mapped_logical = grouped_logical.input_dependencies[0]
    read_logical = mapped_logical.input_dependencies[0]
    plan, _ = Planner().plan(dataset._logical_plan)
    operators = tuple(plan.dag.post_order_iter())
    region_id = next(
        op.gpu_operator_spec.region_id
        for op in operators
        if isinstance(op, GPUOperator)
    )
    fragment = GPUPlanFragment(
        root=plan.dag,
        new_operators=operators,
        consumed_logical_operators=(read_logical, mapped_logical, grouped_logical),
        op_map=plan.op_map,
        region_id=region_id,
    )
    validate_gpu_plan_fragment(fragment)

    with pytest.raises(ValueError, match="exactly cover"):
        validate_gpu_plan_fragment(
            replace(fragment, new_operators=operators[1:])
        )
    with pytest.raises(ValueError, match="duplicate physical"):
        validate_gpu_plan_fragment(
            replace(fragment, new_operators=operators + (operators[-1],))
        )
    parent_logical = dataset.limit(1)._logical_plan.dag
    with pytest.raises(ValueError, match="every and only consumed"):
        validate_gpu_plan_fragment(
            replace(
                fragment,
                consumed_logical_operators=(
                    read_logical,
                    mapped_logical,
                    grouped_logical,
                    parent_logical,
                ),
            )
        )

    mapped_gpu = next(
        op
        for op in operators
        if isinstance(op, GPUOperator)
        and op.gpu_operator_spec.logical_provenance == ("MapBatches",)
    )
    mapped_gpu.set_logical_operators(grouped_logical)
    with pytest.raises(ValueError, match="disagrees"):
        validate_gpu_plan_fragment(fragment)


def test_provider_dispatch_continues_not_applicable_and_stops_on_claim(
    tmp_path, fusion_context, monkeypatch
):
    anchor = _pipeline(_write_parquet(tmp_path))._logical_plan.dag
    calls = []

    class Provider:
        def __init__(self, result, name):
            self.result = result
            self.name = name

        def plan(self, logical_op, data_context, consumer_counts):
            assert logical_op is anchor
            assert data_context is fusion_context
            assert consumer_counts == {}
            calls.append(self.name)
            return self.result

    sentinel = object()
    providers = (
        Provider(GPUProviderNotApplicable(), "first"),
        Provider(GPUProviderAccepted(sentinel), "second"),
    )
    monkeypatch.setitem(
        gpu_plan_provider._GPU_PROVIDER_LOADERS_BY_EXACT_ROOT,
        type(anchor),
        tuple((lambda provider=provider: provider) for provider in providers),
    )
    assert (
        gpu_plan_provider.plan_with_gpu_provider(anchor, fusion_context, {})
        is sentinel
    )
    assert calls == ["first", "second"]

    calls.clear()
    providers = (
        Provider(GPUProviderRejected("unsafe request"), "rejected"),
        Provider(GPUProviderAccepted(sentinel), "must-not-run"),
    )
    monkeypatch.setitem(
        gpu_plan_provider._GPU_PROVIDER_LOADERS_BY_EXACT_ROOT,
        type(anchor),
        tuple((lambda provider=provider: provider) for provider in providers),
    )
    with pytest.raises(ValueError, match="unsafe request"):
        gpu_plan_provider.plan_with_gpu_provider(anchor, fusion_context, {})
    assert calls == ["rejected"]


def test_flag_off_does_not_construct_gpu_provider(
    tmp_path, fusion_context, monkeypatch
):
    dataset = _pipeline(_write_parquet(tmp_path))
    dataset.context.enable_gpu_operator_fusion = False

    def fail_loader():
        raise AssertionError("flag-off planning constructed a GPU provider")

    monkeypatch.setitem(
        gpu_plan_provider._GPU_PROVIDER_LOADERS_BY_EXACT_ROOT,
        type(dataset._logical_plan.dag),
        (fail_loader,),
    )
    with pytest.raises(ValueError, match="enable_gpu_operator_fusion=True"):
        Planner().plan(dataset._logical_plan)


def test_map_group_partitions_requires_flag(tmp_path, fusion_context):
    dataset = _pipeline(_write_parquet(tmp_path))
    dataset.context.enable_gpu_operator_fusion = False
    with pytest.raises(ValueError, match="enable_gpu_operator_fusion=True"):
        Planner().plan(dataset._logical_plan)


def test_missing_partition_assertion_is_rejected(tmp_path, fusion_context):
    dataset = (
        ray.data.read_parquet(
            _write_parquet(tmp_path), columns=["User", "Card", "Value"]
        )
        .map_batches(
            _Tokenizer,
            batch_size=2,
            batch_format="cudf",
            compute=ActorPoolStrategy(size=1),
            num_gpus=1,
        )
        .groupby(["User", "Card"], num_partitions=2)
        .map_group_partitions(_partition_fn)
    )
    with pytest.raises(ValueError, match="preserves_partitioning=True"):
        Planner().plan(dataset._logical_plan)


def test_standalone_cudf_map_batches_stays_stock(tmp_path, fusion_context):
    dataset = ray.data.read_parquet(_write_parquet(tmp_path)).map_batches(
        _Tokenizer,
        batch_size=2,
        batch_format="cudf",
        compute=ActorPoolStrategy(size=1),
        num_gpus=1,
    )
    plan, _ = Planner().plan(dataset._logical_plan)
    assert not any(isinstance(op, GPUOperator) for op in plan.dag.post_order_iter())
