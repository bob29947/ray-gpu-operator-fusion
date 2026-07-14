"""Tests for the closed V1 GPU planner shape.

The production requirement-analysis registry was intentionally removed.  These
tests cover the smaller contract that replaced it: one unshared
Read -> preserving cuDF MapBatches -> MapGroupPartitions region is lowered to
three deferred physical nodes.
"""

import pytest

from ray.data import ActorPoolStrategy, UDFProperties
from ray.data._internal.execution.gpu import (
    CompletePartitionBundles,
    Evidence,
    GPUOperatorCreationOptions,
    GPUOperatorSpec,
    GPUTransformSpec,
    PayloadKind,
    PlannedGPUOperator,
    PropertyFact,
    RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    RANGE_DATA_TERMINAL_ADAPTER_KEY,
    RANGE_WORK_ADAPTER_KEY,
    RangePartitioning,
    RangePartitioningRequired,
)
from ray.data._internal.logical.operators import (
    Limit,
    MapBatches,
    MapGroupPartitions,
    Read,
    Union,
)
from ray.data._internal.planner.gpu_plan_provider import (
    create_gpu_required_fusion_planner,
)
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)
from ray.data._internal.planner.gpu_read_source_provider import (
    GPUReadSourceArtifacts,
    GPUReadSourceLowering,
    GPUReadSourceSupport,
)
from ray.data.context import DataContext
from ray.data.datasource.datasource import Datasource


class _Source(Datasource):
    pass


class _GPUMap:
    def __call__(self, batch):
        return batch


def _partition_fn(frame, context):
    del context
    return frame


class _StubReadProvider:
    """Plan-local source double; it is not installed in any global registry."""

    provider_name = "test_v1_source"

    def __init__(self):
        self.support_calls = 0
        self.lower_calls = 0

    def supports(self, logical_op, demand):
        self.support_calls += 1
        accepted = type(logical_op) is Read and bool(demand.group_keys)
        return GPUReadSourceSupport(accepted, "logical_read" if not accepted else "")

    def lower(self, logical_op, request):
        assert type(logical_op) is Read
        self.lower_calls += 1
        properties = (
            PropertyFact(
                RangePartitioning((request.group_keys[0],), "test-range-scheme"),
                Evidence.OPERATOR_GUARANTEE,
            ),
            PropertyFact(
                CompletePartitionBundles(),
                Evidence.OPERATOR_GUARANTEE,
            ),
        )
        spec = GPUOperatorSpec(
            spec_instance_id=request.logical_node_id,
            transforms=(
                GPUTransformSpec(
                    transform_id=request.transform_id(0, "read_parquet_range"),
                    kind="read_parquet_range",
                    input_payload=PayloadKind.RANGE_WORK,
                    output_payload=PayloadKind.RANGE_DATA,
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
            output_properties=properties,
            logical_provenance=(request.logical_node_id,),
        )
        return GPUReadSourceLowering(
            spec=spec,
            creation_options=GPUOperatorCreationOptions(
                name="GPUReadParquetRange",
                min_rows_per_bundle=1,
                target_max_block_size_override=request.target_max_block_size,
            ),
            artifacts=GPUReadSourceArtifacts(
                input_data_factory=lambda _target_size: [],
                provider_artifacts="test-range-work",
            ),
        )


def _read():
    source = _Source()
    return Read(
        datasource=source,
        datasource_or_legacy_reader=source,
        parallelism=1,
    )


def _map(parent, *, preserves=True, batch_format="cudf"):
    return MapBatches(
        fn=_GPUMap,
        input_dependencies=[parent],
        batch_size=8,
        batch_format=batch_format,
        compute=ActorPoolStrategy(size=1),
        ray_remote_args={"num_cpus": 1, "num_gpus": 1},
        udf_properties=(
            UDFProperties(preserves_partitioning=True) if preserves else None
        ),
    )


def _mgp(parent):
    return MapGroupPartitions(
        key=["user", "card"],
        num_partitions=2,
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
    return (
        create_gpu_required_fusion_planner(
            root,
            context,
            source_provider=provider or _StubReadProvider(),
        ),
        context,
    )


def _planned_nodes(root):
    return [
        op for op in root.post_order_iter() if isinstance(op, PlannedGPUOperator)
    ]


def test_exact_v1_planner_emits_three_independent_required_fusion_nodes():
    read = _read()
    mapped = _map(read)
    grouped = _mgp(mapped)
    provider = _StubReadProvider()
    planner, context = _planner(grouped, provider)

    assert provider.support_calls == 0
    assert provider.lower_calls == 0

    planned = planner.try_plan(grouped, context)

    assert planned is not None
    nodes = _planned_nodes(planned.root)
    assert len(nodes) == 3
    read_node, map_node, mgp_node = nodes
    assert read_node.input_dependencies == [next(iter(planned.op_map))]
    assert map_node.input_dependencies == [read_node]
    assert mgp_node.input_dependencies == [map_node]
    assert [
        tuple(transform.kind for transform in node.gpu_operator_spec.transforms)
        for node in nodes
    ] == [
        ("read_parquet_range",),
        ("map_batches", "validate_range_ownership"),
        ("sort_partition", "build_group_boundaries", "map_group_partitions"),
    ]

    specs = [node.gpu_operator_spec for node in nodes]
    assert all(spec.requires_complete_fusion for spec in specs)
    assert len({spec.fusion_group_id for spec in specs}) == 1
    required_provenance = tuple(spec.logical_provenance[0] for spec in specs)
    assert all(
        spec.required_logical_provenance == required_provenance for spec in specs
    )
    assert [spec.owns_input_parallelism for spec in specs] == [True, False, False]
    assert [spec.uses_terminal_output_adapter for spec in specs] == [
        False,
        False,
        True,
    ]
    assert map_node.gpu_operator_spec.input_requirements == (
        RangePartitioningRequired(("user",), "test-range-scheme"),
        CompletePartitionBundles(),
    )
    assert mgp_node.gpu_operator_spec.output_properties == ()
    assert planned.op_map[read_node] is read
    assert planned.op_map[map_node] is mapped
    assert planned.op_map[mgp_node] is grouped
    assert provider.support_calls == 1
    assert provider.lower_calls == 1


@pytest.mark.parametrize(
    "build",
    [
        lambda: _mgp(_read()),
        lambda: _mgp(_map(_map(_read()))),
        lambda: _mgp(_mgp(_map(_read()))),
    ],
    ids=("missing-map", "repeated-map", "repeated-mgp"),
)
def test_exact_v1_shape_rejects_missing_or_repeated_operators(build):
    root = build()
    planner, context = _planner(root)

    with pytest.raises(GPUPlanningError) as captured:
        planner.try_plan(root, context)

    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "exactly Read -> one cuDF MapBatches" in str(captured.value)


@pytest.mark.parametrize(
    ("mapped", "detail"),
    [
        (lambda read: _map(read, preserves=False), "preserves_partitioning"),
        (lambda read: _map(read, batch_format="pyarrow"), "batch_format='cudf'"),
    ],
    ids=("not-preserving", "not-cudf"),
)
def test_exact_v1_rejects_an_ineligible_map(mapped, detail):
    grouped = _mgp(mapped(_read()))
    planner, context = _planner(grouped)

    with pytest.raises(GPUPlanningError) as captured:
        planner.try_plan(grouped, context)

    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert detail in str(captured.value)


def test_exact_v1_rejects_a_shared_internal_map():
    mapped = _map(_read())
    left = _mgp(mapped)
    right = _mgp(mapped)
    root = Union([left, right])
    planner, context = _planner(root)

    with pytest.raises(GPUPlanningError) as captured:
        planner.try_plan(left, context)

    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "shared by multiple logical consumers" in str(captured.value)


def test_exact_v1_region_may_feed_one_stock_consumer():
    grouped = _mgp(_map(_read()))
    root = Limit(1, input_dependencies=[grouped])
    planner, context = _planner(root)

    planned = planner.try_plan(grouped, context)

    assert planned is not None
    assert len(_planned_nodes(planned.root)) == 3


def test_non_mgp_operator_is_not_claimed():
    mapped = _map(_read())
    planner, context = _planner(mapped)

    assert planner.try_plan(mapped, context) is None
