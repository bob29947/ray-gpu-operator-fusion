"""Focused contracts for the three concrete V1 GPU lowerers."""

from types import SimpleNamespace

import pytest

from ray.data import ActorPoolStrategy, UDFProperties
from ray.data._internal.execution.gpu import (
    CompletePartitionBundles,
    Evidence,
    GPUCompositionError,
    GPUCompositionErrorReason,
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUOperatorCreationOptions,
    GPUOperatorSpec,
    GPUTransformSpec,
    KeysColocated,
    PayloadKind,
    PropertyFact,
    RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    RANGE_DATA_TERMINAL_ADAPTER_KEY,
    RANGE_WORK_ADAPTER_KEY,
    RangePartitioning,
    RangePartitioningRequired,
    RowsOwnedByRange,
    compose_gpu_operator_specs,
)
from ray.data._internal.logical.operators import (
    MapBatches,
    MapGroupPartitions,
    Read,
)
from ray.data._internal.planner.gpu_lowering.interfaces import GPULoweringContext
from ray.data._internal.planner.gpu_lowering.map_batches_lowerer import (
    GPUMapBatchesLowerer,
)
from ray.data._internal.planner.gpu_lowering.map_group_partitions_lowerer import (
    GPUMapGroupPartitionsLowerer,
)
from ray.data._internal.planner.gpu_lowering.parquet_read_lowerer import GPUReadLowerer
from ray.data._internal.planner.gpu_read_source_provider import (
    GPUReadSourceArtifacts,
    GPUReadSourceLowering,
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


def _artifacts():
    return GPUReadSourceArtifacts(
        input_data_factory=lambda _target_size: [],
        provider_artifacts=SimpleNamespace(work=("range-0", "range-1")),
    )


def _requirements():
    return GPUExecutionRequirements(
        compatibility_key=GPUExecutionCompatibilityKey(num_gpus=0)
    )


def _range_properties(*, scheme_id="range-scheme"):
    return (
        PropertyFact(
            RangePartitioning(("user",), scheme_id),
            Evidence.OPERATOR_GUARANTEE,
        ),
        PropertyFact(
            CompletePartitionBundles(),
            Evidence.OPERATOR_GUARANTEE,
        ),
    )


def _context(
    *,
    logical_node_id,
    input_properties=(),
    source_artifacts=None,
    source_provider=None,
):
    return GPULoweringContext(
        stage_id="required-v1-stage",
        logical_node_id=logical_node_id,
        group_keys=("user", "card"),
        source_provider=source_provider or object(),
        execution_requirements=_requirements(),
        target_max_block_size=1024,
        retried_io_errors=("network",),
        target_ranges=2,
        num_workers=1,
        input_properties=input_properties,
        source_artifacts=source_artifacts,
    )


def _read_spec(*, scheme_id="range-scheme"):
    return GPUOperatorSpec(
        spec_instance_id="read-node",
        transforms=(
            GPUTransformSpec(
                transform_id="read-node/0/read_parquet_range",
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
        execution_requirements=_requirements(),
        output_properties=_range_properties(scheme_id=scheme_id),
        logical_provenance=("read-node",),
    )


def test_concrete_lowerer_analysis_describes_the_exact_v1_capabilities():
    read = _read()
    preserving_map = _map(read)
    grouped = _mgp(preserving_map)
    context = DataContext.get_current()

    map_analysis = GPUMapBatchesLowerer().analyze(preserving_map, context)
    mgp_analysis = GPUMapGroupPartitionsLowerer().analyze(grouped, context)

    assert map_analysis.accepted
    assert map_analysis.preserves_partitioning
    assert map_analysis.execution_requirements is not None
    assert mgp_analysis.accepted
    assert mgp_analysis.partition_keys == ("user", "card")
    assert mgp_analysis.requested_partitions == 2

    nonpreserving = GPUMapBatchesLowerer().analyze(
        _map(read, preserves=False), context
    )
    non_cudf = GPUMapBatchesLowerer().analyze(
        _map(read, batch_format="pyarrow"), context
    )
    assert nonpreserving.accepted and not nonpreserving.preserves_partitioning
    assert not non_cudf.accepted


def test_read_lowerer_uses_one_plan_local_source_without_a_registry():
    artifacts = _artifacts()
    spec = _read_spec()

    class _PlanLocalSource:
        def __init__(self):
            self.request = None

        def lower(self, logical_op, request):
            assert type(logical_op) is Read
            self.request = request
            return GPUReadSourceLowering(
                spec=spec,
                creation_options=GPUOperatorCreationOptions(name="GPURead"),
                artifacts=artifacts,
            )

    source = _PlanLocalSource()
    lowered = GPUReadLowerer().lower(
        _read(),
        _context(logical_node_id="read-node", source_provider=source),
    )

    assert lowered.spec is spec
    assert lowered.source_artifacts is artifacts
    assert lowered.downstream_context.source_artifacts is artifacts
    assert lowered.downstream_context.input_properties == spec.output_properties
    assert source.request.logical_node_id == "read-node"
    assert source.request.group_keys == ("user", "card")


def test_preserving_map_spec_requires_and_readvertises_the_exact_range_scheme():
    logical = _map(_read())
    lowered = GPUMapBatchesLowerer().lower(
        logical,
        _context(
            logical_node_id="map-node",
            input_properties=_range_properties(),
            source_artifacts=_artifacts(),
        ),
    )

    spec = lowered.spec
    assert spec.transform_ids == (
        "map-node/0/map_batches",
        "map-node/1/validate_range_ownership",
    )
    assert spec.input_requirements == (
        RangePartitioningRequired(("user",), "range-scheme"),
        CompletePartitionBundles(),
    )
    assert spec.output_properties == _range_properties()
    obligation = RowsOwnedByRange("user", "range-scheme")
    assert spec.transforms[0].creates_obligations == (obligation,)
    assert spec.transforms[1].discharges_obligations == (obligation,)
    assert spec.native_input_payload is PayloadKind.RANGE_DATA
    assert spec.native_output_payload is PayloadKind.RANGE_DATA
    assert not spec.requires_complete_fusion
    assert spec.fusion_group_id is None
    assert not spec.uses_terminal_output_adapter


def test_mgp_spec_consumes_complete_colocated_ranges_and_advertises_no_output_fact():
    mapped = _map(_read())
    grouped = _mgp(mapped)
    lowered = GPUMapGroupPartitionsLowerer().lower(
        grouped,
        _context(
            logical_node_id="mgp-node",
            input_properties=_range_properties(),
            source_artifacts=_artifacts(),
        ),
    )

    spec = lowered.spec
    assert spec.transform_ids == (
        "mgp-node/0/sort_partition",
        "mgp-node/1/build_group_boundaries",
        "mgp-node/2/map_group_partitions",
    )
    assert spec.input_requirements == (
        KeysColocated(("user", "card")),
        CompletePartitionBundles(),
    )
    assert spec.output_properties == ()
    assert spec.native_input_payload is PayloadKind.RANGE_DATA
    assert spec.native_output_payload is PayloadKind.RANGE_DATA
    # The exact planner, not the lowerer, marks this occurrence as region egress.
    assert not spec.uses_terminal_output_adapter


def test_independent_lowerer_specs_form_the_optimizer_composition_contract():
    artifacts = _artifacts()
    read_spec = _read_spec()
    mapped = GPUMapBatchesLowerer().lower(
        _map(_read()),
        _context(
            logical_node_id="map-node",
            input_properties=read_spec.output_properties,
            source_artifacts=artifacts,
        ),
    )
    grouped = GPUMapGroupPartitionsLowerer().lower(
        _mgp(_map(_read())),
        mapped.downstream_context.for_operator(logical_node_id="mgp-node"),
    )

    # This calls the pure composition primitive directly only to validate the
    # contracts emitted by each lowerer.  Production planning leaves all three
    # specs independent for the physical optimizer.
    composed = compose_gpu_operator_specs(
        compose_gpu_operator_specs(read_spec, mapped.spec),
        grouped.spec,
    )

    assert composed.transform_ids == (
        "read-node/0/read_parquet_range",
        "map-node/0/map_batches",
        "map-node/1/validate_range_ownership",
        "mgp-node/0/sort_partition",
        "mgp-node/1/build_group_boundaries",
        "mgp-node/2/map_group_partitions",
    )
    assert composed.logical_provenance == ("read-node", "map-node", "mgp-node")
    assert composed.output_properties == ()


def test_map_spec_rejects_a_different_range_scheme_during_composition():
    map_spec = GPUMapBatchesLowerer().lower(
        _map(_read()),
        _context(
            logical_node_id="map-node",
            input_properties=_range_properties(),
            source_artifacts=_artifacts(),
        ),
    ).spec

    with pytest.raises(GPUCompositionError) as captured:
        compose_gpu_operator_specs(
            _read_spec(scheme_id="different-scheme"),
            map_spec,
        )

    assert (
        captured.value.reason
        is GPUCompositionErrorReason.INPUT_REQUIREMENTS_UNSATISFIED
    )
