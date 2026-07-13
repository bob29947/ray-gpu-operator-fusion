from dataclasses import dataclass
from types import SimpleNamespace

from ray.data import ActorPoolStrategy, UDFProperties
from ray.data._internal.execution.gpu import (
    CompletePartitionBundles,
    Evidence,
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUOperatorCreationOptions,
    GPUOperatorSpec,
    GPUTransformSpec,
    KeysColocated,
    PartitioningProperty,
    PartitioningValidation,
    PayloadKind,
    PropertyFact,
    RangePartitioning,
    UnknownPartitioning,
    ValidationObligation,
    compile_gpu_region,
    register_partitioning_preservation,
    register_property_satisfaction,
    unregister_partitioning_preservation,
    unregister_property_satisfaction,
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
from ray.data._internal.planner.gpu_requirement_analysis import (
    GPUPropertyState,
    create_default_property_transfer_registry,
)
from ray.data.datasource.datasource import Datasource


class _Source(Datasource):
    pass


class _GPUMap:
    def __call__(self, batch):
        return batch


@dataclass(frozen=True)
class _HashPartitioning(PartitioningProperty):
    keys: tuple[str, ...]
    scheme_id: str


@dataclass(frozen=True)
class _RowsOwnedByHash(ValidationObligation):
    scheme_id: str


def _hash_partitioning_validations(value):
    assert isinstance(value, _HashPartitioning)
    obligation = _RowsOwnedByHash(value.scheme_id)
    return (
        PartitioningValidation(
            kind="validate_hash_ownership",
            config=(value.keys, value.scheme_id),
            obligations=(obligation,),
        ),
    )


def _hash_satisfies_colocation(value, requirement):
    assert isinstance(value, _HashPartitioning)
    assert isinstance(requirement, KeysColocated)
    return requirement.keys[: len(value.keys)] == value.keys


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


def _map(parent, *, preserves):
    return MapBatches(
        fn=_GPUMap,
        input_dependencies=[parent],
        batch_size=8,
        batch_format="cudf",
        compute=ActorPoolStrategy(size=1),
        ray_remote_args={"num_cpus": 1, "num_gpus": 1},
        udf_properties=(
            UDFProperties(preserves_partitioning=True) if preserves else None
        ),
    )


def _mgp(parent, *, preserves):
    return MapGroupPartitions(
        key=["user", "card"],
        num_partitions=2,
        fn=_partition_fn,
        udf_properties=(
            UDFProperties(preserves_partitioning=True) if preserves else None
        ),
        input_dependencies=[parent],
    )


def _artifacts():
    work = ("range-0", "range-1")
    return GPUReadSourceArtifacts(
        input_data_factory=lambda _target_size: [],
        provider_artifacts=SimpleNamespace(work=work),
    )


def _context(
    *,
    logical_node_id,
    group_keys=("user", "card"),
    input_properties=(),
    source_artifacts=None,
    source_provider=None,
):
    return GPULoweringContext(
        stage_id="stage-1",
        logical_node_id=logical_node_id,
        group_keys=group_keys,
        source_provider=source_provider or object(),
        execution_requirements=GPUExecutionRequirements(
            compatibility_key=GPUExecutionCompatibilityKey(num_gpus=0)
        ),
        target_max_block_size=1024,
        retried_io_errors=("network",),
        target_ranges=2,
        num_workers=1,
        input_properties=input_properties,
        source_artifacts=source_artifacts,
    )


def _range_properties():
    return (
        PropertyFact(
            RangePartitioning(("user",), "range-scheme"),
            Evidence.OPERATOR_GUARANTEE,
        ),
        PropertyFact(
            CompletePartitionBundles(),
            Evidence.OPERATOR_GUARANTEE,
        ),
    )


def test_read_lowerer_is_source_provider_dispatch_not_parquet_logic():
    artifacts = _artifacts()
    requirements = GPUExecutionRequirements(
        compatibility_key=GPUExecutionCompatibilityKey(num_gpus=0)
    )
    spec = GPUOperatorSpec(
        spec_instance_id="source-node",
        transforms=(
            GPUTransformSpec(
                "source-node/0/custom_source",
                "custom_source",
                PayloadKind.RANGE_WORK,
                PayloadKind.RANGE_DATA,
            ),
        ),
        external_input_payload=PayloadKind.RAY_BLOCK_STREAM,
        external_output_payload=PayloadKind.RAY_BLOCK_STREAM,
        native_input_payload=PayloadKind.RANGE_WORK,
        native_output_payload=PayloadKind.RANGE_DATA,
        input_adapter_key="custom-source-import-v1",
        output_adapter_key="custom-source-export-v1",
        terminal_output_adapter_key="custom-source-terminal-v1",
        execution_requirements=requirements,
        output_properties=_range_properties(),
    )

    class _Provider:
        def __init__(self):
            self.request = None

        def lower(self, logical_op, request):
            assert type(logical_op) is Read
            self.request = request
            return GPUReadSourceLowering(
                spec,
                GPUOperatorCreationOptions(name="CustomGPURead"),
                artifacts,
            )

    provider = _Provider()
    context = _context(
        logical_node_id="source-node",
        group_keys=("user",),
        source_provider=provider,
    )
    lowered = GPUReadLowerer().lower(_read(), context)

    assert lowered.spec is spec
    assert lowered.creation_options.name == "CustomGPURead"
    assert lowered.source_artifacts is artifacts
    assert lowered.downstream_context.source_artifacts is artifacts
    assert lowered.downstream_context.input_properties == spec.output_properties
    assert provider.request.logical_node_id == "source-node"


def test_preserving_map_has_stable_ids_and_explicitly_readvertises_facts():
    read = _read()
    logical = _map(read, preserves=True)
    context = _context(
        logical_node_id="map-node-7",
        input_properties=_range_properties(),
        source_artifacts=_artifacts(),
    )

    lowered = GPUMapBatchesLowerer().lower(logical, context)

    assert lowered.spec.transform_ids == (
        "map-node-7/0/map_batches",
        "map-node-7/1/validate_range_ownership",
    )
    assert lowered.spec.output_properties == lowered.downstream_context.input_properties
    assert isinstance(lowered.spec.output_properties[0].value, RangePartitioning)
    assert isinstance(lowered.spec.output_properties[1].value, CompletePartitionBundles)
    assert lowered.spec.external_input_payload is PayloadKind.RAY_BLOCK_STREAM
    assert lowered.spec.external_output_payload is PayloadKind.RAY_BLOCK_STREAM
    assert lowered.spec.native_input_payload is PayloadKind.RANGE_DATA
    assert lowered.spec.native_output_payload is PayloadKind.RANGE_DATA


def test_partitioning_invalidation_cannot_be_recreated_by_later_preservation():
    read = _read()
    artifacts = _artifacts()
    invalidating_op = _map(read, preserves=False)
    invalidating = GPUMapBatchesLowerer().lower(
        invalidating_op,
        _context(
            logical_node_id="map-invalidates",
            input_properties=_range_properties(),
            source_artifacts=artifacts,
        ),
    )
    analyzed = create_default_property_transfer_registry().transfer(
        invalidating_op,
        GPUPropertyState(_range_properties(), partition_count=2),
    )
    assert invalidating.spec.output_properties == analyzed.facts
    assert isinstance(
        invalidating.spec.output_properties[1].value,
        CompletePartitionBundles,
    )
    preserving_context = invalidating.downstream_context.for_operator(
        logical_node_id="map-cannot-recreate"
    )
    later = GPUMapBatchesLowerer().lower(
        _map(read, preserves=True),
        preserving_context,
    )

    assert later.spec.transform_ids == ("map-cannot-recreate/0/map_batches",)
    assert len(later.spec.output_properties) == 2
    assert isinstance(later.spec.output_properties[0].value, UnknownPartitioning)
    assert isinstance(
        later.spec.output_properties[1].value,
        CompletePartitionBundles,
    )


def test_mgp_is_native_producer_and_preservation_validates_only_range_key():
    read = _read()
    logical = _mgp(read, preserves=True)
    context = _context(
        logical_node_id="mgp-node-3",
        input_properties=_range_properties(),
        source_artifacts=_artifacts(),
    )

    lowered = GPUMapGroupPartitionsLowerer().lower(logical, context)

    assert lowered.spec.transform_ids == (
        "mgp-node-3/0/sort_partition",
        "mgp-node-3/1/build_group_boundaries",
        "mgp-node-3/2/map_group_partitions",
        "mgp-node-3/3/validate_range_ownership",
    )
    assert lowered.spec.native_output_payload is PayloadKind.RANGE_DATA
    assert lowered.spec.external_output_payload is PayloadKind.RAY_BLOCK_STREAM
    assert lowered.spec.input_requirements == (
        KeysColocated(("user", "card")),
        CompletePartitionBundles(),
    )
    validation_config = lowered.spec.transforms[-1].config
    assert validation_config.range_key == "user"
    assert validation_config.group_keys == ("user",)
    assert isinstance(lowered.spec.output_properties[0].value, RangePartitioning)
    assert isinstance(lowered.spec.output_properties[1].value, CompletePartitionBundles)


def test_lowered_map_and_mgp_compile_with_same_structural_contract():
    read = _read()
    artifacts = _artifacts()
    mapped = GPUMapBatchesLowerer().lower(
        _map(read, preserves=True),
        _context(
            logical_node_id="map-node",
            input_properties=_range_properties(),
            source_artifacts=artifacts,
        ),
    )
    grouped = GPUMapGroupPartitionsLowerer().lower(
        _mgp(read, preserves=False),
        mapped.downstream_context.for_operator(logical_node_id="mgp-node"),
    )

    compiled = compile_gpu_region((mapped.spec, grouped.spec))
    assert (
        compiled.transform_ids == mapped.spec.transform_ids + grouped.spec.transform_ids
    )
    assert compiled.output_properties == grouped.spec.output_properties
    assert isinstance(compiled.output_properties[0].value, UnknownPartitioning)


def test_registered_partitioning_type_is_readvertised_without_lowerer_changes():
    read = _read()
    mapped_op = _map(read, preserves=True)
    grouped_op = _mgp(mapped_op, preserves=True)
    artifacts = _artifacts()
    input_properties = (
        PropertyFact(
            _HashPartitioning(("user",), "hash-scheme"),
            Evidence.OPERATOR_GUARANTEE,
        ),
        PropertyFact(
            CompletePartitionBundles(),
            Evidence.OPERATOR_GUARANTEE,
        ),
    )
    transfers = create_default_property_transfer_registry()

    register_partitioning_preservation(
        _HashPartitioning,
        _hash_partitioning_validations,
    )
    register_property_satisfaction(
        _HashPartitioning,
        KeysColocated,
        _hash_satisfies_colocation,
    )
    try:
        input_state = GPUPropertyState(input_properties, partition_count=2)
        analyzed_map = transfers.transfer(mapped_op, input_state)
        lowered_map = GPUMapBatchesLowerer().lower(
            mapped_op,
            _context(
                logical_node_id="map-hash",
                input_properties=input_properties,
                source_artifacts=artifacts,
            ),
        )
        assert lowered_map.spec.output_properties == analyzed_map.facts
        assert isinstance(
            lowered_map.spec.output_properties[0].value,
            _HashPartitioning,
        )
        assert lowered_map.spec.transform_ids[-1] == (
            "map-hash/1/validate_hash_ownership"
        )

        analyzed_grouped = transfers.transfer(grouped_op, analyzed_map)
        lowered_grouped = GPUMapGroupPartitionsLowerer().lower(
            grouped_op,
            lowered_map.downstream_context.for_operator(
                logical_node_id="mgp-hash",
            ),
        )
        assert lowered_grouped.spec.output_properties == analyzed_grouped.facts
        assert isinstance(
            lowered_grouped.spec.output_properties[0].value,
            _HashPartitioning,
        )
        assert lowered_grouped.spec.transform_ids[-1] == (
            "mgp-hash/3/validate_hash_ownership"
        )

        repeated_grouped_op = _mgp(grouped_op, preserves=True)
        analyzed_repeated = transfers.transfer(
            repeated_grouped_op,
            analyzed_grouped,
        )
        lowered_repeated = GPUMapGroupPartitionsLowerer().lower(
            repeated_grouped_op,
            lowered_grouped.downstream_context.for_operator(
                logical_node_id="mgp-hash-repeated",
            ),
        )
        compiled = compile_gpu_region(
            (lowered_map.spec, lowered_grouped.spec, lowered_repeated.spec)
        )
        assert compiled.output_properties == analyzed_repeated.facts
        assert isinstance(compiled.output_properties[0].value, _HashPartitioning)
        assert compiled.transform_ids[-1] == (
            "mgp-hash-repeated/3/validate_hash_ownership"
        )
    finally:
        unregister_property_satisfaction(
            _HashPartitioning,
            KeysColocated,
            expected=_hash_satisfies_colocation,
        )
        unregister_partitioning_preservation(
            _HashPartitioning,
            expected=_hash_partitioning_validations,
        )
