from dataclasses import dataclass, replace

import pytest
from ray import cloudpickle

from ray.data._internal.execution.gpu import (
    CompletePartitionBundles,
    Evidence,
    GPUBoundaryAdapter,
    GPUBoundaryDirection,
    GPUCompositionError,
    GPUCompositionErrorReason,
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUOperatorSpec,
    GPUStageRetryPolicy,
    GPUTransformSpec,
    GroupedRangeData,
    KeysColocated,
    PartitionIdentity,
    PayloadKind,
    PropertyFact,
    RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    RANGE_DATA_TERMINAL_ADAPTER_KEY,
    RANGE_WORK_ADAPTER_KEY,
    RangeData,
    RangePartitioning,
    RangePartitioningRequired,
    RangePartitionIdentity,
    RowsOwnedByRange,
    UnknownPartitioning,
    compile_gpu_region,
    compose_gpu_operator_specs,
    get_gpu_boundary_adapter,
    property_satisfies,
    register_gpu_boundary_adapter,
    unregister_gpu_boundary_adapter,
)
from ray.data._internal.execution.gpu.runtime import (
    GPURegionBlockTransform,
    GPURegionWorker,
)


class _ExplodingEq:
    def __eq__(self, other):
        raise AssertionError("opaque UDF equality must not be called")

    def __call__(self, value, _context):
        return value


def _spec(
    transform_id,
    native_input,
    native_output,
    *,
    key=None,
    requirements=(),
    properties=(),
    input_adapter=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    output_adapter=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    terminal_output_adapter=RANGE_DATA_TERMINAL_ADAPTER_KEY,
    creates=(),
    discharges=(),
    requires_resolved=False,
    config=None,
    provenance=None,
    fusion_group_id=None,
    requires_complete_fusion=False,
    required_provenance=(),
    owns_input_parallelism=False,
    uses_terminal_output_adapter=False,
):
    if key is None:
        key = GPUExecutionCompatibilityKey(num_gpus=0)
    return GPUOperatorSpec(
        spec_instance_id=transform_id,
        transforms=(
            GPUTransformSpec(
                transform_id=transform_id,
                kind=transform_id,
                input_payload=native_input,
                output_payload=native_output,
                config=config or _ExplodingEq(),
                creates_obligations=creates,
                discharges_obligations=discharges,
                requires_resolved_obligations=requires_resolved,
            ),
        ),
        external_input_payload=PayloadKind.RAY_BLOCK_STREAM,
        external_output_payload=PayloadKind.RAY_BLOCK_STREAM,
        native_input_payload=native_input,
        native_output_payload=native_output,
        input_adapter_key=input_adapter,
        output_adapter_key=output_adapter,
        terminal_output_adapter_key=terminal_output_adapter,
        execution_requirements=GPUExecutionRequirements(compatibility_key=key),
        input_requirements=requirements,
        output_properties=properties,
        logical_provenance=(transform_id,) if provenance is None else provenance,
        fusion_group_id=fusion_group_id,
        requires_complete_fusion=requires_complete_fusion,
        required_logical_provenance=required_provenance,
        owns_input_parallelism=owns_input_parallelism,
        uses_terminal_output_adapter=uses_terminal_output_adapter,
    )


def test_compose_is_pairwise_pure_associative_and_serializable():
    range_fact = PropertyFact(
        RangePartitioning(("User",), "scheme"), Evidence.OPERATOR_GUARANTEE
    )
    complete_fact = PropertyFact(
        CompletePartitionBundles(), Evidence.OPERATOR_GUARANTEE
    )
    a = _spec(
        "read/0/read",
        PayloadKind.RANGE_WORK,
        PayloadKind.RANGE_DATA,
        properties=(range_fact, complete_fact),
        input_adapter=RANGE_WORK_ADAPTER_KEY,
    )
    b = _spec(
        "map/0/map",
        PayloadKind.RANGE_DATA,
        PayloadKind.RANGE_DATA,
        properties=(range_fact, complete_fact),
    )
    c = _spec(
        "mgp/0/group",
        PayloadKind.RANGE_DATA,
        PayloadKind.RANGE_DATA,
        requirements=(
            KeysColocated(("User", "Card")),
            CompletePartitionBundles(),
        ),
    )
    before = (a.canonical_contract(), b.canonical_contract(), c.canonical_contract())

    left = compose_gpu_operator_specs(compose_gpu_operator_specs(a, b), c)
    right = compose_gpu_operator_specs(a, compose_gpu_operator_specs(b, c))
    folded = compile_gpu_region((a, b, c))

    assert left.canonical_contract() == right.canonical_contract()
    assert left.canonical_contract() == folded.canonical_contract()
    assert folded.spec_instance_id == ("gpu-stage[read/0/read,map/0/map,mgp/0/group]")
    assert folded.external_input_payload is PayloadKind.RAY_BLOCK_STREAM
    assert folded.external_output_payload is PayloadKind.RAY_BLOCK_STREAM
    assert folded.native_input_payload is PayloadKind.RANGE_WORK
    assert folded.native_output_payload is PayloadKind.RANGE_DATA
    assert folded.input_adapter_key == RANGE_WORK_ADAPTER_KEY
    assert folded.output_adapter_key == RANGE_DATA_BOUNDARY_ADAPTER_KEY
    assert folded.terminal_output_adapter_key == RANGE_DATA_TERMINAL_ADAPTER_KEY
    assert folded.active_output_adapter_key == RANGE_DATA_BOUNDARY_ADAPTER_KEY
    assert before == (
        a.canonical_contract(),
        b.canonical_contract(),
        c.canonical_contract(),
    )
    restored = cloudpickle.loads(cloudpickle.dumps(folded))
    assert restored.canonical_contract() == folded.canonical_contract()


@pytest.mark.parametrize(
    ("upstream", "downstream", "reason"),
    [
        (
            _spec(
                "a",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
            ),
            _spec(
                "b",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
                key=GPUExecutionCompatibilityKey(runtime_id="different", num_gpus=0),
            ),
            GPUCompositionErrorReason.EXECUTION_INCOMPATIBLE,
        ),
        (
            _spec(
                "a",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
            ),
            _spec(
                "b",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
                requirements=(KeysColocated(("key",)),),
            ),
            GPUCompositionErrorReason.INPUT_REQUIREMENTS_UNSATISFIED,
        ),
        (
            _spec("a", PayloadKind.CUDF_BATCH_STREAM, PayloadKind.CUDF_PARTITION),
            _spec(
                "b",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
            ),
            GPUCompositionErrorReason.PAYLOAD_INCOMPATIBLE,
        ),
        (
            _spec(
                "a",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
                output_adapter="different-adapter-v1",
            ),
            _spec(
                "b",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
            ),
            GPUCompositionErrorReason.ADAPTER_INCOMPATIBLE,
        ),
        (
            _spec(
                "a",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
                creates=(RowsOwnedByRange("User", "scheme"),),
            ),
            _spec(
                "b",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
                requires_resolved=True,
            ),
            GPUCompositionErrorReason.UNRESOLVED_VALIDATION_OBLIGATION,
        ),
        (
            _spec(
                "duplicate",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
            ),
            _spec(
                "duplicate",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
            ),
            GPUCompositionErrorReason.DUPLICATE_TRANSFORM_ID,
        ),
        (
            _spec(
                "required-a",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
                fusion_group_id="group-a",
                requires_complete_fusion=True,
                required_provenance=("required-a", "required-b"),
            ),
            _spec(
                "required-b",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
                fusion_group_id="group-b",
                requires_complete_fusion=True,
                required_provenance=("required-a", "required-b"),
            ),
            GPUCompositionErrorReason.FUSION_GROUP_INCOMPATIBLE,
        ),
        (
            _spec(
                "required-a",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
                fusion_group_id="group",
                requires_complete_fusion=True,
                required_provenance=("required-a", "required-b"),
            ),
            _spec(
                "required-b",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
                fusion_group_id="group",
                requires_complete_fusion=True,
                required_provenance=("required-a", "different"),
            ),
            GPUCompositionErrorReason.FUSION_GROUP_INCOMPATIBLE,
        ),
    ],
)
def test_compose_stable_error_codes(upstream, downstream, reason):
    with pytest.raises(GPUCompositionError) as exc_info:
        compose_gpu_operator_specs(upstream, downstream)
    assert exc_info.value.reason is reason


def test_terminal_egress_cannot_be_composed_as_an_upstream_boundary():
    terminal = _spec(
        "terminal",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
        uses_terminal_output_adapter=True,
    )
    adjacent = _spec(
        "adjacent",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
    )

    with pytest.raises(GPUCompositionError) as captured:
        compose_gpu_operator_specs(terminal, adjacent)

    assert captured.value.reason is GPUCompositionErrorReason.ADAPTER_INCOMPATIBLE
    assert "terminal GPU egress" in str(captured.value)


def test_retry_policy_is_canonical_and_part_of_execution_compatibility():
    disabled_with_irrelevant_limits = GPUStageRetryPolicy(
        actor_init_retry_on_errors=False,
        actor_init_max_retries=99,
        retried_map_errors=False,
        max_map_retries=99,
        retried_io_errors=["slow down"],
    )
    disabled_canonical = GPUStageRetryPolicy(
        actor_init_retry_on_errors=False,
        actor_init_max_retries=0,
        retried_map_errors=False,
        max_map_retries=0,
        retried_io_errors=("slow down",),
    )
    assert disabled_with_irrelevant_limits == disabled_canonical
    assert GPUStageRetryPolicy(
        actor_task_retry_on_errors=[], retried_map_errors=[]
    ) == GPUStageRetryPolicy(actor_task_retry_on_errors=False, retried_map_errors=False)

    class Retryable(RuntimeError):
        pass

    retrying = GPUStageRetryPolicy(
        actor_max_restarts=4,
        actor_max_task_retries=2,
        actor_task_retry_on_errors=[Retryable],
        actor_init_retry_on_errors=True,
        actor_init_max_retries=5,
        retried_map_errors=["Retryable"],
        max_map_retries=7,
        retried_io_errors=["network"],
    )
    assert retrying.actor_task_retry_on_errors == (Retryable,)
    assert retrying.retried_map_errors == ("Retryable",)
    assert retrying.retried_io_errors == ("network",)
    assert GPUExecutionCompatibilityKey(
        num_gpus=0, retry_policy=retrying
    ) != GPUExecutionCompatibilityKey(num_gpus=0, retry_policy=disabled_canonical)


def test_range_and_complete_partition_facts_satisfy_exact_requirements():
    range_fact = PropertyFact(
        RangePartitioning(("User",), "scheme"), Evidence.USER_ASSERTED
    )
    complete_fact = PropertyFact(
        CompletePartitionBundles(), Evidence.OPERATOR_GUARANTEE
    )

    assert property_satisfies(range_fact, KeysColocated(("User", "Card")))
    assert not property_satisfies(range_fact, KeysColocated(("Card", "User")))
    assert property_satisfies(
        range_fact,
        RangePartitioningRequired(("User",), "scheme"),
    )
    assert not property_satisfies(
        range_fact,
        RangePartitioningRequired(("Card",), "scheme"),
    )
    assert not property_satisfies(
        range_fact,
        RangePartitioningRequired(("User",), "different-scheme"),
    )
    assert property_satisfies(complete_fact, CompletePartitionBundles())
    assert not property_satisfies(
        PropertyFact(UnknownPartitioning(), Evidence.DERIVED),
        KeysColocated(("User",)),
    )


def test_composition_uses_only_explicit_downstream_readvertisement():
    range_fact = PropertyFact(
        RangePartitioning(("User",), "scheme"), Evidence.OPERATOR_GUARANTEE
    )
    upstream = _spec(
        "upstream",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
        properties=(range_fact,),
    )
    passthrough_without_contract = _spec(
        "passthrough",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
    )

    composed = compose_gpu_operator_specs(upstream, passthrough_without_contract)
    assert composed.output_properties == ()


def test_range_validation_obligation_orders_dependent_transforms():
    obligation = RowsOwnedByRange("User", "scheme")
    range_fact = PropertyFact(
        RangePartitioning(("User",), "scheme"), Evidence.USER_ASSERTED
    )
    map_spec = _spec(
        "map",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
        properties=(range_fact,),
        creates=(obligation,),
    )
    validator = _spec(
        "validate",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
        properties=(range_fact,),
        discharges=(obligation,),
    )
    grouped = _spec(
        "grouped",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
        requirements=(KeysColocated(("User", "Card")),),
        requires_resolved=True,
    )

    composed = compile_gpu_region((map_spec, validator, grouped))
    assert composed.transform_ids == ("map", "validate", "grouped")


def test_partial_region_probe_allows_a_later_spec_to_discharge_obligation():
    obligation = RowsOwnedByRange("User", "scheme")
    creator = _spec(
        "create-obligation",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
        creates=(obligation,),
    )
    discharger = _spec(
        "discharge-obligation",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
        discharges=(obligation,),
    )

    with pytest.raises(GPUCompositionError) as exc_info:
        compile_gpu_region((creator,))
    assert (
        exc_info.value.reason
        is GPUCompositionErrorReason.UNRESOLVED_VALIDATION_OBLIGATION
    )

    prefix = compile_gpu_region((creator,), require_resolved_output=False)
    assert prefix.transform_ids == ("create-obligation",)
    completed = compile_gpu_region((creator, discharger))
    assert completed.transform_ids == (
        "create-obligation",
        "discharge-obligation",
    )


def test_required_fusion_group_and_provenance_propagate_through_composition():
    range_fact = PropertyFact(
        RangePartitioning(("User",), "scheme"), Evidence.OPERATOR_GUARANTEE
    )
    complete_fact = PropertyFact(
        CompletePartitionBundles(), Evidence.OPERATOR_GUARANTEE
    )
    required_provenance = ("read", "map", "mgp")
    common = {
        "fusion_group_id": "required-v1-region",
        "requires_complete_fusion": True,
        "required_provenance": required_provenance,
    }
    read = _spec(
        "read",
        PayloadKind.RANGE_WORK,
        PayloadKind.RANGE_DATA,
        input_adapter=RANGE_WORK_ADAPTER_KEY,
        properties=(range_fact, complete_fact),
        owns_input_parallelism=True,
        **common,
    )
    mapped = _spec(
        "map",
        PayloadKind.RANGE_DATA,
        PayloadKind.RANGE_DATA,
        properties=(range_fact, complete_fact),
        **common,
    )
    grouped = _spec(
        "mgp",
        PayloadKind.RANGE_DATA,
        PayloadKind.RANGE_DATA,
        requirements=(
            KeysColocated(("User", "Card")),
            CompletePartitionBundles(),
        ),
        uses_terminal_output_adapter=True,
        **common,
    )

    assert read.logical_provenance != read.required_logical_provenance
    stage = compile_gpu_region((read, mapped, grouped))

    assert stage.fusion_group_id == "required-v1-region"
    assert stage.requires_complete_fusion
    assert stage.required_logical_provenance == required_provenance
    assert stage.logical_provenance == required_provenance
    assert stage.owns_input_parallelism
    assert stage.uses_terminal_output_adapter
    assert stage.active_output_adapter_key == RANGE_DATA_TERMINAL_ADAPTER_KEY


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"requires_complete_fusion": True}, "requires a fusion_group_id"),
        (
            {
                "fusion_group_id": "required-v1-region",
                "requires_complete_fusion": True,
            },
            "requires logical provenance",
        ),
        (
            {"required_logical_provenance": ("read",)},
            "requires a fusion_group_id",
        ),
    ],
)
def test_incomplete_required_spec_declaration_is_rejected(changes, message):
    base = _spec(
        "read",
        PayloadKind.RANGE_WORK,
        PayloadKind.RANGE_DATA,
        input_adapter=RANGE_WORK_ADAPTER_KEY,
    )

    with pytest.raises(ValueError, match=message):
        replace(base, **changes)


def test_v1_boundary_registry_exposes_only_implemented_directions():
    assert (
        get_gpu_boundary_adapter(
            key=RANGE_WORK_ADAPTER_KEY,
            direction=GPUBoundaryDirection.IMPORT,
            external_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_payload=PayloadKind.RANGE_WORK,
        )
        is not None
    )
    terminal = get_gpu_boundary_adapter(
        key=RANGE_DATA_TERMINAL_ADAPTER_KEY,
        direction=GPUBoundaryDirection.EXPORT,
        external_payload=PayloadKind.RAY_BLOCK_STREAM,
        native_payload=PayloadKind.RANGE_DATA,
    )
    assert terminal is not None
    for direction in (
        GPUBoundaryDirection.IMPORT,
        GPUBoundaryDirection.EXPORT,
    ):
        assert (
            get_gpu_boundary_adapter(
                key=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
                direction=direction,
                external_payload=PayloadKind.RAY_BLOCK_STREAM,
                native_payload=PayloadKind.RANGE_DATA,
            )
            is None
        )

    spec = _spec("map", PayloadKind.RANGE_DATA, PayloadKind.RANGE_DATA)
    assert spec.output_adapter_key == RANGE_DATA_BOUNDARY_ADAPTER_KEY
    assert spec.terminal_output_adapter_key == RANGE_DATA_TERMINAL_ADAPTER_KEY
    identity = RangePartitionIdentity(0, "range", 1, 1)
    blocks = list(terminal.adapt(RangeData(identity, iter(({"value": [1]},))), None))
    assert blocks and all(not isinstance(block, RangeData) for block in blocks)
    assert blocks[0].column_names == ["value"]


def test_boundary_registry_is_additive_and_direction_aware():
    def import_value(value, _context):
        return value

    register_gpu_boundary_adapter(
        GPUBoundaryAdapter(
            key="test-additive-adapter-v1",
            direction=GPUBoundaryDirection.IMPORT,
            external_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_payload=PayloadKind.CUDF_PARTITION,
            adapt=import_value,
        ),
        replace_existing=True,
    )

    assert (
        get_gpu_boundary_adapter(
            key="test-additive-adapter-v1",
            direction=GPUBoundaryDirection.IMPORT,
            external_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_payload=PayloadKind.CUDF_PARTITION,
        )
        is not None
    )
    assert (
        get_gpu_boundary_adapter(
            key="test-additive-adapter-v1",
            direction=GPUBoundaryDirection.EXPORT,
            external_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_payload=PayloadKind.CUDF_PARTITION,
        )
        is None
    )


def test_driver_resolved_boundary_handlers_serialize_with_stage():
    def passthrough(value, _context):
        return value

    key = "test-serialized-adapter-v1"
    for direction in (
        GPUBoundaryDirection.IMPORT,
        GPUBoundaryDirection.EXPORT,
    ):
        register_gpu_boundary_adapter(
            GPUBoundaryAdapter(
                key=key,
                direction=direction,
                external_payload=PayloadKind.RAY_BLOCK_STREAM,
                native_payload=PayloadKind.CUDF_PARTITION,
                adapt=passthrough,
            )
        )
    spec = _spec(
        "adapter-transform",
        PayloadKind.CUDF_PARTITION,
        PayloadKind.CUDF_PARTITION,
        input_adapter=key,
        output_adapter=key,
        config=passthrough,
    )
    restored = cloudpickle.loads(cloudpickle.dumps(GPURegionBlockTransform(spec)))
    for direction in (
        GPUBoundaryDirection.IMPORT,
        GPUBoundaryDirection.EXPORT,
    ):
        unregister_gpu_boundary_adapter(
            key=key,
            direction=direction,
            external_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_payload=PayloadKind.CUDF_PARTITION,
        )

    assert restored([1, 2], None) == [1, 2]


def test_non_range_source_composes_through_distribution_neutral_envelopes():
    @dataclass(frozen=True)
    class HashPartitionIdentity(PartitionIdentity):
        bucket: int
        scheme_id: str

    @dataclass(frozen=True)
    class CustomRangeWork:
        partition: PartitionIdentity
        values: tuple[int, ...]

    exported_partitions = []

    def import_custom_work(values, _context):
        (work,) = tuple(values)
        assert isinstance(work, CustomRangeWork)
        return work

    def export_range_data(value, _context):
        assert isinstance(value, RangeData)
        exported_partitions.append(value.partition)
        return value.value

    def read_custom_range(work, _context):
        # No ParquetRangeWork is fabricated: the provider-owned work descriptor
        # is consumed only by its own source transform.
        return RangeData(work.partition, iter(work.values))

    def double_range(value, _context):
        return RangeData(value.partition, (item * 2 for item in value.value))

    def group_range(value, _context):
        values = tuple(value.value)
        return GroupedRangeData(value.partition, values, (0, len(values)))

    def emit_grouped_range(value, _context):
        assert isinstance(value, GroupedRangeData)
        return RangeData(value.partition, iter(value.frame))

    source_key = "test-custom-range-source-v1"
    data_key = "test-custom-range-data-v1"
    adapters = (
        GPUBoundaryAdapter(
            key=source_key,
            direction=GPUBoundaryDirection.IMPORT,
            external_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_payload=PayloadKind.RANGE_WORK,
            adapt=import_custom_work,
        ),
        GPUBoundaryAdapter(
            key=data_key,
            direction=GPUBoundaryDirection.EXPORT,
            external_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_payload=PayloadKind.RANGE_DATA,
            adapt=export_range_data,
        ),
    )
    for adapter in adapters:
        register_gpu_boundary_adapter(adapter)
    try:
        source = _spec(
            "custom-source",
            PayloadKind.RANGE_WORK,
            PayloadKind.RANGE_DATA,
            input_adapter=source_key,
            output_adapter=data_key,
            config=read_custom_range,
        )
        unary = _spec(
            "custom-unary",
            PayloadKind.RANGE_DATA,
            PayloadKind.RANGE_DATA,
            input_adapter=data_key,
            output_adapter=data_key,
            config=double_range,
        )
        grouping = _spec(
            "custom-group",
            PayloadKind.RANGE_DATA,
            PayloadKind.GROUPED_RANGE_DATA,
            input_adapter=data_key,
            output_adapter=data_key,
            config=group_range,
        )
        grouped_map = _spec(
            "custom-group-map",
            PayloadKind.GROUPED_RANGE_DATA,
            PayloadKind.RANGE_DATA,
            input_adapter=data_key,
            output_adapter=data_key,
            config=emit_grouped_range,
        )
        stage = compile_gpu_region((source, unary, grouping, grouped_map))
        identity = HashPartitionIdentity(7, "hash-scheme")
        work = CustomRangeWork(
            identity,
            (10, 12),
        )

        assert list(GPURegionBlockTransform(stage)([work], None)) == [20, 24]
        assert exported_partitions == [identity]
    finally:
        for adapter in adapters:
            unregister_gpu_boundary_adapter(
                key=adapter.key,
                direction=adapter.direction,
                external_payload=adapter.external_payload,
                native_payload=adapter.native_payload,
            )


def test_generic_region_worker_runs_synthetic_transforms_in_order():
    def add_one(values, _context):
        return (value + 1 for value in values)

    def double(values, _context):
        return (value * 2 for value in values)

    spec = GPUOperatorSpec(
        spec_instance_id="synthetic",
        transforms=(
            GPUTransformSpec(
                "synthetic/0/add",
                "synthetic_add",
                PayloadKind.RAY_BLOCK_STREAM,
                PayloadKind.RAY_BLOCK_STREAM,
                config=add_one,
            ),
            GPUTransformSpec(
                "synthetic/1/double",
                "synthetic_double",
                PayloadKind.RAY_BLOCK_STREAM,
                PayloadKind.RAY_BLOCK_STREAM,
                config=double,
            ),
        ),
        external_input_payload=PayloadKind.RAY_BLOCK_STREAM,
        external_output_payload=PayloadKind.RAY_BLOCK_STREAM,
        native_input_payload=PayloadKind.RAY_BLOCK_STREAM,
        native_output_payload=PayloadKind.RAY_BLOCK_STREAM,
        input_adapter_key="synthetic-v1",
        output_adapter_key="synthetic-v1",
        terminal_output_adapter_key="synthetic-terminal-v1",
        execution_requirements=GPUExecutionRequirements(
            compatibility_key=GPUExecutionCompatibilityKey(num_gpus=0)
        ),
    )
    worker = GPURegionWorker(spec)
    assert list(worker.apply([1, 2], None)) == [4, 6]
