from dataclasses import replace

import pytest
from ray import cloudpickle

from ray.data._internal.execution.gpu import (
    Evidence,
    GPUBoundaryPolicy,
    GPUCompositionError,
    GPUCompositionErrorReason,
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUOperatorSpec,
    GPUTransformSpec,
    KeysColocated,
    PayloadKind,
    PropertyFact,
    RangePartitioning,
    RowsOwnedByRange,
    compile_gpu_region,
    compose_gpu_operator_specs,
)
from ray.data._internal.execution.gpu.runtime import GPURegionWorker


class _ExplodingEq:
    def __eq__(self, other):
        raise AssertionError("opaque UDF equality must not be called")

    def __call__(self, value, _context):
        return value


def _spec(
    transform_id,
    input_payload,
    output_payload,
    *,
    key=None,
    requirements=(),
    properties=(),
    boundary=GPUBoundaryPolicy.ELIDABLE,
    creates=(),
    discharges=(),
    requires_resolved=False,
    config=None,
):
    if key is None:
        key = GPUExecutionCompatibilityKey(num_gpus=0)
    return GPUOperatorSpec(
        transforms=(
            GPUTransformSpec(
                transform_id=transform_id,
                kind=transform_id,
                input_payload=input_payload,
                output_payload=output_payload,
                config=config or _ExplodingEq(),
                creates_obligations=creates,
                discharges_obligations=discharges,
                requires_resolved_obligations=requires_resolved,
            ),
        ),
        input_payload=input_payload,
        output_payload=output_payload,
        execution_requirements=GPUExecutionRequirements(
            compatibility_key=key
        ),
        input_requirements=requirements,
        output_properties=properties,
        logical_provenance=(transform_id,),
        region_id="test-region",
    )


def test_compose_is_pairwise_pure_associative_and_serializable():
    range_fact = PropertyFact(
        RangePartitioning(("User",), "scheme"), Evidence.OPERATOR_GUARANTEE
    )
    a = _spec(
        "a",
        PayloadKind.RAY_BLOCK_STREAM,
        PayloadKind.CUDF_BATCH_STREAM,
        properties=(range_fact,),
    )
    b = _spec(
        "b",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.CUDF_PARTITION,
        requirements=(KeysColocated(("User", "Card")),),
        properties=(range_fact,),
    )
    c = _spec(
        "c",
        PayloadKind.CUDF_PARTITION,
        PayloadKind.OUTPUT_BATCH_STREAM,
    )
    before = (a.canonical_contract(), b.canonical_contract(), c.canonical_contract())

    left = compose_gpu_operator_specs(compose_gpu_operator_specs(a, b), c)
    right = compose_gpu_operator_specs(a, compose_gpu_operator_specs(b, c))
    folded = compile_gpu_region((a, b, c))

    assert left.canonical_contract() == right.canonical_contract()
    assert left.canonical_contract() == folded.canonical_contract()
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
                PayloadKind.RAY_BLOCK_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
            ),
            _spec(
                "b",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.OUTPUT_BATCH_STREAM,
                key=GPUExecutionCompatibilityKey(
                    runtime_id="different", num_gpus=0
                ),
            ),
            GPUCompositionErrorReason.EXECUTION_INCOMPATIBLE,
        ),
        (
            _spec(
                "a",
                PayloadKind.RAY_BLOCK_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
            ),
            _spec(
                "b",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.OUTPUT_BATCH_STREAM,
                requirements=(KeysColocated(("key",)),),
            ),
            GPUCompositionErrorReason.INPUT_REQUIREMENTS_UNSATISFIED,
        ),
        (
            _spec(
                "a", PayloadKind.RAY_BLOCK_STREAM, PayloadKind.CUDF_PARTITION
            ),
            _spec(
                "b",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.OUTPUT_BATCH_STREAM,
            ),
            GPUCompositionErrorReason.PAYLOAD_INCOMPATIBLE,
        ),
        (
            replace(
                _spec(
                    "a",
                    PayloadKind.RAY_BLOCK_STREAM,
                    PayloadKind.CUDF_BATCH_STREAM,
                ),
                output_boundary=GPUBoundaryPolicy.REQUIRED,
            ),
            _spec(
                "b",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.OUTPUT_BATCH_STREAM,
            ),
            GPUCompositionErrorReason.PHYSICAL_BOUNDARY_REQUIRED,
        ),
        (
            _spec(
                "a",
                PayloadKind.RAY_BLOCK_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
                creates=(RowsOwnedByRange("User", "scheme"),),
            ),
            _spec(
                "b",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.OUTPUT_BATCH_STREAM,
                requires_resolved=True,
            ),
            GPUCompositionErrorReason.UNRESOLVED_VALIDATION_OBLIGATION,
        ),
        (
            _spec(
                "duplicate",
                PayloadKind.RAY_BLOCK_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
            ),
            _spec(
                "duplicate",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.OUTPUT_BATCH_STREAM,
            ),
            GPUCompositionErrorReason.DUPLICATE_TRANSFORM_ID,
        ),
    ],
)
def test_compose_stable_error_codes(upstream, downstream, reason):
    with pytest.raises(GPUCompositionError) as exc_info:
        compose_gpu_operator_specs(upstream, downstream)
    assert exc_info.value.reason is reason


def test_range_prefix_semantically_satisfies_colocation_after_validation():
    obligation = RowsOwnedByRange("User", "scheme")
    range_fact = PropertyFact(
        RangePartitioning(("User",), "scheme"), Evidence.USER_ASSERTED
    )
    map_spec = GPUOperatorSpec(
        transforms=(
            GPUTransformSpec(
                "map",
                "map",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
                creates_obligations=(obligation,),
            ),
            GPUTransformSpec(
                "validate",
                "validate",
                PayloadKind.CUDF_BATCH_STREAM,
                PayloadKind.CUDF_BATCH_STREAM,
                discharges_obligations=(obligation,),
            ),
        ),
        input_payload=PayloadKind.CUDF_BATCH_STREAM,
        output_payload=PayloadKind.CUDF_BATCH_STREAM,
        execution_requirements=GPUExecutionRequirements(
            compatibility_key=GPUExecutionCompatibilityKey(num_gpus=0)
        ),
        output_properties=(range_fact,),
        logical_provenance=("MapBatches",),
        region_id="test-region",
    )
    grouped_spec = _spec(
        "grouped",
        PayloadKind.CUDF_BATCH_STREAM,
        PayloadKind.OUTPUT_BATCH_STREAM,
        requirements=(KeysColocated(("User", "Card")),),
        requires_resolved=True,
    )
    composed = compose_gpu_operator_specs(map_spec, grouped_spec)
    assert composed.transform_ids == ("map", "validate", "grouped")


def test_generic_region_worker_runs_synthetic_transforms_in_order():
    def add_one(values, _context):
        return (value + 1 for value in values)

    def double(values, _context):
        return (value * 2 for value in values)

    spec = GPUOperatorSpec(
        transforms=(
            GPUTransformSpec(
                "add",
                "synthetic_add",
                PayloadKind.RAY_BLOCK_STREAM,
                PayloadKind.RAY_BLOCK_STREAM,
                config=add_one,
            ),
            GPUTransformSpec(
                "double",
                "synthetic_double",
                PayloadKind.RAY_BLOCK_STREAM,
                PayloadKind.RAY_BLOCK_STREAM,
                config=double,
            ),
        ),
        input_payload=PayloadKind.RAY_BLOCK_STREAM,
        output_payload=PayloadKind.RAY_BLOCK_STREAM,
        execution_requirements=GPUExecutionRequirements(
            compatibility_key=GPUExecutionCompatibilityKey(num_gpus=0)
        ),
    )
    worker = GPURegionWorker(spec)
    assert list(worker.apply([1, 2], None)) == [4, 6]
