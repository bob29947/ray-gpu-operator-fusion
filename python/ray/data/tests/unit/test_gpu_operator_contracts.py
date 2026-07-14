from dataclasses import dataclass

import pytest
from ray import cloudpickle

from ray.data._internal.execution.gpu.composition import (
    GPUCompositionError,
    GPUCompositionErrorReason,
    foldable_edge,
    fuse_gpu_operator_specs,
)
from ray.data._internal.execution.gpu.operator_spec import (
    DEVICE_HANDLE,
    RANGE_DATA,
    RANGE_WORK,
    RAY_BLOCK_STREAM,
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUOperatorSpec,
    GPUStageRetryPolicy,
    GPUTransformSpec,
    PayloadKind,
    RuntimeKey,
    merge_gpu_execution_requirements,
    preserve_range,
)
from ray.data._internal.execution.gpu.properties import (
    KeysColocated,
    PropertyConflict,
    RangePartitioning,
    RangePartitioningRequired,
    property_satisfies,
    range_partitioning_fact,
    requirements_satisfied,
)
from ray.data._internal.execution.gpu.region_compiler import (
    compile_gpu_region,
    compile_region,
)
from ray.data._internal.execution.gpu.runtime import (
    GPURegionBlockTransform,
    TransformRuntimeRegistration,
    register_gpu_transform_runtime,
    resolve_gpu_transform_runtime_registration,
    supports_gpu_transform_runtime,
    unregister_gpu_transform_runtime,
)
from ray.data._internal.execution.gpu.transforms.boundary import (
    EXPORT_RANGE_DATA_KIND,
    IMPORT_RANGE_WORK_KIND,
)
from ray.data._internal.execution.gpu.transforms.map_batches import (
    MAP_BATCHES_KIND,
    MapBatchesConfig,
)


NATIVE = PayloadKind("test_native")


class _ExplodingEq:
    def __eq__(self, other):
        raise AssertionError("opaque configs must not be compared")


def _execution(runtime_id="ray-data-gpu-v1"):
    return GPUExecutionRequirements(
        compatibility_key=GPUExecutionCompatibilityKey(
            runtime_id=runtime_id,
            num_gpus=0,
        )
    )


def _spec(
    kind,
    input_payload=NATIVE,
    output_payload=NATIVE,
    *,
    execution=None,
    requirements=(),
    facts=(),
    config=None,
):
    return GPUOperatorSpec(
        transforms=(
            GPUTransformSpec(
                kind,
                input_payload,
                output_payload,
                _ExplodingEq() if config is None else config,
            ),
        ),
        execution_requirements=execution or _execution(),
        input_requirements=requirements,
        output_facts=facts,
    )


def test_payload_kind_is_open_and_boundaries_are_payload_properties():
    custom = PayloadKind("plugin_owned_payload")
    boundary = PayloadKind("plugin_device_boundary", True)

    assert custom.key == custom.value == "plugin_owned_payload"
    assert not custom.materializable_boundary
    assert boundary.materializable_boundary
    assert RAY_BLOCK_STREAM.materializable_boundary
    assert DEVICE_HANDLE.materializable_boundary
    assert not RANGE_DATA.materializable_boundary
    with pytest.raises(ValueError):
        PayloadKind("device_handle")


def test_operator_spec_derives_ports_and_excludes_opaque_config_from_contract():
    spec = GPUOperatorSpec(
        transforms=(
            GPUTransformSpec("one", RANGE_WORK, RANGE_DATA, _ExplodingEq()),
            GPUTransformSpec("two", RANGE_DATA, NATIVE, _ExplodingEq()),
        ),
        execution_requirements=_execution(),
    )

    assert spec.input_payload == RANGE_WORK
    assert spec.output_payload == NATIVE
    assert spec.canonical_contract()
    restored = cloudpickle.loads(cloudpickle.dumps(spec))
    assert restored.canonical_contract() == spec.canonical_contract()


def test_operator_spec_rejects_mismatch_and_internal_materialization():
    with pytest.raises(ValueError, match="payloads do not match"):
        GPUOperatorSpec(
            transforms=(
                GPUTransformSpec("one", RANGE_WORK, RANGE_DATA),
                GPUTransformSpec("two", NATIVE, RANGE_DATA),
            ),
            execution_requirements=_execution(),
        )
    internal_boundary = PayloadKind("test_internal_boundary", True)
    with pytest.raises(ValueError, match="materializable payload"):
        GPUOperatorSpec(
            transforms=(
                GPUTransformSpec("export", NATIVE, internal_boundary),
                GPUTransformSpec("import", internal_boundary, NATIVE),
            ),
            execution_requirements=_execution(),
        )


def test_composition_is_pure_associative_and_supports_repeated_map_kinds():
    fact = RangePartitioning(("user",), "range-v1")
    a = _spec("read", RANGE_WORK, RANGE_DATA, facts=(fact,))
    b = _spec("map_batches", RANGE_DATA, RANGE_DATA, facts=(fact,))
    c = _spec(
        "map_batches",
        RANGE_DATA,
        RANGE_DATA,
        requirements=(RangePartitioningRequired(fact.keys, fact.scheme_id),),
        facts=(fact,),
    )
    before = tuple(spec.canonical_contract() for spec in (a, b, c))

    left = fuse_gpu_operator_specs(fuse_gpu_operator_specs(a, b), c)
    right = fuse_gpu_operator_specs(a, fuse_gpu_operator_specs(b, c))
    folded = compile_gpu_region((a, b, c))

    assert left.canonical_contract() == right.canonical_contract()
    assert folded.canonical_contract() == left.canonical_contract()
    assert tuple(transform.kind for transform in folded.transforms) == (
        "read",
        "map_batches",
        "map_batches",
    )
    assert before == tuple(spec.canonical_contract() for spec in (a, b, c))


@pytest.mark.parametrize("count", [1, 3, 10])
def test_one_three_and_ten_specs_compile_to_one_ordered_program(count):
    specs = tuple(_spec("map_batches") for _ in range(count))
    compiled = compile_gpu_region(specs)

    assert len(compiled.transforms) == count
    assert all(transform.kind == "map_batches" for transform in compiled.transforms)


def test_composition_has_stable_payload_property_and_execution_refusals():
    with pytest.raises(GPUCompositionError) as payload_error:
        fuse_gpu_operator_specs(
            _spec("a", NATIVE, RANGE_DATA),
            _spec("b", RANGE_WORK, NATIVE),
        )
    assert payload_error.value.reason is GPUCompositionErrorReason.PAYLOAD_INCOMPATIBLE

    with pytest.raises(GPUCompositionError) as property_error:
        fuse_gpu_operator_specs(
            _spec("a"),
            _spec("b", requirements=(KeysColocated(("user",)),)),
        )
    assert (
        property_error.value.reason
        is GPUCompositionErrorReason.INPUT_REQUIREMENTS_UNSATISFIED
    )

    with pytest.raises(GPUCompositionError) as execution_error:
        fuse_gpu_operator_specs(
            _spec("a"),
            _spec("b", execution=_execution("different-runtime")),
        )
    assert (
        execution_error.value.reason is GPUCompositionErrorReason.EXECUTION_INCOMPATIBLE
    )


def test_payload_keyed_foldability_refuses_materializable_edges():
    boundary = PayloadKind("synthetic_materializable_boundary", True)
    upstream = _spec("export", NATIVE, boundary)
    downstream = _spec("import", boundary, NATIVE)

    assert not foldable_edge(upstream, downstream)
    with pytest.raises(GPUCompositionError) as error:
        fuse_gpu_operator_specs(upstream, downstream)
    assert error.value.reason is GPUCompositionErrorReason.MATERIALIZABLE_BOUNDARY


def test_range_protocol_uses_absence_for_unknown_and_preserves_exact_fact():
    fact = RangePartitioning(("user",), "range-v1")

    assert range_partitioning_fact(()) is None
    assert preserve_range((), RANGE_DATA, object()).output_facts == ()
    preservation = preserve_range((fact,), RANGE_DATA, "validation-config")
    assert preservation.input_requirements == (
        RangePartitioningRequired(fact.keys, fact.scheme_id),
    )
    assert preservation.output_facts == (fact,)
    assert tuple(t.kind for t in preservation.validation_transforms) == (
        "validate_range_ownership",
    )
    assert requirements_satisfied((fact,), (KeysColocated(("user", "card")),))
    assert property_satisfies(
        fact,
        RangePartitioningRequired(("user",), "range-v1"),
    )

    with pytest.raises(PropertyConflict):
        range_partitioning_fact(
            (
                fact,
                RangePartitioning(("other",), "other-range"),
            )
        )


def test_execution_requirements_merge_inherited_and_reject_conflicts():
    concrete = _execution()
    inherited = GPUExecutionRequirements()

    assert merge_gpu_execution_requirements(inherited, concrete) == concrete
    assert merge_gpu_execution_requirements(concrete, inherited) == concrete
    with pytest.raises(ValueError, match="runtime_id"):
        merge_gpu_execution_requirements(concrete, _execution("other-runtime"))
    with pytest.raises(ValueError, match="no compatible"):
        merge_gpu_execution_requirements(
            GPUExecutionRequirements(frozenset(("one",))),
            GPUExecutionRequirements(frozenset(("two",))),
        )


def test_execution_requirements_resolve_unset_placement_constraints():
    unset = _execution()
    spread = GPUExecutionRequirements(
        compatibility_key=GPUExecutionCompatibilityKey(
            num_gpus=0,
            scheduling_strategy="SPREAD",
            label_selector=(("gpu-generation", "test"),),
        )
    )

    merged = merge_gpu_execution_requirements(unset, spread)

    assert merged.compatibility_key.scheduling_strategy == "SPREAD"
    assert merged.compatibility_key.label_selector == (("gpu-generation", "test"),)
    with pytest.raises(ValueError, match="scheduling_strategy"):
        merge_gpu_execution_requirements(
            spread,
            GPUExecutionRequirements(
                compatibility_key=GPUExecutionCompatibilityKey(
                    num_gpus=0,
                    scheduling_strategy="DEFAULT",
                )
            ),
        )


def test_execution_requirements_add_source_bootstrap_constraints():
    base = _execution()
    source = GPUExecutionRequirements(
        compatibility_key=GPUExecutionCompatibilityKey(
            num_gpus=0,
            actor_bootstrap=(("source_kind", "s3"), ("region", "us-west-2")),
        )
    )

    merged = merge_gpu_execution_requirements(base, source)

    assert merged.compatibility_key.actor_bootstrap == (
        ("region", "us-west-2"),
        ("source_kind", "s3"),
    )
    with pytest.raises(ValueError, match="bootstrap requirement"):
        merge_gpu_execution_requirements(
            source,
            GPUExecutionRequirements(
                compatibility_key=GPUExecutionCompatibilityKey(
                    num_gpus=0,
                    actor_bootstrap=(("source_kind", "local"),),
                )
            ),
        )


def test_retry_policy_normalizes_disabled_scopes_and_compares_effective_policy():
    normalized = GPUStageRetryPolicy(
        actor_init_retry_on_errors=False,
        actor_init_max_retries=99,
        retried_map_errors=False,
        max_map_retries=99,
        retried_io_errors=["slow down"],
    )
    assert normalized.actor_init_max_retries == 0
    assert normalized.max_map_retries == 0
    assert normalized.retried_io_errors == ("slow down",)
    assert normalized != GPUStageRetryPolicy(retried_io_errors=("timeout",))


def test_compile_region_adds_ordinary_transform_boundaries():
    core = _spec("core")
    ingress = _spec("import", RAY_BLOCK_STREAM, NATIVE)
    egress = _spec("export", NATIVE, RAY_BLOCK_STREAM)

    closed = compile_region(core, ingress, egress)

    assert closed.input_payload == RAY_BLOCK_STREAM
    assert closed.output_payload == RAY_BLOCK_STREAM
    assert tuple(transform.kind for transform in closed.transforms) == (
        "import",
        "core",
        "export",
    )


def test_builtin_boundary_runtimes_are_payload_exact():
    ingress = GPUTransformSpec(
        IMPORT_RANGE_WORK_KIND,
        RAY_BLOCK_STREAM,
        RANGE_WORK,
    )
    egress = GPUTransformSpec(
        EXPORT_RANGE_DATA_KIND,
        RANGE_DATA,
        RAY_BLOCK_STREAM,
    )
    wrong_port = GPUTransformSpec(
        IMPORT_RANGE_WORK_KIND,
        RAY_BLOCK_STREAM,
        RANGE_DATA,
    )

    assert supports_gpu_transform_runtime(ingress)
    assert supports_gpu_transform_runtime(egress)
    assert not supports_gpu_transform_runtime(wrong_port)


def test_builtin_map_gate_validates_config_without_constructing_udf():
    class NeverConstructed:
        def __init__(self):
            raise AssertionError("planning must not construct the UDF")

    transform = GPUTransformSpec(
        MAP_BATCHES_KIND,
        RANGE_DATA,
        RANGE_DATA,
        MapBatchesConfig(NeverConstructed, 16, False),
    )

    assert supports_gpu_transform_runtime(transform)


@dataclass(frozen=True)
class _Delta:
    value: int


class _DeltaRuntime:
    def __init__(self, delta):
        self._delta = delta

    def initialize(self):
        pass

    def apply(self, values, _context):
        return [value + self._delta for value in values]


def _delta_factory(transform, _context):
    return _DeltaRuntime(transform.config.value)


def _delta_validator(config):
    return isinstance(config, _Delta)


def test_runtime_registry_validates_exact_contract_without_constructing_runtime():
    key = RuntimeKey("test-runtime", "delta", NATIVE, NATIVE)
    registration = TransformRuntimeRegistration(_delta_validator, _delta_factory)
    register_gpu_transform_runtime(key, registration)
    try:
        supported = GPUTransformSpec(
            "delta",
            NATIVE,
            NATIVE,
            _Delta(1),
            runtime_id="test-runtime",
        )
        wrong_config = GPUTransformSpec(
            "delta",
            NATIVE,
            NATIVE,
            lambda value: value,
            runtime_id="test-runtime",
        )
        wrong_payload = GPUTransformSpec(
            "delta",
            NATIVE,
            RANGE_DATA,
            _Delta(1),
            runtime_id="test-runtime",
        )
        assert supports_gpu_transform_runtime(supported)
        assert resolve_gpu_transform_runtime_registration(supported) is registration
        assert not supports_gpu_transform_runtime(wrong_config)
        assert not supports_gpu_transform_runtime(wrong_payload)
    finally:
        unregister_gpu_transform_runtime(key, expected=registration)


def test_unregistered_callable_config_is_not_an_execution_escape_hatch():
    transform = GPUTransformSpec(
        "unregistered-callable",
        NATIVE,
        NATIVE,
        lambda value: value,
        runtime_id="test-runtime",
    )
    assert not supports_gpu_transform_runtime(transform)


def test_driver_resolved_runtime_registrations_execute_after_unregister():
    import_key = RuntimeKey("test-runtime", "import", RAY_BLOCK_STREAM, NATIVE)
    delta_key = RuntimeKey("test-runtime", "delta", NATIVE, NATIVE)
    export_key = RuntimeKey("test-runtime", "export", NATIVE, RAY_BLOCK_STREAM)
    registrations = {
        import_key: TransformRuntimeRegistration(_delta_validator, _delta_factory),
        delta_key: TransformRuntimeRegistration(_delta_validator, _delta_factory),
        export_key: TransformRuntimeRegistration(_delta_validator, _delta_factory),
    }
    for key, registration in registrations.items():
        register_gpu_transform_runtime(key, registration)
    core = GPUOperatorSpec(
        transforms=(
            GPUTransformSpec("delta", NATIVE, NATIVE, _Delta(1), "test-runtime"),
            GPUTransformSpec("delta", NATIVE, NATIVE, _Delta(2), "test-runtime"),
        ),
        execution_requirements=_execution(),
    )
    ingress = GPUOperatorSpec(
        (
            GPUTransformSpec(
                "import", RAY_BLOCK_STREAM, NATIVE, _Delta(0), "test-runtime"
            ),
        ),
        _execution(),
    )
    egress = GPUOperatorSpec(
        (
            GPUTransformSpec(
                "export", NATIVE, RAY_BLOCK_STREAM, _Delta(0), "test-runtime"
            ),
        ),
        _execution(),
    )
    closed = compile_region(core, ingress, egress)
    resolved = tuple(
        resolve_gpu_transform_runtime_registration(transform)
        for transform in closed.transforms
    )
    block_transform = cloudpickle.loads(
        cloudpickle.dumps(GPURegionBlockTransform(closed, resolved))
    )
    for key, registration in registrations.items():
        unregister_gpu_transform_runtime(key, expected=registration)

    assert block_transform([1, 2], None) == [4, 5]
