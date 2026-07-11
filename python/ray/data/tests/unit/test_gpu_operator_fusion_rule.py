from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from ray.data.context import DataContext
from ray.data._internal.execution.gpu import (
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUBoundaryPolicy,
    GPUOperator,
    GPUOperatorCreationOptions,
    GPUOperatorSpec,
    GPUTransformSpec,
    PayloadKind,
    create_gpu_operator,
    register_gpu_operator_factory,
)
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import PhysicalPlan
from ray.data._internal.logical.optimizers import get_physical_ruleset
from ray.data._internal.logical.rules import (
    FuseGPUOperators,
    GPURequiredFusionError,
)


def _identity(value, _context):
    return value


class _FakeSessionGPUOperator(PhysicalOperator, GPUOperator):
    def __init__(self, spec, input_op, context, options):
        self._spec = spec
        self._options = options or GPUOperatorCreationOptions()
        super().__init__("FakeSessionGPU", [input_op], context)

    @property
    def gpu_operator_spec(self):
        return self._spec

    @property
    def execution_backend_name(self):
        return "fake_session"

    @property
    def gpu_creation_options(self):
        return self._options


class _FakeSessionFactory:
    def supports(self, spec):
        return spec.execution_requirements.backend_name == "fake_session"

    def create(self, spec, input_op, data_context, creation_options=None):
        return _FakeSessionGPUOperator(
            spec, input_op, data_context, creation_options
        )


def _spec(
    transform_id,
    input_payload,
    output_payload,
    provenance,
    *,
    complete=False,
):
    required = ("Read", "MapBatches", "MapGroupPartitions") if complete else ()
    return GPUOperatorSpec(
        transforms=(
            GPUTransformSpec(
                transform_id,
                transform_id,
                input_payload,
                output_payload,
                config=_identity,
            ),
        ),
        input_payload=input_payload,
        output_payload=output_payload,
        execution_requirements=GPUExecutionRequirements(
            compatibility_key=GPUExecutionCompatibilityKey(num_gpus=0)
        ),
        logical_provenance=(provenance,),
        required_logical_provenance=required,
        region_id="region",
        requires_complete_fusion=complete,
    )


def _three_node_plan(*, complete=False):
    context = DataContext.get_current().copy()
    context.enable_gpu_operator_fusion = True
    source = InputDataBuffer(context, input_data=[])
    read = create_gpu_operator(
        _spec(
            "read",
            PayloadKind.RANGE_WORK,
            PayloadKind.CUDF_BATCH_STREAM,
            "Read",
            complete=complete,
        ),
        source,
        context,
        GPUOperatorCreationOptions(min_rows_per_bundle=1),
    )
    mapped = create_gpu_operator(
        _spec(
            "map",
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.CUDF_BATCH_STREAM,
            "MapBatches",
            complete=complete,
        ),
        read,
        context,
    )
    grouped = create_gpu_operator(
        _spec(
            "group",
            PayloadKind.CUDF_BATCH_STREAM,
            PayloadKind.OUTPUT_BATCH_STREAM,
            "MapGroupPartitions",
            complete=complete,
        ),
        mapped,
        context,
    )
    logical = [MagicMock(name=name) for name in ("read", "map", "group")]
    read.set_logical_operators(logical[0])
    mapped.set_logical_operators(logical[1])
    grouped.set_logical_operators(logical[2])
    return PhysicalPlan(
        grouped,
        {source: logical[0], read: logical[0], mapped: logical[1], grouped: logical[2]},
        context,
    )


def test_gpu_rule_fuses_maximal_chain_and_rewrites_op_map_idempotently():
    plan = _three_node_plan()
    result = FuseGPUOperators().apply(plan)

    assert isinstance(result.dag, GPUOperator)
    assert result.dag.gpu_operator_spec.transform_ids == ("read", "map", "group")
    assert len(result.dag.input_dependencies) == 1
    assert isinstance(result.dag.input_dependencies[0], InputDataBuffer)
    assert set(result.op_map) == {result.dag.input_dependencies[0], result.dag}
    assert result.dag.input_dependencies[0].output_dependencies == [result.dag]

    again = FuseGPUOperators().apply(result)
    assert again.dag is result.dag
    assert again.dag.gpu_operator_spec.canonical_contract() == (
        result.dag.gpu_operator_spec.canonical_contract()
    )


def test_gpu_rule_flag_off_is_exact_noop():
    plan = _three_node_plan()
    plan.context.enable_gpu_operator_fusion = False
    assert FuseGPUOperators().apply(plan) is plan


def test_required_complete_region_fails_if_a_boundary_stops_fusion():
    plan = _three_node_plan(complete=True)
    middle = plan.dag.input_dependencies[0]
    middle._gpu_operator_spec = replace(
        middle.gpu_operator_spec,
        output_boundary=GPUBoundaryPolicy.REQUIRED,
    )
    with pytest.raises(GPURequiredFusionError, match="could not be fused"):
        FuseGPUOperators().apply(plan)


def test_required_complete_region_fuses_and_validates_provenance():
    result = FuseGPUOperators().apply(_three_node_plan(complete=True))
    assert result.dag.gpu_operator_spec.logical_provenance == (
        "Read",
        "MapBatches",
        "MapGroupPartitions",
    )


def test_gpu_physical_rule_order_is_explicit():
    names = [rule.__name__ for rule in get_physical_ruleset()]
    assert names == [
        "InheritTargetMaxBlockSizeRule",
        "FuseGPUOperators",
        "SetReadParallelismRule",
        "FuseOperators",
        "ConfigureMapTaskMemoryUsingOutputSize",
    ]


def test_backend_factory_can_be_replaced_without_actor_pool_dependencies():
    context = DataContext.get_current().copy()
    source = InputDataBuffer(context, input_data=[])
    spec = _spec(
        "session_transform",
        PayloadKind.RAY_BLOCK_STREAM,
        PayloadKind.RAY_BLOCK_STREAM,
        "Synthetic",
    )
    spec = replace(
        spec,
        execution_requirements=GPUExecutionRequirements(
            backend_name="fake_session",
            compatibility_key=replace(
                spec.compatibility_key, worker_lifecycle="session"
            ),
        ),
    )
    register_gpu_operator_factory(
        "fake_session", _FakeSessionFactory(), replace_existing=True
    )
    operator = create_gpu_operator(spec, source, context)
    assert isinstance(operator, _FakeSessionGPUOperator)
    assert operator.gpu_operator_spec is spec
