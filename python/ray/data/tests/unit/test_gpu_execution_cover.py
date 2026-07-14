from dataclasses import dataclass

import pytest

from ray.data._internal.execution.gpu.cover import validate_gpu_execution_plan
from ray.data._internal.execution.gpu.planning_state import GPUFusionPlanState
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import PhysicalPlan
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)
from ray.data.context import DataContext


class _Logical:
    def __init__(self, name, inputs=()):
        self.name = name
        self.input_dependencies = list(inputs)

    def post_order_iter(self):
        for input_op in self.input_dependencies:
            yield from input_op.post_order_iter()
        yield self


@dataclass(frozen=True)
class _Payload:
    key: str
    materializable_boundary: bool = False


class _Spec:
    transforms = ()


class _Physical(PhysicalOperator):
    def __init__(self, name, inputs, context):
        super().__init__(name, list(inputs), context)


class _ExecutableRegion(_Physical):
    def __init__(self, name, inputs, context, spec, backend="test"):
        self._spec = spec
        self._backend = backend
        super().__init__(name, inputs, context)

    @property
    def gpu_operator_spec(self):
        return self._spec

    @property
    def execution_backend_name(self):
        return self._backend


def _context():
    context = DataContext()
    context.enable_gpu_operator_fusion = True
    return context


def _strict_plan(logical_ops, *, internal_payloads=()):
    context = _context()
    input_buffer = InputDataBuffer(context, input_data=[])
    spec = _Spec()
    region = _ExecutableRegion("SyntheticGPU", [input_buffer], context, spec)
    region.set_logical_operators(*logical_ops)
    state = GPUFusionPlanState()
    state.initialize_logical_universe(logical_ops[-1])
    state.register_infrastructure(input_buffer)
    state.register_materialized_region(
        region,
        spec,
        "test",
        input_payload=_Payload("ray_block_stream", True),
        output_payload=_Payload("ray_block_stream", True),
        internal_payloads=internal_payloads,
    )
    return PhysicalPlan(
        region,
        {region: logical_ops[-1]},
        context,
        gpu_fusion_state=state,
    )


def test_strict_cover_accepts_one_generic_attested_region():
    read = _Logical("Read")
    synthetic = _Logical("NewUnaryAPI", (read,))
    plan = _strict_plan((read, synthetic))

    validate_gpu_execution_plan(plan)


def test_strict_cover_accepts_optional_terminal_output():
    read = _Logical("Read")
    transform = _Logical("Transform", (read,))
    write = _Logical("Write", (transform,))
    context = _context()
    input_buffer = InputDataBuffer(context, input_data=[])
    spec = _Spec()
    region = _ExecutableRegion("SyntheticGPU", [input_buffer], context, spec)
    region.set_logical_operators(read, transform)
    terminal = _Physical("Write", [region], context)
    terminal.set_logical_operators(write)
    state = GPUFusionPlanState()
    state.initialize_logical_universe(write, terminal_logical=(write,))
    state.register_infrastructure(input_buffer)
    state.register_terminal(write, terminal)
    state.register_materialized_region(
        region,
        spec,
        "test",
        input_payload=_Payload("ray_block_stream", True),
        output_payload=_Payload("ray_block_stream", True),
    )
    plan = PhysicalPlan(
        terminal,
        {region: transform, terminal: write},
        context,
        gpu_fusion_state=state,
    )

    validate_gpu_execution_plan(plan)


def test_strict_cover_rejects_stock_transform_around_gpu_region():
    read = _Logical("Read")
    gpu = _Logical("GPU", (read,))
    cpu = _Logical("CPU", (gpu,))
    plan = _strict_plan((read, gpu, cpu))
    region = plan.dag
    region.set_logical_operators(read, gpu)
    stock = _Physical("StockMap", [region], plan.context)
    stock.set_logical_operators(cpu)
    plan._dag = stock
    plan.gpu_fusion_state.register_stock_fallback(
        cpu,
        GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
        "no registered GPU lowerer for CPU",
        (stock,),
    )

    with pytest.raises(GPUPlanningError) as captured:
        validate_gpu_execution_plan(plan)

    assert captured.value.reason is GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY
    assert "StockMap" in str(captured.value)
    assert "CPU" in str(captured.value)


def test_strict_cover_rejects_missing_and_duplicate_lineage():
    read = _Logical("Read")
    first = _Logical("First", (read,))
    second = _Logical("Second", (first,))
    plan = _strict_plan((read, first, second))
    plan.dag.set_logical_operators(read, first, first)

    with pytest.raises(GPUPlanningError) as captured:
        validate_gpu_execution_plan(plan)

    assert captured.value.reason is GPUPlanningErrorReason.INCOMPLETE_FUSION
    assert "lineage" in str(captured.value) or "partition" in str(captured.value)


def test_strict_cover_rejects_internal_materializable_boundary():
    read = _Logical("Read")
    transform = _Logical("Transform", (read,))
    plan = _strict_plan(
        (read, transform),
        internal_payloads=(_Payload("device_handle", True),),
    )

    with pytest.raises(GPUPlanningError) as captured:
        validate_gpu_execution_plan(plan)

    assert captured.value.reason is GPUPlanningErrorReason.INCOMPLETE_FUSION
    assert "unattested" in str(captured.value)


def test_strict_cover_rejects_fresh_source_only_plan():
    source = _Logical("InputData")
    context = _context()
    input_buffer = InputDataBuffer(context, input_data=[])
    input_buffer.set_logical_operators(source)
    state = GPUFusionPlanState()
    state.initialize_logical_universe(source)
    state.register_infrastructure(input_buffer)
    plan = PhysicalPlan(
        input_buffer,
        {input_buffer: source},
        context,
        gpu_fusion_state=state,
    )

    with pytest.raises(GPUPlanningError) as captured:
        validate_gpu_execution_plan(plan)

    assert captured.value.reason is GPUPlanningErrorReason.INCOMPLETE_FUSION
    assert "data_region" in str(captured.value)


def test_flag_off_cover_is_stock_noop():
    context = DataContext()
    context.enable_gpu_operator_fusion = False
    input_buffer = InputDataBuffer(context, input_data=[])
    plan = PhysicalPlan(input_buffer, {}, context)

    validate_gpu_execution_plan(plan)
