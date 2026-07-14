"""CPU-only retry coverage for a complete required-fusion ActorPool stage.

These tests deliberately use test payload adapters and ``num_gpus=0``.  They
exercise the real ``ActorPoolGPUOperator`` and stock ``StreamingExecutor``
without requiring CUDA or cuDF.
"""

import os
import sys

import pandas as pd
import pytest

import ray
from ray import cloudpickle
from ray.data._internal.execution.gpu.boundary_adapters import (
    GPUBoundaryAdapter,
    GPUBoundaryDirection,
    register_gpu_boundary_adapter,
    unregister_gpu_boundary_adapter,
)
from ray.data._internal.execution.gpu.operator import create_gpu_operator
from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUOperatorSpec,
    GPUStageRetryPolicy,
    GPUTransformSpec,
    PayloadKind,
)
from ray.data._internal.execution.gpu.transforms._user_code import (
    raise_user_code_exception,
)
from ray.data._internal.execution.gpu.runtime import (
    GPUTransformRuntimeContext,
    register_gpu_transform_runtime,
    unregister_gpu_transform_runtime,
)
from ray.data._internal.execution.operators.actor_pool_gpu_operator import (
    ActorPoolGPUOperator,
)
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.streaming_executor import StreamingExecutor
from ray.data._internal.execution.util import make_ref_bundles
from ray.data.context import DataContext
from ray.data.exceptions import UserCodeException


_TEST_ADAPTER_KEY = "ray-data-gpu-test-block-stream-v1"


def _import_test_blocks(blocks, task_context):
    del task_context
    return list(blocks)


def _export_test_blocks(blocks, task_context):
    del task_context
    return iter(blocks)


@pytest.fixture
def gpu_test_block_adapters():
    # The test module itself is not installed in worker environments.  Serialize
    # the test-only adapters and runtimes by value, as user callables normally
    # are, rather than making the workers import this pytest module.
    module = sys.modules[__name__]
    cloudpickle.register_pickle_by_value(module)
    adapters = (
        GPUBoundaryAdapter(
            key=_TEST_ADAPTER_KEY,
            direction=GPUBoundaryDirection.IMPORT,
            external_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_payload=PayloadKind.CUDF_BATCH_STREAM,
            adapt=_import_test_blocks,
        ),
        GPUBoundaryAdapter(
            key=_TEST_ADAPTER_KEY,
            direction=GPUBoundaryDirection.EXPORT,
            external_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_payload=PayloadKind.CUDF_BATCH_STREAM,
            adapt=_export_test_blocks,
        ),
    )
    for adapter in adapters:
        register_gpu_boundary_adapter(adapter)
    try:
        yield
    finally:
        for adapter in adapters:
            unregister_gpu_boundary_adapter(
                key=adapter.key,
                direction=adapter.direction,
                external_payload=adapter.external_payload,
                native_payload=adapter.native_payload,
            )
        cloudpickle.unregister_pickle_by_value(module)


def _retry_policy(
    context: DataContext,
    *,
    actor_max_restarts: int = -1,
    actor_max_task_retries: int = -1,
) -> GPUStageRetryPolicy:
    return GPUStageRetryPolicy(
        actor_max_restarts=actor_max_restarts,
        actor_max_task_retries=actor_max_task_retries,
        actor_task_retry_on_errors=context.actor_task_retry_on_errors,
        actor_init_retry_on_errors=context.actor_init_retry_on_errors,
        actor_init_max_retries=context.actor_init_max_retries,
        retried_map_errors=context.retried_map_errors,
        max_map_retries=context.max_map_retries,
        retried_io_errors=tuple(context.retried_io_errors),
    )


def _complete_required_test_spec(
    config,
    context: DataContext,
    *,
    actor_max_restarts: int = -1,
    actor_max_task_retries: int = -1,
    transform_kind: str = "test_cpu_retry_runtime",
) -> GPUOperatorSpec:
    compatibility_key = GPUExecutionCompatibilityKey(
        min_workers=1,
        initial_workers=1,
        max_workers=1,
        num_cpus=1,
        num_gpus=0,
        retry_policy=_retry_policy(
            context,
            actor_max_restarts=actor_max_restarts,
            actor_max_task_retries=actor_max_task_retries,
        ),
    )
    spec = GPUOperatorSpec(
        spec_instance_id="test-gpu-stage",
        transforms=(
            GPUTransformSpec(
                transform_id="test-transform",
                kind=transform_kind,
                input_payload=PayloadKind.CUDF_BATCH_STREAM,
                output_payload=PayloadKind.CUDF_BATCH_STREAM,
                config=config,
            ),
        ),
        external_input_payload=PayloadKind.RAY_BLOCK_STREAM,
        external_output_payload=PayloadKind.RAY_BLOCK_STREAM,
        native_input_payload=PayloadKind.CUDF_BATCH_STREAM,
        native_output_payload=PayloadKind.CUDF_BATCH_STREAM,
        input_adapter_key=_TEST_ADAPTER_KEY,
        output_adapter_key=_TEST_ADAPTER_KEY,
        terminal_output_adapter_key=_TEST_ADAPTER_KEY,
        execution_requirements=GPUExecutionRequirements(
            compatibility_key=compatibility_key
        ),
        logical_provenance=("TestGPUStage",),
        fusion_group_id="test-required-gpu-stage",
        requires_complete_fusion=True,
        required_logical_provenance=("TestGPUStage",),
        uses_terminal_output_adapter=True,
    )
    return spec


def _execute_gpu_stage(config, input_ranges, context, **spec_kwargs):
    input_op = InputDataBuffer(context, make_ref_bundles(input_ranges))
    spec = _complete_required_test_spec(config, context, **spec_kwargs)
    operator = create_gpu_operator(spec, input_op, context)
    assert isinstance(operator, ActorPoolGPUOperator)
    assert operator.gpu_operator_spec.requires_complete_fusion
    assert (
        operator.gpu_operator_spec.logical_provenance
        == operator.gpu_operator_spec.required_logical_provenance
    )
    assert operator.gpu_operator_spec.uses_terminal_output_adapter
    assert operator._ray_remote_args["num_gpus"] == 0

    return _execute_gpu_operator(operator, context)


def _execute_gpu_operator(operator, context):
    executor = StreamingExecutor(context)
    values = []
    for bundle in executor.execute(operator):
        for block_ref in bundle.block_refs:
            values.extend(ray.get(block_ref)["id"].tolist())
    return values


class _OpaqueRegisteredConfig:
    def __init__(self, increment):
        self.increment = increment


class _DriverRegisteredRuntime:
    def __init__(self, increment):
        self._increment = increment

    def initialize(self):
        pass

    def apply(self, blocks, task_context):
        del task_context
        return [block.assign(id=block["id"] + self._increment) for block in blocks]


def _driver_registered_runtime_factory(transform_spec, runtime_context):
    del runtime_context
    return _DriverRegisteredRuntime(transform_spec.config.increment)


def test_driver_registered_opaque_runtime_is_serialized_to_actor(
    ray_start_regular_shared_2_cpus,
    restore_data_context,
    gpu_test_block_adapters,
):
    context = DataContext.get_current()
    kind = "test_driver_registered_opaque_runtime"
    spec = _complete_required_test_spec(
        _OpaqueRegisteredConfig(increment=10),
        context,
        transform_kind=kind,
    )
    input_op = InputDataBuffer(context, make_ref_bundles([[1, 2]]))

    register_gpu_transform_runtime(kind, _driver_registered_runtime_factory)
    try:
        operator = create_gpu_operator(spec, input_op, context)
    finally:
        # Remove it before StreamingExecutor launches the actor.  Success below
        # therefore requires the selected factory to travel with the operator.
        unregister_gpu_transform_runtime(
            kind,
            expected=_driver_registered_runtime_factory,
        )

    assert _execute_gpu_operator(operator, context) == [11, 12]


class _FlakyWrappedMap:
    def __init__(self, failures: int, message: str):
        self._failures = failures
        self._message = message
        self._attempt = 0

    def __call__(self, blocks, task_context):
        del blocks, task_context
        self._attempt += 1
        if self._attempt <= self._failures:
            raise_user_code_exception(
                ValueError(f"{self._message}; attempt={self._attempt}")
            )
        return [pd.DataFrame({"id": [self._attempt]})]


class _InitFlakyConfig:
    def __init__(self, failures: int):
        self._failures = failures

    def create_runtime(self, context: GPUTransformRuntimeContext):
        return _InitFlakyRuntime(context, self._failures)


class _InitFlakyRuntime:
    def __init__(self, context: GPUTransformRuntimeContext, failures: int):
        self._context = context
        self._failures = failures

    def initialize(self):
        attempt = self._context.state.get("init_attempt", 0) + 1
        self._context.state["init_attempt"] = attempt
        if attempt <= self._failures:
            raise_user_code_exception(ValueError(f"init-transient; attempt={attempt}"))

    def apply(self, blocks, task_context):
        del blocks, task_context
        return [pd.DataFrame({"id": [self._context.state["init_attempt"]]})]


class _FailFirstRange:
    def __call__(self, blocks, task_context):
        del task_context
        first_id = int(blocks[0]["id"].iloc[0])
        if first_id == 0:
            raise_user_code_exception(ValueError("failed whole range=0"))
        return blocks


class _KillActorOnce:
    def __init__(self, attempt_counter):
        self._attempt_counter = attempt_counter

    def __call__(self, blocks, task_context):
        del task_context
        if ray.get(self._attempt_counter.next.remote()) == 1:
            os._exit(23)
        return blocks


def test_gpu_stage_retried_map_errors_matches_wrapped_cause_and_attempt_limit(
    ray_start_regular_shared_2_cpus,
    restore_data_context,
    gpu_test_block_adapters,
):
    context = DataContext.get_current()
    context.retried_map_errors = ["wrapped-transient"]
    context.max_map_retries = 2

    # The public UserCodeException message doesn't contain the match string.
    # Reaching attempt 3 proves that stock map retry matched its ValueError cause.
    assert _execute_gpu_stage(
        _FlakyWrappedMap(failures=2, message="wrapped-transient"),
        [[0]],
        context,
    ) == [3]


@pytest.mark.parametrize(
    "match,expected_final_attempt",
    [
        (["persistent-transient"], 3),
        (["different-error"], 1),
    ],
)
def test_gpu_stage_retried_map_errors_exhausted_or_not_matched(
    ray_start_regular_shared_2_cpus,
    restore_data_context,
    gpu_test_block_adapters,
    match,
    expected_final_attempt,
):
    context = DataContext.get_current()
    context.retried_map_errors = match
    context.max_map_retries = 2

    with pytest.raises(
        ray.exceptions.RayTaskError, match=f"attempt={expected_final_attempt}"
    ) as exc_info:
        _execute_gpu_stage(
            _FlakyWrappedMap(failures=99, message="persistent-transient"),
            [[0]],
            context,
        )
    assert isinstance(exc_info.value, UserCodeException)


def test_gpu_stage_actor_initialization_uses_stock_retry_loop(
    ray_start_regular_shared_2_cpus,
    restore_data_context,
    gpu_test_block_adapters,
):
    context = DataContext.get_current()
    context.actor_init_retry_on_errors = True
    context.actor_init_max_retries = 2

    assert _execute_gpu_stage(_InitFlakyConfig(failures=2), [[0]], context) == [3]


def test_gpu_stage_actor_death_restarts_and_replays_the_input(
    ray_start_regular_shared_2_cpus,
    restore_data_context,
    gpu_test_block_adapters,
):
    @ray.remote(num_cpus=0)
    class AttemptCounter:
        def __init__(self):
            self._attempt = 0

        def next(self):
            self._attempt += 1
            return self._attempt

    context = DataContext.get_current()
    counter = AttemptCounter.remote()
    assert _execute_gpu_stage(
        _KillActorOnce(counter),
        [[0, 1]],
        context,
        actor_max_restarts=-1,
        actor_max_task_retries=-1,
    ) == [0, 1]
    assert ray.get(counter.next.remote()) == 3


@pytest.mark.parametrize("max_errored_blocks,should_raise", [(0, True), (1, False)])
def test_gpu_stage_max_errored_blocks_counts_a_whole_input_range(
    ray_start_regular_shared_2_cpus,
    restore_data_context,
    gpu_test_block_adapters,
    max_errored_blocks,
    should_raise,
):
    context = DataContext.get_current()
    context.max_errored_blocks = max_errored_blocks

    if should_raise:
        with pytest.raises(
            ray.exceptions.RayTaskError, match="failed whole range=0"
        ) as exc_info:
            _execute_gpu_stage(_FailFirstRange(), [[0, 1, 2], [3, 4]], context)
        assert isinstance(exc_info.value, UserCodeException)
    else:
        # One tolerated failure drops the entire three-row input/range, not one
        # stock-sized output block or only the row that raised.
        assert _execute_gpu_stage(_FailFirstRange(), [[0, 1, 2], [3, 4]], context) == [
            3,
            4,
        ]
