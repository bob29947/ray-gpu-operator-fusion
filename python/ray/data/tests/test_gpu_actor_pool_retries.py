"""CPU-only retry coverage for a closed ActorPool GPU region.

These tests use ordinary test transform runtimes and ``num_gpus=0``.  They
exercise the real ``ActorPoolGPUOperator`` and stock ``StreamingExecutor``
without requiring CUDA or cuDF.
"""

import os
import sys
import time

import pandas as pd
import pytest

import ray
from ray import cloudpickle
from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUOperatorSpec,
    GPUStageRetryPolicy,
    GPUTransformSpec,
    PayloadKind,
    RAY_BLOCK_STREAM,
    RuntimeKey,
)
from ray.data._internal.execution.gpu.transforms._user_code import (
    raise_user_code_exception,
)
from ray.data._internal.execution.gpu.runtime import (
    GPUTransformRuntimeContext,
    TransformRuntimeRegistration,
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


_TEST_IMPORT_KIND = "test_import_blocks"
_TEST_NATIVE = PayloadKind("test_retry_native")
_TEST_EXPORT_KIND = "test_export_blocks"
_DEFAULT_RUNTIME_KIND = "test_cpu_retry_runtime"


class _CallableRuntime:
    def __init__(self, function):
        self._function = function

    def initialize(self):
        pass

    def apply(self, value, task_context):
        return self._function(value, task_context)


def _import_test_blocks(blocks, task_context):
    del task_context
    return list(blocks)


def _export_test_blocks(blocks, task_context):
    del task_context
    return iter(blocks)


def _default_runtime_factory(transform_spec, runtime_context):
    config = transform_spec.config
    create_runtime = getattr(config, "create_runtime", None)
    if create_runtime is not None:
        return create_runtime(runtime_context)
    return _CallableRuntime(config)


def _callable_config(config):
    return callable(config) or callable(getattr(config, "create_runtime", None))


def _test_runtime_key(kind, input_payload, output_payload):
    return RuntimeKey(
        "ray-data-gpu-v1",
        kind,
        input_payload,
        output_payload,
    )


@pytest.fixture
def gpu_test_transform_runtimes():
    # The test module itself is not installed in worker environments.  Serialize
    # the test-only transforms and runtimes by value, as user callables normally
    # are, rather than making the workers import this pytest module.
    module = sys.modules[__name__]
    cloudpickle.register_pickle_by_value(module)
    registrations = (
        (
            _test_runtime_key(
                _TEST_IMPORT_KIND,
                RAY_BLOCK_STREAM,
                _TEST_NATIVE,
            ),
            TransformRuntimeRegistration(
                lambda config: config is None,
                lambda transform, context: _CallableRuntime(_import_test_blocks),
            ),
        ),
        (
            _test_runtime_key(
                _DEFAULT_RUNTIME_KIND,
                _TEST_NATIVE,
                _TEST_NATIVE,
            ),
            TransformRuntimeRegistration(
                _callable_config,
                _default_runtime_factory,
            ),
        ),
        (
            _test_runtime_key(
                _TEST_EXPORT_KIND,
                _TEST_NATIVE,
                RAY_BLOCK_STREAM,
            ),
            TransformRuntimeRegistration(
                lambda config: config is None,
                lambda transform, context: _CallableRuntime(_export_test_blocks),
            ),
        ),
    )
    for key, registration in registrations:
        register_gpu_transform_runtime(key, registration)
    try:
        yield
    finally:
        for key, registration in registrations:
            unregister_gpu_transform_runtime(key, expected=registration)
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


def _closed_test_spec(
    config,
    context: DataContext,
    *,
    actor_max_restarts: int = -1,
    actor_max_task_retries: int = -1,
    transform_kind: str = _DEFAULT_RUNTIME_KIND,
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
    return GPUOperatorSpec(
        transforms=(
            GPUTransformSpec(
                kind=_TEST_IMPORT_KIND,
                input_payload=RAY_BLOCK_STREAM,
                output_payload=_TEST_NATIVE,
            ),
            GPUTransformSpec(
                kind=transform_kind,
                input_payload=_TEST_NATIVE,
                output_payload=_TEST_NATIVE,
                config=config,
            ),
            GPUTransformSpec(
                kind=_TEST_EXPORT_KIND,
                input_payload=_TEST_NATIVE,
                output_payload=RAY_BLOCK_STREAM,
            ),
        ),
        execution_requirements=GPUExecutionRequirements(
            compatibility_key=compatibility_key
        ),
    )


def _execute_gpu_stage(config, input_ranges, context, **spec_kwargs):
    input_op = InputDataBuffer(context, make_ref_bundles(input_ranges))
    spec = _closed_test_spec(config, context, **spec_kwargs)
    operator = ActorPoolGPUOperator(spec, input_op, context)
    assert isinstance(operator, ActorPoolGPUOperator)
    assert operator.gpu_operator_spec.input_payload == RAY_BLOCK_STREAM
    assert operator.gpu_operator_spec.output_payload == RAY_BLOCK_STREAM
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
    gpu_test_transform_runtimes,
):
    context = DataContext.get_current()
    kind = "test_driver_registered_opaque_runtime"
    spec = _closed_test_spec(
        _OpaqueRegisteredConfig(increment=10),
        context,
        transform_kind=kind,
    )
    input_op = InputDataBuffer(context, make_ref_bundles([[1, 2]]))
    runtime_key = _test_runtime_key(
        kind,
        _TEST_NATIVE,
        _TEST_NATIVE,
    )
    registration = TransformRuntimeRegistration(
        lambda config: isinstance(config, _OpaqueRegisteredConfig),
        _driver_registered_runtime_factory,
    )
    register_gpu_transform_runtime(runtime_key, registration)
    try:
        operator = ActorPoolGPUOperator(spec, input_op, context)
    finally:
        # Remove it before StreamingExecutor launches the actor.  Success below
        # therefore requires the selected factory to travel with the operator.
        unregister_gpu_transform_runtime(
            runtime_key,
            expected=registration,
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


class _YieldPauseKillActorOnce:
    def __init__(self, control):
        self._control = control

    def __call__(self, blocks, task_context):
        del task_context
        block = blocks[0]
        yield block.iloc[:1].copy()
        ray.get(self._control.prefix_yielded.remote())
        while not ray.get(self._control.is_released.remote()):
            time.sleep(0.01)
        if ray.get(self._control.next_attempt.remote()) == 1:
            os._exit(23)
        yield block.iloc[1:].copy()


def test_gpu_stage_retried_map_errors_matches_wrapped_cause_and_attempt_limit(
    ray_start_regular_shared_2_cpus,
    restore_data_context,
    gpu_test_transform_runtimes,
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
    gpu_test_transform_runtimes,
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
    gpu_test_transform_runtimes,
):
    context = DataContext.get_current()
    context.actor_init_retry_on_errors = True
    context.actor_init_max_retries = 2

    assert _execute_gpu_stage(_InitFlakyConfig(failures=2), [[0]], context) == [3]


def test_gpu_stage_actor_death_restarts_and_replays_the_input(
    ray_start_regular_shared_2_cpus,
    restore_data_context,
    gpu_test_transform_runtimes,
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


def test_gpu_stage_actor_death_after_streamed_prefix_recovers_without_duplicates(
    ray_start_regular_shared_2_cpus,
    restore_data_context,
    gpu_test_transform_runtimes,
):
    """Recoverable death replays a streamed task without missing or duplicate rows.

    If retries exhaust after an observed prefix, stock streaming cannot retract it;
    V1 deliberately keeps that case as a limitation instead of adding a commit barrier.
    """

    @ray.remote(num_cpus=0)
    class Control:
        def __init__(self):
            self._prefixes = 0
            self._released = False
            self._attempts = 0

        def prefix_yielded(self):
            self._prefixes += 1

        def prefix_count(self):
            return self._prefixes

        def release(self):
            self._released = True

        def is_released(self):
            return self._released

        def next_attempt(self):
            self._attempts += 1
            return self._attempts

        def attempt_count(self):
            return self._attempts

    context = DataContext.get_current()
    control = Control.remote()
    input_op = InputDataBuffer(context, make_ref_bundles([[0, 1]]))
    spec = _closed_test_spec(
        _YieldPauseKillActorOnce(control),
        context,
        actor_max_restarts=-1,
        actor_max_task_retries=-1,
    )
    operator = ActorPoolGPUOperator(spec, input_op, context)
    executor = StreamingExecutor(context)
    output = iter(executor.execute(operator))

    first_bundle = next(output)
    observed = []
    for block_ref in first_bundle.block_refs:
        observed.extend(ray.get(block_ref)["id"].tolist())
    assert observed == [0]

    deadline = time.monotonic() + 10
    while ray.get(control.prefix_count.remote()) < 1:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    time.sleep(0.2)
    ray.get(control.release.remote())

    for bundle in output:
        for block_ref in bundle.block_refs:
            observed.extend(ray.get(block_ref)["id"].tolist())

    assert observed == [0, 1]
    assert ray.get(control.prefix_count.remote()) == 2
    assert ray.get(control.attempt_count.remote()) == 2


@pytest.mark.parametrize("max_errored_blocks,should_raise", [(0, True), (1, False)])
def test_gpu_stage_max_errored_blocks_counts_a_whole_input_range(
    ray_start_regular_shared_2_cpus,
    restore_data_context,
    gpu_test_transform_runtimes,
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
