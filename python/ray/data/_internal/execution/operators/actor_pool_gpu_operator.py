"""Actor-pool backend for compiled declarative GPU regions."""

from __future__ import annotations

from dataclasses import replace

from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.execution.gpu.composition import GPUCompositionError
from ray.data._internal.execution.gpu.operator import (
    ExecutableGPUOperator,
    GPUBackendRefusal,
    GPUOperatorCreationOptions,
    _gpu_name,
    _register_builtin_gpu_operator_factory,
)
from ray.data._internal.execution.gpu.operator_spec import (
    GPUOperatorSpec,
    GPUTransformSpec,
    RANGE_DATA,
    RANGE_WORK,
    RAY_BLOCK_STREAM,
)
from ray.data._internal.execution.gpu.region_compiler import compile_region
from ray.data._internal.execution.gpu.runtime import (
    GPURegionBlockTransform,
    TransformRuntimeRegistration,
    resolve_gpu_transform_runtime_registration,
    supports_gpu_transform_runtime,
)
from ray.data._internal.execution.gpu.transforms.boundary import (
    EXPORT_RANGE_DATA_KIND,
    IMPORT_RANGE_WORK_KIND,
)
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.actor_pool_map_operator import (
    ActorPoolMapOperator,
)
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    MapTransformer,
)
from ray.data.context import DataContext


_BACKEND = "actor_pool"
_RuntimeRegistrations = tuple[TransformRuntimeRegistration, ...]


class ActorPoolGPUOperator(ActorPoolMapOperator, ExecutableGPUOperator):
    """One executable actor-pool shell around a closed GPU transform program."""

    def __init__(
        self,
        operator_spec: GPUOperatorSpec,
        input_op: PhysicalOperator,
        data_context: DataContext,
        creation_options: GPUOperatorCreationOptions | None = None,
        resolved_runtime_registrations: _RuntimeRegistrations | None = None,
    ):
        creation_options = creation_options or GPUOperatorCreationOptions()
        if not operator_spec.execution_requirements.is_resolved:
            raise ValueError("actor-pool GPU execution requirements are unresolved")
        if (
            operator_spec.input_payload != RAY_BLOCK_STREAM
            or operator_spec.output_payload != RAY_BLOCK_STREAM
        ):
            raise ValueError("actor-pool GPU operator requires Ray-block endpoints")
        _validate_context_retry_policy(operator_spec, data_context)
        self._gpu_operator_spec = operator_spec
        self._gpu_creation_options = creation_options
        resolved = resolved_runtime_registrations or tuple(
            resolve_gpu_transform_runtime_registration(transform)
            for transform in operator_spec.transforms
        )
        transform = GPURegionBlockTransform(operator_spec, resolved)
        transformer = MapTransformer(
            [BlockMapTransformFn(transform, disable_block_shaping=True)],
            init_fn=transform.initialize,
        )
        key = operator_spec.compatibility_key
        compute_strategy = ActorPoolStrategy(
            min_size=key.min_workers,
            max_size=key.max_workers,
            initial_size=key.initial_workers,
            max_tasks_in_flight_per_actor=key.max_tasks_in_flight_per_worker,
            enable_true_multi_threading=False,
        )
        super().__init__(
            map_transformer=transformer,
            input_op=input_op,
            data_context=data_context,
            compute_strategy=compute_strategy,
            name=creation_options.name or _gpu_name("GPU", operator_spec),
            min_rows_per_bundle=creation_options.min_rows_per_bundle,
            supports_fusion=False,
            ray_remote_args=_ray_remote_args(operator_spec),
            target_max_block_size_override=(
                creation_options.target_max_block_size_override
            ),
        )


def _ray_remote_args(spec: GPUOperatorSpec) -> dict[str, object]:
    key = spec.compatibility_key
    retry_policy = key.retry_policy
    args: dict[str, object] = {
        "num_cpus": key.num_cpus,
        "num_gpus": key.num_gpus,
        "max_concurrency": 1,
        "max_restarts": retry_policy.actor_max_restarts,
        "max_task_retries": retry_policy.actor_max_task_retries,
    }
    if key.memory is not None:
        args["memory"] = key.memory
    if key.custom_resources:
        args["resources"] = dict(key.custom_resources)
    if key.accelerator_type is not None:
        args["accelerator_type"] = key.accelerator_type
    if key.scheduling_strategy is not None:
        args["scheduling_strategy"] = key.scheduling_strategy
    if key.label_selector:
        args["label_selector"] = dict(key.label_selector)
    if key.runtime_env_serialized is not None:
        from ray.runtime_env import RuntimeEnv

        args["runtime_env"] = RuntimeEnv.deserialize(key.runtime_env_serialized)
    return args


def _validate_context_retry_policy(
    spec: GPUOperatorSpec, data_context: DataContext
) -> None:
    expected = spec.compatibility_key.retry_policy
    actual = replace(
        expected,
        actor_task_retry_on_errors=data_context.actor_task_retry_on_errors,
        actor_init_retry_on_errors=data_context.actor_init_retry_on_errors,
        actor_init_max_retries=data_context.actor_init_max_retries,
        retried_map_errors=data_context.retried_map_errors,
        max_map_retries=data_context.max_map_retries,
        retried_io_errors=tuple(data_context.retried_io_errors),
    )
    if actual != expected:
        raise ValueError(
            "GPU stage retry policy does not match the execution DataContext: "
            f"declared={expected!r}, effective={actual!r}"
        )


class ActorPoolGPUOperatorFactory:
    def prepare(self, core_spec: GPUOperatorSpec) -> GPUOperatorSpec:
        if not core_spec.execution_requirements.is_resolved:
            raise GPUBackendRefusal(
                _BACKEND, "compiled execution requirements are unresolved"
            )
        if core_spec.input_payload != RANGE_WORK:
            raise GPUBackendRefusal(
                _BACKEND,
                "no Ray-block import transform is registered for "
                f"{core_spec.input_payload.key!r}",
            )
        if core_spec.output_payload != RANGE_DATA:
            raise GPUBackendRefusal(
                _BACKEND,
                "no Ray-block export transform is registered for "
                f"{core_spec.output_payload.key!r}",
            )
        ingress = _boundary_spec(
            IMPORT_RANGE_WORK_KIND, RAY_BLOCK_STREAM, RANGE_WORK, core_spec
        )
        egress = _boundary_spec(
            EXPORT_RANGE_DATA_KIND, RANGE_DATA, RAY_BLOCK_STREAM, core_spec
        )
        try:
            return compile_region(core_spec, ingress, egress)
        except GPUCompositionError as error:
            raise GPUBackendRefusal(_BACKEND, str(error)) from error

    def supports(self, spec: GPUOperatorSpec) -> bool:
        return (
            spec.execution_requirements.is_resolved
            and spec.execution_requirements.backend_name == _BACKEND
            and spec.compatibility_key.worker_lifecycle == _BACKEND
            and spec.input_payload == RAY_BLOCK_STREAM
            and spec.output_payload == RAY_BLOCK_STREAM
            and all(
                supports_gpu_transform_runtime(transform)
                for transform in spec.transforms
            )
        )

    def create(
        self,
        spec: GPUOperatorSpec,
        input_op: PhysicalOperator,
        data_context: DataContext,
        creation_options: GPUOperatorCreationOptions | None = None,
    ) -> PhysicalOperator:
        resolved = tuple(
            resolve_gpu_transform_runtime_registration(transform)
            for transform in spec.transforms
        )
        return ActorPoolGPUOperator(
            spec,
            input_op,
            data_context,
            creation_options=creation_options,
            resolved_runtime_registrations=resolved,
        )

    def attests(
        self, operator: PhysicalOperator, compiled_spec: GPUOperatorSpec
    ) -> bool:
        return (
            isinstance(operator, ActorPoolGPUOperator)
            and operator.gpu_operator_spec is compiled_spec
            and operator.execution_backend_name == _BACKEND
            and compiled_spec.input_payload == RAY_BLOCK_STREAM
            and compiled_spec.output_payload == RAY_BLOCK_STREAM
        )


def _boundary_spec(
    kind: str,
    input_payload: object,
    output_payload: object,
    core_spec: GPUOperatorSpec,
) -> GPUOperatorSpec:
    return GPUOperatorSpec(
        (GPUTransformSpec(kind, input_payload, output_payload),),
        core_spec.execution_requirements,
    )


_register_builtin_gpu_operator_factory(_BACKEND, ActorPoolGPUOperatorFactory())
