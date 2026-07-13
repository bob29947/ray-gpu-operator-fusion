"""Disposable ActorPoolMapOperator adapter for declarative GPU specifications."""

from __future__ import annotations

from typing import Dict, Optional

from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.execution.gpu.operator import (
    GPUOperator,
    GPUOperatorCreationOptions,
    GPUOperatorFactory,
    register_gpu_operator_factory,
)
from ray.data._internal.execution.gpu.boundary_adapters import (
    GPUBoundaryDirection,
    get_gpu_boundary_adapter,
)
from ray.data._internal.execution.gpu.operator_spec import (
    GPUOperatorSpec,
    GPUStageRetryPolicy,
)
from ray.data._internal.execution.gpu.runtime import (
    GPURegionBlockTransform,
    supports_gpu_transform_runtime,
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


class ActorPoolGPUOperator(ActorPoolMapOperator, GPUOperator):
    """V1 physical adapter used for both modular and fused GPU nodes."""

    def __init__(
        self,
        operator_spec: GPUOperatorSpec,
        input_op: PhysicalOperator,
        data_context: DataContext,
        creation_options: Optional[GPUOperatorCreationOptions] = None,
    ):
        if creation_options is None:
            creation_options = GPUOperatorCreationOptions()
        _validate_context_retry_policy(operator_spec, data_context)
        self._gpu_operator_spec = operator_spec
        self._gpu_creation_options = creation_options

        transform = GPURegionBlockTransform(operator_spec)
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
            name=creation_options.name
            or "GPU["
            + ",".join(transform.kind for transform in operator_spec.transforms)
            + "]",
            min_rows_per_bundle=creation_options.min_rows_per_bundle,
            supports_fusion=False,
            ray_remote_args=_ray_remote_args(operator_spec),
            target_max_block_size_override=(
                creation_options.target_max_block_size_override
            ),
        )

    @property
    def gpu_operator_spec(self) -> GPUOperatorSpec:
        return self._gpu_operator_spec

    @property
    def execution_backend_name(self) -> str:
        return self._gpu_operator_spec.execution_requirements.backend_name

    @property
    def gpu_creation_options(self) -> GPUOperatorCreationOptions:
        return self._gpu_creation_options


def _ray_remote_args(spec: GPUOperatorSpec) -> Dict[str, object]:
    key = spec.compatibility_key
    retry_policy = key.retry_policy
    args: Dict[str, object] = {
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
    """Keep stock ActorPool retry behavior identical to the declared key."""

    expected = spec.compatibility_key.retry_policy
    actual = GPUStageRetryPolicy(
        # Actor creation/task limits are carried by Ray remote args rather than
        # DataContext; retain the declared values in this comparison.
        actor_max_restarts=expected.actor_max_restarts,
        actor_max_task_retries=expected.actor_max_task_retries,
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


class ActorPoolGPUOperatorFactory(GPUOperatorFactory):
    def supports(self, spec: GPUOperatorSpec) -> bool:
        return (
            spec.execution_requirements.backend_name == "actor_pool"
            and spec.compatibility_key.worker_lifecycle == "actor_pool"
            and all(
                supports_gpu_transform_runtime(transform)
                for transform in spec.transforms
            )
            and get_gpu_boundary_adapter(
                key=spec.input_adapter_key,
                direction=GPUBoundaryDirection.IMPORT,
                external_payload=spec.external_input_payload,
                native_payload=spec.native_input_payload,
            )
            is not None
            and get_gpu_boundary_adapter(
                key=spec.active_output_adapter_key,
                direction=GPUBoundaryDirection.EXPORT,
                external_payload=spec.external_output_payload,
                native_payload=spec.native_output_payload,
            )
            is not None
        )

    def create(
        self,
        spec: GPUOperatorSpec,
        input_op: PhysicalOperator,
        data_context: DataContext,
        creation_options: Optional[GPUOperatorCreationOptions] = None,
    ) -> PhysicalOperator:
        return ActorPoolGPUOperator(
            spec, input_op, data_context, creation_options=creation_options
        )


register_gpu_operator_factory(
    "actor_pool", ActorPoolGPUOperatorFactory(), replace_existing=True
)
