"""Payload-exact runtime registry and actor-local GPU region execution."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

from ray.data._internal.execution.gpu.operator_spec import (
    GPUOperatorSpec,
    GPUTransformSpec,
    RAY_BLOCK_STREAM,
    RuntimeKey,
)
from ray.data._internal.execution.interfaces import TaskContext
from ray.data.block import Block


@dataclass
class GPUTransformRuntimeContext:
    operator_spec: GPUOperatorSpec
    state: Dict[str, Any] = field(default_factory=dict)

    def get_or_create(self, key: str, factory: Callable[[], Any]) -> Any:
        if key not in self.state:
            self.state[key] = factory()
        return self.state[key]


class GPUTransformRuntime(Protocol):
    def initialize(self) -> None: ...

    def apply(self, value: Any, task_context: TaskContext) -> Any: ...


GPUTransformRuntimeFactory = Callable[
    [GPUTransformSpec, GPUTransformRuntimeContext], GPUTransformRuntime
]
GPUTransformConfigValidator = Callable[[Any], bool]


@dataclass(frozen=True)
class TransformRuntimeRegistration:
    validate_config: GPUTransformConfigValidator = field(compare=False, repr=False)
    factory: GPUTransformRuntimeFactory = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(self.validate_config):
            raise TypeError("GPU transform config validator must be callable")
        if not callable(self.factory):
            raise TypeError("GPU transform runtime factory must be callable")


_GPU_TRANSFORM_RUNTIMES: Dict[RuntimeKey, TransformRuntimeRegistration] = {}
_BUILTINS_LOAD_ATTEMPTED = False


def register_gpu_transform_runtime(
    key: RuntimeKey,
    registration: TransformRuntimeRegistration,
    *,
    replace_existing: bool = False,
) -> None:
    _ensure_builtin_transform_runtimes_loaded()
    if not isinstance(key, RuntimeKey):
        raise TypeError("GPU transform runtime key must be a RuntimeKey")
    if not isinstance(registration, TransformRuntimeRegistration):
        raise TypeError(
            "GPU transform runtime registration must be TransformRuntimeRegistration"
        )
    if key in _GPU_TRANSFORM_RUNTIMES and not replace_existing:
        raise ValueError(f"GPU transform runtime already registered: {key!r}")
    _GPU_TRANSFORM_RUNTIMES[key] = registration


def unregister_gpu_transform_runtime(
    key: RuntimeKey,
    *,
    expected: TransformRuntimeRegistration | None = None,
) -> TransformRuntimeRegistration:
    try:
        registration = _GPU_TRANSFORM_RUNTIMES[key]
    except KeyError as error:
        raise KeyError(f"GPU transform runtime is not registered: {key!r}") from error
    if expected is not None and registration is not expected:
        raise ValueError(
            f"GPU transform runtime {key!r} is not the expected implementation"
        )
    del _GPU_TRANSFORM_RUNTIMES[key]
    return registration


def _ensure_builtin_transform_runtimes_loaded() -> None:
    global _BUILTINS_LOAD_ATTEMPTED
    if _BUILTINS_LOAD_ATTEMPTED:
        return
    _BUILTINS_LOAD_ATTEMPTED = True
    try:
        importlib.import_module("ray.data._internal.execution.gpu.transforms.runtime")
    except ModuleNotFoundError as error:
        if error.name != "ray.data._internal.execution.gpu.transforms.runtime":
            raise


def get_gpu_transform_runtime_registration(
    key: RuntimeKey,
) -> TransformRuntimeRegistration | None:
    _ensure_builtin_transform_runtimes_loaded()
    return _GPU_TRANSFORM_RUNTIMES.get(key)


def resolve_gpu_transform_runtime_registration(
    transform_spec: GPUTransformSpec,
) -> TransformRuntimeRegistration:
    registration = get_gpu_transform_runtime_registration(transform_spec.runtime_key)
    if registration is None:
        raise ValueError(
            "No GPU transform runtime registered for " f"{transform_spec.runtime_key!r}"
        )
    try:
        valid = registration.validate_config(transform_spec.config)
    except Exception as error:
        raise ValueError(
            f"Invalid config for GPU transform {transform_spec.runtime_key!r}: "
            f"{error}"
        ) from error
    if valid is not True:
        raise ValueError(
            f"Invalid config for GPU transform {transform_spec.runtime_key!r}"
        )
    return registration


def supports_gpu_transform_runtime(transform_spec: GPUTransformSpec) -> bool:
    try:
        resolve_gpu_transform_runtime_registration(transform_spec)
    except ValueError:
        return False
    return True


def create_gpu_transform_runtime(
    transform_spec: GPUTransformSpec,
    context: GPUTransformRuntimeContext,
) -> GPUTransformRuntime:
    registration = resolve_gpu_transform_runtime_registration(transform_spec)
    return registration.factory(transform_spec, context)


class GPURegionWorker:
    def __init__(
        self,
        operator_spec: GPUOperatorSpec,
        resolved_runtime_registrations: Optional[
            Tuple[TransformRuntimeRegistration, ...]
        ] = None,
    ):
        self._operator_spec = operator_spec
        if resolved_runtime_registrations is None:
            resolved_runtime_registrations = tuple(
                resolve_gpu_transform_runtime_registration(transform)
                for transform in operator_spec.transforms
            )
        else:
            resolved_runtime_registrations = tuple(resolved_runtime_registrations)
        if len(resolved_runtime_registrations) != len(operator_spec.transforms):
            raise ValueError(
                "resolved GPU runtime registrations must align with transform specs"
            )
        self._resolved_runtime_registrations = resolved_runtime_registrations
        self._context = GPUTransformRuntimeContext(operator_spec)
        self._transform_runtimes: List[GPUTransformRuntime] = []
        self._initialized = False

    @property
    def runtime_context(self) -> GPUTransformRuntimeContext:
        return self._context

    def initialize(self) -> None:
        if self._initialized:
            return
        self._transform_runtimes = [
            registration.factory(transform, self._context)
            for transform, registration in zip(
                self._operator_spec.transforms,
                self._resolved_runtime_registrations,
            )
        ]
        for runtime in self._transform_runtimes:
            initialize = getattr(runtime, "initialize", None)
            if initialize is not None:
                initialize()
        self._initialized = True

    def apply(self, value: Any, task_context: TaskContext) -> Any:
        self.initialize()
        for runtime in self._transform_runtimes:
            value = runtime.apply(value, task_context)
        return value


class GPURegionBlockTransform:
    def __init__(
        self,
        operator_spec: GPUOperatorSpec,
        resolved_runtime_registrations: Optional[
            Tuple[TransformRuntimeRegistration, ...]
        ] = None,
    ):
        if (
            operator_spec.input_payload != RAY_BLOCK_STREAM
            or operator_spec.output_payload != RAY_BLOCK_STREAM
        ):
            raise ValueError(
                "GPURegionBlockTransform requires Ray-block input and output transforms"
            )
        self._operator_spec = operator_spec
        self._worker = GPURegionWorker(
            operator_spec,
            resolved_runtime_registrations=resolved_runtime_registrations,
        )

    def initialize(self) -> None:
        self._worker.initialize()

    def __call__(
        self,
        blocks: Iterable[Block],
        task_context: TaskContext,
    ) -> Iterable[Block]:
        return self._worker.apply(blocks, task_context)
