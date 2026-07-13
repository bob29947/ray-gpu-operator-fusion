"""Backend-neutral actor-local runtime for an ordered GPU transform region."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

from ray.data._internal.execution.gpu.boundary_adapters import (
    GPUBoundaryDirection,
    require_gpu_boundary_adapter,
)
from ray.data._internal.execution.gpu.operator_spec import (
    GPUOperatorSpec,
    GPUTransformSpec,
)
from ray.data._internal.execution.interfaces import TaskContext
from ray.data.block import Block


@dataclass
class GPUTransformRuntimeContext:
    """Mutable state shared by all transform runtimes in one actor process."""

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
_GPU_TRANSFORM_RUNTIME_FACTORIES: Dict[str, GPUTransformRuntimeFactory] = {}
_BUILTINS_LOAD_ATTEMPTED = False


def register_gpu_transform_runtime(
    kind: str,
    factory: GPUTransformRuntimeFactory,
    *,
    replace_existing: bool = False,
) -> None:
    if not kind:
        raise ValueError("GPU transform runtime kind must be nonempty")
    if not callable(factory):
        raise TypeError("GPU transform runtime factory must be callable")
    if kind in _GPU_TRANSFORM_RUNTIME_FACTORIES and not replace_existing:
        raise ValueError(f"GPU transform runtime already registered: {kind}")
    _GPU_TRANSFORM_RUNTIME_FACTORIES[kind] = factory


def unregister_gpu_transform_runtime(
    kind: str,
    *,
    expected: GPUTransformRuntimeFactory | None = None,
) -> GPUTransformRuntimeFactory:
    """Remove one runtime implementation, with optional teardown guarding."""

    try:
        factory = _GPU_TRANSFORM_RUNTIME_FACTORIES[kind]
    except KeyError as exc:
        raise KeyError(f"GPU transform runtime is not registered: {kind}") from exc
    if expected is not None and factory is not expected:
        raise ValueError(
            f"GPU transform runtime {kind!r} is not the expected implementation"
        )
    del _GPU_TRANSFORM_RUNTIME_FACTORIES[kind]
    return factory


def _ensure_builtin_transform_runtimes_loaded() -> None:
    global _BUILTINS_LOAD_ATTEMPTED
    if _BUILTINS_LOAD_ATTEMPTED:
        return
    _BUILTINS_LOAD_ATTEMPTED = True
    # The TFM runtime module is optional while the core package is unit tested in
    # isolation.  Importing it here ensures registrations also happen in workers.
    try:
        importlib.import_module("ray.data._internal.execution.gpu.transforms.runtime")
    except ModuleNotFoundError as exc:
        if exc.name != "ray.data._internal.execution.gpu.transforms.runtime":
            raise


def supports_gpu_transform_runtime(transform_spec: GPUTransformSpec) -> bool:
    """Return whether a transform can be constructed, without constructing it."""

    try:
        resolve_gpu_transform_runtime_factory(transform_spec)
    except ValueError:
        return False
    return True


def resolve_gpu_transform_runtime_factory(
    transform_spec: GPUTransformSpec,
) -> Optional[GPUTransformRuntimeFactory]:
    """Resolve a registry factory without constructing its runtime.

    ``None`` means the implementation is carried by ``config`` itself through
    ``create_runtime`` or ``__call__``.  A returned registry callable is stored
    in the physical transform so workers never need the driver's registry.
    """

    _ensure_builtin_transform_runtimes_loaded()
    factory = _GPU_TRANSFORM_RUNTIME_FACTORIES.get(transform_spec.kind)
    if factory is not None:
        return factory
    config = transform_spec.config
    if callable(getattr(config, "create_runtime", None)) or callable(config):
        return None
    raise ValueError(
        f"No GPU transform runtime registered for kind {transform_spec.kind!r}"
    )


class _CallableTransformRuntime:
    def __init__(self, fn: Callable[[Any, TaskContext], Any]):
        self._fn = fn

    def initialize(self) -> None:
        pass

    def apply(self, value: Any, task_context: TaskContext) -> Any:
        return self._fn(value, task_context)


def create_gpu_transform_runtime(
    transform_spec: GPUTransformSpec,
    context: GPUTransformRuntimeContext,
) -> GPUTransformRuntime:
    factory = resolve_gpu_transform_runtime_factory(transform_spec)
    return _create_gpu_transform_runtime_from_resolved(
        transform_spec,
        context,
        factory,
    )


def _create_gpu_transform_runtime_from_resolved(
    transform_spec: GPUTransformSpec,
    context: GPUTransformRuntimeContext,
    factory: Optional[GPUTransformRuntimeFactory],
) -> GPUTransformRuntime:
    if factory is not None:
        return factory(transform_spec, context)

    config = transform_spec.config
    create_runtime = getattr(config, "create_runtime", None)
    if callable(create_runtime):
        return create_runtime(context)
    if callable(config):
        return _CallableTransformRuntime(config)
    raise ValueError(
        f"No GPU transform runtime registered for kind {transform_spec.kind!r}"
    )


class GPURegionWorker:
    """Runs the ordered transform list in one actor-local task."""

    def __init__(
        self,
        operator_spec: GPUOperatorSpec,
        resolved_runtime_factories: Optional[
            Tuple[Optional[GPUTransformRuntimeFactory], ...]
        ] = None,
    ):
        self._operator_spec = operator_spec
        if resolved_runtime_factories is None:
            resolved_runtime_factories = tuple(
                resolve_gpu_transform_runtime_factory(transform)
                for transform in operator_spec.transforms
            )
        else:
            resolved_runtime_factories = tuple(resolved_runtime_factories)
        if len(resolved_runtime_factories) != len(operator_spec.transforms):
            raise ValueError(
                "resolved GPU runtime factories must align with transform specs"
            )
        self._resolved_runtime_factories = resolved_runtime_factories
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
            _create_gpu_transform_runtime_from_resolved(
                transform,
                self._context,
                factory,
            )
            for transform, factory in zip(
                self._operator_spec.transforms,
                self._resolved_runtime_factories,
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
    """Cloudpickle-friendly bridge used by a stock BlockMapTransformFn."""

    def __init__(self, operator_spec: GPUOperatorSpec):
        self._operator_spec = operator_spec
        self._worker = GPURegionWorker(operator_spec)
        self._input_adapter = require_gpu_boundary_adapter(
            key=operator_spec.input_adapter_key,
            direction=GPUBoundaryDirection.IMPORT,
            external_payload=operator_spec.external_input_payload,
            native_payload=operator_spec.native_input_payload,
        )
        self._output_adapter = require_gpu_boundary_adapter(
            key=operator_spec.active_output_adapter_key,
            direction=GPUBoundaryDirection.EXPORT,
            external_payload=operator_spec.external_output_payload,
            native_payload=operator_spec.native_output_payload,
        )

    def initialize(self) -> None:
        self._worker.initialize()

    def __call__(
        self, blocks: Iterable[Block], task_context: TaskContext
    ) -> Iterable[Block]:
        native_value = self._input_adapter.adapt(blocks, task_context)
        output_value = self._worker.apply(native_value, task_context)
        return self._output_adapter.adapt(output_value, task_context)
