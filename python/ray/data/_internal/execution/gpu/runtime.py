"""Backend-neutral actor-local runtime for an ordered GPU transform region."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Protocol

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
    def initialize(self) -> None:
        ...

    def apply(self, value: Any, task_context: TaskContext) -> Any:
        ...


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
    if kind in _GPU_TRANSFORM_RUNTIME_FACTORIES and not replace_existing:
        raise ValueError(f"GPU transform runtime already registered: {kind}")
    _GPU_TRANSFORM_RUNTIME_FACTORIES[kind] = factory


def _ensure_builtin_transform_runtimes_loaded() -> None:
    global _BUILTINS_LOAD_ATTEMPTED
    if _BUILTINS_LOAD_ATTEMPTED:
        return
    _BUILTINS_LOAD_ATTEMPTED = True
    # The TFM runtime module is optional while the core package is unit tested in
    # isolation.  Importing it here ensures registrations also happen in workers.
    try:
        importlib.import_module(
            "ray.data._internal.execution.gpu.transforms.runtime"
        )
    except ModuleNotFoundError as exc:
        if exc.name != "ray.data._internal.execution.gpu.transforms.runtime":
            raise


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
    _ensure_builtin_transform_runtimes_loaded()
    factory = _GPU_TRANSFORM_RUNTIME_FACTORIES.get(transform_spec.kind)
    if factory is not None:
        return factory(transform_spec, context)

    config = transform_spec.config
    create_runtime = getattr(config, "create_runtime", None)
    if create_runtime is not None:
        return create_runtime(context)
    if callable(config):
        return _CallableTransformRuntime(config)
    raise ValueError(
        f"No GPU transform runtime registered for kind {transform_spec.kind!r}"
    )


class GPURegionWorker:
    """Runs the ordered transform list in one actor-local task."""

    def __init__(self, operator_spec: GPUOperatorSpec):
        self._operator_spec = operator_spec
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
            create_gpu_transform_runtime(transform, self._context)
            for transform in self._operator_spec.transforms
        ]
        for runtime in self._transform_runtimes:
            initialize = getattr(runtime, "initialize", None)
            if initialize is not None:
                initialize()
        self._initialized = True

    def apply(
        self, blocks: Iterable[Block], task_context: TaskContext
    ) -> Iterable[Block]:
        self.initialize()
        value: Any = blocks
        for runtime in self._transform_runtimes:
            value = runtime.apply(value, task_context)
        return value


class GPURegionBlockTransform:
    """Cloudpickle-friendly bridge used by a stock BlockMapTransformFn."""

    def __init__(self, operator_spec: GPUOperatorSpec):
        self._worker = GPURegionWorker(operator_spec)

    def initialize(self) -> None:
        self._worker.initialize()

    def __call__(
        self, blocks: Iterable[Block], task_context: TaskContext
    ) -> Iterable[Block]:
        return self._worker.apply(blocks, task_context)
