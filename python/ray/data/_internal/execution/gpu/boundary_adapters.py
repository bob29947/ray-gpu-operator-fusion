"""Direction-aware materialization adapters for GPU operator boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

from ray.data._internal.execution.gpu.operator_spec import (
    PayloadKind,
    RANGE_DATA_TERMINAL_ADAPTER_KEY,
    RANGE_WORK_ADAPTER_KEY,
)


class GPUBoundaryDirection(str, Enum):
    IMPORT = "import"
    EXPORT = "export"


BoundaryAdapterFn = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class GPUBoundaryAdapter:
    key: str
    direction: GPUBoundaryDirection
    external_payload: PayloadKind
    native_payload: PayloadKind
    adapt: BoundaryAdapterFn

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("GPU boundary adapter key must be nonempty")
        if not callable(self.adapt):
            raise TypeError("GPU boundary adapter adapt function must be callable")


_AdapterIdentity = Tuple[
    str,
    GPUBoundaryDirection,
    PayloadKind,
    PayloadKind,
]
_GPU_BOUNDARY_ADAPTERS: Dict[_AdapterIdentity, GPUBoundaryAdapter] = {}


def _identity(
    key: str,
    direction: GPUBoundaryDirection,
    external_payload: PayloadKind,
    native_payload: PayloadKind,
) -> _AdapterIdentity:
    return key, direction, external_payload, native_payload


def register_gpu_boundary_adapter(
    adapter: GPUBoundaryAdapter,
    *,
    replace_existing: bool = False,
) -> None:
    identity = _identity(
        adapter.key,
        adapter.direction,
        adapter.external_payload,
        adapter.native_payload,
    )
    if identity in _GPU_BOUNDARY_ADAPTERS and not replace_existing:
        raise ValueError(
            "GPU boundary adapter already registered: "
            f"{adapter.key!r}/{adapter.direction.value}/"
            f"{adapter.external_payload.value}/{adapter.native_payload.value}"
        )
    _GPU_BOUNDARY_ADAPTERS[identity] = adapter


def get_gpu_boundary_adapter(
    *,
    key: str,
    direction: GPUBoundaryDirection,
    external_payload: PayloadKind,
    native_payload: PayloadKind,
) -> Optional[GPUBoundaryAdapter]:
    return _GPU_BOUNDARY_ADAPTERS.get(
        _identity(key, direction, external_payload, native_payload)
    )


def unregister_gpu_boundary_adapter(
    *,
    key: str,
    direction: GPUBoundaryDirection,
    external_payload: PayloadKind,
    native_payload: PayloadKind,
) -> Optional[GPUBoundaryAdapter]:
    """Remove one exact registration, primarily for isolated tests/plugins."""

    return _GPU_BOUNDARY_ADAPTERS.pop(
        _identity(key, direction, external_payload, native_payload),
        None,
    )


def require_gpu_boundary_adapter(
    *,
    key: str,
    direction: GPUBoundaryDirection,
    external_payload: PayloadKind,
    native_payload: PayloadKind,
) -> GPUBoundaryAdapter:
    adapter = get_gpu_boundary_adapter(
        key=key,
        direction=direction,
        external_payload=external_payload,
        native_payload=native_payload,
    )
    if adapter is None:
        raise ValueError(
            "No GPU boundary adapter is registered for "
            f"{key!r}/{direction.value}/"
            f"{external_payload.value}/{native_payload.value}"
        )
    return adapter


def _range_work_import(value: Any, task_context: Any) -> Any:
    del task_context
    from ray.data._internal.execution.gpu.transforms.boundary import (
        decode_one_range_work,
    )

    return decode_one_range_work(value)


def _range_data_export(value: Any, task_context: Any) -> Any:
    del task_context
    from ray.data._internal.execution.gpu.payloads import RangeData
    from ray.data.block import BlockAccessor

    if not isinstance(value, RangeData):
        raise TypeError("the range-data export adapter expected RangeData")
    if value.value is None:
        return iter(())

    def convert():
        for batch in value.value:
            yield BlockAccessor.batch_to_block(batch)

    return convert()


# V1 executable ports.  RANGE_DATA boundary import and export are intentionally
# absent: the first exchange enforcer registers both preserving directions.
# Only terminal egress, which may discard PartitionIdentity, exists today.
register_gpu_boundary_adapter(
    GPUBoundaryAdapter(
        key=RANGE_WORK_ADAPTER_KEY,
        direction=GPUBoundaryDirection.IMPORT,
        external_payload=PayloadKind.RAY_BLOCK_STREAM,
        native_payload=PayloadKind.RANGE_WORK,
        adapt=_range_work_import,
    )
)
register_gpu_boundary_adapter(
    GPUBoundaryAdapter(
        key=RANGE_DATA_TERMINAL_ADAPTER_KEY,
        direction=GPUBoundaryDirection.EXPORT,
        external_payload=PayloadKind.RAY_BLOCK_STREAM,
        native_payload=PayloadKind.RANGE_DATA,
        adapt=_range_data_export,
    )
)


__all__ = [
    "GPUBoundaryAdapter",
    "GPUBoundaryDirection",
    "get_gpu_boundary_adapter",
    "register_gpu_boundary_adapter",
    "require_gpu_boundary_adapter",
    "unregister_gpu_boundary_adapter",
]
