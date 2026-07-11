"""Backend-neutral marker and factory seam for GPU physical operators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Dict, Optional, Protocol, TYPE_CHECKING

from ray.data._internal.execution.gpu.operator_spec import GPUOperatorSpec

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import PhysicalOperator
    from ray.data.context import DataContext


@dataclass(frozen=True)
class GPUOperatorCreationOptions:
    name: Optional[str] = None
    min_rows_per_bundle: Optional[int] = None
    target_max_block_size_override: Optional[int] = None

    def for_fused_spec(
        self,
        downstream: "GPUOperatorCreationOptions",
        spec: GPUOperatorSpec,
    ) -> "GPUOperatorCreationOptions":
        """Keep options owned by the fused operator's external boundaries.

        Input bundling belongs to the upstream-most node, output block sizing
        belongs to the downstream-most node, and the name describes the full
        fused transform sequence.
        """

        target = (
            downstream.target_max_block_size_override
            if downstream.target_max_block_size_override is not None
            else self.target_max_block_size_override
        )
        return replace(
            self,
            name="GPU[" + ",".join(spec.transform_ids) + "]",
            target_max_block_size_override=target,
        )


class GPUOperator(ABC):
    """Marker implemented by every physical node owned by this GPU backend."""

    @property
    @abstractmethod
    def gpu_operator_spec(self) -> GPUOperatorSpec:
        pass

    @property
    @abstractmethod
    def execution_backend_name(self) -> str:
        pass

    @property
    @abstractmethod
    def gpu_creation_options(self) -> GPUOperatorCreationOptions:
        pass


class GPUOperatorFactory(Protocol):
    def supports(self, spec: GPUOperatorSpec) -> bool:
        ...

    def create(
        self,
        spec: GPUOperatorSpec,
        input_op: "PhysicalOperator",
        data_context: "DataContext",
        creation_options: Optional[GPUOperatorCreationOptions] = None,
    ) -> "PhysicalOperator":
        ...


_GPU_OPERATOR_FACTORIES: Dict[str, GPUOperatorFactory] = {}


def register_gpu_operator_factory(
    backend_name: str,
    factory: GPUOperatorFactory,
    *,
    replace_existing: bool = False,
) -> None:
    if backend_name in _GPU_OPERATOR_FACTORIES and not replace_existing:
        raise ValueError(f"GPU operator factory already registered: {backend_name}")
    _GPU_OPERATOR_FACTORIES[backend_name] = factory


def _ensure_default_factory_loaded(backend_name: str) -> None:
    if backend_name == "actor_pool" and backend_name not in _GPU_OPERATOR_FACTORIES:
        # This is the only durable-to-ActorPool import, and it is lazy.  Provider
        # and optimizer modules continue to depend on this generic function only.
        from ray.data._internal.execution.operators import (  # noqa: F401
            actor_pool_gpu_operator,
        )


def create_gpu_operator(
    spec: GPUOperatorSpec,
    input_op: "PhysicalOperator",
    data_context: "DataContext",
    creation_options: Optional[GPUOperatorCreationOptions] = None,
) -> "PhysicalOperator":
    backend_name = spec.execution_requirements.backend_name
    _ensure_default_factory_loaded(backend_name)
    try:
        factory = _GPU_OPERATOR_FACTORIES[backend_name]
    except KeyError as exc:
        raise ValueError(
            f"No GPU operator factory for backend {backend_name!r}"
        ) from exc
    if not factory.supports(spec):
        raise ValueError(
            f"GPU operator factory {backend_name!r} does not support this specification"
        )
    return factory.create(spec, input_op, data_context, creation_options)
