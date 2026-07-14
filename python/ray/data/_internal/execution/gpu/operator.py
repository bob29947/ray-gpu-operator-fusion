"""Nominal physical operators and backend factory seam for GPU regions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Protocol, TYPE_CHECKING

from ray.data._internal.execution.gpu.operator_spec import GPUOperatorSpec
from ray.data._internal.execution.interfaces import PhysicalOperator

if TYPE_CHECKING:
    from ray.data.context import DataContext


class GPURequiredFusionError(ValueError):
    """A deferred GPU operator reached an execution-only interface."""


class GPUBackendRefusal(ValueError):
    """A registered backend cannot close or materialize a compiled GPU region."""

    def __init__(self, backend_name: str, detail: str):
        self.backend_name = backend_name
        self.detail = detail
        super().__init__(f"backend={backend_name!r}; {detail}")


def _gpu_name(prefix: str, spec: GPUOperatorSpec) -> str:
    kinds = ",".join(transform.kind for transform in spec.transforms)
    return f"{prefix}[{kinds}]"


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
        target = downstream.target_max_block_size_override
        if target is None:
            target = self.target_max_block_size_override
        return replace(
            self,
            name=_gpu_name("GPU", spec),
            target_max_block_size_override=target,
        )


class GPUOperator:
    """Nominal marker for physical nodes owned by the GPU subsystem."""

    @property
    def gpu_operator_spec(self) -> GPUOperatorSpec:
        return self._gpu_operator_spec

    @property
    def execution_backend_name(self) -> str:
        keys = sorted(self._gpu_operator_spec.execution_requirements.backend_keys)
        return "|".join(keys)

    @property
    def gpu_creation_options(self) -> GPUOperatorCreationOptions:
        return self._gpu_creation_options


class ExecutableGPUOperator(GPUOperator):
    """Nominal marker for a backend-materialized executable GPU region."""


class PlannedGPUOperator(PhysicalOperator, GPUOperator):
    """Backend-neutral, deliberately non-executable physical GPU fragment."""

    def __init__(
        self,
        spec: GPUOperatorSpec,
        input_op: PhysicalOperator,
        data_context: "DataContext",
        creation_options: Optional[GPUOperatorCreationOptions] = None,
        *,
        owns_input_parallelism: bool = False,
    ) -> None:
        self._gpu_operator_spec = spec
        self._gpu_creation_options = creation_options or GPUOperatorCreationOptions()
        self._owns_input_parallelism = bool(owns_input_parallelism)
        name = self._gpu_creation_options.name or _gpu_name("PlannedGPU", spec)
        super().__init__(
            name,
            [input_op],
            data_context,
            target_max_block_size_override=(
                self._gpu_creation_options.target_max_block_size_override
            ),
        )

    @property
    def owns_input_parallelism(self) -> bool:
        return self._owns_input_parallelism

    def start(self, *_args, **_kwargs) -> None:
        transforms = tuple(t.kind for t in self._gpu_operator_spec.transforms)
        raise GPURequiredFusionError(
            "a PlannedGPUOperator reached execution before physical fusion: "
            f"transforms={transforms!r}"
        )


class GPUOperatorFactory(Protocol):
    def prepare(self, core_spec: GPUOperatorSpec) -> GPUOperatorSpec: ...

    def supports(self, compiled_spec: GPUOperatorSpec) -> bool: ...

    def create(
        self,
        compiled_spec: GPUOperatorSpec,
        input_op: PhysicalOperator,
        data_context: "DataContext",
        creation_options: Optional[GPUOperatorCreationOptions] = None,
    ) -> PhysicalOperator: ...

    def attests(
        self, operator: PhysicalOperator, compiled_spec: GPUOperatorSpec
    ) -> bool: ...


@dataclass(frozen=True)
class GPUBackendMaterialization:
    operator: PhysicalOperator
    compiled_spec: GPUOperatorSpec
    backend_name: str
    attested: bool


_GPU_OPERATOR_FACTORIES: Dict[str, GPUOperatorFactory] = {}


def register_gpu_operator_factory(
    backend_name: str,
    factory: GPUOperatorFactory,
    *,
    replace_existing: bool = False,
) -> None:
    if not backend_name:
        raise ValueError("GPU backend name must be nonempty")
    if backend_name in _GPU_OPERATOR_FACTORIES and not replace_existing:
        raise ValueError(f"GPU operator factory already registered: {backend_name}")
    _GPU_OPERATOR_FACTORIES[backend_name] = factory


def _register_builtin_gpu_operator_factory(
    backend_name: str, factory: GPUOperatorFactory
) -> None:
    """Register a built-in without replacing an earlier plugin registration."""

    _GPU_OPERATOR_FACTORIES.setdefault(backend_name, factory)


def unregister_gpu_operator_factory(
    backend_name: str,
    *,
    expected: Optional[GPUOperatorFactory] = None,
) -> Optional[GPUOperatorFactory]:
    factory = _GPU_OPERATOR_FACTORIES.get(backend_name)
    if factory is None:
        return None
    if expected is not None and factory is not expected:
        raise ValueError(
            f"GPU operator factory {backend_name!r} is not the expected factory"
        )
    return _GPU_OPERATOR_FACTORIES.pop(backend_name)


def _ensure_default_factory_loaded(backend_name: str) -> None:
    if backend_name == "actor_pool" and backend_name not in _GPU_OPERATOR_FACTORIES:
        from ray.data._internal.execution.operators import (  # noqa: F401
            actor_pool_gpu_operator,
        )


def get_gpu_operator_factory(backend_name: str) -> Optional[GPUOperatorFactory]:
    _ensure_default_factory_loaded(backend_name)
    return _GPU_OPERATOR_FACTORIES.get(backend_name)


def has_gpu_operator_factory(backend_name: str) -> bool:
    return get_gpu_operator_factory(backend_name) is not None


def materialize_gpu_operator(
    core_spec: GPUOperatorSpec,
    input_op: PhysicalOperator,
    data_context: "DataContext",
    creation_options: Optional[GPUOperatorCreationOptions] = None,
) -> GPUBackendMaterialization:
    refusals = []
    for backend_name in sorted(core_spec.execution_requirements.backend_keys):
        factory = get_gpu_operator_factory(backend_name)
        if factory is None:
            refusals.append(f"{backend_name}: factory is not registered")
            continue
        operator = None
        try:
            execution = replace(
                core_spec.execution_requirements,
                backend_keys=frozenset((backend_name,)),
            )
            selected_core = replace(core_spec, execution_requirements=execution)
            compiled_spec = factory.prepare(selected_core)
            if not factory.supports(compiled_spec):
                raise GPUBackendRefusal(
                    backend_name, "factory rejected the prepared region"
                )
            operator = factory.create(
                compiled_spec,
                input_op,
                data_context,
                creation_options,
            )
            if not isinstance(operator, PhysicalOperator) or isinstance(
                operator, PlannedGPUOperator
            ):
                raise GPUBackendRefusal(
                    backend_name,
                    "factory did not return an executable PhysicalOperator",
                )
            if not isinstance(operator, ExecutableGPUOperator):
                raise GPUBackendRefusal(
                    backend_name,
                    "factory result lacks the nominal ExecutableGPUOperator marker",
                )
            if not factory.attests(operator, compiled_spec):
                raise GPUBackendRefusal(
                    backend_name, "factory could not attest its materialized region"
                )
            return GPUBackendMaterialization(
                operator=operator,
                compiled_spec=compiled_spec,
                backend_name=backend_name,
                attested=True,
            )
        except GPUBackendRefusal as error:
            _detach_rejected_operator(operator)
            refusals.append(str(error))
        except Exception:
            _detach_rejected_operator(operator)
            raise
    detail = "; ".join(refusals) or "no candidate backends"
    backends = "|".join(sorted(core_spec.execution_requirements.backend_keys))
    raise GPUBackendRefusal(backends, detail)


def _detach_rejected_operator(operator: Optional[PhysicalOperator]) -> None:
    if not isinstance(operator, PhysicalOperator):
        return
    for input_op in operator.input_dependencies:
        input_op._output_dependencies = [
            dependency
            for dependency in input_op._output_dependencies
            if dependency is not operator
        ]


def create_gpu_operator(
    spec: GPUOperatorSpec,
    input_op: PhysicalOperator,
    data_context: "DataContext",
    creation_options: Optional[GPUOperatorCreationOptions] = None,
) -> PhysicalOperator:
    result = materialize_gpu_operator(spec, input_op, data_context, creation_options)
    return result.operator
