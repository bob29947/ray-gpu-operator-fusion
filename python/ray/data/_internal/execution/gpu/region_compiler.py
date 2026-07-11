"""Region fold shared by the physical fusion rule and contract tests."""

from functools import reduce
from typing import Iterable

from ray.data._internal.execution.gpu.composition import compose_gpu_operator_specs
from ray.data._internal.execution.gpu.operator_spec import GPUOperatorSpec


def compile_gpu_region(specs: Iterable[GPUOperatorSpec]) -> GPUOperatorSpec:
    specs = tuple(specs)
    if not specs:
        raise ValueError("cannot compile an empty GPU region")
    return reduce(compose_gpu_operator_specs, specs)
