"""Pure folds for native GPU chains and their external boundaries."""

from __future__ import annotations

from functools import reduce
from typing import Iterable

from ray.data._internal.execution.gpu.composition import (
    GPUCompositionError,
    GPUCompositionErrorReason,
    fuse_gpu_operator_specs,
)
from ray.data._internal.execution.gpu.operator_spec import (
    GPUOperatorSpec,
    RAY_BLOCK_STREAM,
)


def compile_gpu_region(specs: Iterable[GPUOperatorSpec]) -> GPUOperatorSpec:
    specs = tuple(specs)
    if not specs:
        raise ValueError("cannot compile an empty GPU region")
    return reduce(fuse_gpu_operator_specs, specs)


def compile_region(
    core: GPUOperatorSpec,
    ingress: GPUOperatorSpec,
    egress: GPUOperatorSpec,
) -> GPUOperatorSpec:
    closed = compile_gpu_region((ingress, core, egress))
    if (
        closed.input_payload != RAY_BLOCK_STREAM
        or closed.output_payload != RAY_BLOCK_STREAM
    ):
        raise GPUCompositionError(
            GPUCompositionErrorReason.PAYLOAD_INCOMPATIBLE,
            "compiled GPU region must have Ray-block input and output boundaries",
        )
    return closed
