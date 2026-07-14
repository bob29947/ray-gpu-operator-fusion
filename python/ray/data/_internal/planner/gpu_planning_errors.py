"""Stable planning failures for explicit Ray Data GPU operations."""

from enum import Enum


class GPUPlanningErrorReason(str, Enum):
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    CONFLICTING_REQUIREMENTS = "CONFLICTING_REQUIREMENTS"
    INCOMPLETE_FUSION = "INCOMPLETE_FUSION"


class GPUPlanningError(ValueError):
    def __init__(
        self,
        reason: GPUPlanningErrorReason,
        detail: str,
        logical_ops=(),
        physical_ops=(),
    ):
        self.reason = reason
        self.detail = detail
        self.logical_ops = tuple(logical_ops)
        self.physical_ops = tuple(physical_ops)
        offenders = []
        if self.logical_ops:
            offenders.append(f"logical_ops={self.logical_ops!r}")
        if self.physical_ops:
            offenders.append(f"physical_ops={self.physical_ops!r}")
        suffix = f"; {', '.join(offenders)}" if offenders else ""
        recovery = (
            "; recovery=disable enable_gpu_operator_fusion or remove the "
            "unsupported operators"
        )
        super().__init__(f"{reason.value}: {detail}{suffix}{recovery}")


__all__ = ["GPUPlanningError", "GPUPlanningErrorReason"]
