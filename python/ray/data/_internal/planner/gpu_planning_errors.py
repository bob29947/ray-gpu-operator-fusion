"""Stable planning failures for explicit Ray Data GPU operations."""

from enum import Enum


class GPUPlanningErrorReason(str, Enum):
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    EXCHANGE_REQUIRED = "EXCHANGE_REQUIRED"
    CONFLICTING_REQUIREMENTS = "CONFLICTING_REQUIREMENTS"


class GPUPlanningError(ValueError):
    def __init__(self, reason: GPUPlanningErrorReason, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}")


__all__ = ["GPUPlanningError", "GPUPlanningErrorReason"]
