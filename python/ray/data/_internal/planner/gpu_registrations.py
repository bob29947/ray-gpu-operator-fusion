"""Durable registration hooks for additive gate-2 GPU planner extensions.

Registrations affect subsequent ordinary :class:`Planner` calls.  Each call
snapshots these process-wide registrations into plan-local registries, so a
planning pass remains immutable even if a plugin is later unloaded.
"""

from ray.data._internal.planner.gpu_read_source_provider import (
    register_gpu_read_source_provider,
    unregister_gpu_read_source_provider,
)
from ray.data._internal.planner.gpu_requirement_analysis import (
    register_gpu_logical_lowerer,
    register_gpu_property_enforcer,
    register_gpu_property_transfer,
    unregister_gpu_logical_lowerer,
    unregister_gpu_property_enforcer,
    unregister_gpu_property_transfer,
)

__all__ = [
    "register_gpu_logical_lowerer",
    "register_gpu_property_enforcer",
    "register_gpu_property_transfer",
    "register_gpu_read_source_provider",
    "unregister_gpu_logical_lowerer",
    "unregister_gpu_property_enforcer",
    "unregister_gpu_property_transfer",
    "unregister_gpu_read_source_provider",
]
