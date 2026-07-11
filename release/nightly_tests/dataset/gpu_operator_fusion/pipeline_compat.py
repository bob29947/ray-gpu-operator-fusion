"""Narrow compatibility helpers for stock and modified Ray APIs."""

from __future__ import annotations

import inspect
from typing import Any


def _load_ray_data_api() -> tuple[Any, Any, Any, Any]:
    import ray.data as ray_data
    from ray.data.context import DataContext
    from ray.data.dataset import Dataset
    from ray.data.grouped_data import GroupedData

    return ray_data, DataContext, Dataset, GroupedData


def configure_data_context(
    *, enable_gpu_operator_fusion: bool
) -> dict[str, object]:
    """Apply only settings shared by the three-arm benchmark contract."""
    ray_data, DataContext, Dataset, GroupedData = _load_ray_data_api()
    context = DataContext.get_current()
    if not hasattr(context, "use_datasource_v2"):
        raise RuntimeError(
            "benchmark Ray does not expose DataContext.use_datasource_v2"
        )
    context.use_datasource_v2 = False
    if context.use_datasource_v2 is not False:
        raise RuntimeError("benchmark requires DataContext.use_datasource_v2=False")

    fusion_flag_supported = hasattr(context, "enable_gpu_operator_fusion")
    if fusion_flag_supported:
        context.enable_gpu_operator_fusion = enable_gpu_operator_fusion
        if context.enable_gpu_operator_fusion is not enable_gpu_operator_fusion:
            raise RuntimeError("failed to configure enable_gpu_operator_fusion")

    map_batches_has_udf_properties = (
        "udf_properties" in inspect.signature(Dataset.map_batches).parameters
    )
    udf_properties_exported = hasattr(ray_data, "UDFProperties")
    if map_batches_has_udf_properties != udf_properties_exported:
        raise RuntimeError(
            "UDFProperties export and Dataset.map_batches support must appear together"
        )

    return {
        "use_datasource_v2": context.use_datasource_v2,
        "fusion_flag_supported": fusion_flag_supported,
        "enable_gpu_operator_fusion": (
            context.enable_gpu_operator_fusion if fusion_flag_supported else None
        ),
        "udf_properties_supported": udf_properties_exported,
        "map_group_partitions_api_supported": hasattr(
            GroupedData, "map_group_partitions"
        ),
    }


def partition_preservation_kwargs() -> dict[str, object]:
    """Return the one modified-Ray-only ``map_batches`` keyword when supported."""
    ray_data, _, Dataset, _ = _load_ray_data_api()
    supports_parameter = (
        "udf_properties" in inspect.signature(Dataset.map_batches).parameters
    )
    properties_type = getattr(ray_data, "UDFProperties", None)
    if supports_parameter != (properties_type is not None):
        raise RuntimeError(
            "UDFProperties export and Dataset.map_batches support must appear together"
        )
    if not supports_parameter:
        return {}
    return {
        "udf_properties": properties_type(preserves_partitioning=True),
    }
