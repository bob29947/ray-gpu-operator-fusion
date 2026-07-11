from typing import List

from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.operators import MapGroupPartitions
from ray.data.context import DataContext


def plan_map_group_partitions_op(
    op: MapGroupPartitions,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
) -> PhysicalOperator:
    """Fail when the anchored GPU provider did not claim this explicit API."""

    if not data_context.enable_gpu_operator_fusion:
        raise ValueError(
            "map_group_partitions requires "
            "DataContext.enable_gpu_operator_fusion=True; set the flag before "
            "executing this Dataset"
        )
    raise ValueError(
        "map_group_partitions could not be lowered by a compatible GPU plan "
        "provider; this V1 API has no stock execution fallback"
    )
