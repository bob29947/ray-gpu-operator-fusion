"""Validation helpers for physical fragments returned by GPU providers."""

from ray.data._internal.execution.gpu import GPUOperator
from ray.data._internal.planner.gpu_plan_provider import GPUPlanFragment


def validate_gpu_plan_fragment(fragment: GPUPlanFragment) -> None:
    reachable = set(fragment.root.post_order_iter())
    new_operators = set(fragment.new_operators)
    mapped_operators = set(fragment.op_map)

    if len(new_operators) != len(fragment.new_operators):
        raise ValueError("GPU provider returned duplicate physical operators")
    if len(set(fragment.consumed_logical_operators)) != len(
        fragment.consumed_logical_operators
    ):
        raise ValueError("GPU provider returned duplicate consumed logical operators")
    if fragment.root not in new_operators:
        raise ValueError("GPU provider fragment root is missing from new_operators")
    if new_operators != reachable:
        missing = reachable - new_operators
        unreachable = new_operators - reachable
        raise ValueError(
            "GPU provider new_operators must exactly cover its reachable physical "
            f"fragment; missing={len(missing)}, unreachable={len(unreachable)}"
        )
    if mapped_operators != new_operators:
        raise ValueError(
            "GPU provider must provide exactly one op_map entry for every new operator"
        )

    consumed = set(fragment.consumed_logical_operators)
    if not consumed:
        raise ValueError("GPU provider consumed no logical operators")
    if set(fragment.op_map.values()) != consumed:
        raise ValueError(
            "GPU provider op_map must represent every and only consumed logical "
            "operator"
        )
    for physical_op, mapped_logical_op in fragment.op_map.items():
        logical_lineage = tuple(physical_op._logical_operators)
        if mapped_logical_op not in logical_lineage:
            raise ValueError(
                "GPU provider op_map disagrees with a physical operator's logical "
                "lineage"
            )
        if any(logical_op not in consumed for logical_op in logical_lineage):
            raise ValueError(
                "GPU provider physical lineage escapes its consumed logical region"
            )

    if not fragment.region_id:
        raise ValueError("GPU provider fragment requires a nonempty region_id")
    for physical_op in reachable:
        if not isinstance(physical_op, GPUOperator):
            continue
        operator_region_id = physical_op.gpu_operator_spec.region_id
        if operator_region_id != fragment.region_id:
            raise ValueError(
                "GPU provider fragment region_id disagrees with a GPU operator spec"
            )
