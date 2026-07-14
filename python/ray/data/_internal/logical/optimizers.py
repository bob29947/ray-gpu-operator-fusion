from typing import TYPE_CHECKING, List, Tuple

from ray.data._internal.planner import create_planner

if TYPE_CHECKING:
    from ray.data._internal.execution.execution_callback import ExecutionCallback

from .ruleset import Ruleset
from ray.data._internal.logical.interfaces import (
    LogicalPlan,
    Optimizer,
    PhysicalPlan,
    Rule,
)
from ray.data._internal.logical.rules import (
    CombineShuffles,
    CommonSubExprElimination,
    ConfigureMapTaskMemoryUsingOutputSize,
    FuseGPUOperators,
    FuseOperators,
    InheritTargetMaxBlockSizeRule,
    LimitPushdownRule,
    PredicatePushdown,
    ProjectionPushdown,
    SetReadParallelismRule,
)
from ray.util.annotations import DeveloperAPI

_LOGICAL_RULESET = Ruleset(
    [
        LimitPushdownRule,
        ProjectionPushdown,
        PredicatePushdown,
        CombineShuffles,
    ]
)

_PHYSICAL_RULESET = Ruleset(
    [
        InheritTargetMaxBlockSizeRule,
        SetReadParallelismRule,
        FuseGPUOperators,
        FuseOperators,
        ConfigureMapTaskMemoryUsingOutputSize,
    ]
)


@DeveloperAPI
def get_logical_ruleset() -> Ruleset:
    return _LOGICAL_RULESET


@DeveloperAPI
def get_physical_ruleset() -> Ruleset:
    return _PHYSICAL_RULESET


class LogicalOptimizer(Optimizer):
    """The optimizer for logical operators."""

    @property
    def rules(self) -> List[Rule]:
        return [rule_cls() for rule_cls in get_logical_ruleset()]

    def _post_optimize(self, plan: LogicalPlan) -> LogicalPlan:
        # CommonSubExprElimination is only supposed to run once
        # isolated from the optimizer rule loop as it applies to
        # a single Projection operator not a chain of operators.
        return CommonSubExprElimination().apply(plan)


class PhysicalOptimizer(Optimizer):
    """The optimizer for physical operators."""

    @property
    def rules(self) -> List[Rule]:
        return [rule_cls() for rule_cls in get_physical_ruleset()]

    def _post_optimize(self, plan: PhysicalPlan) -> PhysicalPlan:
        # Planned GPU nodes are deliberately non-executable.  Keep this check
        # outside the fusion rule so a custom physical ruleset cannot silently
        # omit required materialization and hand deferred nodes to the executor.
        from ray.data._internal.execution.gpu.operator import (
            GPURequiredFusionError,
            PlannedGPUOperator,
        )

        leftovers = tuple(
            op
            for op in plan.dag.post_order_iter()
            if isinstance(op, PlannedGPUOperator)
        )
        if leftovers:
            groups = sorted(
                {
                    op.gpu_operator_spec.fusion_group_id or "<missing>"
                    for op in leftovers
                }
            )
            raise GPURequiredFusionError(
                "physical optimization left non-executable planned GPU "
                f"operators: groups={groups!r}"
            )
        return plan


def get_execution_plan(
    logical_plan: LogicalPlan,
) -> Tuple[PhysicalPlan, List["ExecutionCallback"]]:
    """Get the physical execution plan for the provided logical plan.

    This process has 3 steps:
    (1) logical optimization: optimize logical operators.
    (2) planning: convert logical to physical operators.
    (3) physical optimization: optimize physical operators.
    """
    # 1. Logical -> Logical (Optimized)
    optimized_logical_plan = LogicalOptimizer().optimize(logical_plan)

    # 2. Rewire Logical -> Logical (Optimized)
    logical_plan._dag = optimized_logical_plan.dag

    # 3. Logical (Optimized) -> Physical
    physical_plan, callbacks = create_planner().plan(optimized_logical_plan)

    # 4. Physical (Optimized) -> Physical
    return PhysicalOptimizer().optimize(physical_plan), callbacks
