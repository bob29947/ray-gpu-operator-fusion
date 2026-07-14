"""Physical materialization rule for declarative planned GPU regions."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Set

from ray.data._internal.execution.gpu.composition import GPUCompositionError
from ray.data._internal.execution.gpu.operator import (
    GPUOperator,
    GPURequiredFusionError,
    PlannedGPUOperator,
    create_gpu_operator,
)
from ray.data._internal.execution.gpu.operator_spec import GPUOperatorSpec
from ray.data._internal.execution.gpu.region_compiler import compile_gpu_region
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule


class FuseGPUOperators(Rule):
    """Atomically compile and materialize complete planned GPU fusion groups."""

    @classmethod
    def dependencies(cls):
        from ray.data._internal.logical.rules.set_read_parallelism import (
            SetReadParallelismRule,
        )

        return [SetReadParallelismRule]

    @classmethod
    def dependents(cls):
        from ray.data._internal.logical.rules.operator_fusion import FuseOperators

        return [FuseOperators]

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        planned = tuple(
            op
            for op in _reachable_operators(plan.dag)
            if isinstance(op, PlannedGPUOperator)
        )
        if not getattr(plan.context, "enable_gpu_operator_fusion", False):
            if planned:
                raise GPURequiredFusionError(
                    "planned GPU operators require GPU physical fusion to be enabled"
                )
            return plan
        if not planned:
            return plan

        self._op_map = plan.op_map.copy()
        self._consumer_counts = _consumer_counts(plan.dag)
        self._planned_by_group = self._index_required_groups(planned)
        fused_dag = self._fuse_from_downstream(plan.dag, plan.context)

        leftovers = tuple(
            op
            for op in _reachable_operators(fused_dag)
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
                "required GPU fusion left non-executable planned operators: "
                f"groups={groups!r}"
            )

        _rebuild_output_dependencies(fused_dag)
        return PhysicalPlan(fused_dag, self._op_map, plan.context)

    def _index_required_groups(
        self, planned: Iterable[PlannedGPUOperator]
    ) -> Dict[str, Set[PlannedGPUOperator]]:
        groups: Dict[str, Set[PlannedGPUOperator]] = {}
        required_provenance: Dict[str, tuple[str, ...]] = {}
        for op in planned:
            spec = op.gpu_operator_spec
            group_id = spec.fusion_group_id
            if (
                not spec.requires_complete_fusion
                or not group_id
                or not spec.required_logical_provenance
            ):
                raise GPURequiredFusionError(
                    "every PlannedGPUOperator must declare a complete fusion group"
                )
            expected = required_provenance.setdefault(
                group_id, spec.required_logical_provenance
            )
            if expected != spec.required_logical_provenance:
                raise GPURequiredFusionError(
                    f"fusion group {group_id!r} declares inconsistent required "
                    "logical provenance"
                )
            if self._consumer_counts.get(op, 0) > 1:
                raise GPURequiredFusionError(
                    f"fusion group {group_id!r} is branched at {op.name!r}"
                )
            groups.setdefault(group_id, set()).add(op)
        return groups

    def _fuse_from_downstream(
        self, op: PhysicalOperator, data_context
    ) -> PhysicalOperator:
        if isinstance(op, PlannedGPUOperator):
            op = self._materialize_complete_group(op, data_context)
        return self._rewrite_inputs(op, data_context)

    def _rewrite_inputs(self, op: PhysicalOperator, data_context) -> PhysicalOperator:
        op._input_dependencies = [
            self._fuse_from_downstream(input_op, data_context)
            for input_op in op.input_dependencies
        ]
        return op

    def _materialize_complete_group(
        self,
        downstream: PlannedGPUOperator,
        data_context,
    ) -> PhysicalOperator:
        group_id = downstream.gpu_operator_spec.fusion_group_id
        assert group_id is not None
        chain = self._full_chain_ending_at(downstream, group_id)
        declared_nodes = self._planned_by_group[group_id]
        if set(chain) != declared_nodes:
            raise GPURequiredFusionError(
                f"fusion group {group_id!r} is not one complete linear chain; "
                f"reachable={len(chain)}, declared={len(declared_nodes)}"
            )

        expected = downstream.gpu_operator_spec.required_logical_provenance
        actual = tuple(
            logical_id
            for node in chain
            for logical_id in node.gpu_operator_spec.logical_provenance
        )
        if actual != expected:
            raise GPURequiredFusionError(
                f"fusion group {group_id!r} has incomplete or reordered logical "
                f"provenance: actual={actual!r}, required={expected!r}"
            )

        external_inputs = chain[0].input_dependencies
        if len(external_inputs) != 1:
            raise GPURequiredFusionError(
                f"fusion group {group_id!r} must have exactly one external input"
            )
        if not downstream.gpu_operator_spec.uses_terminal_output_adapter:
            raise GPURequiredFusionError(
                f"fusion group {group_id!r} does not declare terminal egress"
            )

        try:
            compiled = compile_gpu_region(
                node.gpu_operator_spec for node in chain
            )
        except (GPUCompositionError, ValueError) as error:
            raise GPURequiredFusionError(
                f"fusion group {group_id!r} could not be composed: {error}"
            ) from error

        self._validate_compiled_contract(compiled, group_id, expected)
        creation_options = chain[0].gpu_creation_options
        for node in chain[1:]:
            creation_options = creation_options.for_fused_spec(
                node.gpu_creation_options,
                compiled,
            )
        try:
            fused = create_gpu_operator(
                compiled,
                external_inputs[0],
                data_context,
                creation_options,
            )
        except Exception as error:
            raise GPURequiredFusionError(
                f"fusion group {group_id!r} backend materialization failed: {error}"
            ) from error

        if not isinstance(fused, PhysicalOperator) or isinstance(
            fused, PlannedGPUOperator
        ):
            raise GPURequiredFusionError(
                "GPU operator factory must return an executable PhysicalOperator"
            )
        if not isinstance(fused, GPUOperator):
            raise GPURequiredFusionError(
                "GPU operator factory result must implement GPUOperator"
            )
        if fused.gpu_operator_spec is not compiled:
            raise GPURequiredFusionError(
                "GPU operator factory must retain the exact compiled specification"
            )
        if fused.execution_backend_name != compiled.execution_requirements.backend_name:
            raise GPURequiredFusionError(
                "GPU operator factory returned the wrong execution backend"
            )
        if fused.input_dependencies != [external_inputs[0]]:
            raise GPURequiredFusionError(
                "GPU operator factory changed the compiled region's external input"
            )

        logical_lineage = []
        downstream_logical = self._op_map.get(downstream)
        for node in chain:
            if node._logical_operators:
                logical_lineage.extend(node._logical_operators)
            else:
                logical = self._op_map.get(node)
                if logical is not None:
                    logical_lineage.append(logical)
            self._op_map.pop(node, None)
        if logical_lineage:
            fused.set_logical_operators(*logical_lineage)
        if downstream_logical is not None:
            self._op_map[fused] = downstream_logical
        return fused

    def _full_chain_ending_at(
        self,
        downstream: PlannedGPUOperator,
        group_id: str,
    ) -> List[PlannedGPUOperator]:
        chain = [downstream]
        cursor = downstream
        while len(cursor.input_dependencies) == 1:
            upstream = cursor.input_dependencies[0]
            if not isinstance(upstream, PlannedGPUOperator):
                break
            if upstream.gpu_operator_spec.fusion_group_id != group_id:
                break
            if self._consumer_counts.get(upstream, 0) != 1:
                break
            chain.insert(0, upstream)
            cursor = upstream
        return chain

    @staticmethod
    def _validate_compiled_contract(
        compiled: GPUOperatorSpec,
        group_id: str,
        required_provenance: tuple[str, ...],
    ) -> None:
        if (
            not compiled.requires_complete_fusion
            or compiled.fusion_group_id != group_id
            or compiled.required_logical_provenance != required_provenance
            or compiled.logical_provenance != required_provenance
            or not compiled.uses_terminal_output_adapter
        ):
            raise GPURequiredFusionError(
                f"fusion group {group_id!r} compiled to an incomplete contract"
            )


def _reachable_operators(root: PhysicalOperator) -> Iterable[PhysicalOperator]:
    visited: Set[PhysicalOperator] = set()
    stack = [root]
    while stack:
        op = stack.pop()
        if op in visited:
            continue
        visited.add(op)
        yield op
        stack.extend(op.input_dependencies)


def _consumer_counts(root: PhysicalOperator) -> Counter:
    counts: Counter = Counter()
    for op in _reachable_operators(root):
        counts.update(op.input_dependencies)
    return counts


def _rebuild_output_dependencies(root: PhysicalOperator) -> None:
    operators = list(_reachable_operators(root))
    for op in operators:
        op._output_dependencies = []
    for op in operators:
        for input_op in op.input_dependencies:
            if op not in input_op._output_dependencies:
                input_op._output_dependencies.append(op)
