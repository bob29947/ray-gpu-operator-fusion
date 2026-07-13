"""Physical optimizer rule for declarative marker-based GPU operators."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional, Set, Tuple

from ray.data._internal.execution.gpu.composition import (
    GPUCompositionError,
)
from ray.data._internal.execution.gpu.operator import (
    GPUOperator,
    create_gpu_operator,
)
from ray.data._internal.execution.gpu.operator_spec import GPUOperatorSpec
from ray.data._internal.execution.gpu.region_compiler import compile_gpu_region
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule


class FuseGPUOperators(Rule):
    """Fuse maximal linear chains of compatible :class:`GPUOperator` nodes."""

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
        if not getattr(plan.context, "enable_gpu_operator_fusion", False):
            return plan

        self._op_map = plan.op_map.copy()
        self._consumer_counts = _consumer_counts(plan.dag)
        fused_dag = self._fuse_from_downstream(plan.dag, plan.context)
        _rebuild_output_dependencies(fused_dag)
        return PhysicalPlan(fused_dag, self._op_map, plan.context)

    def _fuse_from_downstream(
        self, op: PhysicalOperator, data_context
    ) -> PhysicalOperator:
        chain, compiled_spec = self._maximal_chain_ending_at(op)
        if len(chain) > 1:
            assert compiled_spec is not None
            external_inputs = chain[0].input_dependencies
            if len(external_inputs) != 1:
                return self._rewrite_inputs(op, data_context)

            upstream_gpu = chain[0]
            downstream_gpu = chain[-1]
            assert isinstance(upstream_gpu, GPUOperator)
            assert isinstance(downstream_gpu, GPUOperator)
            creation_options = upstream_gpu.gpu_creation_options.for_fused_spec(
                downstream_gpu.gpu_creation_options, compiled_spec
            )
            try:
                fused = create_gpu_operator(
                    compiled_spec,
                    external_inputs[0],
                    data_context,
                    creation_options,
                )
            except ValueError:
                # A backend may support each materialized stage but not the
                # combined contract.  Optional fusion must retain the existing
                # physical boundary in that case.
                return self._rewrite_inputs(op, data_context)
            if not isinstance(fused, PhysicalOperator):
                raise TypeError("GPU operator factory must return a PhysicalOperator")

            logical_lineage = []
            downstream_logical = self._op_map.get(chain[-1])
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

            op = fused

        return self._rewrite_inputs(op, data_context)

    def _rewrite_inputs(self, op: PhysicalOperator, data_context) -> PhysicalOperator:
        op._input_dependencies = [
            self._fuse_from_downstream(input_op, data_context)
            for input_op in op.input_dependencies
        ]
        return op

    def _maximal_chain_ending_at(
        self, downstream: PhysicalOperator
    ) -> Tuple[List[PhysicalOperator], Optional[GPUOperatorSpec]]:
        if not isinstance(downstream, GPUOperator):
            return [downstream], None
        if downstream.gpu_operator_spec.gate2_composed:
            # Gate 2 already selected and compiled this executable stage.  It is
            # atomic to the optional physical optimizer, even when another GPU
            # marker happens to be adjacent.
            return [downstream], downstream.gpu_operator_spec

        chain: List[PhysicalOperator] = [downstream]
        compiled_spec = downstream.gpu_operator_spec
        cursor = downstream
        while len(cursor.input_dependencies) == 1:
            upstream = cursor.input_dependencies[0]
            if not isinstance(upstream, GPUOperator):
                break
            if upstream.gpu_operator_spec.gate2_composed:
                break
            if self._consumer_counts.get(upstream, 0) != 1:
                break
            if upstream.execution_backend_name != downstream.execution_backend_name:
                break

            candidate = [upstream] + chain
            try:
                # Rebuild the candidate with compile_gpu_region instead of
                # prepending to the previous result. This keeps fusion on the
                # canonical code path and avoids assuming future spec fields
                # compose identically in either order.
                compiled_spec = compile_gpu_region(
                    node.gpu_operator_spec  # type: ignore[attr-defined]
                    for node in candidate
                )
            except GPUCompositionError:
                # Fusion is optional.  The already-valid physical boundary is
                # retained and no composition failure escapes optimization.
                break
            chain = candidate
            cursor = upstream

        return chain, compiled_spec


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
