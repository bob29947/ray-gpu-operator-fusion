"""Physical fusion for declarative, non-executable GPU fragments."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from ray.data._internal.execution.gpu.composition import (
    GPUCompositionError,
    GPUCompositionErrorReason,
    foldable_edge,
)
from ray.data._internal.execution.gpu.operator import (
    GPUBackendRefusal,
    GPUOperatorCreationOptions,
    GPURequiredFusionError,
    PlannedGPUOperator,
    materialize_gpu_operator,
)
from ray.data._internal.execution.gpu.region_compiler import compile_gpu_region
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.planner.gpu_planning_errors import GPUPlanningErrorReason


class FuseGPUOperators(Rule):
    """Fold each maximal linear GPU chain and materialize it atomically."""

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
        operators = tuple(_reachable_operators(plan.dag))
        if not any(isinstance(operator, PlannedGPUOperator) for operator in operators):
            return plan
        self._context = plan.context
        self._state = plan.gpu_fusion_state
        if self._state is None:
            raise GPURequiredFusionError(
                "strict GPU fusion requires plan-local fusion state"
            )
        self._op_map = plan.op_map.copy()
        self._consumer_counts = Counter(
            input_operator
            for operator in operators
            for input_operator in operator.input_dependencies
        )
        self._memo: dict[PhysicalOperator, PhysicalOperator] = {}
        fused_dag = self._rewrite(plan.dag)
        _rebuild_output_dependencies(fused_dag)
        return PhysicalPlan(
            fused_dag,
            self._op_map,
            plan.context,
            gpu_fusion_state=self._state,
        )

    def _rewrite(self, operator: PhysicalOperator) -> PhysicalOperator:
        cached = self._memo.get(operator)
        if cached is not None:
            return cached
        if isinstance(operator, PlannedGPUOperator):
            chain = self._maximal_chain_ending_at(operator)
            external_inputs = chain[0].input_dependencies
            if len(external_inputs) != 1:
                self._state.register_fusion_refusal(
                    chain,
                    GPUPlanningErrorReason.INCOMPLETE_FUSION,
                    "a GPU region must have exactly one external input",
                )
                chain[0]._input_dependencies = [
                    self._rewrite(item) for item in external_inputs
                ]
                for node in chain:
                    self._memo[node] = node
                return chain[-1]
            rewritten_input = self._rewrite(external_inputs[0])
            chain[0]._input_dependencies = [rewritten_input]
            result = self._materialize(chain, rewritten_input)
            for node in chain:
                self._memo[node] = result if result is not chain[-1] else node
            return result
        operator._input_dependencies = [
            self._rewrite(input_operator)
            for input_operator in operator.input_dependencies
        ]
        self._memo[operator] = operator
        return operator

    def _maximal_chain_ending_at(
        self, downstream: PlannedGPUOperator
    ) -> list[PlannedGPUOperator]:
        chain = [downstream]
        cursor = downstream
        while len(cursor.input_dependencies) == 1:
            upstream = cursor.input_dependencies[0]
            if not (
                isinstance(upstream, PlannedGPUOperator)
                and self._consumer_counts.get(upstream, 0) == 1
                and foldable_edge(upstream.gpu_operator_spec, cursor.gpu_operator_spec)
            ):
                break
            chain.insert(0, upstream)
            cursor = upstream
        return chain

    def _materialize(
        self,
        chain: list[PlannedGPUOperator],
        external_input: PhysicalOperator,
    ) -> PhysicalOperator:
        try:
            core = compile_gpu_region(node.gpu_operator_spec for node in chain)
        except GPUCompositionError as error:
            reason = (
                GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS
                if error.reason
                in {
                    GPUCompositionErrorReason.EXECUTION_INCOMPATIBLE,
                    GPUCompositionErrorReason.INPUT_REQUIREMENTS_UNSATISFIED,
                }
                else GPUPlanningErrorReason.INCOMPLETE_FUSION
            )
            self._state.register_fusion_refusal(chain, reason, str(error))
            return chain[-1]
        options = self._creation_options(chain, core)
        try:
            materialized = materialize_gpu_operator(
                core,
                external_input,
                self._context,
                options,
            )
        except GPUBackendRefusal as error:
            self._state.register_fusion_refusal(
                chain,
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                str(error),
            )
            return chain[-1]
        fused = materialized.operator
        logical_lineage = []
        downstream_logical = self._op_map.get(chain[-1])
        for node in chain:
            lineage = tuple(node._logical_operators)
            if not lineage:
                logical = self._op_map.get(node)
                if logical is not None:
                    lineage = (logical,)
            logical_lineage.extend(lineage)
        if logical_lineage:
            fused.set_logical_operators(*logical_lineage)
        for node in chain:
            self._op_map.pop(node, None)
        if downstream_logical is None and logical_lineage:
            downstream_logical = logical_lineage[-1]
        if downstream_logical is not None:
            self._op_map[fused] = downstream_logical
        self._state.replace_physical(chain, fused)
        self._state.register_materialized_region(
            fused,
            materialized.compiled_spec,
            materialized.backend_name,
            attested=materialized.attested,
        )
        return fused

    @staticmethod
    def _creation_options(
        chain: list[PlannedGPUOperator], core
    ) -> GPUOperatorCreationOptions:
        options = chain[0].gpu_creation_options
        for node in chain[1:]:
            options = options.for_fused_spec(node.gpu_creation_options, core)
        if options.name is None:
            options = options.for_fused_spec(options, core)
        return options


def _reachable_operators(root: PhysicalOperator) -> Iterable[PhysicalOperator]:
    visited: set[PhysicalOperator] = set()
    stack = [root]
    while stack:
        operator = stack.pop()
        if operator in visited:
            continue
        visited.add(operator)
        yield operator
        stack.extend(operator.input_dependencies)


def _rebuild_output_dependencies(root: PhysicalOperator) -> None:
    operators = list(_reachable_operators(root))
    for operator in operators:
        operator._output_dependencies = []
    for operator in operators:
        for input_operator in operator.input_dependencies:
            if operator not in input_operator._output_dependencies:
                input_operator._output_dependencies.append(operator)
