"""Physical optimizer rule for declarative marker-based GPU operators."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ray.data._internal.execution.gpu.composition import (
    GPUCompositionError,
    unresolved_validation_obligations,
)
from ray.data._internal.execution.gpu.operator import (
    GPUOperator,
    create_gpu_operator,
)
from ray.data._internal.execution.gpu.operator_spec import GPUOperatorSpec, PayloadKind
from ray.data._internal.execution.gpu.region_compiler import compile_gpu_region
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule


class GPURequiredFusionError(ValueError):
    """A region whose correctness requires actor-local fusion did not fully fuse."""


class FuseGPUOperators(Rule):
    """Fuse maximal linear chains of compatible :class:`GPUOperator` nodes."""

    @classmethod
    def dependencies(cls):
        from ray.data._internal.logical.rules.inherit_target_max_block_size import (
            InheritTargetMaxBlockSizeRule,
        )

        return [InheritTargetMaxBlockSizeRule]

    @classmethod
    def dependents(cls):
        from ray.data._internal.logical.rules.set_read_parallelism import (
            SetReadParallelismRule,
        )

        return [SetReadParallelismRule]

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        if not getattr(plan.context, "enable_gpu_operator_fusion", False):
            return plan

        self._op_map = plan.op_map.copy()
        self._consumer_counts = _consumer_counts(plan.dag)
        self._composition_failures: Dict[str, GPUCompositionError] = {}
        fused_dag = self._fuse_from_downstream(plan.dag, plan.context)
        _rebuild_output_dependencies(fused_dag)
        result = PhysicalPlan(fused_dag, self._op_map, plan.context)
        self._validate_required_complete_regions(result)
        return result

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
            fused = create_gpu_operator(
                compiled_spec,
                external_inputs[0],
                data_context,
                creation_options,
            )
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

        chain: List[PhysicalOperator] = [downstream]
        compiled_spec = downstream.gpu_operator_spec
        cursor = downstream
        while len(cursor.input_dependencies) == 1:
            upstream = cursor.input_dependencies[0]
            if not isinstance(upstream, GPUOperator):
                break
            if self._consumer_counts.get(upstream, 0) != 1:
                break
            if upstream.execution_backend_name != downstream.execution_backend_name:
                break
            if (
                upstream.gpu_operator_spec.region_id is not None
                and downstream.gpu_operator_spec.region_id is not None
                and upstream.gpu_operator_spec.region_id
                != downstream.gpu_operator_spec.region_id
            ):
                break

            candidate = [upstream] + chain
            try:
                compiled_spec = compile_gpu_region(
                    node.gpu_operator_spec  # type: ignore[attr-defined]
                    for node in candidate
                )
            except GPUCompositionError as exc:
                region_id = (
                    downstream.gpu_operator_spec.region_id
                    or upstream.gpu_operator_spec.region_id
                    or "<anonymous>"
                )
                self._composition_failures.setdefault(region_id, exc)
                break
            chain = candidate
            cursor = upstream

        return chain, compiled_spec

    def _validate_required_complete_regions(self, plan: PhysicalPlan) -> None:
        gpu_ops = [
            op for op in _reachable_operators(plan.dag) if isinstance(op, GPUOperator)
        ]
        required_ops = [
            op for op in gpu_ops if op.gpu_operator_spec.requires_complete_fusion
        ]
        if not required_ops:
            return

        required_region_ids = {
            op.gpu_operator_spec.region_id for op in required_ops
        }
        if None in required_region_ids or len(required_region_ids) != 1:
            raise GPURequiredFusionError(
                "GPU V1 supports exactly one complete-fusion region per Dataset"
            )
        region_id = next(iter(required_region_ids))
        region_ops = [
            op for op in gpu_ops if op.gpu_operator_spec.region_id == region_id
        ]
        if len(region_ops) != 1:
            failure = self._composition_failures.get(region_id)
            detail = str(failure) if failure is not None else (
                "the region contains a branch, boundary, or multiple physical nodes"
            )
            raise GPURequiredFusionError(
                "map_group_partitions GPU region could not be fused into one "
                f"physical operator: {detail}. V1 requires a complete fused "
                "range-read region."
            )
        if len(gpu_ops) != 1:
            raise GPURequiredFusionError(
                "GPU V1 supports only one activated GPU region per Dataset execution"
            )

        op = region_ops[0]
        spec = op.gpu_operator_spec
        if not spec.required_logical_provenance:
            raise GPURequiredFusionError(
                "complete-fusion region is missing required logical provenance"
            )
        if spec.logical_provenance != spec.required_logical_provenance:
            raise GPURequiredFusionError(
                "complete-fusion region has incomplete logical provenance: "
                f"actual={spec.logical_provenance!r}, "
                f"required={spec.required_logical_provenance!r}"
            )
        pending = unresolved_validation_obligations(spec.transforms)
        if pending:
            raise GPURequiredFusionError(
                "complete-fusion region has unresolved validation obligations: "
                f"{pending!r}"
            )
        if spec.input_payload is not PayloadKind.RANGE_WORK:
            raise GPURequiredFusionError(
                "complete-fusion region must consume RangeWork descriptors"
            )
        if spec.output_payload is not PayloadKind.OUTPUT_BATCH_STREAM:
            raise GPURequiredFusionError(
                "complete-fusion region must produce an output batch stream"
            )
        if (
            len(op.input_dependencies) != 1
            or not isinstance(op.input_dependencies[0], InputDataBuffer)
        ):
            raise GPURequiredFusionError(
                "complete-fusion region must consume one InputDataBuffer of RangeWork"
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
