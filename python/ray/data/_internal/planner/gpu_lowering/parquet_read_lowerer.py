"""Generic Read lowerer dispatched through a selected source provider."""

from ray.data._internal.logical.operators import Read
from ray.data._internal.planner.gpu_lowering.interfaces import (
    GPULogicalAnalysis,
    GPULoweredOperator,
    GPULoweringContext,
    GPURequirementDemand,
)
from ray.data._internal.planner.gpu_read_source_provider import (
    GPUReadLoweringRequest,
)


class GPUReadLowerer:
    def analyze(self, logical_op: Read, data_context) -> GPULogicalAnalysis:
        del data_context
        if type(logical_op) is not Read:
            return GPULogicalAnalysis(False, "logical_read")
        return GPULogicalAnalysis(True)

    def transfer_demands(
        self,
        logical_op: Read,
        downstream_demands: tuple[GPURequirementDemand, ...],
        analysis: GPULogicalAnalysis,
    ) -> tuple[tuple[GPURequirementDemand, ...], ...]:
        del logical_op, downstream_demands, analysis
        return ()

    def lower(
        self, logical_op: Read, context: GPULoweringContext
    ) -> GPULoweredOperator:
        if type(logical_op) is not Read:
            raise TypeError("GPUReadLowerer requires an exact Read")
        provider = context.source_provider
        requirements = context.execution_requirements
        if provider is None or requirements is None:
            raise ValueError("GPU Read lowering requires a resolved source demand")
        lowered = provider.lower(
            logical_op,
            GPUReadLoweringRequest(
                provider_state=context.source_analysis,
                execution_requirements=requirements,
                target_max_block_size=context.target_max_block_size,
                retried_io_errors=context.retried_io_errors,
            ),
        )
        return GPULoweredOperator(
            lowered.spec, lowered.creation_options, lowered.artifacts
        )
