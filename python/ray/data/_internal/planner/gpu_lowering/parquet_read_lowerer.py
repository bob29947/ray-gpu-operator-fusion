"""Generic Read lowerer dispatched through the selected source provider."""

from ray.data._internal.logical.operators import Read
from ray.data._internal.planner.gpu_lowering.interfaces import (
    GPULogicalAnalysis,
    GPULoweredOperator,
    GPULoweringContext,
)
from ray.data._internal.planner.gpu_read_source_provider import (
    GPUReadLoweringRequest,
)


class GPUReadLowerer:
    def analyze(self, logical_op: Read, data_context) -> GPULogicalAnalysis:
        del data_context
        if type(logical_op) is not Read:
            return GPULogicalAnalysis(False, "logical_read")
        # Full source recognition and concrete range work remain inside lower().
        return GPULogicalAnalysis(True, None)

    def lower(
        self, logical_op: Read, context: GPULoweringContext
    ) -> GPULoweredOperator:
        if type(logical_op) is not Read:
            raise TypeError("GPUReadLowerer requires an exact Read")
        if context.source_artifacts is not None:
            raise ValueError("Read lowering requires an unplanned source context")

        lowered = context.source_provider.lower(
            logical_op,
            GPUReadLoweringRequest(
                logical_node_id=context.logical_node_id,
                group_keys=context.group_keys,
                execution_requirements=context.execution_requirements,
                target_max_block_size=context.target_max_block_size,
                retried_io_errors=context.retried_io_errors,
                target_ranges=context.target_ranges,
                num_workers=context.num_workers,
            ),
        )
        downstream_context = context.for_lowered_output(
            lowered.spec,
            source_artifacts=lowered.artifacts,
        )
        return GPULoweredOperator(
            spec=lowered.spec,
            creation_options=lowered.creation_options,
            downstream_context=downstream_context,
            source_artifacts=lowered.artifacts,
        )


# Compatibility name for the original registry construction.  The
# implementation itself is source-agnostic; adding a source registers a provider.
ParquetRangeReadLowerer = GPUReadLowerer


__all__ = ["GPUReadLowerer", "ParquetRangeReadLowerer"]
