from ray.data._internal.execution.gpu import (
    Evidence,
    GPUBoundaryPolicy,
    GPUOperatorCreationOptions,
    GPUOperatorSpec,
    GPUTransformSpec,
    PayloadKind,
    PropertyFact,
    RangePartitioning,
)
from ray.data._internal.execution.gpu.transforms.parquet_read import (
    READ_PARQUET_RANGE_KIND,
    ReadParquetRangeConfig,
)
from ray.data._internal.logical.operators import Read
from ray.data._internal.planner.gpu_lowering.interfaces import (
    GPULoweredOperator,
    GPULoweringContext,
)


class ParquetRangeReadLowerer:
    def lower(
        self, logical_op: Read, context: GPULoweringContext
    ) -> GPULoweredOperator:
        if type(logical_op) is not Read:
            raise TypeError("ParquetRangeReadLowerer requires an exact Read")
        transform = GPUTransformSpec(
            transform_id="read_parquet_range",
            kind=READ_PARQUET_RANGE_KIND,
            input_payload=PayloadKind.RANGE_WORK,
            output_payload=PayloadKind.CUDF_BATCH_STREAM,
            config=ReadParquetRangeConfig(
                source_kind=context.read_descriptor.source_kind,
                region=context.read_descriptor.region,
                projection=context.read_descriptor.projection,
                range_key=context.range_plan.range_key,
                retried_io_errors=context.retried_io_errors,
            ),
        )
        spec = GPUOperatorSpec(
            transforms=(transform,),
            input_payload=PayloadKind.RANGE_WORK,
            output_payload=PayloadKind.CUDF_BATCH_STREAM,
            execution_requirements=context.execution_requirements,
            output_properties=(
                PropertyFact(
                    RangePartitioning(
                        (context.range_plan.range_key,),
                        context.range_plan.scheme_id,
                    ),
                    Evidence.OPERATOR_GUARANTEE,
                ),
            ),
            input_boundary=GPUBoundaryPolicy.REQUIRED,
            output_boundary=GPUBoundaryPolicy.ELIDABLE,
            logical_provenance=("Read",),
            required_logical_provenance=context.required_logical_provenance,
            region_id=context.region_id,
            requires_complete_fusion=True,
        )
        return GPULoweredOperator(
            spec,
            GPUOperatorCreationOptions(
                name="GPUReadParquetRange",
                min_rows_per_bundle=1,
                target_max_block_size_override=context.target_max_block_size,
            ),
        )
