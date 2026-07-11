from ray.data._internal.execution.gpu import (
    Evidence,
    GPUBoundaryPolicy,
    GPUOperatorCreationOptions,
    GPUOperatorSpec,
    GPUTransformSpec,
    KeysColocated,
    PayloadKind,
    PropertyFact,
    RangePartitioning,
    RowsOwnedByRange,
)
from ray.data._internal.execution.gpu.transforms.map_batches import (
    MAP_BATCHES_KIND,
    VALIDATE_RANGE_OWNERSHIP_KIND,
    MapBatchesConfig,
    ValidateRangeOwnershipConfig,
)
from ray.data._internal.logical.operators import MapBatches
from ray.data._internal.planner.gpu_lowering.interfaces import (
    GPULoweredOperator,
    GPULoweringContext,
)


class GPUMapBatchesLowerer:
    def lower(
        self, logical_op: MapBatches, context: GPULoweringContext
    ) -> GPULoweredOperator:
        if type(logical_op) is not MapBatches:
            raise TypeError("GPUMapBatchesLowerer requires an exact MapBatches")
        obligation = RowsOwnedByRange(
            context.range_plan.range_key,
            context.range_plan.scheme_id,
        )
        map_transform = GPUTransformSpec(
            transform_id="map_batches",
            kind=MAP_BATCHES_KIND,
            input_payload=PayloadKind.CUDF_BATCH_STREAM,
            output_payload=PayloadKind.CUDF_BATCH_STREAM,
            config=MapBatchesConfig.from_logical_op(logical_op),
            creates_obligations=(obligation,),
        )
        validate_transform = GPUTransformSpec(
            transform_id="validate_range_ownership",
            kind=VALIDATE_RANGE_OWNERSHIP_KIND,
            input_payload=PayloadKind.CUDF_BATCH_STREAM,
            output_payload=PayloadKind.CUDF_BATCH_STREAM,
            config=ValidateRangeOwnershipConfig(
                context.range_plan.range_key,
                context.group_keys,
            ),
            discharges_obligations=(obligation,),
        )
        range_property = PropertyFact(
            RangePartitioning(
                (context.range_plan.range_key,),
                context.range_plan.scheme_id,
            ),
            Evidence.OPERATOR_GUARANTEE,
        )
        spec = GPUOperatorSpec(
            transforms=(map_transform, validate_transform),
            input_payload=PayloadKind.CUDF_BATCH_STREAM,
            output_payload=PayloadKind.CUDF_BATCH_STREAM,
            execution_requirements=context.execution_requirements,
            input_requirements=(KeysColocated((context.range_plan.range_key,)),),
            output_properties=(range_property,),
            input_boundary=GPUBoundaryPolicy.ELIDABLE,
            output_boundary=GPUBoundaryPolicy.ELIDABLE,
            logical_provenance=("MapBatches",),
            required_logical_provenance=context.required_logical_provenance,
            region_id=context.region_id,
            requires_complete_fusion=True,
        )
        return GPULoweredOperator(
            spec,
            GPUOperatorCreationOptions(
                name="GPUMapBatches",
                target_max_block_size_override=context.target_max_block_size,
            ),
        )

