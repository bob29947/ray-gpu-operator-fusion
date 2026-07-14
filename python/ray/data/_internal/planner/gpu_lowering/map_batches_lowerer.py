"""Independent GPU lowering for one exact MapBatches occurrence."""

import inspect
import numbers

from ray.data._internal.execution.gpu.operator import GPUOperatorCreationOptions
from ray.data._internal.execution.gpu.operator_spec import (
    GPUOperatorSpec,
    GPUTransformSpec,
    RANGE_DATA,
    preserve_range,
)
from ray.data._internal.execution.gpu.properties import (
    PropertyConflict,
    RangePartitioning,
)
from ray.data._internal.execution.gpu.transforms.map_batches import (
    MAP_BATCHES_KIND,
    MapBatchesConfig,
    ValidateRangeOwnershipConfig,
)
from ray.data._internal.logical.operators import MapBatches
from ray.data._internal.planner.gpu_execution_profile import (
    canonical_gpu_execution_requirements,
)
from ray.data._internal.planner.gpu_lowering.interfaces import (
    GPULogicalAnalysis,
    GPULoweredOperator,
    GPULoweringContext,
    GPURequirementDemand,
    is_synchronous_callable,
)
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)
from ray.data._internal.planner.gpu_requirement_analysis import RangePartitionDemand


class GPUMapBatchesLowerer:
    def analyze(self, logical_op: MapBatches, data_context) -> GPULogicalAnalysis:
        if type(logical_op) is not MapBatches:
            return GPULogicalAnalysis(False, "logical_map_batches")
        supported = (
            inspect.isclass(logical_op.fn)
            and is_synchronous_callable(logical_op.fn)
            and logical_op.batch_format == "cudf"
            and isinstance(logical_op.batch_size, numbers.Integral)
            and not isinstance(logical_op.batch_size, bool)
            and logical_op.batch_size > 0
            and logical_op.per_block_limit is None
            and logical_op.ray_remote_args_fn is None
        )
        if not supported:
            return GPULogicalAnalysis(
                False,
                "GPU MapBatches requires a synchronous callable class, "
                "batch_format='cudf', a positive integer batch_size, and no "
                "dynamic remote arguments or per-block limit",
            )
        try:
            requirements = canonical_gpu_execution_requirements(
                compute=logical_op.compute,
                resources=logical_op.ray_remote_args,
                data_context=data_context,
                consumer=f"MapBatches({logical_op.name})",
            )
        except GPUPlanningError as error:
            return GPULogicalAnalysis(False, error.detail)
        return GPULogicalAnalysis(
            True,
            execution_requirements=requirements,
        )

    def transfer_demands(
        self,
        logical_op: MapBatches,
        downstream_demands: tuple[GPURequirementDemand, ...],
        analysis: GPULogicalAnalysis,
    ) -> tuple[tuple[GPURequirementDemand, ...], ...]:
        if len(logical_op.input_dependencies) != 1:
            return tuple(() for _ in logical_op.input_dependencies)
        preserves = bool(
            logical_op.udf_properties is not None
            and logical_op.udf_properties.preserves_partitioning
        )
        if not preserves:
            return ((),)
        requirements = analysis.execution_requirements
        if requirements is None:
            raise AssertionError("accepted GPU MapBatches must declare execution")
        transferred = tuple(
            demand.with_execution(requirements)
            if isinstance(demand, RangePartitionDemand)
            else demand
            for demand in downstream_demands
        )
        return (transferred,)

    def lower(
        self, logical_op: MapBatches, context: GPULoweringContext
    ) -> GPULoweredOperator:
        if type(logical_op) is not MapBatches:
            raise TypeError("GPUMapBatchesLowerer requires an exact MapBatches")
        requirements = context.execution_requirements
        if requirements is None or requirements.compatibility_key is None:
            raise ValueError("GPU MapBatches requires a concrete execution profile")
        preserves = bool(
            logical_op.udf_properties is not None
            and logical_op.udf_properties.preserves_partitioning
        )
        range_fact = next(
            (
                fact
                for fact in context.input_facts
                if preserves and isinstance(fact, RangePartitioning)
            ),
            None,
        )
        validation_config = (
            ValidateRangeOwnershipConfig(range_fact.keys[0], range_fact.keys)
            if range_fact is not None
            else None
        )
        try:
            preservation = preserve_range(
                context.input_facts if preserves else (), RANGE_DATA, validation_config
            )
        except PropertyConflict as error:
            raise GPUPlanningError(
                GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS,
                str(error),
            ) from error
        spec = GPUOperatorSpec(
            transforms=(
                GPUTransformSpec(
                    kind=MAP_BATCHES_KIND,
                    input_payload=RANGE_DATA,
                    output_payload=RANGE_DATA,
                    config=MapBatchesConfig.from_logical_op(logical_op),
                ),
                *preservation.validation_transforms,
            ),
            execution_requirements=requirements,
            input_requirements=preservation.input_requirements,
            output_facts=preservation.output_facts,
        )
        return GPULoweredOperator(
            spec=spec,
            creation_options=GPUOperatorCreationOptions(
                name="GPUMapBatches",
                target_max_block_size_override=context.target_max_block_size,
            ),
        )
