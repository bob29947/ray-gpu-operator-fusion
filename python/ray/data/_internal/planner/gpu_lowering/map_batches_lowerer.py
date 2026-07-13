import functools
import inspect
import numbers

from ray.data._internal.execution.gpu import (
    GPUOperatorCreationOptions,
    GPUOperatorSpec,
    GPUTransformSpec,
    PayloadKind,
    RANGE_DATA_BOUNDARY_ADAPTER_KEY,
    RANGE_DATA_TERMINAL_ADAPTER_KEY,
)
from ray.data._internal.execution.gpu.properties import (
    PartitioningPreservationError,
    readvertise_preserved_properties,
)
from ray.data._internal.execution.gpu.transforms.map_batches import (
    MAP_BATCHES_KIND,
    MapBatchesConfig,
)
from ray.data._internal.logical.operators import MapBatches
from ray.data._internal.planner.gpu_lowering.interfaces import (
    GPUComponentRole,
    GPULogicalAnalysis,
    GPULoweredOperator,
    GPULoweringContext,
)
from ray.data._internal.planner.gpu_execution_profile import (
    canonical_gpu_execution_requirements,
)
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)


class GPUMapBatchesLowerer:
    component_role = GPUComponentRole.UNARY

    def analyze(self, logical_op: MapBatches, data_context) -> GPULogicalAnalysis:
        if type(logical_op) is not MapBatches:
            return GPULogicalAnalysis(False, "logical_map_batches")
        supported = (
            inspect.isclass(logical_op.fn)
            and _is_synchronous(logical_op.fn)
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
            None,
            execution_requirements=requirements,
            preserves_partitioning=bool(
                logical_op.udf_properties is not None
                and logical_op.udf_properties.preserves_partitioning
            ),
        )

    def lower(
        self, logical_op: MapBatches, context: GPULoweringContext
    ) -> GPULoweredOperator:
        if type(logical_op) is not MapBatches:
            raise TypeError("GPUMapBatchesLowerer requires an exact MapBatches")
        if context.source_artifacts is None:
            raise ValueError("GPU MapBatches lowering requires Read source artifacts")

        preserves = bool(
            logical_op.udf_properties is not None
            and logical_op.udf_properties.preserves_partitioning
        )
        try:
            readvertisement = readvertise_preserved_properties(
                context.input_properties,
                preserves_partitioning=preserves,
                # MapBatches keeps one incoming planner unit intact even when
                # its UDF invalidates the semantic partitioning property.
                preserves_complete_partition_bundles=True,
            )
        except PartitioningPreservationError as error:
            raise GPUPlanningError(
                GPUPlanningErrorReason.UNSUPPORTED_CAPABILITY,
                f"consumer=MapBatches({logical_op.name}); {error}",
            ) from error

        transforms = [
            GPUTransformSpec(
                transform_id=context.transform_id(0, MAP_BATCHES_KIND),
                kind=MAP_BATCHES_KIND,
                input_payload=PayloadKind.RANGE_DATA,
                output_payload=PayloadKind.RANGE_DATA,
                config=MapBatchesConfig.from_logical_op(logical_op),
                creates_obligations=readvertisement.created_obligations,
            )
        ]
        for ordinal, validation in enumerate(
            readvertisement.partitioning_validations,
            start=1,
        ):
            transforms.append(
                GPUTransformSpec(
                    transform_id=context.transform_id(ordinal, validation.kind),
                    kind=validation.kind,
                    input_payload=PayloadKind.RANGE_DATA,
                    output_payload=PayloadKind.RANGE_DATA,
                    config=validation.config,
                    discharges_obligations=validation.obligations,
                )
            )

        spec = GPUOperatorSpec(
            spec_instance_id=context.logical_node_id,
            transforms=tuple(transforms),
            external_input_payload=PayloadKind.RAY_BLOCK_STREAM,
            external_output_payload=PayloadKind.RAY_BLOCK_STREAM,
            native_input_payload=PayloadKind.RANGE_DATA,
            native_output_payload=PayloadKind.RANGE_DATA,
            input_adapter_key=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
            output_adapter_key=RANGE_DATA_BOUNDARY_ADAPTER_KEY,
            terminal_output_adapter_key=RANGE_DATA_TERMINAL_ADAPTER_KEY,
            execution_requirements=context.execution_requirements,
            output_properties=readvertisement.output_properties,
            logical_provenance=(context.logical_node_id,),
        )
        downstream_context = context.for_lowered_output(spec)
        return GPULoweredOperator(
            spec=spec,
            creation_options=GPUOperatorCreationOptions(
                name="GPUMapBatches",
                target_max_block_size_override=context.target_max_block_size,
            ),
            downstream_context=downstream_context,
            source_artifacts=context.source_artifacts,
        )


def _is_synchronous(callable_) -> bool:
    if not callable(callable_):
        return False
    if inspect.isroutine(callable_) or isinstance(callable_, functools.partial):
        function = callable_
    else:
        function = callable_.__call__
    return not inspect.iscoroutinefunction(function) and not inspect.isasyncgenfunction(
        function
    )
