"""Focused contracts for independent V1 GPU logical lowerers."""

from types import SimpleNamespace

import pytest

from ray.data import ActorPoolStrategy, UDFProperties
from ray.data._internal.execution.gpu.composition import fuse_gpu_operator_specs
from ray.data._internal.execution.gpu.operator import GPUOperatorCreationOptions
from ray.data._internal.execution.gpu.operator_spec import (
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    GPUOperatorSpec,
    GPUTransformSpec,
    RANGE_DATA,
    RANGE_PARTITION_DATA,
    RANGE_WORK,
)
from ray.data._internal.execution.gpu.properties import (
    KeysColocated,
    RangePartitioning,
    RangePartitioningRequired,
)
from ray.data._internal.execution.gpu.transforms.map_batches import (
    VALIDATE_RANGE_OWNERSHIP_KIND,
)
from ray.data._internal.logical.operators import MapBatches, MapGroupPartitions, Read
from ray.data._internal.planner.gpu_lowering.interfaces import GPULoweringContext
from ray.data._internal.planner.gpu_lowering.map_batches_lowerer import (
    GPUMapBatchesLowerer,
)
from ray.data._internal.planner.gpu_lowering.map_group_partitions_lowerer import (
    GPUMapGroupPartitionsLowerer,
)
from ray.data._internal.planner.gpu_lowering.parquet_read_lowerer import GPUReadLowerer
from ray.data._internal.planner.gpu_planning_errors import (
    GPUPlanningError,
    GPUPlanningErrorReason,
)
from ray.data._internal.planner.gpu_read_source_provider import (
    GPUReadSourceArtifacts,
    GPUReadSourceDemand,
    GPUReadSourceLowering,
)
from ray.data._internal.planner.gpu_requirement_analysis import RangePartitionDemand
from ray.data.context import DataContext
from ray.data.datasource.datasource import Datasource


class _Source(Datasource):
    pass


class _GPUMap:
    def __call__(self, batch):
        return batch


def _partition_fn(frame, context):
    del context
    return frame


def _read():
    source = _Source()
    return Read(
        datasource=source,
        datasource_or_legacy_reader=source,
        parallelism=1,
    )


def _map(parent, *, preserves=True, batch_format="cudf"):
    return MapBatches(
        fn=_GPUMap,
        input_dependencies=[parent],
        batch_size=8,
        batch_format=batch_format,
        compute=ActorPoolStrategy(size=1),
        ray_remote_args={"num_cpus": 1, "num_gpus": 1},
        udf_properties=(
            UDFProperties(preserves_partitioning=True) if preserves else None
        ),
    )


def _mgp(parent):
    return MapGroupPartitions(
        key=["user", "card"],
        num_partitions=2,
        fn=_partition_fn,
        input_dependencies=[parent],
    )


def _requirements():
    return GPUExecutionRequirements(
        compatibility_key=GPUExecutionCompatibilityKey(
            min_workers=1,
            initial_workers=1,
            max_workers=1,
            num_cpus=1,
            num_gpus=1,
        )
    )


def _facts(scheme="range-scheme"):
    return (RangePartitioning(("user",), scheme),)


def _context(*, input_facts=(), execution=None, source_provider=None, demand=None):
    return GPULoweringContext(
        target_max_block_size=1024,
        input_facts=input_facts,
        execution_requirements=execution or _requirements(),
        source_provider=source_provider,
        source_analysis=demand,
        retried_io_errors=("network",),
    )


def test_map_analysis_is_local_and_accepts_nonpreserving_gpu_map():
    read = _read()
    context = DataContext.get_current()
    preserving = GPUMapBatchesLowerer().analyze(_map(read), context)
    nonpreserving = GPUMapBatchesLowerer().analyze(_map(read, preserves=False), context)
    non_cudf = GPUMapBatchesLowerer().analyze(
        _map(read, batch_format="pyarrow"), context
    )
    assert preserving.accepted
    assert nonpreserving.accepted
    assert not non_cudf.accepted


def test_preserving_map_uses_only_input_facts_and_appends_validation():
    lowered = GPUMapBatchesLowerer().lower(
        _map(_read()),
        _context(input_facts=_facts()),
    )
    spec = lowered.spec
    assert tuple(transform.kind for transform in spec.transforms) == (
        "map_batches",
        VALIDATE_RANGE_OWNERSHIP_KIND,
    )
    assert spec.input_requirements == (
        RangePartitioningRequired(("user",), "range-scheme"),
    )
    assert spec.output_facts == _facts()
    assert spec.input_payload == RANGE_DATA
    assert spec.output_payload == RANGE_DATA
    assert lowered.source_artifacts is None


def test_preserving_map_reports_conflicting_range_facts_stably():
    facts = _facts() + (RangePartitioning(("other",), "other-scheme"),)
    with pytest.raises(GPUPlanningError) as captured:
        GPUMapBatchesLowerer().lower(
            _map(_read()),
            _context(input_facts=facts),
        )
    assert captured.value.reason is GPUPlanningErrorReason.CONFLICTING_REQUIREMENTS


def test_map_without_known_fact_lowers_independently_and_advertises_nothing():
    preserving = GPUMapBatchesLowerer().lower(_map(_read()), _context()).spec
    nonpreserving = (
        GPUMapBatchesLowerer()
        .lower(
            _map(_read(), preserves=False),
            _context(input_facts=_facts()),
        )
        .spec
    )
    assert tuple(transform.kind for transform in preserving.transforms) == (
        "map_batches",
    )
    assert preserving.input_requirements == preserving.output_facts == ()
    assert nonpreserving.input_requirements == nonpreserving.output_facts == ()


def test_map_demand_transfer_is_fact_generic_not_mgp_aware():
    logical = _map(_read())
    lowerer = GPUMapBatchesLowerer()
    analysis = lowerer.analyze(logical, DataContext.get_current())
    demand = RangePartitionDemand(("user", "card"), 4)
    transferred = lowerer.transfer_demands(logical, (demand,), analysis)
    assert transferred[0][0].keys == ("user", "card")
    assert transferred[0][0].execution_requirements == analysis.execution_requirements
    nonpreserving = _map(_read(), preserves=False)
    assert lowerer.transfer_demands(
        nonpreserving,
        (demand,),
        lowerer.analyze(nonpreserving, DataContext.get_current()),
    ) == ((),)


def test_mgp_emits_range_demand_and_lowers_with_inherited_execution():
    logical = _mgp(_map(_read()))
    lowerer = GPUMapGroupPartitionsLowerer()
    analysis = lowerer.analyze(logical, DataContext.get_current())
    transfer = lowerer.transfer_demands(logical, (), analysis)
    assert transfer == ((RangePartitionDemand(("user", "card"), 2),),)
    spec = lowerer.lower(logical, _context(input_facts=_facts())).spec
    assert tuple(transform.kind for transform in spec.transforms) == (
        "sort_partition",
        "build_group_boundaries",
        "map_group_partitions",
    )
    assert spec.input_requirements == (KeysColocated(("user", "card")),)
    assert spec.output_facts == ()
    assert spec.transforms[0].output_payload == RANGE_PARTITION_DATA


def test_read_lowerer_uses_resolved_demand_and_plan_local_provider():
    artifacts = GPUReadSourceArtifacts(
        input_data_factory=lambda _target_size: [],
        provider_artifacts=SimpleNamespace(work=("range-0", "range-1")),
    )
    spec = GPUOperatorSpec(
        transforms=(GPUTransformSpec("read_parquet_range", RANGE_WORK, RANGE_DATA),),
        execution_requirements=_requirements(),
        output_facts=_facts(),
    )

    class _Provider:
        def __init__(self):
            self.request = None

        def lower(self, logical_op, request):
            assert type(logical_op) is Read
            self.request = request
            return GPUReadSourceLowering(
                spec,
                GPUOperatorCreationOptions(name="GPURead"),
                artifacts,
            )

    provider = _Provider()
    demand = GPUReadSourceDemand(("user", "card"), 2, 1)
    lowered = GPUReadLowerer().lower(
        _read(),
        _context(source_provider=provider, demand=demand),
    )
    assert lowered.spec is spec
    assert lowered.source_artifacts is artifacts
    assert provider.request.provider_state is demand


def test_independent_read_map_mgp_specs_form_one_pure_composition_contract():
    read = GPUOperatorSpec(
        transforms=(GPUTransformSpec("read", RANGE_WORK, RANGE_DATA),),
        execution_requirements=_requirements(),
        output_facts=_facts(),
    )
    mapped = (
        GPUMapBatchesLowerer()
        .lower(
            _map(_read()),
            _context(input_facts=read.output_facts),
        )
        .spec
    )
    grouped = (
        GPUMapGroupPartitionsLowerer()
        .lower(
            _mgp(_map(_read())),
            _context(input_facts=mapped.output_facts),
        )
        .spec
    )
    composed = fuse_gpu_operator_specs(
        fuse_gpu_operator_specs(read, mapped),
        grouped,
    )
    assert tuple(transform.kind for transform in composed.transforms) == (
        "read",
        "map_batches",
        "validate_range_ownership",
        "sort_partition",
        "build_group_boundaries",
        "map_group_partitions",
    )
    assert composed.output_facts == ()


def test_preserving_map_rejects_conflicting_range_facts():
    with pytest.raises(ValueError, match="conflicting range partitioning"):
        GPUMapBatchesLowerer().lower(
            _map(_read()),
            _context(input_facts=_facts("one") + _facts("two")),
        )
