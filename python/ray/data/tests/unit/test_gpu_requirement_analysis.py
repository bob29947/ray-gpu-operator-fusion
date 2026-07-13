from ray.data import ActorPoolStrategy, UDFProperties
from ray.data._internal.logical.operators import (
    Limit,
    MapBatches,
    MapGroupPartitions,
    Read,
    Union,
)
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.planner.gpu_lowering import (
    GPUComponentRole,
    GPULogicalAnalysis,
)
from ray.data._internal.planner.gpu_requirement_analysis import (
    GPUEnforcerRegistry,
    GPUPlanningError,
    GPUPlanningErrorReason,
    GPUPropertyState,
    analyze_gpu_requirements,
    create_default_gpu_lowerer_registry,
)
from ray.data._internal.execution.gpu import (
    CompletePartitionBundles,
    Evidence,
    GPUExecutionCompatibilityKey,
    GPUExecutionRequirements,
    PropertyFact,
    RangePartitioning,
)
from ray.data._internal.planner.gpu_read_source_provider import (
    GPUReadSourceProviderRegistry,
    GPUReadSourceSupport,
)
from ray.data.context import DataContext
from ray.data.datasource.datasource import Datasource


class _Source(Datasource):
    pass


class _GPUMap:
    def __call__(self, batch):
        return batch


class _CustomLogicalSource(LogicalOperator):
    def __init__(self):
        object.__setattr__(self, "_input_dependencies", [])
        object.__setattr__(self, "_name", "CustomGPUInput")


class _CustomSourceLowerer:
    component_role = GPUComponentRole.SOURCE

    def __init__(self, execution_requirements):
        self._execution_requirements = execution_requirements

    def analyze(self, logical_op, data_context):
        del data_context
        if type(logical_op) is not _CustomLogicalSource:
            return GPULogicalAnalysis(False, "custom_source")
        return GPULogicalAnalysis(
            True,
            None,
            execution_requirements=self._execution_requirements,
        )

    def lower(self, logical_op, context):
        del logical_op, context
        raise AssertionError("requirement analysis must not lower a source")


class _TestSourceProvider:
    provider_name = "test"

    def supports(self, logical_op, demand):
        del logical_op, demand
        return GPUReadSourceSupport(True)

    def lower(self, logical_op, request):
        del logical_op, request
        raise AssertionError("requirement analysis must not lower a source")


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


def _map(parent, *, preserves=True, size=1):
    return MapBatches(
        fn=_GPUMap,
        input_dependencies=[parent],
        batch_size=8,
        batch_format="cudf",
        compute=ActorPoolStrategy(size=size),
        ray_remote_args={"num_cpus": 1, "num_gpus": 1},
        udf_properties=(
            UDFProperties(preserves_partitioning=True) if preserves else None
        ),
    )


def _mgp(parent, *, keys=("user", "card"), preserves=False, count=2):
    return MapGroupPartitions(
        key=list(keys),
        num_partitions=count,
        fn=_partition_fn,
        udf_properties=(
            UDFProperties(preserves_partitioning=True) if preserves else None
        ),
        input_dependencies=[parent],
    )


def _analyze(root, *, enforcers=None):
    providers = GPUReadSourceProviderRegistry()
    providers.register(_TestSourceProvider())
    return analyze_gpu_requirements(
        root,
        DataContext.get_current(),
        source_providers=providers,
        enforcers=enforcers,
    )


def test_repeated_mgp_is_one_maximal_component():
    read = _read()
    mapped = _map(read)
    first = _mgp(mapped, preserves=True)
    last = _mgp(first)

    analysis = _analyze(last)

    assert tuple(analysis.components_by_anchor) == (last,)
    component = analysis.components_by_anchor[last]
    assert component.operators == (read, mapped, first, last)
    assert [requirement.operator for requirement in component.mgp_requirements] == [
        first,
        last,
    ]
    assert [
        membership.component_id
        for membership in analysis.annotations[first].memberships
    ] == [component.component_id]


def test_trailing_map_is_stock_and_not_component_member():
    read = _read()
    mapped = _map(read)
    grouped = _mgp(mapped)
    trailing = _map(grouped, preserves=True)

    analysis = _analyze(trailing)

    assert tuple(analysis.components_by_anchor) == (grouped,)
    assert analysis.components_by_anchor[grouped].operators == (
        read,
        mapped,
        grouped,
    )
    assert analysis.annotations[trailing].memberships == ()


def test_invalidation_before_later_mgp_requires_exchange():
    read = _read()
    first = _mgp(_map(read), preserves=True)
    invalidating_map = _map(first, preserves=False)
    last = _mgp(invalidating_map)

    try:
        _analyze(last)
    except GPUPlanningError as error:
        assert error.reason is GPUPlanningErrorReason.EXCHANGE_REQUIRED
        assert "first_invalidator=MapBatches" in str(error)
        assert "no GPU exchange is registered" in str(error)
    else:
        raise AssertionError("property invalidation must refuse the later MGP")


def test_branch_ending_at_earlier_mgp_gets_its_own_component():
    read = _read()
    first = _mgp(_map(read), preserves=True)
    continued = _mgp(first)
    stock_branch = Limit(1, input_dependencies=[first])
    root = Union([continued, stock_branch])

    analysis = _analyze(root)

    assert set(analysis.components_by_anchor) == {first, continued}
    assert analysis.components_by_anchor[first].operators[-1] is first
    assert analysis.components_by_anchor[continued].operators[-1] is continued
    assert len(analysis.annotations[first].memberships) == 2


class _RecordingEnforcer:
    def __init__(self):
        self.requests = []

    def supports(self, request):
        del request
        return True

    def established_state(self, request):
        self.requests.append(request)
        return GPUPropertyState(
            facts=(
                PropertyFact(
                    RangePartitioning((request.required_keys[0],), "test-exchange"),
                    Evidence.OPERATOR_GUARANTEE,
                ),
                PropertyFact(
                    CompletePartitionBundles(),
                    Evidence.OPERATOR_GUARANTEE,
                ),
            ),
            partition_count=request.required_partition_count,
        )

    def plan(self, request, *, upstream, data_context):
        del request, upstream, data_context
        raise AssertionError("analysis must not construct an enforcer")


def test_every_adjacent_execution_profile_boundary_is_annotated():
    read = _read()
    profile_a = _map(read, size=1)
    profile_b = _map(profile_a, size=2)
    profile_a_again = _map(profile_b, size=1)
    grouped = _mgp(profile_a_again, count=None)
    enforcer = _RecordingEnforcer()
    registry = GPUEnforcerRegistry()
    registry.register(enforcer)

    analysis = _analyze(grouped, enforcers=registry)
    component = analysis.components_by_anchor[grouped]

    assert len(enforcer.requests) == 2
    assert [
        (
            request.upstream_execution_requirements.compatibility_key.max_workers,
            request.downstream_execution_requirements.compatibility_key.max_workers,
        )
        for request in enforcer.requests
    ] == [(1, 2), (2, 1)]
    assert component.mgp_requirements[0].partition_count == 2
    assert [
        analysis.annotations[op]
        .memberships[0]
        .execution_requirements.compatibility_key.max_workers
        for op in (read, profile_a, profile_b, profile_a_again, grouped)
    ] == [1, 1, 2, 1, 1]


def test_registered_custom_source_is_classified_without_read_fields():
    source = _CustomLogicalSource()
    grouped = _mgp(source, count=None)
    source_requirements = GPUExecutionRequirements(
        compatibility_key=GPUExecutionCompatibilityKey(
            min_workers=2,
            initial_workers=2,
            max_workers=2,
            num_cpus=1,
            num_gpus=1,
        )
    )
    lowerers = create_default_gpu_lowerer_registry()
    lowerers.register(
        _CustomLogicalSource,
        _CustomSourceLowerer(source_requirements),
    )
    providers = GPUReadSourceProviderRegistry()
    providers.register(_TestSourceProvider())

    analysis = analyze_gpu_requirements(
        grouped,
        DataContext.get_current(),
        lowerers=lowerers,
        source_providers=providers,
    )

    component = analysis.components_by_anchor[grouped]
    assert component.source is source
    assert component.operators == (source, grouped)
    assert component.execution_requirements == source_requirements
    assert component.source_demand.source is source
    assert component.mgp_requirements[0].partition_count == 4


def test_registered_custom_source_can_inherit_first_unary_profile():
    source = _CustomLogicalSource()
    mapped = _map(source, size=2)
    grouped = _mgp(mapped, count=None)
    lowerers = create_default_gpu_lowerer_registry()
    lowerers.register(
        _CustomLogicalSource,
        _CustomSourceLowerer(None),
    )
    providers = GPUReadSourceProviderRegistry()
    providers.register(_TestSourceProvider())

    analysis = analyze_gpu_requirements(
        grouped,
        DataContext.get_current(),
        lowerers=lowerers,
        source_providers=providers,
    )

    component = analysis.components_by_anchor[grouped]
    assert [
        analysis.annotations[op]
        .memberships[0]
        .execution_requirements.compatibility_key.max_workers
        for op in (source, mapped, grouped)
    ] == [2, 2, 2]
    assert component.source_demand.execution_requirements == (
        analysis.annotations[mapped].memberships[0].execution_requirements
    )
    assert component.mgp_requirements[0].partition_count == 4
