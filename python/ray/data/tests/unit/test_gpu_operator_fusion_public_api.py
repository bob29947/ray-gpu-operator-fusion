import inspect
from dataclasses import FrozenInstanceError

import pytest

import ray.data
from ray.data import MapGroupPartitionsContext, UDFProperties
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators import (
    InputData,
    MapBatches,
    MapGroupPartitions,
)
from ray.data.context import DataContext
from ray.data.dataset import Dataset
from ray.data.grouped_data import GroupedData


def identity(batch, context=None):
    return batch


class _FakeDataset:
    _map_batches_without_batch_size_validation = (
        Dataset._map_batches_without_batch_size_validation
    )

    def __init__(self, logical_plan, context):
        self._logical_plan = logical_plan
        self._context = context

    @property
    def context(self):
        return self._context


@pytest.fixture
def source_dataset(monkeypatch):
    context = DataContext()

    def from_parent(parent, logical_plan):
        return _FakeDataset(logical_plan, parent.context)

    monkeypatch.setattr(Dataset, "_from_parent", staticmethod(from_parent))
    source_op = InputData(input_data=[])
    return _FakeDataset(LogicalPlan(source_op, context), context)


def test_gpu_operator_fusion_flag_defaults_false():
    assert DataContext().enable_gpu_operator_fusion is False


def test_udf_properties_is_minimal_frozen_public_contract():
    properties = UDFProperties(preserves_partitioning=True)

    assert ray.data.UDFProperties is UDFProperties
    assert properties.preserves_partitioning is True
    assert list(inspect.signature(UDFProperties).parameters) == [
        "preserves_partitioning"
    ]
    with pytest.raises(FrozenInstanceError):
        properties.preserves_partitioning = False


@pytest.mark.parametrize("properties", [None, UDFProperties(), UDFProperties(True)])
def test_map_batches_plumbs_udf_properties(source_dataset, properties):
    dataset = Dataset.map_batches(
        source_dataset,
        identity,
        batch_size=None,
        udf_properties=properties,
    )

    op = dataset._logical_plan.dag
    assert isinstance(op, MapBatches)
    assert op.udf_properties is properties
    assert op.can_modify_num_rows is True


def test_map_batches_legacy_call_defaults_to_no_udf_properties(source_dataset):
    dataset = Dataset.map_batches(source_dataset, identity, batch_size=None)

    op = dataset._logical_plan.dag
    assert isinstance(op, MapBatches)
    assert op.udf_properties is None


def test_map_group_partitions_builds_logical_operator(source_dataset):
    dataset = GroupedData(
        source_dataset,
        ["User", "Card"],
        num_partitions=8,
    ).map_group_partitions(identity)

    op = dataset._logical_plan.dag
    assert isinstance(op, MapGroupPartitions)
    assert op.input_dependencies == [source_dataset._logical_plan.dag]
    assert op.key == ["User", "Card"]
    assert op.num_partitions == 8
    assert op.fn is identity
    assert op.num_outputs == 8
    assert op.name == "MapGroupPartitions"


def test_map_group_partitions_contract_is_documented():
    doc = inspect.getdoc(GroupedData.map_group_partitions)

    assert "enable_gpu_operator_fusion" in doc
    assert "UDFProperties(preserves_partitioning=True)" in doc
    assert "deterministic" not in doc
    assert "retry-safe" not in doc


@pytest.mark.parametrize(
    ("key", "error"),
    [
        (None, "one or more named"),
        ([], "one or more named"),
        (["User", ""], "non-empty string"),
        (["User", 1], "non-empty string"),
    ],
)
def test_map_group_partitions_rejects_invalid_keys(
    source_dataset, key, error
):
    grouped = GroupedData(source_dataset, key, num_partitions=None)
    with pytest.raises(ValueError, match=error):
        grouped.map_group_partitions(identity)


def test_map_group_partitions_rejects_non_callable(source_dataset):
    grouped = GroupedData(source_dataset, "User", num_partitions=None)
    with pytest.raises(TypeError, match="fn must be callable"):
        grouped.map_group_partitions(None)


def test_partition_context_is_frozen_and_reports_group_count():
    context = MapGroupPartitionsContext(
        group_keys=("User", "Card"),
        input_group_boundaries=(0, 3, 8),
    )

    assert ray.data.MapGroupPartitionsContext is MapGroupPartitionsContext
    assert context.num_groups == 2
    with pytest.raises(FrozenInstanceError):
        context.group_keys = ("Other",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_keys": (), "input_group_boundaries": (0,)},
        {"group_keys": ("key",), "input_group_boundaries": ()},
        {"group_keys": ("key",), "input_group_boundaries": (1, 2)},
        {"group_keys": ("key",), "input_group_boundaries": (0, 2, 2)},
        {"group_keys": ("key",), "input_group_boundaries": (0, True)},
    ],
)
def test_partition_context_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        MapGroupPartitionsContext(**kwargs)
