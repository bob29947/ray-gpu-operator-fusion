import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from ray.data._internal.execution.gpu.parquet_types import (
    ParquetFileWork,
    ParquetRangeWork,
    ParquetSourceIdentity,
)
from ray.data._internal.execution.gpu.payloads import GroupedRangeData, RangeData
from ray.data._internal.execution.gpu.transforms.grouped_partitions import (
    MapGroupPartitionsConfig,
    MapGroupPartitionsRuntime,
    SortPartitionConfig,
    SortPartitionRuntime,
)
from ray.data._internal.execution.gpu.transforms.map_batches import (
    MapBatchesConfig,
    MapBatchesRuntime,
    ValidateRangeOwnershipConfig,
    ValidateRangeOwnershipRuntime,
)
from ray.data.context import DataContext
from ray.data.exceptions import UserCodeException


class _UserError(RuntimeError):
    pass


class _Iloc:
    def __init__(self, frame):
        self._frame = frame

    def __getitem__(self, item):
        return self._frame


class _FakeFrame:
    def __init__(self, rows=2):
        self._rows = rows
        self.iloc = _Iloc(self)

    def __len__(self):
        return self._rows

    def reset_index(self, *, drop):
        assert drop
        return self

    def copy(self, *, deep):
        return _FakeFrame(self._rows)


class _FakeCudf:
    DataFrame = (_FakeFrame, pa.Table)

    @staticmethod
    def concat(frames, *, ignore_index):
        assert ignore_index
        return _FakeFrame(sum(len(frame) for frame in frames))


class _FakeCudfContext:
    cudf = _FakeCudf

    def ensure_cudf(self):
        return self


def _work():
    return ParquetRangeWork(
        range_id=0,
        scheme_id="scheme",
        lower=1,
        upper=5,
        files=(
            ParquetFileWork(
                "file",
                ParquetSourceIdentity(100, "identity"),
                (0,),
            ),
        ),
    )


def _map_runtime(udf_cls):
    return MapBatchesRuntime(
        MapBatchesConfig(
            udf_cls=udf_cls,
            batch_size=2,
            zero_copy_batch=True,
        ),
        _FakeCudfContext(),
    )


def _map_input():
    return RangeData(_work().partition, iter((_FakeFrame(),)))


def _group_input():
    return GroupedRangeData(
        partition=_work().partition,
        frame=pa.table({"key": [1], "value": [2]}),
        boundaries=(0, 1),
    )


class _ConstructorFailure:
    def __init__(self):
        raise _UserError("constructor failed")


class _CallFailure:
    def __call__(self, frame):
        raise _UserError("call failed")


class _LazyFailure:
    def __call__(self, frame):
        def outputs():
            yield frame
            raise _UserError("lazy output failed")

        return outputs()


class _WrongOutput:
    def __call__(self, frame):
        return object()


def test_gpu_map_batches_wraps_constructor_failure():
    with pytest.raises(UserCodeException) as exc_info:
        _map_runtime(_ConstructorFailure).initialize()

    assert isinstance(exc_info.value.__cause__, _UserError)


def test_gpu_map_batches_honors_original_exception_for_constructor():
    context = DataContext.get_current().copy()
    context.raise_original_map_exception = True

    with DataContext.current(context), pytest.raises(_UserError, match="constructor"):
        _map_runtime(_ConstructorFailure).initialize()


def test_gpu_map_batches_wraps_call_and_output_contract_failures():
    call_runtime = _map_runtime(_CallFailure)
    call_runtime.initialize()
    with pytest.raises(UserCodeException) as call_error:
        list(call_runtime.apply(_map_input(), None).value)
    assert isinstance(call_error.value.__cause__, _UserError)

    output_runtime = _map_runtime(_WrongOutput)
    output_runtime.initialize()
    with pytest.raises(UserCodeException) as output_error:
        list(output_runtime.apply(_map_input(), None).value)
    assert isinstance(output_error.value.__cause__, TypeError)


def test_gpu_map_batches_wraps_lazy_iterator_failure():
    runtime = _map_runtime(_LazyFailure)
    runtime.initialize()
    output = iter(runtime.apply(_map_input(), None).value)

    assert isinstance(next(output), _FakeFrame)
    with pytest.raises(UserCodeException) as exc_info:
        next(output)
    assert isinstance(exc_info.value.__cause__, _UserError)


def test_gpu_map_batches_honors_original_exception_for_lazy_failure():
    context = DataContext.get_current().copy()
    context.raise_original_map_exception = True
    runtime = _map_runtime(_LazyFailure)
    runtime.initialize()

    with DataContext.current(context):
        output = iter(runtime.apply(_map_input(), None).value)
        next(output)
        with pytest.raises(_UserError, match="lazy output"):
            next(output)


def test_gpu_map_group_partitions_wraps_call_failure():
    def fail(frame, context):
        raise _UserError("group call failed")

    runtime = MapGroupPartitionsRuntime(
        MapGroupPartitionsConfig(fail, ("key",), 1024),
        _FakeCudfContext(),
    )
    with pytest.raises(UserCodeException) as exc_info:
        list(runtime.apply(_group_input(), None).value)
    assert isinstance(exc_info.value.__cause__, _UserError)


def test_gpu_map_group_partitions_accepts_numpy_tensor_batch():
    expected = np.arange(16, dtype=np.uint16).reshape(2, 8)

    def return_tensor_batch(frame, context):
        return {"input_ids": expected}

    runtime = MapGroupPartitionsRuntime(
        MapGroupPartitionsConfig(return_tensor_batch, ("key",), 8),
        _FakeCudfContext(),
    )

    outputs = list(runtime.apply(_group_input(), None).value)

    assert len(outputs) == 2
    assert all(output["input_ids"].dtype == np.uint16 for output in outputs)
    np.testing.assert_array_equal(
        np.concatenate([output["input_ids"] for output in outputs]), expected
    )


def test_gpu_map_group_partitions_wraps_invalid_output_batch():
    def return_invalid_batch(frame, context):
        return object()

    runtime = MapGroupPartitionsRuntime(
        MapGroupPartitionsConfig(return_invalid_batch, ("key",), 1024),
        _FakeCudfContext(),
    )

    with pytest.raises(UserCodeException) as exc_info:
        list(runtime.apply(_group_input(), None).value)
    assert isinstance(exc_info.value.__cause__, TypeError)


def test_gpu_map_group_partitions_wraps_lazy_iterator_failure():
    def fail_lazily(frame, context):
        yield _FakeFrame(1)
        raise _UserError("group lazy output failed")

    runtime = MapGroupPartitionsRuntime(
        MapGroupPartitionsConfig(fail_lazily, ("key",), 1024),
        _FakeCudfContext(),
    )
    output = iter(runtime.apply(_group_input(), None).value)

    assert isinstance(next(output), _FakeFrame)
    with pytest.raises(UserCodeException) as exc_info:
        next(output)
    assert isinstance(exc_info.value.__cause__, _UserError)


def test_gpu_map_group_partitions_honors_original_exception():
    def fail_lazily(frame, context):
        yield _FakeFrame(1)
        raise _UserError("original group failure")

    context = DataContext.get_current().copy()
    context.raise_original_map_exception = True
    runtime = MapGroupPartitionsRuntime(
        MapGroupPartitionsConfig(fail_lazily, ("key",), 1024),
        _FakeCudfContext(),
    )

    with DataContext.current(context):
        output = iter(runtime.apply(_group_input(), None).value)
        next(output)
        with pytest.raises(_UserError, match="original group"):
            next(output)


def test_gpu_range_validation_wraps_removed_group_key():
    runtime = ValidateRangeOwnershipRuntime(
        ValidateRangeOwnershipConfig("key", ("key", "removed"))
    )
    output = runtime.apply(
        RangeData(_work().partition, iter((pd.DataFrame({"key": [2]}),))), None
    )

    with pytest.raises(UserCodeException) as exc_info:
        list(output.value)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "removed a group key" in str(exc_info.value.__cause__)


def test_gpu_range_validation_wraps_out_of_range_group_key():
    runtime = ValidateRangeOwnershipRuntime(
        ValidateRangeOwnershipConfig("key", ("key",))
    )
    output = runtime.apply(
        RangeData(_work().partition, iter((pd.DataFrame({"key": [10]}),))), None
    )

    with pytest.raises(UserCodeException) as exc_info:
        list(output.value)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "outside its assigned range" in str(exc_info.value.__cause__)


def test_gpu_range_validation_honors_original_exception():
    context = DataContext.get_current().copy()
    context.raise_original_map_exception = True
    runtime = ValidateRangeOwnershipRuntime(
        ValidateRangeOwnershipConfig("key", ("key",))
    )
    output = runtime.apply(
        RangeData(_work().partition, iter((pd.DataFrame({"key": [10]}),))), None
    )

    with (
        DataContext.current(context),
        pytest.raises(ValueError, match="outside its assigned range"),
    ):
        list(output.value)


def test_gpu_range_validation_wraps_incompatible_key_dtype():
    runtime = ValidateRangeOwnershipRuntime(
        ValidateRangeOwnershipConfig("key", ("key",))
    )
    output = runtime.apply(
        RangeData(
            _work().partition,
            iter((pd.DataFrame({"key": ["not-an-integer"]}),)),
        ),
        None,
    )

    with pytest.raises(UserCodeException) as exc_info:
        list(output.value)
    assert isinstance(exc_info.value.__cause__, TypeError)


def test_gpu_range_validation_dtype_error_honors_original_exception():
    context = DataContext.get_current().copy()
    context.raise_original_map_exception = True
    runtime = ValidateRangeOwnershipRuntime(
        ValidateRangeOwnershipConfig("key", ("key",))
    )
    output = runtime.apply(
        RangeData(
            _work().partition,
            iter((pd.DataFrame({"key": ["not-an-integer"]}),)),
        ),
        None,
    )

    with DataContext.current(context), pytest.raises(TypeError):
        list(output.value)


def test_gpu_sort_partition_wraps_removed_group_key():
    runtime = SortPartitionRuntime(
        SortPartitionConfig(("removed",)), _FakeCudfContext()
    )
    runtime.initialize()

    with pytest.raises(UserCodeException) as exc_info:
        runtime.apply(
            RangeData(_work().partition, iter((pd.DataFrame({"key": [2]}),))),
            None,
        )
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "removed a group key" in str(exc_info.value.__cause__)
