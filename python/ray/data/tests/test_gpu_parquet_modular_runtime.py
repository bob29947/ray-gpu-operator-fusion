import pickle

from ray.data._internal.execution.gpu.parquet_types import (
    ParquetFileWork,
    ParquetFooterSummary,
    ParquetRangeWork,
    ParquetRowGroup,
    ParquetSourceIdentity,
)
from ray.data._internal.execution.gpu.payloads import RangeData, RangePartitionIdentity
from ray.data._internal.execution.gpu.transforms.boundary import (
    decode_one_range_work,
    deserialize_range_work,
    serialize_range_work,
)
from ray.data._internal.execution.gpu.transforms.map_batches import (
    MapBatchesConfig,
    MapBatchesRuntime,
)
from ray.data._internal.execution.gpu.transforms.parquet_read import (
    ReadParquetRangeConfig,
    ReadParquetRangeRuntime,
)
from ray.data._internal.planner.parquet_cudf_range_planner import (
    build_range_plan,
    plan_parquet_ranges,
)
from ray.data._internal.planner.parquet_cudf_read_recognizer import (
    ParquetCudfReadDescriptor,
)


def _identity(marker="a"):
    return ParquetSourceIdentity(100, marker)


def test_range_work_is_serializable_and_boundary_round_trips():
    work = ParquetRangeWork(
        range_id=0,
        scheme_id="scheme",
        lower=1,
        upper=5,
        files=(ParquetFileWork("file", _identity(), (0, 2)),),
    )
    assert pickle.loads(pickle.dumps(work)) == work
    assert deserialize_range_work(serialize_range_work(work)) == work
    assert decode_one_range_work(work) is work


def test_range_plan_is_disjoint_stable_and_tracks_overlapping_row_groups():
    rows = (
        ParquetRowGroup("a", _identity("a"), 0, 10, 0, 4, 20, 40),
        ParquetRowGroup("a", _identity("a"), 1, 10, 5, 9, 20, 40),
        ParquetRowGroup("b", _identity("b"), 0, 10, 3, 7, 20, 40),
    )
    footer = ParquetFooterSummary(rows, footer_bytes=50, elapsed_s=0.1)
    first = build_range_plan(footer, range_key="user", target_ranges=3, num_workers=2)
    second = build_range_plan(footer, range_key="user", target_ranges=3, num_workers=2)

    assert first.scheme_id == second.scheme_id
    assert first.work == second.work
    assert all(
        left.upper < right.lower for left, right in zip(first.work, first.work[1:])
    )
    assert first.work[1].files == (
        ParquetFileWork("a", _identity("a"), (0, 1)),
        ParquetFileWork("b", _identity("b"), (0,)),
    )


def test_range_planner_reads_real_footer_and_emits_serializable_work(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "input.parquet"
    pq.write_table(
        pa.table(
            {
                "user": pa.array([0, 1, 5, 6], type=pa.int64()),
                "value": pa.array([10, 11, 12, 13], type=pa.int64()),
            }
        ),
        path,
        row_group_size=2,
    )
    filesystem = pa.fs.LocalFileSystem()
    info = filesystem.get_file_info(str(path))
    descriptor = ParquetCudfReadDescriptor(
        filesystem=filesystem,
        source_kind="local",
        paths=(str(path),),
        listed_file_sizes=(info.size,),
        projection=("user", "value"),
        file_schema=pq.read_schema(path),
        region=None,
        read_ray_remote_args=(),
    )
    plan = plan_parquet_ranges(
        descriptor, range_key="user", target_ranges=2, num_workers=1
    )

    assert tuple((work.lower, work.upper) for work in plan.work) == ((0, 2), (3, 6))
    assert pickle.loads(pickle.dumps(plan.work)) == plan.work
    assert plan.footer_bytes > 0


class _Iloc:
    def __init__(self, frame):
        self._frame = frame

    def __getitem__(self, item):
        return self._frame


class _FakeFrame:
    copy_calls = []

    def __init__(self, rows=2):
        self._rows = rows
        self.iloc = _Iloc(self)

    def __len__(self):
        return self._rows

    def reset_index(self, *, drop):
        assert drop
        return self

    def copy(self, *, deep):
        self.copy_calls.append(deep)
        return _FakeFrame(self._rows)


class _FakeCudf:
    DataFrame = _FakeFrame

    @staticmethod
    def concat(frames, *, ignore_index):
        assert ignore_index
        return _FakeFrame(sum(len(frame) for frame in frames))


class _FakeCudfContext:
    cudf = _FakeCudf

    def ensure_cudf(self):
        return self


class _IdentityUDF:
    def __call__(self, frame):
        return frame


def test_zero_copy_false_performs_deep_copy_before_udf():
    _FakeFrame.copy_calls.clear()
    runtime = MapBatchesRuntime(
        MapBatchesConfig(
            udf_cls=_IdentityUDF,
            batch_size=2,
            zero_copy_batch=False,
        ),
        _FakeCudfContext(),
    )
    runtime.initialize()
    work = ParquetRangeWork(
        range_id=0,
        scheme_id="scheme",
        lower=1,
        upper=5,
        files=(ParquetFileWork("file", _identity(), (0,)),),
    )
    result = runtime.apply(RangeData(work.partition, iter((_FakeFrame(),))), None)
    assert list(result.value)
    assert _FakeFrame.copy_calls == [True]


class _RetryingCudf:
    def __init__(self, failures):
        self._failures = iter(failures)
        self.calls = 0

    def read_parquet(self, path, **kwargs):
        import pandas as pd

        del path, kwargs
        self.calls += 1
        error = next(self._failures, None)
        if error is not None:
            raise error
        return pd.DataFrame({"user": [2], "value": [10]})


class _ReadContext:
    def __init__(self, cudf):
        self.cudf = cudf

    def ensure_cudf(self, **kwargs):
        del kwargs
        return self


def _read_runtime(cudf, retried_io_errors):
    runtime = ReadParquetRangeRuntime(
        ReadParquetRangeConfig(
            source_kind="local",
            region=None,
            projection=("user", "value"),
            range_key="user",
            retried_io_errors=tuple(retried_io_errors),
        ),
        _ReadContext(cudf),
    )
    runtime._verify_identity = lambda _file: None
    return runtime


def _read_work():
    return ParquetRangeWork(
        range_id=0,
        scheme_id="scheme",
        lower=1,
        upper=5,
        files=(ParquetFileWork("file", _identity(), (0,)),),
    )


def test_parquet_row_group_read_honors_retried_io_errors(monkeypatch):
    from ray.data._internal import util

    monkeypatch.setattr(util.time, "sleep", lambda _seconds: None)
    cudf = _RetryingCudf([OSError("temporary network failure")])

    output = _read_runtime(cudf, ("temporary network",)).apply(_read_work(), None)

    assert output.partition == RangePartitionIdentity(0, "scheme", 1, 5)
    assert not hasattr(output, "work")
    assert [frame.to_dict("records") for frame in output.value] == [
        [{"user": 2, "value": 10}]
    ]
    assert cudf.calls == 2


def test_parquet_row_group_read_does_not_retry_unmatched_error(monkeypatch):
    from ray.data._internal import util

    monkeypatch.setattr(util.time, "sleep", lambda _seconds: None)
    cudf = _RetryingCudf([OSError("permanent failure")])
    output = _read_runtime(cudf, ("temporary network",)).apply(_read_work(), None)

    import pytest

    with pytest.raises(OSError, match="permanent failure"):
        list(output.value)
    assert cudf.calls == 1
