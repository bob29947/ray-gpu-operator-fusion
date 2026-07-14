"""Actor-runtime transform for one planned Parquet range."""

from __future__ import annotations

import numbers
import urllib.parse
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Optional

from ray.data._internal.execution.gpu.parquet_types import (
    ParquetFileWork,
    ParquetRangeWork,
    ParquetSourceIdentity,
)
from ray.data._internal.execution.gpu.payloads import RangeData
from ray.data._internal.execution.gpu.transforms._runtime_support import (
    CudfTransformRuntime,
)

READ_PARQUET_RANGE_KIND = "read_parquet_range"


@dataclass(frozen=True)
class ReadParquetRangeConfig:
    source_kind: Literal["local", "s3"]
    region: Optional[str]
    projection: tuple[str, ...]
    range_key: str
    retried_io_errors: tuple[str, ...]
    row_groups_per_read: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "projection", tuple(self.projection))
        object.__setattr__(self, "retried_io_errors", tuple(self.retried_io_errors))
        if self.source_kind not in ("local", "s3"):
            raise ValueError(f"Unsupported source kind: {self.source_kind!r}")
        if self.source_kind == "s3" and not self.region:
            raise ValueError("An S3 direct read requires a region")
        if not self.projection or not self.range_key:
            raise ValueError("Projection and range key must be nonempty")
        if self.range_key not in self.projection:
            raise ValueError("The range key must be projected")
        if self.row_groups_per_read < 1:
            raise ValueError("row_groups_per_read must be positive")


def _split_s3_path(path: str) -> tuple[str, str]:
    bucket, separator, key = path.partition("/")
    if not separator or not bucket or not key:
        raise RuntimeError(f"Invalid planned S3 path: {path!r}")
    return bucket, key


def _s3_uri(path: str) -> str:
    return "s3://" + urllib.parse.quote(path, safe="/")


class ReadParquetRangeRuntime(CudfTransformRuntime):
    """Read exact planned row groups and filter them to one disjoint range."""

    config_type = ReadParquetRangeConfig
    kind = READ_PARQUET_RANGE_KIND

    def initialize(self) -> None:
        self._context.ensure_cudf(
            source_kind=self._config.source_kind,
            region=self._config.region,
        )

    def _identity(self, path: str) -> ParquetSourceIdentity:
        if self._config.source_kind == "local":
            import pyarrow.fs as pafs

            info = pafs.LocalFileSystem().get_file_info(path)
            if (
                info.type != pafs.FileType.File
                or info.size is None
                or info.size < 0
                or info.mtime_ns is None
            ):
                raise RuntimeError(f"Parquet source is unavailable: {path!r}")
            return ParquetSourceIdentity(
                int(info.size), f"mtime_ns:{int(info.mtime_ns)}"
            )

        bucket, key = _split_s3_path(path)
        response = self._context.s3_client.head_object(Bucket=bucket, Key=key)
        size = response.get("ContentLength")
        if not isinstance(size, numbers.Integral) or isinstance(size, bool) or size < 0:
            raise RuntimeError(f"Parquet source has no stable size: {path!r}")
        version = response.get("VersionId")
        if version and version != "null":
            marker = f"version:{version}"
        else:
            etag = response.get("ETag")
            modified = response.get("LastModified")
            if not etag or modified is None:
                raise RuntimeError(
                    f"Parquet source has no stable S3 identity: {path!r}"
                )
            timestamp = (
                modified.isoformat()
                if hasattr(modified, "isoformat")
                else str(modified)
            )
            marker = f"etag:{etag};last_modified:{timestamp}"
        return ParquetSourceIdentity(int(size), marker)

    def _verify_identity(self, file: ParquetFileWork) -> None:
        if self._identity(file.path) != file.identity:
            raise RuntimeError(f"Parquet source changed after planning: {file.path!r}")

    def _read_frames(self, work: ParquetRangeWork) -> Iterator[Any]:
        from ray.data._internal.util import iterate_with_retry

        cudf = self._context.cudf
        for file in work.files:
            path = _s3_uri(file.path) if self._config.source_kind == "s3" else file.path
            for start in range(
                0, len(file.row_groups), self._config.row_groups_per_read
            ):
                kwargs = {
                    "columns": list(self._config.projection),
                    "row_groups": list(
                        file.row_groups[
                            start : start + self._config.row_groups_per_read
                        ]
                    ),
                }
                if self._config.source_kind == "s3":
                    if not cudf.get_option("kvikio_remote_io"):
                        raise RuntimeError("cuDF disabled KvikIO remote I/O")
                    kwargs.update(
                        engine="cudf",
                        dataset_kwargs={"partitioning": None},
                        use_pandas_metadata=False,
                        categorical_partitions=False,
                    )

                def read_chunk() -> tuple[Any]:
                    if self._config.source_kind == "s3":
                        self._context.refresh_s3_for_task()
                    self._verify_identity(file)
                    frame = cudf.read_parquet(path, **kwargs)
                    self._verify_identity(file)
                    return (frame,)

                description = (
                    "read a Parquet row-group chunk with KvikIO"
                    if self._config.source_kind == "s3"
                    else "read a Parquet row-group chunk"
                )
                frame = next(
                    iterate_with_retry(
                        read_chunk,
                        description,
                        match=list(self._config.retried_io_errors),
                    )
                )
                frame = frame[
                    (frame[self._config.range_key] >= work.lower)
                    & (frame[self._config.range_key] <= work.upper)
                ].reset_index(drop=True)
                if len(frame):
                    yield frame

    def apply(self, value: Any, task_context: Any) -> RangeData:
        del task_context
        if not isinstance(value, ParquetRangeWork):
            raise TypeError("read_parquet_range expected native ParquetRangeWork")
        return RangeData(value.partition, self._read_frames(value))
