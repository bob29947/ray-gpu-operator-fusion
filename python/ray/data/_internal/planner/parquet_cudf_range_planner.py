"""Read-shape-independent footer and range planning for direct cuDF reads."""

from __future__ import annotations

import hashlib
import json
import logging
import numbers
import time
from typing import Any

from ray.data._internal.execution.gpu.parquet_types import (
    ParquetFileWork,
    ParquetFooterSummary,
    ParquetRangePlan,
    ParquetRangeWork,
    ParquetRowGroup,
    ParquetSourceIdentity,
)
from ray.data._internal.planner.parquet_cudf_read_recognizer import (
    ParquetCudfReadDescriptor,
)

logger = logging.getLogger(__name__)

MAX_RECOMMENDED_AMPLIFICATION = 1.25
MAX_RECOMMENDED_SKEW = 1.5


class ParquetCudfPlanningError(ValueError):
    """A stable source or range invariant required by V1 was unavailable."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


_REASON_MESSAGES = {
    "source_unavailable": "a Parquet source file is unavailable",
    "source_identity": "a stable Parquet source identity is unavailable",
    "source_changed": "a Parquet source changed while its footer was read",
    "listed_size_changed": "a Parquet source changed after dataset construction",
    "missing_column": "a projected column is missing from a Parquet file",
    "nested_range_key": "the leading group key must be a top-level Parquet column",
    "key_statistics": (
        "the leading group key needs non-null integer min/max row-group statistics"
    ),
    "size_statistics": "projected Parquet columns need row-group size statistics",
    "empty_source": "the Parquet input has no nonempty row groups",
    "s3_path": "an S3 source path is invalid",
    "range_target": "the range target and worker count must be positive",
    "empty_ranges": "footer statistics produced no executable ranges",
}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ParquetCudfPlanningError(reason, _REASON_MESSAGES.get(reason, reason))


def split_s3_path(path: str) -> tuple[str, str]:
    bucket, separator, key = path.partition("/")
    _require(bool(separator and bucket and key), "s3_path")
    return bucket, key


def _new_s3_client(region: str) -> Any:
    """Create a driver client after the recognizer validated endpoint policy."""

    import botocore.session

    session = botocore.session.get_session()
    _require(session.get_credentials() is not None, "source_identity")
    return session.create_client("s3", region_name=region)


def source_identity(
    read: ParquetCudfReadDescriptor,
    path: str,
    s3_client: Any = None,
) -> ParquetSourceIdentity:
    """Read a stable local or S3 identity without opening Parquet data."""

    if read.source_kind == "local":
        import pyarrow.fs as pafs

        info = read.filesystem.get_file_info(path)
        _require(info.type == pafs.FileType.File, "source_unavailable")
        _require(info.size is not None and info.size >= 0, "source_identity")
        _require(info.mtime_ns is not None, "source_identity")
        return ParquetSourceIdentity(int(info.size), f"mtime_ns:{int(info.mtime_ns)}")

    if s3_client is None:
        s3_client = _new_s3_client(read.region)
    bucket, key = split_s3_path(path)
    response = s3_client.head_object(Bucket=bucket, Key=key)
    size = response.get("ContentLength")
    _require(
        isinstance(size, numbers.Integral) and not isinstance(size, bool) and size >= 0,
        "source_identity",
    )
    version = response.get("VersionId")
    etag = response.get("ETag")
    modified = response.get("LastModified")
    if version and version != "null":
        marker = f"version:{version}"
    else:
        _require(bool(etag) and modified is not None, "source_identity")
        timestamp = (
            modified.isoformat() if hasattr(modified, "isoformat") else str(modified)
        )
        marker = f"etag:{etag};last_modified:{timestamp}"
    return ParquetSourceIdentity(int(size), marker)


def read_footer_summary(
    read: ParquetCudfReadDescriptor,
    range_key: str,
) -> ParquetFooterSummary:
    """Read and validate all source footers once on the driver."""

    import pyarrow.parquet as pq

    started = time.perf_counter()
    rows: list[ParquetRowGroup] = []
    footer_bytes = 0
    s3_client = _new_s3_client(read.region) if read.source_kind == "s3" else None

    for path, listed_size in zip(read.paths, read.listed_file_sizes):
        identity = source_identity(read, path, s3_client)
        _require(identity.size == listed_size, "listed_size_changed")
        with read.filesystem.open_input_file(path) as source:
            metadata = pq.ParquetFile(source).metadata
        _require(source_identity(read, path, s3_client) == identity, "source_changed")
        footer_bytes += int(getattr(metadata, "serialized_size", 0) or 0)

        column_indices: dict[str, list[int]] = {}
        for index in range(metadata.num_columns):
            name = metadata.schema.column(index).path.split(".", 1)[0]
            column_indices.setdefault(name, []).append(index)
        _require(
            all(column in column_indices for column in read.projection),
            "missing_column",
        )
        _require(len(column_indices.get(range_key, ())) == 1, "nested_range_key")
        range_key_index = column_indices[range_key][0]

        for row_group_id in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_id)
            if row_group.num_rows == 0:
                continue
            statistics = row_group.column(range_key_index).statistics
            statistics_ok = (
                statistics is not None
                and statistics.has_min_max
                and statistics.min is not None
                and statistics.max is not None
                and statistics.null_count == 0
            )
            _require(statistics_ok, "key_statistics")
            lower, upper = statistics.min, statistics.max
            _require(
                isinstance(lower, numbers.Integral)
                and not isinstance(lower, bool)
                and isinstance(upper, numbers.Integral)
                and not isinstance(upper, bool)
                and lower <= upper,
                "key_statistics",
            )

            compressed = 0
            uncompressed = 0
            for column in read.projection:
                for index in column_indices[column]:
                    chunk = row_group.column(index)
                    compressed_size = chunk.total_compressed_size
                    uncompressed_size = chunk.total_uncompressed_size
                    _require(
                        compressed_size is not None
                        and compressed_size >= 0
                        and uncompressed_size is not None
                        and uncompressed_size >= 0,
                        "size_statistics",
                    )
                    compressed += int(compressed_size)
                    uncompressed += int(uncompressed_size)
            _require(compressed > 0 and uncompressed > 0, "size_statistics")
            rows.append(
                ParquetRowGroup(
                    path=path,
                    identity=identity,
                    row_group=row_group_id,
                    num_rows=int(row_group.num_rows),
                    lower=int(lower),
                    upper=int(upper),
                    compressed_bytes=compressed,
                    uncompressed_bytes=uncompressed,
                )
            )

    _require(bool(rows), "empty_source")
    rows.sort(key=lambda row: (row.lower, row.upper, row.path, row.row_group))
    return ParquetFooterSummary(
        tuple(rows), footer_bytes, time.perf_counter() - started
    )


def _scheme_id(
    range_key: str,
    ranges: list[tuple[int, int, tuple[ParquetRowGroup, ...]]],
) -> str:
    payload = {
        "version": 1,
        "range_key": range_key,
        "ranges": [
            {
                "lower": lower,
                "upper": upper,
                "row_groups": [
                    [
                        row.path,
                        row.identity.size,
                        row.identity.marker,
                        row.row_group,
                    ]
                    for row in selected
                ],
            }
            for lower, upper, selected in ranges
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "parquet-range-v1:" + hashlib.sha256(encoded).hexdigest()


def build_range_plan(
    footer: ParquetFooterSummary,
    *,
    range_key: str,
    target_ranges: int,
    num_workers: int,
) -> ParquetRangePlan:
    """Build disjoint ranges and the overlapping physical work for each."""

    _require(target_ranges > 0 and num_workers > 0, "range_target")
    rows = footer.row_groups
    total_rows = sum(row.num_rows for row in rows)
    total_compressed = sum(row.compressed_bytes for row in rows)
    total_uncompressed = sum(row.uncompressed_bytes for row in rows)

    domain_lower = min(row.lower for row in rows)
    domain_upper = max(row.upper for row in rows)
    domain_size = domain_upper - domain_lower + 1
    count = min(target_ranges, domain_size)
    if count < target_ranges:
        logger.warning(
            "Parquet/cuDF capped ranges from %d to key-domain size %d; "
            "execution may underutilize GPUs",
            target_ranges,
            count,
        )

    ranges: list[tuple[int, int, tuple[ParquetRowGroup, ...]]] = []
    loads: list[tuple[int, int, int]] = []
    for index in range(count):
        lower = domain_lower + domain_size * index // count
        upper = domain_lower + domain_size * (index + 1) // count - 1
        selected = tuple(
            row for row in rows if row.upper >= lower and row.lower <= upper
        )
        if not selected:
            continue
        ranges.append((lower, upper, selected))
        loads.append(
            (
                sum(row.num_rows for row in selected),
                sum(row.compressed_bytes for row in selected),
                sum(row.uncompressed_bytes for row in selected),
            )
        )

    _require(bool(ranges), "empty_ranges")
    duplicated_totals = tuple(sum(load[i] for load in loads) for i in range(3))
    source_totals = (total_rows, total_compressed, total_uncompressed)
    amplification = tuple(duplicated_totals[i] / source_totals[i] for i in range(3))
    skew = max(
        max(load[i] for load in loads) / (duplicated_totals[i] / len(loads))
        for i in range(3)
    )

    scheme_id = _scheme_id(range_key, ranges)
    work: list[ParquetRangeWork] = []
    for range_id, (lower, upper, selected) in enumerate(ranges):
        files: dict[tuple[str, ParquetSourceIdentity], list[int]] = {}
        for row in selected:
            files.setdefault((row.path, row.identity), []).append(row.row_group)
        file_work = tuple(
            ParquetFileWork(path, identity, tuple(sorted(row_groups)))
            for (path, identity), row_groups in sorted(
                files.items(), key=lambda item: item[0][0]
            )
        )
        work.append(
            ParquetRangeWork(
                range_id=range_id,
                scheme_id=scheme_id,
                lower=lower,
                upper=upper,
                files=file_work,
            )
        )

    if max(amplification) > MAX_RECOMMENDED_AMPLIFICATION:
        logger.warning(
            "Parquet/cuDF estimated read amplification is %.3fx "
            "(recommended <= %.2fx); execution may be slow",
            max(amplification),
            MAX_RECOMMENDED_AMPLIFICATION,
        )
    if skew > MAX_RECOMMENDED_SKEW:
        logger.warning(
            "Parquet/cuDF estimated range skew is %.3fx "
            "(recommended <= %.2fx); execution may be slow",
            skew,
            MAX_RECOMMENDED_SKEW,
        )
    if len(work) < num_workers:
        logger.warning(
            "Parquet/cuDF planned %d nonempty ranges for %d workers; "
            "execution may underutilize GPUs",
            len(work),
            num_workers,
        )
    return ParquetRangePlan(
        range_key=range_key,
        scheme_id=scheme_id,
        work=tuple(work),
        amplification=amplification,
        skew=skew,
        footer_bytes=footer.footer_bytes,
        footer_elapsed_s=footer.elapsed_s,
    )


def plan_parquet_ranges(
    read: ParquetCudfReadDescriptor,
    *,
    range_key: str,
    target_ranges: int,
    num_workers: int,
    retried_io_errors: tuple[str, ...] = (),
) -> ParquetRangePlan:
    """Perform footer analysis and return serializable actor work."""

    from ray.data._internal.util import iterate_with_retry

    footer = next(
        iterate_with_retry(
            lambda: (read_footer_summary(read, range_key),),
            "read Parquet footer metadata",
            match=list(retried_io_errors),
        )
    )
    return build_range_plan(
        footer,
        range_key=range_key,
        target_ranges=target_ranges,
        num_workers=num_workers,
    )
