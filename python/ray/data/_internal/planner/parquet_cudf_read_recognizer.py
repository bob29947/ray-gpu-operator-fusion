"""Single V1 recognition seam for direct Parquet/cuDF range reads."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class ParquetCudfReadDescriptor:
    """Driver-side input to range planning.

    The filesystem and schema are intentionally confined to this driver record;
    neither may be embedded in a GPU operator specification or actor payload.
    """

    filesystem: Any
    source_kind: Literal["local", "s3"]
    paths: tuple[str, ...]
    listed_file_sizes: tuple[int, ...]
    projection: tuple[str, ...]
    file_schema: Any
    region: Optional[str]
    read_ray_remote_args: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class ParquetCudfReadRecognition:
    """A recognized read descriptor or a stable incompatibility reason."""

    descriptor: Optional[ParquetCudfReadDescriptor] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.descriptor is None) == (self.reason is None):
            raise ValueError("Exactly one of descriptor or reason must be set")

    @property
    def accepted(self) -> bool:
        return self.descriptor is not None


def _reject(reason: str) -> ParquetCudfReadRecognition:
    return ParquetCudfReadRecognition(reason=reason)


def _ambient_s3_credentials_are_available() -> bool:
    """Check capability without exporting credentials into a plan object."""

    try:
        import botocore.session

        credentials = botocore.session.get_session().get_credentials()
        if credentials is None:
            return False
        frozen = credentials.get_frozen_credentials()
        return bool(frozen.access_key and frozen.secret_key)
    except Exception:
        return False


def recognize_parquet_cudf_read(
    read_op: Any,
    group_keys: tuple[str, ...],
) -> ParquetCudfReadRecognition:
    """Recognize the V1 legacy-Read shape without performing footer I/O.

    This is the only V1 function that jointly understands logical ``Read``,
    ``ParquetDatasource``, source kind/region, ambient credential policy, and
    endpoint overrides.  A future datasource-V2 recognizer can return the same
    descriptor without changing range planning.
    """

    import pyarrow as pa

    from ray.data._internal.datasource.parquet_datasource import ParquetDatasource
    from ray.data._internal.logical.operators import Read

    if type(read_op) is not Read:
        return _reject("logical_read")
    datasource = read_op.datasource
    if (
        type(datasource) is not ParquetDatasource
        or read_op.datasource_or_legacy_reader is not datasource
    ):
        return _reject("parquet_datasource")
    if not group_keys or any(not isinstance(key, str) or not key for key in group_keys):
        return _reject("group_keys")
    if read_op.per_block_limit is not None:
        return _reject("per_block_limit")
    allowed_read_args = {"scheduling_strategy", "label_selector"}
    if set(read_op.ray_remote_args) - allowed_read_args:
        return _reject("read_remote_args")

    result = datasource._get_parquet_cudf_direct_read_spec()
    if result.spec is None:
        return _reject(f"direct_read:{result.reason}")
    spec = result.spec
    if any(key not in spec.projection for key in group_keys):
        return _reject("group_key_projection")
    if group_keys[0] not in spec.file_schema.names:
        return _reject("range_key_source_column")
    if not pa.types.is_integer(spec.file_schema.field(group_keys[0]).type):
        return _reject("range_key_integer")

    if spec.source_kind == "s3":
        if os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3"):
            return _reject("s3_endpoint_override")
        if not isinstance(spec.region, str) or not spec.region:
            return _reject("s3_region")
        if not _ambient_s3_credentials_are_available():
            return _reject("s3_ambient_credentials")

    return ParquetCudfReadRecognition(
        descriptor=ParquetCudfReadDescriptor(
            filesystem=spec.filesystem,
            source_kind=spec.source_kind,
            paths=spec.paths,
            listed_file_sizes=spec.listed_file_sizes,
            projection=spec.projection,
            file_schema=spec.file_schema,
            region=spec.region,
            read_ray_remote_args=tuple(
                sorted(read_op.ray_remote_args.items(), key=lambda item: item[0])
            ),
        )
    )
