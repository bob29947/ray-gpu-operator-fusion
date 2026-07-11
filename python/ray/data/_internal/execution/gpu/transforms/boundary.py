"""Stock-compatible external boundary helpers for GPU range operators."""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from typing import Any

from ray.data._internal.execution.gpu.parquet_types import ParquetRangeWork


def serialize_range_work(work: ParquetRangeWork) -> bytes:
    return pickle.dumps(work, protocol=5)


def deserialize_range_work(value: Any) -> ParquetRangeWork:
    work = pickle.loads(bytes(value))
    if not isinstance(work, ParquetRangeWork):
        raise TypeError("Range descriptor did not contain ParquetRangeWork")
    return work


def decode_one_range_work(value: Any) -> ParquetRangeWork:
    """Decode exactly one descriptor from a block stream or direct test value."""

    if isinstance(value, ParquetRangeWork):
        return value
    from ray.data.block import BlockAccessor

    descriptors = []
    for block in value:
        column = BlockAccessor.for_block(block).to_numpy("work")
        descriptors.extend(column.tolist())
    if len(descriptors) != 1:
        raise ValueError(
            "A fused Parquet/cuDF actor task requires exactly one RangeWork "
            f"descriptor, but received {len(descriptors)}"
        )
    return deserialize_range_work(descriptors[0])


def descriptor_bundles(work: Iterable[ParquetRangeWork]) -> list[Any]:
    """Create one one-row owned RefBundle for every planned range."""

    import pyarrow as pa
    import ray
    from ray.data._internal.execution.interfaces import BlockEntry, RefBundle
    from ray.data.block import BlockAccessor

    bundles = []
    for descriptor in work:
        block = pa.table({"work": [serialize_range_work(descriptor)]})
        metadata = BlockAccessor.for_block(block).get_metadata()
        bundles.append(
            RefBundle(
                blocks=(BlockEntry(ray.put(block), metadata),),
                schema=block.schema,
                owns_blocks=True,
            )
        )
    return bundles
