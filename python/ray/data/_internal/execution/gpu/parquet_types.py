"""Serializable records shared by Parquet planning and GPU execution.

Only immutable data required by an actor task belongs here.  Driver-only
objects such as PyArrow filesystems and schemas intentionally live in the read
recognizer instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParquetSourceIdentity:
    """Stable identity observed for a source file during planning."""

    size: int
    marker: str

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError("Parquet source size must be nonnegative")
        if not self.marker:
            raise ValueError("Parquet source identity marker must be nonempty")


@dataclass(frozen=True)
class ParquetRowGroup:
    """Footer-derived facts for one nonempty projected row group."""

    path: str
    identity: ParquetSourceIdentity
    row_group: int
    num_rows: int
    lower: int
    upper: int
    compressed_bytes: int
    uncompressed_bytes: int


@dataclass(frozen=True)
class ParquetFileWork:
    """The row groups to read from one identity-checked source file."""

    path: str
    identity: ParquetSourceIdentity
    row_groups: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("Parquet file work path must be nonempty")
        if not self.row_groups:
            raise ValueError("Parquet file work must contain at least one row group")
        if any(index < 0 for index in self.row_groups):
            raise ValueError("Parquet row-group indices must be nonnegative")
        if tuple(sorted(set(self.row_groups))) != self.row_groups:
            raise ValueError("Parquet row-group indices must be sorted and unique")


@dataclass(frozen=True)
class ParquetRangeWork:
    """One disjoint logical range and the physical row groups that overlap it."""

    range_id: int
    scheme_id: str
    lower: int
    upper: int
    files: tuple[ParquetFileWork, ...]

    def __post_init__(self) -> None:
        if self.range_id < 0:
            raise ValueError("Parquet range id must be nonnegative")
        if not self.scheme_id:
            raise ValueError("Parquet range scheme id must be nonempty")
        if self.lower > self.upper:
            raise ValueError("Parquet range lower bound must not exceed upper bound")
        if not self.files:
            raise ValueError("Parquet range work must contain at least one file")


@dataclass(frozen=True)
class ParquetFooterSummary:
    """Driver-only footer scan result without a live filesystem handle."""

    row_groups: tuple[ParquetRowGroup, ...]
    footer_bytes: int
    elapsed_s: float


@dataclass(frozen=True)
class ParquetRangePlan:
    """Complete immutable output of driver-side range planning."""

    range_key: str
    scheme_id: str
    work: tuple[ParquetRangeWork, ...]
    amplification: tuple[float, float, float]
    skew: float
    footer_bytes: int
    footer_elapsed_s: float


@dataclass(frozen=True, eq=False)
class RangeData:
    """Actor-local value that preserves range identity across transforms.

    ``value`` may be a generator or a mutable cuDF object and is therefore
    deliberately excluded from structural equality.
    """

    work: ParquetRangeWork
    value: Any
