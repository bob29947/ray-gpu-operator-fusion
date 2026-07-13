"""Distribution-neutral actor-local payload envelopes for GPU stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class PartitionIdentity:
    """Open marker for an implementation-defined partition identity.

    Range partitioning is the only V1 implementation.  Future distribution
    strategies, such as hash partitioning, subclass this marker and flow
    through the existing envelopes without changing their structure.
    """


@dataclass(frozen=True)
class RangePartitionIdentity(PartitionIdentity):
    """Identity and ownership bounds for one logical range partition.

    This record deliberately contains no source work.  A source provider may
    use any private work descriptor to produce this identity; downstream unary
    and partition operators therefore do not depend on that source's format.
    """

    partition_id: int
    scheme_id: str
    lower: Any
    upper: Any

    def __post_init__(self) -> None:
        if self.partition_id < 0:
            raise ValueError("Range partition id must be nonnegative")
        if not self.scheme_id:
            raise ValueError("Range partition scheme id must be nonempty")


@dataclass(frozen=True, eq=False)
class RangeData:
    """Actor-local data carrying a distribution-neutral partition identity.

    ``value`` is either a frame iterable or a materialized frame, depending on
    the transform payload kind.  It may be a generator or mutable GPU object,
    so it is deliberately excluded from structural equality.
    """

    partition: PartitionIdentity
    value: Any

    def __post_init__(self) -> None:
        if not isinstance(self.partition, PartitionIdentity):
            raise TypeError("RangeData.partition must be a PartitionIdentity")


@dataclass(frozen=True, eq=False)
class GroupedRangeData:
    """One materialized partition plus its group boundaries."""

    partition: PartitionIdentity
    frame: Any
    boundaries: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.partition, PartitionIdentity):
            raise TypeError("GroupedRangeData.partition must be a PartitionIdentity")
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        if not self.boundaries:
            raise ValueError("GroupedRangeData boundaries must be nonempty")


__all__ = [
    "GroupedRangeData",
    "PartitionIdentity",
    "RangeData",
    "RangePartitionIdentity",
]
