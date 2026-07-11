"""Small, conservative property vocabulary used by GPU fusion V1."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Union


class Evidence(str, Enum):
    DERIVED = "derived"
    USER_ASSERTED = "user_asserted"
    OPERATOR_GUARANTEE = "operator_guarantee"


@dataclass(frozen=True)
class UnknownPartitioning:
    pass


@dataclass(frozen=True)
class RangePartitioning:
    keys: Tuple[str, ...]
    scheme_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))
        if not self.keys:
            raise ValueError("RangePartitioning requires at least one key")


@dataclass(frozen=True)
class UnknownOrdering:
    pass


@dataclass(frozen=True)
class SortKey:
    key: str
    descending: bool = False
    nulls_first: bool = False


@dataclass(frozen=True)
class LocalOrdering:
    sort_keys: Tuple[SortKey, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort_keys", tuple(self.sort_keys))


PartitioningProperty = Union[UnknownPartitioning, RangePartitioning]
OrderingProperty = Union[UnknownOrdering, LocalOrdering]
PropertyValue = Union[PartitioningProperty, OrderingProperty]


@dataclass(frozen=True)
class PropertyFact:
    value: PropertyValue
    evidence: Evidence


@dataclass(frozen=True)
class KeysColocated:
    keys: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))
        if not self.keys:
            raise ValueError("KeysColocated requires at least one key")


@dataclass(frozen=True)
class OrderedBy:
    sort_keys: Tuple[SortKey, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort_keys", tuple(self.sort_keys))
        if not self.sort_keys:
            raise ValueError("OrderedBy requires at least one sort key")


PropertyRequirement = Union[KeysColocated, OrderedBy]


@dataclass(frozen=True)
class RowsOwnedByRange:
    leading_key: str
    scheme_id: str


ValidationObligation = RowsOwnedByRange


def property_satisfies(
    fact: PropertyFact, requirement: PropertyRequirement
) -> bool:
    """Return whether one conservative fact semantically satisfies a requirement."""

    value = fact.value
    if isinstance(requirement, KeysColocated):
        if not isinstance(value, RangePartitioning):
            return False
        # Disjoint ranges on a nonempty prefix guarantee that equal values for
        # all requested keys are assigned to the same range.
        return (
            len(value.keys) <= len(requirement.keys)
            and requirement.keys[: len(value.keys)] == value.keys
        )
    if isinstance(requirement, OrderedBy):
        if not isinstance(value, LocalOrdering):
            return False
        return value.sort_keys[: len(requirement.sort_keys)] == requirement.sort_keys
    return False


def requirements_satisfied(
    facts: Tuple[PropertyFact, ...],
    requirements: Tuple[PropertyRequirement, ...],
) -> bool:
    return all(
        any(property_satisfies(fact, requirement) for fact in facts)
        for requirement in requirements
    )
