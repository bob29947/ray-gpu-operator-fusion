"""Typed facts and requirements exchanged by declarative GPU operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


class PropertyFact:
    """Marker for a fact proven at one physical edge."""


class PropertyRequirement:
    """Marker for a fact required at one physical edge."""


def _normalize_keys(value: Sequence[str], owner: str) -> Tuple[str, ...]:
    keys = tuple(value)
    if not keys or not all(isinstance(key, str) and key for key in keys):
        raise ValueError(f"{owner} requires nonempty string keys")
    return keys


@dataclass(frozen=True)
class _RangeProperty:
    keys: Tuple[str, ...]
    scheme_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "keys", _normalize_keys(self.keys, type(self).__name__)
        )
        if not isinstance(self.scheme_id, str) or not self.scheme_id:
            raise ValueError(f"{type(self).__name__} requires a nonempty scheme_id")


class RangePartitioning(_RangeProperty, PropertyFact):
    pass


class RangePartitioningRequired(_RangeProperty, PropertyRequirement):
    pass


@dataclass(frozen=True)
class KeysColocated(PropertyRequirement):
    keys: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "keys", _normalize_keys(self.keys, type(self).__name__)
        )


class PropertyConflict(ValueError):
    pass


def property_satisfies(
    fact: PropertyFact,
    requirement: PropertyRequirement,
) -> bool:
    if not isinstance(fact, RangePartitioning):
        return False
    if isinstance(requirement, RangePartitioningRequired):
        return fact.keys == requirement.keys and fact.scheme_id == requirement.scheme_id
    if isinstance(requirement, KeysColocated):
        return (
            len(fact.keys) <= len(requirement.keys)
            and requirement.keys[: len(fact.keys)] == fact.keys
        )
    return False


def requirements_satisfied(
    facts: Sequence[PropertyFact],
    requirements: Sequence[PropertyRequirement],
) -> bool:
    return all(
        any(property_satisfies(fact, requirement) for fact in facts)
        for requirement in requirements
    )


def range_partitioning_fact(
    facts: Sequence[PropertyFact],
) -> RangePartitioning | None:
    ranges = tuple(fact for fact in facts if isinstance(fact, RangePartitioning))
    if len(ranges) > 1:
        raise PropertyConflict(
            f"edge advertises conflicting range partitioning facts: {ranges!r}"
        )
    return ranges[0] if ranges else None
