"""Small property vocabulary for required GPU-region composition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple


class Evidence(str, Enum):
    DERIVED = "derived"
    USER_ASSERTED = "user_asserted"
    OPERATOR_GUARANTEE = "operator_guarantee"


class PropertyValue:
    """Marker base for a fact advertised on a physical edge."""


class PartitioningProperty(PropertyValue):
    """Marker base for semantic partitioning facts."""


class PropertyRequirement:
    """Marker base for a fact required at an operator input."""


@dataclass(frozen=True)
class UnknownPartitioning(PartitioningProperty):
    pass


@dataclass(frozen=True)
class RangePartitioning(PartitioningProperty):
    keys: Tuple[str, ...]
    scheme_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))
        if not self.keys:
            raise ValueError("RangePartitioning requires at least one key")
        if not self.scheme_id:
            raise ValueError("RangePartitioning requires a nonempty scheme_id")


@dataclass(frozen=True)
class CompletePartitionBundles(PropertyValue, PropertyRequirement):
    """Each logical partition is delivered as one complete planner unit."""


@dataclass(frozen=True)
class PropertyFact:
    value: PropertyValue
    evidence: Evidence

    def __post_init__(self) -> None:
        if not isinstance(self.value, PropertyValue):
            raise TypeError("PropertyFact.value must be a PropertyValue")


@dataclass(frozen=True)
class KeysColocated(PropertyRequirement):
    keys: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))
        if not self.keys:
            raise ValueError("KeysColocated requires at least one key")


@dataclass(frozen=True)
class RangePartitioningRequired(PropertyRequirement):
    """Require the exact range scheme used to lower a preserving operator."""

    keys: Tuple[str, ...]
    scheme_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))
        if not self.keys:
            raise ValueError("RangePartitioningRequired requires at least one key")
        if not self.scheme_id:
            raise ValueError("RangePartitioningRequired requires a scheme_id")


class ValidationObligation:
    """Marker for a runtime proof that must be discharged inside one stage."""


@dataclass(frozen=True)
class RowsOwnedByRange(ValidationObligation):
    leading_key: str
    scheme_id: str


@dataclass(frozen=True)
class PartitioningValidation:
    kind: str
    config: Any
    obligations: Tuple[ValidationObligation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligations", tuple(self.obligations))
        if not self.kind:
            raise ValueError("partitioning validation kind must be nonempty")
        if not self.obligations:
            raise ValueError("partitioning validation must discharge an obligation")
        if not all(
            isinstance(obligation, ValidationObligation)
            for obligation in self.obligations
        ):
            raise TypeError("invalid partitioning validation obligation")


@dataclass(frozen=True)
class PropertyReadvertisement:
    output_properties: Tuple[PropertyFact, ...]
    partitioning_validations: Tuple[PartitioningValidation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_properties", tuple(self.output_properties))
        object.__setattr__(
            self,
            "partitioning_validations",
            tuple(self.partitioning_validations),
        )

    @property
    def created_obligations(self) -> Tuple[ValidationObligation, ...]:
        return tuple(
            obligation
            for validation in self.partitioning_validations
            for obligation in validation.obligations
        )


class PartitioningPreservationError(ValueError):
    pass


def property_satisfies(
    fact: PropertyFact,
    requirement: PropertyRequirement,
) -> bool:
    value = fact.value
    if isinstance(value, RangePartitioning) and isinstance(
        requirement, KeysColocated
    ):
        return (
            len(value.keys) <= len(requirement.keys)
            and requirement.keys[: len(value.keys)] == value.keys
        )
    if isinstance(value, RangePartitioning) and isinstance(
        requirement, RangePartitioningRequired
    ):
        return (
            value.keys == requirement.keys
            and value.scheme_id == requirement.scheme_id
        )
    return isinstance(value, CompletePartitionBundles) and isinstance(
        requirement, CompletePartitionBundles
    )


def requirements_satisfied(
    facts: Tuple[PropertyFact, ...],
    requirements: Tuple[PropertyRequirement, ...],
) -> bool:
    return all(
        any(property_satisfies(fact, requirement) for fact in facts)
        for requirement in requirements
    )


def usable_partitioning_fact(
    facts: Tuple[PropertyFact, ...],
) -> Optional[PropertyFact]:
    partitioning = tuple(
        fact for fact in facts if isinstance(fact.value, PartitioningProperty)
    )
    if len(partitioning) > 1:
        raise PartitioningPreservationError(
            f"edge advertises conflicting partitioning facts: {partitioning!r}"
        )
    if not partitioning or isinstance(partitioning[0].value, UnknownPartitioning):
        return None
    return partitioning[0]


def readvertise_preserved_properties(
    facts: Tuple[PropertyFact, ...],
    *,
    preserves_partitioning: bool,
    preserves_complete_partition_bundles: bool,
    plan_validations: bool = True,
) -> PropertyReadvertisement:
    """Return only facts explicitly preserved by the current operator."""

    facts = tuple(facts)
    partitioning = usable_partitioning_fact(facts)
    validations: Tuple[PartitioningValidation, ...] = ()
    if preserves_partitioning and partitioning is not None:
        value = partitioning.value
        if not isinstance(value, RangePartitioning):
            raise PartitioningPreservationError(
                f"V1 cannot preserve {type(value).__name__}"
            )
        if plan_validations:
            validations = _range_partitioning_validations(value)
        output = [PropertyFact(value, Evidence.OPERATOR_GUARANTEE)]
    else:
        output = [PropertyFact(UnknownPartitioning(), Evidence.DERIVED)]

    if preserves_complete_partition_bundles and any(
        isinstance(fact.value, CompletePartitionBundles) for fact in facts
    ):
        output.append(
            PropertyFact(
                CompletePartitionBundles(),
                Evidence.OPERATOR_GUARANTEE,
            )
        )
    return PropertyReadvertisement(tuple(output), validations)


def _range_partitioning_validations(
    value: RangePartitioning,
) -> Tuple[PartitioningValidation, ...]:
    from ray.data._internal.execution.gpu.transforms.map_batches import (
        VALIDATE_RANGE_OWNERSHIP_KIND,
        ValidateRangeOwnershipConfig,
    )

    obligation = RowsOwnedByRange(value.keys[0], value.scheme_id)
    return (
        PartitioningValidation(
            kind=VALIDATE_RANGE_OWNERSHIP_KIND,
            config=ValidateRangeOwnershipConfig(value.keys[0], value.keys),
            obligations=(obligation,),
        ),
    )
