"""Extensible planner properties used by declarative GPU operators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Type


class Evidence(str, Enum):
    DERIVED = "derived"
    USER_ASSERTED = "user_asserted"
    OPERATOR_GUARANTEE = "operator_guarantee"


class PropertyValue:
    """Marker base for a fact that an operator may advertise."""


class PartitioningProperty(PropertyValue):
    """Open property family for one edge's semantic partitioning.

    A new partitioning implementation subclasses this marker and registers both
    its requirement-satisfaction rule and its preservation validator.  The
    built-in lowerers then preserve it without knowing its concrete type.
    """


class PropertyRequirement:
    """Marker base for a fact that an operator may require."""


@dataclass(frozen=True)
class UnknownPartitioning(PartitioningProperty):
    """The operator does not advertise a usable partitioning guarantee."""


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
class UnknownOrdering(PropertyValue):
    pass


@dataclass(frozen=True)
class SortKey:
    key: str
    descending: bool = False
    nulls_first: bool = False


@dataclass(frozen=True)
class LocalOrdering(PropertyValue):
    sort_keys: Tuple[SortKey, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort_keys", tuple(self.sort_keys))


@dataclass(frozen=True)
class CompletePartitionBundles(PropertyValue, PropertyRequirement):
    """Every logical partition is delivered as one complete planner unit.

    This is intentionally separate from semantic partitioning.  A future
    exchange must re-establish both its distribution property and this delivery
    fact before a partition-consuming operator can run.
    """


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
class OrderedBy(PropertyRequirement):
    sort_keys: Tuple[SortKey, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort_keys", tuple(self.sort_keys))
        if not self.sort_keys:
            raise ValueError("OrderedBy requires at least one sort key")


class ValidationObligation:
    """Marker base for a proof that must be discharged inside a GPU stage."""


@dataclass(frozen=True)
class RowsOwnedByRange(ValidationObligation):
    leading_key: str
    scheme_id: str


@dataclass(frozen=True)
class PartitioningValidation:
    """One property-specific validation transform declared by a registry entry."""

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
            raise TypeError(
                "partitioning validation obligations must be "
                "ValidationObligation instances"
            )


@dataclass(frozen=True)
class PropertyReadvertisement:
    """Output facts and proofs for one explicit preservation declaration."""

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
    """A declared property cannot be safely re-advertised."""


PartitioningValidationFactory = Callable[
    [PartitioningProperty], Tuple[PartitioningValidation, ...]
]
_PARTITIONING_VALIDATION_FACTORIES: Dict[
    Type[PartitioningProperty], PartitioningValidationFactory
] = {}

PropertySatisfactionHandler = Callable[[PropertyValue, PropertyRequirement], bool]
_PROPERTY_SATISFACTION_HANDLERS: Dict[
    Tuple[Type[PropertyValue], Type[PropertyRequirement]],
    PropertySatisfactionHandler,
] = {}


def register_property_satisfaction(
    fact_type: Type[PropertyValue],
    requirement_type: Type[PropertyRequirement],
    handler: PropertySatisfactionHandler,
    *,
    replace: bool = False,
) -> None:
    """Register one exact-type satisfaction rule.

    The resolver always calls :func:`property_satisfies`; adding a future
    partitioning family therefore adds a handler without changing resolver
    control flow.
    """

    if not issubclass(fact_type, PropertyValue):
        raise TypeError("fact_type must derive from PropertyValue")
    if not issubclass(requirement_type, PropertyRequirement):
        raise TypeError("requirement_type must derive from PropertyRequirement")
    key = (fact_type, requirement_type)
    if key in _PROPERTY_SATISFACTION_HANDLERS and not replace:
        raise ValueError(
            "property satisfaction handler already registered for "
            f"{fact_type.__name__} -> {requirement_type.__name__}"
        )
    _PROPERTY_SATISFACTION_HANDLERS[key] = handler


def unregister_property_satisfaction(
    fact_type: Type[PropertyValue],
    requirement_type: Type[PropertyRequirement],
    *,
    expected: Optional[PropertySatisfactionHandler] = None,
) -> PropertySatisfactionHandler:
    """Remove an extension's exact-type satisfaction rule during teardown."""

    key = (fact_type, requirement_type)
    try:
        handler = _PROPERTY_SATISFACTION_HANDLERS[key]
    except KeyError as exc:
        raise KeyError(
            "property satisfaction handler is not registered for "
            f"{fact_type.__name__} -> {requirement_type.__name__}"
        ) from exc
    if expected is not None and handler is not expected:
        raise ValueError(
            "property satisfaction handler for "
            f"{fact_type.__name__} -> {requirement_type.__name__} is not the "
            "expected handler"
        )
    del _PROPERTY_SATISFACTION_HANDLERS[key]
    return handler


def property_satisfies(fact: PropertyFact, requirement: PropertyRequirement) -> bool:
    """Return whether one fact semantically satisfies one requirement."""

    handler = _PROPERTY_SATISFACTION_HANDLERS.get((type(fact.value), type(requirement)))
    return False if handler is None else handler(fact.value, requirement)


def requirements_satisfied(
    facts: Tuple[PropertyFact, ...],
    requirements: Tuple[PropertyRequirement, ...],
) -> bool:
    return all(
        any(property_satisfies(fact, requirement) for fact in facts)
        for requirement in requirements
    )


def register_partitioning_preservation(
    fact_type: Type[PartitioningProperty],
    validation_factory: PartitioningValidationFactory,
    *,
    replace: bool = False,
) -> None:
    """Register how a preserving UDF proves one partitioning property.

    Registration is exact-type, like :func:`register_property_satisfaction`.
    Adding a future partitioning family therefore requires a property type, a
    satisfaction rule, a validation transform/runtime, and this registration;
    the built-in MapBatches and MGP lowerers remain unchanged.
    """

    if not issubclass(fact_type, PartitioningProperty):
        raise TypeError("fact_type must derive from PartitioningProperty")
    if fact_type is UnknownPartitioning:
        raise ValueError("UnknownPartitioning cannot be preserved")
    if not callable(validation_factory):
        raise TypeError("validation_factory must be callable")
    if fact_type in _PARTITIONING_VALIDATION_FACTORIES and not replace:
        raise ValueError(
            "partitioning preservation already registered for " f"{fact_type.__name__}"
        )
    _PARTITIONING_VALIDATION_FACTORIES[fact_type] = validation_factory


def unregister_partitioning_preservation(
    fact_type: Type[PartitioningProperty],
    *,
    expected: Optional[PartitioningValidationFactory] = None,
) -> PartitioningValidationFactory:
    """Remove an extension's preservation registration during teardown."""

    try:
        factory = _PARTITIONING_VALIDATION_FACTORIES[fact_type]
    except KeyError as exc:
        raise KeyError(
            "partitioning preservation is not registered for " f"{fact_type.__name__}"
        ) from exc
    if expected is not None and factory is not expected:
        raise ValueError(
            f"partitioning preservation for {fact_type.__name__} is not the "
            "expected factory"
        )
    del _PARTITIONING_VALIDATION_FACTORIES[fact_type]
    return factory


def usable_partitioning_fact(
    facts: Tuple[PropertyFact, ...],
) -> Optional[PropertyFact]:
    """Return the single concrete partitioning fact advertised on an edge."""

    partitioning = tuple(
        fact for fact in facts if isinstance(fact.value, PartitioningProperty)
    )
    if len(partitioning) > 1:
        raise PartitioningPreservationError(
            f"edge advertises conflicting partitioning facts: {partitioning!r}"
        )
    if not partitioning or isinstance(
        partitioning[0].value,
        UnknownPartitioning,
    ):
        return None
    return partitioning[0]


def readvertise_preserved_properties(
    facts: Tuple[PropertyFact, ...],
    *,
    preserves_partitioning: bool,
    preserves_complete_partition_bundles: bool,
    plan_validations: bool = True,
) -> PropertyReadvertisement:
    """Plan the facts an operator explicitly re-advertises.

    No fact passes through implicitly.  Semantic partitioning is republished
    only after its registered runtime validation, while complete-bundle delivery
    is republished independently when the operator preserves that physical fact.
    """

    facts = tuple(facts)
    partitioning = usable_partitioning_fact(facts)
    validations: Tuple[PartitioningValidation, ...] = ()
    if preserves_partitioning and partitioning is not None:
        value = partitioning.value
        assert isinstance(value, PartitioningProperty)
        factory = _PARTITIONING_VALIDATION_FACTORIES.get(type(value))
        if factory is None:
            raise PartitioningPreservationError(
                "no partitioning preservation validator is registered for "
                f"{type(value).__name__}"
            )
        if plan_validations:
            validations = tuple(factory(value))
            if not validations or not all(
                isinstance(validation, PartitioningValidation)
                for validation in validations
            ):
                raise PartitioningPreservationError(
                    "partitioning preservation factory for "
                    f"{type(value).__name__} must return validations"
                )
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


def _range_satisfies_colocation(
    value: PropertyValue, requirement: PropertyRequirement
) -> bool:
    assert isinstance(value, RangePartitioning)
    assert isinstance(requirement, KeysColocated)
    # Disjoint ranges on a nonempty prefix guarantee that equal values for all
    # requested keys are assigned to the same range.  This does not discharge
    # a RowsOwnedByRange validation obligation.
    return (
        len(value.keys) <= len(requirement.keys)
        and requirement.keys[: len(value.keys)] == value.keys
    )


def _unknown_satisfies_nothing(
    _value: PropertyValue, _requirement: PropertyRequirement
) -> bool:
    return False


def _complete_satisfies_complete(
    value: PropertyValue, requirement: PropertyRequirement
) -> bool:
    return isinstance(value, CompletePartitionBundles) and isinstance(
        requirement, CompletePartitionBundles
    )


def _local_ordering_satisfies_ordered_by(
    value: PropertyValue, requirement: PropertyRequirement
) -> bool:
    assert isinstance(value, LocalOrdering)
    assert isinstance(requirement, OrderedBy)
    return value.sort_keys[: len(requirement.sort_keys)] == requirement.sort_keys


def _range_partitioning_validations(
    value: PartitioningProperty,
) -> Tuple[PartitioningValidation, ...]:
    """Declare the exact V1 range-ownership proof for preserving UDFs."""

    assert isinstance(value, RangePartitioning)
    # Keep the property vocabulary independent of transform implementation at
    # import time.  The concrete V1 implementation is loaded only when a planner
    # asks to preserve a RangePartitioning fact.
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


register_property_satisfaction(
    RangePartitioning, KeysColocated, _range_satisfies_colocation
)
register_property_satisfaction(
    UnknownPartitioning, KeysColocated, _unknown_satisfies_nothing
)
register_property_satisfaction(
    CompletePartitionBundles,
    CompletePartitionBundles,
    _complete_satisfies_complete,
)
# Retain the already-declared ordering vocabulary while partitioning V1 remains
# deliberately limited to UnknownPartitioning and RangePartitioning.
register_property_satisfaction(
    LocalOrdering, OrderedBy, _local_ordering_satisfies_ordered_by
)
register_partitioning_preservation(
    RangePartitioning,
    _range_partitioning_validations,
)
