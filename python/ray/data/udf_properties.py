from dataclasses import dataclass

from ray.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
@dataclass(frozen=True)
class UDFProperties:
    """User-supplied semantic properties of a Ray Data UDF.

    These properties are assertions used by opt-in execution backends. They don't
    change the behavior of ordinary Ray Data execution.

    Args:
        preserves_partitioning: Whether every output row remains owned by the same
            known logical input partition. This doesn't imply that the UDF preserves
            row count, key values, ordering, schema, determinism, purity, or batch
            independence. Backends are responsible for validating the concrete facts
            they rely on at runtime.
    """

    preserves_partitioning: bool = False
