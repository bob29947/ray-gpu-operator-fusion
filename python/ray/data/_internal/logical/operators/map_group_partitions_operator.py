from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Union

from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data.block import BlockMetadata, DataBatch
from ray.data.udf_properties import UDFProperties

__all__ = ["MapGroupPartitions"]


@dataclass(frozen=True, repr=False, eq=False)
class MapGroupPartitions(LogicalOperator):
    """Logical operator for :meth:`GroupedData.map_group_partitions`."""

    key: Union[str, List[str]]
    num_partitions: Optional[int]
    fn: Callable[..., Union[DataBatch, Iterator[DataBatch]]]
    udf_properties: Optional[UDFProperties] = None
    input_dependencies: List[LogicalOperator] = field(repr=False, kw_only=True)

    def __post_init__(self):
        assert len(self.input_dependencies) == 1, len(self.input_dependencies)
        if self.udf_properties is not None and not isinstance(
            self.udf_properties, UDFProperties
        ):
            raise TypeError("udf_properties must be a UDFProperties instance or None")
        object.__setattr__(self, "_name", "MapGroupPartitions")

    @property
    def num_outputs(self) -> Optional[int]:
        return self.num_partitions

    def infer_metadata(self) -> BlockMetadata:
        input_metadata = self.input_dependencies[0].infer_metadata()
        return BlockMetadata(
            num_rows=None,
            size_bytes=None,
            input_files=input_metadata.input_files,
            exec_stats=None,
        )
