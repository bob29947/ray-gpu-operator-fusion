#!/usr/bin/env python3
"""Run the fused ``map_group_partitions`` pipeline on modified Ray."""

from __future__ import annotations

import argparse

from pipeline_common import build_tokenized, run_pipeline
from pipeline_compat import partition_preservation_kwargs
from workload import GROUP_KEYS, build_sequences_partition_gpu


def build_pipeline(args: argparse.Namespace):
    tokenized = build_tokenized(
        args,
        map_batches_compat_kwargs=partition_preservation_kwargs(),
    )
    grouped = tokenized.groupby(
        list(GROUP_KEYS), num_partitions=args.num_partitions
    )
    return grouped.map_group_partitions(build_sequences_partition_gpu)


def main(argv: list[str] | None = None) -> int:
    return run_pipeline(
        build_pipeline,
        execution_api="map_group_partitions",
        enable_gpu_operator_fusion=True,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
