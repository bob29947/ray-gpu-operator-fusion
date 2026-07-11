#!/usr/bin/env python3
"""Run the shared legacy ``map_groups`` pipeline on stock or modified Ray."""

from __future__ import annotations

import argparse

from pipeline_common import build_tokenized, run_pipeline
from workload import (
    GROUP_KEYS,
    SEQUENCE_CHUNK_SIZE,
    SEQUENCE_LENGTH,
    build_sequences_gpu,
)


def build_pipeline(args: argparse.Namespace):
    import ray.data as ray_data

    tokenized = build_tokenized(args)
    grouped = tokenized.groupby(
        list(GROUP_KEYS), num_partitions=args.num_partitions
    )
    return grouped.map_groups(
        build_sequences_gpu,
        fn_kwargs={
            "sequence_length": SEQUENCE_LENGTH,
            "chunk_size": SEQUENCE_CHUNK_SIZE,
        },
        batch_format="cudf",
        compute=ray_data.TaskPoolStrategy(size=args.actors),
        num_cpus=args.num_cpus_per_worker,
        num_gpus=args.num_gpus_per_worker,
        zero_copy_batch=True,
    )


def main(argv: list[str] | None = None) -> int:
    return run_pipeline(
        build_pipeline,
        execution_api="map_groups",
        enable_gpu_operator_fusion=False,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
