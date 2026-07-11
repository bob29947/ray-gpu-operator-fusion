#!/usr/bin/env python3
"""Shared machinery for the three-arm GPU operator-fusion benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from workload import (
    GROUP_KEYS,
    MERCHANT_HASH_SIZE,
    PROJECTION,
    SEQUENCE_CHUNK_SIZE,
    SEQUENCE_LENGTH,
    OptimalCudfTokenizer,
)


FAST_OPERATOR = "parquetcudfmapgrouppartitions"
MATRIX_PIPELINE_SOURCES = (
    "pipeline_common.py",
    "pipeline_compat.py",
    "pipeline_map_groups.py",
    "pipeline_map_group_partitions.py",
    "run_matrix.py",
    "workload.py",
)


def parse_args(
    argv: list[str] | None = None,
    *,
    default_batch_size: int,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local path or s3:// URI")
    parser.add_argument("--output", required=True, help="Local path or s3:// URI")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--phase", choices=("fresh", "cold", "warm"), required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--expected-ray-commit")
    parser.add_argument("--ray-wheel-sha256")
    parser.add_argument("--benchmark-commit", required=True)
    parser.add_argument("--run-matrix-sha256", required=True)
    parser.add_argument(
        "--expected-map-group-partitions-api",
        choices=("true", "false"),
    )
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--num-partitions", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--num-cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--num-gpus-per-worker", type=float, default=1.0)
    parser.add_argument("--cluster-cpus", type=int, default=64)
    parser.add_argument("--cluster-gpus", type=int, default=4)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--ray-temp-dir", type=Path)
    parser.add_argument("--write-complete-marker", type=Path)
    parser.add_argument("--compression", choices=("none", "zstd"), default="zstd")
    parser.add_argument("--compression-level", type=int, default=1)
    args = parser.parse_args(argv)
    for name in (
        "actors",
        "num_partitions",
        "batch_size",
        "cluster_cpus",
        "cluster_gpus",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_partitions % args.actors:
        parser.error("--num-partitions must be divisible by --actors")
    if args.num_cpus_per_worker <= 0 or args.num_gpus_per_worker <= 0:
        parser.error("worker resources must be positive")
    if args.compression_level < 0:
        parser.error("--compression-level must be nonnegative")
    if args.trial_index < 0:
        parser.error("--trial-index must be nonnegative")
    if args.ray_wheel_sha256 and (
        len(args.ray_wheel_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in args.ray_wheel_sha256
        )
    ):
        parser.error("--ray-wheel-sha256 must be a lowercase SHA-256 digest")
    if len(args.benchmark_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.benchmark_commit
    ):
        parser.error("--benchmark-commit must be a 40-character Git SHA")
    if len(args.run_matrix_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in args.run_matrix_sha256
    ):
        parser.error("--run-matrix-sha256 must be a lowercase SHA-256 digest")
    return args


def semantic_configuration(args: argparse.Namespace) -> dict[str, object]:
    """Settings that must be identical across all three benchmark arms."""
    return {
        "input": args.input.rstrip("/"),
        "projection": list(PROJECTION),
        "group_keys": list(GROUP_KEYS),
        "actors": args.actors,
        "num_partitions": args.num_partitions,
        "batch_size": args.batch_size,
        "worker_resources": {
            "num_cpus": args.num_cpus_per_worker,
            "num_gpus": args.num_gpus_per_worker,
        },
        "cluster_resources": {
            "num_cpus": args.cluster_cpus,
            "num_gpus": args.cluster_gpus,
        },
        "merchant_hash_size": MERCHANT_HASH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_chunk_size": SEQUENCE_CHUNK_SIZE,
        "shuffle": "hash",
        "writer": {
            "format": "parquet",
            "compression": args.compression,
            "compression_level": (
                args.compression_level if args.compression != "none" else None
            ),
            "use_dictionary": False,
            "write_statistics": False,
        },
    }


def build_tokenized(
    args: argparse.Namespace,
    *,
    map_batches_compat_kwargs: Mapping[str, object] | None = None,
):
    """Build the read and tokenizer prefix shared by every benchmark arm."""
    import ray.data as ray_data
    from ray.data.context import DataContext, ShuffleStrategy

    DataContext.get_current().shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE
    source = ray_data.read_parquet(args.input, columns=list(PROJECTION))
    compatibility_kwargs = dict(map_batches_compat_kwargs or {})
    unknown = set(compatibility_kwargs) - {"udf_properties"}
    if unknown:
        raise ValueError(
            f"unsupported map_batches compatibility keys: {sorted(unknown)}"
        )
    return source.map_batches(
        OptimalCudfTokenizer,
        fn_constructor_kwargs={"merchant_hash_size": MERCHANT_HASH_SIZE},
        batch_size=args.batch_size,
        batch_format="cudf",
        compute=ray_data.ActorPoolStrategy(size=args.actors),
        num_cpus=args.num_cpus_per_worker,
        num_gpus=args.num_gpus_per_worker,
        **compatibility_kwargs,
    )


def _walk_summaries(summary: object) -> Iterable[object]:
    yield summary
    for parent in getattr(summary, "parents", ()) or ():
        yield from _walk_summaries(parent)


def plan_evidence(dataset: object, stats: str) -> dict[str, object]:
    names: list[str] = []
    try:
        for summary in _walk_summaries(dataset.get_stats_summary()):
            names.extend(
                str(operator.operator_name)
                for operator in (getattr(summary, "operators_stats", ()) or ())
            )
        source = "stats_summary"
    except (AttributeError, RuntimeError, TypeError, ValueError):
        for line in stats.splitlines():
            if "Operator " in line and ":" in line:
                names.append(line.split(":", 1)[0].split(maxsplit=2)[-1].strip())
        source = "stats_text"
    normalized = [name.replace("_", "").replace("-", "").lower() for name in names]
    optimized = [name for name in normalized if FAST_OPERATOR in name]
    gpu_operators = [
        name
        for name in normalized
        if "gpu[" in name or "actorpoolgpuoperator" in name
    ]
    exchanges = [
        name
        for name in normalized
        if "hashshuffle" in name
        or "gpushuffle" in name
        or name.startswith("shuffle(")
        or name.startswith("repartition(")
    ]
    return {
        "source": source,
        "operator_names": names,
        "optimizer_operator_count": len(optimized),
        "gpu_operator_count": len(gpu_operators),
        "gpu_operator_names": gpu_operators,
        "exchange_operator_names": exchanges,
    }


def _tensor_metadata(field: object) -> tuple[tuple[int, ...] | None, object | None]:
    field_type = getattr(field, "type", None)
    shape = getattr(field_type, "shape", None)
    dtype = getattr(field_type, "value_type", None)
    metadata = {
        (key.decode() if isinstance(key, bytes) else str(key)): (
            value.decode() if isinstance(value, bytes) else str(value)
        )
        for key, value in (getattr(field, "metadata", None) or {}).items()
    }
    if shape is None and "ray.data.fixed_size_binary_tensor.shape" in metadata:
        shape = tuple(json.loads(metadata["ray.data.fixed_size_binary_tensor.shape"]))
    if dtype is None:
        dtype = metadata.get("ray.data.fixed_size_binary_tensor.dtype")
    if dtype is not None and not isinstance(dtype, str):
        try:
            dtype = dtype.to_pandas_dtype()
        except (AttributeError, TypeError):
            dtype = str(dtype)
    return (tuple(shape) if shape is not None else None), dtype


def _array_values(array: object) -> Iterable[object]:
    try:
        values = array.to_numpy(zero_copy_only=False)
    except (AttributeError, TypeError, ValueError):
        values = None
    if isinstance(values, np.ndarray) and values.ndim >= 2:
        yield from values
    else:
        yield from array.to_pylist()


def parquet_digest(output: str) -> dict[str, object]:
    """Validate and order-independently hash every uint16[4096] output row."""
    import pyarrow.dataset as pads

    dataset = pads.dataset(output, format="parquet")
    if dataset.schema.names != ["input_ids"]:
        raise ValueError(
            "output schema must contain only input_ids, got "
            f"{dataset.schema.names!r}"
        )
    shape, dtype = _tensor_metadata(dataset.schema.field("input_ids"))
    expected_shape = (SEQUENCE_LENGTH,)
    expected_dtype = np.dtype("uint16")
    if shape != expected_shape or np.dtype(dtype) != expected_dtype:
        raise ValueError(
            f"input_ids must be uint16[{SEQUENCE_LENGTH}], got dtype={dtype}, shape={shape}"
        )
    rows: list[bytes] = []
    for batch in dataset.scanner(columns=["input_ids"], batch_size=4096).to_batches():
        for value in _array_values(batch.column(0)):
            array = np.asarray(value, dtype="<u2")
            if array.shape != expected_shape:
                raise ValueError(f"unexpected tensor row shape: {array.shape}")
            rows.append(hashlib.sha256(array.tobytes(order="C")).digest())
    digest = hashlib.sha256()
    digest.update(b"ray-parquet-range-map-groups-canonical-v1\0")
    digest.update(len(rows).to_bytes(8, "little"))
    for row_hash in sorted(rows):
        digest.update(row_hash)
    return {
        "algorithm": "sha256-sorted-row-sha256-v1",
        "rows": len(rows),
        "sha256": digest.hexdigest(),
        "dtype": "uint16",
        "shape": [SEQUENCE_LENGTH],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(uri: str) -> list[dict[str, object]]:
    """Return a stable size/mtime inventory for a local or S3 Parquet input."""
    import pyarrow.fs as pafs

    normalized = uri if "://" in uri else str(Path(uri).expanduser().resolve())
    filesystem, path = pafs.FileSystem.from_uri(normalized)
    info = filesystem.get_file_info(path)
    items = (
        [info]
        if info.is_file
        else filesystem.get_file_info(pafs.FileSelector(path, recursive=True))
    )
    return sorted(
        [
            {
                "path": item.path,
                "bytes": item.size,
                "mtime_ns": item.mtime_ns,
            }
            for item in items
            if item.is_file and item.path.endswith(".parquet")
        ],
        key=lambda item: str(item["path"]),
    )


def dependency_versions() -> dict[str, str]:
    candidates = {
        "numpy": ("numpy",),
        "scipy": ("scipy",),
        "pandas": ("pandas",),
        "pyarrow": ("pyarrow",),
        "fsspec": ("fsspec",),
        "s3fs": ("s3fs",),
        "boto3": ("boto3",),
        "botocore": ("botocore",),
        "cudf": ("cudf-cu12", "cudf"),
        "cupy": ("cupy-cuda12x", "cupy"),
        "kvikio": ("kvikio-cu12", "kvikio"),
        "rmm": ("rmm-cu12", "rmm"),
    }
    versions = {}
    for logical_name, names in candidates.items():
        for distribution in names:
            try:
                versions[logical_name] = importlib.metadata.version(distribution)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        else:
            versions[logical_name] = "not-installed"
    return versions


def installed_ray_wheel_sha256() -> str | None:
    direct_url = importlib.metadata.distribution("ray").read_text("direct_url.json")
    if not direct_url:
        return None
    archive = json.loads(direct_url).get("archive_info", {})
    digest = archive.get("hashes", {}).get("sha256")
    if digest is None and str(archive.get("hash", "")).startswith("sha256="):
        digest = str(archive["hash"]).removeprefix("sha256=")
    return str(digest) if digest else None


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_pipeline(
    build_pipeline: Callable[[argparse.Namespace], object],
    *,
    execution_api: str,
    enable_gpu_operator_fusion: bool = False,
    argv: list[str] | None = None,
) -> int:
    """Run one explicit benchmark arm and persist its evidence."""
    if execution_api not in {"map_groups", "map_group_partitions"}:
        raise ValueError(f"unknown execution API: {execution_api}")
    process_started_at_unix_ns = time.time_ns()
    default_batch_size = (
        1_048_576 if execution_api == "map_groups" else 2_097_152
    )
    args = parse_args(argv, default_batch_size=default_batch_size)
    runner_name = "run_matrix.py"
    expected_runner_sha256 = args.run_matrix_sha256
    source_files = MATRIX_PIPELINE_SOURCES
    runner_sha256 = _sha256(Path(__file__).with_name(runner_name))
    if runner_sha256 != expected_runner_sha256:
        raise RuntimeError(
            f"{runner_name} hash differs from the clean benchmark provenance"
        )
    benchmark_provenance = {
        "git_commit": args.benchmark_commit,
        runner_name.replace(".py", "_sha256"): runner_sha256,
    }
    from pipeline_compat import configure_data_context

    context_evidence = configure_data_context(
        enable_gpu_operator_fusion=enable_gpu_operator_fusion
    )
    context_evidence["udf_properties_applied"] = (
        execution_api == "map_group_partitions"
        and bool(context_evidence["udf_properties_supported"])
    )
    import ray
    from ray.data.grouped_data import GroupedData

    api_supported = hasattr(GroupedData, "map_group_partitions")
    expected_api_support = (
        execution_api == "map_group_partitions"
        if args.expected_map_group_partitions_api is None
        else args.expected_map_group_partitions_api == "true"
    )
    if api_supported is not expected_api_support:
        raise RuntimeError(
            f"{execution_api} entrypoint expected map_group_partitions API "
            f"support to be {expected_api_support}, got {api_supported}"
        )

    init_kwargs: dict[str, object] = {}
    if args.ray_address:
        init_kwargs["address"] = args.ray_address
    else:
        init_kwargs.update(num_cpus=args.cluster_cpus, num_gpus=args.cluster_gpus)
        if args.ray_temp_dir:
            args.ray_temp_dir.mkdir(parents=True, exist_ok=True)
            init_kwargs["_temp_dir"] = str(args.ray_temp_dir.absolute())
    ray.init(**init_kwargs)
    if (
        args.expected_ray_commit
        and getattr(ray, "__commit__", None) != args.expected_ray_commit
    ):
        raise RuntimeError(
            f"expected Ray commit {args.expected_ray_commit}, got "
            f"{getattr(ray, '__commit__', None)}"
        )
    installed_wheel_sha256 = installed_ray_wheel_sha256()
    if args.ray_wheel_sha256 and installed_wheel_sha256 != args.ray_wheel_sha256:
        raise RuntimeError(
            f"expected Ray wheel {args.ray_wheel_sha256}, got {installed_wheel_sha256}"
        )
    input_before = input_identity(args.input)
    if not input_before:
        raise RuntimeError(f"input contains no Parquet files: {args.input}")
    started = time.perf_counter()
    try:
        dataset = build_pipeline(args)
        writer_args: dict[str, object] = {
            "mode": "overwrite",
            "compression": None if args.compression == "none" else args.compression,
            "use_dictionary": False,
            "write_statistics": False,
        }
        if args.compression != "none":
            writer_args["compression_level"] = args.compression_level
        dataset.write_parquet(args.output, **writer_args)
        elapsed = time.perf_counter() - started
        if args.write_complete_marker:
            args.write_complete_marker.parent.mkdir(parents=True, exist_ok=True)
            args.write_complete_marker.touch()
        stats = dataset.stats()
        input_after = input_identity(args.input)
        if input_after != input_before:
            raise RuntimeError("input identity changed while the pipeline was running")
        logical_digest = parquet_digest(args.output)
        process_completed_at_unix_ns = time.time_ns()
        result = {
            "schema_version": 1,
            "label": args.label,
            "phase": args.phase,
            "cohort": args.cohort,
            "trial_index": args.trial_index,
            "started_at_unix_ns": process_started_at_unix_ns,
            "completed_at_unix_ns": process_completed_at_unix_ns,
            "semantic_configuration": semantic_configuration(args),
            "input_identity": input_before,
            "execution_api": execution_api,
            "map_group_partitions_api_supported": api_supported,
            "data_context": context_evidence,
            "pipeline_elapsed_s": elapsed,
            "logical_digest": logical_digest,
            "plan": plan_evidence(dataset, stats),
            "stats": stats,
            "runtime": {
                "python": sys.executable,
                "python_implementation": platform.python_implementation(),
                "python_version": sys.version,
                "python_cache_tag": sys.implementation.cache_tag,
                "sys_prefix": sys.prefix,
                "ray_file": ray.__file__,
                "ray_version": ray.__version__,
                "ray_commit": getattr(ray, "__commit__", None),
                "ray_wheel_sha256": installed_wheel_sha256,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "dependencies": dependency_versions(),
            "source_sha256": {
                name: _sha256(Path(__file__).with_name(name))
                for name in source_files
            },
            "benchmark_provenance": benchmark_provenance,
        }
        _write_json(args.result, result)
    finally:
        ray.shutdown()
    return 0
