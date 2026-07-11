#!/usr/bin/env python3
"""Run and validate the local three-arm Ray GPU-fusion benchmark matrix."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from pipeline_common import MATRIX_PIPELINE_SOURCES


EXPECTED_OUTPUT_ROWS = 1_229_748
EXPECTED_OUTPUT_SHA256 = (
    "cd440f692ed6e69f3980600a73afdb730b29ec67a8725bfbb484cc0b15edec85"
)
LEGACY_BATCH_SIZE = 1_048_576
NEW_API_BATCH_SIZE = 2_097_152
LEGACY_RELATIVE_TOLERANCE = 0.10
V1_MAX_SECONDS = 17.15
V1_MIN_SPEEDUP = 7.0


@dataclass(frozen=True)
class ArmSpec:
    build: str
    entrypoint: str
    execution_api: str
    batch_size: int
    api_supported: bool
    fusion_flag_supported: bool
    fusion_enabled: bool | None
    udf_properties_supported: bool


ARM_SPECS: dict[str, ArmSpec] = {
    "stock_legacy": ArmSpec(
        build="stock",
        entrypoint="pipeline_map_groups.py",
        execution_api="map_groups",
        batch_size=LEGACY_BATCH_SIZE,
        api_supported=False,
        fusion_flag_supported=False,
        fusion_enabled=None,
        udf_properties_supported=False,
    ),
    "modified_legacy": ArmSpec(
        build="modified",
        entrypoint="pipeline_map_groups.py",
        execution_api="map_groups",
        batch_size=LEGACY_BATCH_SIZE,
        api_supported=True,
        fusion_flag_supported=True,
        fusion_enabled=False,
        udf_properties_supported=True,
    ),
    "modified_v1": ArmSpec(
        build="modified",
        entrypoint="pipeline_map_group_partitions.py",
        execution_api="map_group_partitions",
        batch_size=NEW_API_BATCH_SIZE,
        api_supported=True,
        fusion_flag_supported=True,
        fusion_enabled=True,
        udf_properties_supported=True,
    ),
}
ARMS = tuple(ARM_SPECS)
BUILDS = ("stock", "modified")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _benchmark_provenance() -> dict[str, str]:
    root = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"benchmark Git inspection failed: {completed.stdout.strip()}"
            )
        return completed.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=normal")
    if status:
        raise RuntimeError(
            "benchmark checkout must be clean before a measured run:\n" + status
        )
    commit = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"benchmark Git commit is invalid: {commit!r}")
    return {
        "git_commit": commit,
        "run_matrix_sha256": _sha256(Path(__file__)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).with_name("artifacts") / "matrix-current",
    )
    parser.add_argument("--stock-python", type=Path, required=True)
    parser.add_argument("--modified-python", type=Path, required=True)
    parser.add_argument("--stock-commit", required=True)
    parser.add_argument("--modified-commit", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--num-partitions", type=int, default=8)
    parser.add_argument("--cluster-cpus", type=int, default=64)
    parser.add_argument("--cluster-gpus", type=int, default=4)
    parser.add_argument("--gpu-devices", default="0,1,2,3")
    parser.add_argument("--compression", choices=("none", "zstd"), default="zstd")
    parser.add_argument("--compression-level", type=int, default=1)
    args = parser.parse_args(argv)

    for name in (
        "repetitions",
        "actors",
        "num_partitions",
        "cluster_cpus",
        "cluster_gpus",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_partitions % args.actors:
        parser.error("--num-partitions must be divisible by --actors")
    if args.compression_level < 0:
        parser.error("--compression-level must be nonnegative")
    if len([item for item in args.gpu_devices.split(",") if item]) != args.cluster_gpus:
        parser.error("--gpu-devices count must equal --cluster-gpus")
    commits = []
    for build in BUILDS:
        commit = getattr(args, f"{build}_commit")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            parser.error(f"--{build}-commit must be a 40-character Git SHA")
        commits.append(commit)
    if len(set(commits)) != len(commits):
        parser.error("stock and modified commits must be distinct")
    return args


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "RAY_ADDRESS"):
        env.pop(name, None)
    env.update(PYTHONHASHSEED="0", PYTHONNOUSERSITE="1")
    return env


def _probe_python(python: Path) -> dict[str, object]:
    code = r'''
import importlib.metadata
import inspect
import json
import platform
import sys
import ray
import ray.data as ray_data
from ray.data.context import DataContext
from ray.data.dataset import Dataset
from ray.data.grouped_data import GroupedData

context = DataContext.get_current()
if not hasattr(context, "use_datasource_v2"):
    raise RuntimeError("missing DataContext.use_datasource_v2")
context.use_datasource_v2 = False
if context.use_datasource_v2 is not False:
    raise RuntimeError("failed to disable datasource V2")
direct_url = importlib.metadata.distribution("ray").read_text("direct_url.json")
archive = json.loads(direct_url).get("archive_info", {}) if direct_url else {}
wheel_sha256 = archive.get("hashes", {}).get("sha256")
if wheel_sha256 is None and str(archive.get("hash", "")).startswith("sha256="):
    wheel_sha256 = str(archive["hash"]).removeprefix("sha256=")
document = {
    "python": sys.executable,
    "python_implementation": platform.python_implementation(),
    "python_version": sys.version,
    "python_cache_tag": sys.implementation.cache_tag,
    "sys_prefix": sys.prefix,
    "ray_file": ray.__file__,
    "ray_commit": getattr(ray, "__commit__", None),
    "ray_wheel_sha256": wheel_sha256,
    "use_datasource_v2": context.use_datasource_v2,
    "map_group_partitions_api_supported": hasattr(GroupedData, "map_group_partitions"),
    "fusion_flag_supported": hasattr(context, "enable_gpu_operator_fusion"),
    "udf_properties_exported": hasattr(ray_data, "UDFProperties"),
    "map_batches_udf_properties_supported": (
        "udf_properties" in inspect.signature(Dataset.map_batches).parameters
    ),
}
print("RAY_MATRIX_PROBE=" + json.dumps(document, sort_keys=True))
'''
    completed = subprocess.run(
        [str(python.expanduser().absolute()), "-c", code],
        env=_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    marker = "RAY_MATRIX_PROBE="
    line = next(
        (line for line in completed.stdout.splitlines() if line.startswith(marker)),
        None,
    )
    if completed.returncode or line is None:
        raise RuntimeError(
            f"Ray preflight failed for {python} with exit code "
            f"{completed.returncode}:\n{completed.stdout}"
        )
    document = json.loads(line.removeprefix(marker))
    if not isinstance(document, dict):
        raise RuntimeError(f"invalid Ray preflight document from {python}")
    return document


def _preflight(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    configured = {
        build: getattr(args, f"{build}_python").expanduser().absolute()
        for build in BUILDS
    }
    if len(set(configured.values())) != len(configured):
        raise ValueError("stock and modified require distinct interpreters")
    for build, interpreter in configured.items():
        if not interpreter.is_file():
            raise ValueError(f"{build} Python does not exist: {interpreter}")

    probes = {build: _probe_python(path) for build, path in configured.items()}
    expected_capabilities = {
        "stock": (False, False, False),
        "modified": (True, True, True),
    }
    for build, probe in probes.items():
        expected_commit = getattr(args, f"{build}_commit")
        if probe.get("ray_commit") != expected_commit:
            raise RuntimeError(
                f"{build} expected Ray commit {expected_commit}, "
                f"got {probe.get('ray_commit')}"
            )
        if probe.get("use_datasource_v2") is not False:
            raise RuntimeError(f"{build} did not disable datasource V2 in preflight")
        if probe.get("udf_properties_exported") is not probe.get(
            "map_batches_udf_properties_supported"
        ):
            raise RuntimeError(
                f"{build} UDFProperties export and map_batches parameter disagree"
            )
        observed = (
            probe.get("map_group_partitions_api_supported"),
            probe.get("fusion_flag_supported"),
            probe.get("udf_properties_exported"),
        )
        if observed != expected_capabilities[build]:
            raise RuntimeError(
                f"{build} capabilities {observed!r} differ from "
                f"{expected_capabilities[build]!r}"
            )
        wheel_hash = str(probe.get("ray_wheel_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", wheel_hash):
            raise RuntimeError(f"{build} Ray is not installed from an identified wheel")
        reported = Path(str(probe.get("python", ""))).absolute()
        if reported != configured[build]:
            raise RuntimeError(
                f"{build} reported interpreter {reported}, expected {configured[build]}"
            )
        ray_file = Path(str(probe.get("ray_file", ""))).resolve()
        prefix = Path(str(probe.get("sys_prefix", ""))).resolve()
        if not ray_file.is_relative_to(prefix):
            raise RuntimeError(f"{build} imported Ray outside its environment")

    for field in ("sys_prefix", "ray_file", "ray_wheel_sha256"):
        if len({str(probes[build].get(field)) for build in BUILDS}) != len(BUILDS):
            raise RuntimeError(f"benchmark builds share preflight {field}")
    for field in ("python_implementation", "python_version", "python_cache_tag"):
        if len({str(probes[build].get(field)) for build in BUILDS}) != 1:
            raise RuntimeError(f"benchmark builds differ in {field}")
    return probes


def _pipeline_command(
    *,
    arm: str,
    python: Path,
    args: argparse.Namespace,
    result_path: Path,
    output: str,
    label: str,
    ordinal: int,
    preflight: Mapping[str, object],
    benchmark_provenance: Mapping[str, str],
    ray_temp_dir: Path,
    write_complete_marker: Path,
) -> list[str]:
    spec = ARM_SPECS[arm]
    return [
        str(python.expanduser().absolute()),
        str(Path(__file__).with_name(spec.entrypoint)),
        "--input",
        args.input,
        "--output",
        output,
        "--result",
        str(result_path),
        "--label",
        label,
        "--phase",
        "fresh",
        "--cohort",
        "local-matrix",
        "--trial-index",
        str(ordinal // len(ARMS)),
        "--expected-ray-commit",
        str(preflight["ray_commit"]),
        "--ray-wheel-sha256",
        str(preflight["ray_wheel_sha256"]),
        "--benchmark-commit",
        benchmark_provenance["git_commit"],
        "--run-matrix-sha256",
        benchmark_provenance["run_matrix_sha256"],
        "--expected-map-group-partitions-api",
        str(spec.api_supported).lower(),
        "--actors",
        str(args.actors),
        "--num-partitions",
        str(args.num_partitions),
        "--batch-size",
        str(spec.batch_size),
        "--cluster-cpus",
        str(args.cluster_cpus),
        "--cluster-gpus",
        str(args.cluster_gpus),
        "--ray-temp-dir",
        str(ray_temp_dir),
        "--write-complete-marker",
        str(write_complete_marker),
        "--compression",
        args.compression,
        "--compression-level",
        str(args.compression_level),
    ]


def _output_path(root: str, label: str) -> str:
    return root.rstrip("/") + "/" + label if "://" in root else str(Path(root) / label)


@contextlib.contextmanager
def _short_ray_temp(target: Path) -> Iterator[Path]:
    target.mkdir(parents=True, exist_ok=True)
    parent = Path(tempfile.mkdtemp(prefix="ray-matrix-"))
    link = parent / "r"
    link.symlink_to(target.resolve(), target_is_directory=True)
    try:
        yield link
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def _load(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"result is not a JSON object: {path}")
    return document


def _run_one(
    *,
    arm: str,
    ordinal: int,
    args: argparse.Namespace,
    stage: Path,
    preflights: Mapping[str, Mapping[str, object]],
    benchmark_provenance: Mapping[str, str],
) -> dict[str, object]:
    spec = ARM_SPECS[arm]
    python = getattr(args, f"{spec.build}_python")
    preflight = preflights[spec.build]
    label = f"{ordinal:02d}-{arm}"
    result_path = stage / arm / f"{ordinal:02d}.json"
    log_path = stage / arm / f"{ordinal:02d}.log"
    output = _output_path(args.output_root, label)
    runtime = stage / ".runtime" / label
    marker = runtime / "write-complete"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    env = _child_env()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    with _short_ray_temp(runtime) as short_runtime:
        command = _pipeline_command(
            arm=arm,
            python=python,
            args=args,
            result_path=result_path,
            output=output,
            label=label,
            ordinal=ordinal,
            preflight=preflight,
            benchmark_provenance=benchmark_provenance,
            ray_temp_dir=short_runtime,
            write_complete_marker=marker,
        )
        with log_path.open("w") as log:
            started_at_unix_ns = time.time_ns()
            completed = subprocess.run(
                command,
                cwd=Path(__file__).parent,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        process_end_to_end_s = (
            (marker.stat().st_mtime_ns - started_at_unix_ns) / 1_000_000_000
            if completed.returncode == 0 and marker.is_file()
            else None
        )
    shutil.rmtree(runtime, ignore_errors=True)
    if "://" not in output:
        shutil.rmtree(output, ignore_errors=True)
    if completed.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
        raise RuntimeError(
            f"{arm} pipeline failed with exit code {completed.returncode}\n{tail}"
        )
    if process_end_to_end_s is None or process_end_to_end_s <= 0:
        raise RuntimeError(f"{arm} did not record valid write completion")

    result = _load(result_path)
    if result.get("benchmark_provenance") != benchmark_provenance:
        raise RuntimeError(f"{arm} benchmark provenance drifted")
    runtime_evidence = result.get("runtime", {})
    for field in (
        "ray_commit",
        "ray_wheel_sha256",
        "python_implementation",
        "python_version",
        "python_cache_tag",
    ):
        if runtime_evidence.get(field) != preflight.get(field):
            raise RuntimeError(f"{arm} runtime {field} drifted from preflight")
    result["process_end_to_end_s"] = process_end_to_end_s
    result["matrix_arm"] = arm
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values or any(value <= 0 for value in values):
        raise AssertionError("elapsed times must be positive")
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _relative_delta(actual: float, baseline: float) -> float:
    return actual / baseline - 1.0


def validate_and_summarize(
    results: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    expected_benchmark_commit: str | None = None,
    expected_run_matrix_sha256: str | None = None,
    enforce_performance_gates: bool = True,
) -> dict[str, object]:
    if set(results) != set(ARMS) or any(not results[arm] for arm in ARMS):
        raise AssertionError("all three benchmark arms are required")
    counts = {len(results[arm]) for arm in ARMS}
    if len(counts) != 1:
        raise AssertionError("all benchmark arms require equal trial counts")
    flattened = [result for arm in ARMS for result in results[arm]]

    configurations = {}
    shared_configurations = set()
    for arm in ARMS:
        unique = {
            json.dumps(result["semantic_configuration"], sort_keys=True)
            for result in results[arm]
        }
        if len(unique) != 1:
            raise AssertionError(f"{arm} semantic configuration changed across trials")
        configuration = dict(results[arm][0]["semantic_configuration"])
        configurations[arm] = configuration
        shared = dict(configuration)
        shared.pop("batch_size", None)
        shared_configurations.add(json.dumps(shared, sort_keys=True))
        if configuration.get("batch_size") != ARM_SPECS[arm].batch_size:
            raise AssertionError(f"{arm} used the wrong tuned batch size")
    if len(shared_configurations) != 1:
        raise AssertionError("matrix semantics differ beyond tuned batch size")

    for field, error in (
        ("source_sha256", "matrix arms did not run identical source files"),
        ("dependencies", "matrix dependency versions differ"),
        ("input_identity", "matrix input identity differs"),
        ("logical_digest", "matrix outputs differ"),
        ("benchmark_provenance", "matrix benchmark provenance differs"),
    ):
        values = {json.dumps(result[field], sort_keys=True) for result in flattened}
        if len(values) != 1:
            raise AssertionError(error)

    source_hashes = dict(flattened[0]["source_sha256"])
    if set(source_hashes) != set(MATRIX_PIPELINE_SOURCES):
        raise AssertionError("matrix source hashes do not cover every shared source")
    if any(
        not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in source_hashes.values()
    ):
        raise AssertionError("matrix source hashes must be lowercase SHA-256 values")

    provenance = dict(flattened[0]["benchmark_provenance"])
    benchmark_commit = provenance.get("git_commit")
    runner_hash = provenance.get("run_matrix_sha256")
    if not isinstance(benchmark_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", benchmark_commit
    ):
        raise AssertionError("results must identify one benchmark Git commit")
    if not isinstance(runner_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", runner_hash
    ):
        raise AssertionError("results must identify one run_matrix.py SHA-256")
    if expected_benchmark_commit and benchmark_commit != expected_benchmark_commit:
        raise AssertionError("result benchmark commit differs from expected")
    if expected_run_matrix_sha256 and runner_hash != expected_run_matrix_sha256:
        raise AssertionError("result run_matrix.py hash differs from expected")

    digest = dict(flattened[0]["logical_digest"])
    expected_digest = {
        "algorithm": "sha256-sorted-row-sha256-v1",
        "dtype": "uint16",
        "rows": EXPECTED_OUTPUT_ROWS,
        "sha256": EXPECTED_OUTPUT_SHA256,
        "shape": [4096],
    }
    if digest != expected_digest:
        raise AssertionError("matrix output does not match the canonical digest")

    build_runtime: dict[str, dict[str, object]] = {}
    for arm, spec in ARM_SPECS.items():
        for result in results[arm]:
            if result.get("matrix_arm") not in (None, arm):
                raise AssertionError(f"{arm} result has the wrong matrix arm")
            if result.get("execution_api") != spec.execution_api:
                raise AssertionError(f"{arm} used the wrong execution API")
            if (
                result.get("map_group_partitions_api_supported")
                is not spec.api_supported
            ):
                raise AssertionError(f"{arm} reported the wrong API support")
            context = result.get("data_context", {})
            expected_context = {
                "use_datasource_v2": False,
                "fusion_flag_supported": spec.fusion_flag_supported,
                "enable_gpu_operator_fusion": spec.fusion_enabled,
                "udf_properties_supported": spec.udf_properties_supported,
                "udf_properties_applied": (
                    spec.execution_api == "map_group_partitions"
                    and spec.udf_properties_supported
                ),
                "map_group_partitions_api_supported": spec.api_supported,
            }
            if context != expected_context:
                raise AssertionError(
                    f"{arm} data context/compatibility evidence differs"
                )

            plan = result["plan"]
            exchanges = [
                str(name).replace("_", "").replace("-", "").lower()
                for name in plan["exchange_operator_names"]
            ]
            legacy = arm in {"stock_legacy", "modified_legacy"}
            if legacy:
                if not any("hashshuffle" in name for name in exchanges):
                    raise AssertionError(
                        f"{arm} did not execute the stock hash shuffle"
                    )
                if int(plan.get("optimizer_operator_count", 0)) != 0:
                    raise AssertionError(f"{arm} selected a specialized operator")
                if int(plan.get("gpu_operator_count", 0)) != 0:
                    raise AssertionError(f"{arm} selected the new GPU operator")
            elif exchanges:
                raise AssertionError(f"{arm} executed a shuffle or repartition")
            else:
                if int(plan.get("optimizer_operator_count", 0)) != 0:
                    raise AssertionError("modified V1 used a monolithic operator")
                if int(plan.get("gpu_operator_count", 0)) != 1:
                    raise AssertionError(
                        "modified V1 did not execute one fused GPU node"
                    )

        runtimes = {
            json.dumps(result["runtime"], sort_keys=True)
            for result in results[arm]
        }
        if len(runtimes) != 1:
            raise AssertionError(f"{arm} runtime changed across trials")
        runtime = dict(results[arm][0]["runtime"])
        identity = {
            field: runtime.get(field)
            for field in (
                "python",
                "python_implementation",
                "python_version",
                "python_cache_tag",
                "sys_prefix",
                "ray_file",
                "ray_commit",
                "ray_wheel_sha256",
            )
        }
        previous = build_runtime.setdefault(spec.build, identity)
        if previous != identity:
            raise AssertionError(f"{spec.build} runtime differs between matrix arms")

    if len({build_runtime[build]["ray_commit"] for build in BUILDS}) != len(BUILDS):
        raise AssertionError("matrix builds must use distinct Ray commits")
    if len(
        {build_runtime[build]["ray_wheel_sha256"] for build in BUILDS}
    ) != len(BUILDS):
        raise AssertionError("matrix builds must use distinct Ray wheels")
    python_runtime = {
        json.dumps(
            {
                field: build_runtime[build][field]
                for field in (
                    "python_implementation",
                    "python_version",
                    "python_cache_tag",
                )
            },
            sort_keys=True,
        )
        for build in BUILDS
    }
    if len(python_runtime) != 1:
        raise AssertionError("matrix builds use different Python runtimes")

    timing = {
        arm: _distribution(
            [float(result["pipeline_elapsed_s"]) for result in results[arm]]
        )
        for arm in ARMS
    }
    medians = {arm: timing[arm]["median"] for arm in ARMS}
    gates = {
        "modified_legacy_relative_to_stock": _relative_delta(
            medians["modified_legacy"], medians["stock_legacy"]
        ),
        "modified_v1_seconds": medians["modified_v1"],
        "modified_v1_speedup_over_stock": (
            medians["stock_legacy"] / medians["modified_v1"]
        ),
    }
    if enforce_performance_gates:
        if (
            abs(gates["modified_legacy_relative_to_stock"])
            > LEGACY_RELATIVE_TOLERANCE
        ):
            raise AssertionError(
                "modified legacy performance differs from stock by >10%"
            )
        if gates["modified_v1_seconds"] > V1_MAX_SECONDS:
            raise AssertionError("modified V1 median exceeds 17.15 seconds")
        if gates["modified_v1_speedup_over_stock"] < V1_MIN_SPEEDUP:
            raise AssertionError("modified V1 speedup over stock is below 7x")

    return {
        "schema_version": 3,
        "validated": True,
        "runs_per_arm": counts.pop(),
        "logical_digest": digest,
        "semantic_configurations": configurations,
        "arm_specs": {arm: asdict(spec) for arm, spec in ARM_SPECS.items()},
        "timing_seconds": timing,
        "performance_gates": {
            "enforced": enforce_performance_gates,
            "legacy_relative_tolerance": LEGACY_RELATIVE_TOLERANCE,
            "v1_max_seconds": V1_MAX_SECONDS,
            "v1_min_speedup": V1_MIN_SPEEDUP,
            "observed": gates,
        },
        "build_runtime": build_runtime,
        "benchmark_provenance": provenance,
    }


def _clean_staging(parent: Path) -> None:
    for path in parent.glob(".matrix-staging-*"):
        shutil.rmtree(path, ignore_errors=True)


def _promote(stage: Path, destination: Path) -> None:
    backup = destination.with_name(".matrix-previous")
    shutil.rmtree(backup, ignore_errors=True)
    if destination.exists():
        destination.replace(backup)
    try:
        stage.replace(destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _run_local(args: argparse.Namespace, stage: Path) -> dict[str, object]:
    benchmark_provenance = _benchmark_provenance()
    preflights = _preflight(args)
    results: dict[str, list[dict[str, object]]] = {arm: [] for arm in ARMS}
    schedule = []
    ordinal = 0
    for repetition in range(args.repetitions):
        offset = repetition % len(ARMS)
        order = ARMS[offset:] + ARMS[:offset]
        for arm in order:
            schedule.append(arm)
            results[arm].append(
                _run_one(
                    arm=arm,
                    ordinal=ordinal,
                    args=args,
                    stage=stage,
                    preflights=preflights,
                    benchmark_provenance=benchmark_provenance,
                )
            )
            ordinal += 1
    shutil.rmtree(stage / ".runtime", ignore_errors=True)
    summary = validate_and_summarize(
        results,
        expected_benchmark_commit=benchmark_provenance["git_commit"],
        expected_run_matrix_sha256=benchmark_provenance["run_matrix_sha256"],
    )
    summary["schedule"] = schedule
    summary["preflight"] = preflights
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    destination = args.artifact_root.expanduser().resolve()
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    _clean_staging(parent)
    stage = parent / f".matrix-staging-{os.getpid()}"
    stage.mkdir()
    try:
        summary = _run_local(args, stage)
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "schema_version": 3,
            "argv": list(sys.argv[1:] if argv is None else argv),
            "summary": "summary.json",
            "benchmark_provenance": summary["benchmark_provenance"],
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        _promote(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
