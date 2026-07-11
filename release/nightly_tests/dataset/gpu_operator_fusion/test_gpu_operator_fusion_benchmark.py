from __future__ import annotations

import argparse
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pipeline_compat
import pipeline_common
import pipeline_map_group_partitions
import pipeline_map_groups
import run_matrix
import workload


BENCHMARK_COMMIT = "b" * 40
RUN_MATRIX_SHA256 = "f" * 64


def _runtime(build: str) -> dict[str, object]:
    identity = {
        "stock": ("s" * 40, "1" * 64),
        "modified": ("m" * 40, "2" * 64),
    }
    commit, wheel = identity[build]
    return {
        "python": f"/env/{build}/bin/python",
        "python_implementation": "CPython",
        "python_version": "3.11.15 (benchmark build)",
        "python_cache_tag": "cpython-311",
        "sys_prefix": f"/env/{build}",
        "ray_file": f"/env/{build}/lib/python3.11/site-packages/ray/__init__.py",
        "ray_commit": commit,
        "ray_wheel_sha256": wheel,
        "cuda_visible_devices": "0,1,2,3",
    }


def _result(arm: str, elapsed: float) -> dict[str, object]:
    spec = run_matrix.ARM_SPECS[arm]
    legacy = spec.execution_api == "map_groups"
    return {
        "matrix_arm": arm,
        "semantic_configuration": {
            "input": "same",
            "actors": 4,
            "batch_size": spec.batch_size,
        },
        "source_sha256": {
            name: f"{index:x}" * 64
            for index, name in enumerate(
                (
                    "pipeline_common.py",
                    "pipeline_compat.py",
                    "pipeline_map_groups.py",
                    "pipeline_map_group_partitions.py",
                    "run_matrix.py",
                    "workload.py",
                )
            )
        },
        "benchmark_provenance": {
            "git_commit": BENCHMARK_COMMIT,
            "run_matrix_sha256": RUN_MATRIX_SHA256,
        },
        "dependencies": {"numpy": "same"},
        "input_identity": [{"path": "input.parquet", "bytes": 10}],
        "logical_digest": {
            "algorithm": "sha256-sorted-row-sha256-v1",
            "dtype": "uint16",
            "rows": run_matrix.EXPECTED_OUTPUT_ROWS,
            "sha256": run_matrix.EXPECTED_OUTPUT_SHA256,
            "shape": [4096],
        },
        "execution_api": spec.execution_api,
        "map_group_partitions_api_supported": spec.api_supported,
        "data_context": {
            "use_datasource_v2": False,
            "fusion_flag_supported": spec.fusion_flag_supported,
            "enable_gpu_operator_fusion": spec.fusion_enabled,
            "udf_properties_supported": spec.udf_properties_supported,
            "udf_properties_applied": (
                spec.execution_api == "map_group_partitions"
                and spec.udf_properties_supported
            ),
            "map_group_partitions_api_supported": spec.api_supported,
        },
        "pipeline_elapsed_s": elapsed,
        "plan": {
            "optimizer_operator_count": 0,
            "gpu_operator_count": int(arm == "modified_v1"),
            "gpu_operator_names": ["gpu[all]"] if arm == "modified_v1" else [],
            "exchange_operator_names": ["HashShuffle"] if legacy else [],
        },
        "runtime": _runtime(spec.build),
    }


def _matrix_results() -> dict[str, list[dict[str, object]]]:
    elapsed = {
        "stock_legacy": 100.0,
        "modified_legacy": 102.0,
        "modified_v1": 10.5,
    }
    return {arm: [_result(arm, elapsed[arm])] for arm in run_matrix.ARMS}


class GPUOperatorFusionBenchmarkTest(unittest.TestCase):
    def test_plan_evidence_recognizes_wrapped_fused_gpu_operator(self):
        class Operator:
            operator_name = "ActorPoolMapOperator[GPU[Read,Map,Groups]]"

        class Summary:
            operators_stats = [Operator()]
            parents = []

        class Dataset:
            @staticmethod
            def get_stats_summary():
                return Summary()

        evidence = pipeline_common.plan_evidence(Dataset(), "")
        self.assertEqual(evidence["gpu_operator_count"], 1)
        self.assertEqual(evidence["optimizer_operator_count"], 0)

    def test_shared_legacy_pipeline_has_no_new_api_keywords(self):
        tokenized = mock.Mock()
        grouped = tokenized.groupby.return_value
        grouped.map_groups.return_value = "legacy"
        args = mock.Mock(
            num_partitions=8,
            actors=4,
            num_cpus_per_worker=1,
            num_gpus_per_worker=1,
        )
        with mock.patch.object(
            pipeline_map_groups, "build_tokenized", return_value=tokenized
        ):
            result = pipeline_map_groups.build_pipeline(args)

        tokenized.groupby.assert_called_once_with(
            list(workload.GROUP_KEYS), num_partitions=8
        )
        grouped.map_groups.assert_called_once()
        self.assertEqual(result, "legacy")

    def test_shared_new_api_pipeline_uses_narrow_compatibility_kwargs(self):
        tokenized = mock.Mock()
        grouped = tokenized.groupby.return_value
        grouped.map_group_partitions.return_value = "new-api"
        properties = object()
        args = mock.Mock(num_partitions=8)
        with (
            mock.patch.object(
                pipeline_map_group_partitions,
                "partition_preservation_kwargs",
                return_value={"udf_properties": properties},
            ),
            mock.patch.object(
                pipeline_map_group_partitions,
                "build_tokenized",
                return_value=tokenized,
            ) as build_tokenized,
        ):
            result = pipeline_map_group_partitions.build_pipeline(args)

        build_tokenized.assert_called_once_with(
            args, map_batches_compat_kwargs={"udf_properties": properties}
        )
        grouped.map_group_partitions.assert_called_once_with(
            workload.build_sequences_partition_gpu
        )
        self.assertEqual(result, "new-api")

    def test_modified_compatibility_sets_flag_and_udf_property(self):
        context = SimpleNamespace(
            use_datasource_v2=True,
            enable_gpu_operator_fusion=False,
        )

        class DataContext:
            @staticmethod
            def get_current():
                return context

        class UDFProperties:
            def __init__(self, *, preserves_partitioning):
                self.preserves_partitioning = preserves_partitioning

        class Dataset:
            def map_batches(self, *, udf_properties=None):
                pass

        class GroupedData:
            def map_group_partitions(self, fn):
                pass

        ray_data = SimpleNamespace(UDFProperties=UDFProperties)
        api = (ray_data, DataContext, Dataset, GroupedData)
        with mock.patch.object(pipeline_compat, "_load_ray_data_api", return_value=api):
            evidence = pipeline_compat.configure_data_context(
                enable_gpu_operator_fusion=True
            )
            kwargs = pipeline_compat.partition_preservation_kwargs()

        self.assertFalse(evidence["use_datasource_v2"])
        self.assertTrue(evidence["fusion_flag_supported"])
        self.assertTrue(evidence["enable_gpu_operator_fusion"])
        self.assertTrue(evidence["udf_properties_supported"])
        self.assertTrue(kwargs["udf_properties"].preserves_partitioning)

    def test_stock_compatibility_leaves_absent_extensions_absent(self):
        context = SimpleNamespace(use_datasource_v2=True)

        class DataContext:
            @staticmethod
            def get_current():
                return context

        class Dataset:
            def map_batches(self):
                pass

        class GroupedData:
            pass

        api = (SimpleNamespace(), DataContext, Dataset, GroupedData)
        with mock.patch.object(pipeline_compat, "_load_ray_data_api", return_value=api):
            evidence = pipeline_compat.configure_data_context(
                enable_gpu_operator_fusion=True
            )
            kwargs = pipeline_compat.partition_preservation_kwargs()

        self.assertFalse(evidence["use_datasource_v2"])
        self.assertFalse(evidence["fusion_flag_supported"])
        self.assertIsNone(evidence["enable_gpu_operator_fusion"])
        self.assertFalse(evidence["udf_properties_supported"])
        self.assertEqual(kwargs, {})

    def test_matrix_parser_uses_two_builds_and_tuned_arm_defaults(self):
        args = run_matrix.parse_args(
            [
                "--stock-python",
                "/env/stock/bin/python",
                "--modified-python",
                "/env/modified/bin/python",
                "--stock-commit",
                "1" * 40,
                "--modified-commit",
                "2" * 40,
                "--input",
                "/input",
                "--output-root",
                "/output",
            ]
        )

        self.assertEqual(args.repetitions, 3)
        self.assertEqual(
            run_matrix.ARMS,
            ("stock_legacy", "modified_legacy", "modified_v1"),
        )
        self.assertEqual(run_matrix.BUILDS, ("stock", "modified"))
        self.assertEqual(run_matrix.ARM_SPECS["stock_legacy"].batch_size, 1_048_576)
        self.assertEqual(run_matrix.ARM_SPECS["modified_v1"].batch_size, 2_097_152)

    def test_matrix_preflight_pins_datasource_v2_and_capabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            interpreters = {}
            for build in run_matrix.BUILDS:
                interpreter = root / build / "bin" / "python"
                interpreter.parent.mkdir(parents=True)
                interpreter.touch()
                interpreters[build] = interpreter.absolute()

            capabilities = {
                "stock": (False, False, False),
                "modified": (True, True, True),
            }
            commits = {
                "stock": "1" * 40,
                "modified": "2" * 40,
            }

            def probe(interpreter):
                build = interpreter.parts[-3]
                api, flag, properties = capabilities[build]
                prefix = root / build
                return {
                    "python": str(interpreters[build]),
                    "python_implementation": "CPython",
                    "python_version": "3.11.15",
                    "python_cache_tag": "cpython-311",
                    "sys_prefix": str(prefix),
                    "ray_file": str(prefix / "site-packages" / "ray" / "__init__.py"),
                    "ray_commit": commits[build],
                    "ray_wheel_sha256": str(1 + list(run_matrix.BUILDS).index(build))
                    * 64,
                    "use_datasource_v2": False,
                    "map_group_partitions_api_supported": api,
                    "fusion_flag_supported": flag,
                    "udf_properties_exported": properties,
                    "map_batches_udf_properties_supported": properties,
                }

            args = argparse.Namespace(
                **{
                    f"{build}_python": interpreters[build]
                    for build in run_matrix.BUILDS
                },
                **{
                    f"{build}_commit": commits[build]
                    for build in run_matrix.BUILDS
                },
            )
            with mock.patch.object(run_matrix, "_probe_python", side_effect=probe):
                evidence = run_matrix._preflight(args)

        self.assertFalse(evidence["stock"]["use_datasource_v2"])
        self.assertTrue(evidence["modified"]["fusion_flag_supported"])
        self.assertTrue(evidence["modified"]["udf_properties_exported"])

    def test_pipeline_commands_reuse_two_shared_entrypoints(self):
        args = argparse.Namespace(
            input="/input",
            actors=4,
            num_partitions=8,
            cluster_cpus=64,
            cluster_gpus=4,
            compression="zstd",
            compression_level=1,
        )
        provenance = {
            "git_commit": BENCHMARK_COMMIT,
            "run_matrix_sha256": RUN_MATRIX_SHA256,
        }
        preflight = {"ray_commit": "m" * 40, "ray_wheel_sha256": "2" * 64}
        commands = {}
        for arm in run_matrix.ARMS:
            commands[arm] = run_matrix._pipeline_command(
                arm=arm,
                python=Path(f"/env/{run_matrix.ARM_SPECS[arm].build}/bin/python"),
                args=args,
                result_path=Path("result.json"),
                output="/output",
                label=arm,
                ordinal=0,
                preflight=preflight,
                benchmark_provenance=provenance,
                ray_temp_dir=Path("/tmp/ray"),
                write_complete_marker=Path("/tmp/complete"),
            )

        self.assertIn("pipeline_map_groups.py", commands["stock_legacy"][1])
        self.assertEqual(
            commands["stock_legacy"][1], commands["modified_legacy"][1]
        )
        self.assertIn(
            "pipeline_map_group_partitions.py",
            commands["modified_v1"][1],
        )
        modified = commands["modified_v1"]
        self.assertEqual(
            modified[modified.index("--expected-map-group-partitions-api") + 1],
            "true",
        )
        self.assertIn("--run-matrix-sha256", modified)

    def test_matrix_validation_accepts_all_three_arms(self):
        summary = run_matrix.validate_and_summarize(
            _matrix_results(),
            expected_benchmark_commit=BENCHMARK_COMMIT,
            expected_run_matrix_sha256=RUN_MATRIX_SHA256,
        )

        self.assertTrue(summary["validated"])
        self.assertEqual(summary["runs_per_arm"], 1)
        observed = summary["performance_gates"]["observed"]
        self.assertGreater(observed["modified_v1_speedup_over_stock"], 7)

    def test_matrix_validation_rejects_modified_legacy_fusion(self):
        results = _matrix_results()
        results["modified_legacy"][0]["data_context"][
            "enable_gpu_operator_fusion"
        ] = True

        with self.assertRaisesRegex(AssertionError, "data context"):
            run_matrix.validate_and_summarize(results)

    def test_matrix_validation_requires_one_fused_v1_node(self):
        results = _matrix_results()
        results["modified_v1"][0]["plan"]["gpu_operator_count"] = 0

        with self.assertRaisesRegex(AssertionError, "one fused GPU node"):
            run_matrix.validate_and_summarize(results)

    def test_matrix_validation_requires_canonical_output_digest(self):
        results = _matrix_results()
        for arm in run_matrix.ARMS:
            results[arm][0]["logical_digest"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(AssertionError, "canonical digest"):
            run_matrix.validate_and_summarize(results)

    def test_matrix_validation_requires_stock_hash_shuffle(self):
        results = _matrix_results()
        results["stock_legacy"][0]["plan"]["exchange_operator_names"] = []

        with self.assertRaisesRegex(AssertionError, "stock hash shuffle"):
            run_matrix.validate_and_summarize(results)

    def test_matrix_validation_enforces_performance_gates(self):
        results = _matrix_results()
        results["modified_v1"][0]["pipeline_elapsed_s"] = 20.0

        with self.assertRaisesRegex(AssertionError, "exceeds 17.15 seconds"):
            run_matrix.validate_and_summarize(results)

        summary = run_matrix.validate_and_summarize(
            results, enforce_performance_gates=False
        )
        self.assertFalse(summary["performance_gates"]["enforced"])

    def test_matrix_validation_enforces_legacy_parity(self):
        results = _matrix_results()
        results["modified_legacy"][0]["pipeline_elapsed_s"] = 120.0

        with self.assertRaisesRegex(AssertionError, "legacy performance"):
            run_matrix.validate_and_summarize(results)

    def test_matrix_validation_enforces_v1_speedup(self):
        results = _matrix_results()
        results["modified_v1"][0]["pipeline_elapsed_s"] = 15.0

        with self.assertRaisesRegex(AssertionError, "speedup over stock"):
            run_matrix.validate_and_summarize(results)

    def test_matrix_performance_gate_accepts_a_v1_improvement(self):
        results = _matrix_results()
        results["modified_v1"][0]["pipeline_elapsed_s"] = 5.0

        summary = run_matrix.validate_and_summarize(results)

        self.assertGreater(
            summary["performance_gates"]["observed"][
                "modified_v1_speedup_over_stock"
            ],
            7,
        )

    def test_partition_preservation_signature_is_the_only_compat_keyword(self):
        signature = inspect.signature(
            pipeline_map_group_partitions.build_pipeline
        )
        self.assertEqual(list(signature.parameters), ["args"])
        self.assertEqual(
            set(pipeline_compat.partition_preservation_kwargs.__annotations__),
            {"return"},
        )


if __name__ == "__main__":
    unittest.main()
