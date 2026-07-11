# Ray GPU operator fusion benchmark

This directory is the self-contained stock-versus-modified benchmark harness.
It uses stock Ray commit `2741c6461d2bd3e5ff114af67be7a1190453dadd`
from this repository's history and the checked-out commit as modified Ray. It
does not require the earlier candidate checkout, cloud configuration, or an
external benchmark repository. Benchmark Parquet input is intentionally an
argument; this repository does not publish the multi-gigabyte corpus.

## Prerequisites

Use Linux x86_64 with a CUDA 12.2-compatible driver and four visible GPUs.
Building the stock CPython 3.11 wheel uses Ray's pinned `build-wheel.sh`, so
Git, `uv`, Docker/raymake prerequisites, network access, and full Git history
are required. Environment creation requires conda, mamba, or micromamba.

From the Ray repository root:

```bash
suite=release/nightly_tests/dataset/gpu_operator_fusion
"$suite/scripts/build_wheels.sh"
"$suite/scripts/create_envs.sh"
"$suite/scripts/run_benchmark.sh" /absolute/path/to/parquet-input /absolute/path/to/output
```

All generated wheels, the detached stock worktree, and two isolated conda
environments live under `.work/` in this directory and are ignored by Git.
Set `GPU_FUSION_WORK_DIR` or pass a work directory to the first two scripts to
place them elsewhere. The checked-in explicit conda lock and hash-locked pip
runtime requirements are verified against `environment/SHA256SUMS` before
installation.

`run_benchmark.sh` runs three fresh-process arms: stock legacy, modified
legacy with fusion disabled, and modified V1 with fusion enabled. Extra
runner options follow the output argument, for example:

```bash
"$suite/scripts/run_benchmark.sh" /data/parquet /tmp/gpu-fusion-output \
  --repetitions 1 --cluster-cpus 64 --cluster-gpus 4 --gpu-devices 0,1,2,3
```

The default is three repetitions. Untouched stock Ray has a previously
observed intermittent `cudaErrorInvalidValue` during large Arrow-to-cuDF batch
conversion; a stock-arm failure at that location is retained as a failed run,
not retried or attributed to GPU fusion. See `VALIDATION.md` for the preserved
acceptance result and thresholds.
