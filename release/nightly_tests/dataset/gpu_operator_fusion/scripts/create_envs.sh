#!/usr/bin/env bash
set -euo pipefail

readonly STOCK_COMMIT="2741c6461d2bd3e5ff114af67be7a1190453dadd"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SUITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
RAY_ROOT="$(git -C "$SUITE_DIR" rev-parse --show-toplevel)"
MODIFIED_COMMIT="$(git -C "$RAY_ROOT" rev-parse HEAD)"
WORK_DIR="${1:-${GPU_FUSION_WORK_DIR:-$SUITE_DIR/.work}}"
mkdir -p "$WORK_DIR"
WORK_DIR="$(cd "$WORK_DIR" && pwd -P)"
LOCK_DIR="$SUITE_DIR/environment"
ENV_DIR="$WORK_DIR/envs"
WHEEL_DIR="$WORK_DIR/wheels"

if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    conda_command=("$CONDA_EXE")
elif command -v micromamba >/dev/null 2>&1; then
    conda_command=(micromamba)
elif command -v mamba >/dev/null 2>&1; then
    conda_command=(mamba)
elif command -v conda >/dev/null 2>&1; then
    conda_command=(conda)
else
    echo "Install conda, mamba, or micromamba before creating environments." >&2
    exit 1
fi
(
    cd "$LOCK_DIR"
    sha256sum -c SHA256SUMS
)
if [[ ! -f "$WHEEL_DIR/SHA256SUMS" ]]; then
    echo "Run scripts/build_wheels.sh first; wheel SHA-256 manifest is missing." >&2
    exit 1
fi
(
    cd "$WHEEL_DIR"
    sha256sum -c SHA256SUMS
)

shopt -s nullglob
stock_wheels=("$WHEEL_DIR/stock"/ray-*-cp311-cp311-*.whl)
modified_wheels=("$WHEEL_DIR/modified"/ray-*-cp311-cp311-*.whl)
if (( ${#stock_wheels[@]} != 1 || ${#modified_wheels[@]} != 1 )); then
    echo "Run scripts/build_wheels.sh first; exactly one wheel per arm is required." >&2
    exit 1
fi

mkdir -p "$ENV_DIR"
create_base_environment() {
    local prefix="$1"
    if [[ -e "$prefix/conda-meta/history" ]]; then
        return
    fi
    if [[ -e "$prefix" ]]; then
        echo "$prefix exists but is not a complete conda environment." >&2
        exit 1
    fi
    "${conda_command[@]}" create --yes --prefix "$prefix" \
        --file "$LOCK_DIR/rapids-25.12-linux-64.explicit.txt"
}

install_arm() {
    local arm="$1"
    local expected_commit="$2"
    local wheel="$3"
    local prefix="$ENV_DIR/$arm"
    create_base_environment "$prefix"
    "$prefix/bin/python" -m pip install --no-deps --require-hashes \
        -r "$LOCK_DIR/ray-runtime-linux-py311.lock"
    "$prefix/bin/python" -m pip install --no-deps --force-reinstall "$wheel"
    "$prefix/bin/python" -c \
        'import sys, ray, cudf; expected=sys.argv[1]; assert sys.version_info[:2] == (3, 11); assert ray.__commit__ == expected, (ray.__commit__, expected); print(ray.__version__, ray.__commit__, cudf.__version__)' \
        "$expected_commit"
}

install_arm stock "$STOCK_COMMIT" "${stock_wheels[0]}"
install_arm modified "$MODIFIED_COMMIT" "${modified_wheels[0]}"
echo "Isolated benchmark interpreters are under $ENV_DIR"
