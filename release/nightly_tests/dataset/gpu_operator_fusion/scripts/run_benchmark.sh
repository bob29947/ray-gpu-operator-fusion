#!/usr/bin/env bash
set -euo pipefail

readonly STOCK_COMMIT="2741c6461d2bd3e5ff114af67be7a1190453dadd"
if (( $# < 1 )); then
    echo "Usage: $0 INPUT_PATH_OR_URI [OUTPUT_ROOT] [run_matrix.py options]" >&2
    exit 2
fi
INPUT="$1"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SUITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
RAY_ROOT="$(git -C "$SUITE_DIR" rev-parse --show-toplevel)"
MODIFIED_COMMIT="$(git -C "$RAY_ROOT" rev-parse HEAD)"
WORK_DIR="${GPU_FUSION_WORK_DIR:-$SUITE_DIR/.work}"
OUTPUT_ROOT="$SUITE_DIR/outputs"
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    OUTPUT_ROOT="$1"
    shift
fi
STOCK_PYTHON="$WORK_DIR/envs/stock/bin/python"
MODIFIED_PYTHON="$WORK_DIR/envs/modified/bin/python"
if [[ ! -x "$STOCK_PYTHON" || ! -x "$MODIFIED_PYTHON" ]]; then
    echo "Run scripts/create_envs.sh first." >&2
    exit 1
fi
exec env -u PYTHONPATH -u PYTHONHOME -u RAY_ADDRESS \
    "$MODIFIED_PYTHON" "$SUITE_DIR/run_matrix.py" \
    --stock-python "$STOCK_PYTHON" \
    --modified-python "$MODIFIED_PYTHON" \
    --stock-commit "$STOCK_COMMIT" \
    --modified-commit "$MODIFIED_COMMIT" \
    --input "$INPUT" \
    --output-root "$OUTPUT_ROOT" \
    "$@"
