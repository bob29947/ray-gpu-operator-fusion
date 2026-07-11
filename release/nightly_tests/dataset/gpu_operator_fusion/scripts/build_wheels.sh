#!/usr/bin/env bash
set -euo pipefail

readonly STOCK_COMMIT="2741c6461d2bd3e5ff114af67be7a1190453dadd"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SUITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
RAY_ROOT="$(git -C "$SUITE_DIR" rev-parse --show-toplevel)"
WORK_DIR="${1:-${GPU_FUSION_WORK_DIR:-$SUITE_DIR/.work}}"
mkdir -p "$WORK_DIR"
WORK_DIR="$(cd "$WORK_DIR" && pwd -P)"
STOCK_SOURCE="$WORK_DIR/stock-source"
STOCK_WHEEL_DIR="$WORK_DIR/wheels/stock"
MODIFIED_WHEEL_DIR="$WORK_DIR/wheels/modified"

for command in git python3 sha256sum uv; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command not found: $command" >&2
        exit 1
    fi
done
if [[ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]]; then
    echo "This pinned benchmark environment requires Linux x86_64." >&2
    exit 1
fi
if ! git -C "$RAY_ROOT" cat-file -e "$STOCK_COMMIT^{commit}"; then
    echo "Pinned stock commit is absent. Fetch full repository history first." >&2
    exit 1
fi
if ! git -C "$RAY_ROOT" merge-base --is-ancestor "$STOCK_COMMIT" HEAD; then
    echo "Current branch is not based on the pinned stock commit." >&2
    exit 1
fi
if ! git -C "$RAY_ROOT" diff --quiet HEAD -- python/ray/data || \
        ! git -C "$RAY_ROOT" diff --cached --quiet HEAD -- python/ray/data; then
    echo "Commit or discard Ray Data source changes before building the overlay." >&2
    exit 1
fi

if [[ -e "$STOCK_SOURCE/.git" ]]; then
    actual_stock_commit="$(git -C "$STOCK_SOURCE" rev-parse HEAD)"
    if [[ "$actual_stock_commit" != "$STOCK_COMMIT" ]]; then
        echo "$STOCK_SOURCE is at $actual_stock_commit, expected $STOCK_COMMIT" >&2
        exit 1
    fi
elif [[ -e "$STOCK_SOURCE" ]]; then
    echo "$STOCK_SOURCE exists but is not the pinned Git worktree." >&2
    exit 1
else
    git -C "$RAY_ROOT" worktree add --detach "$STOCK_SOURCE" "$STOCK_COMMIT"
fi
if ! git -C "$STOCK_SOURCE" diff --quiet HEAD || \
        ! git -C "$STOCK_SOURCE" diff --cached --quiet HEAD; then
    echo "Pinned stock worktree has tracked source changes." >&2
    exit 1
fi

mkdir -p "$STOCK_WHEEL_DIR" "$MODIFIED_WHEEL_DIR"
shopt -s nullglob
stock_wheels=("$STOCK_WHEEL_DIR"/ray-*-cp311-cp311-*.whl)
if (( ${#stock_wheels[@]} == 0 )); then
    (
        cd "$STOCK_SOURCE"
        ./build-wheel.sh 3.11 "$STOCK_WHEEL_DIR" --no-java
    )
    stock_wheels=("$STOCK_WHEEL_DIR"/ray-*-cp311-cp311-*.whl)
fi
if (( ${#stock_wheels[@]} != 1 )); then
    echo "Expected exactly one stock CPython 3.11 Ray wheel in $STOCK_WHEEL_DIR." >&2
    exit 1
fi
actual_wheel_commit="$(python3 -c \
    'import re, sys, zipfile; data=zipfile.ZipFile(sys.argv[1]).read("ray/_version.py").decode(); match=re.search(r"^commit = \"([0-9a-f]{40})\"$", data, re.MULTILINE); print(match.group(1) if match else "")' \
    "${stock_wheels[0]}")"
if [[ "$actual_wheel_commit" != "$STOCK_COMMIT" ]]; then
    echo "Stock wheel reports $actual_wheel_commit, expected $STOCK_COMMIT." >&2
    exit 1
fi

python3 "$SCRIPT_DIR/build_modified_wheel.py" \
    --repo "$RAY_ROOT" \
    --base-commit "$STOCK_COMMIT" \
    --stock-wheel "${stock_wheels[0]}" \
    --output-dir "$MODIFIED_WHEEL_DIR"
modified_wheels=("$MODIFIED_WHEEL_DIR"/ray-*-cp311-cp311-*.whl)
if (( ${#modified_wheels[@]} != 1 )); then
    echo "Expected exactly one modified CPython 3.11 Ray wheel." >&2
    exit 1
fi
(
    cd "$WORK_DIR/wheels"
    sha256sum "stock/${stock_wheels[0]##*/}" "modified/${modified_wheels[0]##*/}" > SHA256SUMS
)
echo "Wheels and SHA-256 manifest are under $WORK_DIR/wheels"
