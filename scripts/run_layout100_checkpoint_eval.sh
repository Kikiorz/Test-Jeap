#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: $0 RUN_ID CONFIG_NAME CHECKPOINT_STEP_DIR" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$1"
CONFIG_NAME="$2"
CHECKPOINT_DIR="$3"
PLUS_ROOT="${LIBERO_PLUS_ROOT:-/workspace/Test-Jeap/third_party/LIBERO-plus}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/artifacts/evals/layout100/$RUN_ID}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT/eval_manifests/libero_plus_layout100}"
BASE_PORT="${BASE_PORT:-8100}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
SUITES=(libero_spatial libero_object libero_goal libero_10)
SERVER_PIDS=()
EVAL_PIDS=()

mkdir -p "$RESULTS_ROOT/logs"

cleanup() {
    local pid
    for pid in "${EVAL_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${SERVER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${EVAL_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    for pid in "${SERVER_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# Refuse to connect to a server left behind by another run. Without this
# preflight, a failed bind can be hidden by the existing listener and the
# evaluation may silently query the wrong checkpoint.
for gpu in 0 1 2 3; do
    port=$((BASE_PORT + gpu))
    if (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; then
        echo "Port $port is already occupied; refusing to start an ambiguous evaluation." >&2
        exit 1
    fi
done

for gpu in 0 1 2 3; do
    port=$((BASE_PORT + gpu))
    CONFIG_NAME="$CONFIG_NAME" \
    CHECKPOINT_DIR="$CHECKPOINT_DIR" \
    PORT="$port" \
    GPU_IDS="$gpu" \
    XLA_MEMORY_FRACTION=0.80 \
        "$ROOT/scripts/run_libero_policy_server.sh" start \
        >"$RESULTS_ROOT/logs/server-gpu${gpu}.log" 2>&1 &
    SERVER_PIDS+=("$!")
done

for gpu in 0 1 2 3; do
    port=$((BASE_PORT + gpu))
    ready=0
    for _ in $(seq 1 180); do
        if (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "${SERVER_PIDS[$gpu]}" 2>/dev/null; then
            echo "Policy server $gpu exited before becoming ready" >&2
            tail -100 "$RESULTS_ROOT/logs/server-gpu${gpu}.log" >&2 || true
            exit 1
        fi
        sleep 2
    done
    if [[ "$ready" != 1 ]]; then
        echo "Timed out waiting for policy server $gpu on port $port" >&2
        exit 1
    fi
done

for gpu in 0 1 2 3; do
    suite="${SUITES[$gpu]}"
    port=$((BASE_PORT + gpu))
    manifest="$MANIFEST_ROOT/${suite}.json"
    RUN_ID="$RUN_ID" \
    HOST=127.0.0.1 \
    PORT="$port" \
    TASK_SUITE="$suite" \
    TASK_IDS_PATH="$manifest" \
    RESULTS_PATH="$RESULTS_ROOT/${suite}.jsonl" \
    VIDEO_ROOT="$RESULTS_ROOT/videos/$suite" \
    REPLAN_STEPS="$REPLAN_STEPS" \
    SAVE_VIDEO=0 \
    RESUME=1 \
    EVAL_GPU="$gpu" \
    LIBERO_PLUS_ROOT="$PLUS_ROOT" \
    LIBERO_CONFIG_PATH="$RESULTS_ROOT/libero-config-$gpu" \
        "$ROOT/scripts/run_libero_evaluation.sh" plus \
        >"$RESULTS_ROOT/logs/eval-${suite}.log" 2>&1 &
    EVAL_PIDS+=("$!")
done

status=0
for pid in "${EVAL_PIDS[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [[ "$status" != 0 ]]; then
    echo "At least one Layout100 evaluation worker failed; see $RESULTS_ROOT/logs" >&2
    exit "$status"
fi

echo "Layout100 evaluation complete: $RESULTS_ROOT"
