#!/usr/bin/env bash
# test_p2p.sh — Integration test: two vLLM+LMCache instances connected via
# RemoteController in P2P mode.
#
# Each instance runs on its own GPU. The two LMCache MP servers peer with each
# other via ZMQ so that a cache miss on instance 1 can be satisfied by a hit
# on instance 0 (and vice-versa).
#
# Usage:
#   [MODEL=<hf-model>] [GPU_0=0] [GPU_1=1] ./test_p2p.sh
#
# Prerequisites:
#   pip install lmcache vllm nixl
#   nvidia-smi showing >= 2 GPUs
#
# Ports used (all on 127.0.0.1 by default):
#   5550, 5551  — LMCache ZMQ (worker ↔ server)
#   5200, 5201  — RemoteController ZMQ (server ↔ server P2P)
#   8090, 8091  — LMCache HTTP healthcheck
#   8010, 8011  — vLLM OpenAI-compatible API

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via env vars
# ---------------------------------------------------------------------------
MODEL="${MODEL:-/data0/weishu-model/gpt-oss-20b}"
GPU_0="${GPU_0:-0}"
GPU_1="${GPU_1:-1}"

LMCACHE_ZMQ_PORT_0="${LMCACHE_ZMQ_PORT_0:-5550}"
LMCACHE_ZMQ_PORT_1="${LMCACHE_ZMQ_PORT_1:-5551}"
LMCACHE_HTTP_PORT_0="${LMCACHE_HTTP_PORT_0:-8090}"
LMCACHE_HTTP_PORT_1="${LMCACHE_HTTP_PORT_1:-8091}"

REMOTE_CTRL_PORT_0="${REMOTE_CTRL_PORT_0:-5200}"
REMOTE_CTRL_PORT_1="${REMOTE_CTRL_PORT_1:-5202}"

PROMETHEUS_PORT_0="${PROMETHEUS_PORT_0:-9090}"
PROMETHEUS_PORT_1="${PROMETHEUS_PORT_1:-9092}"

VLLM_PORT_0="${VLLM_PORT_0:-8010}"
VLLM_PORT_1="${VLLM_PORT_1:-8011}"

# L1 size in GB for each LMCache server
L1_SIZE_GB="${L1_SIZE_GB:-8}"

# Bench settings
BENCH_WORKLOAD="${BENCH_WORKLOAD:-long-doc-qa}"
BENCH_TOKENS_PER_GB="${BENCH_TOKENS_PER_GB:-512}"

LOG_DIR="$(dirname "${BASH_SOURCE[0]}")/logs"
mkdir -p "$LOG_DIR"

PIDS=()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cleanup() {
    echo ""
    echo "=== Shutting down all processes ==="
    trap - INT TERM EXIT
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in "${PIDS[@]}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    echo "Done."
}
trap cleanup INT TERM EXIT

check_prerequisites() {
    local missing=0
    for cmd in lmcache vllm curl nvidia-smi; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "ERROR: '$cmd' not found in PATH" >&2
            missing=1
        fi
    done
    for pkg in nixl; do
        python3 -c "import $pkg" 2>/dev/null || {
            echo "ERROR: Python package '$pkg' not installed" >&2
            missing=1
        }
    done
    local num_gpus
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [[ "$num_gpus" -lt 2 ]]; then
        echo "ERROR: Need >= 2 GPUs; found $num_gpus" >&2
        missing=1
    fi
    [[ "$missing" -eq 0 ]] || exit 1
}

# Poll a URL until HTTP 2xx or timeout (seconds).
wait_for_http() {
    local url="$1"
    local timeout="${2:-300}"
    local deadline=$(( SECONDS + timeout ))
    echo "Waiting for $url  (timeout ${timeout}s)..."
    while ! curl -sf "$url" >/dev/null 2>&1; do
        if [[ $SECONDS -ge $deadline ]]; then
            echo "TIMEOUT waiting for $url" >&2
            exit 1
        fi
        sleep 3
    done
    echo "  → $url ready"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

check_prerequisites

echo "=== P2P Remote Controller integration test ==="
echo "  Model:       $MODEL"
echo "  GPU 0:       $GPU_0  (LMCache port $LMCACHE_ZMQ_PORT_0, vLLM port $VLLM_PORT_0)"
echo "  GPU 1:       $GPU_1  (LMCache port $LMCACHE_ZMQ_PORT_1, vLLM port $VLLM_PORT_1)"
echo "  P2P ports:   $REMOTE_CTRL_PORT_0 <-> $REMOTE_CTRL_PORT_1"
echo "  Logs:        $LOG_DIR"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Start LMCache MP server 0
#   • CUDA_VISIBLE_DEVICES=GPU_0 — sees only the same GPU as its vLLM workers
#   • --remote-mode p2p  — enables RemoteController
#   • --remote-peer      — points to server 1's ZMQ control port
# ---------------------------------------------------------------------------
echo "Starting LMCache server 0 (GPU $GPU_0)..."
CUDA_VISIBLE_DEVICES="$GPU_0" \
PYTHONHASHSEED=0 \
lmcache server \
    --host 127.0.0.1 \
    --port "$LMCACHE_ZMQ_PORT_0" \
    --http-host 127.0.0.1 \
    --http-port "$LMCACHE_HTTP_PORT_0" \
    --prometheus-port "$PROMETHEUS_PORT_0" \
    --l1-size-gb "$L1_SIZE_GB" \
    --eviction-policy LRU \
    --remote-mode p2p \
    --remote-serve-port "$REMOTE_CTRL_PORT_0" \
    --remote-peer "peer1:127.0.0.1:$REMOTE_CTRL_PORT_1" \
    >"$LOG_DIR/lmcache0.log" 2>&1 &
PIDS+=($!)

# ---------------------------------------------------------------------------
# Step 2: Start LMCache MP server 1
#   • CUDA_VISIBLE_DEVICES=GPU_1 — sees only the same GPU as its vLLM workers
# ---------------------------------------------------------------------------
echo "Starting LMCache server 1 (GPU $GPU_1)..."
CUDA_VISIBLE_DEVICES="$GPU_1" \
PYTHONHASHSEED=0 \
lmcache server \
    --host 127.0.0.1 \
    --port "$LMCACHE_ZMQ_PORT_1" \
    --http-host 127.0.0.1 \
    --http-port "$LMCACHE_HTTP_PORT_1" \
    --prometheus-port "$PROMETHEUS_PORT_1" \
    --l1-size-gb "$L1_SIZE_GB" \
    --eviction-policy LRU \
    --remote-mode p2p \
    --remote-serve-port "$REMOTE_CTRL_PORT_1" \
    --remote-peer "peer0:127.0.0.1:$REMOTE_CTRL_PORT_0" \
    >"$LOG_DIR/lmcache1.log" 2>&1 &
PIDS+=($!)

wait_for_http "http://127.0.0.1:$LMCACHE_HTTP_PORT_0/api/healthcheck"
wait_for_http "http://127.0.0.1:$LMCACHE_HTTP_PORT_1/api/healthcheck"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Start vLLM instance 0 — connects to LMCache server 0
# ---------------------------------------------------------------------------
echo "Starting vLLM instance 0 (GPU $GPU_0)..."
CUDA_VISIBLE_DEVICES="$GPU_0" \
PYTHONHASHSEED=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "$MODEL" \
    --port "$VLLM_PORT_0" \
    --host 127.0.0.1 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-transfer-config \
    "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$LMCACHE_ZMQ_PORT_0}}" \
    >"$LOG_DIR/vllm0.log" 2>&1 &
PIDS+=($!)

# ---------------------------------------------------------------------------
# Step 4: Start vLLM instance 1 — connects to LMCache server 1
# ---------------------------------------------------------------------------
echo "Starting vLLM instance 1 (GPU $GPU_1)..."
CUDA_VISIBLE_DEVICES="$GPU_1" \
PYTHONHASHSEED=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "$MODEL" \
    --port "$VLLM_PORT_1" \
    --host 127.0.0.1 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-transfer-config \
    "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$LMCACHE_ZMQ_PORT_1}}" \
    >"$LOG_DIR/vllm1.log" 2>&1 &
PIDS+=($!)

wait_for_http "http://127.0.0.1:$VLLM_PORT_0/health" 600
wait_for_http "http://127.0.0.1:$VLLM_PORT_1/health" 600
echo ""

# ---------------------------------------------------------------------------
# Step 5: Warm instance 0 — populate its L1 cache via bench
# ---------------------------------------------------------------------------
echo "=== Phase 1: Warm instance 0 (populates LMCache server 0 L1) ==="
lmcache bench engine \
    --engine-url "http://127.0.0.1:$VLLM_PORT_0" \
    --workload "$BENCH_WORKLOAD" \
    --tokens-per-gb-kvcache "$BENCH_TOKENS_PER_GB" \
    --no-interactive \
    --quiet

echo ""

# ---------------------------------------------------------------------------
# Step 6: Run bench on instance 1 — should get P2P cache hits from server 0
# ---------------------------------------------------------------------------
echo "=== Phase 2: Bench instance 1 (expect P2P hits from server 0) ==="
lmcache bench engine \
    --engine-url "http://127.0.0.1:$VLLM_PORT_1" \
    --workload "$BENCH_WORKLOAD" \
    --tokens-per-gb-kvcache "$BENCH_TOKENS_PER_GB" \
    --no-interactive \
    --quiet

echo ""
echo "=== Test complete. Inspect $LOG_DIR for detailed logs. ==="

# Report LMCache HTTP metrics for both servers
for port in "$LMCACHE_HTTP_PORT_0" "$LMCACHE_HTTP_PORT_1"; do
    echo ""
    echo "--- LMCache server (HTTP $port) status ---"
    curl -sf "http://127.0.0.1:$port/api/status" 2>/dev/null | python3 -m json.tool || true
done
