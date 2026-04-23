#!/usr/bin/env bash
# test_cxl.sh — Integration test: two vLLM+LMCache instances sharing a CXL
# memory region.  KV cache written by instance 0 is read by instance 1 via
# direct shared-memory access (no RDMA, no GPU-to-GPU copy).
#
# The CXL shared region is simulated using a regular file in /dev/shm that
# both servers mmap(MAP_SHARED).  On real hardware replace CXL_DAX_DEVICE
# with the actual /dev/dax*.* path shared between the two hosts.
#
# Usage:
#   [MODEL=<hf-model>] [GPU_0=0] [GPU_1=1] ./test_cxl.sh
#
# Prerequisites:
#   pip install lmcache vllm
#   nvidia-smi showing >= 2 GPUs
#   ~4 GiB free in /dev/shm (or set CXL_DAX_DEVICE to another path)
#
# Ports used (all on 127.0.0.1 by default):
#   5550, 5551  — LMCache ZMQ (worker ↔ server)
#   5300, 5301  — CxlRemoteController REP + PULL (server 0)
#   5302, 5303  — CxlRemoteController REP + PULL (server 1)
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

# CXL controller ports: server 0 REP=5300/PULL=5301, server 1 REP=5302/PULL=5303
CXL_SERVE_PORT_0="${CXL_SERVE_PORT_0:-5300}"
CXL_SERVE_PORT_1="${CXL_SERVE_PORT_1:-5302}"

PROMETHEUS_PORT_0="${PROMETHEUS_PORT_0:-9090}"
PROMETHEUS_PORT_1="${PROMETHEUS_PORT_1:-9092}"

VLLM_PORT_0="${VLLM_PORT_0:-8010}"
VLLM_PORT_1="${VLLM_PORT_1:-8011}"

# Simulated CXL region: 4 GiB file in /dev/shm split into two 2 GiB sub-regions.
# Replace CXL_DAX_DEVICE with /dev/dax0.0 etc. for real hardware.
CXL_DAX_DEVICE="${CXL_DAX_DEVICE:-/dev/shm/cxl_test_region.bin}"
CXL_REGION_SIZE_GB="${CXL_REGION_SIZE_GB:-4}"
CXL_SUBREGION_SIZE_GB="${CXL_SUBREGION_SIZE_GB:-2}"

# L1 DRAM budget per server (KV data lands in CXL, so this can be small)
L1_SIZE_GB="${L1_SIZE_GB:-2}"

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
    for cmd in lmcache vllm curl nvidia-smi truncate; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "ERROR: '$cmd' not found in PATH" >&2
            missing=1
        fi
    done
    local num_gpus
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [[ "$num_gpus" -lt 2 ]]; then
        echo "ERROR: Need >= 2 GPUs; found $num_gpus" >&2
        missing=1
    fi
    # Verify /dev/shm has enough space when using simulated region
    if [[ "$CXL_DAX_DEVICE" == /dev/shm/* ]]; then
        local shm_avail_kb
        shm_avail_kb=$(df --output=avail /dev/shm | tail -1)
        local need_kb=$(( CXL_REGION_SIZE_GB * 1024 * 1024 ))
        if [[ "$shm_avail_kb" -lt "$need_kb" ]]; then
            echo "ERROR: /dev/shm has ${shm_avail_kb} KiB free; need ${need_kb} KiB" >&2
            missing=1
        fi
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
    echo "  -> $url ready"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

check_prerequisites

echo "=== CXL shared-memory integration test ==="
echo "  Model:         $MODEL"
echo "  GPU 0:         $GPU_0  (LMCache port $LMCACHE_ZMQ_PORT_0, vLLM port $VLLM_PORT_0)"
echo "  GPU 1:         $GPU_1  (LMCache port $LMCACHE_ZMQ_PORT_1, vLLM port $VLLM_PORT_1)"
echo "  CXL device:    $CXL_DAX_DEVICE (${CXL_REGION_SIZE_GB} GiB)"
echo "  CXL ports:     $CXL_SERVE_PORT_0 (server 0) <-> $CXL_SERVE_PORT_1 (server 1)"
echo "  Logs:          $LOG_DIR"
echo ""

# ---------------------------------------------------------------------------
# Step 0: Create / resize the simulated CXL shared region.
#   On real CXL hardware (or a persistent DAX mount) skip this step.
# ---------------------------------------------------------------------------
if [[ "$CXL_DAX_DEVICE" == /dev/shm/* ]]; then
    echo "Creating simulated CXL region: $CXL_DAX_DEVICE (${CXL_REGION_SIZE_GB} GiB)..."
    truncate -s "${CXL_REGION_SIZE_GB}G" "$CXL_DAX_DEVICE"
    echo "  -> $CXL_DAX_DEVICE ready"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 1: Start LMCache server 0
#   Sub-region: [0, CXL_SUBREGION_SIZE_GB)
#   Peer:       server 1 at CXL_SERVE_PORT_1
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
    --cxl-dax-device "$CXL_DAX_DEVICE" \
    --cxl-region-size-gb "$CXL_REGION_SIZE_GB" \
    --cxl-subregion-offset-gb 0 \
    --cxl-subregion-size-gb "$CXL_SUBREGION_SIZE_GB" \
    --cxl-serve-port "$CXL_SERVE_PORT_0" \
    --cxl-peer "peer1:127.0.0.1:$CXL_SERVE_PORT_1" \
    --cxl-tiering-policy always_cxl \
    >"$LOG_DIR/lmcache0.log" 2>&1 &
PIDS+=($!)

# ---------------------------------------------------------------------------
# Step 2: Start LMCache server 1
#   Sub-region: [CXL_SUBREGION_SIZE_GB, CXL_REGION_SIZE_GB)
#   Peer:       server 0 at CXL_SERVE_PORT_0
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
    --cxl-dax-device "$CXL_DAX_DEVICE" \
    --cxl-region-size-gb "$CXL_REGION_SIZE_GB" \
    --cxl-subregion-offset-gb "$CXL_SUBREGION_SIZE_GB" \
    --cxl-subregion-size-gb "$CXL_SUBREGION_SIZE_GB" \
    --cxl-serve-port "$CXL_SERVE_PORT_1" \
    --cxl-peer "peer0:127.0.0.1:$CXL_SERVE_PORT_0" \
    --cxl-tiering-policy always_cxl \
    >"$LOG_DIR/lmcache1.log" 2>&1 &
PIDS+=($!)

wait_for_http "http://127.0.0.1:$LMCACHE_HTTP_PORT_0/api/healthcheck"
wait_for_http "http://127.0.0.1:$LMCACHE_HTTP_PORT_1/api/healthcheck"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Start vLLM instance 0
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
# Step 4: Start vLLM instance 1
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
# Step 5: Warm instance 0 — writes KV data into server 0's CXL sub-region
# ---------------------------------------------------------------------------
echo "=== Phase 1: Warm instance 0 (populates CXL sub-region 0) ==="
lmcache bench engine \
    --engine-url "http://127.0.0.1:$VLLM_PORT_0" \
    --workload "$BENCH_WORKLOAD" \
    --tokens-per-gb-kvcache "$BENCH_TOKENS_PER_GB" \
    --no-interactive \
    --quiet

echo ""

# ---------------------------------------------------------------------------
# Step 6: Bench instance 1 — should satisfy misses via CXL cross-server lookup
#   Server 1's CxlRemoteL2Adapter queries server 0's CxlRemoteController.
#   Server 0 returns byte offsets; server 1 reads directly from the shared mmap.
# ---------------------------------------------------------------------------
echo "=== Phase 2: Bench instance 1 (expect CXL cross-server hits from server 0) ==="
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
