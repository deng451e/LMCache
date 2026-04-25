#!/usr/bin/env bash
# test_cxl_cross.sh — Two hosts (c1 + c2) sharing the SAME /dev/dax0.0
# CXL region. Each host runs one LMCache MP server (using a distinct
# sub-region of the shared DAX) + one vLLM instance. KV cache written by
# one side is read by the other via direct shared-memory access; the
# CxlRemoteController coordinates byte-offset lookups over ZMQ.
#
# Usage: ./test_cxl_cross.sh <role 0|1> <peer_ip>
# Env:   MODEL, L1_SIZE_GB, CXL_SUBREGION_SIZE_GB

set -euo pipefail
ROLE="${1:?usage: $0 <0|1> <peer_ip>}"
PEER_IP="${2:?peer_ip required}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MY_IP="$(hostname -I | awk '{print $1}')"
L1_SIZE_GB="${L1_SIZE_GB:-8}"

# Shared CXL config — MUST match on both hosts
CXL_DAX_DEVICE="${CXL_DAX_DEVICE:-/dev/dax0.0}"
CXL_REGION_SIZE_GB="${CXL_REGION_SIZE_GB:-32}"
CXL_SUBREGION_SIZE_GB="${CXL_SUBREGION_SIZE_GB:-16}"

# Ports (same on both hosts — each binds its own)
LMC_ZMQ=5550
LMC_HTTP=8090
CXL_SERVE=5300
PROM=9090
VLLM=8010

# Sub-region offset differs by role
SUB_OFFSET=$(( ROLE * CXL_SUBREGION_SIZE_GB ))
PEER_ROLE=$(( 1 - ROLE ))

LOG_DIR="$(dirname "${BASH_SOURCE[0]}")/logs-cxl${ROLE}"
mkdir -p "$LOG_DIR"

PIDS=()
cleanup() {
    trap - INT TERM EXIT
    for p in "${PIDS[@]}"; do kill $p 2>/dev/null || true; done
    sleep 2
    for p in "${PIDS[@]}"; do kill -9 $p 2>/dev/null || true; done
}
trap cleanup INT TERM EXIT

wait_for_http() {
    local url="$1"; local timeout="${2:-300}"; local deadline=$(( SECONDS + timeout ))
    echo "Waiting for $url (timeout ${timeout}s)..."
    while ! curl -sf "$url" >/dev/null 2>&1; do
        [[ $SECONDS -ge $deadline ]] && { echo "TIMEOUT $url" >&2; exit 1; }
        sleep 3
    done
    echo "  -> $url ready"
}

echo "=== Cross-host CXL test host$ROLE ==="
echo "  MY_IP        : $MY_IP"
echo "  PEER_IP      : $PEER_IP"
echo "  MODEL        : $MODEL"
echo "  CXL device   : $CXL_DAX_DEVICE (${CXL_REGION_SIZE_GB} GiB total)"
echo "  Sub-region   : offset ${SUB_OFFSET} GiB, size ${CXL_SUBREGION_SIZE_GB} GiB"
echo "  CXL peer     : peer${PEER_ROLE} at ${PEER_IP}:${CXL_SERVE}"
echo "  LOG_DIR      : $LOG_DIR"

if [ ! -e "$CXL_DAX_DEVICE" ]; then
    echo "ERROR: $CXL_DAX_DEVICE does not exist on this host" >&2
    exit 1
fi

source "$HOME/LMCache/.venv/bin/activate"

# LMCache server
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
lmcache server \
    --host 0.0.0.0 \
    --port "$LMC_ZMQ" \
    --http-host 0.0.0.0 \
    --http-port "$LMC_HTTP" \
    --prometheus-port "$PROM" \
    --l1-size-gb "$L1_SIZE_GB" \
    --eviction-policy LRU \
    --cxl-dax-device "$CXL_DAX_DEVICE" \
    --cxl-region-size-gb "$CXL_REGION_SIZE_GB" \
    --cxl-subregion-offset-gb "$SUB_OFFSET" \
    --cxl-subregion-size-gb "$CXL_SUBREGION_SIZE_GB" \
    --cxl-serve-port "$CXL_SERVE" \
    --cxl-peer "peer${PEER_ROLE}:${PEER_IP}:${CXL_SERVE}" \
    --cxl-tiering-policy always_cxl \
    --remote-zmq-timeout-ms 60000 \
    >"$LOG_DIR/lmcache.log" 2>&1 &
PIDS+=($!)

wait_for_http "http://127.0.0.1:$LMC_HTTP/api/healthcheck"

# vLLM
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "$MODEL" \
    --port "$VLLM" \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$LMC_ZMQ}}" \
    >"$LOG_DIR/vllm.log" 2>&1 &
PIDS+=($!)

wait_for_http "http://127.0.0.1:$VLLM/health" 600

echo ""
echo "Host $ROLE READY (CXL)."
echo "  LMCache HTTP : http://$MY_IP:$LMC_HTTP"
echo "  vLLM API     : http://$MY_IP:$VLLM"

wait
