#!/usr/bin/env bash
# Cross-host P2P: launch one LMCache MP server + one vLLM instance on this host
# and peer with the other host via ZMQ over TCP.
#
# Usage:  ./test_p2p_cross.sh <role 0|1> <peer_ip>
# Env:    MODEL, L1_SIZE_GB

set -euo pipefail
ROLE="${1:?usage: $0 <0|1> <peer_ip>}"
PEER_IP="${2:?peer_ip required}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MY_IP="$(hostname -I | awk '{print $1}')"
L1_SIZE_GB="${L1_SIZE_GB:-8}"

LMC_ZMQ=5550
LMC_HTTP=8090
REMOTE_CTRL=5200
PROM=9090
VLLM=8010

LOG_DIR="$(dirname "${BASH_SOURCE[0]}")/logs-host${ROLE}"
mkdir -p "$LOG_DIR"

PIDS=()
cleanup() { trap - INT TERM EXIT; for p in "${PIDS[@]}"; do kill $p 2>/dev/null || true; done; sleep 2; for p in "${PIDS[@]}"; do kill -9 $p 2>/dev/null || true; done; }
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

echo "=== Cross-host P2P host$ROLE ==="
echo "  MY_IP   : $MY_IP"
echo "  PEER_IP : $PEER_IP"
echo "  MODEL   : $MODEL"
echo "  LOG_DIR : $LOG_DIR"

source "$HOME/LMCache/.venv/bin/activate"

# Compute peer label
PEER_ROLE=$(( 1 - ROLE ))

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
    --remote-mode p2p \
    --remote-serve-port "$REMOTE_CTRL" \
    --remote-zmq-timeout-ms 60000 \
    --remote-peer "peer${PEER_ROLE}:${PEER_IP}:${REMOTE_CTRL}" \
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
echo "Host $ROLE READY."
echo "  LMCache HTTP: http://$MY_IP:$LMC_HTTP"
echo "  vLLM API:     http://$MY_IP:$VLLM"

wait
