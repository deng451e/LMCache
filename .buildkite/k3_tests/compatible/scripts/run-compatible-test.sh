#!/usr/bin/env bash
#
# vLLM × LMCache compatibility matrix (CI).
#
# Flow: resolve version lists → filter by cutoffs → per vLLM install_vllm → per (vLLM, LMCache)
# test_pair → write compat_matrix.rst (Sphinx .. csv-table::) to OUT_FILE.
#
# Env: VLLM_VERSIONS, LMCACHE_VERSIONS (comma-separated); MIN_FREE_MEM_MB for pick-free-gpu.sh.
#
set -euo pipefail

# =============================================================================
# Configuration & constants
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMCACHE_DIR="${LMCACHE_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
WORKDIR="${WORKDIR:-/tmp/lmcache_compat_runs}"
OUT_FILE="${OUT_FILE:-${SCRIPT_DIR}/../compat_matrix.rst}"
MODEL_ID="${MODEL_ID:-facebook/opt-125m}"
PORT_BASE=18000

OK="✅"
BAD="❌"
CANDLE="🕯️"

# =============================================================================
# Logging & exit
# =============================================================================
log() { echo -e "\033[1;34m[INFO]\033[0m $*" >&2; }
log_fail() { echo -e "\033[1;31m[FAIL]\033[0m $*" >&2; }
die() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }

# report_vllm_install_failure: $1 ver, $2 log path; tail log on install_vllm failure.
report_vllm_install_failure() {
    local ver="$1"
    local install_log="$2"
    echo -e "\033[1;31m[ERROR]\033[0m Failed to install vLLM ${ver}" >&2
    echo "Full log: ${install_log}" >&2
    if [[ -f "${install_log}" ]]; then
        echo "----- last 80 lines of ${install_log} -----" >&2
        tail -n 80 "${install_log}" >&2
    else
        echo "(log file missing)" >&2
    fi
}

# =============================================================================
# Process tree
# =============================================================================
# kill_tree: SIGKILL PID $1 and descendants.
kill_tree() {
    local p="$1"
    [[ -z "$p" ]] && return
    for c in $(pgrep -P "$p" 2>/dev/null); do
        kill_tree "$c"
    done
    kill -9 "$p" 2>/dev/null || true
}

# =============================================================================
# Version strings (matrix keys)
# =============================================================================
# norm: version string -> RESULTS key (e.g. 0.11.x -> 0.11.0).
norm() {
    local v="${1%.x}"
    [[ "$v" =~ ^[0-9]+\.[0-9]+$ ]] && echo "${v}.0" || echo "$v"
}

# version_gt: true if $1 > $2 (sort -V).
version_gt() {
    local lhs="$1"
    local rhs="$2"
    [[ "$lhs" != "$rhs" ]] && [[ "$(printf '%s\n%s\n' "$lhs" "$rhs" | sort -V | tail -n1)" == "$lhs" ]]
}

# =============================================================================
# Python venv (LMCache repo)
# =============================================================================
# setup_env: create/activate .venv, safe Python env.
setup_env() {
    [[ -d "$LMCACHE_DIR/.venv" ]] || uv venv "$LMCACHE_DIR/.venv"
    # shellcheck disable=SC1091
    source "$LMCACHE_DIR/.venv/bin/activate"
    # Prevent accidental imports from repo cwd / custom PYTHONPATH.
    unset PYTHONPATH
    export PYTHONSAFEPATH=1
    export HF_TOKEN="${HF_TOKEN:-}"
}

# =============================================================================
# PyPI & CUDA (for install_vllm)
# =============================================================================
# get_deps_for_vllm: $1=x.y.z; PyPI min deps on stdout for install_vllm.
get_deps_for_vllm() {
    local v_ver="$1"
    python - <<PY
import json
import re
import urllib.request

version = "${v_ver}"
url = f"https://pypi.org/pypi/vllm/{version}/json"

with urllib.request.urlopen(url, timeout=30) as response:
    payload = json.load(response)

reqs = payload.get("info", {}).get("requires_dist") or []

min_tf = None
min_torch = None
min_torchaudio = None
min_torchvision = None

for req in reqs:
    if not req:
        continue
    pkg = req.split("[")[0].strip().lower()
    name = re.match(r"^(\w+)", pkg)
    name = name.group(1) if name else ""
    match = re.search(r"(?:>=|==)\s*([\d.]+)", req)
    if not match:
        continue

    if name == "transformers":
        min_tf = match.group(1)
    elif name == "torch":
        min_torch = match.group(1)
    elif name == "torchaudio":
        min_torchaudio = match.group(1)
    elif name == "torchvision":
        min_torchvision = match.group(1)

print(f"torch=={min_torch}" if min_torch else "torch:unspecified")
print(f"torchaudio=={min_torchaudio}" if min_torchaudio else "torchaudio:unspecified")
print(f"torchvision=={min_torchvision}" if min_torchvision else "torchvision:unspecified")
print(f"transformers>={min_tf}" if min_tf else "transformers:unspecified")
PY
}

# get_cuda_suffix_from_nvcc: nvcc -> cu121; rc 1 if missing.
get_cuda_suffix_from_nvcc() {
    local nvcc_ver
    nvcc_ver=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -1)
    [[ -z "$nvcc_ver" ]] && return 1
    echo "cu${nvcc_ver//./}"
}

# =============================================================================
# Package installs (active venv)
# =============================================================================
# install_vllm: $1 train; pinned deps + vllm; log WORKDIR/vllm_install_*.log; rc!=0 on uv fail.
install_vllm() {
    local ver="$1"
    local base="${ver//[xX]/0}"
    local major_minor_patch
    major_minor_patch=$(echo "$base" | sed -n 's/^\([0-9]\+\.[0-9]\+\.[0-9]\+\).*$/\1/p')
    local next_patch="${major_minor_patch%.*}.$((${major_minor_patch##*.} + 1))"
    local deps min_tf min_torch min_torchaudio min_torchvision cuda_suffix
    local safe_ver="${ver//[^a-zA-Z0-9._-]/_}"
    local install_log="${WORKDIR}/vllm_install_${safe_ver}.log"

    mkdir -p "${WORKDIR}"
    {
        echo "=== vLLM install ${ver} ==="
        echo "date: $(date -Is 2>/dev/null || date)"
        echo "base=${base} major_minor_patch=${major_minor_patch} next_patch=${next_patch}"
        echo "WORKDIR=${WORKDIR}"
        echo "=========="
    } >"${install_log}"

    deps="$(get_deps_for_vllm "$base" 2>>"${install_log}")" || {
        echo "[WARN] get_deps_for_vllm failed for base=${base}; continuing with empty deps" >>"${install_log}"
        deps=""
    }
    min_tf=$(echo "$deps" | sed -n 's/^transformers[>=:]*//p')
    min_torch=$(echo "$deps" | sed -n 's/^torch==//p')
    min_torchaudio=$(echo "$deps" | sed -n 's/^torchaudio==//p')
    min_torchvision=$(echo "$deps" | sed -n 's/^torchvision==//p')
    cuda_suffix="$(get_cuda_suffix_from_nvcc 2>>"${install_log}")" || cuda_suffix=""

    {
        echo "resolved deps: transformers=${min_tf} torch=${min_torch} torchaudio=${min_torchaudio} torchvision=${min_torchvision}"
        echo "cuda_suffix(nvcc)=${cuda_suffix:-<none>}"
    } >>"${install_log}"

    log "Installing vLLM $ver deps (transformers=$min_tf, torch=$min_torch+$cuda_suffix, cuda=$cuda_suffix)..."
    log "vLLM install log: ${install_log}"

    if [[ -n "$min_tf" && "$min_tf" != "unspecified" ]]; then
        echo "---- uv pip install transformers==${min_tf} ----" >>"${install_log}"
        if ! uv pip install "transformers==${min_tf}" >>"${install_log}" 2>&1; then
            report_vllm_install_failure "$ver" "${install_log}"
            return 1
        fi
    fi

    if [[ -n "$cuda_suffix" && -n "$min_torch" && "$min_torch" != "unspecified" ]]; then
        local torch_spec="torch==${min_torch}+${cuda_suffix}"
        local torchaudio_spec="torchaudio==${min_torchaudio:-$min_torch}+${cuda_suffix}"
        local torchvision_spec="torchvision==${min_torchvision:-0.23.0}+${cuda_suffix}"
        echo "---- uv pip install torch stack (${cuda_suffix}) ----" >>"${install_log}"
        if ! uv pip install "$torch_spec" "$torchaudio_spec" "$torchvision_spec" \
            --index-url "https://download.pytorch.org/whl/${cuda_suffix}" >>"${install_log}" 2>&1; then
            report_vllm_install_failure "$ver" "${install_log}"
            return 1
        fi
    fi

    echo "---- uv pip install vllm>=${base},<${next_patch} ----" >>"${install_log}"
    if ! uv pip install "vllm>=${base},<${next_patch}" >>"${install_log}" 2>&1; then
        report_vllm_install_failure "$ver" "${install_log}"
        return 1
    fi
}

# install_lmcache: $1 ver, $2 isolation, $3 run_dir; install + c_ops check.
install_lmcache() {
    local ver="$1" use_isolation="${2:-true}" run_dir="${3:-}"
    local install_log="${run_dir:-$WORKDIR}/lmcache_install.log"
    mkdir -p "$(dirname "$install_log")"

    uv pip uninstall -y lmcache >/dev/null 2>&1 || true

    if [[ "$use_isolation" == "false" ]]; then
        uv pip install --no-build-isolation \
            "lmcache @ git+https://github.com/LMCache/LMCache.git@v${ver}" \
            2>&1 | tee "$install_log" >&2
    else
        uv pip install "lmcache==$ver" 2>&1 | tee "$install_log" >&2
    fi

    local installed_ver
    installed_ver=$(python -c "from importlib.metadata import version; print(version('lmcache'))" 2>/dev/null) || return 1
    if [[ "$installed_ver" != "$ver" ]]; then
        log_fail "LMCache version mismatch: expected $ver, got $installed_ver"
        return 1
    fi

    local lmcache_path
    lmcache_path=$(python -c "import os, lmcache; print(os.path.realpath(lmcache.__file__))" 2>/dev/null) || return 1
    if [[ "$lmcache_path" != "$VIRTUAL_ENV"/lib/* ]]; then
        log_fail "lmcache imported from unexpected path: $lmcache_path"
        echo "Expected lmcache under $VIRTUAL_ENV/lib" >> "$install_log"
        echo "Actual lmcache path: $lmcache_path" >> "$install_log"
        return 1
    fi

    local torch_lib
    torch_lib=$(python -c 'import os, torch; print(os.path.dirname(torch.__file__) + "/lib")')
    export LD_LIBRARY_PATH="${torch_lib}:${LD_LIBRARY_PATH:-}"
    local c_ops_out
    c_ops_out=$(python -c "import lmcache.c_ops" 2>&1)
    local ret=$?
    echo "$c_ops_out" >> "$install_log"
    echo "$c_ops_out" > "${run_dir:-$WORKDIR}/lmcache_c_ops.log"
    return $ret
}

# =============================================================================
# One (vLLM, LMCache) pair: LMCache install, vllm serve, health + completion probe
# =============================================================================
# test_pair: $1 vLLM, $2 LMCache; prints OK / CANDLE / BAD.
test_pair() {
    local v_ver="$1" l_ver="$2"
    local port=$((PORT_BASE + RANDOM % 1000))
    local run_dir="$WORKDIR/vllm${v_ver}_lmcache${l_ver}"
    mkdir -p "$run_dir"

    # 1. Install LMCache (with retry logic) and verify lmcache.c_ops import.
    local isolated=true fail_reason=""
    local lmcache_ok=false
    if install_lmcache "$l_ver" "true" "$run_dir"; then
        lmcache_ok=true
    fi
    if [[ "$lmcache_ok" != "true" ]]; then
        isolated=false
        log "PyPI install failed, retrying with build from source..."
        if install_lmcache "$l_ver" "false" "$run_dir"; then
            lmcache_ok=true
        fi
    fi
    if [[ "$lmcache_ok" != "true" ]]; then
        fail_reason="LMCache install or lmcache.c_ops check failed; see $run_dir/lmcache_c_ops.log"
        log_fail "vLLM $v_ver + LMCache $l_ver: $fail_reason"
        echo "$BAD"
        return 1
    fi

    # 2. Setup LD_LIBRARY_PATH (for vllm serve).
    local torch_lib
    torch_lib=$(python -c 'import os, torch; print(os.path.dirname(torch.__file__) + "/lib")')
    export LD_LIBRARY_PATH="${torch_lib}:${LD_LIBRARY_PATH:-}"

    # Pick a free GPU before each run.
    source "${LMCACHE_DIR}/.buildkite/scripts/pick-free-gpu.sh" \
        "${MIN_FREE_MEM_MB:-10000}" >> "$run_dir/server.log" 2>&1 || die "Failed to pick free GPU"

    # 3. Start Server.
    LMCACHE_CHUNK_SIZE=8 vllm serve "$MODEL_ID" --port "$port" --load-format dummy \
        --gpu-memory-utilization 0.1 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
        >> "$run_dir/server.log" 2>&1 &
    local server_pid=$!

    # Ensure cleanup (kill server and all child processes).
    trap "kill_tree $server_pid" EXIT

    # 4. Health Check & Query.
    local timeout=120 status="$BAD" fail_reason=""
    while (( timeout > 0 )); do
        if curl -s "http://127.0.0.1:$port/v1/models" >/dev/null; then
            if curl -s -X POST "http://127.0.0.1:$port/v1/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"Hello\",\"max_tokens\":5}" >/dev/null; then
                status=$([[ "$isolated" == "true" ]] && echo "$OK" || echo "$CANDLE")
            else
                fail_reason="Completion request failed"
            fi
            break
        fi
        sleep 2
        (( timeout -= 2 ))
    done

    kill_tree "$server_pid"

    if [[ -z "$fail_reason" && "$status" == "$BAD" ]]; then
        fail_reason="Server did not respond within 120s; see $run_dir/server.log"
    fi
    if [[ -n "$fail_reason" ]]; then
        log_fail "vLLM $v_ver + LMCache $l_ver: $fail_reason"
    fi
    echo "$status"
}

# =============================================================================
# main: orchestration
# =============================================================================
main() {
    local -a vllm_versions=()
    local -a lmcache_versions=()
    local docs_matrix_file
    local -a filtered_vllm_versions=()
    local -a filtered_lmcache_versions=()
    local -A RESULTS
    local rv cv key res

    # --- Resolve vLLM / LMCache version lists (env or docs CSV) ---
    if [[ -n "${VLLM_VERSIONS:-}" ]]; then
        IFS=',' read -r -a vllm_versions <<< "${VLLM_VERSIONS}"
    else
        docs_matrix_file="${LMCACHE_DIR}/docs/source/getting_started/installation_compatibility.csv"
        if [[ ! -f "$docs_matrix_file" ]]; then
            die "Compatibility matrix file not found: $docs_matrix_file"
        fi
        mapfile -t vllm_versions < <(
            python3 - "$docs_matrix_file" <<'PY'
import csv
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8", newline="") as f:
    rows = list(csv.reader(f))
# e.g. "vLLM 0.11.x" -> "0.11.x"; "vLLM 0.10.2.x" -> "0.10.2.x"
pat = re.compile(r"vLLM\s+((?:\d+\.)+\d+)\.x")
for row in rows[1:]:
    if not row:
        continue
    m = pat.search(row[0])
    if m:
        print(f"{m.group(1)}.x")
PY
        )
        [[ ${#vllm_versions[@]} -gt 0 ]] || die "No vLLM versions found in $docs_matrix_file"
    fi
    [[ -n "${LMCACHE_VERSIONS:-}" ]] && IFS=',' read -r -a lmcache_versions <<< "${LMCACHE_VERSIONS}"

    # --- Filter by minimum supported versions ---
    filtered_vllm_versions=()
    for rv in "${vllm_versions[@]}"; do
        if version_gt "$(norm "$rv")" "0.8.5"; then
            filtered_vllm_versions+=("$rv")
        fi
    done
    vllm_versions=("${filtered_vllm_versions[@]}")

    filtered_lmcache_versions=()
    for cv in "${lmcache_versions[@]}"; do
        if version_gt "$(norm "$cv")" "0.3.2"; then
            filtered_lmcache_versions+=("$cv")
        fi
    done
    lmcache_versions=("${filtered_lmcache_versions[@]}")

    [[ ${#vllm_versions[@]} -gt 0 ]] || die "No vLLM versions to test after 0.8.5.x"
    [[ ${#lmcache_versions[@]} -gt 0 ]] || die "No LMCache versions to test after 0.3.2"

    echo "vllm_versions: ${vllm_versions[*]}"
    echo "lmcache_versions: ${lmcache_versions[*]}"

    # --- Matrix: install vLLM once per row, then each LMCache column ---
    for rv in "${vllm_versions[@]}"; do
        setup_env
        install_vllm "$rv" || exit 1
        for cv in "${lmcache_versions[@]}"; do
            key="$(norm "$rv")|$(norm "$cv")"
            log "Testing vLLM $rv + LMCache $cv..."
            res=$(test_pair "$rv" "$cv") || res="$BAD"
            RESULTS["$key"]="$res"
            log "Result for vLLM $rv + LMCache $cv: $res"
            echo "Cleaning uv cache..."
            uv cache clean
        done
    done

    # --- Write Sphinx csv-table fragment for check-and-update merge ---
    {
        echo ".. csv-table::"
        printf "   :header: \"\""
        for cv in "${lmcache_versions[@]}"; do
            printf ", \"%s\"" "LMCache $cv"
        done
        echo -e "\n   :widths: 20$(printf ', 15%.0s' "${lmcache_versions[@]}")\n"

        for rv in "${vllm_versions[@]}"; do
            printf "   \"%s\"" "vLLM $rv"
            for cv in "${lmcache_versions[@]}"; do
                printf ", \"%s\"" "${RESULTS["$(norm "$rv")|$(norm "$cv")"]:-$BAD}"
            done
            echo
        done
    } | tee "$OUT_FILE"
}

# Run when executed; omit when sourced (definitions only).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi

