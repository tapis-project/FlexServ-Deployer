#!/bin/bash
# FlexServ Runner Script for TACC HPC Environment
# Supports normal single-node single-process mode through the gateway.
# The multi-node distributed launch path is currently unsupported.

set -e

# Parse arguments (supports both named args in any order and legacy positional args)
print_usage() {
    echo "Usage (named, order-independent):"
    echo "  $0 [--login-port <port>] [--is-distributed <0|1>] [--flexserv-port <port>] [--secret <secret>] [--model-name <model>] [--device <device>] [--serving-profile <profile>] [--dtype <dtype>] [--attn-implementation <impl>] [--model-timeout <seconds>] [--quantization <mode>] [--trust-remote-code <true|false>] [--continuous-batching <policy>] [--cb-hybrid-cache <auto|off|qwen3.5>] [--compile <policy>] [--use-kernels <policy>] [--enable-https]"
    echo "  $0 --login-port 18080 --secret flexserv"
    echo ""
    echo "Usage (legacy positional):"
    echo "  $0 <flexserv_port> <secret> <model_name> [login_port] [is_distributed] [enable_https] [quantization] [trust_remote_code] [continuous_batching]"
    echo ""
    echo "Arguments:"
    echo "  flexserv_port / --flexserv-port FlexServ service port on compute node (default: 8000)"
    echo "  secret / --secret               FlexServ auth secret (default: TAP token / flexserv)"
    echo "  model_name / --model-name       Default FlexServ private model ID (default: Qwen/Qwen3.5-0.8B)"
    echo "  device / --device               Backend device (default: auto)"
    echo "  --serving-profile              auto/baseline/decode/latency/extreme/prefill/memory/balanced (extreme=latency; Python default: balanced)"
    echo "  --dtype / --attn-implementation Optional profile overrides (omitted means inherit)"
    echo "  model_timeout / --model-timeout Backend model timeout seconds (default: 86400)"
    echo "  --quantization                 Explicit none/bnb-4bit/bnb-8bit/checkpoint override"
    echo "  trust_remote_code / --trust-remote-code  Whether to trust remote code execution (default: false)"
    echo "  --continuous-batching / --compile / --use-kernels  true/false/auto/inherit; bare flag means true"
    echo "  --cb-hybrid-cache              auto/off/qwen3.5 (Python default: auto)"
    echo "  --compile-mode / --compile-dynamic / --compile-fullgraph  Advanced compilation overrides"
    echo "  BACKEND_* / FLEXSERV_*          Optional environment overrides; explicit arguments take precedence"
    echo "  login_port / --login-port       Login node port for reverse tunnel (required unless TAP allocates one)"
    echo "  is_distributed / --is-distributed  Keep at 0; distributed mode is currently unsupported (default: 0)"
    echo "  enable_https / --enable-https   Whether to enable HTTPS in gateway mode (default: disabled)"
}

if [ "$#" -eq 0 ]; then
    print_usage
    exit 1
fi

FLEXSERV_PORT=8000
FLEXSERV_SECRET=""
MODEL_NAME="Qwen/Qwen3.5-0.8B"
LOGIN_PORT=""
IS_DISTRIBUTED=0
ENABLE_HTTPS=0
DEVICE="${FLEXSERV_DEVICE:-auto}"
MODEL_TIMEOUT="${FLEXSERV_MODEL_TIMEOUT:-86400}"
TRUST_REMOTE_CODE="${FLEXSERV_TRUST_REMOTE_CODE:-false}"
RUNTIME_FLAGS=()
RUNTIME_SEEN=()
normalize_runtime_policy() {
    case "$1" in
        true|on|1|yes) echo true ;;
        false|off|0|no) echo false ;;
        auto|inherit) echo "$1" ;;
        *) echo "ERROR: $2 must be true, false, auto or inherit" >&2; return 1 ;;
    esac
}

add_runtime_arg() {
    local flag="$1" value="$2" seen
    case "$flag" in
        --continuous-batching|--compile|--use-kernels|--compile-dynamic|--compile-fullgraph|--cb-prefix-caching|--cb-use-cuda-graph|--cb-async-batching)
            value=$(normalize_runtime_policy "$value" "$flag") || exit 1
        ;;
        --prefill-chunk-size)
            if [ "$value" = off ]; then value=false; fi
        ;;
    esac
    for seen in "${RUNTIME_SEEN[@]}"; do
        if [ "$seen" = "$flag" ]; then
            echo "ERROR: duplicate runtime override: $flag" >&2
            exit 1
        fi
    done
    RUNTIME_SEEN+=("$flag")
    RUNTIME_FLAGS+=("$flag" "$value")
}

CONTAINER_TRANSFORMERS_BACKEND_SERVER="${FLEXSERV_TRANSFORMERS_BACKEND_SERVER:-/app/flexserv/backend/backend_server.py}"

if [[ "$1" == -* ]]; then
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --login-port)
                LOGIN_PORT="$2"
                shift 2
            ;;
            --is-distributed)
                IS_DISTRIBUTED="$2"
                shift 2
            ;;
            --distributed)
                IS_DISTRIBUTED=1
                shift
            ;;
            --single-node)
                IS_DISTRIBUTED=0
                shift
            ;;
            --flexserv-port)
                FLEXSERV_PORT="$2"
                shift 2
            ;;
            --secret)
                FLEXSERV_SECRET="$2"
                shift 2
            ;;
            --model-name)
                MODEL_NAME="$2"
                shift 2
            ;;
            --device)
                DEVICE="$2"
                shift 2
            ;;
            --model-timeout)
                MODEL_TIMEOUT="$2"
                shift 2
            ;;
            --enable-https)
                ENABLE_HTTPS=1
                shift
            ;;
            --trust-remote-code)
                TRUST_REMOTE_CODE="$2"
                shift 2
            ;;
            --serving-profile|--dtype|--attn-implementation|--quantization|--compile-mode|--cache-implementation|--max-cache-len|--prefill-chunk-size|--bnb-4bit-compute-dtype|--cb-page-size|--cb-block-size|--cb-num-blocks|--cb-max-blocks-per-request|--cb-max-batch-tokens|--cb-max-requests-per-batch|--cb-max-memory-percent|--cb-compile-level|--cb-hybrid-cache|--cb-scheduler|--cb-safety-margin|--cb-cpu-offload-gib|--cb-max-queue-size|--max-outstanding-requests)
                if [ "$#" -lt 2 ] || [[ "$2" == --* ]]; then
                    echo "ERROR: $1 requires a value" >&2
                    exit 1
                fi
                flag="$1"
                [ "$flag" != "--cb-block-size" ] || flag="--cb-page-size"
                add_runtime_arg "$flag" "$2"
                shift 2
            ;;
            --continuous-batching|--compile|--use-kernels|--compile-dynamic|--compile-fullgraph|--cb-prefix-caching|--cb-use-cuda-graph|--cb-async-batching)
                flag="$1"
                value="true"
                shift
                if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
                    value="$1"
                    shift
                fi
                add_runtime_arg "$flag" "$value"
            ;;
            --no-continuous-batching|--no-compile|--no-use-kernels|--no-compile-dynamic|--no-compile-fullgraph|--no-cb-prefix-caching|--no-cb-use-cuda-graph|--no-cb-async-batching)
                add_runtime_arg "--${1#--no-}" "false"
                shift
            ;;
            -h|--help)
                print_usage
                exit 0
            ;;
            *)
                echo "ERROR: Unknown argument: $1"
                print_usage
                exit 1
            ;;
        esac
    done
else
    if [ "$#" -lt 1 ] || [ "$#" -gt 9 ]; then
        print_usage
        exit 1
    fi

    FLEXSERV_PORT=${1:-8000}
    FLEXSERV_SECRET=${2:-""}
    MODEL_NAME=${3:-"Qwen/Qwen3.5-0.8B"}
    LOGIN_PORT=${4:-""}
    IS_DISTRIBUTED=${5:-0}
    ENABLE_HTTPS=${6:-0}
    if [ "$#" -ge 7 ]; then add_runtime_arg "--quantization" "$7"; fi
    TRUST_REMOTE_CODE=${8:-${TRUST_REMOTE_CODE}}
    if [ "$#" -ge 9 ]; then add_runtime_arg "--continuous-batching" "$9"; fi
fi

HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-""}

is_integer() {
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

parse_gpu_count() {
    # Handles common Slurm values such as "4", "gpu:4", "a100:4", "gpu:a100:4".
    local raw="${1:-}"
    local parsed=""

    if [ -z "$raw" ]; then
        echo 0
        return
    fi

    if [[ "$raw" =~ ^[0-9]+$ ]]; then
        echo "$raw"
        return
    fi

    # Use the last numeric field after ':' if present.
    parsed=$(printf '%s\n' "$raw" | awk -F: '{print $NF}' | grep -E '^[0-9]+$' || true)
    if [ -n "$parsed" ]; then
        echo "$parsed"
        return
    fi

    # Fallback: sum all numbers in the string.
    parsed=$(printf '%s\n' "$raw" | grep -Eo '[0-9]+' | awk '{s+=$1} END {print s+0}')
    echo "${parsed:-0}"
}

count_cuda_visible_devices() {
    local cvd="${1:-}"
    if [ -z "$cvd" ] || [ "$cvd" = "NoDevFiles" ] || [ "$cvd" = "none" ]; then
        echo 0
        return
    fi
    awk -F, '{print NF}' <<< "$cvd"
}

GPU_COUNT=0

# Prefer an already restricted CUDA_VISIBLE_DEVICES from Slurm or the caller.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]; then
    GPU_COUNT=$(count_cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")
    echo "Respecting existing CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    # Prefer Slurm allocation metadata when present, but parse it defensively.
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        GPU_COUNT=$(parse_gpu_count "${SLURM_GPUS_ON_NODE}")
    elif [ -n "${SLURM_GPUS_PER_NODE:-}" ]; then
        GPU_COUNT=$(parse_gpu_count "${SLURM_GPUS_PER_NODE}")
    fi

    # Fall back to nvidia-smi only if it both exists and works.
    if ! is_integer "$GPU_COUNT" || [ "$GPU_COUNT" -eq 0 ]; then
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
            GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
        elif [ -d /proc/driver/nvidia/gpus ]; then
            GPU_COUNT=$(ls -d /proc/driver/nvidia/gpus/* 2>/dev/null | wc -l | tr -d ' ' || true)
        fi
    fi

    if ! is_integer "$GPU_COUNT"; then
        GPU_COUNT=0
    fi

    # If GPUs detected, expose exactly the local GPU ordinals inside the container.
    if [ "$GPU_COUNT" -gt 0 ]; then
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
        export CUDA_VISIBLE_DEVICES
    fi
fi

if ! is_integer "$GPU_COUNT"; then
    GPU_COUNT=0
fi

if [ "$GPU_COUNT" -gt 0 ]; then
    echo "Detected $GPU_COUNT NVIDIA GPU(s)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    echo "No NVIDIA GPUs detected. Running in CPU mode."
    unset CUDA_VISIBLE_DEVICES || true
fi

# Do NOT force IS_DISTRIBUTED=0 here.
# The unsupported distributed launch path remains off by default.

echo "======================================================================"
echo "FlexServ on TACC HPC - Apptainer Runtime"
echo "======================================================================"
echo "FlexServ Port: ${FLEXSERV_PORT}"
echo "Compute Node: $(hostname)"
echo "======================================================================"

export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-/work/projects/aci/cic/apps/flexserv}
mkdir -p "$APPTAINER_CACHEDIR"

# 1. Load TACC Apptainer module
echo "Loading tacc-apptainer module..."
module load tacc-apptainer
module unload xalt
echo "✓ Apptainer loaded: $(apptainer --version)"

find_available_port() {
    for port in $(seq 8000 9000); do
        if ! netstat -tuln 2>/dev/null | grep -q ":${port} "; then
            echo "$port"
            return
        fi
    done
    echo ""
}

# 2. Verify flexserv port is available on compute node, if not suggest an alternative port
echo "Checking if port ${FLEXSERV_PORT} is available on compute node..."
if netstat -tuln 2>/dev/null | grep -q ":${FLEXSERV_PORT} "; then
    echo "ERROR: Port ${FLEXSERV_PORT} is already in use on $(hostname)"
    echo "try to find an available port and rerun the script."
    AVAILABLE_PORT=$(find_available_port)
    if [ -n "$AVAILABLE_PORT" ]; then
        echo "Suggested available port: ${AVAILABLE_PORT}"
        FLEXSERV_PORT=${AVAILABLE_PORT}
        echo "Using port ${FLEXSERV_PORT} instead."
    else
        echo "No available ports found in the range 8000-9000."
        exit 1
    fi
fi
echo "✓ Port ${FLEXSERV_PORT} is available"

############### Set up TAP environment for reverse port forwarding ###############
NODE_HOSTNAME_PREFIX=$(hostname -s)
export NODE_HOSTNAME_DOMAIN=$(hostname -d)
echo "TACC: running on node $NODE_HOSTNAME_PREFIX on $NODE_HOSTNAME_DOMAIN"

TAP_FUNCTIONS="${TAP_FUNCTIONS:-/share/doc/slurm/tap_functions}"
if [ -f "${TAP_FUNCTIONS}" ]; then
    . "${TAP_FUNCTIONS}"
else
    echo "TACC: ERROR - could not find TAP functions file: ${TAP_FUNCTIONS}"
    exit 1
fi

if [ -z "${TAP_CERTFILE:-}" ]; then
    export TAP_CERTFILE="$(cat "${HOME}/.tap/.${SLURM_JOB_ID}")"
fi
if [ ! -f "${TAP_CERTFILE}" ]; then
    echo "TACC: ERROR - could not find TLS cert for secure session"
    exit 1
fi

export TAP_TOKEN="$(tap_get_token)"
if [ -z "${TAP_TOKEN}" ]; then
    echo "TACC: ERROR - could not generate token for app session"
    exit 1
fi

FLEXSERV_SECRET=${FLEXSERV_SECRET:-${FLEXSERV_TOKEN:-${TAP_TOKEN}}}
if [ -z "${FLEXSERV_OWNER:-}" ]; then
    if owner="$(whoami 2>/dev/null)" && [ -n "${owner}" ]; then
        export FLEXSERV_OWNER="${owner}"
    elif [ -n "${USER:-}" ]; then
        export FLEXSERV_OWNER="${USER}"
    else
        export FLEXSERV_OWNER="$(id -u 2>/dev/null || echo unknown)"
    fi
fi

# This is the remote port users will hit (on login nodes)
export LOGIN_PORT=${LOGIN_PORT:-"$(tap_get_port)"}
echo "FlexServ login-node port: ${LOGIN_PORT}"

export LOCAL_PORT="${FLEXSERV_PORT}"
export GATEWAY_BACKEND_PORT=${GATEWAY_BACKEND_PORT:-$((FLEXSERV_PORT + 1))}

if [ "${GATEWAY_BACKEND_PORT}" -eq "${FLEXSERV_PORT}" ] || netstat -tuln 2>/dev/null | grep -q ":${GATEWAY_BACKEND_PORT} "; then
    ALT_BACKEND_PORT=$(find_available_port)
    if [ -n "${ALT_BACKEND_PORT}" ]; then
        GATEWAY_BACKEND_PORT="${ALT_BACKEND_PORT}"
    fi
fi

echo "Gateway backend port on compute node: ${GATEWAY_BACKEND_PORT}"

############## TAP environment set up complete ##############

# 3. Set up environment variables
export PUB_MODEL_HOST=${PUB_MODEL_HOST:-"/work/projects/aci/cic/apps/flexserv/models"}
export PRI_MODEL_HOST=${PRI_MODEL_HOST:-"${SCRATCH}/flexserv/models"}

RAW_ARCH="$(uname -m 2>/dev/null || echo unknown)"
if [ -n "${FLEXSERV_ARCH_CODE:-}" ]; then
    case "${FLEXSERV_ARCH_CODE}" in
        amd64|arm64)
            ARCH_CODE="${FLEXSERV_ARCH_CODE}"
            echo "WARNING: Overriding detected architecture '${RAW_ARCH}' with FLEXSERV_ARCH_CODE='${ARCH_CODE}'."
        ;;
        *)
            echo "ERROR: Invalid FLEXSERV_ARCH_CODE='${FLEXSERV_ARCH_CODE}'. Allowed values: amd64, arm64."
            exit 1
        ;;
    esac
else
    case "${RAW_ARCH}" in
        x86_64|amd64)
            ARCH_CODE="amd64"
        ;;
        aarch64|arm64)
            ARCH_CODE="arm64"
        ;;
        armv8*|armv9*)
            ARCH_CODE="arm64"
            echo "WARNING: Architecture '${RAW_ARCH}' mapped to 'arm64'. Verify your userland/container compatibility."
        ;;
        armv7*|armv6*|armhf|armel|i386|i686|x86|ppc*|s390*|riscv*|mips*)
            echo "WARNING: Incompatible architecture '${RAW_ARCH}' for supported images (amd64, arm64)."
            echo "Set FLEXSERV_ARCH_CODE=amd64 or FLEXSERV_ARCH_CODE=arm64 only if you intentionally want to override."
            exit 1
        ;;
        *)
            echo "WARNING: Unrecognized architecture '${RAW_ARCH}'. Automatic fallback is disabled to prevent wrong-image startup."
            echo "Set FLEXSERV_ARCH_CODE=amd64 or FLEXSERV_ARCH_CODE=arm64 to override explicitly."
            exit 1
        ;;
    esac
fi

APPTAINER_IMAGE_DEFAULT="/work/projects/aci/cic/apps/flexserv/flexserv_${ARCH_CODE}_latest.sif"
export APPTAINER_IMAGE="${APPTAINER_IMAGE:-${APPTAINER_IMAGE_DEFAULT}}"

# Create models directory if it doesn't exist
HF_CACHE_HOST=${HF_CACHE_HOST:-"${PRI_MODEL_HOST}/.hf_cache"}
mkdir -p "${PRI_MODEL_HOST}" "${HF_CACHE_HOST}"

echo "Public model repository (host): ${PUB_MODEL_HOST}"
echo "Private model repository (host): ${PRI_MODEL_HOST}"
echo "HF cache directory (host): ${HF_CACHE_HOST}"
echo "Detected architecture: ${RAW_ARCH} (image arch code: ${ARCH_CODE})"
echo "Apptainer image: ${APPTAINER_IMAGE}"
echo "Default model name: ${MODEL_NAME}"
echo "Device: ${DEVICE}"
echo "Model timeout: ${MODEL_TIMEOUT}"
echo "Runtime argv overrides: ${RUNTIME_FLAGS[*]:-(profile defaults)}"
echo "FlexServ owner: ${FLEXSERV_OWNER}"
echo "FlexServ token: ${FLEXSERV_SECRET}"

# 4. Set up reverse port forwarding to login node
echo ""
echo "======================================================================"
echo "SETTING UP REVERSE PORT FORWARDING"
echo "======================================================================"
echo "FlexServ running on compute node: $(hostname):${FLEXSERV_PORT}"
echo "Forwarding to login node port: ${LOGIN_PORT}"
echo ""

# Create a reverse tunnel on each login node (login1..4)
for i in 1 2 3 4; do
    ssh -o StrictHostKeyChecking=no \
        -o ConnectTimeout=3 \
        -o ExitOnForwardFailure=yes \
        -q -f -g -N \
        -R "${LOGIN_PORT}:${NODE_HOSTNAME_PREFIX}:${LOCAL_PORT}" \
        "login${i}" || true
done

HPC_HOST="${NODE_HOSTNAME_DOMAIN}"

protocol="http"
if [ "$ENABLE_HTTPS" -ne 0 ]; then
    protocol="https"
    export TLS_CERT="${TAP_CERTFILE}"
    export TLS_KEY="${TAP_KEYFILE:-${TAP_CERTFILE}}"
    echo "Gateway TLS cert: ${TLS_CERT}"
    echo "Gateway TLS key:  ${TLS_KEY}"
fi

echo ""
echo "======================================================================"
echo "ACCESS INFORMATION"
echo "======================================================================"
echo "FlexServ address: ${protocol}://${HPC_HOST}:${LOGIN_PORT}  FlexServ token: ${FLEXSERV_SECRET}"
echo "======================================================================"
echo ""

echo "FlexServ address: ${protocol}://${HPC_HOST}:${LOGIN_PORT}  FlexServ token: ${FLEXSERV_SECRET}" > ${_tapisExecSystemOutputDir}/flexserv_access_info.txt

# 5. Run FlexServ in Apptainer container
echo "Starting FlexServ in Apptainer container..."
echo "Service will be available on port ${FLEXSERV_PORT}"

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    if [ -n "${LOGIN_PORT:-}" ]; then
        tap_release_port "${LOGIN_PORT}" || true
    fi
    echo "✓ Cleanup complete"
}
trap cleanup EXIT INT TERM

# Run apptainer with GPU support.
# Apptainer shares the host network by default, so container port is directly accessible on compute node.

SLURM_NNODES_EFFECTIVE=${SLURM_NNODES:-1}
SLURM_NODEID_EFFECTIVE=${SLURM_NODEID:-0}
export GPUS_PER_NODE=$GPU_COUNT
# Allocation size is not a torch.distributed process group. Only the launcher
# below should create WORLD_SIZE/RANK/LOCAL_RANK in actual worker processes.
DISTRIBUTED_WORLD_SIZE=$((SLURM_NNODES_EFFECTIVE * GPUS_PER_NODE))

# ======================= Distributed setup (if enabled) =======================
if [ "$IS_DISTRIBUTED" -ne 0 ]; then
    if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
        echo "ERROR: --distributed requires SLURM_JOB_NODELIST."
        exit 1
    fi

    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_PORT=$(
        srun --nodes=1 --ntasks=1 -w "$MASTER_ADDR" \
        bash -lc 'python - << "PY"
import socket, random
for _ in range(200):
    p = random.randint(20000, 45000)
    s = socket.socket()
    try:
        s.bind(("", p))
        s.close()
        print(p)
        break
    except OSError:
        pass
PY'
    )

    # NCCL bits for explicit multi-node distributed mode.
    export NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
    export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}

    if command -v ibdev2netdev >/dev/null 2>&1; then
        IB_INTERFACE=$(ibdev2netdev | awk 'NR==1 {print $5}')
    else
        IB_INTERFACE=$(ip link show 2>/dev/null | grep 'ib' | head -1 | awk '{print $2}' | tr -d ':' || true)
    fi

    if [ -n "${IB_INTERFACE:-}" ]; then
        export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${IB_INTERFACE}}
    fi
fi

export VENV_PATH=${VENV_PATH:-$WORK/venvs}
echo "VENV_PATH=${VENV_PATH}"

export BACKEND_PATCH_PATH=${BACKEND_PATCH_PATH:-/work/projects/aci/cic/apps/flexserv/patches/backend}
export LANDING_PAGE_PATH=${LANDING_PAGE_PATH:-/work/projects/aci/cic/apps/flexserv/patches/gateway}
export APPLY_PATCH=${APPLY_PATCH:-0}

BACKEND_SERVER_FLAGS=()
GATEWAY_BACKEND_FLAGS=()

if [[ "$TRUST_REMOTE_CODE" == "true" ]] || [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    BACKEND_SERVER_FLAGS+=("--trust-remote-code")
    GATEWAY_BACKEND_FLAGS+=("--backend-trust-remote-code")
fi

# Keep performance argv separate from environment so Python can report its origin.
GATEWAY_RUNTIME_ARGS=()
for ((runtime_index=0; runtime_index<${#RUNTIME_FLAGS[@]}; runtime_index+=2)); do
    GATEWAY_RUNTIME_ARGS+=("--backend-${RUNTIME_FLAGS[runtime_index]#--}" "${RUNTIME_FLAGS[runtime_index+1]}")
done

APPTAINER_GATEWAY_TLS_ENVS=()
if [ "$ENABLE_HTTPS" -ne 0 ]; then
    APPTAINER_GATEWAY_TLS_ENVS+=(--env "TLS_CERT=${TLS_CERT}")
    APPTAINER_GATEWAY_TLS_ENVS+=(--env "TLS_KEY=${TLS_KEY}")
fi

APPTAINER_PATCH_BINDS=()
if [ "$APPLY_PATCH" -ne 0 ]; then
    echo "Applying backend patches from ${BACKEND_PATCH_PATH}..."
    echo "Applying landing page patches from ${LANDING_PAGE_PATH}..."
    APPTAINER_PATCH_BINDS+=(--bind "${BACKEND_PATCH_PATH}:/app/flexserv/backend:ro")
    APPTAINER_PATCH_BINDS+=(--bind "${LANDING_PAGE_PATH}:/app/flexserv/gateway:ro")
fi

COMMON_APPTAINER_ARGS=(
    --nv
    --bind "${PUB_MODEL_HOST}:/app/models/public:rw"
    --bind "${PRI_MODEL_HOST}:/app/models/private:rw"
    --bind "${HF_CACHE_HOST}:/app/models/.hf_cache:rw"
    "${APPTAINER_PATCH_BINDS[@]}"
    --env "FLEXSERV_VENV=/app/venvs/flexserv"
    --env "PUB_MODEL_REPO=/app/models/public"
    --env "PRI_MODEL_REPO=/app/models/private"
    --env "HF_HOME=/app/models/.hf_cache"
    --env "FLEXSERV_BACKEND_TYPE=${FLEXSERV_BACKEND_TYPE:-transformers}"
    --env "FLEXSERV_TOKEN=${FLEXSERV_SECRET}"
    --env "FLEXSERV_OWNER=${FLEXSERV_OWNER}"
    --env "HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}"
    --env "TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache"
    --env "TRITON_CACHE_DIR=/tmp/triton"
    --env "CC=/usr/bin/gcc"
    --env "CXX=/usr/bin/g++"
)

# These values remain environment inputs; explicit argv wins in the Python resolver.
for runtime_env in \
    BACKEND_SERVING_PROFILE FLEXSERV_SERVING_PROFILE BACKEND_DTYPE FLEXSERV_DTYPE \
    BACKEND_ATTN_IMPLEMENTATION FLEXSERV_ATTN_IMPLEMENTATION BACKEND_QUANTIZATION FLEXSERV_QUANTIZATION \
    BACKEND_COMPILE_MODE FLEXSERV_COMPILE_MODE BACKEND_CACHE_IMPLEMENTATION FLEXSERV_CACHE_IMPLEMENTATION \
    BACKEND_MAX_CACHE_LEN FLEXSERV_MAX_CACHE_LEN BACKEND_PREFILL_CHUNK_SIZE FLEXSERV_PREFILL_CHUNK_SIZE \
    BACKEND_BNB_4BIT_COMPUTE_DTYPE FLEXSERV_BNB_4BIT_COMPUTE_DTYPE BACKEND_CB_PAGE_SIZE FLEXSERV_CB_PAGE_SIZE \
    BACKEND_CB_NUM_BLOCKS FLEXSERV_CB_NUM_BLOCKS BACKEND_CB_MAX_BLOCKS_PER_REQUEST FLEXSERV_CB_MAX_BLOCKS_PER_REQUEST \
    BACKEND_CB_MAX_BATCH_TOKENS FLEXSERV_CB_MAX_BATCH_TOKENS BACKEND_CB_MAX_REQUESTS_PER_BATCH FLEXSERV_CB_MAX_REQUESTS_PER_BATCH \
    BACKEND_CB_MAX_MEMORY_PERCENT FLEXSERV_CB_MAX_MEMORY_PERCENT BACKEND_CB_COMPILE_LEVEL FLEXSERV_CB_COMPILE_LEVEL \
    BACKEND_CB_HYBRID_CACHE FLEXSERV_CB_HYBRID_CACHE BACKEND_CB_SCHEDULER FLEXSERV_CB_SCHEDULER BACKEND_CB_SAFETY_MARGIN FLEXSERV_CB_SAFETY_MARGIN \
    BACKEND_CB_CPU_OFFLOAD_GIB FLEXSERV_CB_CPU_OFFLOAD_GIB BACKEND_CB_MAX_QUEUE_SIZE FLEXSERV_CB_MAX_QUEUE_SIZE \
    BACKEND_MAX_OUTSTANDING_REQUESTS FLEXSERV_MAX_OUTSTANDING_REQUESTS BACKEND_CONTINUOUS_BATCHING FLEXSERV_CONTINUOUS_BATCHING \
    BACKEND_COMPILE FLEXSERV_COMPILE BACKEND_USE_KERNELS FLEXSERV_USE_KERNELS \
    BACKEND_COMPILE_DYNAMIC FLEXSERV_COMPILE_DYNAMIC BACKEND_COMPILE_FULLGRAPH FLEXSERV_COMPILE_FULLGRAPH \
    BACKEND_CB_PREFIX_CACHING FLEXSERV_CB_PREFIX_CACHING BACKEND_CB_USE_CUDA_GRAPH FLEXSERV_CB_USE_CUDA_GRAPH \
    BACKEND_CB_ASYNC_BATCHING FLEXSERV_CB_ASYNC_BATCHING BACKEND_CB_BLOCK_SIZE FLEXSERV_CB_BLOCK_SIZE \
    BACKEND_ATTN_IMPL FLEXSERV_ATTN_IMPL
do
    if declare -p "$runtime_env" >/dev/null 2>&1; then
        runtime_value="${!runtime_env}"
        case "$runtime_env" in
            *_CONTINUOUS_BATCHING|*_COMPILE|*_USE_KERNELS|*_COMPILE_DYNAMIC|*_COMPILE_FULLGRAPH|*_CB_PREFIX_CACHING|*_CB_USE_CUDA_GRAPH|*_CB_ASYNC_BATCHING)
                runtime_value=$(normalize_runtime_policy "$runtime_value" "$runtime_env") || exit 1
            ;;
            *_PREFILL_CHUNK_SIZE)
                if [ "$runtime_value" = off ]; then runtime_value=false; fi
            ;;
        esac
        COMMON_APPTAINER_ARGS+=(--env "${runtime_env}=${runtime_value}")
    fi
done

echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

case "${MODEL_NAME}" in
    FLEX:PRI:*|FLEX:PUB:*)
    ;;
    *)
        MODEL_NAME="FLEX:PRI:${MODEL_NAME}"
    ;;
esac

if [ "$IS_DISTRIBUTED" -ne 0 ]; then
    DIRECT_BACKEND_MODEL_NAME="${MODEL_NAME}"
    case "${DIRECT_BACKEND_MODEL_NAME}" in
        FLEX:PRI:*)
            DIRECT_BACKEND_MODEL_NAME="${DIRECT_BACKEND_MODEL_NAME#FLEX:PRI:}"
        ;;
        FLEX:PUB:*)
            echo "ERROR: distributed direct-backend mode cannot resolve public-pool FlexServ IDs. Use a private model or disable --distributed."
            exit 1
        ;;
    esac

    echo "Launching FlexServ container in MULTI-NODE DISTRIBUTED mode..."
    srun --ntasks=${SLURM_NNODES_EFFECTIVE} \
        --ntasks-per-node=1 \
        --cpus-per-task=${SLURM_CPUS_PER_TASK:-8} \
        apptainer run \
        "${COMMON_APPTAINER_ARGS[@]}" \
        --env "MASTER_ADDR=${MASTER_ADDR}" \
        --env "MASTER_PORT=${MASTER_PORT}" \
        "${APPTAINER_IMAGE}" \
        /app/venvs/flexserv/bin/accelerate launch \
        --multi_gpu \
        --num_machines=${SLURM_NNODES_EFFECTIVE} \
        --num_processes=${DISTRIBUTED_WORLD_SIZE} \
        --machine_rank=${SLURM_NODEID_EFFECTIVE} \
        --main_process_ip=${MASTER_ADDR} \
        --main_process_port=${MASTER_PORT} \
        --same_network \
        --mixed_precision=bf16 \
        "${CONTAINER_TRANSFORMERS_BACKEND_SERVER}" \
        --default-model "${DIRECT_BACKEND_MODEL_NAME}" \
        --host 0.0.0.0 \
        --port "${FLEXSERV_PORT}" \
        --flexserv-token "${FLEXSERV_SECRET}" \
        --device "${DEVICE}" \
        --model-timeout "${MODEL_TIMEOUT}" \
        "${RUNTIME_FLAGS[@]}" \
        "${BACKEND_SERVER_FLAGS[@]}"
else
    echo "Launching FlexServ container in SINGLE-NODE NORMAL mode..."

    apptainer run \
        "${COMMON_APPTAINER_ARGS[@]}" \
        "${APPTAINER_GATEWAY_TLS_ENVS[@]}" \
        "${APPTAINER_IMAGE}" \
        /app/flexserv/bin/flexserv-gateway \
        --manage-backend \
        --backend-kind transformers \
        --backend-default-model "${MODEL_NAME}" \
        --port "${FLEXSERV_PORT}" \
        --backend-port "${GATEWAY_BACKEND_PORT}" \
        --backend-host 0.0.0.0 \
        --backend-server "${CONTAINER_TRANSFORMERS_BACKEND_SERVER}" \
        --flexserv-token "${FLEXSERV_SECRET}" \
        --backend-device "${DEVICE}" \
        --backend-model-timeout "${MODEL_TIMEOUT}" \
        "${GATEWAY_RUNTIME_ARGS[@]}" \
        "${GATEWAY_BACKEND_FLAGS[@]}"
fi

# If we reach here, container exited normally
echo "FlexServ container stopped"
