#!/bin/bash
# FlexServ Runner Script for HPC Environments
# Supports:
#   1) normal single-node single-process mode through the gateway
#   2) optional multi-node distributed mode if explicitly enabled

set -e

# Parse arguments (supports both named args in any order and legacy positional args)
print_usage() {
    echo "Usage (named, order-independent):"
    echo "  $0 [--site <name>] [--login-port <port>] [--is-distributed <0|1>] [--flexserv-port <port>] [--secret <secret>] [--model-name <model>] [--device <device>] [--dtype <dtype>] [--attn-implementation <impl>] [--model-timeout <seconds>] [--quantization <mode|bnb-4bit|bnb-8bit>] [--trust-remote-code <true|false>] [--continuous-batching <true|false>] [--enable-https]"
    echo "  $0 --site tacc --login-port 18080 --secret flexserv"
    echo ""
    echo "Usage (legacy positional):"
    echo "  $0 <flexserv_port> <secret> <model_name> [login_port] [is_distributed] [enable_https] [quantization] [trust_remote_code] [continuous_batching]"
    echo ""
    echo "Arguments:"
    echo "  --site                          HPC site name (default: tacc)"
    echo "  --flexserv-port                 FlexServ service port on compute node (default: 8000)"
    echo "  --secret                        FlexServ auth secret (default: TAP token / flexserv)"
    echo "  --model-name                    Default FlexServ private model ID (default: Qwen/Qwen3.5-0.8B)"
    echo "  --device                        Backend device (default: auto)"
    echo "  --dtype                         Backend dtype (default: bfloat16)"
    echo "  --attn-implementation           Attention implementation (default: sdpa)"
    echo "  --model-timeout                 Backend model timeout seconds (default: 86400)"
    echo "  --quantization                  Quantization mode for backend (default: none)"
    echo "  --trust-remote-code             Whether to trust remote code execution (default: false)"
    echo "  --continuous-batching           Whether to enable continuous batching (default: false)"
    echo "  --login-port                    Login node port for reverse tunnel (required unless TAP allocates one)"
    echo "  --is-distributed                Multi-node distributed mode (0/1, default: 0)"
    echo "  --enable-https                  Whether to enable HTTPS in gateway mode (default: disabled)"
}

FLEXSERV_PORT=8000
FLEXSERV_SECRET=""
MODEL_NAME="Qwen/Qwen3.5-0.8B"
LOGIN_PORT=""
HPC_SITE_NAME_CLI=""
IS_DISTRIBUTED=0
ENABLE_HTTPS=0
DEVICE="${FLEXSERV_DEVICE:-auto}"
DTYPE="${FLEXSERV_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${FLEXSERV_ATTN_IMPL:-sdpa}"
MODEL_TIMEOUT="${FLEXSERV_MODEL_TIMEOUT:-86400}"
QUANTIZATION="${FLEXSERV_QUANTIZATION:-none}"
TRUST_REMOTE_CODE="${FLEXSERV_TRUST_REMOTE_CODE:-false}"
CONTINUOUS_BATCHING="${FLEXSERV_CONTINUOUS_BATCHING:-false}"
CONTAINER_TRANSFORMERS_BACKEND_SERVER="${FLEXSERV_TRANSFORMERS_BACKEND_SERVER:-/app/flexserv/backend/backend_server.py}"

if [ "$#" -gt 0 ] && [[ "$1" == -* ]]; then
    while [[ $# -gt 0 ]]; do
        case "$1" in
        --site)
            HPC_SITE_NAME_CLI="$2"
            shift 2
            ;;
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
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --attn-implementation)
            ATTN_IMPLEMENTATION="$2"
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
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --trust-remote-code)
            TRUST_REMOTE_CODE="$2"
            shift 2
            ;;
        --continuous-batching)
            CONTINUOUS_BATCHING="$2"
            shift 2
            ;;
        -h | --help)
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
elif [ "$#" -gt 0 ]; then
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
    QUANTIZATION=${7:-${QUANTIZATION}}
    TRUST_REMOTE_CODE=${8:-${TRUST_REMOTE_CODE}}
    CONTINUOUS_BATCHING=${9:-${CONTINUOUS_BATCHING}}
fi

HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-""}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HPC_SITE_DEFAULT="tacc"

if [ -n "${HPC_SITE_NAME_CLI:-}" ]; then
    HPC_SITE_SCRIPT="${SCRIPT_DIR}/sites/${HPC_SITE_NAME_CLI}.sh"
    echo "Using HPC site from --site argument: ${HPC_SITE_NAME_CLI}"
elif [ -n "${HPC_SITE_SCRIPT:-}" ]; then
    echo "Using HPC site from environment variable HPC_SITE_NAME: ${HPC_SITE_NAME_CLI}"
else
    HPC_SITE_SCRIPT="${SCRIPT_DIR}/sites/${HPC_SITE_DEFAULT}.sh"
    echo "No --site argument or HPC_SITE_NAME env var provided, defaulting to site: ${HPC_SITE_DEFAULT}"
fi

if [ ! -f "${HPC_SITE_SCRIPT}" ]; then
    echo "ERROR: HPC SITE SCRIPT at ${HPC_SITE_SCRIPT} is not a file"
    exit 1
fi

echo "Using HPC site script: ${HPC_SITE_SCRIPT}"

. "${HPC_SITE_SCRIPT}"

setup_environment() {
    concrete_setup_environment
}

prepare_apptainer() {
    echo "Preparing Apptainer environment..."

    concrete_prepare_apptainer || {
        echo "ERROR: Failed to prepare Apptainer environment"
        exit 1
    }

    if ! command -v apptainer >/dev/null 2>&1; then
        echo "ERROR: Apptainer is not available after preparation"
        exit 1
    fi

    echo "✓ Apptainer loaded: $(apptainer --version)"
}

setup_cert_tls() {
    concrete_setup_cert_tls

    if [ "$ENABLE_HTTPS" -ne 0 ]; then
        if [ -z "${TLS_CERT:-}" ] || [ -z "${TLS_KEY:-}" ]; then
            echo "ERROR: HTTPS is enabled but TLS_CERT or TLS_KEY is not set"
            exit 1
        fi

        export PROTOCOL="https"
        echo "Gateway TLS cert: ${TLS_CERT}"
        echo "Gateway TLS key:  ${TLS_KEY}"
    else
        export PROTOCOL="http"
    fi
}

setup_random_token() {
    concrete_setup_random_token

    if [ -z "${RAND_TOKEN:-}" ]; then
        echo "ERROR: Failed to generate random token"
        exit 1
    fi
}

setup_login_port() {
    concrete_setup_login_port

    if [ -z "${LOGIN_PORT:-}" ]; then
        echo "ERROR: LOGIN_PORT is not set after setup"
        exit 1
    fi

    echo "FlexServ login-node port: ${LOGIN_PORT}"
}

setup_reverse_tunnels() {
    concrete_setup_reverse_tunnels
}

setup_access_url() {
    concrete_setup_access_url

    if [ -z "${FLEXSERV_ACCESS_URL:-}" ]; then
        echo "ERROR: FLEXSERV_ACCESS_URL is not set after access URL setup"
        exit 1
    fi
}

cleanup_site_resources() {
    concrete_cleanup_site_resources
}

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
    awk -F, '{print NF}' <<<"$cvd"
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
# Multi-node distributed mode remains off by default, but explicit --distributed still works.

echo "======================================================================"
echo "FlexServ on HPC - Apptainer Runtime"
echo "======================================================================"
echo "FlexServ Port: ${FLEXSERV_PORT}"
echo "Compute Node: $(hostname)"
echo "======================================================================"

setup_environment

# 1. Prepare Apptainer environment
prepare_apptainer

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

############### Set up environment for reverse port forwarding ###############
NODE_HOSTNAME_PREFIX=$(hostname -s)
export NODE_HOSTNAME_DOMAIN=$(hostname -d)
echo "HPC: running on node $NODE_HOSTNAME_PREFIX on $NODE_HOSTNAME_DOMAIN"

setup_cert_tls
setup_random_token
setup_login_port

FLEXSERV_SECRET=${FLEXSERV_SECRET:-${FLEXSERV_TOKEN:-${RAND_TOKEN}}}
if [ -z "${FLEXSERV_OWNER:-}" ]; then
    if owner="$(whoami 2>/dev/null)" && [ -n "${owner}" ]; then
        export FLEXSERV_OWNER="${owner}"
    elif [ -n "${USER:-}" ]; then
        export FLEXSERV_OWNER="${USER}"
    else
        export FLEXSERV_OWNER="$(id -u 2>/dev/null || echo unknown)"
    fi
fi

export LOCAL_PORT="${FLEXSERV_PORT}"
export GATEWAY_BACKEND_PORT=${GATEWAY_BACKEND_PORT:-$((FLEXSERV_PORT + 1))}

if [ "${GATEWAY_BACKEND_PORT}" -eq "${FLEXSERV_PORT}" ] || netstat -tuln 2>/dev/null | grep -q ":${GATEWAY_BACKEND_PORT} "; then
    ALT_BACKEND_PORT=$(find_available_port)
    if [ -n "${ALT_BACKEND_PORT}" ]; then
        GATEWAY_BACKEND_PORT="${ALT_BACKEND_PORT}"
    fi
fi

echo "Gateway backend port on compute node: ${GATEWAY_BACKEND_PORT}"

############## Environment set up complete ##############

# 3. Set up environment variables
export PUB_MODEL_HOST=${PUB_MODEL_HOST:-"/work/projects/aci/cic/apps/flexserv/models"}
export PRI_MODEL_HOST=${PRI_MODEL_HOST:-"${SCRATCH}/flexserv/models"}

RAW_ARCH="$(uname -m 2>/dev/null || echo unknown)"
if [ -n "${FLEXSERV_ARCH_CODE:-}" ]; then
    case "${FLEXSERV_ARCH_CODE}" in
    amd64 | arm64)
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
    x86_64 | amd64)
        ARCH_CODE="amd64"
        ;;
    aarch64 | arm64)
        ARCH_CODE="arm64"
        ;;
    armv8* | armv9*)
        ARCH_CODE="arm64"
        echo "WARNING: Architecture '${RAW_ARCH}' mapped to 'arm64'. Verify your userland/container compatibility."
        ;;
    armv7* | armv6* | armhf | armel | i386 | i686 | x86 | ppc* | s390* | riscv* | mips*)
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
echo "DType: ${DTYPE}"
echo "Attention implementation: ${ATTN_IMPLEMENTATION}"
echo "Model timeout: ${MODEL_TIMEOUT}"
echo "Quantization: ${QUANTIZATION}"
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

setup_reverse_tunnels

setup_access_url

echo ""
echo "======================================================================"
echo "ACCESS INFORMATION"
echo "======================================================================"
echo "FlexServ address: ${FLEXSERV_ACCESS_URL}  FlexServ token: ${FLEXSERV_SECRET}"
echo "======================================================================"
echo ""

# 5. Run FlexServ in Apptainer container
echo "Starting FlexServ in Apptainer container..."
echo "Service will be available on port ${FLEXSERV_PORT}"

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    cleanup_site_resources
    echo "✓ Cleanup complete"
}
trap cleanup EXIT INT TERM

# Run apptainer with GPU support.
# Apptainer shares the host network by default, so container port is directly accessible on compute node.

SLURM_NNODES_EFFECTIVE=${SLURM_NNODES:-1}
SLURM_NODEID_EFFECTIVE=${SLURM_NODEID:-0}
export GPUS_PER_NODE=$GPU_COUNT
export WORLD_SIZE=$((SLURM_NNODES_EFFECTIVE * GPUS_PER_NODE))

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

if [[ "$CONTINUOUS_BATCHING" == "true" ]] || [[ "$CONTINUOUS_BATCHING" == "1" ]]; then
    BACKEND_SERVER_FLAGS+=("--continuous-batching")
    GATEWAY_BACKEND_FLAGS+=("--backend-continuous-batching")
fi

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
)

echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

case "${MODEL_NAME}" in
FLEX:PRI:* | FLEX:PUB:*)
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
        --num_processes=${WORLD_SIZE} \
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
        --dtype "${DTYPE}" \
        --attn-implementation "${ATTN_IMPLEMENTATION}" \
        --model-timeout "${MODEL_TIMEOUT}" \
        --quantization "${QUANTIZATION}" \
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
        --backend-dtype "${DTYPE}" \
        --backend-model-timeout "${MODEL_TIMEOUT}" \
        --backend-quantization "${QUANTIZATION}" \
        --backend-attn-implementation "${ATTN_IMPLEMENTATION}" \
        "${GATEWAY_BACKEND_FLAGS[@]}"
fi

# If we reach here, container exited normally
echo "FlexServ container stopped"
