#!/bin/bash
# FlexServ Runner Script for TACC HPC Environment
# Usage: ./run_flexserv.sh <port> <secret>

set -e

# Parse arguments (supports both named args in any order and legacy positional args)
print_usage() {
    echo "Usage (named, order-independent):"
    echo "  $0 [--login-port <port>] [--is-distributed <0|1>] [--flexserv-port <port>] [--secret <secret>] [--model-name <model>]"
    echo "  $0 --login-port 18080 --secret flexserv"
    echo ""
    echo "Usage (legacy positional):"
    echo "  $0 <flexserv_port> <secret> <model_name> [login_port] [is_distributed] "
    echo ""
    echo "Arguments:"
    echo "  flexserv_port / --flexserv-port FlexServ service port on compute node (default: 8000)"
    echo "  secret / --secret               FlexServ auth secret (default: flexserv)"
    echo "  model_name / --model-name       Default model name/path (default: Qwen/Qwen3-0.6B)"
    echo "  login_port / --login-port       Login node port for reverse tunnel (required)"
    echo "  is_distributed / --is-distributed  Whether to run distributed (0/1, default: 0)"
}

if [ "$#" -eq 0 ]; then
    print_usage
    exit 1
fi

FLEXSERV_PORT=8000
FLEXSERV_SECRET=""
MODEL_NAME="Qwen/Qwen3-0.6B"
LOGIN_PORT=""
IS_DISTRIBUTED=0

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
    if [ "$#" -lt 1 ] || [ "$#" -gt 5 ]; then
        print_usage
        exit 1
    fi
    
    FLEXSERV_PORT=${1:-8000}
    FLEXSERV_SECRET=${2:-""}
    MODEL_NAME=${3:-"Qwen/Qwen3-0.6B"}
    LOGIN_PORT=$4
    IS_DISTRIBUTED=${5:-0}
fi

if [ -z "$LOGIN_PORT" ]; then
    echo "ERROR: login_port is required (use --login-port <port> or positional arg #1)."
    print_usage
    exit 1
fi

HUGGINGFACE_TOKEN=${HF_TOKEN:-""}

GPU_COUNT=0

# Try nvidia-smi only if it both exists AND works
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
elif [ -d /proc/driver/nvidia/gpus ]; then
    GPU_COUNT=$(ls -d /proc/driver/nvidia/gpus/* 2>/dev/null | wc -l || true)
fi

# Check from Slurm environment variables 
GPU_COUNT=${SLURM_GPUS_ON_NODE:-${GPU_COUNT}}

# If GPUs detected → set CUDA_VISIBLE_DEVICES
if [ "$GPU_COUNT" -gt 0 ]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
    export CUDA_VISIBLE_DEVICES
    echo "Detected $GPU_COUNT NVIDIA GPU(s)"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
    echo "No NVIDIA GPUs detected. Running in CPU mode."
    unset CUDA_VISIBLE_DEVICES || true
fi

# Temporarily disable distributed mode:

IS_DISTRIBUTED=0

echo "======================================================================"
echo "FlexServ on TACC HPC - Apptainer Runtime"
echo "======================================================================"
echo "FlexServ Port: ${FLEXSERV_PORT}"
echo "Compute Node: $(hostname)"
echo "======================================================================"

export APPTAINER_CACHEDIR=$WORK/cache/apptainer
mkdir -p $APPTAINER_CACHEDIR

# 1. Load TACC Apptainer module
echo "Loading tacc-apptainer module..."
module load tacc-apptainer
module unload xalt
echo "✓ Apptainer loaded: $(apptainer --version)"

find_available_port() {
    for port in $(seq 8000 9000); do
        if ! netstat -tuln 2>/dev/null | grep -q ":${port} "; then
            echo $port
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

TAP_FUNCTIONS="/share/doc/slurm/tap_functions"
if [ -f "${TAP_FUNCTIONS}" ]; then
    . "${TAP_FUNCTIONS}"
else
    echo "TACC: ERROR - could not find TAP functions file: ${TAP_FUNCTIONS}"
    exit 1
fi

mkdir -p "${HOME}/.tap"
export TAP_CERTFILE="$(cat "${HOME}/.tap/.${SLURM_JOB_ID}")"
if [ ! -f "${TAP_CERTFILE}" ]; then
    echo "TACC: ERROR - could not find TLS cert for secure session"
    exit 1
fi

export TAP_TOKEN="$(tap_get_token)"
if [ -z "${TAP_TOKEN}" ]; then
    echo "TACC: ERROR - could not generate token for app session"
    exit 1
fi

FLEXSERV_SECRET=${FLEXSERV_SECRET:-${TAP_TOKEN}}

# This is the remote port users will hit (on login nodes)
export LOGIN_PORT=${LOGIN_PORT:-"$(tap_get_port)"}
echo "FlexServ login-node port: ${LOGIN_PORT}"


export LOCAL_PORT="${FLEXSERV_PORT}"

############## TAP environment set up complete ##############

# 3. Set up environment variables
export MODEL_REPO="${SCRATCH}/flexserv/models"
export HF_HOME="${SCRATCH}/flexserv/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
# export APPTAINER_IMAGE="${SCRATCH}/flexserv/flexserv_latest.sif"
export APPTAINER_IMAGE="docker://zhangwei217245/flexserv-transformers:1.3.0"

# Create models directory if it doesn't exist
mkdir -p "${MODEL_REPO}"
mkdir -p "${HUGGINGFACE_HUB_CACHE}"

echo "Model repository: ${MODEL_REPO}"
echo "Apptainer image: ${APPTAINER_IMAGE}"
echo "Default model name: ${MODEL_NAME}"

# Check if image exists
# if [ ! -f "${APPTAINER_IMAGE}" ]; then
#     echo "ERROR: Apptainer image not found at ${APPTAINER_IMAGE}"
#     echo "Please build or pull the image first:"
#     echo "  apptainer pull docker://zhangwei217245/flexserv:latest"
#     exit 1
# fi

# 4. Set up reverse port forwarding to login node
echo ""
echo "======================================================================"
echo "SETTING UP REVERSE PORT FORWARDING"
echo "======================================================================"
echo "FlexServ running on compute node: $(hostname):${FLEXSERV_PORT}"
echo "Forwarding to login node port: ${LOGIN_PORT}"
echo ""

# # Determine login node (try to use the one we're connected through)
# LOGIN_NODE=${TACC_LOGIN_NODE:-"${LOGIN_HOST_PREFIX}.tacc.utexas.edu"}
# echo "Using login node: ${LOGIN_NODE}"

# # Start reverse SSH tunnel in background
# -R forwards remote (login node) port to local (compute node) port
# -N means no remote command, just port forwarding
# -f runs in background
# -o ServerAliveInterval=60 keeps connection alive

# Create a reverse tunnel on each login node (login1..4)
for i in 1 2 3 4; do
    ssh -o StrictHostKeyChecking=no \
    -o ConnectTimeout=3 \
    -o ExitOnForwardFailure=yes \
    -q -f -g -N \
    -R "${LOGIN_PORT}:${NODE_HOSTNAME_PREFIX}:${LOCAL_PORT}" \
    "login${i}" || true
done

# echo "Starting reverse SSH tunnel..."
# ssh -f -N -R ${LOGIN_PORT}:localhost:${FLEXSERV_PORT} \
# -o ServerAliveInterval=60 \
# -o ServerAliveCountMax=3 \
# -o StrictHostKeyChecking=no \
# ${LOGIN_NODE} &
# SSH_TUNNEL_PID=$!
# sleep 3

# if ps -p ${SSH_TUNNEL_PID} > /dev/null 2>&1; then
#     echo "✓ Reverse tunnel established (PID: ${SSH_TUNNEL_PID})"
# else
#     echo "WARNING: Could not verify tunnel PID (may still be running)"
# fi

HPC_HOST="${NODE_HOSTNAME_DOMAIN}"

echo ""
echo "======================================================================"
echo "ACCESS INFORMATION"
echo "======================================================================"
echo "FlexServ is now accessible at:"
echo ""
# echo "  http://${LOGIN_NODE}:${LOGIN_PORT}"
echo "  https://${HPC_HOST}:${LOGIN_PORT}"
echo " Your TAP token is ${TAP_TOKEN}"
echo "Anyone on the TACC network or with VPN can access this URL directly!"
echo "No additional SSH tunneling required from client machines."
echo "======================================================================"
echo ""

# 5. Run FlexServ in Apptainer container
echo "Starting FlexServ in Apptainer container..."
echo "Service will be available on port ${FLEXSERV_PORT}"

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    # if [ ! -z "$SSH_TUNNEL_PID" ]; then
    #     echo "Stopping reverse SSH tunnel..."
    #     kill ${SSH_TUNNEL_PID} 2>/dev/null || true
    # fi
    if [ -n "${LOGIN_PORT:-}" ]; then
        tap_release_port "${LOGIN_PORT}" || true
    fi
    echo "✓ Cleanup complete"
}
trap cleanup EXIT INT TERM

# Run apptainer with GPU support in background
# Apptainer shares the host network by default, so container port is directly accessible on compute node

export GPUS_PER_NODE=$GPU_COUNT
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))


# ======================= Distributed setup (if enabled) =======================
if [ "$IS_DISTRIBUTED" -ne 0 ]; then
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

    # NCCL bits (adjust iface names to your cluster)
    export NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0

    if command -v ibdev2netdev &> /dev/null; then
        # ibdev2netdev exists, use it
        IB_INTERFACE=$(ibdev2netdev | head -n1 | awk '{print $5}')
    else
        # Fallback to ip link
        IB_INTERFACE=$(ip link show | grep 'ib' | head -1 | awk '{print $2}'|tr -d ':')
    fi

    export NCCL_SOCKET_IFNAME=${IB_INTERFACE}



export VENV_PATH=${VENV_PATH:-$WORK/venvs}
echo "VENV_PATH=${VENV_PATH}"


if [ "$IS_DISTRIBUTED" -ne 0 ]; then
    echo "Launching FlexServ container in DISTRIBUTED mode..."
    srun --ntasks=${SLURM_NNODES} \
    --ntasks-per-node=1 \
    --cpus-per-task=${SLURM_CPUS_PER_TASK:-8} \
    apptainer run --nv \
    --bind ${MODEL_REPO}:/app/models:ro \
    --bind ${HF_HOME}:/root/.cache/huggingface \
    --env MASTER_ADDR=${MASTER_ADDR} \
    --env MASTER_PORT=${MASTER_PORT} \
    --env HF_TOKEN=${HUGGINGFACE_TOKEN} \
    ${APPTAINER_IMAGE} \
    /app/venvs/transformers/bin/accelerate launch \
    --multi_gpu \
    --num_machines=${SLURM_NNODES} \
    --num_processes=${WORLD_SIZE} \
    --machine_rank=${SLURM_NODEID} \
    --main_process_ip=${MASTER_ADDR} \
    --main_process_port=${MASTER_PORT} \
    --same_network \
    --mixed_precision=bf16 \
    /app/flexserv/python/backend/transformers/backend_server.py \
    ${MODEL_NAME} \
    --host 0.0.0.0 \
    --port ${FLEXSERV_PORT} \
    --flexserv-token ${FLEXSERV_SECRET}
else
    echo "Launching FlexServ container in SINGLE-NODE mode..."
    apptainer run --nv \
    --bind ${MODEL_REPO}:/app/models:rw \
    --bind ${HF_HOME}:/root/.cache/huggingface \
    --env HF_TOKEN=${HUGGINGFACE_TOKEN} \
    ${APPTAINER_IMAGE} \
    /app/boot_loader.sh \
    --default-model ${MODEL_NAME} \
    --host 0.0.0.0 \
    --port ${FLEXSERV_PORT} \
    --flexserv-token ${FLEXSERV_SECRET} \
    --attn-implementation ${FLEXSERV_ATTN_IMPL:-sdpa}
fi
# If we reach here, container exited normally
echo "FlexServ container stopped"