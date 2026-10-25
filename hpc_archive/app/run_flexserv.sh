#!/usr/bin/env bash
# Tapis enters inside an existing Slurm allocation. Keep orchestration sequential.
set -euo pipefail
umask 077
# Resolve an administrator's directory alias before Rust anchors private job files.
export FLEXSERV_JOB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ ${1:-} == --help ]]; then
    echo 'Usage: run_flexserv.sh [--control-plane-only]'
    echo 'Runs Gateway on the first Slurm node and one FCS on every allocated node.'
    exit 0
fi
source "$FLEXSERV_JOB_DIR/scripts/install.sh"
source "$FLEXSERV_JOB_DIR/scripts/launch-flexserv.sh"
if [[ ${1:-} == --fcs-worker ]]; then
    fcs_worker "$FLEXSERV_JOB_DIR" "${2:?Missing runtime directory}"
fi
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_JOB_NODELIST:-} ]] || { echo 'Run inside a Slurm allocation.' >&2; exit 2; }
first_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
if [[ ${SLURMD_NODENAME:-$(hostname -s)} != "$first_node" ]]; then
    [[ ${FLEXSERV_ON_HEAD:-0} == 0 ]] || { echo 'Could not enter the first Slurm node.' >&2; exit 2; }
    export FLEXSERV_ON_HEAD=1
    exec srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --nodelist="$first_node" --export=ALL \
        bash "$FLEXSERV_JOB_DIR/app/run_flexserv.sh" "$@"
fi
ensure_environment() {
    local identity="${HOST_TYPE:-} ${SLURM_CLUSTER_NAME:-} $(hostname -f)" site=${FLEXSERV_SITE:-}
    if [[ -z "$site" ]]; then
        case "$identity" in
            *vista*|*stampede3*|*horizon*|*dgx*) site=tacc ;;
            *ascend*|*cardinal*|*pitzer*|*osc.edu*) site=osc ;;
            *) site=template ;;
        esac
    fi
    case "$site" in tacc|osc|template) ;; *) echo 'Unknown FLEXSERV_SITE.' >&2; return 1 ;; esac
    export FLEXSERV_SITE=$site
    source "$FLEXSERV_JOB_DIR/scripts/site/$site.sh"
    source "$FLEXSERV_JOB_DIR/scripts/flexserv.env"
    source "$FLEXSERV_JOB_DIR/scripts/tunnel.sh"
}
if [[ ${1:-} == --control-plane-only ]]; then export FLEXSERV_PREPARE_BACKENDS=0
elif [[ $# != 0 ]]; then echo 'Unknown option. Use --help.' >&2; exit 2; fi
ensure_environment
TUNNEL_PIDS=()
trap cleanup_flexserv EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
install_flexserv
prepare_site_access
launch_flexserv
setup_tunnel
publish_access_info
wait_for_services
