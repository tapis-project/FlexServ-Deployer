concrete_setup_environment() {
    local flexserv_root

    flexserv_root="$(get_flexserv_root)" || return 1

    export VENV_PATH=${VENV_PATH:-"/venvs"}
    export HPC_HOST=${HPC_HOST:-"$(get_cluster_name)-login01.hpc.osc.edu"}

    export PUB_MODEL_HOST=${PUB_MODEL_HOST:-"${flexserv_root}/models/public"}
    export PRI_MODEL_HOST=${PRI_MODEL_HOST:-"${flexserv_root}/models/private"}
    export APPTAINER_IMAGE="${APPTAINER_IMAGE:-"${flexserv_root}/flexserv.sif"}"

    export BACKEND_PATCH_PATH=${BACKEND_PATCH_PATH:-"${flexserv_root}/patches/backend"}
    export LANDING_PAGE_PATH=${LANDING_PAGE_PATH:-"${flexserv_root}/patches/gateway"}

    mkdir -p "${flexserv_root}" "${PUB_MODEL_HOST}" "${PRI_MODEL_HOST}"
    chmod 700 "${flexserv_root}" 2>/dev/null || true
}

concrete_prepare_apptainer() {
    local flexserv_root

    flexserv_root="$(get_flexserv_root)" || return 1

    export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-"${flexserv_root}/apptainer_cache"}
    mkdir -p "$APPTAINER_CACHEDIR"
    chmod 700 "$APPTAINER_CACHEDIR" 2>/dev/null || true

    if command -v apptainer >/dev/null 2>&1; then
        echo "Apptainer already in PATH: $(apptainer --version)"
        return 0
    fi

    echo "ERROR: Apptainer not found in PATH"
    return 1
}

file_contains_private_key() {
    grep -Eq -- "-----BEGIN (RSA |EC |ENCRYPTED |)PRIVATE KEY-----" "$1"
}

set_certfile() {
    if [ -z "${FLEXSERV_CERTFILE:-}" ]; then
        echo "ERROR: FLEXSERV_CERTFILE is not set. HTTPS cannot be enabled."
        echo "Set FLEXSERV_CERTFILE=/path/to/cert.pem and, if needed, FLEXSERV_KEYFILE=/path/to/key.pem."
        return 1
    fi

    # File not exisiting check
    if [ ! -f "${FLEXSERV_CERTFILE}" ]; then
        echo "ERROR: FLEXSERV_CERTFILE is set but does not exist: ${FLEXSERV_CERTFILE}"
        return 1
    fi

    export TLS_CERT="${FLEXSERV_CERTFILE}"
}

set_cert_key() {
    # Check if the cert file contains a private key
    local certkey

    if file_contains_private_key "${FLEXSERV_CERTFILE}"; then
        certkey="${FLEXSERV_CERTFILE}"
    else
        if [ -z "${FLEXSERV_KEYFILE:-}" ]; then
            echo "ERROR: FLEXSERV_CERTFILE does not contain a private key and FLEXSERV_KEYFILE is not set. HTTPS cannot be enabled."
            echo "Set FLEXSERV_KEYFILE=/path/to/key.pem or use a combined cert/key PEM file."
            return 1
        fi

        if [ ! -f "${FLEXSERV_KEYFILE}" ]; then
            echo "ERROR: FLEXSERV_KEYFILE is set but does not exist: ${FLEXSERV_KEYFILE}"
            return 1
        fi

        if ! file_contains_private_key "${FLEXSERV_KEYFILE}"; then
            echo "ERROR: FLEXSERV_KEYFILE does not contain a private key: ${FLEXSERV_KEYFILE}"
            return 1
        fi

        certkey="${FLEXSERV_KEYFILE}"
    fi

    export TLS_KEY="${certkey}"
}

concrete_setup_cert_tls() {
    if [ "$ENABLE_HTTPS" -eq 0 ]; then
        echo "WARNING: HTTPS is managed by OSC Open OnDemand. Skipping TLS cert and key setup."
        return 0
    else
        echo "WARNING: HTTPS is enabled. FLEXSERV_CERTFILE and FLEXSERV_KEYFILE are obsolete on OSC. The TLS cert and key will be managed by OSC Open OnDemand."
    fi

    export TLS_CERT="${FLEXSERV_CERTFILE:-}"
    export TLS_KEY="${FLEXSERV_KEYFILE:-}"
}

concrete_setup_random_token() {
    # Random token generated
    export RAND_TOKEN=$(echo "$$ $RANDOM $(date)" | sha1sum | awk '{print $1}')
}

get_cluster_name() {
    # OSC Slurm exposes the cluster name (pitzer, ascend, cardinal). Fall back to
    # scontrol for shells where SLURM_CLUSTER_NAME was not exported.
    if [ -n "${SLURM_CLUSTER_NAME:-}" ]; then
        echo "${SLURM_CLUSTER_NAME}"
    elif command -v scontrol >/dev/null 2>&1; then
        scontrol show config 2>/dev/null | grep -i '^ClusterName' | awk '{print($3);}'
    fi
}

get_project_name() {
    local project_name

    if [ -n "${OSC_ACCOUNT_NAME:-}" ]; then
        project_name="${OSC_ACCOUNT_NAME}"
    elif [ -n "${SLURM_JOB_ACCOUNT:-}" ]; then
        project_name="${SLURM_JOB_ACCOUNT}"
    else
        echo "ERROR: Could not determine OSC project. Set OSC_ACCOUNT_NAME or run inside a Slurm job with SLURM_JOB_ACCOUNT." >&2
        return 1
    fi

    printf '%s\n' "${project_name}" | tr '[:lower:]' '[:upper:]'
}

get_flexserv_root() {
    local project_name

    project_name="$(get_project_name)" || return 1
    echo "/fs/scratch/${project_name}/${USER}/flexserv"
}

get_number_of_login_nodes() {
    # OSC cluster hostnames resolve to the current login-node addresses, so DNS
    # gives us the login-node count without hardcoding per-cluster values.
    getent ahostsv4 $(get_cluster_name).osc.edu | awk '{ print($1); }' | uniq | wc -l
}

generate_login_nodes() {
    local login_node_prefix="$(get_cluster_name)-login0"
    local login_node_count="$(get_number_of_login_nodes)"
    local i

    for i in $(seq 1 "${login_node_count}"); do
        echo "${login_node_prefix}${i}"
    done
}

get_compute_node_fqdn() {
    hostname -f
}

concrete_setup_access_url() {
    local compute_node_fqdn

    compute_node_fqdn="$(get_compute_node_fqdn)"

    export OSC_ONDEMAND_HOST=${OSC_ONDEMAND_HOST:-"ondemand.osc.edu"}

    export FLEXSERV_ACCESS_URL="https://${OSC_ONDEMAND_HOST}/rnode/${compute_node_fqdn}/${LOCAL_PORT}/"

    echo "OSC Open OnDemand access URL: ${FLEXSERV_ACCESS_URL}"
}

list_listening_ports() {
    # Almost all modern Linux systems provide ss; netstat covers older environments.
    if command -v ss >/dev/null 2>&1; then
        ss -Htln
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tln
    else
        echo "ERROR: Neither 'ss' nor 'netstat' command is available to list listening ports." >&2
        return 1
    fi
}

concrete_setup_login_port() {
    local start_port=60000
    local end_port=65000
    local tmp
    local login_node
    local port

    LOCKDIR="${HOME}/flexserv_cache/cache/locks"
    mkdir -p "${LOCKDIR}"

    # This is the remote port users will hit (on login nodes)
    if [ -n "${LOGIN_PORT:-}" ]; then
        if mkdir "${LOCKDIR}/${LOGIN_PORT}.lock" 2>/dev/null; then
            export LOGIN_PORT_LOCKFILE="${LOCKDIR}/${LOGIN_PORT}.lock"

            echo "Using pre-set LOGIN_PORT: ${LOGIN_PORT} with lock file: ${LOGIN_PORT_LOCKFILE}"
            return 0
        fi

        export LOGIN_PORT=""
        echo "ERROR: ${LOGIN_PORT} is already in use. Please set a different LOGIN_PORT or leave it unset to auto-select."
        return 1
    fi

    tmp=$(mktemp)

    for login_node in $(generate_login_nodes); do
        # List listening ports on the login node and filter them to find ports in the specified range
        ssh -q "${login_node}" "$(declare -f list_listening_ports); list_listening_ports" | awk -v start="${start_port}" -v end="${end_port}" '
            {
                split($4, a, ":")
                port=a[length(a)]
                if (port >= start && port <= end) {
                    print(port)
                }
            }
            ' >>"${tmp}"
    done

    for port in $(seq "${start_port}" "${end_port}"); do
        if grep -q "^${port}$" "${tmp}"; then
            continue
        fi

        # mkdir is atomic so race conditions are avoided by using a lock directory
        if mkdir "${LOCKDIR}/${port}.lock" 2>/dev/null; then
            export LOGIN_PORT="${port}"
            export LOGIN_PORT_LOCKFILE="${LOCKDIR}/${port}.lock"

            echo "Selected LOGIN_PORT: ${LOGIN_PORT} with lock file: ${LOGIN_PORT_LOCKFILE}"

            rm -f "${tmp}"
            return 0
        fi
    done

    echo "ERROR: Could not find an available login port in range ${start_port}-${end_port}"
    rm -f "${tmp}"
    return 1
}

concrete_setup_reverse_tunnels() {
    local login_node

    # Create a reverse tunnel on each login node
    for login_node in $(generate_login_nodes); do
        ssh -o StrictHostKeyChecking=accept-new \
            -o ConnectTimeout=3 \
            -o ExitOnForwardFailure=yes \
            -q -f -g -N \
            -R "127.0.0.1:${LOGIN_PORT}:${NODE_HOSTNAME_PREFIX}:${LOCAL_PORT}" \
            "${login_node}" || true
    done
}

release_login_port() {
    if [ -n "${LOGIN_PORT_LOCKFILE:-}" ]; then
        rmdir "${LOGIN_PORT_LOCKFILE}" 2>/dev/null || true
        echo "Released login port lock: ${LOGIN_PORT_LOCKFILE}"
    fi
}

concrete_cleanup_site_resources() {
    release_login_port
}
