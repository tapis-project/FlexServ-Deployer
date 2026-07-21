concrete_setup_environment() {
    export VENV_PATH=${VENV_PATH:-/venvs}
    export HPC_HOST=${HPC_HOST:-$(hostname)}

    export LOGIN_NODE_PREFIX=${LOGIN_NODE_PREFIX:-login0}
    export LOGIN_NODE_COUNT=${LOGIN_NODE_COUNT:-4}

    export PUB_MODEL_HOST=${PUB_MODEL_HOST:-"${HOME}/flexserv/models"}
    export PRI_MODEL_HOST=${PRI_MODEL_HOST:-"${HOME}/flexserv/models"}
    export APPTAINER_IMAGE="${APPTAINER_IMAGE:-"${HOME}/flexserv/flexserv.sif"}"

    export BACKEND_PATCH_PATH=${BACKEND_PATCH_PATH:-"${HOME}/flexserv/patches/backend"}
    export LANDING_PAGE_PATH=${LANDING_PAGE_PATH:-"${HOME}/flexserv/patches/gateway"}
}

concrete_prepare_apptainer() {
    export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-"${HOME}/flexserv_cache"}
    mkdir -p "$APPTAINER_CACHEDIR"

    if command -v apptainer >/dev/null 2>&1; then
        echo "Apptainer already in PATH: $(apptainer --version)"
        return 0
    fi

    echo "Apptainer not found in PATH, attempting to load module..."
    module load apptainer 2>/dev/null
}

file_contains_private_key() {
    grep -Eq -- "-----BEGIN (RSA |EC |ENCRYPTED |)PRIVATE KEY-----" "$1"
}

set_certfile() {
    if [ -z "${FLEXSERV_CERTFILE:-}" ]; then
        echo "ERROR: FLEXSERV_CERTFILE is not set. HTTPS cannot be enabled."
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
        echo "HTTPS is disabled."
        return 0
    fi

    set_certfile
    set_cert_key
}

concrete_setup_random_token() {
    # Random token generated
    export RAND_TOKEN=$(echo "$$ $RANDOM $(date)" | sha1sum | awk '{print $1}')
}

generate_login_nodes() {
    local login_node_prefix="${LOGIN_NODE_PREFIX}"
    local login_node_count="${LOGIN_NODE_COUNT}"
    local i

    for i in $(seq 1 "${login_node_count}"); do
        echo "${login_node_prefix}${i}"
    done
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
