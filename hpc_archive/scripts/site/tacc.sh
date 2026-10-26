#!/usr/bin/env bash
# TACC path conventions and TAP access, independent of inference installation.
get_host_type() {
    local identity="${HOST_TYPE:-} ${SLURM_CLUSTER_NAME:-} $(hostname -f)" name
    for name in vista stampede3 horizon dgx; do
        if [[ "$identity" == *"$name"* ]]; then printf '%s\n' "$name"; return; fi
    done
    echo 'Cannot determine TACC host type; set HOST_TYPE.' >&2; return 1
}
get_flexserv_venv_root() {
    case "$HOST_TYPE" in
        vista|stampede3) printf '%s\n' /work/projects/aci/cic/apps/flexserv/virt_envs ;;
        dgx) printf '%s\n' /opt/share/virt_envs ;;
        *) printf '%s\n' "$HOME/tapis-app/flexserv/virt_envs" ;;
    esac
}
concrete_setup_environment() {
    load_tap_functions
}
concrete_setup_random_token() {
    if [[ -z ${RAND_TOKEN:-} ]]; then
        if [[ "$TAP_AVAILABLE" == 1 ]]; then RAND_TOKEN=$(tap_get_token)
        else RAND_TOKEN=$(secure_random_hex); fi
    fi
    [[ -n "$RAND_TOKEN" ]] || { echo 'No access token could be generated.' >&2; return 1; }
    export RAND_TOKEN
}
get_login_port() {
    # Probe once per reachable login node; actual SSH forwarding still owns the bind.
    local node listing used=' ' reachable=0 value candidate attempt
    for node in ${FLEXSERV_LOGIN_NODES:-login1 login2 login3 login4}; do
        if listing=$(ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=3 "$node" \
            'if command -v ss >/dev/null; then ss -H -ltn; else netstat -ltn; fi' 2>/dev/null); then
            reachable=$((reachable+1))
            while IFS= read -r value; do used+="${value##*:} "; done < <(awk '{print $4}' <<< "$listing")
        fi
    done
    (( reachable > 0 )) || { echo 'Cannot inspect any login node; configure LOGIN_PORT or SSH access.' >&2; return 1; }
    for ((attempt=0; attempt<32; attempt++)); do
        value=$(od -An -N2 -tu2 /dev/urandom); candidate=$((value%40000+20000))
        if [[ "$used" != *" $candidate "* ]]; then printf '%s\n' "$candidate"; return; fi
    done
    echo 'No login port candidate found.' >&2; return 1
}
concrete_setup_login_port() {
    export TAP_PORT_ALLOCATED=0
    if [[ -z ${LOGIN_PORT:-} ]]; then
        if [[ "$TAP_AVAILABLE" == 1 ]]; then LOGIN_PORT=$(tap_get_port); export TAP_PORT_ALLOCATED=1
        else LOGIN_PORT=$(get_login_port); fi
    fi
    valid_port "$LOGIN_PORT" || { echo 'Invalid LOGIN_PORT.' >&2; return 1; }
    export LOGIN_PORT
    FLEXSERV_LOGIN_PUBLIC_HOSTS=""
    local login host public
    for login in ${FLEXSERV_LOGIN_NODES:-login1 login2 login3 login4}; do
        host=$(ssh -G "$login" 2>/dev/null | awk '$1=="hostname" {print $2; exit}')
        public=$(getent hosts "${host:-$login}" | awk 'NF {print $2; exit}') || true
        public=${public:-${host:-$login}}
        FLEXSERV_LOGIN_PUBLIC_HOSTS+="$public "
    done
    export FLEXSERV_LOGIN_PUBLIC_HOSTS
}
concrete_setup_cert_tls() {
    setup_public_tls
}
concrete_setup_reverse_tunnels() {
    local node remote_bind=${FLEXSERV_TUNNEL_BIND:-}
    remote_bind=${remote_bind:-0.0.0.0}
    for node in ${FLEXSERV_LOGIN_NODES:-login1 login2 login3 login4}; do
        ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=3 \
            -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -g -N -T \
            -R "${remote_bind}:${LOGIN_PORT}:127.0.0.1:${LOCAL_PORT}" "$node" \
            >>"$FLEXSERV_JOB_DIR/logs/tunnel.log" 2>&1 &
        TUNNEL_PIDS+=("$!")
    done
    sleep 1
    local alive=0 pid
    local index=0 host
    local public_hosts=( $FLEXSERV_LOGIN_PUBLIC_HOSTS )
    for pid in ${TUNNEL_PIDS[@]+"${TUNNEL_PIDS[@]}"}; do
        if kill -0 "$pid" 2>/dev/null; then
            alive=$((alive+1))
            if [[ -z "${FLEXSERV_PUBLIC_HOST:-}" ]]; then
                FLEXSERV_PUBLIC_HOST=${public_hosts[$index]}
                export FLEXSERV_PUBLIC_HOST
            fi
        fi
        index=$((index+1))
    done
    (( alive > 0 )) || { echo 'No TACC reverse tunnel could be established.' >&2; return 1; }
}
concrete_setup_access_url() {
    if [[ ${FLEXSERV_TUNNEL_BIND:-0.0.0.0} == 127.0.0.1 ]]; then
        FLEXSERV_ACCESS_URL="http://127.0.0.1:${LOGIN_PORT}"
        printf 'SSH-only access: forward laptop port %s to 127.0.0.1:%s on %s.\n' "$LOGIN_PORT" "$LOGIN_PORT" "$FLEXSERV_PUBLIC_HOST"
    else
        FLEXSERV_ACCESS_URL="${PROTOCOL}://${FLEXSERV_PUBLIC_HOST:?Missing successful login host}:${LOGIN_PORT}"
    fi
    export FLEXSERV_ACCESS_URL
}
concrete_cleanup_site_resources() {
    if [[ ${TAP_PORT_ALLOCATED:-0} == 1 ]] && declare -F tap_release_port >/dev/null; then tap_release_port "$LOGIN_PORT" || true; fi
}
