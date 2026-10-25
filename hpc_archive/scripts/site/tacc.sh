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
    local functions=${TAP_FUNCTIONS:-/share/doc/slurm/tap_functions}
    [[ -r "$functions" ]] || { echo "TAP functions not found: $functions" >&2; return 1; }
    source "$functions"
    mkdir -p "$HOME/.tap"
}
concrete_setup_random_token() {
    RAND_TOKEN=${RAND_TOKEN:-$(tap_get_token)}
    [[ -n "$RAND_TOKEN" ]] || { echo 'TAP did not provide an access token.' >&2; return 1; }
    export RAND_TOKEN
}
concrete_setup_login_port() {
    LOGIN_PORT=${LOGIN_PORT:-$(tap_get_port)}
    [[ "$LOGIN_PORT" =~ ^[0-9]+$ ]] && (( LOGIN_PORT > 0 && LOGIN_PORT < 65536 ))
    export LOGIN_PORT
    FLEXSERV_LOGIN_PUBLIC_HOSTS=""
    local login host public
    for login in ${FLEXSERV_LOGIN_NODES:-login1 login2 login3 login4}; do
        host=$(ssh -G "$login" 2>/dev/null | awk '$1=="hostname" {print $2; exit}')
        public=$(python3 -c 'import socket,sys; print(socket.getfqdn(sys.argv[1]))' "${host:-$login}")
        FLEXSERV_LOGIN_PUBLIC_HOSTS+="$public "
    done
    export FLEXSERV_LOGIN_PUBLIC_HOSTS
}
concrete_setup_cert_tls() {
    if [[ "$ENABLE_HTTPS" == 1 ]]; then
        local pointer="$HOME/.tap/.$SLURM_JOB_ID"
        [[ -r "$pointer" ]] || { echo 'TAP certificate pointer is missing.' >&2; return 1; }
        TLS_CERT=$(cat "$pointer"); TLS_KEY=${FLEXSERV_KEYFILE:-$TLS_CERT}
        [[ -r "$TLS_CERT" && -r "$TLS_KEY" ]] || { echo 'TAP certificate/key unavailable.' >&2; return 1; }
        export TLS_CERT TLS_KEY
    fi
}
concrete_setup_reverse_tunnels() {
    local node remote_bind=${FLEXSERV_TUNNEL_BIND:-}
    if [[ -z "$remote_bind" ]]; then
        if [[ "$PROTOCOL" == https ]]; then remote_bind=0.0.0.0; else remote_bind=127.0.0.1; fi
    fi
    if [[ "$PROTOCOL" == http && "$remote_bind" != 127.0.0.1 ]]; then
        echo 'Public TAP access requires HTTPS; use a loopback remote bind for an SSH-only HTTP session.' >&2
        return 1
    fi
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
    if [[ "$PROTOCOL" == http ]]; then
        FLEXSERV_ACCESS_URL="http://127.0.0.1:${LOGIN_PORT}"
        printf 'SSH-only access: forward laptop port %s to 127.0.0.1:%s on %s.\n' "$LOGIN_PORT" "$LOGIN_PORT" "$FLEXSERV_PUBLIC_HOST"
    else
        FLEXSERV_ACCESS_URL="https://${FLEXSERV_PUBLIC_HOST:?Missing successful login host}:${LOGIN_PORT}"
    fi
    export FLEXSERV_ACCESS_URL
}
concrete_cleanup_site_resources() {
    if [[ -n "${LOGIN_PORT:-}" ]] && declare -F tap_release_port >/dev/null; then tap_release_port "$LOGIN_PORT" || true; fi
}
