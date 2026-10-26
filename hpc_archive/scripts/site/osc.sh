#!/usr/bin/env bash
# OSC uses Open OnDemand HTTPS termination; no SSH tunnel or container is required.
get_host_type() { printf '%s\n' "${HOST_TYPE:-${SLURM_CLUSTER_NAME:-osc}}"; }
get_flexserv_venv_root() { printf '%s\n' "$HOME/tapis-app/flexserv/virt_envs"; }
concrete_setup_environment() { load_tap_functions; }
concrete_setup_random_token() { RAND_TOKEN=${RAND_TOKEN:-$(secure_random_hex)}; export RAND_TOKEN; }
concrete_setup_login_port() { :; }
concrete_setup_cert_tls() {
    setup_public_tls
    # Only the explicitly approved OOD proxy may reach a plaintext public listener.
    [[ -n "${FLEXSERV_TRUSTED_PROXY_NETWORKS:-}" ]] || {
        echo 'Set FLEXSERV_TRUSTED_PROXY_NETWORKS to the approved OOD proxy CIDRs.' >&2; return 1;
    }
    export FLEXSERV_MCP_ALLOWED_ORIGINS=${FLEXSERV_MCP_ALLOWED_ORIGINS:-https://${OSC_ONDEMAND_HOST:-ondemand.osc.edu}}
    GATEWAY_BIND_HOST=${GATEWAY_BIND_HOST:-0.0.0.0}; export GATEWAY_BIND_HOST
}
concrete_setup_reverse_tunnels() { :; }
concrete_setup_access_url() {
    FLEXSERV_ACCESS_URL="https://${OSC_ONDEMAND_HOST:-ondemand.osc.edu}/rnode/${NODE_HOSTNAME_DOMAIN}/${LOCAL_PORT}/"
    export FLEXSERV_ACCESS_URL
}
concrete_cleanup_site_resources() { :; }
