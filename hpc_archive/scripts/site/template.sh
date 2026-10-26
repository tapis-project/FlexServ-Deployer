#!/usr/bin/env bash
# Site defaults for direct internal access; override values through the environment.
get_host_type() { printf '%s\n' "${HOST_TYPE:-${SLURM_CLUSTER_NAME:-generic}}"; }
get_flexserv_venv_root() { printf '%s\n' "$HOME/tapis-app/flexserv/virt_envs"; }
concrete_setup_environment() { load_tap_functions; }
concrete_setup_cert_tls() { setup_public_tls; }
concrete_setup_random_token() { RAND_TOKEN=${RAND_TOKEN:-$(secure_random_hex)}; export RAND_TOKEN; }
concrete_setup_login_port() { :; }
concrete_setup_reverse_tunnels() { :; }
concrete_setup_access_url() {
    local host=${GATEWAY_BIND_HOST:-0.0.0.0}
    if [[ "$host" == 0.0.0.0 || "$host" == :: ]]; then host=$NODE_HOSTNAME_DOMAIN; fi
    FLEXSERV_ACCESS_URL="${PROTOCOL}://${host}:${LOCAL_PORT}"; export FLEXSERV_ACCESS_URL
}
concrete_cleanup_site_resources() { :; }
