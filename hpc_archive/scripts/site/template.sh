#!/usr/bin/env bash
# Site defaults for direct internal access; override values through the environment.
get_host_type() { printf '%s\n' "${HOST_TYPE:-${SLURM_CLUSTER_NAME:-generic}}"; }
get_flexserv_venv_root() { printf '%s\n' "$HOME/tapis-app/flexserv/virt_envs"; }
concrete_setup_environment() { :; }
concrete_setup_cert_tls() { unset TLS_CERT TLS_KEY; }
concrete_setup_random_token() { RAND_TOKEN=${RAND_TOKEN:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}; export RAND_TOKEN; }
concrete_setup_login_port() { :; }
concrete_setup_reverse_tunnels() { :; }
concrete_setup_access_url() { FLEXSERV_ACCESS_URL="${PROTOCOL}://${GATEWAY_BIND_HOST:-127.0.0.1}:${LOCAL_PORT}"; export FLEXSERV_ACCESS_URL; }
concrete_cleanup_site_resources() { :; }
