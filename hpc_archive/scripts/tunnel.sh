#!/usr/bin/env bash
# The selected site owns access preparation; no per-site dispatch is repeated here.
prepare_site_access() {
    concrete_setup_environment
    concrete_setup_random_token
    concrete_setup_login_port
    concrete_setup_cert_tls
    PROTOCOL=http
    if [[ -n "${TLS_CERT:-}" && -n "${TLS_KEY:-}" ]]; then PROTOCOL=https; fi
    export PROTOCOL
}
setup_tunnel() {
    concrete_setup_reverse_tunnels
    concrete_setup_access_url
}
publish_access_info() {
    # This explicitly private file is the only normal place containing the access token.
    printf 'FlexServ URL: %s\nFlexServ token: %s\n' "$FLEXSERV_ACCESS_URL" "$RAND_TOKEN" > "$FLEXSERV_JOB_DIR/flexserv_access_info.txt"
    chmod 600 "$FLEXSERV_JOB_DIR/flexserv_access_info.txt"
    printf 'FlexServ URL: %s\nAccess credentials: %s/flexserv_access_info.txt\n' "$FLEXSERV_ACCESS_URL" "$FLEXSERV_JOB_DIR"
}
