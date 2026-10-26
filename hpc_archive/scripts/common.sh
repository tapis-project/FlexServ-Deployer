#!/usr/bin/env bash
# Shared Bash helpers for private paths, JSON inputs and finite startup budgets.
require_commands() {
    local name
    for name in "$@"; do
        command -v "$name" >/dev/null || { echo "Required command not found: $name" >&2; return 1; }
    done
}
secure_random_hex() {
    od -An -N "${1:-32}" -tx1 /dev/urandom | tr -d ' \n'
}
valid_port() {
    [[ ${1:-} =~ ^[0-9]{1,5}$ ]] && (( 10#$1 > 0 && 10#$1 <= 65535 ))
}
file_sha256() {
    local result
    result=$(sha256sum -- "$1") || return
    printf '%s\n' "${result%% *}"
}
safe_relative_path() {
    case "/$1/" in *'/../'*|*'/./'*|*'//'*) return 1 ;; esac
    [[ -n "$1" && "$1" != /* && "$1" != *\\* && "$1" != *$'\n'* && "$1" != *$'\r'* ]]
}
no_link_path() {
    local root=$1 rest=$2 part current=$1
    [[ ! -L "$root" ]] || return 1
    while [[ -n "$rest" ]]; do
        part=${rest%%/*}; current="$current/$part"
        [[ ! -L "$current" ]] || return 1
        if [[ "$rest" == */* ]]; then rest=${rest#*/}; else rest=; fi
    done
}
verify_inventory() {
    local root=$1 inventory=$2 line digest name actual
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^([a-f0-9]{64})\ \ (.+)$ ]] || { echo 'Invalid checksum entry.' >&2; return 1; }
        digest=${BASH_REMATCH[1]}; name=${BASH_REMATCH[2]}
        safe_relative_path "$name" && no_link_path "$root" "$name" && [[ -f "$root/$name" ]] || {
            echo "Unsafe checksum path: $name" >&2; return 1;
        }
        actual=$(file_sha256 "$root/$name") || return
        [[ "$actual" == "$digest" ]] || { echo "Checksum mismatch: $name" >&2; return 1; }
    done < "$inventory"
}
utc_epoch() {
    date -u -d "$1" +%s 2>/dev/null || date -u -j -f '%Y-%m-%dT%H:%M:%S' "$1" +%s
}
utc_timestamp() {
    date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ
}
node_ipv4_addresses() {
    getent ahostsv4 "$1" | awk '{print $1}' | LC_ALL=C sort -u
}
load_tap_functions() {
    local file=${TAP_FUNCTIONS:-/share/doc/slurm/tap_functions}
    export TAP_AVAILABLE=0
    if [[ -r "$file" ]]; then source "$file"; export TAP_AVAILABLE=1; fi
}
setup_public_tls() {
    unset TLS_CERT TLS_KEY HTTP_REDIRECT_PORT
    if [[ "$ENABLE_HTTPS" == 1 ]]; then
        local pointer="$HOME/.tap/.$SLURM_JOB_ID"
        [[ ${TAP_AVAILABLE:-0} == 1 && -r "$pointer" ]] || {
            echo '--enable-https requires TAP certificate/key files.' >&2; return 1;
        }
        TLS_CERT=$(cat "$pointer"); TLS_KEY=${FLEXSERV_KEYFILE:-$TLS_CERT}
        [[ -r "$TLS_CERT" && -r "$TLS_KEY" ]] || { echo 'TAP certificate/key unavailable.' >&2; return 1; }
        export TLS_CERT TLS_KEY
    fi
}
