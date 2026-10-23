#!/usr/bin/env bash
# Forward laptop loopback to the first compute node through the site's login host.
# SSH owns the foreground connection; no background tunnel or PID file is created.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h || $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 USER@LOGIN_HOST GATEWAY_NODE [GATEWAY_PORT=8000] [LOCAL_PORT=GATEWAY_PORT]"
    [[ ${1:-} == --help || ${1:-} == -h ]] && exit 0
    exit 2
fi
login=$1
node=$2
remote_port=${3:-8000}
local_port=${4:-$remote_port}
[[ $login != -* && $login != *[[:space:]]* ]] || { echo "Invalid SSH login target." >&2; exit 2; }
[[ $node =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || { echo "Invalid Gateway node hostname." >&2; exit 2; }
for port in "$remote_port" "$local_port"; do
    [[ $port =~ ^[0-9]{1,5}$ ]] && (( 10#$port >= 1 && 10#$port <= 65535 )) || {
        echo "Ports must be integers between 1 and 65535." >&2; exit 2;
    }
done
printf 'Gateway: http://127.0.0.1:%s ; MCP: http://127.0.0.1:%s/mcp\n' "$local_port" "$local_port"
printf 'If the local port differs, configure the MCP client Host as localhost:%s.\n' "$remote_port"
exec ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L "127.0.0.1:${local_port}:${node}:${remote_port}" "$login"
