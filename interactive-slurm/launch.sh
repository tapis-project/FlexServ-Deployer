#!/usr/bin/env bash
# Run the existing trusted Bootstrap on the first allocated node. Bootstrap owns
# the manifest, job mTLS, one FCS per node, Gateway discovery and process cleanup.
set -euo pipefail
umask 077

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage: source interactive-slurm/site.env; bash interactive-slurm/launch.sh

Run inside an interactive Slurm allocation. Required environment:
  BOOTSTRAP_BIN, GATEWAY_BIN, FCS_BIN  Shared Linux release executables
  FCS_PACK                          Operator-approved v0.13 Pack payload
  FCS_MCP_TOKEN                     Public MCP bearer credential
  FLEXSERV_ADMIN_SECRET             Separate Gateway administrator credential

See env.example and README.md for optional site settings and SSH forwarding.
Keep this command in the foreground; Ctrl-C asks Bootstrap to stop its Run.
HELP
    exit 0
fi
[[ $# == 0 ]] || { echo "Unexpected arguments; use --help." >&2; exit 2; }

: "${SLURM_JOB_ID:?Start an interactive Slurm allocation first}"
: "${SLURM_JOB_NODELIST:?Slurm must provide SLURM_JOB_NODELIST}"
: "${BOOTSTRAP_BIN:?Set BOOTSTRAP_BIN}"
: "${GATEWAY_BIN:?Set GATEWAY_BIN}"
: "${FCS_BIN:?Set FCS_BIN}"
: "${FCS_PACK:?Set FCS_PACK to an approved payload directory}"
: "${FCS_MCP_TOKEN:?Set FCS_MCP_TOKEN without putting it in command arguments}"
: "${FLEXSERV_ADMIN_SECRET:?Set a separate FLEXSERV_ADMIN_SECRET}"

export TZ=UTC SLURM_EXPORT_ENV=ALL
SRUN_BIN=$(command -v srun)
SCONTROL_BIN=$(command -v scontrol)
export SRUN_BIN SCONTROL_BIN
command -v python3 >/dev/null
nodes=$("$SCONTROL_BIN" show hostnames "$SLURM_JOB_NODELIST")
first_node=${nodes%%$'\n'*}
[[ -n $first_node ]] || { echo "The allocation has no nodes." >&2; exit 1; }

# salloc may leave the shell on the login node; move the driver to node zero.
current_node=${SLURMD_NODENAME:-$(hostname -s)}
if [[ ${current_node%%.*} != "${first_node%%.*}" ]]; then
    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
    exec "$SRUN_BIN" --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=1 \
        --nodelist="$first_node" --export=ALL bash "$script_dir/launch.sh"
fi

export GATEWAY_PORT=${GATEWAY_PORT:-8000}
export GATEWAY_BIND_HOST=${GATEWAY_BIND_HOST:-0.0.0.0}
export MCP_PRINCIPAL=${MCP_PRINCIPAL:-hpc-user}
export STATE_ROOT=${STATE_ROOT:-$HOME/.local/state/flexserv/slurm-$SLURM_JOB_ID}
mkdir -p -- "$STATE_ROOT"
STATE_ROOT=$(cd -- "$STATE_ROOT" && pwd -P)
export STATE_ROOT
session_dir=$(mktemp -d "$STATE_ROOT/launch-XXXXXXXX")
export SESSION_DIR=$session_dir
export FCS_NODE_NAMES=$nodes

# Generate the current strict Bootstrap contract; no legacy registration POSTs.
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timedelta, timezone


def path(value):
    """Canonical shared paths avoid symlink ancestry in protected Run files."""
    return str(Path(value).expanduser().resolve(strict=True))


def pin(value):
    """Pin executable bytes without loading a large release binary into memory."""
    filename = path(value)
    digest = hashlib.sha256()
    with open(filename, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": filename, "sha256": digest.hexdigest()}


env = os.environ
root = Path(env["SESSION_DIR"]).resolve()
nodes = env["FCS_NODE_NAMES"].splitlines()
for directory in ("runs", "archive", "gateway-runtime", "fetch", "public-models", "private-models"):
    (root / directory).mkdir(mode=0o700)

# Both this query and Bootstrap's independent query interpret Slurm time in UTC.
job = subprocess.check_output(
    [env["SCONTROL_BIN"], "show", "job", "-o", env["SLURM_JOB_ID"]], text=True
)
fields = dict(word.split("=", 1) for word in job.split() if "=" in word)
end = datetime.fromisoformat(fields["EndTime"]).replace(tzinfo=timezone.utc)
seconds = int(env.get("RUN_SECONDS", "3600"))
if not 1 <= seconds <= 600000:
    raise SystemExit("RUN_SECONDS must be between 1 and 600000")
cutoff = min(end - timedelta(seconds=10), datetime.now(timezone.utc) + timedelta(seconds=seconds))
if cutoff <= datetime.now(timezone.utc):
    raise SystemExit("The allocation has no usable walltime remaining")

pack = path(env["FCS_PACK"])
inventory = hashlib.sha256((Path(pack) / "checksums.sha256").read_bytes()).hexdigest()
approved = env.get("FCS_PACK_SHA256", inventory)
if approved != inventory:
    raise SystemExit("FCS_PACK_SHA256 does not match the selected Pack inventory")
fcs_args = ["--pack", pack, "--approved-checksums", approved]
for variable, option in (("FCS_DEPLOY_POLICY", "--deploy-policy"),
                         ("FCS_CONNECTIONS", "--connections"),
                         ("FCS_WORKFLOW_POLICY", "--workflow-policy")):
    if env.get(variable):
        fcs_args.extend([option, path(env[variable])])

tools = env.get("MCP_ALLOWED_TOOLS", " ".join(
    "io.flexserv.deployments." + leaf
    for leaf in ("vllm", "inspect", "status", "result", "cancel", "output", "stop")
)).split()
tools += ["gateway.operations." + name for name in ("status", "result", "cancel", "list")]
port = int(env["GATEWAY_PORT"])
if not 1 <= port <= 65535:
    raise SystemExit("GATEWAY_PORT must be between 1 and 65535")
gateway_args = [
    "--host", env["GATEWAY_BIND_HOST"], "--port", str(port), "--workers", "2",
    "--mcp-allowed-host", f"localhost:{port}",
    "--mcp-allowed-host", f"127.0.0.1:{port}",
    "--mcp-allowed-host", f"{nodes[0]}:{port}",
    "--log-file", str(root / "gateway.log"),
    "--trace-log-file", str(root / "trace.log"),
    "--startup-lock-file", str(root / "startup.lock"),
    "--readiness-lock-file", str(root / "ready.lock"),
    "--runtime-config-dir", str(root / "gateway-runtime"),
    "--fetch-state-dir", str(root / "fetch"),
    "--pub-model-repo", path(env["PUB_MODEL_REPO"]) if env.get("PUB_MODEL_REPO") else str(root / "public-models"),
    "--pri-model-repo", path(env["PRI_MODEL_REPO"]) if env.get("PRI_MODEL_REPO") else str(root / "private-models"),
]
config = {
    "schema_version": 1, "job_id": env["SLURM_JOB_ID"],
    "deployment_id": "interactive-" + env["SLURM_JOB_ID"],
    "run_parent": str(root / "runs"), "archive": {"root": str(root / "archive")},
    "run_not_after_utc": cutoff.isoformat().replace("+00:00", "Z"),
    "allocation": {"mode": "slurm", "srun": pin(env["SRUN_BIN"]),
                   "scontrol": pin(env["SCONTROL_BIN"]),
                   "srun_arguments": ["--overlap", "--exact", "--cpus-per-task=" + env.get("FCS_CPUS_PER_TASK", "1")]},
    "fcs_binary": pin(env["FCS_BIN"]), "gateway_binary": pin(env["GATEWAY_BIN"]),
    "fcs_arguments": fcs_args, "gateway_arguments": gateway_args,
    "instances": [{"instance_id": "node" + str(rank), "rank": rank,
                   "roles": ["worker", "head"] if rank == 0 else ["worker"],
                   "node": node, "bind": "0.0.0.0:0", "advertise_host": node}
                  for rank, node in enumerate(nodes)],
    "observer_principal": "gateway-observer",
    "observer_scopes": ["mcp:read", "node:read", "services:read"],
    "observer_owners": [env["MCP_PRINCIPAL"]],
    "delegated_principals": [env["MCP_PRINCIPAL"]], "delegated_scopes": ["mcp:read", "mcp:call"],
    "mcp": {"schema_version": 1, "operations": {}, "callers": [{
        "principal": env["MCP_PRINCIPAL"], "token_env": "FCS_MCP_TOKEN",
        "scopes": ["mcp:read", "mcp:call"], "allowed_tools": list(dict.fromkeys(tools))}]},
    "startup_timeout_seconds": int(env.get("STARTUP_TIMEOUT_SECONDS", "120")),
}
if env.get("GATEWAY_ROUTING_POLICY"):
    config["routing"] = json.loads(Path(path(env["GATEWAY_ROUTING_POLICY"])).read_text())
(root / "launch.json").write_text(json.dumps(config, indent=2) + "\n")
(root / "launch.json").chmod(0o600)
print(f"Gateway node: {nodes[0]}; FCS instances: {len(nodes)}; HTTP port: {port}")
print(f"Launch configuration and retained diagnostics: {root}")
PY

printf 'From your laptop: bash interactive-slurm/tunnel.sh USER@LOGIN_HOST %q %q\n' \
    "$first_node" "$GATEWAY_PORT"
printf 'Starting Bootstrap in the foreground; Ctrl-C stops this Run.\n'
exec "$BOOTSTRAP_BIN" --config "$session_dir/launch.json"
