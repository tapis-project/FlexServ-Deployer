# Minimal interactive Slurm launch for FlexServ 1.5

This standalone directory launches the `feature/1.5.0` Gateway and current v0.13
FCS through the existing `flexserv-bootstrap`. It does not use the old deployer
library, old FCS registration endpoints, Tapis or a batch-job wrapper.

The first hostname returned by `scontrol show hostnames "$SLURM_JOB_NODELIST"`
hosts Gateway and Bootstrap. Every allocated node, including the first, runs one
FCS through Bootstrap's `srun` step. Ranks start at zero; all FCS instances have
the `worker` role and rank zero also has `head`. Registration means Bootstrap's
trusted manifest plus Gateway's active Streamable HTTP discovery over job mTLS;
FCS does not POST a registration or heartbeat.

## Prepare once

Use Linux builds matching the compute nodes, Python 3, Bash, `srun` and `scontrol`.
Binaries, scripts, Pack, site policies and `STATE_ROOT` must be visible at the
same paths on every node. Load any required site modules before launching.
With Rust 1.99 and the two source checkouts already available on shared storage:

```bash

# Confirm GCC/G++ locations
which gcc
which g++

# if cannot be found, install gcc and g++ using your package manager, e.g., apt or yum
# or load a module providing GCC/G++ if your site uses environment modules

# module load gcc

# Explicitly select GCC/G++ for native builds
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"

# Explicitly select GCC/G++ for AWS-LC
export AWS_LC_SYS_CC="$CC"
export AWS_LC_SYS_CXX="$CXX"


cd /shared/flexserv  # Parent directory containing both source checkouts.
# FlexServ checkout: feature/1.5.0
cargo build --release --manifest-path FlexServ/flexserv/gateway/Cargo.toml
cargo build --release --manifest-path FlexServ/flexserv/bootstrap/Cargo.toml
# FCS checkout: current v0.13 implementation on feature/new_fcs
cargo build --release --manifest-path fcs-deploy-node-v013/Cargo.toml --bin mcp-bashkit-server
```

Supply an operator-approved v1alpha2 Pack payload containing `checksums.sha256`.
The launcher pins the current inventory and Bootstrap pins binary bytes. To
require a previously reviewed inventory, set `FCS_PACK_SHA256` to its SHA-256.
This is operator approval of the selected files, not publisher-signature checking.
Unsigned demo mode is not used on Slurm. The default tool grants match the seven
public vLLM wrappers; set `MCP_ALLOWED_TOOLS` for another Pack. Site deployment
permissions still authorize the principal and tools independently.

## Launch inside an interactive allocation

Obtain the allocation using your site's normal command, for example
`salloc -N 2 -t 01:00:00` with the required account/partition options. Then:

```bash
cp interactive-slurm/env.example interactive-slurm/site.env
# Edit site.env to use your shared binaries, approved Pack and actual host policy.
source interactive-slurm/site.env

read -r -s -p 'MCP bearer token: ' FCS_MCP_TOKEN; echo
export FCS_MCP_TOKEN
read -r -s -p 'Gateway administrator secret (different): ' FLEXSERV_ADMIN_SECRET; echo
export FLEXSERV_ADMIN_SECRET

bash interactive-slurm/launch.sh
```

The launcher moves itself to the first node with a small overlapping `srun` step
if the allocation shell is elsewhere. Bootstrap starts another overlapping step
for all FCS instances. The site must permit overlapping steps and shared files.
Both steps request one CPU per task by default; `FCS_CPUS_PER_TASK` changes the
FCS request. GPU allocation and binding remain the site's Slurm responsibility.

The Run ends at the earlier of `RUN_SECONDS` (default 3600) or ten seconds before
the scheduler's finite EndTime. Keep the launch command in the foreground.
Ctrl-C asks Bootstrap to clean up its owned processes and ephemeral Run keys.
The printed private session directory retains launch configuration, Gateway logs
and diagnostic archives for inspection. Remove that specific directory after
reviewing it and confirming shutdown; shared model repositories are not removed.

This starts the control plane, not a vLLM model. Backend deployment and inference
route approval are later manual calls; `GATEWAY_ROUTING_POLICY` optionally supplies
the existing site-approved routing contract. No model download, Python install,
port retry, restart recovery or PID adoption is performed here.

## Forward Gateway to your laptop

In a second terminal **on your laptop**, use the Gateway node printed at launch:

```bash
bash interactive-slurm/tunnel.sh USER@LOGIN_HOST COMPUTE_NODE_0 8000
curl -f http://127.0.0.1:8000/healthz/ready
```

The tunnel binds laptop loopback only. It connects through the login host to the
Gateway HTTP listener on the first node; that login-to-compute hop must be allowed
by the site. Internal Gateway-to-FCS connections continue to use job mTLS.
For an entirely SSH-encrypted route where compute-node SSH is permitted, bind
Gateway to `127.0.0.1` and use
`ssh -J USER@LOGIN_HOST -N -L 8000:127.0.0.1:8000 USER@COMPUTE_NODE_0` instead.

Use `http://localhost:8000/mcp` with an MCP Streamable HTTP client and
`Authorization: Bearer <FCS_MCP_TOKEN>` for tool discovery and calls. The separate
administrator secret is not an MCP credential. The helper's optional fourth
argument selects a different laptop port; in that case, override the MCP client's
Host header to `localhost:<GATEWAY_PORT>` to match the approved authority.
Gateway operation status/result/cancel/list controls are enabled for the same
principal. Live WebSocket subscriptions and browser-Origin configuration are not
enabled by this minimal launcher.

## Local checks

`bash -n` checks both scripts; the embedded Python is syntax-checked separately.
No real Slurm allocation, SSH tunnel or HPC startup has been executed by this
change. Actual site configuration and first deployment will be checked on HPC.
