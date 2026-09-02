# FlexServ HPC Archive Runner

This directory contains the shell runner used by the archived Tapis HPC app. The entry point is [app/run_flexserv.sh](app/run_flexserv.sh), with site-specific behavior implemented by scripts in [app/sites](app/sites).

The runner starts FlexServ in Apptainer, sets up model/cache paths, selects or receives a login-node port, creates reverse SSH tunnels, and prints the access URL and token.

## Common Usage

Run from `hpc_archive/app`:

```bash
cd /path/to/FlexServ-Deployer/hpc_archive/app
./run_flexserv.sh --site tacc --secret flexserv
```

Useful options:

```bash
./run_flexserv.sh \
    --site <tacc|osc> \
    --flexserv-port 8000 \
    --secret flexserv \
    --model-name Qwen/Qwen3.5-0.8B \
    --device auto \
    --dtype bfloat16
```

The selected login-node port is printed in the logs as `FlexServ login-node port`.

## Pulling The Apptainer Image

On sites where the FlexServ `.sif` image is not already available on shared storage, pull the latest FlexServ OCI/Docker image into an Apptainer image file and point `APPTAINER_IMAGE` at it.

```bash
mkdir -p "$HOME/flexserv" "$HOME/flexserv_cache"
export APPTAINER_CACHEDIR="$HOME/flexserv_cache"

export FLEXSERV_IMAGE_REF="docker://zhangwei217245/flexserv-transformers:1.4.6"

apptainer pull --force "$HOME/flexserv/flexserv.sif" "$FLEXSERV_IMAGE_REF"
export APPTAINER_IMAGE="$HOME/flexserv/flexserv.sif"
```

Equivalently, from the directory where you want the image:

```bash
apptainer pull flexserv.sif docker://zhangwei217245/flexserv-transformers:1.4.6
export APPTAINER_IMAGE="$PWD/flexserv.sif"
```

Verify the image before launching:

```bash
ls -lh "$APPTAINER_IMAGE"
apptainer inspect "$APPTAINER_IMAGE" >/dev/null
```

Pull on the same CPU architecture as the job nodes, or use the image/tag intended for that architecture.

## TACC

TACC is the default site:

```bash
./run_flexserv.sh --site tacc --secret flexserv
```

The TACC site script uses TAP functionality from `/share/doc/slurm/tap_functions` to:

- load the TACC Apptainer environment
- obtain a TAP token when a FlexServ secret is not provided
- allocate and release the login-node port
- use TAP-provided certificate material when HTTPS is enabled
- create reverse SSH tunnels through `login1` through `login4`

HTTPS can be enabled on TACC with:

```bash
./run_flexserv.sh --site tacc --secret flexserv --enable-https
```

## OSC

For OSC clusters, use the OSC site script:

```bash
cd /path/to/FlexServ-Deployer/hpc_archive/app
./run_flexserv.sh --site osc --secret flexserv
```

The OSC site script is intended for Pitzer, Ascend, and Cardinal. It:

- uses `apptainer` from `PATH`
- generates a FlexServ token when one is not provided
- selects a login-node port in the `60000-65000` range
- detects the OSC cluster from Slurm
- detects the OSC project from `OSC_ACCOUNT_NAME` or `SLURM_JOB_ACCOUNT`
- uses OSC DNS to determine how many login nodes to target
- creates reverse SSH tunnels to the OSC login nodes

OSC uses the allocation from `OSC_ACCOUNT_NAME` when set; otherwise it uses the current job's `SLURM_JOB_ACCOUNT`. The project name is normalized to uppercase. Set `OSC_ACCOUNT_NAME` when a different allocation should be used:

```bash
export OSC_ACCOUNT_NAME=PAA1234  # replace with the desired OSC allocation
./run_flexserv.sh --site osc --secret flexserv
```

OSC keeps private resources under `/fs/scratch/<PROJECT>/$USER/flexserv` and shared resources under `/fs/scratch/<PROJECT>/flexserv`:

```bash
APPTAINER_CACHEDIR=/fs/scratch/<PROJECT>/flexserv/apptainer_cache
PRI_MODEL_HOST=/fs/scratch/<PROJECT>/$USER/flexserv/models/private
PUB_MODEL_HOST=/fs/scratch/<PROJECT>/flexserv/models/public
APPTAINER_IMAGE=/fs/scratch/<PROJECT>/flexserv/flexserv.sif
```

The resolved paths are printed in the runner logs, making it possible to verify which allocation is being used.

If the default `APPTAINER_IMAGE` does not exist yet, pull the image first as shown in [Pulling The Apptainer Image](#pulling-the-apptainer-image), or set `APPTAINER_IMAGE` to another `.sif` path.

Models, container images, and Apptainer caches can be large, so check the quota and purge policy for your OSC scratch project. If you have another suitable project or shared filesystem, override the paths before launching:

```bash
export OSC_ACCOUNT_NAME=PAA1234
export APPTAINER_CACHEDIR=/path/to/cache
export PRI_MODEL_HOST=/path/to/private/models
export PUB_MODEL_HOST=/path/to/public/models
export APPTAINER_IMAGE=/path/to/flexserv.sif
```

OSC Open OnDemand manages the browser-facing HTTPS connection, so `--enable-https`, `FLEXSERV_CERTFILE`, and `FLEXSERV_KEYFILE` are not needed for OSC. FlexServ runs over HTTP behind the Open OnDemand proxy. If `--enable-https` is supplied, the OSC site script prints a warning and disables it.

You can test from an OSC login node with the printed login-node port:

```bash
export LOGIN_PORT=<printed-login-port>

curl -H "x-flexserv-secret: flexserv" "http://localhost:${LOGIN_PORT}/v1/flexserv/health"
curl -s \
    -H "x-flexserv-secret: flexserv" \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"What is 2+2? Answer in one sentence."}]}' \
    "http://localhost:${LOGIN_PORT}/v1/chat/completions"
```

## Creating A Site Implementation

Use [app/sites/template.sh](app/sites/template.sh) as the starting point for a new site:

```bash
cp app/sites/template.sh app/sites/<site>.sh
./run_flexserv.sh --site <site> --secret flexserv
```


The main runner sources the selected site script and calls these functions:

- `concrete_setup_environment`
- `concrete_prepare_apptainer`
- `concrete_setup_cert_tls`
- `concrete_setup_random_token`
- `concrete_setup_login_port`
- `concrete_setup_reverse_tunnels`
- `concrete_cleanup_site_resources`

Keep site-specific policy in the site script: module names, model/cache defaults, login-node naming, port selection, TLS source, token source, and cleanup behavior. The common runner should stay focused on validating arguments, preparing FlexServ settings, and starting the container.

At minimum, a new site should decide:

- where to store model directories, Apptainer cache, and the Apptainer image
- whether Apptainer is already in `PATH` or must be loaded through a module
- how to choose and reserve the login-node port
- how to enumerate login nodes for reverse SSH tunnels
- whether HTTPS is supported, and where cert/key files come from
- how to generate a fallback FlexServ token


The template provides general implementations for random token generation, optional cert/key handling, login-port selection, and reverse SSH tunnel setup; site-specific code usually only needs to adjust environment paths, Apptainer preparation, and login-node discovery.

For basic cases, the template includes a simple `LOGIN_NODE_PREFIX` plus `LOGIN_NODE_COUNT` implementation for generating login-node names. Sites with different naming, discovery, or routing rules can replace `generate_login_nodes` with their own logic.
