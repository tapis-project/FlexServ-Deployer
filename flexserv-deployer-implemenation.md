# FlexServ Deployer

`flexserv-deployer` is a Rust library and CLI for creating and managing
FlexServ model serving deployments on Tapis Pods and Tapis Jobs.

The CLI is the most complete user interface. It owns command parsing, saved
configuration, output formatting, ID-prefix resolution, and direct list/status
calls to the Tapis SDKs. Creation and lifecycle operations are delegated to the
library deployment types.


## Build And Run

```bash
cargo build
cargo run --bin flexserv-deployer -- --help
cargo run --bin flexserv-deployer -- manage show
```

After installation, the examples below assume the executable is available as
`flexserv-deployer`.

## CLI Shape

The CLI has one top-level command:

```text
flexserv-deployer <manage|pod|job> ...
```

The CLI does not use global deployment options or environment-backed command
options. Authentication, Hugging Face token, and output format are saved through
`manage` commands. Model and backend are create-time options on `pod create` and
`job create`.

Command aliases:

- `manage` can also be invoked as `m`
- `pod` can also be invoked as `p`, `pods`, or `ps`
- `job` can also be invoked as `j`, `jobs`, or `js`

## Configuration Model

The CLI resolves effective configuration in this order:

1. Command arguments for the command being run
2. Saved manager config
3. JWT-derived tenant and user values, where required
4. Built-in defaults, where available

The saved config path is:

```text
~/.flexserv-deployer/config.json
```

Saved config fields:

```json
{
  "tapis_token": "...",
  "hf_token": "...",
  "output": "text"
}
```

Tokens are stored as plain JSON values. `manage show` currently does not mask
them; it prints raw token strings when they are available. Treat the config file
and `manage show` output as sensitive.

The Tapis username is decoded from the JWT payload claim `tapis/username`.
The Tapis tenant URL is decoded from `tapis/tenant_id` as
`https://<tenant_id>.tapis.io`. Malformed JWTs or tokens without those claims
fail before Tapis clients are built.

## `manage` Commands

`manage` commands update local saved configuration. They do not call Tapis APIs.

### `manage authenticate <JWT_TOKEN>`

Aliases: `manage auth`, `manage login`

Implementation:

1. Reads the existing saved config with `read_saved_config()`.
2. Starts from `SavedConfig::default()` if no file exists.
3. Stores the provided token in `tapis_token`.
4. Writes pretty JSON through `write_saved_config()`.
5. Prints suggested next commands.

Example:

```bash
flexserv-deployer manage authenticate <JWT_TOKEN>
```

### `manage hf-token <HF_TOKEN>`

Alias: `manage hf`

Implementation:

1. Reads or initializes saved config.
2. Stores the value in `hf_token`.
3. Writes the updated config.

Example:

```bash
flexserv-deployer manage hf-token <HF_TOKEN>
```

### `manage output [OUTPUT_FORMAT]`

Implementation:

1. Reads or initializes saved config.
2. Parses `text` or `json` with `OutputFormat::from_str`.
3. Stores the output format.
4. Writes the updated config.

Example:

```bash
flexserv-deployer manage output json
```

### `manage show`

Aliases: `manage config`, `manage whoami`

Implementation:

1. Resolves the effective tenant URL, Tapis user, and output.
2. Resolves token and HF token values.
3. Decodes the token `exp` claim and prints `Expires in ...` or `Expired`.
4. Prints the config file path.

This command exercises the same resolution path as deployment commands, so it is
a good first check before creating resources.

## Pod Commands

Pod commands manage FlexServ deployments backed by Tapis Pods and Tapis
Volumes. They use `FlexServPodDeployment` from `src/deployment/pod.rs`.

### Pod IDs And Volume IDs

For new deployments, pod and volume IDs are deterministic:

- If `--deployment-id` is provided, it is normalized to lowercase ASCII
  alphanumeric characters.
- Otherwise the suffix is the first 12 base62 characters of a SHA-256 hash of
  `tapis_user`, `tenant_url`, `model`, and `backend`.

The final IDs are:

```text
p<suffix>  pod id
v<suffix>  volume id
```

For existing pod operations, the CLI accepts a full pod id or a unique prefix.
It lists visible pods, finds prefix matches, and errors if the prefix is missing
or ambiguous. If `--volume-id` is not provided, it infers the volume by replacing
a leading `p` with `v`.

### `pod create`

Creates or replaces the pod and creates or reuses its backing volume.

Options:

| Option | Purpose | Default |
| --- | --- | --- |
| `--model <MODEL_ID>` | Hugging Face model id. | `openai-community/gpt2` |
| `--backend <BACKEND>` | Backend: `transformers`, `vllm`, `sglang`, or `trtllm`. | `transformers` |
| `--deployment-id <ID>` | Stable external id used to derive pod/volume ids. | deployment hash |
| `--volume-size-mb <MB>` | Tapis volume size. | `10240` |
| `--image <IMAGE>` | Pod image. | `zhangwei217245/flexserv-transformers:1.4.6` |
| `--cpu-request <MILLICPUS>` | CPU request. | `1000` |
| `--cpu-limit <MILLICPUS>` | CPU limit. | `2000` |
| `--mem-request-mb <MB>` | Memory request. | `4096` |
| `--mem-limit-mb <MB>` | Memory limit. | `8192` |
| `--gpus <COUNT>` | GPU count. | `0` |
| `--flexserv-secret <SECRET>` | FlexServ auth token. | empty |

CLI implementation in `run_pod()`:

1. Builds a `FlexServInstance` with `flexserv_server(args.model, args.backend)`.
2. Converts CLI args into `PodDeploymentOptions`.
3. Creates `FlexServPodDeployment::with_options(server, token, options)`.
4. Calls `create().await`.
5. Prints `DeploymentResult::PodResult` as text or JSON.

Example:

```bash
flexserv-deployer \
  pod create \
  --model openai-community/gpt2 \
  --backend transformers \
  --volume-size-mb 20480 \
  --cpu-request 2000 \
  --mem-limit-mb 16384
```

### `pod list`

Lists pods visible to the authenticated Tapis user.

Options:

| Option | Purpose | Default |
| --- | --- | --- |
| `--limit <N>` | Maximum number of rows to print. Negative values print all rows. | `25` |

Implementation:

1. Builds a `TapisPods` client with `pods_client()`.
2. Calls `client.pods.list_pods().await`.
3. Applies the limit locally.
4. In JSON mode, pretty-prints the raw SDK models.
5. In text mode, prints `POD ID`, `DESCRIPTION`, `STATUS`, `URL`, and `IMAGE`.

Example:

```bash
flexserv-deployer pod list --limit 10
```

### `pod monitor <ID_OR_PREFIX>`

Fetches current pod state.

Implementation:

1. Resolves the supplied id or prefix against `pod list`.
2. Infers or accepts the volume id.
3. Builds `FlexServPodDeployment::from_existing(...)`.
4. Calls `monitor().await`.
5. `monitor()` calls `get_pod` and then tries `get_volume`.
6. Returns a `PodResult` with status, URL, pod debug info, and volume debug info.

Example:

```bash
flexserv-deployer pod monitor pabc123
```

### `pod start <ID_OR_PREFIX>`

Requests that Tapis start a stopped pod.

Implementation:

1. Resolves the id or prefix.
2. Builds an existing deployment handle.
3. Calls `client.pods.start_pod(&pod_id)`.
4. Converts the returned pod model into a `PodResult`.

Example:

```bash
flexserv-deployer pod start pabc123
```

### `pod stop <ID_OR_PREFIX>`

Requests that Tapis stop a running pod.

Implementation:

1. Resolves the id or prefix.
2. Builds an existing deployment handle.
3. Calls `client.pods.stop_pod(&pod_id)`.
4. Converts the returned pod model into a `PodResult`.

Example:

```bash
flexserv-deployer pod stop pabc123
```

### `pod terminate <ID_OR_PREFIX>`

Aliases: `pod delete`, `pod rm`

Deletes the pod. It does not delete the associated volume.

Implementation:

1. Resolves the id or prefix.
2. Builds an existing deployment handle.
3. Calls `delete_pod`.
4. Logs delete errors but still returns a synthetic `PodResult`.
5. The synthetic result has `status = "TERMINATED"` and no URL.

Example:

```bash
flexserv-deployer pod terminate pabc123
```

## Job Commands

Job commands manage HPC deployments through Tapis Jobs. They use
`FlexServHPCDeployment` from `src/deployment/hpc.rs`.

Existing job operations accept a full UUID or a unique prefix. The CLI lists
recent jobs, resolves the prefix locally, and errors on missing or ambiguous
matches.

### `job create`

Submits a new FlexServ HPC job.

Arguments:

| Argument | Purpose |
| --- | --- |
| `<APP_ID>` | Tapis app id. |
| `<APP_VERSION>` | Tapis app version. |
| `<EXEC_SYSTEM_ID>` | Execution system id. |
| `<QUEUE>` | Execution system logical queue. |
| `<MAX_MINUTES>` | Max job runtime. |
| `<ALLOCATION>` | HPC allocation. |

Options:

| Option | Purpose | Default |
| --- | --- | --- |
| `--model <MODEL_ID>` | Hugging Face model id. | `openai-community/gpt2` |
| `--backend <BACKEND>` | Backend: `transformers`, `vllm`, `sglang`, or `trtllm`. | `transformers` |

CLI implementation in `run_job()`:

1. Builds `HpcDeploymentOptions` from CLI args.
2. Builds a `FlexServInstance` with `flexserv_server(args.model, args.backend)`.
3. Creates `FlexServHPCDeployment::new(server, token, options)`.
4. Calls `create().await`.
5. Prints `DeploymentResult::HPCResult`.

Example:

```bash
./flexserv-deployer \
  job create \
  --model Qwen/Qwen3.5-0.8B \
  --backend transformers \
  FlexServ-1.4.0 \
  1.4.0 \
  vista-tapis \
  gh-dev \
  60 \
  TACC-ACI-CIC
```

### `job list`

Lists recent jobs visible to the authenticated Tapis user.

Options:

| Option | Purpose | Default |
| --- | --- | --- |
| `--limit <N>` | Maximum number of rows to print. Negative values print all rows. | `25` |

Implementation:

1. Builds a Tapis Jobs `Configuration`.
2. Calls `jobs_api::get_job_list` with:
   - no API-side limit when `--limit` is negative
   - otherwise an API-side limit matching `--limit`
   - order `created(desc)`
3. Applies the requested local limit.
4. In JSON mode, pretty-prints the raw SDK DTOs.
5. In text mode, prints `JOB UUID`, `NAME`, and `STATUS`.

Example:

```bash
flexserv-deployer job list --limit 20
```

### `job monitor <UUID_OR_PREFIX>`

Aliases: `job status`, `job view`

Fetches current job state and, when possible, the running FlexServ endpoint.

Implementation:

1. Resolves the UUID or prefix against recent jobs.
2. Builds `FlexServHPCDeployment::from_existing(token, uuid)`.
3. Sets `tenant_url` on the deployment because existing handles do not include a
   full `FlexServInstance`.
4. Calls `get_job_status`.
5. Calls `get_job`.
6. If status is `RUNNING`, tries to fetch the job output log from the Tapis
   Files API at:

   ```text
   /files/content/<exec_system_id>/<exec_system_output_dir>/tapisjob.out
   ```

7. Reads up to five pages using the `more` header.
8. Parses a line containing both `FlexServ address:` and `FlexServ token:`.
9. Returns `hpc_url` and `flexserv_token` only when those fields are found.

Example:

```bash
flexserv-deployer job monitor 550e8400
```

### `job start <UUID_OR_PREFIX>`

Alias: `job resubmit`

Resubmits a previous Tapis job.

Implementation:

1. Resolves the UUID or prefix.
2. Builds an existing deployment handle.
3. Calls `jobs_api::resubmit_job`.
4. Uses the returned job UUID when present, otherwise falls back to the prior
   UUID.
5. If the resulting job is already `RUNNING`, tries to parse endpoint metadata
   from logs.

Example:

```bash
flexserv-deployer job start 550e8400
```

### `job cancel <UUID_OR_PREFIX>`

Aliases: `job stop`, `job terminate`

Cancels a Tapis job.

Implementation:

1. Resolves the UUID or prefix.
2. Builds an existing deployment handle.
3. Calls `jobs_api::cancel_job`.
4. Returns an `HPCResult` containing the UUID and empty status/job/endpoint
   fields.

Example:

```bash
flexserv-deployer job cancel 550e8400
```

## Output Formats

Text mode prints compact, purpose-built summaries.

Pod result:

```text
Pod deployment result:
  pod_id:     p...
  volume_id:  v...
  status:     RUNNING
  pod_url:    https://...
  tapis_user: username
  tapis_tenant: https://tacc.tapis.io
  model_id:   openai-community/gpt2
```

HPC result:

```text
HPC deployment result:
  job_uuid:   ...
  status:     RUNNING
  hpc_url:    https://...
  flexserv_secret: ...
```

JSON mode serializes `DeploymentResult` or raw list DTOs with
`serde_json::to_string_pretty`.
