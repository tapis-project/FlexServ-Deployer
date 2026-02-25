# FlexServ Deployer

A Rust library and web server for deploying FlexServ on Tapis Pods or HPC systems via Tapis Jobs, with various backends (Transformers, vLLM, SGLang, TRT-LLM).

## Production Readiness

| Area | Status | Notes |
|------|--------|--------|
| **Pod deployment** | Ready | Create, start, stop, terminate, monitor; pod-side model download from Hugging Face; volume mounts; error mapping and `Display`/`Error` for `DeploymentError`. |
| **HPC deployment** | Not implemented | `FlexServHPCDeployment` methods are `todo!()`. |
| **Error handling** | Good | `DeploymentError` has `Display` and `std::error::Error`; TAPIS errors mapped; cleanup on pod create failure. |
| **Logging** | Good | Uses `log::warn!` / `log::error!` for deployment; no secrets in log messages. |
| **Config** | Env-based | `FLEXSERV_SECRET`, `HF_TOKEN` from env; image and resources hardcoded (`tapis/flexserv:1.0`, 10GB volume, 2 CPU / 8GB). Make configurable for multi-tenant or sizing. |
| **Input validation** | Minimal | No validation of `tenant_url`, `tapis_user`, or `model_id` format. Callers should validate. |
| **Binary (main.rs)** | Stub | Only `/health` and `/models`; no deployment API. Use the library directly or add HTTP endpoints for create/start/stop/terminate/monitor. |
| **Tests** | Good | Unit tests for deployment, server, backend, base62; integration tests for pod ops (require TAPIS credentials). |

**Recommendations for production:** (1) Add validation for `FlexServInstance` (URL format, non-empty user/model if required). (2) Make image, volume size, and resources configurable. (3) Implement or remove HPC deployment. (4) Add deployment HTTP API to the binary if needed. (5) Optionally use a shared `tokio::Runtime` instead of creating one per call.

## Library Components

### Backend Module (`backend.rs`)
- **Backend enum**: Supports Transformers, vLLM, SGLang, and TRT-LLM
- **BackendParameterSet**: Built parameter set (command, params, env) for a backend
- **BuildBackendParameterSet** trait on `Backend`: `build_params_for_pod(server)` and `build_params_for_hpc(server)` return a `BackendParameterSet`
- **Parameter set builders** (each has `.build()` → `BackendParameterSet`):
  - `TransformersParameterSetBuilder`
  - `VLlmParameterSetBuilder`
  - `SGLangParameterSetBuilder`
  - `TrtLlmParameterSetBuilder`

### Server Module (`server.rs`)
- **FlexServInstance**: Server instance configuration with tenant URL, user, model info, and backend

### Deployment Module (`deployment.rs`)
- **FlexServDeployment trait**: Common interface for deployment operations
- **FlexServPodDeployment**: Pod-based deployment implementation
- **FlexServHPCDeployment**: HPC-based deployment implementation
- **DeploymentResult & DeploymentError**: Result types for deployment operations

### Binary (`main.rs`)
A web server built with Actix-web that exposes REST APIs:
- Runs on `127.0.0.1:8080`

## Building

### Build the library
```bash
cargo build --lib
```

### Build the server binary
```bash
cargo build --bin flexserv-deployer-server
```

### Build everything
```bash
cargo build
```

## Running

### Run the web server
```bash
cargo run --bin flexserv-deployer-server
```

The server will start on `http://127.0.0.1:8080`

## Testing

### Run all tests (unit + integration)
```bash
cargo test
```

### Run only unit tests
```bash
cargo test --lib
```

### Run only integration tests (require TAPIS_TENANT_URL, TAPIS_TOKEN; pod tests also need POD_ID, VOLUME_ID for start/stop/terminate/monitor)
```bash
cargo test --test pod_create_integration -- --nocapture
```

### Run tests with output
```bash
cargo test -- --nocapture
```

<!-- ## API Endpoints

- `GET /search` - Search for models
- `POST /info` - Get model information -->

## Using the library

This crate is a library first: you use it from your own Rust code to deploy and manage FlexServ pods (and eventually HPC jobs). The binary `flexserv-deployer-server` is a stub; deployment is done via the library.

### Add the dependency

In your project’s `Cargo.toml`:

```toml
[dependencies]
flexserv-deployer = { path = "../FlexServ-Deployer" }  # path relative to your project
# Or once published: flexserv-deployer = "0.1"
```

The deployer uses the Tapis Pods client (path to tapis-rust-sdk/tapis-pods for now; see Cargo.toml). When [tapis-sdk](https://crates.io/crates/tapis-sdk) publishes a version that includes the `source_id` fix for volume mounts, you can switch to `tapis-sdk = "0.1"` and use `tapis_sdk::pods::*`.

### Imports

```rust
use flexserv_deployer::{
    Backend,
    DeploymentError,
    DeploymentResult,
    FlexServDeployment,
    FlexServInstance,
    FlexServPodDeployment,
    PodDeploymentOptions,
};
```

Optional: `FlexServHPCDeployment`, backend types (`BackendParameterSet`, `TransformersParameterSetBuilder`, etc.).

### Create a new pod deployment

1. Build a **server instance** (tenant, user, model, backend).
2. Build a **pod deployment** with `FlexServPodDeployment::new(server, tapis_token)`.
3. Call **`deployment.create()`**; the pod will create a volume and pod, then download the model from Hugging Face at startup.

```rust
use flexserv_deployer::{
    Backend, DeploymentError, DeploymentResult, FlexServDeployment, FlexServPodDeployment,
    FlexServInstance,
};

fn main() -> Result<(), DeploymentError> {
    let tenant_url = std::env::var("TAPIS_TENANT_URL").expect("TAPIS_TENANT_URL");
    let tapis_token = std::env::var("TAPIS_TOKEN").expect("TAPIS_TOKEN");

    let server = FlexServInstance::new(
        tenant_url,
        "your_tapis_username".to_string(),
        "openai-community/gpt2".to_string(),  // model_id; pod downloads from HF
        None,  // model_revision
        None,  // hf_token (optional; for gated/private HF models)
        None,  // default_embedding_model
        Backend::Transformers {
            command_prefix: vec!["python".to_string()],
        },
    );

    let mut deployment = FlexServPodDeployment::new(server, tapis_token);
    let result = deployment.create()?;

    match result {
        DeploymentResult::PodResult {
            pod_id,
            volume_id,
            pod_url,
            tapis_user,
            tapis_tenant,
            model_id,
            ..
        } => {
            println!("Pod created: {}", pod_id);
            println!("Volume: {}", volume_id);
            if let Some(ref url) = pod_url {
                println!("Pod URL (for inference): {}", url);
            }
            // Auth token for the pod API: model_id with "/" replaced by "_", e.g. openai-community_gpt2
            let auth_token = model_id.replace('/', "_");
            println!("Use Authorization: Bearer {} when calling the pod", auth_token);
        }
        DeploymentResult::HPCResult { .. } => unreachable!("pod deployment returns PodResult"),
    }

    Ok(())
}
```

### Passing options (volume size, image, resources, secrets, deployment id)

Use **`FlexServPodDeployment::with_options(server, tapis_token, options)`** and **`PodDeploymentOptions`** to pass values at the call site instead of env or hardcoded defaults:

- **`deployment_id`** – Optional deployment id (e.g. UUID from MLHub). When set, **pod_id** and **volume_id** are derived from it (normalized to lowercase alphanumeric, e.g. UUID without dashes), so you can create **multiple pods for the same model**. When unset, ids are derived from server config (one pod per user+model).
- **`volume_size_mb`** – Volume size in MB (default 10240 = 10 GB).
- **`image`** – Container image (default `"tapis/flexserv:1.0"`).
- **`cpu_request`** / **`cpu_limit`** – CPU in millicpus, 1000 = 1 CPU (defaults 1000 / 2000).
- **`mem_request_mb`** / **`mem_limit_mb`** – Memory in MB (defaults 4096 / 8192).
- **`gpus`** – Number of GPUs (default 0).
- **`flexserv_secret`** – Optional secret prepended to the pod auth token; if `None`, uses `FLEXSERV_SECRET` env.

User (TAPIS username) and **`hf_token`** (for gated/private Hugging Face models) are set via **`FlexServInstance::new(..., hf_token, ...)`**; if `hf_token` is `None`, the pod falls back to `HF_TOKEN` env.

```rust
use flexserv_deployer::{
    Backend, DeploymentError, FlexServDeployment, FlexServInstance,
    FlexServPodDeployment, PodDeploymentOptions,
};

let server = FlexServInstance::new(
    tenant_url,
    "your_tapis_username".to_string(),
    "openai-community/gpt2".to_string(),
    None,  // model_revision
    None,  // hf_token (optional; for gated/private HF models)
    None,
    Backend::Transformers { command_prefix: vec!["python".to_string()] },
);

let options = PodDeploymentOptions {
    deployment_id: Some(mlhub_deployment_uuid.clone()),  // e.g. from MLHub; enables multiple pods per model
    volume_size_mb: Some(20 * 1024),  // 20 GB
    cpu_request: Some(2000),           // 2 CPU
    mem_limit_mb: Some(16384),         // 16 GB RAM
    ..Default::default()
};
let mut deployment = FlexServPodDeployment::with_options(server, tapis_token, options);
let result = deployment.create()?;
// result.pod_id / volume_id will be p{uuid_no_dashes} and v{uuid_no_dashes}
```

### Manage an existing deployment

If you already have `pod_id` and `volume_id` (e.g. from a previous `create()` or from your own storage), use **`from_existing`**:

```rust
let server = FlexServInstance::new(/* ... */);
let deployment = FlexServPodDeployment::from_existing(
    server,
    tapis_token,
    pod_id,
    volume_id,
);

deployment.start()?;   // start the pod
deployment.monitor()?; // get status
deployment.stop()?;    // stop the pod
deployment.terminate()?; // delete pod and volume
```

All of these return `Result<DeploymentResult, DeploymentError>`. For pod deployments the result is always `DeploymentResult::PodResult { pod_id, volume_id, pod_url, ... }`.

### Call the pod (inference)

The library does **not** implement HTTP calls to the running pod. Use any HTTP client (e.g. `reqwest`) and the **pod URL** and **auth token** from `create()`:

- **Pod URL**: use the `pod_url` from `DeploymentResult::PodResult` (e.g. `https://<pod-id>.pods.tacc.tapis.io`). Use HTTPS and no port.
- **Auth**: send `Authorization: Bearer <token>` where `<token>` is the model dir name, e.g. `openai-community_gpt2` (i.e. `model_id` with `/` replaced by `_`). Optionally also send `X-FlexServ-Secret` with the same value if your ingress forwards it.

Example endpoints: `GET /v1/flexserv/health`, `GET /v1/models`, `POST /v1/completions` (for GPT-2 use `"prompt"` and `"model": "/app/models/openai-community_gpt2"`). See `examples/call_pod.rs` and the curl block in that file.

### Error handling

`DeploymentError` implements `std::error::Error` and `Display`. You can use `?`, `map_err`, or `match` in your code; no need to depend on the library’s logging. Initialize a logger (e.g. `env_logger::init()`) if you want the library’s `log::warn!` / `log::error!` output.

### Examples in this repo

- **`create_pod`**: create a new pod (volume + pod, model downloaded in pod). Run with `TAPIS_TENANT_URL`, `TAPIS_TOKEN`; optional `HF_TOKEN`, `FLEXSERV_NO_MODEL=1`.
- **`terminate_pod`**: delete an existing pod and volume. Needs `POD_ID`, `VOLUME_ID` (and tenant/token).
- **`call_pod`**: HTTP client example (health, models, completions) using `reqwest`; set `POD_URL` and `FLEXSERV_TOKEN`.

Run with: `cargo run --example create_pod` (and similarly for `terminate_pod`, `call_pod`).

## Dependencies

- **actix-web**: Web server (binary only)
- **tokio**: Async runtime
- **serde** / **serde_json**: Serialization
- **reqwest**: HTTP client (examples and optional use from your code)
- **tapis_pods**: TAPIS Pods and Volumes API (path to tapis-rust-sdk; see Cargo.toml)
- **log** / **env_logger**: Logging
