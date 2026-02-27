## FlexServ Deployer

Rust library for deploying FlexServ model servers onto Tapis Pods (and, in the future, HPC via Jobs).

The **library** is the main product. The Actix-web binary is a thin wrapper and intentionally minimal.

---

## Library Overview

### Core modules

- **Backend module** (`backend.rs`)
  - `Backend` enum: `Transformers`, `VLlm`, `SGLang`, `TrtLlm`
  - `BackendParameterSet` (command_prefix + params + env)
  - `BuildBackendParameterSet` trait:
    - `build_params_for_pod(&self, server)`
    - `build_params_for_hpc(&self, server)`
  - Per-backend builders: `TransformersParameterSetBuilder`, `VLlmParameterSetBuilder`, `SGLangParameterSetBuilder`, `TrtLlmParameterSetBuilder`

- **Server module** (`server.rs`)
  - `FlexServInstance`: tenant URL, user, model id, optional revision/HF token, backend
  - `FlexServInstanceBuilder`: validated builder
  - `ModelConfig`, `TapisConfig`, `ValidationError`

- **Deployment module** (`deployment/mod.rs`, `deployment/pod.rs`, `deployment/hpc.rs`)
  - `FlexServDeployment` trait: async `create/start/stop/terminate/monitor`
  - `FlexServPodDeployment` with `PodDeploymentOptions`
  - `FlexServHPCDeployment` (methods currently `todo!()`)
  - `DeploymentResult` and `DeploymentError`

---

## Using the Library

### Add dependency

In your project’s `Cargo.toml`:

```toml
[dependencies]
flexserv-deployer = { path = "../FlexServ-Deployer" }  # adjust path as needed
```

### Basic imports

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

### Quickstart: create a pod (async)

This example:
- builds a `FlexServInstance`
- wraps it in `FlexServPodDeployment`
- calls `create()` to create a volume and pod

```rust
use flexserv_deployer::{
    Backend, DeploymentError, DeploymentResult, FlexServDeployment,
    FlexServInstance, FlexServPodDeployment,
};

#[tokio::main]
async fn main() -> Result<(), DeploymentError> {
    env_logger::init();

    let tenant_url = std::env::var("TAPIS_TENANT_URL").expect("TAPIS_TENANT_URL");
    let tapis_token = std::env::var("TAPIS_TOKEN").expect("TAPIS_TOKEN");

    // Model must already exist on the attached volume at /app/models/<model_dir_name>.
    let model_id = std::env::var("FLEXSERV_MODEL_ID")
        .unwrap_or_else(|_| "no-model-yet".to_string());

    let server = FlexServInstance::new(
        tenant_url,
        "your_tapis_username".to_string(),
        model_id,
        None,                         // model_revision
        std::env::var("HF_TOKEN").ok(), // optional HF token
        None,                         // default_embedding_model
        Backend::Transformers {
            command_prefix: vec!["python".to_string()],
        },
    );

    let mut deployment = FlexServPodDeployment::new(server, tapis_token);
    let result = deployment.create().await?;

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
            println!("Tapis user: {}", tapis_user);
            println!("Tapis tenant: {}", tapis_tenant);
            if let Some(url) = pod_url {
                println!("Pod URL (for inference): {}", url);
            }
            let auth_token = model_id.replace('/', "_");
            println!("Auth token for pod: {}", auth_token);
        }
        DeploymentResult::HPCResult { .. } => unreachable!("pod deployment returns PodResult"),
    }

    Ok(())
}
```

### Pod options

Use `PodDeploymentOptions` with `with_options`:

```rust
let server = FlexServInstance::new(
    tenant_url,
    "your_tapis_username".to_string(),
    "openai-community/gpt2".to_string(),
    None,
    std::env::var("HF_TOKEN").ok(),
    None,
    Backend::Transformers { command_prefix: vec!["python".to_string()] },
);

let options = PodDeploymentOptions {
    deployment_id: Some("550e8400-e29b-41d4-a716-446655440000".to_string()),
    volume_size_mb: Some(20 * 1024),
    image: Some("tapis/flexserv:1.0".to_string()),
    cpu_request: Some(2000),
    mem_limit_mb: Some(16384),
    gpus: Some(0),
    flexserv_secret: Some("mysecret-".to_string()),
    ..Default::default()
};

let mut deployment = FlexServPodDeployment::with_options(server, tapis_token, options);
let result = deployment.create().await?;
```

### Manage an existing pod

```rust
let server = FlexServInstance::new(/* same config as create */);
let deployment = FlexServPodDeployment::from_existing(
    server,
    tapis_token,
    pod_id,
    volume_id,
);

deployment.start().await?;
deployment.monitor().await?;
deployment.stop().await?;
deployment.terminate().await?;
```

All methods return `Result<DeploymentResult, DeploymentError>`.

### Calling the pod HTTP API

Once you have:
- `pod_url` from `DeploymentResult::PodResult`
- `auth_token = model_id.replace('/', "_")`

Use any HTTP client (e.g. `reqwest`) to call the pod:

- Health: `GET {pod_url}/v1/flexserv/health`
- Models: `GET {pod_url}/v1/models`
- Completions/Chat: `POST {pod_url}/v1/completions` / `/v1/chat/completions`

Headers:
- `Authorization: Bearer <auth_token>`

---

## Running Tests

There are two main categories:

- **Unit tests** – internal logic (backend, server, deployment).
- **Integration tests** – call real TAPIS Pods APIs.

### Helper script: `run-tests.sh`

At the repo root:

```bash
# Unit tests only
./run-tests.sh unit

# All tests (unit + integration)
./run-tests.sh all

# Single integration tests (pass parameters instead of exporting envs)
./run-tests.sh create    https://tacc.tapis.io "<TAPIS_TOKEN>"
./run-tests.sh start     https://tacc.tapis.io "<TAPIS_TOKEN>" "<your-pod-id>" "<your-volume-id>"
./run-tests.sh stop      https://tacc.tapis.io "<TAPIS_TOKEN>" "<your-pod-id>" "<your-volume-id>"
./run-tests.sh monitor   https://tacc.tapis.io "<TAPIS_TOKEN>" "<your-pod-id>" ["<your-volume-id>"]
./run-tests.sh terminate https://tacc.tapis.io "<TAPIS_TOKEN>" "<your-pod-id>" "<your-volume-id>"
```

If no argument is provided, the script defaults to `unit`.

### Manual commands

```bash
# Unit tests
cargo test --lib

# All tests (unit + integration)
cargo test

# Run a single integration test with output
TAPIS_TENANT_URL=https://tacc.tapis.io TAPIS_TOKEN=<jwt> \
  cargo test --test pod_create_integration -- --nocapture
```

### Env vars for integration tests

The Rust integration tests under `tests/` still **read env vars** (`TAPIS_TENANT_URL`, `TAPIS_TOKEN`, `POD_ID`, `VOLUME_ID`, etc.) and **skip themselves** if they are missing.

`run-tests.sh` simply maps its CLI arguments into those env vars for you.  
You can also bypass the script and export the env vars manually if you prefer.

---

## Examples

Examples are in `examples/`:

- `create_pod.rs` – create volume + pod using env:
  - `TAPIS_TENANT_URL`, `TAPIS_TOKEN`
  - `FLEXSERV_MODEL_ID` (default `no-model-yet`)
  - optional `HF_TOKEN`
- `terminate_pod.rs` – terminate an existing pod and volume (`POD_ID`, `VOLUME_ID`).
- `call_pod.rs` – call a running pod’s HTTP API.
- `hash_demo.rs` – demonstrate deployment hash generation.

Example:

```bash
export TAPIS_TENANT_URL=https://tacc.tapis.io
export TAPIS_TOKEN=<your-jwt>
export FLEXSERV_MODEL_ID=openai-community/gpt2

cargo run --example create_pod -- --nocapture
```

---

## Notes and Limitations

- Pods expect models pre-populated on the volume under `/app/models/<model_dir_name>`.  
  The deployer does **not** download models at pod startup.
- `FlexServHPCDeployment` is not implemented yet (`todo!()`); do not use it in production.
- An empty `flexserv_secret`/`FLEXSERV_SECRET` is allowed but insecure; production deployments should use a strong secret.
