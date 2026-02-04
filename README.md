# FlexServ Deployer

A Rust library and web server for deploying FlexServ on Tapis Pods or HPC systems via Tapis Jobs, with various backends (Transformers, vLLM, SGLang, TRT-LLM).


## Library Components

### Backend Module (`backend.rs`)
- **Backend enum**: Supports Transformers, vLLM, SGLang, and TRT-LLM
- **BackendParameters**: Flexible configuration for backend-specific settings
- **Parameter Builders**: Type-safe builders for each backend:
  - `TransformersParametersBuilder`
  - `VLlmParametersBuilder`
  - `SGLangParametersBuilder`
  - `TrtLlmParametersBuilder`

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

### Run only integration tests
```bash
cargo test --test integration_test
```

### Run tests with output
```bash
cargo test -- --nocapture
```

<!-- ## API Endpoints

- `GET /search` - Search for models
- `POST /info` - Get model information -->

## Development

### Using the library in other projects
Add to your `Cargo.toml`:
```toml
[dependencies]
flexserv-deployer = { path = "../FlexServ-deployer" }
```

Then import:
```rust
use flexserv_deployer::{model, deployment};
```

## Dependencies

- **actix-web**: Web framework
- **tokio**: Async runtime
- **serde**: Serialization/deserialization
- **reqwest**: HTTP client
- **tapis-sdk**: Tapis API integration
- **hf-hub**: HuggingFace Hub integration
