//! Example: run Pod deployment create against TAPIS.
//!
//! Creates a volume and pod; the pod expects the model to already be present in the
//! volume at /app/models/<model_name> (FlexServ does not download models at startup).
//!
//! Set env vars then run:
//!
//!   export TAPIS_TENANT_URL=https://tacc.tapis.io
//!   export TAPIS_TOKEN=<your-jwt>
//!   cargo run --example create_pod
//!
//! Set the model (no built-in default; pod expects model at /app/models/<model_name>):
//!
//!   export FLEXSERV_MODEL_ID=openai-community/gpt2
//!   cargo run --example create_pod
//!
//! Placeholder (no real model path):
//!
//!   FLEXSERV_MODEL_ID=no-model-yet cargo run --example create_pod
//!
//! Optional HF_TOKEN (for backends that need it):
//!
//!   export HF_TOKEN=<your-hf-token>
//!   cargo run --example create_pod
//!
//! Or pass tenant and token as args:
//!
//!   cargo run --example create_pod -- https://tacc.tapis.io <jwt>

use flexserv_deployer::{
    Backend, DeploymentError, DeploymentResult, FlexServDeployment, FlexServPodDeployment,
    FlexServInstance,
};

#[tokio::main]
async fn main() -> Result<(), DeploymentError> {
    env_logger::init();

    let (tenant_url, tapis_token) = if let (Ok(t), Ok(token)) = (
        std::env::var("TAPIS_TENANT_URL"),
        std::env::var("TAPIS_TOKEN"),
    ) {
        (t, token)
    } else if let [tenant, token, ..] = std::env::args().collect::<Vec<_>>().as_slice() {
        (tenant.clone(), token.clone())
    } else {
        eprintln!("Usage: TAPIS_TENANT_URL=... TAPIS_TOKEN=... cargo run --example create_pod");
        eprintln!("   or: cargo run --example create_pod -- <TAPIS_TENANT_URL> <TAPIS_TOKEN>");
        std::process::exit(1);
    };

    // Model id: from FLEXSERV_MODEL_ID (no built-in default; pod expects model at /app/models/<model_name>).
    let model_id = std::env::var("FLEXSERV_MODEL_ID").unwrap_or_else(|_| "no-model-yet".to_string());
    let hf_token = std::env::var("HF_TOKEN").ok();
    let server = FlexServInstance::new(
        tenant_url,
        "testuser".to_string(),
        model_id,
        None,
        hf_token,
        None,
        Backend::Transformers {
            command_prefix: vec![
                "/app/venvs/transformers/bin/python".to_string(),
                "/app/flexserv/python/backend/transformers/backend_server.py".to_string(),
            ],
        },
    );

    let mut deployment = FlexServPodDeployment::new(server, tapis_token);

    let result = deployment.create().await?;

    match result {
        DeploymentResult::PodResult {
            pod_id,
            volume_id,
            pod_url,
            pod_info,
            volume_info: _,
            tapis_user,
            tapis_tenant,
            model_id,
        } => {
            println!("Create succeeded:");
            println!("  pod_id:     {}  (use for start/stop/terminate/monitor)", pod_id);
            println!("  volume_id:  {}", volume_id);
            if let Some(ref url) = pod_url {
                println!("  pod_url:    {}  (use for inference or health checks)", url);
            } else {
                println!("  pod_url:    (not yet available)");
            }
            println!("  tapis_user: {}", tapis_user);
            println!("  tapis_tenant: {}", tapis_tenant);
            println!("  model_id:   {}  (use this in request JSON: \"model\": \"...\" )", model_id);
            let auth_token = model_id.replace('/', "_");
            println!("  auth_token: {} (use as Authorization: Bearer or X-FlexServ-Secret; add FLEXSERV_SECRET prefix if you set it)", auth_token);
            println!("  pod_info (first 400 chars): {}", pod_info.chars().take(400).collect::<String>());
        }
        DeploymentResult::HPCResult { .. } => unreachable!("pod deployment returns PodResult"),
    }

    Ok(())
}
