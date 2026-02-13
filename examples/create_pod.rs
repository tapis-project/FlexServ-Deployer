//! Example: run Pod deployment create against TAPIS.
//!
//! By default uses openai-community/gpt2: the pod downloads the model from Hugging Face
//! to the volume at startup, then starts the backend server (no deployer bandwidth used).
//!
//! Set env vars then run:
//!
//!   export TAPIS_TENANT_URL=https://tacc.tapis.io
//!   export TAPIS_TOKEN=<your-jwt>
//!   cargo run --example create_pod
//!
//! To skip model download (pod starts without a model):
//!
//!   FLEXSERV_NO_MODEL=1 cargo run --example create_pod
//!
//! For gated/private models, set HF_TOKEN so the pod can download:
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

fn main() -> Result<(), DeploymentError> {
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

    // Default: GPT-2 — pod downloads from HF at startup. Use FLEXSERV_NO_MODEL=1 to skip.
    let model_id = match std::env::var("FLEXSERV_NO_MODEL").ok().as_deref() {
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => "no-model-yet".to_string(),
        _ => "openai-community/gpt2".to_string(),
    };
    let hf_token = std::env::var("HF_TOKEN").ok();
    let server = FlexServInstance::new(
        tenant_url,
        "testuser".to_string(),
        model_id,
        None,
        hf_token,
        None,
        Backend::Transformers {
            command: vec!["python".to_string()],
        },
    );

    let mut deployment = FlexServPodDeployment::new(server, tapis_token);

    println!("Creating pod deployment (volume + pod; pod will download model at startup)...");
    let result = deployment.create()?;

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
            println!("  model_id:   {}", model_id);
            let auth_hint = model_id.replace('/', "_");
            println!("  auth_token: {} (use as Authorization: Bearer or X-FlexServ-Secret; add FLEXSERV_SECRET prefix if you set it)", auth_hint);
            println!("  pod_info (first 400 chars): {}", pod_info.chars().take(400).collect::<String>());
        }
        DeploymentResult::HPCResult { .. } => unreachable!("pod deployment returns PodResult"),
    }

    Ok(())
}
