//! Example: terminate (delete) an existing Pod deployment.
//!
//! Set env vars then run:
//!
//!   export TAPIS_TENANT_URL=https://tacc.tapis.io
//!   export TAPIS_TOKEN=<your-jwt>
//!   export POD_ID=<pod-id>   # e.g. pfbslvnwv6oik (from create_pod output)
//!   export VOLUME_ID=<volume-id>   # e.g. vfbslvnwv6oik
//!   cargo run --example terminate_pod
//!
//! Or pass as args (in order): tenant_url tapis_token pod_id volume_id
//!
//!   cargo run --example terminate_pod -- https://tacc.tapis.io <jwt> <pod_id> <volume_id>

use flexserv_deployer::{
    Backend, DeploymentError, DeploymentResult, FlexServDeployment, FlexServPodDeployment,
    FlexServInstance,
};

#[tokio::main]
async fn main() -> Result<(), DeploymentError> {
    env_logger::init();

    let (tenant_url, tapis_token, pod_id, volume_id) = if let (Ok(t), Ok(token), Ok(pod), Ok(vol)) = (
        std::env::var("TAPIS_TENANT_URL"),
        std::env::var("TAPIS_TOKEN"),
        std::env::var("POD_ID"),
        std::env::var("VOLUME_ID"),
    ) {
        (t, token, pod, vol)
    } else if let [tenant, token, pod, vol, ..] = std::env::args().collect::<Vec<_>>().as_slice() {
        (tenant.clone(), token.clone(), pod.clone(), vol.clone())
    } else {
        eprintln!("Usage: TAPIS_TENANT_URL=... TAPIS_TOKEN=... POD_ID=... VOLUME_ID=... cargo run --example terminate_pod");
        eprintln!("   or: cargo run --example terminate_pod -- <TAPIS_TENANT_URL> <TAPIS_TOKEN> <POD_ID> <VOLUME_ID>");
        std::process::exit(1);
    };

    let server = FlexServInstance::new(
        tenant_url.clone(),
        "testuser".to_string(),
        "openai-community/gpt2".to_string(),
        None,
        None,
        None,
        Backend::Transformers {
            command: vec!["python".to_string()],
        },
    );
    let deployment = FlexServPodDeployment::from_existing(server, tapis_token, pod_id.clone(), volume_id.clone());

    println!("Terminating pod {} and volume {}...", pod_id, volume_id);
    let result = deployment.terminate().await?;

    match result {
        DeploymentResult::PodResult {
            pod_id: p,
            volume_id: v,
            pod_url,
            pod_info,
            ..
        } => {
            println!("Terminate succeeded:");
            println!("  pod_id:    {}", p);
            println!("  volume_id: {}", v);
            println!("  pod_url:   {:?}", pod_url);
            println!("  info:      {} chars", pod_info.len());
        }
        DeploymentResult::HPCResult { .. } => unreachable!("pod deployment returns PodResult"),
    }

    Ok(())
}
