//! Fetch Pods API state for an existing pod + volume (`FlexServPodDeployment::monitor`).
//!
//! ```text
//! export TAPIS_TENANT_URL=https://tacc.tapis.io
//! export TAPIS_TOKEN=...
//! export TAPIS_USER=<tapis-user-matching-pod-owner>
//! export POD_ID=p...
//! export VOLUME_ID=v...
//! cargo run --example pod_monitor
//! ```

use flexserv_deployer::{
    Backend, DeploymentResult, FlexServDeployment, FlexServInstance, FlexServPodDeployment,
};

#[tokio::main]
async fn main() -> Result<(), flexserv_deployer::DeploymentError> {
    env_logger::init();

    let tenant_url = std::env::var("TAPIS_TENANT_URL").expect("TAPIS_TENANT_URL is required");
    let tapis_token = std::env::var("TAPIS_TOKEN").expect("TAPIS_TOKEN is required");
    let tapis_user = std::env::var("TAPIS_USER").unwrap_or_else(|_| "testuser".to_string());
    let pod_id = std::env::var("POD_ID").expect("POD_ID is required");
    let volume_id = std::env::var("VOLUME_ID").expect("VOLUME_ID is required");
    let model_id =
        std::env::var("FLEXSERV_MODEL_ID").unwrap_or_else(|_| "no-model-yet".to_string());

    let server = FlexServInstance::new(
        tenant_url,
        tapis_user,
        model_id,
        None,
        std::env::var("HF_TOKEN").ok(),
        None,
        Backend::Transformers { command: vec![] },
    );

    let deployment = FlexServPodDeployment::from_existing(server, tapis_token, pod_id, volume_id);

    let result = deployment.monitor().await?;

    match result {
        DeploymentResult::PodResult {
            pod_id,
            volume_id,
            pod_url,
            pod_info,
            volume_info,
            ..
        } => {
            println!("pod_id:     {}", pod_id);
            println!("volume_id:  {}", volume_id);
            println!("pod_url:    {:?}", pod_url);
            println!(
                "volume_info (first 500 chars): {}...",
                volume_info.chars().take(500).collect::<String>()
            );
            println!(
                "pod_info (first 800 chars): {}...",
                pod_info.chars().take(800).collect::<String>()
            );
        }
        DeploymentResult::HPCResult { .. } => unreachable!(),
    }

    Ok(())
}
