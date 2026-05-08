//! Start then stop an existing pod (`FlexServPodDeployment::start` / `stop`).
//!
//! Does not delete the pod or volume (use `terminate_pod` for that).
//!
//! ```text
//! export TAPIS_TENANT_URL=https://tacc.tapis.io
//! export TAPIS_TOKEN=...
//! export TAPIS_USER=<owner>
//! export POD_ID=p...
//! export VOLUME_ID=v...
//! cargo run --example pod_start_stop
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

    let deployment = FlexServPodDeployment::from_existing(
        server,
        tapis_token,
        pod_id.clone(),
        volume_id.clone(),
    );

    println!("Starting pod {}...", pod_id);
    let started = deployment.start().await?;
    println!("start => {:#?}", started);

    println!("Stopping pod {}...", pod_id);
    let stopped = deployment.stop().await?;
    println!("stop => {:#?}", stopped);

    match stopped {
        DeploymentResult::PodResult { pod_url, .. } => {
            println!("After stop, pod_url hint: {:?}", pod_url);
        }
        DeploymentResult::HPCResult { .. } => unreachable!(),
    }

    Ok(())
}
