//! Cancel a running FlexServ HPC job (`FlexServHPCDeployment::stop` / `terminate`).
//!
//! ```text
//! export TAPIS_TOKEN=...
//! export TAPIS_USER=...
//! export TAPIS_HPC_JOB_UUID=<uuid>
//! export TAPIS_TENANT_URL=https://public.tapis.io   # optional
//! cargo run --example hpc_cancel_job
//! ```

use flexserv_deployer::{Backend, FlexServDeployment, FlexServHPCDeployment, FlexServInstance};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tenant_url =
        std::env::var("TAPIS_TENANT_URL").unwrap_or_else(|_| "https://public.tapis.io".to_string());
    let tapis_user = std::env::var("TAPIS_USER").expect("TAPIS_USER is required");
    let job_uuid = std::env::var("TAPIS_HPC_JOB_UUID").expect("TAPIS_HPC_JOB_UUID is required");
    let tapis_token = std::env::var("TAPIS_TOKEN").expect("TAPIS_TOKEN is required");

    let server = FlexServInstance::new(
        tenant_url,
        tapis_user,
        std::env::var("FLEXSERV_MODEL_ID").unwrap_or_else(|_| "no-model-yet".to_string()),
        None,
        None,
        None,
        Backend::Transformers { command: vec![] },
    );

    let deployment = FlexServHPCDeployment::from_existing(server, tapis_token, job_uuid.clone());

    println!("Canceling job {}...", job_uuid);
    let result = deployment.stop().await?;
    println!("{:#?}", result);

    Ok(())
}
