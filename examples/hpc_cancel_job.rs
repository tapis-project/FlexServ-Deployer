//! Cancel a running FlexServ HPC job (`FlexServHPCDeployment::stop` / `terminate`).
//!
//! ```text
//! export TAPIS_TOKEN=...
//! export TAPIS_USER=...
//! export TAPIS_HPC_JOB_UUID=<uuid>
//! export TAPIS_TENANT_URL=https://public.tapis.io   # required for from_existing
//! cargo run --example hpc_cancel_job
//! ```

use flexserv_deployer::{FlexServDeployment, FlexServHPCDeployment};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let job_uuid = std::env::var("TAPIS_HPC_JOB_UUID").expect("TAPIS_HPC_JOB_UUID is required");
    let tapis_token = std::env::var("TAPIS_TOKEN").expect("TAPIS_TOKEN is required");
    let tenant_url = std::env::var("TAPIS_TENANT_URL").expect("TAPIS_TENANT_URL is required");
    let mut deployment = FlexServHPCDeployment::from_existing(tapis_token, job_uuid.clone());
    deployment.tenant_url = Some(tenant_url);

    println!("Canceling job {}...", job_uuid);
    let result = deployment.stop().await?;
    println!("{:#?}", result);

    Ok(())
}
