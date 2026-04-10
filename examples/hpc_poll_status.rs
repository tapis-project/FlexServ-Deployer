//! Poll Tapis Jobs status for an existing FlexServ HPC job (library: `FlexServHPCDeployment::job_status`).
//!
//! ```text
//! export TAPIS_TOKEN=...
//! export TAPIS_USER=...
//! export TAPIS_HPC_JOB_UUID=<uuid-from-submit>
//! export TAPIS_TENANT_URL=https://public.tapis.io   # optional
//! cargo run --example hpc_poll_status
//! ```
//!
//! Optional: `POLL_INTERVAL_SECS` (default 10), `POLL_MAX_ITER` (default 360, ~1h at 10s).

use flexserv_deployer::{Backend, FlexServDeployment, FlexServHPCDeployment, FlexServInstance};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tenant_url =
        std::env::var("TAPIS_TENANT_URL").unwrap_or_else(|_| "https://public.tapis.io".to_string());
    let tapis_user = std::env::var("TAPIS_USER").expect("TAPIS_USER is required");
    let job_uuid = std::env::var("TAPIS_HPC_JOB_UUID").expect("TAPIS_HPC_JOB_UUID is required");
    let tapis_token = std::env::var("TAPIS_TOKEN").expect("TAPIS_TOKEN is required");

    let interval = std::env::var("POLL_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(10);
    let max_iter = std::env::var("POLL_MAX_ITER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(360);

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

    println!(
        "Polling job {} every {}s (max {} iterations)...",
        job_uuid, interval, max_iter
    );

    for i in 0..max_iter {
        let status = deployment.job_status().await?;
        println!("[{}] status = {}", i + 1, status);
        if matches!(
            status.as_str(),
            "FINISHED" | "FAILED" | "CANCELLED" | "CANCELED"
        ) {
            break;
        }
        tokio::time::sleep(Duration::from_secs(interval)).await;
    }

    let detail = deployment.monitor().await?;
    println!("Final monitor snapshot:\n{:#?}", detail);

    Ok(())
}
