use flexserv_deployer::{
    Backend, FlexServDeployment, FlexServHPCDeployment, FlexServInstance, HpcDeploymentOptions,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Required env:
    // - TAPIS_TOKEN
    // - TAPIS_USER
    // - TAPIS_HPC_EXEC_SYSTEM_ID (for example: vista-tapis)
    //
    // Also required for submission:
    // - TAPIS_HPC_APP_ID
    // - TAPIS_HPC_APP_VERSION
    // - TAPIS_HPC_EXEC_SYSTEM_LOGICAL_QUEUE
    // - TAPIS_HPC_MAX_MINUTES
    // - TAPIS_HPC_ALLOCATION 
    //
    // Optional env:
    // - TAPIS_TENANT_URL (default: https://public.tapis.io)
    // - FLEXSERV_MODEL_ID (default: Qwen/Qwen3.5-0.8B)
    // - FLEXSERV_BACKEND (default: transformers)
    //
    let tenant_url =
        std::env::var("TAPIS_TENANT_URL").unwrap_or_else(|_| "https://public.tapis.io".to_string());
    let tapis_user = std::env::var("TAPIS_USER").expect("TAPIS_USER is required");
    let model_id =
        std::env::var("FLEXSERV_MODEL_ID").unwrap_or_else(|_| "Qwen/Qwen3.5-0.8B".to_string());
    let backend_name =
        std::env::var("FLEXSERV_BACKEND").unwrap_or_else(|_| "transformers".to_string());
    let tapis_token = std::env::var("TAPIS_TOKEN").expect("TAPIS_TOKEN is required");
    let exec_system_id =
        std::env::var("TAPIS_HPC_EXEC_SYSTEM_ID").expect("TAPIS_HPC_EXEC_SYSTEM_ID is required");
    let app_id = std::env::var("TAPIS_HPC_APP_ID").expect("TAPIS_HPC_APP_ID is required");
    let app_version =
        std::env::var("TAPIS_HPC_APP_VERSION").expect("TAPIS_HPC_APP_VERSION is required");
    let logical_queue = std::env::var("TAPIS_HPC_EXEC_SYSTEM_LOGICAL_QUEUE")
        .expect("TAPIS_HPC_EXEC_SYSTEM_LOGICAL_QUEUE is required");
    let max_minutes = std::env::var("TAPIS_HPC_MAX_MINUTES")
        .expect("TAPIS_HPC_MAX_MINUTES is required")
        .parse::<i32>()
        .expect("TAPIS_HPC_MAX_MINUTES must parse as i32");
    let allocation =
        std::env::var("TAPIS_HPC_ALLOCATION").expect("TAPIS_HPC_ALLOCATION is required");

    let backend = match backend_name.to_lowercase().as_str() {
        "transformers" => Backend::Transformers { command: vec![] },
        "vllm" => Backend::VLlm { command: vec![] },
        "sglang" => Backend::SGLang { command: vec![] },
        "trtllm" => Backend::TrtLlm { command: vec![] },
        other => {
            return Err(format!(
                "Unsupported FLEXSERV_BACKEND '{}'; use transformers|vllm|sglang|trtllm",
                other
            )
            .into())
        }
    };

    let server = FlexServInstance::new(
        tenant_url,
        tapis_user,
        model_id,
        None,
        std::env::var("HF_TOKEN").ok(),
        None,
        backend,
    );

    let options = HpcDeploymentOptions::new(
        app_id,
        app_version,
        exec_system_id,
        logical_queue,
        max_minutes,
        allocation,
    );

    let mut deployment = FlexServHPCDeployment::new(server, tapis_token, options);

    println!("Submitting HPC job...");
    let created = deployment.create().await?;
    println!("CREATE RESULT:\n{:#?}", created);

    println!("Checking job status/details...");
    let monitored = deployment.monitor().await?;
    println!("MONITOR RESULT:\n{:#?}", monitored);

    // Optional cleanup:
    // let stopped = deployment.stop().await?;
    // println!("STOP RESULT:\n{:#?}", stopped);

    Ok(())
}
