//! Integration tests for Pod deployment: monitor operation.
//!
//! Requires real TAPIS credentials and existing pod/volume. Set env vars and run:
//!
//!   TAPIS_TENANT_URL=https://tacc.tapis.io TAPIS_TOKEN=<your-jwt> \
//!   POD_ID=p<your-pod-id> VOLUME_ID=v<your-volume-id> \
//!   cargo test --test pod_monitor_integration -- --nocapture
//!
//! Optional: FLEXSERV_NO_MODEL=1 — use no-model config (for demos).
//! If TAPIS_TENANT_URL, TAPIS_TOKEN, POD_ID, or VOLUME_ID is unset, tests are skipped.

use flexserv_deployer::{
    Backend, DeploymentResult, FlexServDeployment, FlexServPodDeployment, FlexServInstance,
};

fn env_or_skip() -> Option<(String, String)> {
    let tenant = std::env::var("TAPIS_TENANT_URL").ok()?;
    let token = std::env::var("TAPIS_TOKEN").ok()?;
    if tenant.is_empty() || token.is_empty() {
        return None;
    }
    Some((tenant, token))
}

fn make_server(tenant_url: &str, model_id: &str) -> FlexServInstance {
    FlexServInstance::new(
        tenant_url.to_string(),
        "testuser".to_string(),
        model_id.to_string(),
        None,
        None,
        None,
        Backend::Transformers {
            command: vec!["python".to_string()],
        },
    )
}

/// Model id for this test run. Set FLEXSERV_NO_MODEL=1 (or true) for no-model (demos).
fn test_model_id() -> String {
    match std::env::var("FLEXSERV_NO_MODEL").ok().as_deref() {
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => "no-model-yet".to_string(),
        _ => "openai-community/gpt2".to_string(),
    }
}

/// Helper to create deployment from existing pod/volume IDs (from env vars).
/// Returns None if POD_ID or VOLUME_ID are not set.
fn make_existing_deployment(tenant_url: &str, tapis_token: &str) -> Option<FlexServPodDeployment> {
    let pod_id = std::env::var("POD_ID").ok()?;
    let volume_id = std::env::var("VOLUME_ID").ok()?;
    if pod_id.is_empty() || volume_id.is_empty() {
        return None;
    }
    let server = make_server(tenant_url, &test_model_id());
    Some(FlexServPodDeployment::from_existing(
        server,
        tapis_token.to_string(),
        pod_id,
        volume_id,
    ))
}

/// Test monitor() functionality: verify return values and that pod/volume info is retrieved.
#[test]
fn test_monitor_functionality() {
    let (tenant_url, tapis_token) = match env_or_skip() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: set TAPIS_TENANT_URL and TAPIS_TOKEN to run");
            return;
        }
    };

    let deployment = match make_existing_deployment(&tenant_url, &tapis_token) {
        Some(d) => {
            eprintln!("Using existing pod: {}, volume: {}", d.pod_id, d.volume_id);
            d
        }
        None => {
            eprintln!("Skipping: set POD_ID and VOLUME_ID env vars to test existing deployment");
            return;
        }
    };

    let monitor_result = deployment.monitor().expect("monitor should succeed");
    match monitor_result {
        DeploymentResult::PodResult {
            pod_id,
            volume_id,
            pod_url,
            pod_info,
            volume_info,
            tapis_user,
            model_id,
            ..
        } => {
            assert_eq!(pod_id, deployment.pod_id, "monitor() should return correct pod_id");
            assert_eq!(volume_id, deployment.volume_id, "monitor() should return correct volume_id");
            assert_eq!(tapis_user, "testuser");
            assert_eq!(model_id, "no-model-yet");
            assert!(!pod_info.is_empty(), "monitor() should return pod_info");
            assert!(!volume_info.is_empty(), "monitor() should return volume_info");
            eprintln!("Monitor OK -> pod_id: {}, volume_id: {}, pod_url: {:?}", pod_id, volume_id, pod_url);
            eprintln!("pod_info length: {} chars, volume_info length: {} chars", pod_info.len(), volume_info.len());
        }
        _ => panic!("monitor() should return PodResult"),
    }
}

/// Test monitor() multiple times: verify it can be called repeatedly.
#[test]
fn test_monitor_repeated() {
    let (tenant_url, tapis_token) = match env_or_skip() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: set TAPIS_TENANT_URL and TAPIS_TOKEN to run");
            return;
        }
    };

    let deployment = match make_existing_deployment(&tenant_url, &tapis_token) {
        Some(d) => {
            eprintln!("Using existing pod: {}, volume: {}", d.pod_id, d.volume_id);
            d
        }
        None => {
            eprintln!("Skipping: set POD_ID and VOLUME_ID env vars to test existing deployment");
            return;
        }
    };

    // Monitor first time
    let monitor1 = deployment.monitor().expect("first monitor should succeed");
    match monitor1 {
        DeploymentResult::PodResult { pod_id, .. } => {
            assert_eq!(pod_id, deployment.pod_id);
            eprintln!("First monitor OK");
        }
        _ => panic!("expected PodResult"),
    }

    // Wait a moment
    std::thread::sleep(std::time::Duration::from_secs(1));

    // Monitor second time
    let monitor2 = deployment.monitor().expect("second monitor should succeed");
    match monitor2 {
        DeploymentResult::PodResult { pod_id, pod_url, .. } => {
            assert_eq!(pod_id, deployment.pod_id);
            eprintln!("Second monitor OK -> pod_url: {:?}", pod_url);
        }
        _ => panic!("expected PodResult"),
    }
}
