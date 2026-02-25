//! Integration tests for Pod deployment: monitor operation.
//!
//! Requires real TAPIS credentials and existing pod/volume. Set env vars and run:
//!
//!   TAPIS_TENANT_URL=https://tacc.tapis.io TAPIS_TOKEN=<your-jwt> \
//!   POD_ID=p<your-pod-id> [VOLUME_ID=v<your-volume-id>] \
//!   cargo test --test pod_monitor_integration -- --nocapture
//!
//! VOLUME_ID is optional (use for pods with no volume, e.g. POD_ID=pmingyutest).
//! Optional: FLEXSERV_NO_MODEL=1 — use no-model config (for demos).

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
            command_prefix: vec!["python".to_string()],
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

/// Extract pod status from Debug-formatted pod_info (Tapis PodResponseModel: status: Some("AVAILABLE") etc.).
/// Tapis uses AVAILABLE for running pods; status_container.phase may be "Running".
fn status_from_pod_info(pod_info: &str) -> Option<&'static str> {
    if pod_info.contains("Some(\"AVAILABLE\")") || pod_info.contains("Some(\"Available\")") {
        return Some("AVAILABLE");
    }
    if pod_info.contains("Some(\"RUNNING\")") || pod_info.contains("Some(\"Running\")") {
        return Some("RUNNING");
    }
    if pod_info.contains("Some(\"STOPPED\")") || pod_info.contains("Some(\"Stopped\")") {
        return Some("STOPPED");
    }
    if pod_info.contains("Some(\"FAILED\")") || pod_info.contains("Some(\"Failed\")") {
        return Some("FAILED");
    }
    if pod_info.contains("Some(\"PENDING\")") || pod_info.contains("Some(\"Pending\")") {
        return Some("PENDING");
    }
    if pod_info.contains("Some(\"CREATING\")") {
        return Some("CREATING");
    }
    None
}

/// Helper to create deployment from existing pod (and optional volume) IDs from env vars.
/// Returns None if POD_ID is not set. VOLUME_ID is optional (empty = pod has no volume).
fn make_existing_deployment(tenant_url: &str, tapis_token: &str) -> Option<FlexServPodDeployment> {
    let pod_id = std::env::var("POD_ID").ok().filter(|s| !s.is_empty())?;
    let volume_id = std::env::var("VOLUME_ID").unwrap_or_default();
    let server = make_server(tenant_url, &test_model_id());
    Some(FlexServPodDeployment::from_existing(
        server,
        tapis_token.to_string(),
        pod_id,
        volume_id,
    ))
}

/// Test monitor() functionality: verify return values and that pod/volume info is retrieved.
#[tokio::test]
async fn test_monitor_functionality() {
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
            eprintln!("Skipping: set POD_ID env var to test existing deployment (VOLUME_ID optional)");
            return;
        }
    };

    let monitor_result = deployment.monitor().await.expect("monitor should succeed");
    match monitor_result {
        DeploymentResult::PodResult {
            pod_id,
            volume_id,
            pod_url,
            pod_info,
            volume_info,
            tapis_user,
            model_id: _model_id,
            ..
        } => {
            assert_eq!(pod_id, deployment.pod_id, "monitor() should return correct pod_id");
            assert_eq!(volume_id, deployment.volume_id, "monitor() should return correct volume_id");
            assert_eq!(tapis_user, "testuser");
            assert!(!pod_info.is_empty(), "monitor() should return pod_info");
            if !deployment.volume_id.is_empty() {
                assert!(!volume_info.is_empty(), "monitor() should return volume_info when volume_id set");
            }
            let state = status_from_pod_info(&pod_info);
            eprintln!("Monitor OK -> pod_id: {}, volume_id: {:?}, pod_url: {:?}", pod_id, if volume_id.is_empty() { "none" } else { &volume_id }, pod_url);
            eprintln!("pod state: {}", state.unwrap_or("(unknown)"));
            if state.is_none() {
                if let Some(idx) = pod_info.find("status: Some") {
                    let snippet = pod_info.get(idx..(idx + 60).min(pod_info.len())).unwrap_or("");
                    eprintln!("  (status snippet from pod_info: {:?})", snippet);
                }
            }
            eprintln!("pod_info length: {} chars, volume_info length: {} chars", pod_info.len(), volume_info.len());
        }
        _ => panic!("monitor() should return PodResult"),
    }
}

/// Test monitor() multiple times: verify it can be called repeatedly.
#[tokio::test]
async fn test_monitor_repeated() {
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
            eprintln!("Skipping: set POD_ID env var to test existing deployment (VOLUME_ID optional)");
            return;
        }
    };

    // Monitor first time
    let monitor1 = deployment.monitor().await.expect("first monitor should succeed");
    match monitor1 {
        DeploymentResult::PodResult { pod_id, .. } => {
            assert_eq!(pod_id, deployment.pod_id);
            eprintln!("First monitor OK");
        }
        _ => panic!("expected PodResult"),
    }

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Monitor second time
    let monitor2 = deployment.monitor().await.expect("second monitor should succeed");
    match monitor2 {
        DeploymentResult::PodResult { pod_id, pod_url, .. } => {
            assert_eq!(pod_id, deployment.pod_id);
            eprintln!("Second monitor OK -> pod_url: {:?}", pod_url);
        }
        _ => panic!("expected PodResult"),
    }
}
