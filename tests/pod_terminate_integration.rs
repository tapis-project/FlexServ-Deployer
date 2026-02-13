//! Integration tests for Pod deployment: terminate operation.
//!
//! Requires real TAPIS credentials and existing pod/volume. Set env vars and run:
//!
//!   TAPIS_TENANT_URL=https://tacc.tapis.io TAPIS_TOKEN=<your-jwt> \
//!   POD_ID=p<your-pod-id> VOLUME_ID=v<your-volume-id> \
//!   cargo test --test pod_terminate_integration -- --nocapture
//!
//! Optional: FLEXSERV_NO_MODEL=1 — use no-model config (for demos).
//! WARNING: These tests will DELETE the pod and volume!
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

/// Test terminate() functionality: verify return values and that pod/volume are deleted.
/// WARNING: This will delete the pod and volume!
#[test]
fn test_terminate_functionality() {
    let (tenant_url, tapis_token) = match env_or_skip() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: set TAPIS_TENANT_URL and TAPIS_TOKEN to run");
            return;
        }
    };

    let deployment = match make_existing_deployment(&tenant_url, &tapis_token) {
        Some(d) => {
            eprintln!("WARNING: This will DELETE pod {} and volume {}!", d.pod_id, d.volume_id);
            d
        }
        None => {
            eprintln!("Skipping: set POD_ID and VOLUME_ID env vars to test existing deployment");
            return;
        }
    };

    let terminate_result = deployment.terminate().expect("terminate should succeed");
    match terminate_result {
        DeploymentResult::PodResult {
            pod_id,
            volume_id,
            pod_url,
            pod_info,
            tapis_user,
            model_id,
            ..
        } => {
            assert_eq!(pod_id, deployment.pod_id, "terminate() should return correct pod_id");
            assert_eq!(volume_id, deployment.volume_id, "terminate() should return correct volume_id");
            assert_eq!(tapis_user, "testuser");
            assert_eq!(model_id, "no-model-yet");
            assert!(pod_url.is_none(), "terminate() should return None for pod_url");
            assert!(!pod_info.is_empty(), "terminate() should return pod_info");
            eprintln!("Terminate OK -> pod_id: {}, volume_id: {}", pod_id, volume_id);
            eprintln!("pod_info length: {} chars", pod_info.len());
        }
        _ => panic!("terminate() should return PodResult"),
    }

    // Verify pod/volume are actually deleted: monitor() should return error (pod not found)
    let monitor_result = deployment.monitor();
    assert!(monitor_result.is_err(), "monitor() should return error after terminate()");
    if let Err(e) = monitor_result {
        eprintln!("Verified: monitor() returns error after terminate (pod not found): {:?}", e);
    }
    eprintln!("Verified: pod and volume are deleted");
}
