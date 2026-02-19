//! Integration tests for Pod deployment: create operation.
//!
//! Requires real TAPIS credentials. Set env vars and run:
//!
//!   TAPIS_TENANT_URL=https://tacc.tapis.io TAPIS_TOKEN=<your-jwt> \
//!   cargo test --test pod_create_integration -- --nocapture
//!
//! Optional: FLEXSERV_NO_MODEL=1 — use no-model deployment (no Hugging Face download; good for demos).
//! If unset, defaults to downloading openai-community/gpt2 from HF and uploading to the volume.
//! If TAPIS_TENANT_URL or TAPIS_TOKEN is unset, tests are skipped (pass without calling API).

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

/// Model id for this test run. Set FLEXSERV_NO_MODEL=1 (or true) for no-model deployment (demos).
fn test_model_id() -> String {
    match std::env::var("FLEXSERV_NO_MODEL").ok().as_deref() {
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => "no-model-yet".to_string(),
        _ => "openai-community/gpt2".to_string(),
    }
}

fn make_server(tenant_url: &str, model_id: &str) -> FlexServInstance {
    FlexServInstance::new(
        tenant_url.to_string(),
        "testuser".to_string(),
        model_id.to_string(),
        None,
        std::env::var("HF_TOKEN").ok(),
        None,
        Backend::Transformers {
            command: vec!["python".to_string()],
        },
    )
}

/// Test create() functionality: verify return values and that pod/volume are created.
#[tokio::test]
async fn test_create_functionality() {
    let (tenant_url, tapis_token) = match env_or_skip() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: set TAPIS_TENANT_URL and TAPIS_TOKEN to run");
            return;
        }
    };

    let expected_model_id = test_model_id();
    let server = make_server(&tenant_url, &expected_model_id);
    let mut deployment = FlexServPodDeployment::new(server, tapis_token);

    // Test create() - this also tests cleanup of existing pods/volumes (handled inside create_impl)
    let result = deployment.create().await;
    let create_result = result.as_ref().map_err(|e| panic!("create failed: {:?}", e)).unwrap();
    
    match create_result {
        DeploymentResult::PodResult {
            pod_id,
            volume_id,
            pod_url,
            pod_info,
            volume_info,
            tapis_user,
            tapis_tenant: _,
            model_id,
        } => {
            assert!(!pod_id.is_empty(), "create() should return non-empty pod_id");
            assert!(!volume_id.is_empty(), "create() should return non-empty volume_id");
            assert_eq!(tapis_user, "testuser", "create() should return correct tapis_user");
            assert_eq!(model_id, &expected_model_id, "create() should return correct model_id");
            assert!(!pod_info.is_empty(), "create() should return pod_info");
            assert!(!volume_info.is_empty(), "create() should return volume_info");
            eprintln!("Create OK -> pod_id: {}, volume_id: {}, pod_url: {:?}", pod_id, volume_id, pod_url);
            eprintln!("pod_info length: {} chars, volume_info length: {} chars", pod_info.len(), volume_info.len());
        }
        _ => panic!("create() should return PodResult"),
    }
}
