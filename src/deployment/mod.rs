//! Deployment module: common types and traits, plus Pod and HPC implementations.

use serde::Serialize;
use std::fmt;

mod hpc;
mod pod;

pub use hpc::FlexServHPCDeployment;
pub use pod::{FlexServPodDeployment, PodDeploymentOptions};

/// Deployment result enum.
/// Implements Serialize so HTTP handlers can return it as JSON (e.g. `HttpResponse::Ok().json(result)`).
#[derive(Debug, Serialize)]
pub enum DeploymentResult {
    PodResult {
        /// TAPIS Pod id (e.g. `p{deployment_hash}`). Use this for start/stop/terminate/monitor.
        pod_id: String,
        /// TAPIS Volume id (e.g. `v{deployment_hash}`).
        volume_id: String,
        /// URL to reach the pod (e.g. from networking.default.url). Use for inference or health checks.
        pod_url: Option<String>,
        pod_info: String,
        volume_info: String,
        tapis_user: String,
        tapis_tenant: String,
        model_id: String,
    },
    HPCResult {
        job_info: String,
        tapis_user: String,
        tapis_tenant: String,
        model_id: String,
    },
}

/// Deployment related errors
/// We can bind the message to this enum variant for more detailed error information
/// 1. TapisAuthFailed(String) - Authentication to Tapis failed
/// 2. TapisAPIUnreachable(String) - Tapis API is unreachable
/// 3. TapisBadRequest(String) - Bad request to Tapis API
/// 4. TapisTimeout(String) - Request to Tapis API timed out
/// 5. TapisInternalServerError(String) - Tapis API internal server error
/// 6. UnknownError(String) - Unknown error
/// 7. ModelUploadingFailed(String) - Model uploading failed not because of any of the reasons from 1-6.
/// 8. PodCreationFailed(String) - Pod creation failed not because of any of the reasons from 1-6.
/// 9. JobCreationFailed(String) - Job creation failed not because of any of the reasons from 1-6.
/// Each variant carries a message; implements Display, Error, and Serialize so call sites can
/// show messages, use `?`, and return JSON from HTTP handlers (e.g. `HttpResponse::BadRequest().json(err)`).
#[derive(Debug, Serialize)]
pub enum DeploymentError {
    TapisAuthFailed(String),
    TapisAPIUnreachable(String),
    TapisBadRequest(String),
    TapisTimeout(String),
    TapisInternalServerError(String),
    UnknownError(String),
    ModelUploadingFailed(String),
    PodCreationFailed(String),
    JobCreationFailed(String),
}

impl fmt::Display for DeploymentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeploymentError::TapisAuthFailed(msg) => write!(f, "TAPIS auth failed: {}", msg),
            DeploymentError::TapisAPIUnreachable(msg) => write!(f, "TAPIS API unreachable: {}", msg),
            DeploymentError::TapisBadRequest(msg) => write!(f, "TAPIS bad request: {}", msg),
            DeploymentError::TapisTimeout(msg) => write!(f, "TAPIS timeout: {}", msg),
            DeploymentError::TapisInternalServerError(msg) => write!(f, "TAPIS server error: {}", msg),
            DeploymentError::UnknownError(msg) => write!(f, "Unknown error: {}", msg),
            DeploymentError::ModelUploadingFailed(msg) => write!(f, "Model upload failed: {}", msg),
            DeploymentError::PodCreationFailed(msg) => write!(f, "Pod creation failed: {}", msg),
            DeploymentError::JobCreationFailed(msg) => write!(f, "Job creation failed: {}", msg),
        }
    }
}

impl std::error::Error for DeploymentError {}

/// FlexServ deployment trait.
/// All methods are async to avoid deadlocks when called from an async runtime.
#[allow(async_fn_in_trait)]
pub trait FlexServDeployment {
    async fn create(&mut self) -> Result<DeploymentResult, DeploymentError>;
    async fn start(&self) -> Result<DeploymentResult, DeploymentError>;
    async fn stop(&self) -> Result<DeploymentResult, DeploymentError>;
    async fn terminate(&self) -> Result<DeploymentResult, DeploymentError>;
    async fn monitor(&self) -> Result<DeploymentResult, DeploymentError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_deployment_error_debug_display() {
        let e = DeploymentError::TapisBadRequest("msg".to_string());
        let s = format!("{:?}", e);
        assert!(s.contains("TapisBadRequest"));
        assert!(s.contains("msg"));
    }

    #[test]
    fn test_deployment_error_display_and_error() {
        let e = DeploymentError::TapisBadRequest("bad".to_string());
        assert!(format!("{}", e).contains("bad request"));
        assert!(e.source().is_none());
    }

    #[test]
    fn test_deployment_result_pod_variant() {
        let r = DeploymentResult::PodResult {
            pod_id: "p1".to_string(),
            volume_id: "v1".to_string(),
            pod_url: Some("http://pod:8000".to_string()),
            pod_info: "info".to_string(),
            volume_info: "vol".to_string(),
            tapis_user: "u".to_string(),
            tapis_tenant: "t".to_string(),
            model_id: "m".to_string(),
        };
        match &r {
            DeploymentResult::PodResult { pod_id, pod_url, .. } => {
                assert_eq!(pod_id, "p1");
                assert_eq!(pod_url.as_deref(), Some("http://pod:8000"));
            }
            _ => panic!("expected PodResult"),
        }
    }
}
