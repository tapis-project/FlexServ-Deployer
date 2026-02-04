use crate::server::FlexServInstance;

/// Deployment result enum
#[derive(Debug)]
pub enum DeploymentResult {
    PodResult {
        pod_info: String, // Using String for now since we don't have tapis_pods in dependencies yet
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
#[derive(Debug)]
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

/// FlexServ deployment trait
pub trait FlexServDeployment {
    fn create(&mut self) -> Result<DeploymentResult, DeploymentError>;
    fn start(&self) -> Result<DeploymentResult, DeploymentError>;
    fn stop(&self) -> Result<DeploymentResult, DeploymentError>;
    fn terminate(&self) -> Result<DeploymentResult, DeploymentError>;
    fn monitor(&self) -> Result<DeploymentResult, DeploymentError>;
}

/// Pod-based deployment
pub struct FlexServPodDeployment {
    pub server: FlexServInstance,
    // Note: These fields will need proper types once tapis-pods is properly integrated
    // For now using placeholder types to allow compilation
    pub new_volume: String,
    pub new_pod: String,
    pub volume_info: Option<String>,
    pub pod_info: Option<String>,
}

impl FlexServPodDeployment {
    pub fn new(server: FlexServInstance) -> Self {
        Self {
            server,
            new_volume: String::new(),
            new_pod: String::new(),
            volume_info: None,
            pod_info: None,
        }
    }
}

impl FlexServDeployment for FlexServPodDeployment {
    fn create(&mut self) -> Result<DeploymentResult, DeploymentError> {
        // Create volume and pod
        todo!("Implement pod and volume creation")
    }

    fn start(&self) -> Result<DeploymentResult, DeploymentError> {
        // Start pod
        todo!("Implement pod start")
    }

    fn stop(&self) -> Result<DeploymentResult, DeploymentError> {
        // Stop pod
        todo!("Implement pod stop")
    }

    fn terminate(&self) -> Result<DeploymentResult, DeploymentError> {
        // Terminate pod and delete volume
        todo!("Implement pod termination and volume deletion")
    }

    fn monitor(&self) -> Result<DeploymentResult, DeploymentError> {
        let deployment_hash = self.server.deployment_hash();
        let pod_id = format!("p{}", deployment_hash);
        let volume_id = format!("v{}", deployment_hash);
        // collect pod and volume info using SDK

        let mut pod_result = DeploymentResult::PodResult {
            pod_info: self.pod_info.clone().unwrap_or_default(),
            volume_info: self.volume_info.clone().unwrap_or_default(),
            tapis_user: self.server.tapis_user.clone(),
            tapis_tenant: self.server.tenant_url.clone(),
            model_id: self.server.default_model.clone(),
        };
        // Monitor pod status
        Ok(pod_result)
    }
}

/// HPC-based deployment
pub struct FlexServHPCDeployment {
    pub server: FlexServInstance,
}

impl FlexServHPCDeployment {
    pub fn new(server: FlexServInstance) -> Self {
        Self { server }
    }
}

impl FlexServDeployment for FlexServHPCDeployment {
    fn create(&mut self) -> Result<DeploymentResult, DeploymentError> {
        // Create HPC job
        todo!("Implement HPC job creation")
    }

    fn start(&self) -> Result<DeploymentResult, DeploymentError> {
        // Start HPC job
        todo!("Implement HPC job start")
    }

    fn stop(&self) -> Result<DeploymentResult, DeploymentError> {
        // Stop HPC job
        todo!("Implement HPC job stop")
    }

    fn terminate(&self) -> Result<DeploymentResult, DeploymentError> {
        // Terminate HPC job
        todo!("Implement HPC job termination")
    }

    fn monitor(&self) -> Result<DeploymentResult, DeploymentError> {
        // Monitor HPC job status
        todo!("Implement HPC job monitoring")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::server::FlexServInstance;

    #[test]
    fn test_pod_deployment_creation() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "testuser".to_string(),
            "meta-llama/Llama-2-7b".to_string(),
            None,
            Backend::Transformers {
                command: vec!["python".to_string()],
            },
        );

        let deployment = FlexServPodDeployment::new(server);
        assert_eq!(deployment.server.tapis_user, "testuser");
    }

    #[test]
    fn test_hpc_deployment_creation() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "testuser".to_string(),
            "meta-llama/Llama-2-7b".to_string(),
            None,
            Backend::VLlm {
                command: vec!["python".to_string(), "-m".to_string(), "vllm".to_string()],
            },
        );

        let deployment = FlexServHPCDeployment::new(server);
        assert_eq!(deployment.server.tapis_user, "testuser");
    }
}
