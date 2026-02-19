use super::{DeploymentError, DeploymentResult, FlexServDeployment};
use crate::server::FlexServInstance;

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
    async fn create(&mut self) -> Result<DeploymentResult, DeploymentError> {
        todo!("Implement HPC job creation")
    }

    async fn start(&self) -> Result<DeploymentResult, DeploymentError> {
        todo!("Implement HPC job start")
    }

    async fn stop(&self) -> Result<DeploymentResult, DeploymentError> {
        todo!("Implement HPC job stop")
    }

    async fn terminate(&self) -> Result<DeploymentResult, DeploymentError> {
        todo!("Implement HPC job termination")
    }

    async fn monitor(&self) -> Result<DeploymentResult, DeploymentError> {
        todo!("Implement HPC job monitoring")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;

    #[test]
    fn test_hpc_deployment_creation() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "testuser".to_string(),
            "meta-llama/Llama-2-7b".to_string(),
            None,
            None,
            None,
            Backend::VLlm {
                command: vec!["python".to_string(), "-m".to_string(), "vllm".to_string()],
            },
        );

        let deployment = FlexServHPCDeployment::new(server);
        assert_eq!(deployment.server.tapis_user, "testuser");
    }
}
