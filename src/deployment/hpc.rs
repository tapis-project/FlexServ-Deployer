//! HPC-based deployment implementation for FlexServ.
//!
//! This module handles deployment to HPC systems via batch job submission.
//! Currently a stub - implementation to be completed.

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
