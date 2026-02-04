use crate::backend::Backend;
use crate::base62;
use sha2::{Digest, Sha256};

/// FlexServ server configuration
pub struct FlexServInstance {
    /// tenant url
    pub tenant_url: String,

    /// tapis username
    pub tapis_user: String,

    /// model to deploy (e.g., "meta-llama/Llama-3-70b-hf")
    pub default_model: String,

    /// default embedding model
    pub default_embedding_model: Option<String>,

    /// backend to use
    pub backend: Backend,
}

impl FlexServInstance {
    pub fn new(
        tenant_url: String,
        tapis_user: String,
        default_model: String,
        default_embedding_model: Option<String>,
        backend: Backend,
    ) -> Self {
        FlexServInstance {
            tenant_url,
            tapis_user,
            default_model,
            default_embedding_model,
            backend,
        }
    }

    pub fn deployment_hash(&self) -> String {
        // Create a unique hash for the deployment configuration
        let config_string = format!(
            "{}@{}-{}-{:?}",
            self.tapis_user, self.tenant_url, self.default_model, self.backend
        );
        let digest = Sha256::digest(config_string.as_bytes());

        // Encode full 256-bit SHA256 to base62 and take first 12 characters
        let base62_str = base62::encode(&digest);
        base62_str.chars().take(12).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;

    #[test]
    fn test_flexserv_creation() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "testuser".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
            None,
            Backend::Transformers {
                command: vec!["python".to_string()],
            },
        );

        assert_eq!(server.tenant_url, "https://tacc.tapis.io");
        assert_eq!(server.tapis_user, "testuser");
    }

    #[test]
    fn test_deployment_hash() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "testuser".to_string(),
            "Qwen/Qwen3-0.6B".to_string(),
            None,
            Backend::Transformers {
                command: vec!["python".to_string()],
            },
        );

        let hash = server.deployment_hash();
        assert!(!hash.is_empty());
        // Hash should be consistent
        assert_eq!(hash, server.deployment_hash());
    }
}
