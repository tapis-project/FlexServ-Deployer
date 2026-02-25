use crate::backend::Backend;
use crate::base62;
use crate::utils::is_absolute_http_url;
use sha2::{Digest, Sha256};
use std::fmt;

pub use crate::utils::normalize_tenant_url;

/// Server-side TAPIS connection config (tenant, user, token).
#[derive(Clone, Debug)]
pub struct TapisConfig {
    /// TAPIS tenant URL (e.g. "https://tacc.tapis.io")
    pub tenant_url: String,
    /// TAPIS username
    pub tapis_user: String,
    /// JWT used to authenticate against TAPIS Pods API
    pub tapis_token: String,
}

/// Model-related config (what to deploy, how to fetch it).
#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// Hugging Face model id (e.g. "openai-community/gpt2")
    pub model_id: String,
    /// Revision (branch/tag/commit); None = repo default
    pub model_revision: Option<String>,
    /// HF token for gated/private models; None = pod uses HF_TOKEN env
    pub hf_token: Option<String>,
    /// Optional default embedding model
    pub default_embedding_model: Option<String>,
}

/// Input validation error (URL format, non-empty fields, etc.).
#[derive(Clone, Debug)]
pub enum ValidationError {
    InvalidTenantUrl(String),
    EmptyTapisUser,
    EmptyModelId,
    InvalidModelRevision(String),
    MissingBackend,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::InvalidTenantUrl(msg) => write!(f, "invalid tenant URL: {}", msg),
            ValidationError::EmptyTapisUser => write!(f, "tapis_user must be non-empty"),
            ValidationError::EmptyModelId => write!(f, "model_id must be non-empty"),
            ValidationError::InvalidModelRevision(msg) => write!(f, "invalid model_revision: {}", msg),
            ValidationError::MissingBackend => write!(f, "backend is required"),
        }
    }
}

impl std::error::Error for ValidationError {}

/// FlexServ server configuration
#[derive(Debug)]
pub struct FlexServInstance {
    /// tenant url
    pub tenant_url: String,

    /// tapis username
    pub tapis_user: String,

    /// model to deploy (e.g., "meta-llama/Llama-3-70b-hf")
    pub default_model: String,

    /// Hugging Face revision (branch, tag, or commit; e.g. "main"). If None, repo default is used.
    pub model_revision: Option<String>,

    /// Hugging Face token for gated/private models. If None, pod falls back to HF_TOKEN env.
    pub hf_token: Option<String>,

    /// default embedding model
    pub default_embedding_model: Option<String>,

    /// backend to use
    pub backend: Backend,
}

/// Builder for [FlexServInstance] with optional validation.
#[derive(Clone)]
pub struct FlexServInstanceBuilder {
    tenant_url: Option<String>,
    tapis_user: Option<String>,
    default_model: Option<String>,
    model_revision: Option<String>,
    hf_token: Option<String>,
    default_embedding_model: Option<String>,
    backend: Option<Backend>,
}

impl FlexServInstanceBuilder {
    pub fn new() -> Self {
        Self {
            tenant_url: None,
            tapis_user: None,
            default_model: None,
            model_revision: None,
            hf_token: None,
            default_embedding_model: None,
            backend: None,
        }
    }

    pub fn tenant_url(mut self, url: impl Into<String>) -> Self {
        self.tenant_url = Some(url.into());
        self
    }

    pub fn tapis_user(mut self, user: impl Into<String>) -> Self {
        self.tapis_user = Some(user.into());
        self
    }

    pub fn model(mut self, model_id: impl Into<String>) -> Self {
        self.default_model = Some(model_id.into());
        self
    }

    pub fn model_revision(mut self, revision: impl Into<String>) -> Self {
        self.model_revision = Some(revision.into());
        self
    }

    pub fn hf_token(mut self, token: Option<String>) -> Self {
        self.hf_token = token;
        self
    }

    pub fn default_embedding_model(mut self, model: Option<String>) -> Self {
        self.default_embedding_model = model;
        self
    }

    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Build and validate (non-empty user/model, tenant URL; URL normalized if no scheme, e.g. `tacc.tapis.io` → `https://tacc.tapis.io`).
    pub fn build(self) -> Result<FlexServInstance, ValidationError> {
        let tenant_url = normalize_tenant_url(&self.tenant_url.unwrap_or_default());
        let tenant_url = tenant_url.trim();
        if tenant_url.is_empty() || !is_absolute_http_url(tenant_url) {
            return Err(ValidationError::InvalidTenantUrl(
                "must be non-empty; use e.g. https://tacc.tapis.io or tacc.tapis.io".to_string(),
            ));
        }
        let tapis_user = self.tapis_user.unwrap_or_default().trim().to_string();
        if tapis_user.is_empty() {
            return Err(ValidationError::EmptyTapisUser);
        }
        let default_model = self.default_model.unwrap_or_default().trim().to_string();
        if default_model.is_empty() {
            return Err(ValidationError::EmptyModelId);
        }
        let backend = self.backend.ok_or(ValidationError::MissingBackend)?;
        Ok(FlexServInstance {
            tenant_url: tenant_url.to_string(),
            tapis_user,
            default_model,
            model_revision: self.model_revision,
            hf_token: self.hf_token,
            default_embedding_model: self.default_embedding_model,
            backend,
        })
    }
}

impl Default for FlexServInstanceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FlexServInstance {
    /// Start a builder for [FlexServInstance] (use [FlexServInstanceBuilder::build] for validation).
    pub fn builder() -> FlexServInstanceBuilder {
        FlexServInstanceBuilder::new()
    }

    /// Build from [TapisConfig] and [ModelConfig] (no validation; tenant_url normalized if no scheme).
    pub fn from_configs(tapis: &TapisConfig, model: &ModelConfig, backend: Backend) -> Self {
        FlexServInstance {
            tenant_url: normalize_tenant_url(&tapis.tenant_url),
            tapis_user: tapis.tapis_user.clone(),
            default_model: model.model_id.clone(),
            model_revision: model.model_revision.clone(),
            hf_token: model.hf_token.clone(),
            default_embedding_model: model.default_embedding_model.clone(),
            backend,
        }
    }

    pub fn new(
        tenant_url: String,
        tapis_user: String,
        default_model: String,
        model_revision: Option<String>,
        hf_token: Option<String>,
        default_embedding_model: Option<String>,
        backend: Backend,
    ) -> Self {
        FlexServInstance {
            tenant_url: normalize_tenant_url(&tenant_url),
            tapis_user,
            default_model,
            model_revision,
            hf_token,
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
            None, 
            None,
            Backend::Transformers {
                command_prefix: vec!["python".to_string()],
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
            None, 
            None,
            Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            },
        );

        let hash = server.deployment_hash();
        assert!(!hash.is_empty());
        // Hash should be consistent
        assert_eq!(hash, server.deployment_hash());
    }

    #[test]
    fn test_builder_ok() {
        let server = FlexServInstance::builder()
            .tenant_url("https://tacc.tapis.io")
            .tapis_user("myuser")
            .model("openai-community/gpt2")
            .backend(Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            })
            .build()
            .unwrap();
        assert_eq!(server.tenant_url, "https://tacc.tapis.io");
        assert_eq!(server.tapis_user, "myuser");
        assert_eq!(server.default_model, "openai-community/gpt2");
    }

    #[test]
    fn test_builder_validation_empty_user() {
        let err = FlexServInstance::builder()
            .tenant_url("https://tacc.tapis.io")
            .model("gpt2")
            .backend(Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            })
            .build()
            .unwrap_err();
        assert!(matches!(err, ValidationError::EmptyTapisUser));
    }

    #[test]
    fn test_builder_validation_empty_model() {
        let err = FlexServInstance::builder()
            .tenant_url("https://tacc.tapis.io")
            .tapis_user("u")
            .backend(Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            })
            .build()
            .unwrap_err();
        assert!(matches!(err, ValidationError::EmptyModelId));
    }

    #[test]
    fn test_builder_validation_bad_url() {
        let err = FlexServInstance::builder()
            .tenant_url("not-a-url")
            .tapis_user("u")
            .model("m")
            .backend(Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            })
            .build()
            .unwrap_err();
        assert!(matches!(err, ValidationError::InvalidTenantUrl(_)));
    }

    #[test]
    fn test_from_configs() {
        let tapis = TapisConfig {
            tenant_url: "https://tacc.tapis.io".to_string(),
            tapis_user: "u".to_string(),
            tapis_token: "token".to_string(),
        };
        let model = ModelConfig {
            model_id: "openai-community/gpt2".to_string(),
            model_revision: Some("main".to_string()),
            hf_token: None,
            default_embedding_model: None,
        };
        let server = FlexServInstance::from_configs(&tapis, &model, Backend::Transformers {
            command_prefix: vec!["python".to_string()],
        });
        assert_eq!(server.tenant_url, tapis.tenant_url);
        assert_eq!(server.default_model, model.model_id);
    }

    #[test]
    fn test_builder_normalizes_tenant_url() {
        let server = FlexServInstance::builder()
            .tenant_url("tacc.tapis.io")
            .tapis_user("u")
            .model("gpt2")
            .backend(Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            })
            .build()
            .unwrap();
        assert_eq!(server.tenant_url, "https://tacc.tapis.io");
    }
}
