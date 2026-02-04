use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Supported ML inference backends
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[serde(rename = "transformers")]
    Transformers { command: Vec<String> },
    #[serde(rename = "vllm")]
    VLlm { command: Vec<String> },
    #[serde(rename = "sglang")]
    SGLang { command: Vec<String> },
    #[serde(rename = "trtllm")]
    TrtLlm { command: Vec<String> },
}

impl Backend {
    pub fn as_str(&self) -> &str {
        match self {
            Backend::Transformers { .. } => "transformers",
            Backend::VLlm { .. } => "vllm",
            Backend::SGLang { .. } => "sglang",
            Backend::TrtLlm { .. } => "trtllm",
        }
    }

    /// Create a Transformers parameter builder
    pub fn transformers(&self) -> TransformersParametersBuilder {
        match self {
            Backend::Transformers { command } => {
                TransformersParametersBuilder::new(command.clone())
            }
            _ => panic!("Backend is not Transformers"),
        }
    }

    /// Create a vLLM parameter builder
    pub fn vllm(&self) -> VLlmParametersBuilder {
        match self {
            Backend::VLlm { command } => VLlmParametersBuilder::new(command.clone()),
            _ => panic!("Backend is not vLLM"),
        }
    }

    /// Create an SGLang parameter builder
    pub fn sglang(&self) -> SGLangParametersBuilder {
        match self {
            Backend::SGLang { command } => SGLangParametersBuilder::new(command.clone()),
            _ => panic!("Backend is not SGLang"),
        }
    }

    /// Create a TRT-LLM parameter builder
    pub fn trtllm(&self) -> TrtLlmParametersBuilder {
        match self {
            Backend::TrtLlm { command } => TrtLlmParametersBuilder::new(command.clone()),
            _ => panic!("Backend is not TRT-LLM"),
        }
    }
}

/// Backend-specific parameters
/// This is a flexible JSON object that varies by backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendParameters {
    pub command: Vec<String>,
    pub params: HashMap<String, Value>,
    pub env: HashMap<String, String>,
}

impl BackendParameters {
    pub fn new(command: Vec<String>) -> Self {
        Self {
            command,
            params: HashMap::new(),
            env: HashMap::new(),
        }
    }

    pub fn insert_param<T: Serialize>(&mut self, key: impl Into<String>, value: T) -> &mut Self {
        self.params
            .insert(key.into(), serde_json::to_value(value).unwrap());
        self
    }

    pub fn insert_env(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.env.insert(key.into(), value.into());
        self
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.params.get(key)
    }
}

/// Builder for transformers parameters
pub struct TransformersParametersBuilder {
    params: BackendParameters,
}

impl TransformersParametersBuilder {
    pub fn new(command: Vec<String>) -> Self {
        Self {
            params: BackendParameters::new(command),
        }
    }

    pub fn default_model(mut self, model: &str) -> Self {
        self.params.insert_param("default-model", model);
        self
    }

    pub fn default_embedding_model(mut self, model: &str) -> Self {
        self.params.insert_param("default-embedding-model", model);
        self
    }

    pub fn host(mut self, host: &str) -> Self {
        self.params.insert_param("host", host);
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.params.insert_param("port", port);
        self
    }

    pub fn device(mut self, device: &str) -> Self {
        self.params.insert_param("device", device);
        self
    }

    pub fn dtype(mut self, dtype: &str) -> Self {
        self.params.insert_param("dtype", dtype);
        self
    }

    pub fn continuous_batching(mut self, enabled: bool) -> Self {
        self.params.insert_param("continuous-batching", enabled);
        self
    }

    pub fn flexserv_token(mut self, token: &str) -> Self {
        self.params.insert_param("flexserv-token", token);
        self
    }

    pub fn force_default_model(mut self, force: bool) -> Self {
        self.params.insert_param("force-default-model", force);
        self
    }

    pub fn force_default_embedding_model(mut self, force: bool) -> Self {
        self.params
            .insert_param("force-default-embedding-model", force);
        self
    }

    pub fn log_level(mut self, level: &str) -> Self {
        self.params.insert_param("log-level", level);
        self
    }

    pub fn quantization(mut self, quant: &str) -> Self {
        self.params.insert_param("quantization", quant);
        self
    }

    pub fn trust_remote_code(mut self, trust: bool) -> Self {
        self.params.insert_param("trust-remote-code", trust);
        self
    }

    pub fn attn_implementation(mut self, implementation: &str) -> Self {
        self.params
            .insert_param("attn-implementation", implementation);
        self
    }

    pub fn enable_cors(mut self, enable: bool) -> Self {
        self.params.insert_param("enable-cors", enable);
        self
    }

    pub fn non_blocking(mut self, non_blocking: bool) -> Self {
        self.params.insert_param("non-blocking", non_blocking);
        self
    }

    pub fn insert_env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert_env(key, value);
        self
    }

    pub fn build(self) -> BackendParameters {
        self.params
    }
}

/// Builder for vLLM parameters
pub struct VLlmParametersBuilder {
    params: BackendParameters,
}

impl VLlmParametersBuilder {
    pub fn new(command: Vec<String>) -> Self {
        Self {
            params: BackendParameters::new(command),
        }
    }

    pub fn tensor_parallel_size(mut self, size: u32) -> Self {
        self.params.insert_param("tensor_parallel_size", size);
        self
    }

    pub fn pipeline_parallel_size(mut self, size: u32) -> Self {
        self.params.insert_param("pipeline_parallel_size", size);
        self
    }

    pub fn max_model_len(mut self, len: u32) -> Self {
        self.params.insert_param("max_model_len", len);
        self
    }

    pub fn gpu_memory_utilization(mut self, util: f32) -> Self {
        self.params.insert_param("gpu_memory_utilization", util);
        self
    }

    pub fn build(self) -> BackendParameters {
        self.params
    }
}

/// Builder for SGLang parameters
pub struct SGLangParametersBuilder {
    params: BackendParameters,
}

impl SGLangParametersBuilder {
    pub fn new(command: Vec<String>) -> Self {
        Self {
            params: BackendParameters::new(command),
        }
    }

    pub fn tp_size(mut self, size: u32) -> Self {
        self.params.insert_param("tp_size", size);
        self
    }

    pub fn mem_fraction_static(mut self, fraction: f32) -> Self {
        self.params.insert_param("mem_fraction_static", fraction);
        self
    }

    pub fn build(self) -> BackendParameters {
        self.params
    }
}

/// Builder for TRT-LLM parameters
pub struct TrtLlmParametersBuilder {
    params: BackendParameters,
}

impl TrtLlmParametersBuilder {
    pub fn new(command: Vec<String>) -> Self {
        Self {
            params: BackendParameters::new(command),
        }
    }

    pub fn max_batch_size(mut self, size: u32) -> Self {
        self.params.insert_param("max_batch_size", size);
        self
    }

    pub fn max_input_len(mut self, len: u32) -> Self {
        self.params.insert_param("max_input_len", len);
        self
    }

    pub fn build(self) -> BackendParameters {
        self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_as_str() {
        let backend = Backend::Transformers {
            command: vec!["python".to_string()],
        };
        assert_eq!(backend.as_str(), "transformers");
    }

    #[test]
    fn test_backend_parameters() {
        let mut params = BackendParameters::new(vec!["python".to_string()]);
        params.insert_param("test", "value");
        params.insert_env("ENV_VAR", "test");

        assert_eq!(params.get("test").unwrap(), "value");
        assert_eq!(params.env.get("ENV_VAR").unwrap(), "test");
    }

    #[test]
    fn test_transformers_builder() {
        let params = TransformersParametersBuilder::new(vec!["python".to_string()])
            .default_model("meta-llama/Llama-2-7b")
            .port(8080)
            .trust_remote_code(true)
            .build();

        assert_eq!(
            params.get("default-model").unwrap(),
            "meta-llama/Llama-2-7b"
        );
        assert_eq!(params.get("port").unwrap(), 8080);
    }
}
