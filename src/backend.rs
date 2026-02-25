//! Backend parameter sets and builders for pod/HPC deployment.
//!
//! - **Parameter set** = built result (`BackendParameterSet`): command, params, env.
//! - **Parameter set builder** = builder type (e.g. `TransformersParameterSetBuilder`) with `.build()`.
//! - **Trait** `BuildBackendParameterSet` on `Backend`: `build_params_for_pod(server)` and `build_params_for_hpc(server)` produce a `BackendParameterSet` for that target.

use crate::server::FlexServInstance;
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

    /// Default command prefix for running this backend in the FlexServ pod image.
    /// Returns [interpreter, script_path] or [python, -m, module] as appropriate.
    /// Non-Transformers backends use a placeholder until pod image support is added.
    pub fn default_pod_command_prefix(&self) -> Vec<String> {
        match self {
            Backend::Transformers { .. } => vec![
                "/app/venvs/transformers/bin/python".to_string(),
                "/app/flexserv/python/backend/transformers/backend_server.py".to_string(),
            ],
            Backend::VLlm { .. } => vec![
                "/bin/echo".to_string(),
                "vllm backend: pod command not yet implemented".to_string(),
            ],
            Backend::SGLang { .. } => vec![
                "/bin/echo".to_string(),
                "sglang backend: pod command not yet implemented".to_string(),
            ],
            Backend::TrtLlm { .. } => vec![
                "/bin/echo".to_string(),
                "trtllm backend: pod command not yet implemented".to_string(),
            ],
        }
    }

    /// Create a Transformers parameter set builder
    pub fn transformers(&self) -> TransformersParameterSetBuilder {
        match self {
            Backend::Transformers { command } => {
                TransformersParameterSetBuilder::new(command.clone())
            }
            _ => panic!("Backend is not Transformers"),
        }
    }

    /// Create a vLLM parameter set builder
    pub fn vllm(&self) -> VLlmParameterSetBuilder {
        match self {
            Backend::VLlm { command } => VLlmParameterSetBuilder::new(command.clone()),
            _ => panic!("Backend is not vLLM"),
        }
    }

    /// Create an SGLang parameter set builder
    pub fn sglang(&self) -> SGLangParameterSetBuilder {
        match self {
            Backend::SGLang { command } => SGLangParameterSetBuilder::new(command.clone()),
            _ => panic!("Backend is not SGLang"),
        }
    }

    /// Create a TRT-LLM parameter set builder
    pub fn trtllm(&self) -> TrtLlmParameterSetBuilder {
        match self {
            Backend::TrtLlm { command } => TrtLlmParameterSetBuilder::new(command.clone()),
            _ => panic!("Backend is not TRT-LLM"),
        }
    }
}

/// Trait to build backend parameter sets for pod or HPC deployment.
pub trait BuildBackendParameterSet {
    fn build_params_for_pod(&self, server: &FlexServInstance) -> BackendParameterSet;
    fn build_params_for_hpc(&self, server: &FlexServInstance) -> BackendParameterSet;
}

impl BuildBackendParameterSet for Backend {
    fn build_params_for_pod(&self, server: &FlexServInstance) -> BackendParameterSet {
        match self {
            Backend::Transformers { .. } => self
                .transformers()
                .default_model(&server.default_model)
                .host("0.0.0.0")
                .port(8000)
                .build(),
            Backend::VLlm { .. } => self.vllm().build(),
            Backend::SGLang { .. } => self.sglang().build(),
            Backend::TrtLlm { .. } => self.trtllm().build(),
        }
    }

    fn build_params_for_hpc(&self, server: &FlexServInstance) -> BackendParameterSet {
        match self {
            Backend::Transformers { .. } => self
                .transformers()
                .default_model(&server.default_model)
                .build(),
            Backend::VLlm { .. } => self.vllm().build(),
            Backend::SGLang { .. } => self.sglang().build(),
            Backend::TrtLlm { .. } => self.trtllm().build(),
        }
    }
}

/// Built parameter set for a backend (command, params, env). Used for pod or HPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendParameterSet {
    pub command: Vec<String>,
    pub params: HashMap<String, Value>,
    pub env: HashMap<String, String>,
}

impl BackendParameterSet {
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

    /// Convert params to CLI args (e.g. `--host 0.0.0.0 --port 8000`).
    /// Skips `flexserv-token` so the pod script can inject `"$FLEXSERV_TOKEN"` at runtime.
    /// Skips `default-model` because the model path is passed as a positional argument
    /// in the pod exec line (`$MODEL_REPO/$MODEL_NAME`), not as a flag.
    /// Bool true => `--key`, bool false => omitted.
    pub fn to_cli_args(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (key, value) in &self.params {
            if key == "flexserv-token" || key == "default-model" {
                continue;
            }
            match value {
                Value::Bool(b) => {
                    if *b {
                        out.push(format!("--{}", key.replace('_', "-")));
                    }
                }
                Value::Null => {}
                v => {
                    out.push(format!("--{}", key.replace('_', "-")));
                    let s = serde_json::to_string(v).unwrap_or_default();
                    let val = if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
                        s[1..s.len() - 1].replace("\\\"", "\"")
                    } else {
                        s
                    };
                    out.push(val);
                }
            }
        }
        out
    }
}

/// Builder for Transformers parameter set (use .build() or Backend::build_params_for_pod/hpc).
pub struct TransformersParameterSetBuilder {
    params: BackendParameterSet,
}

impl TransformersParameterSetBuilder {
    pub fn new(command: Vec<String>) -> Self {
        Self {
            params: BackendParameterSet::new(command),
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

    pub fn build(self) -> BackendParameterSet {
        self.params
    }
}

/// Builder for vLLM parameter set.
pub struct VLlmParameterSetBuilder {
    params: BackendParameterSet,
}

impl VLlmParameterSetBuilder {
    pub fn new(command: Vec<String>) -> Self {
        Self {
            params: BackendParameterSet::new(command),
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

    pub fn build(self) -> BackendParameterSet {
        self.params
    }
}

/// Builder for SGLang parameter set.
pub struct SGLangParameterSetBuilder {
    params: BackendParameterSet,
}

impl SGLangParameterSetBuilder {
    pub fn new(command: Vec<String>) -> Self {
        Self {
            params: BackendParameterSet::new(command),
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

    pub fn build(self) -> BackendParameterSet {
        self.params
    }
}

/// Builder for TRT-LLM parameter set.
pub struct TrtLlmParameterSetBuilder {
    params: BackendParameterSet,
}

impl TrtLlmParameterSetBuilder {
    pub fn new(command: Vec<String>) -> Self {
        Self {
            params: BackendParameterSet::new(command),
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

    pub fn build(self) -> BackendParameterSet {
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
    fn test_backend_parameter_set() {
        let mut params = BackendParameterSet::new(vec!["python".to_string()]);
        params.insert_param("test", "value");
        params.insert_env("ENV_VAR", "test");

        assert_eq!(params.get("test").unwrap(), "value");
        assert_eq!(params.env.get("ENV_VAR").unwrap(), "test");
    }

    #[test]
    fn test_transformers_builder() {
        let params = TransformersParameterSetBuilder::new(vec!["python".to_string()])
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

    #[test]
    fn test_to_cli_args() {
        let params = TransformersParameterSetBuilder::new(vec!["python".to_string()])
            .host("0.0.0.0")
            .port(8000)
            .flexserv_token("secret")
            .build();
        let args = params.to_cli_args();
        assert!(args.contains(&"--host".to_string()));
        assert!(args.contains(&"0.0.0.0".to_string()));
        assert!(args.contains(&"--port".to_string()));
        assert!(args.contains(&"8000".to_string()));
        assert!(!args.iter().any(|a| a == "secret"), "flexserv-token omitted for script");
    }

    #[test]
    fn test_default_pod_command_prefix() {
        let backend = Backend::Transformers { command: vec!["python".to_string()] };
        let prefix = backend.default_pod_command_prefix();
        assert_eq!(prefix.len(), 2);
        assert!(prefix[0].contains("python"));
        assert!(prefix[1].contains("backend_server"));
    }
}
