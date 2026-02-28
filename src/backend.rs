//! Backend parameter sets and builders for pod/HPC deployment.
//!
//! - **PodParameterSet** = parameters for k8 pod (command, arguments, environment_variables).
//! - **HPCParameterSet** = tapis-sdk `JobParameterSet` for HPC job submission.
//! - **BackendParameterSetBuilder** = trait implemented by each backend builder; provides
//!   `build_params_for_pod(server)` -> `PodParameterSet` and `build_params_for_hpc(server)` -> `HPCParameterSet`.

use crate::server::FlexServInstance;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tapis_sdk::jobs::models::JobParameterSet;

/// Supported ML inference backends.
// TODO: The `command` field on each variant is not in use right now; the pod always runs the default startup only.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[serde(rename = "transformers")]
    Transformers {
        #[serde(default)]
        command: Vec<String>,
    },
    #[serde(rename = "vllm")]
    VLlm {
        #[serde(default)]
        command: Vec<String>,
    },
    #[serde(rename = "sglang")]
    SGLang {
        #[serde(default)]
        command: Vec<String>,
    },
    #[serde(rename = "trtllm")]
    TrtLlm {
        #[serde(default)]
        command: Vec<String>,
    },
}

/// Default command to start each backend in the FlexServ pod image. Only Transformers has a
/// defined startup script; VLlm, SGLang, and TrtLlm return None until their startup paths exist.
fn default_pod_command(backend: &Backend) -> Option<Vec<String>> {
    match backend {
        Backend::Transformers { .. } => Some(vec![
            "/app/venvs/transformers/bin/python".to_string(),
            "/app/flexserv/python/backend/transformers/backend_server.py".to_string(),
        ]),
        Backend::VLlm { .. } | Backend::SGLang { .. } | Backend::TrtLlm { .. } => None,
    }
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

    /// User-supplied commands that may be executed (e.g. after or alongside the default startup). Not passed as arguments to the default script.
    pub fn command(&self) -> &[String] {
        match self {
            Backend::Transformers { command, .. } => command,
            Backend::VLlm { command, .. } => command,
            Backend::SGLang { command, .. } => command,
            Backend::TrtLlm { command, .. } => command,
        }
    }

    /// Returns a builder that implements [BackendParameterSetBuilder]. Pod runs the default startup; [command](Backend::command) is for user commands that may be executed separately.
    pub fn parameter_set_builder(&self) -> Box<dyn BackendParameterSetBuilder> {
        let command = default_pod_command(self);
        match self {
            Backend::Transformers { .. } => Box::new(TransformersParameterSetBuilder::new(command)),
            Backend::VLlm { .. } => Box::new(VLlmParameterSetBuilder::new(command)),
            Backend::SGLang { .. } => Box::new(SGLangParameterSetBuilder::new(command)),
            Backend::TrtLlm { .. } => Box::new(TrtLlmParameterSetBuilder::new(command)),
        }
    }
}

/// Parameters for a pod: command, arguments, and environment variables.
/// Use `${pods:secrets:KEY}` in env values to reference secret_map entries.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PodParameterSet {
    /// Optional command (e.g. executable + script).
    pub command: Option<Vec<String>>,
    /// Optional arguments to the command.
    pub arguments: Option<Vec<String>>,
    /// Environment variables to inject into the pod. 
    #[serde(default)]
    pub environment_variables: Option<HashMap<String, Value>>,
}

/// HPC job parameter set (tapis-sdk [JobParameterSet]).
pub type HPCParameterSet = JobParameterSet;

/// Trait to build backend parameter sets for pod or HPC deployment.
/// Implemented by each backend's parameter set builder (e.g. [TransformersParameterSetBuilder]).
pub trait BackendParameterSetBuilder {
    fn build_params_for_pod(&self, server: &FlexServInstance) -> PodParameterSet;
    fn build_params_for_hpc(&self, server: &FlexServInstance) -> HPCParameterSet;
}

/// Push a single param (key, value) onto the arguments list (e.g. `--key` and value string).
fn push_param(args: &mut Vec<String>, key: &str, value: &Value) {
    let flag = format!("--{}", key.replace('_', "-"));
    match value {
        Value::Bool(b) => {
            if *b {
                args.push(flag);
            }
        }
        Value::Null => {}
        v => {
            args.push(flag);
            let s = serde_json::to_string(v).unwrap_or_default();
            let val = if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
                s[1..s.len() - 1].replace("\\\"", "\"")
            } else {
                s
            };
            args.push(val);
        }
    }
}

/// Builder for Transformers parameter set (state is PodParameterSet-shaped).
pub struct TransformersParameterSetBuilder {
    command: Option<Vec<String>>,
    arguments: Vec<String>,
    environment_variables: HashMap<String, Value>,
}

impl BackendParameterSetBuilder for TransformersParameterSetBuilder {
    fn build_params_for_pod(&self, _server: &FlexServInstance) -> PodParameterSet {
        // Always run the default FlexServ startup script; default args + builder args. User commands (backend.command()) are for separate execution, not as args to the default script.
        let mut arguments = vec![
            "--host".to_string(),
            "0.0.0.0".to_string(),
            "--port".to_string(),
            "8000".to_string(),
        ];
        arguments.extend(self.arguments.clone());
        PodParameterSet {
            command: self.command.clone(),
            arguments: Some(arguments),
            environment_variables: Some(self.environment_variables.clone()),
        }
    }

    fn build_params_for_hpc(&self, _server: &FlexServInstance) -> HPCParameterSet {
        HPCParameterSet::default()
    }
}

impl TransformersParameterSetBuilder {
    pub fn new(command: Option<Vec<String>>) -> Self {
        Self {
            command,
            arguments: Vec::new(),
            environment_variables: HashMap::new(),
        }
    }

    pub fn default_model(mut self, model: &str) -> Self {
        push_param(&mut self.arguments, "default-model", &Value::String(model.to_string()));
        self
    }

    pub fn default_embedding_model(mut self, model: &str) -> Self {
        push_param(&mut self.arguments, "default-embedding-model", &Value::String(model.to_string()));
        self
    }

    pub fn host(mut self, host: &str) -> Self {
        push_param(&mut self.arguments, "host", &Value::String(host.to_string()));
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        push_param(&mut self.arguments, "port", &Value::Number(serde_json::Number::from(port as u64)));
        self
    }

    pub fn device(mut self, device: &str) -> Self {
        push_param(&mut self.arguments, "device", &Value::String(device.to_string()));
        self
    }

    pub fn dtype(mut self, dtype: &str) -> Self {
        push_param(&mut self.arguments, "dtype", &Value::String(dtype.to_string()));
        self
    }

    pub fn continuous_batching(mut self, enabled: bool) -> Self {
        push_param(&mut self.arguments, "continuous-batching", &Value::Bool(enabled));
        self
    }

    pub fn flexserv_token(mut self, token: &str) -> Self {
        push_param(&mut self.arguments, "flexserv-token", &Value::String(token.to_string()));
        self
    }

    pub fn force_default_model(mut self, force: bool) -> Self {
        push_param(&mut self.arguments, "force-default-model", &Value::Bool(force));
        self
    }

    pub fn force_default_embedding_model(mut self, force: bool) -> Self {
        push_param(&mut self.arguments, "force-default-embedding-model", &Value::Bool(force));
        self
    }

    pub fn log_level(mut self, level: &str) -> Self {
        push_param(&mut self.arguments, "log-level", &Value::String(level.to_string()));
        self
    }

    pub fn quantization(mut self, quant: &str) -> Self {
        push_param(&mut self.arguments, "quantization", &Value::String(quant.to_string()));
        self
    }

    pub fn trust_remote_code(mut self, trust: bool) -> Self {
        push_param(&mut self.arguments, "trust-remote-code", &Value::Bool(trust));
        self
    }

    pub fn attn_implementation(mut self, implementation: &str) -> Self {
        push_param(&mut self.arguments, "attn-implementation", &Value::String(implementation.to_string()));
        self
    }

    pub fn enable_cors(mut self, enable: bool) -> Self {
        push_param(&mut self.arguments, "enable-cors", &Value::Bool(enable));
        self
    }

    pub fn non_blocking(mut self, non_blocking: bool) -> Self {
        push_param(&mut self.arguments, "non-blocking", &Value::Bool(non_blocking));
        self
    }

    pub fn insert_env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.environment_variables
            .insert(key.into(), Value::String(value.into()));
        self
    }
}

/// Builder for vLLM parameter set (state is PodParameterSet-shaped).
pub struct VLlmParameterSetBuilder {
    command: Option<Vec<String>>,
    arguments: Vec<String>,
    environment_variables: HashMap<String, Value>,
}

impl BackendParameterSetBuilder for VLlmParameterSetBuilder {
    fn build_params_for_pod(&self, _server: &FlexServInstance) -> PodParameterSet {
        let arguments = self.arguments.clone();
        PodParameterSet {
            command: self.command.clone(),
            arguments: Some(arguments),
            environment_variables: Some(self.environment_variables.clone()),
        }
    }

    fn build_params_for_hpc(&self, _server: &FlexServInstance) -> HPCParameterSet {
        HPCParameterSet::default()
    }
}

impl VLlmParameterSetBuilder {
    pub fn new(command: Option<Vec<String>>) -> Self {
        Self {
            command,
            arguments: Vec::new(),
            environment_variables: HashMap::new(),
        }
    }

    pub fn tensor_parallel_size(mut self, size: u32) -> Self {
        push_param(&mut self.arguments, "tensor_parallel_size", &Value::Number(serde_json::Number::from(size as u64)));
        self
    }

    pub fn pipeline_parallel_size(mut self, size: u32) -> Self {
        push_param(&mut self.arguments, "pipeline_parallel_size", &Value::Number(serde_json::Number::from(size as u64)));
        self
    }

    pub fn max_model_len(mut self, len: u32) -> Self {
        push_param(&mut self.arguments, "max_model_len", &Value::Number(serde_json::Number::from(len as u64)));
        self
    }

    pub fn gpu_memory_utilization(mut self, util: f32) -> Self {
        push_param(&mut self.arguments, "gpu_memory_utilization", &Value::Number(serde_json::Number::from_f64(util as f64).unwrap()));
        self
    }

    pub fn insert_env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.environment_variables
            .insert(key.into(), Value::String(value.into()));
        self
    }
}

/// Builder for SGLang parameter set (state is PodParameterSet-shaped).
pub struct SGLangParameterSetBuilder {
    command: Option<Vec<String>>,
    arguments: Vec<String>,
    environment_variables: HashMap<String, Value>,
}

impl BackendParameterSetBuilder for SGLangParameterSetBuilder {
    fn build_params_for_pod(&self, _server: &FlexServInstance) -> PodParameterSet {
        let arguments = self.arguments.clone();
        PodParameterSet {
            command: self.command.clone(),
            arguments: Some(arguments),
            environment_variables: Some(self.environment_variables.clone()),
        }
    }

    fn build_params_for_hpc(&self, _server: &FlexServInstance) -> HPCParameterSet {
        HPCParameterSet::default()
    }
}

impl SGLangParameterSetBuilder {
    pub fn new(command: Option<Vec<String>>) -> Self {
        Self {
            command,
            arguments: Vec::new(),
            environment_variables: HashMap::new(),
        }
    }

    pub fn tp_size(mut self, size: u32) -> Self {
        push_param(&mut self.arguments, "tp_size", &Value::Number(serde_json::Number::from(size as u64)));
        self
    }

    pub fn mem_fraction_static(mut self, fraction: f32) -> Self {
        push_param(&mut self.arguments, "mem_fraction_static", &Value::Number(serde_json::Number::from_f64(fraction as f64).unwrap()));
        self
    }

    pub fn insert_env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.environment_variables
            .insert(key.into(), Value::String(value.into()));
        self
    }
}

/// Builder for TRT-LLM parameter set (state is PodParameterSet-shaped).
pub struct TrtLlmParameterSetBuilder {
    command: Option<Vec<String>>,
    arguments: Vec<String>,
    environment_variables: HashMap<String, Value>,
}

impl BackendParameterSetBuilder for TrtLlmParameterSetBuilder {
    fn build_params_for_pod(&self, _server: &FlexServInstance) -> PodParameterSet {
        let arguments = self.arguments.clone();
        PodParameterSet {
            command: self.command.clone(),
            arguments: Some(arguments),
            environment_variables: Some(self.environment_variables.clone()),
        }
    }

    fn build_params_for_hpc(&self, _server: &FlexServInstance) -> HPCParameterSet {
        HPCParameterSet::default()
    }
}

impl TrtLlmParameterSetBuilder {
    pub fn new(command: Option<Vec<String>>) -> Self {
        Self {
            command,
            arguments: Vec::new(),
            environment_variables: HashMap::new(),
        }
    }

    pub fn max_batch_size(mut self, size: u32) -> Self {
        push_param(&mut self.arguments, "max_batch_size", &Value::Number(serde_json::Number::from(size as u64)));
        self
    }

    pub fn max_input_len(mut self, len: u32) -> Self {
        push_param(&mut self.arguments, "max_input_len", &Value::Number(serde_json::Number::from(len as u64)));
        self
    }

    pub fn insert_env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.environment_variables
            .insert(key.into(), Value::String(value.into()));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::FlexServInstance;

    #[test]
    fn test_backend_as_str() {
        let backend = Backend::Transformers { command: vec![] };
        assert_eq!(backend.as_str(), "transformers");
    }

    #[test]
    fn test_backend_parameter_set() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "user".to_string(),
            "gpt2".to_string(),
            None,
            None,
            None,
            Backend::Transformers {
                command: vec![],
            },
        );
        let pod_params = TransformersParameterSetBuilder::new(None)
            .insert_env_var("ENV_VAR", "test")
            .build_params_for_pod(&server);
        assert_eq!(
            pod_params.environment_variables.as_ref().unwrap().get("ENV_VAR"),
            Some(&serde_json::json!("test"))
        );
    }

    #[test]
    fn test_transformers_builder() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "user".to_string(),
            "meta-llama/Llama-2-7b".to_string(),
            None,
            None,
            None,
            Backend::Transformers {
                command: vec![],
            },
        );
        let builder = TransformersParameterSetBuilder::new(None)
            .default_model("meta-llama/Llama-2-7b")
            .port(8080)
            .trust_remote_code(true);
        let pod_params = builder.build_params_for_pod(&server);
        let pod_args = pod_params.arguments.as_ref().unwrap();
        assert!(pod_args.contains(&"--port".to_string()));
        // Builder's .port(8080) overrides the default, so we expect 8080 in args.
        assert!(pod_args.contains(&"8080".to_string()));
    }

    #[test]
    fn test_to_cli_args() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "user".to_string(),
            "gpt2".to_string(),
            None,
            None,
            None,
            Backend::Transformers {
                command: vec![],
            },
        );
        let pod_params = TransformersParameterSetBuilder::new(None)
            .build_params_for_pod(&server);
        let args = pod_params.arguments.as_ref().unwrap();
        assert!(args.contains(&"--host".to_string()));
        assert!(args.contains(&"0.0.0.0".to_string()));
        assert!(args.contains(&"--port".to_string()));
        assert!(args.contains(&"8000".to_string()));
    }

    #[test]
    fn test_to_cli_args_excluding() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "user".to_string(),
            "gpt2".to_string(),
            None,
            None,
            None,
            Backend::Transformers {
                command: vec![],
            },
        );
        let pod_params = TransformersParameterSetBuilder::new(None)
            .build_params_for_pod(&server);
        let args = pod_params.arguments.as_ref().unwrap();
        assert!(args.contains(&"--host".to_string()));
        assert!(args.contains(&"--port".to_string()));
        assert!(args.contains(&"8000".to_string()));
    }

    #[test]
    fn test_command_accessor() {
        // User's command (appended after default startup) is preserved.
        let backend = Backend::Transformers {
            command: vec!["python".to_string(), "serve.py".to_string()],
        };
        let cmd = backend.command();
        assert_eq!(cmd, &["python", "serve.py"]);
    }

    #[test]
    fn test_to_cli_args_is_stable() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "user".to_string(),
            "gpt2".to_string(),
            None,
            None,
            None,
            Backend::Transformers {
                command: vec![],
            },
        );
        let pod_params = TransformersParameterSetBuilder::new(None)
            .build_params_for_pod(&server);
        let args = pod_params.arguments.as_ref().unwrap();
        assert!(args.contains(&"--port".to_string()));
        assert!(args.contains(&"8000".to_string()));
    }

    #[test]
    fn test_build_params_for_pod_vs_hpc_transformers() {
        let backend = Backend::Transformers { command: vec![] };
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "user".to_string(),
            "openai-community/gpt2".to_string(),
            None,
            None,
            None,
            backend.clone(),
        );

        let builder = backend.parameter_set_builder();
        let pod_params = builder.build_params_for_pod(&server);
        let _hpc_params = builder.build_params_for_hpc(&server);

        assert!(pod_params.command.is_some());
        assert!(pod_params.arguments.is_some());
    }
}
