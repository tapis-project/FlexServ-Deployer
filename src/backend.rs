//! Backend parameter sets and builders for pod/HPC deployment.
//!
//! - **PodParameterSet** = parameters for k8 pod (command, arguments, environment_variables).
//! - **HPCParameterSet** = tapis-sdk `JobParameterSet` for HPC job submission.
//! - **BackendParameterSetBuilder** = trait implemented by each backend builder; provides
//!   `build_params_for_pod(server)` -> `PodParameterSet` and `build_params_for_hpc(server)` -> `HPCParameterSet`.

use crate::server::FlexServInstance;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use tapis_sdk::jobs::models::{JobArgSpec, JobParameterSet, KeyValuePair};

/// Supported ML inference backends.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[serde(rename = "transformers")]
    Transformers {
        /// Commands run inside the pod before the inference server starts (e.g. warmup, model pre-load).
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

    /// Commands to run inside the pod before the inference server starts (e.g. warmup, model pre-load).
    /// Not yet executed in pod deployment.
    pub fn command(&self) -> &[String] {
        match self {
            Backend::Transformers { command, .. } => command,
            Backend::VLlm { command, .. } => command,
            Backend::SGLang { command, .. } => command,
            Backend::TrtLlm { command, .. } => command,
        }
    }

    /// Returns a builder that implements [BackendParameterSetBuilder].
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
    /// Build the Tapis Jobs `parameterSet` needed to run this backend on HPC.
    fn build_params_for_hpc(&self, server: &FlexServInstance) -> HPCParameterSet;
}

fn hpc_job_arg(name: impl Into<String>, arg: impl Into<String>) -> JobArgSpec {
    JobArgSpec {
        name: Some(name.into()),
        description: None,
        include: Some(true),
        arg: Some(arg.into()),
        notes: None,
    }
}

fn hpc_env_var(key: impl Into<String>, value: impl Into<String>) -> KeyValuePair {
    KeyValuePair {
        key: Some(key.into()),
        value: Some(value.into()),
        description: None,
        include: Some(true),
        notes: None,
    }
}

fn set_builder_option(options: &mut BTreeMap<String, Value>, key: &str, value: impl Into<Value>) {
    options.insert(key.to_string(), value.into());
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(v) => v.clone(),
        other => other.to_string(),
    }
}

fn key_to_job_name(key: &str) -> String {
    let mut out = String::new();
    let mut upper = false;
    for c in key.chars() {
        if c == '-' || c == '_' {
            upper = true;
            continue;
        }
        if upper {
            out.push(c.to_ascii_uppercase());
            upper = false;
        } else {
            out.push(c);
        }
    }
    out
}

fn value_to_pod_args(flag: &str, value: &Value) -> Vec<String> {
    let mut args = Vec::new();
    match value {
        Value::Bool(true) => args.push(format!("--{}", flag)),
        Value::Bool(false) | Value::Null => {}
        Value::String(s) => {
            args.push(format!("--{}", flag));
            args.push(s.clone());
        }
        other => {
            args.push(format!("--{}", flag));
            args.push(other.to_string());
        }
    }
    args
}

fn value_to_hpc_args(flag: &str, value: &Value) -> Option<String> {
    match value {
        Value::Bool(true) => Some(format!("--{}", flag)),
        Value::Bool(false) | Value::Null => None,
        Value::String(s) => Some(format!("--{} {}", flag, s)),
        other => Some(format!("--{} {}", flag, other)),
    }
}

fn build_hpc_from_options(
    options: &BTreeMap<String, Value>,
    env: &HashMap<String, Value>,
) -> HPCParameterSet {
    let mut parameter_set = HPCParameterSet::new();
    parameter_set.app_args = Some(
        options
            .iter()
            .filter_map(|(flag, value)| {
                value_to_hpc_args(flag, value).map(|arg| hpc_job_arg(key_to_job_name(flag), arg))
            })
            .collect(),
    );
    parameter_set.env_variables = Some(
        env.iter()
            .map(|(key, value)| hpc_env_var(key.clone(), value_to_string(value)))
            .collect(),
    );
    parameter_set
}

/// Builder for Transformers parameter set (state is PodParameterSet-shaped).
pub struct TransformersParameterSetBuilder {
    command: Option<Vec<String>>,
    options: BTreeMap<String, Value>,
    environment_variables: HashMap<String, Value>,
}

impl BackendParameterSetBuilder for TransformersParameterSetBuilder {
    fn build_params_for_pod(&self, _server: &FlexServInstance) -> PodParameterSet {
        let mut merged = BTreeMap::new();
        merged.insert("host".to_string(), Value::String("0.0.0.0".to_string()));
        merged.insert("port".to_string(), Value::String("8000".to_string()));
        for (k, v) in &self.options {
            merged.insert(k.clone(), v.clone());
        }
        let mut arguments = Vec::new();
        for (flag, value) in &merged {
            arguments.extend(value_to_pod_args(flag, value));
        }
        PodParameterSet {
            command: self.command.clone(),
            arguments: Some(arguments),
            environment_variables: Some(self.environment_variables.clone()),
        }
    }

    fn build_params_for_hpc(&self, server: &FlexServInstance) -> HPCParameterSet {
        let mut merged = BTreeMap::new();
        merged.insert("device".to_string(), Value::String("auto".to_string()));
        merged.insert("dtype".to_string(), Value::String("bfloat16".to_string()));
        merged.insert(
            "attn-implementation".to_string(),
            Value::String("sdpa".to_string()),
        );
        merged.insert(
            "model-timeout".to_string(),
            Value::String("86400".to_string()),
        );
        merged.insert(
            "quantization".to_string(),
            Value::String("none".to_string()),
        );
        if let Some(model) = &server.default_embedding_model {
            merged.insert(
                "default-embedding-model".to_string(),
                Value::String(model.clone()),
            );
        }
        let allowed = [
            "device",
            "dtype",
            "continuous-batching",
            "force-default-embedding-model",
            "log-level",
            "quantization",
            "trust-remote-code",
            "attn-implementation",
            "enable-cors",
            "non-blocking",
            "default-embedding-model",
            "force-default-model",
        ];
        for (key, value) in &self.options {
            if allowed.contains(&key.as_str()) {
                merged.insert(key.clone(), value.clone());
            }
        }
        build_hpc_from_options(&merged, &self.environment_variables)
    }
}

impl TransformersParameterSetBuilder {
    pub fn new(command: Option<Vec<String>>) -> Self {
        Self {
            command,
            options: BTreeMap::new(),
            environment_variables: HashMap::new(),
        }
    }

    pub fn default_model(mut self, model: &str) -> Self {
        set_builder_option(&mut self.options, "default-model", model);
        self
    }

    pub fn default_embedding_model(mut self, model: &str) -> Self {
        set_builder_option(&mut self.options, "default-embedding-model", model);
        self
    }

    pub fn host(mut self, host: &str) -> Self {
        set_builder_option(&mut self.options, "host", host);
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        set_builder_option(&mut self.options, "port", port);
        self
    }

    pub fn device(mut self, device: &str) -> Self {
        set_builder_option(&mut self.options, "device", device);
        self
    }

    pub fn dtype(mut self, dtype: &str) -> Self {
        set_builder_option(&mut self.options, "dtype", dtype);
        self
    }

    pub fn continuous_batching(mut self, enabled: bool) -> Self {
        set_builder_option(&mut self.options, "continuous-batching", enabled);
        self
    }

    pub fn flexserv_token(mut self, token: &str) -> Self {
        set_builder_option(&mut self.options, "flexserv-token", token);
        self
    }

    pub fn force_default_model(mut self, force: bool) -> Self {
        set_builder_option(&mut self.options, "force-default-model", force);
        self
    }

    pub fn force_default_embedding_model(mut self, force: bool) -> Self {
        set_builder_option(&mut self.options, "force-default-embedding-model", force);
        self
    }

    pub fn log_level(mut self, level: &str) -> Self {
        set_builder_option(&mut self.options, "log-level", level);
        self
    }

    pub fn quantization(mut self, quant: &str) -> Self {
        set_builder_option(&mut self.options, "quantization", quant);
        self
    }

    pub fn trust_remote_code(mut self, trust: bool) -> Self {
        set_builder_option(&mut self.options, "trust-remote-code", trust);
        self
    }

    pub fn attn_implementation(mut self, implementation: &str) -> Self {
        set_builder_option(&mut self.options, "attn-implementation", implementation);
        self
    }

    pub fn enable_cors(mut self, enable: bool) -> Self {
        set_builder_option(&mut self.options, "enable-cors", enable);
        self
    }

    pub fn non_blocking(mut self, non_blocking: bool) -> Self {
        set_builder_option(&mut self.options, "non-blocking", non_blocking);
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
    options: BTreeMap<String, Value>,
    environment_variables: HashMap<String, Value>,
}

impl BackendParameterSetBuilder for VLlmParameterSetBuilder {
    fn build_params_for_pod(&self, _server: &FlexServInstance) -> PodParameterSet {
        let mut arguments = Vec::new();
        for (flag, value) in &self.options {
            arguments.extend(value_to_pod_args(flag, value));
        }
        PodParameterSet {
            command: self.command.clone(),
            arguments: Some(arguments),
            environment_variables: Some(self.environment_variables.clone()),
        }
    }

    fn build_params_for_hpc(&self, _server: &FlexServInstance) -> HPCParameterSet {
        build_hpc_from_options(&self.options, &self.environment_variables)
    }
}

impl VLlmParameterSetBuilder {
    pub fn new(command: Option<Vec<String>>) -> Self {
        Self {
            command,
            options: BTreeMap::new(),
            environment_variables: HashMap::new(),
        }
    }

    pub fn tensor_parallel_size(mut self, size: u32) -> Self {
        set_builder_option(&mut self.options, "tensor-parallel-size", size);
        self
    }

    pub fn pipeline_parallel_size(mut self, size: u32) -> Self {
        set_builder_option(&mut self.options, "pipeline-parallel-size", size);
        self
    }

    pub fn max_model_len(mut self, len: u32) -> Self {
        set_builder_option(&mut self.options, "max-model-len", len);
        self
    }

    pub fn gpu_memory_utilization(mut self, util: f32) -> Self {
        set_builder_option(&mut self.options, "gpu-memory-utilization", util);
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
    options: BTreeMap<String, Value>,
    environment_variables: HashMap<String, Value>,
}

impl BackendParameterSetBuilder for SGLangParameterSetBuilder {
    fn build_params_for_pod(&self, _server: &FlexServInstance) -> PodParameterSet {
        let mut arguments = Vec::new();
        for (flag, value) in &self.options {
            arguments.extend(value_to_pod_args(flag, value));
        }
        PodParameterSet {
            command: self.command.clone(),
            arguments: Some(arguments),
            environment_variables: Some(self.environment_variables.clone()),
        }
    }

    fn build_params_for_hpc(&self, _server: &FlexServInstance) -> HPCParameterSet {
        build_hpc_from_options(&self.options, &self.environment_variables)
    }
}

impl SGLangParameterSetBuilder {
    pub fn new(command: Option<Vec<String>>) -> Self {
        Self {
            command,
            options: BTreeMap::new(),
            environment_variables: HashMap::new(),
        }
    }

    pub fn tp_size(mut self, size: u32) -> Self {
        set_builder_option(&mut self.options, "tp-size", size);
        self
    }

    pub fn mem_fraction_static(mut self, fraction: f32) -> Self {
        set_builder_option(&mut self.options, "mem-fraction-static", fraction);
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
    options: BTreeMap<String, Value>,
    environment_variables: HashMap<String, Value>,
}

impl BackendParameterSetBuilder for TrtLlmParameterSetBuilder {
    fn build_params_for_pod(&self, _server: &FlexServInstance) -> PodParameterSet {
        let mut arguments = Vec::new();
        for (flag, value) in &self.options {
            arguments.extend(value_to_pod_args(flag, value));
        }
        PodParameterSet {
            command: self.command.clone(),
            arguments: Some(arguments),
            environment_variables: Some(self.environment_variables.clone()),
        }
    }

    fn build_params_for_hpc(&self, _server: &FlexServInstance) -> HPCParameterSet {
        build_hpc_from_options(&self.options, &self.environment_variables)
    }
}

impl TrtLlmParameterSetBuilder {
    pub fn new(command: Option<Vec<String>>) -> Self {
        Self {
            command,
            options: BTreeMap::new(),
            environment_variables: HashMap::new(),
        }
    }

    pub fn max_batch_size(mut self, size: u32) -> Self {
        set_builder_option(&mut self.options, "max-batch-size", size);
        self
    }

    pub fn max_input_len(mut self, len: u32) -> Self {
        set_builder_option(&mut self.options, "max-input-len", len);
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
            Backend::Transformers { command: vec![] },
        );
        let pod_params = TransformersParameterSetBuilder::new(None)
            .insert_env_var("ENV_VAR", "test")
            .build_params_for_pod(&server);
        assert_eq!(
            pod_params
                .environment_variables
                .as_ref()
                .unwrap()
                .get("ENV_VAR"),
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
            Backend::Transformers { command: vec![] },
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
            Backend::Transformers { command: vec![] },
        );
        let pod_params = TransformersParameterSetBuilder::new(None).build_params_for_pod(&server);
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
            Backend::Transformers { command: vec![] },
        );
        let pod_params = TransformersParameterSetBuilder::new(None).build_params_for_pod(&server);
        let args = pod_params.arguments.as_ref().unwrap();
        assert!(args.contains(&"--host".to_string()));
        assert!(args.contains(&"--port".to_string()));
        assert!(args.contains(&"8000".to_string()));
    }

    #[test]
    fn test_command_accessor() {
        let backend = Backend::Transformers {
            command: vec!["python".to_string(), "serve.py".to_string()],
        };
        assert_eq!(backend.command(), &["python", "serve.py"]);
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
            Backend::Transformers { command: vec![] },
        );
        let pod_params = TransformersParameterSetBuilder::new(None).build_params_for_pod(&server);
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
        let hpc_params = builder.build_params_for_hpc(&server);

        assert!(pod_params.command.is_some());
        assert!(pod_params.arguments.is_some());
        assert!(hpc_params.app_args.is_some());
        assert!(hpc_params.env_variables.is_some());
    }

    #[test]
    fn test_transformers_hpc_defaults_include_backend_args_only() {
        let server = FlexServInstance::new(
            "https://public.tapis.io".to_string(),
            "user".to_string(),
            "Qwen/Qwen3.5-0.8B".to_string(),
            None,
            Some("hf_test_token".to_string()),
            Some("sentence-transformers/all-MiniLM-L6-v2".to_string()),
            Backend::Transformers { command: vec![] },
        );

        let hpc_params = TransformersParameterSetBuilder::new(None).build_params_for_hpc(&server);
        let app_args = hpc_params.app_args.as_ref().unwrap();
        let env_vars = hpc_params.env_variables.as_ref().unwrap();

        assert!(app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--dtype bfloat16")));
        assert!(app_args.iter().any(|arg| arg.arg.as_deref()
            == Some("--default-embedding-model sentence-transformers/all-MiniLM-L6-v2")));
        assert!(!app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--model-name Qwen/Qwen3.5-0.8B")));
        assert!(!app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--enable-https")));
        assert!(!env_vars.iter().any(|env| {
            env.key.as_deref() == Some("FLEXSERV_BACKEND_TYPE")
                || env.key.as_deref() == Some("HUGGINGFACE_TOKEN")
        }));
    }

    #[test]
    fn test_transformers_hpc_builder_overrides_defaults() {
        let server = FlexServInstance::new(
            "https://public.tapis.io".to_string(),
            "user".to_string(),
            "Qwen/Qwen3.5-0.8B".to_string(),
            None,
            None,
            None,
            Backend::Transformers { command: vec![] },
        );

        let hpc_params = TransformersParameterSetBuilder::new(None)
            .dtype("float16")
            .build_params_for_hpc(&server);
        let app_args = hpc_params.app_args.as_ref().unwrap();

        assert!(app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--dtype float16")));
        assert!(!app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--dtype bfloat16")));
    }

    #[test]
    fn test_vllm_hpc_builder_maps_args_and_env_vars() {
        let server = FlexServInstance::new(
            "https://public.tapis.io".to_string(),
            "user".to_string(),
            "Qwen/Qwen3.5-0.8B".to_string(),
            None,
            None,
            None,
            Backend::VLlm { command: vec![] },
        );

        let hpc_params = VLlmParameterSetBuilder::new(None)
            .tensor_parallel_size(4)
            .insert_env_var("CUDA_VISIBLE_DEVICES", "0,1,2,3")
            .build_params_for_hpc(&server);
        let app_args = hpc_params.app_args.as_ref().unwrap();
        let env_vars = hpc_params.env_variables.as_ref().unwrap();

        assert!(app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--tensor-parallel-size 4")));
        assert!(env_vars.iter().any(|env| {
            env.key.as_deref() == Some("CUDA_VISIBLE_DEVICES")
                && env.value.as_deref() == Some("0,1,2,3")
        }));
    }
}
