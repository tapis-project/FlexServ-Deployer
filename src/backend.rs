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

/// Every parameter key the FlexServ Tapis application understands in its
/// `parameterSet.appArgs`.  Keys outside this set are silently ignored by the
/// TAPIS job runner; keeping a single authoritative list here makes any drift
/// between the deployer and the app definition visible at runtime.
///
/// Update this list whenever the FlexServ app spec gains or drops a parameter.
pub const FLEXSERV_APP_HPC_ALLOWED_KEYS: &[&str] = &[
    // ── Transformers ──────────────────────────────────────────────────────────
    "attn-implementation",
    "continuous-batching",
    "default-embedding-model",
    "device",
    "dtype",
    "enable-cors",
    "force-default-embedding-model",
    "force-default-model",
    "log-level",
    "model-timeout",
    "non-blocking",
    "quantization",
    "trust-remote-code",
    // ── vLLM ─────────────────────────────────────────────────────────────────
    "gpu-memory-utilization",
    "max-model-len",
    "pipeline-parallel-size",
    "tensor-parallel-size",
    // ── SGLang ───────────────────────────────────────────────────────────────
    "mem-fraction-static",
    "tp-size",
    // ── TRT-LLM ──────────────────────────────────────────────────────────────
    "max-batch-size",
    "max-input-len",
];

/// Retain only keys present in [`FLEXSERV_APP_HPC_ALLOWED_KEYS`].
///
/// A `log::warn!` is emitted for every dropped key so that stale or
/// mis-spelled parameter names surface immediately at deployment time
/// rather than being silently swallowed by the TAPIS job runner.
fn filter_hpc_options(options: &BTreeMap<String, Value>) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    for (key, value) in options {
        if FLEXSERV_APP_HPC_ALLOWED_KEYS.contains(&key.as_str()) {
            out.insert(key.clone(), value.clone());
        } else {
            log::warn!(
                "build_params_for_hpc: dropping unknown key \"{key}\" \
                 (not listed in FLEXSERV_APP_HPC_ALLOWED_KEYS)"
            );
        }
    }
    out
}

fn push_or_replace_job_arg(
    entries: &mut Vec<JobArgSpec>,
    name: impl Into<String>,
    arg: impl Into<String>,
) {
    let name = name.into();
    let arg = arg.into();
    if let Some(existing) = entries
        .iter_mut()
        .find(|e| e.name.as_deref() == Some(name.as_str()))
    {
        existing.arg = Some(arg);
        existing.include = Some(true);
    } else {
        entries.push(hpc_job_arg(name, arg));
    }
}

fn push_or_replace_env_var(
    entries: &mut Vec<KeyValuePair>,
    key: impl Into<String>,
    value: impl Into<String>,
) {
    let key = key.into();
    let value = value.into();
    if let Some(existing) = entries
        .iter_mut()
        .find(|e| e.key.as_deref() == Some(key.as_str()))
    {
        existing.value = Some(value);
        existing.include = Some(true);
    } else {
        entries.push(hpc_env_var(key, value));
    }
}

/// Apply the FlexServ application contract to an HPC parameter set.
///
/// App args injected:
/// - `--flexserv-port 8000`
/// - `--model-name <server.default_model>`
/// - `--enable-https`
/// - `--is-distributed 0`
///
/// Env vars injected:
/// - `FLEXSERV_BACKEND_TYPE`
/// - `HUGGINGFACE_TOKEN` (only when `server.hf_token` is `Some`)
fn apply_flexserv_hpc_contract(params: &mut HPCParameterSet, server: &FlexServInstance) {
    let mut app_args = params.app_args.take().unwrap_or_default();
    push_or_replace_job_arg(&mut app_args, "flexServPort", "--flexserv-port 8000");
    push_or_replace_job_arg(
        &mut app_args,
        "modelName",
        format!("--model-name {}", server.default_model),
    );
    push_or_replace_job_arg(&mut app_args, "enableHttps", "--enable-https");
    push_or_replace_job_arg(&mut app_args, "isDistributed", "--is-distributed 0");
    params.app_args = Some(app_args);

    let mut env_vars = params.env_variables.take().unwrap_or_default();
    push_or_replace_env_var(
        &mut env_vars,
        "FLEXSERV_BACKEND_TYPE",
        server.backend.as_str().to_string(),
    );
    if let Some(hf_token) = &server.hf_token {
        push_or_replace_env_var(&mut env_vars, "HUGGINGFACE_TOKEN", hf_token.clone());
    }
    params.env_variables = Some(env_vars);
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
        // Caller-supplied options override defaults; unknown keys are dropped
        // with a warning so drift against the FlexServ app spec is visible.
        for (key, value) in filter_hpc_options(&self.options) {
            merged.insert(key, value);
        }
        let mut params = build_hpc_from_options(&merged, &self.environment_variables);
        apply_flexserv_hpc_contract(&mut params, server);
        params
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

    /// Insert an arbitrary option key/value pair.
    ///
    /// For pod deployments all keys pass through; for HPC deployments any key
    /// absent from [`FLEXSERV_APP_HPC_ALLOWED_KEYS`] will be dropped with a
    /// warning at [`build_params_for_hpc`] time.
    pub fn insert_option(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.options.insert(key.into(), value.into());
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

    fn build_params_for_hpc(&self, server: &FlexServInstance) -> HPCParameterSet {
        let mut params =
            build_hpc_from_options(&filter_hpc_options(&self.options), &self.environment_variables);
        apply_flexserv_hpc_contract(&mut params, server);
        params
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

    /// Insert an arbitrary option key/value pair.
    pub fn insert_option(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.options.insert(key.into(), value.into());
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

    fn build_params_for_hpc(&self, server: &FlexServInstance) -> HPCParameterSet {
        let mut params =
            build_hpc_from_options(&filter_hpc_options(&self.options), &self.environment_variables);
        apply_flexserv_hpc_contract(&mut params, server);
        params
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

    /// Insert an arbitrary option key/value pair.
    pub fn insert_option(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.options.insert(key.into(), value.into());
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

    fn build_params_for_hpc(&self, server: &FlexServInstance) -> HPCParameterSet {
        let mut params =
            build_hpc_from_options(&filter_hpc_options(&self.options), &self.environment_variables);
        apply_flexserv_hpc_contract(&mut params, server);
        params
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

    /// Insert an arbitrary option key/value pair.
    pub fn insert_option(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.options.insert(key.into(), value.into());
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
    fn test_transformers_hpc_params_include_backend_and_contract_args() {
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

        // Backend-specific defaults
        assert!(app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--dtype bfloat16")));
        assert!(app_args.iter().any(|arg| arg.arg.as_deref()
            == Some("--default-embedding-model sentence-transformers/all-MiniLM-L6-v2")));

        // FlexServ contract args now live here, not in build_submit_request
        assert!(app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--model-name Qwen/Qwen3.5-0.8B")));
        assert!(app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--enable-https")));
        assert!(app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--flexserv-port 8000")));
        assert!(app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--is-distributed 0")));
        assert!(env_vars.iter().any(|env| {
            env.key.as_deref() == Some("FLEXSERV_BACKEND_TYPE")
                && env.value.as_deref() == Some("transformers")
        }));
        assert!(env_vars.iter().any(|env| {
            env.key.as_deref() == Some("HUGGINGFACE_TOKEN")
                && env.value.as_deref() == Some("hf_test_token")
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

    // Helper: collect all arg strings from an HPCParameterSet.
    fn hpc_arg_strings(params: &HPCParameterSet) -> Vec<String> {
        params
            .app_args
            .as_ref()
            .map(|args| {
                args.iter()
                    .filter_map(|a| a.arg.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn test_unknown_keys_are_dropped_from_hpc_output_transformers() {
        let server = FlexServInstance::new(
            "https://public.tapis.io".to_string(),
            "user".to_string(),
            "gpt2".to_string(),
            None,
            None,
            None,
            Backend::Transformers { command: vec![] },
        );
        // "flexserv-token" is a pod-only key; it must not appear in HPC output.
        let hpc_params = TransformersParameterSetBuilder::new(None)
            .insert_option("flexserv-token", "secret")
            .build_params_for_hpc(&server);
        let args = hpc_arg_strings(&hpc_params);
        assert!(
            !args.iter().any(|a| a.contains("flexserv-token")),
            "pod-only key flexserv-token must not appear in HPC args: {args:?}"
        );
    }

    #[test]
    fn test_unknown_keys_are_dropped_from_hpc_output_vllm() {
        let server = FlexServInstance::new(
            "https://public.tapis.io".to_string(),
            "user".to_string(),
            "gpt2".to_string(),
            None,
            None,
            None,
            Backend::VLlm { command: vec![] },
        );
        // "unknown-flag" is not in FLEXSERV_APP_HPC_ALLOWED_KEYS.
        let hpc_params = VLlmParameterSetBuilder::new(None)
            .insert_option("unknown-flag", "val")
            .build_params_for_hpc(&server);
        let args = hpc_arg_strings(&hpc_params);
        assert!(
            !args.iter().any(|a| a.contains("unknown-flag")),
            "unknown key must be dropped from vLLM HPC args: {args:?}"
        );
    }

    #[test]
    fn test_unknown_keys_are_dropped_from_hpc_output_sglang() {
        let server = FlexServInstance::new(
            "https://public.tapis.io".to_string(),
            "user".to_string(),
            "gpt2".to_string(),
            None,
            None,
            None,
            Backend::SGLang { command: vec![] },
        );
        let hpc_params = SGLangParameterSetBuilder::new(None)
            .insert_option("not-a-real-param", 1)
            .build_params_for_hpc(&server);
        let args = hpc_arg_strings(&hpc_params);
        assert!(
            !args.iter().any(|a| a.contains("not-a-real-param")),
            "unknown key must be dropped from SGLang HPC args: {args:?}"
        );
    }

    #[test]
    fn test_unknown_keys_are_dropped_from_hpc_output_trtllm() {
        let server = FlexServInstance::new(
            "https://public.tapis.io".to_string(),
            "user".to_string(),
            "gpt2".to_string(),
            None,
            None,
            None,
            Backend::TrtLlm { command: vec![] },
        );
        let hpc_params = TrtLlmParameterSetBuilder::new(None)
            .insert_option("bogus-key", "x")
            .build_params_for_hpc(&server);
        let args = hpc_arg_strings(&hpc_params);
        assert!(
            !args.iter().any(|a| a.contains("bogus-key")),
            "unknown key must be dropped from TrtLLM HPC args: {args:?}"
        );
    }

    #[test]
    fn test_known_keys_pass_through_filter_for_each_backend() {
        let make_server = |backend: Backend| {
            FlexServInstance::new(
                "https://public.tapis.io".to_string(),
                "user".to_string(),
                "gpt2".to_string(),
                None,
                None,
                None,
                backend,
            )
        };

        // vLLM: tensor-parallel-size is in the allowed list
        let vllm_params = VLlmParameterSetBuilder::new(None)
            .tensor_parallel_size(8)
            .build_params_for_hpc(&make_server(Backend::VLlm { command: vec![] }));
        assert!(
            hpc_arg_strings(&vllm_params)
                .iter()
                .any(|a| a.contains("tensor-parallel-size")),
            "tensor-parallel-size must survive the filter"
        );

        // SGLang: tp-size is in the allowed list
        let sglang_params = SGLangParameterSetBuilder::new(None)
            .tp_size(4)
            .build_params_for_hpc(&make_server(Backend::SGLang { command: vec![] }));
        assert!(
            hpc_arg_strings(&sglang_params)
                .iter()
                .any(|a| a.contains("tp-size")),
            "tp-size must survive the filter"
        );

        // TRT-LLM: max-batch-size is in the allowed list
        let trtllm_params = TrtLlmParameterSetBuilder::new(None)
            .max_batch_size(32)
            .build_params_for_hpc(&make_server(Backend::TrtLlm { command: vec![] }));
        assert!(
            hpc_arg_strings(&trtllm_params)
                .iter()
                .any(|a| a.contains("max-batch-size")),
            "max-batch-size must survive the filter"
        );
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
