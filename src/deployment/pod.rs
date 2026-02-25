use super::{DeploymentError, DeploymentResult, FlexServDeployment};
use crate::backend::{Backend, BuildBackendParameterSet};
use crate::server::{FlexServInstance, ModelConfig, TapisConfig, ValidationError};
use reqwest::header::{HeaderMap, HeaderValue};
use tapis_sdk::pods::apis;
use tapis_sdk::pods::apis::configuration;
use tapis_sdk::pods::apis::pods_api;
use tapis_sdk::pods::apis::volumes_api;
use tapis_sdk::pods::models;

/// Options for pod-based deployment (volume size, image, resources, secrets, deployment id).
/// Omitted fields use defaults; secrets fall back to env (`FLEXSERV_SECRET`, `HF_TOKEN`) when `None`.
#[derive(Clone, Debug, Default)]
pub struct PodDeploymentOptions {
    /// Optional deployment id (e.g. UUID from MLHub). When set, pod_id and volume_id are derived from this
    pub deployment_id: Option<String>,
    /// Volume size in MB. Default 10240 (10 GB).
    pub volume_size_mb: Option<i32>,
    /// Container image. Default "tapis/flexserv:1.0".
    pub image: Option<String>,
    /// CPU request in millicpus (1000 = 1 CPU). Default 1000.
    pub cpu_request: Option<i32>,
    /// CPU limit in millicpus. Default 2000.
    pub cpu_limit: Option<i32>,
    /// Memory request in MB. Default 4096.
    pub mem_request_mb: Option<i32>,
    /// Memory limit in MB. Default 8192.
    pub mem_limit_mb: Option<i32>,
    /// Number of GPUs. Default 0.
    pub gpus: Option<i32>,
    /// Optional secret prepended to pod auth token. If None, uses `FLEXSERV_SECRET` env.
    pub flexserv_secret: Option<String>,
}

/// Pod-based deployment
#[derive(Debug)]
pub struct FlexServPodDeployment {
    pub server: FlexServInstance,
    /// Tapis JWT token used to authenticate against the Pods API
    pub tapis_token: String,
    /// Options for volume, image, resources, and secrets (used by create())
    pub options: PodDeploymentOptions,
    /// Derived volume id (e.g. v{deployment_hash})
    pub volume_id: String,
    /// Derived pod id (e.g. p{deployment_hash})
    pub pod_id: String,
    pub volume_info: Option<String>,
    pub pod_info: Option<String>,
}

impl FlexServPodDeployment {
    /// Create a new pod deployment with default options (volume 10 GB, image tapis/flexserv:1.0, 2 CPU / 8 GB RAM).
    /// Secrets fall back to env: FLEXSERV_SECRET, HF_TOKEN.
    pub fn new(server: FlexServInstance, tapis_token: String) -> Self {
        Self::with_options(server, tapis_token, PodDeploymentOptions::default())
    }

    /// Create a new pod deployment with explicit options (volume size, image, CPU/memory, secrets, deployment_id).
    /// When `options.deployment_id` is set (e.g. UUID from MLHub), pod_id and volume_id are derived from it
    /// so multiple pods for the same model can coexist. Otherwise they are derived from server config (one pod per user+model).
    pub fn with_options(
        server: FlexServInstance,
        tapis_token: String,
        options: PodDeploymentOptions,
    ) -> Self {
        let (pod_id, volume_id) = Self::ids_from_options(&server, &options);
        Self {
            server,
            tapis_token,
            options,
            volume_id,
            pod_id,
            volume_info: None,
            pod_info: None,
        }
    }

    /// Derive pod_id and volume_id from options.deployment_id (if set) or from server deployment_hash.
    /// deployment_id is normalized to lowercase alphanumeric (e.g. UUID with dashes stripped).
    fn ids_from_options(server: &FlexServInstance, options: &PodDeploymentOptions) -> (String, String) {
        let suffix = if let Some(ref id) = options.deployment_id {
            let normalized = crate::utils::normalize_to_lowercase_alphanumeric(id);
            if normalized.is_empty() {
                server.deployment_hash().to_lowercase()
            } else {
                normalized
            }
        } else {
            server.deployment_hash().to_lowercase()
        };
        (format!("p{}", suffix), format!("v{}", suffix))
    }

    /// Create a deployment from [TapisConfig], [ModelConfig], backend, and options (no validation).
    pub fn from_configs(
        tapis: TapisConfig,
        model: ModelConfig,
        backend: Backend,
        options: PodDeploymentOptions,
    ) -> Self {
        let server = FlexServInstance::from_configs(&tapis, &model, backend);
        Self::with_options(server, tapis.tapis_token, options)
    }

    /// Convenience: create a deployment with individual params (validates inputs).
    /// Use [FlexServPodDeployment::from_configs] for config structs, or [FlexServPodDeployment::with_options] for full control.
    pub fn create_deployment(
        tenant_url: String,
        tapis_user: String,
        tapis_token: String,
        model_id: String,
        deployment_id: Option<String>,
        backend: Backend,
    ) -> Result<Self, ValidationError> {
        let server = FlexServInstance::builder()
            .tenant_url(tenant_url)
            .tapis_user(tapis_user)
            .model(model_id)
            .backend(backend)
            .build()?;
        let mut options = PodDeploymentOptions::default();
        options.deployment_id = deployment_id;
        Ok(Self::with_options(server, tapis_token, options))
    }

    /// Create a deployment from existing pod_id and volume_id (for start/stop/terminate/monitor).
    pub fn from_existing(
        server: FlexServInstance,
        tapis_token: String,
        pod_id: String,
        volume_id: String,
    ) -> Self {
        Self {
            server,
            tapis_token,
            options: PodDeploymentOptions::default(),
            volume_id,
            pod_id,
            volume_info: None,
            pod_info: None,
        }
    }

    /// Build Pods API configuration (base URL + reqwest client with X-Tapis-Token).
    /// Base must be the v3 API root (e.g. https://tacc.tapis.io/v3).
    fn pods_config(&self) -> Result<configuration::Configuration, DeploymentError> {
        let base = self.server.tenant_url.trim_end_matches('/');
        let api_base = format!("{}/v3", base);
        let mut headers = HeaderMap::new();
        headers.insert(
            "X-Tapis-Token",
            HeaderValue::from_str(&self.tapis_token).map_err(|e| DeploymentError::TapisAuthFailed(e.to_string()))?,
        );
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| DeploymentError::TapisAuthFailed(e.to_string()))?;
        let mut config = configuration::Configuration::default();
        config.base_path = api_base;
        config.client = reqwest_middleware::ClientBuilder::new(client).build();
        Ok(config)
    }

    /// Extract pod URL from API response (networking.default.url).
    fn _pod_url_from_result(result: &tapis_sdk::pods::models::PodResponseModel) -> Option<String> {
        result
            .networking
            .as_ref()
            .and_then(|n| n.get("default"))
            .and_then(|net| net.url.clone())
    }

    /// Map a tapis-pods error into our DeploymentError, based on HTTP status / network.
    fn _map_pods_error<E: std::fmt::Debug>(err: apis::Error<E>) -> DeploymentError {
        match err {
            apis::Error::Reqwest(e) => {
                if e.is_timeout() {
                    DeploymentError::TapisTimeout(e.to_string())
                } else if e.is_connect() {
                    DeploymentError::TapisAPIUnreachable(e.to_string())
                } else {
                    DeploymentError::UnknownError(e.to_string())
                }
            }
            apis::Error::ReqwestMiddleware(e) => DeploymentError::UnknownError(e.to_string()),
            apis::Error::Serde(e) => DeploymentError::UnknownError(e.to_string()),
            apis::Error::Io(e) => DeploymentError::UnknownError(e.to_string()),
            apis::Error::ResponseError(resp) => {
                let code = resp.status.as_u16();
                if code == 401 || code == 403 {
                    DeploymentError::TapisAuthFailed(resp.content)
                } else if code == 400 {
                    DeploymentError::TapisBadRequest(resp.content)
                } else if (500..600).contains(&code) {
                    DeploymentError::TapisInternalServerError(resp.content)
                } else {
                    DeploymentError::UnknownError(resp.content)
                }
            }
        }
    }

}

impl FlexServDeployment for FlexServPodDeployment {
    async fn create(&mut self) -> Result<DeploymentResult, DeploymentError> {
        let config = self.pods_config()?;

        // Clean up any existing pod/volume with these ids.
        // Ignore errors (404 means they don't exist, which is fine).
        // Delete pod first, then volume (volume deletion may fail if pod still exists).
        let _ = pods_api::delete_pod(&config, &self.pod_id).await;
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        let _ = volumes_api::delete_volume(&config, &self.volume_id).await;

        // Wait for deletions to complete (volumes can take a moment)
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        // --- Create volume ---
        let volume_size_mb = self.options.volume_size_mb.unwrap_or(10 * 1024);
        let volume_desc = format!(
            "Volume for {}@{}",
            self.server.tapis_user, self.server.default_model
        );
        let new_volume = models::NewVolume {
            volume_id: self.volume_id.clone(),
            description: Some(volume_desc),
            size_limit: Some(volume_size_mb),
        };

        // Try to create volume. If it already exists, delete and retry once.
        let volume_result = volumes_api::create_volume(&config, new_volume.clone()).await;

        match volume_result {
            Ok(_) => {}
            Err(e) => {
                // If volume already exists, try deleting and recreating
                if let apis::Error::ResponseError(ref resp) = e {
                    if resp.content.contains("already exists") || resp.content.contains("UniqueViolation") {
                        log::warn!("Volume {} already exists, deleting and retrying...", self.volume_id);
                        let _ = volumes_api::delete_volume(&config, &self.volume_id).await;
                        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
                        volumes_api::create_volume(&config, new_volume)
                            .await
                            .map_err(Self::_map_pods_error)?;
                    } else {
                        return Err(Self::_map_pods_error(e));
                    }
                } else {
                    return Err(Self::_map_pods_error(e));
                }
            }
        }

        // Model dir in volume: single directory name (e.g. openai-community/gpt2 -> openai-community_gpt2).
        let model_dir_name = self.server.default_model.replace('/', "_");
        // --- Create pod ---
        let image = self
            .options
            .image
            .clone()
            .unwrap_or_else(|| "tapis/flexserv:1.0".to_string());

        // volume_mounts: key = mount path, value = VolumeMountsValue (type, source_id, sub_path).
        const MODEL_REPO_PATH: &str = "/app/models";
        let mut volume_mounts = std::collections::HashMap::new();
        let mut mount = models::VolumeMountsValue::new(models::volume_mounts_value::Type::Tapisvolume);
        mount.source_id = Some(Some(self.volume_id.clone()));
        mount.sub_path = Some(String::new());
        volume_mounts.insert(MODEL_REPO_PATH.to_string(), mount);
        // FlexServ token: secret + MODEL_NAME (from options or FLEXSERV_SECRET env).
        let flexserv_secret = self
            .options
            .flexserv_secret
            .clone()
            .unwrap_or_else(|| std::env::var("FLEXSERV_SECRET").unwrap_or_default());
        let flexserv_token = format!("{}{}", flexserv_secret, model_dir_name);

        let hf_token = self
            .server
            .hf_token
            .clone()
            .or_else(|| std::env::var("HF_TOKEN").ok());

        // Build backend parameter set for pod (trait), then copy directly into NewPod.
        let params = self.server.backend.build_params_for_pod(&self.server);
        let cli_args = params.to_cli_args();
        let model_path = format!("{}/{}", MODEL_REPO_PATH, model_dir_name);
        let arguments: Vec<String> = [vec![model_path], cli_args, vec!["--flexserv-token".to_string(), flexserv_token.clone()]]
            .into_iter()
            .flatten()
            .collect();

        let mut env_vars: std::collections::HashMap<String, serde_json::Value> = [
            ("MODEL_REPO", serde_json::json!(MODEL_REPO_PATH)),
            ("FLEXSERV_PORT", serde_json::json!("8000")),
            ("MODEL_NAME", serde_json::json!(model_dir_name)),
            ("FLEXSERV_SECRET", serde_json::json!(flexserv_secret)),
            ("FLEXSERV_TOKEN", serde_json::json!(flexserv_token)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        for (k, v) in &params.env {
            env_vars.insert(k.clone(), serde_json::json!(v));
        }
        if let Some(ref t) = hf_token {
            env_vars.insert("HF_TOKEN".to_string(), serde_json::json!(t));
        }

        let mut net = models::ModelsPodsNetworking::new();
        net.protocol = Some("http".to_string());
        net.port = Some(8000);
        let mut networking = std::collections::HashMap::new();
        networking.insert("default".to_string(), net);

        let mut resources = models::ModelsPodsResources::new();
        resources.cpu_request = Some(self.options.cpu_request.unwrap_or(1000));
        resources.cpu_limit = Some(self.options.cpu_limit.unwrap_or(2000));
        resources.mem_request = Some(self.options.mem_request_mb.unwrap_or(4096));
        resources.mem_limit = Some(self.options.mem_limit_mb.unwrap_or(8192));
        resources.gpus = Some(self.options.gpus.unwrap_or(0));

        // NewPod run: copy directly from BackendParameterSet (command_prefix, args, env).
        let mut new_pod = models::NewPod::new(self.pod_id.clone());
        new_pod.image = Some(image);
        new_pod.description = Some(format!(
            "FlexServ pod for {}@{}",
            self.server.tapis_user, self.server.default_model
        ));
        new_pod.command = Some(Some(params.command_prefix.clone()));
        new_pod.arguments = Some(Some(arguments));
        new_pod.environment_variables = Some(env_vars);
        new_pod.status_requested = Some("ON".to_string());
        new_pod.volume_mounts = Some(volume_mounts);
        new_pod.time_to_stop_default = Some(-1);
        new_pod.time_to_stop_instance = Some(Some(-1));
        new_pod.networking = Some(networking);
        new_pod.resources = Some(Box::new(resources));

        // Log the exact Pods create_pod request body for debugging.
        if let Ok(body) = serde_json::to_string_pretty(&new_pod) {
            log::info!("Pods create_pod request body:\n{}", body);
        }

        // Create pod. If this fails, clean up the volume we just created.
        let pod_resp = match pods_api::create_pod(&config, new_pod).await {
            Ok(resp) => resp,
            Err(e) => {
                log::error!("Pod creation failed, cleaning up volume {}...", self.volume_id);
                let _ = volumes_api::delete_volume(&config, &self.volume_id).await;
                return Err(Self::_map_pods_error(e));
            }
        };

        // Store minimal info for monitoring later
        self.pod_info = Some(format!("{:#?}", pod_resp.result));
        self.volume_info = Some(self.volume_id.clone());

        let pod_url = Self::_pod_url_from_result(&pod_resp.result);

        Ok(DeploymentResult::PodResult {
            pod_id: self.pod_id.clone(),
            volume_id: self.volume_id.clone(),
            pod_url,
            pod_info: self.pod_info.clone().unwrap_or_default(),
            volume_info: self.volume_info.clone().unwrap_or_default(),
            tapis_user: self.server.tapis_user.clone(),
            tapis_tenant: self.server.tenant_url.clone(),
            model_id: self.server.default_model.clone(),
        })
    }

    async fn start(&self) -> Result<DeploymentResult, DeploymentError> {
        let config = self.pods_config()?;
        let pod_resp = pods_api::start_pod(&config, &self.pod_id)
            .await
            .map_err(Self::_map_pods_error)?;

        let pod_url = Self::_pod_url_from_result(&pod_resp.result);

        Ok(DeploymentResult::PodResult {
            pod_id: self.pod_id.clone(),
            volume_id: self.volume_id.clone(),
            pod_url,
            pod_info: format!("{:#?}", pod_resp.result),
            volume_info: self.volume_id.clone(),
            tapis_user: self.server.tapis_user.clone(),
            tapis_tenant: self.server.tenant_url.clone(),
            model_id: self.server.default_model.clone(),
        })
    }

    async fn stop(&self) -> Result<DeploymentResult, DeploymentError> {
        let config = self.pods_config()?;
        let pod_resp = pods_api::stop_pod(&config, &self.pod_id)
            .await
            .map_err(Self::_map_pods_error)?;

        let pod_url = Self::_pod_url_from_result(&pod_resp.result);

        Ok(DeploymentResult::PodResult {
            pod_id: self.pod_id.clone(),
            volume_id: self.volume_id.clone(),
            pod_url,
            pod_info: format!("{:#?}", pod_resp.result),
            volume_info: self.volume_id.clone(),
            tapis_user: self.server.tapis_user.clone(),
            tapis_tenant: self.server.tenant_url.clone(),
            model_id: self.server.default_model.clone(),
        })
    }

    async fn terminate(&self) -> Result<DeploymentResult, DeploymentError> {
        let config = self.pods_config()?;

        // Delete pod and volume together. Try both even if one fails.
        // Delete pod first (volume deletion may fail if pod still exists).
        let mut pod_resp = None;
        let mut pod_error = None;
        match pods_api::delete_pod(&config, &self.pod_id).await {
            Ok(resp) => pod_resp = Some(resp),
            Err(e) => pod_error = Some(Self::_map_pods_error(e)),
        }

        let mut vol_resp = None;
        let mut vol_error = None;
        if !self.volume_id.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            match volumes_api::delete_volume(&config, &self.volume_id).await {
                Ok(resp) => vol_resp = Some(resp),
                Err(e) => vol_error = Some(Self::_map_pods_error(e)),
            }
        }

        // If both failed, return the pod error (more critical)
        if pod_error.is_some() && vol_error.is_some() {
            return Err(pod_error.unwrap());
        }
        // If only one failed, log it but continue (partial cleanup is better than none)
        if let Some(ref e) = pod_error {
            log::warn!("Pod deletion failed (but volume deleted): {:?}", e);
        }
        if let Some(ref e) = vol_error {
            log::warn!("Volume deletion failed (but pod deleted): {:?}", e);
        }

        let vol_info = if self.volume_id.is_empty() {
            "no volume".to_string()
        } else {
            vol_resp.as_ref().map(|r| format!("{:#?}", r)).unwrap_or_else(|| "deleted".to_string())
        };
        let combined_info = format!(
            "pod: {:#?}\nvolume: {}",
            pod_resp.as_ref().map(|r| format!("{:#?}", r)).unwrap_or_else(|| "deleted".to_string()),
            vol_info
        );

        Ok(DeploymentResult::PodResult {
            pod_id: self.pod_id.clone(),
            volume_id: self.volume_id.clone(),
            pod_url: None, // pod deleted
            pod_info: combined_info,
            volume_info: String::new(),
            tapis_user: self.server.tapis_user.clone(),
            tapis_tenant: self.server.tenant_url.clone(),
            model_id: self.server.default_model.clone(),
        })
    }

    async fn monitor(&self) -> Result<DeploymentResult, DeploymentError> {
        let config = self.pods_config()?;

        let pod_resp = pods_api::get_pod(&config, &self.pod_id, None, None)
            .await
            .map_err(Self::_map_pods_error)?;

        eprintln!("pods_api::get_pod result:\n{:#?}", pod_resp);

        let volume_info = if self.volume_id.is_empty() {
            String::new()
        } else {
            match volumes_api::get_volume(&config, &self.volume_id).await {
                Ok(vol_resp) => format!("{:#?}", vol_resp.result),
                Err(_) => String::new(),
            }
        };

        let pod_info = format!("{:#?}", pod_resp.result);
        let pod_url = Self::_pod_url_from_result(&pod_resp.result);

        Ok(DeploymentResult::PodResult {
            pod_id: self.pod_id.clone(),
            volume_id: self.volume_id.clone(),
            pod_url,
            pod_info,
            volume_info,
            tapis_user: self.server.tapis_user.clone(),
            tapis_tenant: self.server.tenant_url.clone(),
            model_id: self.server.default_model.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::server::{FlexServInstance, ModelConfig, TapisConfig};

    fn is_lowercase_alphanumeric(s: &str) -> bool {
        s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
    }

    #[test]
    fn test_pod_deployment_creation() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "testuser".to_string(),
            "meta-llama/Llama-2-7b".to_string(),
            None,
            None,
            None,
            Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            },
        );

        let deployment = FlexServPodDeployment::new(server, "dummy-token".to_string());
        assert_eq!(deployment.server.tapis_user, "testuser");
    }

    #[test]
    fn test_pod_deployment_with_options() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "myuser".to_string(),
            "openai-community/gpt2".to_string(),
            None,
            None,
            None,
            Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            },
        );
        let options = PodDeploymentOptions {
            volume_size_mb: Some(20 * 1024),
            image: Some("myregistry/flexserv:2.0".to_string()),
            cpu_request: Some(2000),
            flexserv_secret: Some("mysecret".to_string()),
            ..Default::default()
        };
        let deployment = FlexServPodDeployment::with_options(server, "token".to_string(), options);
        assert_eq!(deployment.server.tapis_user, "myuser");
        assert_eq!(deployment.options.volume_size_mb, Some(20 * 1024));
        assert_eq!(deployment.options.image.as_deref(), Some("myregistry/flexserv:2.0"));
        assert_eq!(deployment.options.cpu_request, Some(2000));
        assert_eq!(deployment.options.flexserv_secret.as_deref(), Some("mysecret"));
    }

    #[test]
    fn test_from_configs() {
        let tapis = TapisConfig {
            tenant_url: "https://tacc.tapis.io".to_string(),
            tapis_user: "u".to_string(),
            tapis_token: "jwt".to_string(),
        };
        let model = ModelConfig {
            model_id: "openai-community/gpt2".to_string(),
            model_revision: None,
            hf_token: None,
            default_embedding_model: None,
        };
        let deployment = FlexServPodDeployment::from_configs(
            tapis,
            model,
            Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            },
            PodDeploymentOptions::default(),
        );
        assert_eq!(deployment.server.tapis_user, "u");
        assert_eq!(deployment.server.default_model, "openai-community/gpt2");
        assert_eq!(deployment.tapis_token, "jwt");
    }

    #[test]
    fn test_create_deployment_ok() {
        let deployment = FlexServPodDeployment::create_deployment(
            "https://tacc.tapis.io".to_string(),
            "myuser".to_string(),
            "token".to_string(),
            "openai-community/gpt2".to_string(),
            None,
            Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            },
        )
        .unwrap();
        assert_eq!(deployment.server.tapis_user, "myuser");
        assert_eq!(deployment.server.default_model, "openai-community/gpt2");
    }

    #[test]
    fn test_create_deployment_validation_fails() {
        let err = FlexServPodDeployment::create_deployment(
            "not-a-url".to_string(),
            "u".to_string(),
            "token".to_string(),
            "gpt2".to_string(),
            None,
            Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            },
        )
        .unwrap_err();
        assert!(matches!(err, ValidationError::InvalidTenantUrl(_)));
    }

    #[test]
    fn test_pod_id_volume_id_format() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "testuser".to_string(),
            "no-model-yet".to_string(),
            None,
            None,
            None,
            Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            },
        );
        let deployment = FlexServPodDeployment::new(server, "dummy-token".to_string());
        assert!(deployment.pod_id.starts_with('p'), "pod_id should start with p");
        assert!(deployment.volume_id.starts_with('v'), "volume_id should start with v");
        assert!(is_lowercase_alphanumeric(&deployment.pod_id), "pod_id must be lowercase alphanumeric");
        assert!(is_lowercase_alphanumeric(&deployment.volume_id), "volume_id must be lowercase alphanumeric");
    }

    #[test]
    fn test_pod_id_volume_id_stable() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "user1".to_string(),
            "model-a".to_string(),
            None,
            None,
            None,
            Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            },
        );
        let d1 = FlexServPodDeployment::new(server, "token".to_string());
        let server2 = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "user1".to_string(),
            "model-a".to_string(),
            None,
            None,
            None,
            Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            },
        );
        let d2 = FlexServPodDeployment::new(server2, "token".to_string());
        assert_eq!(d1.pod_id, d2.pod_id);
        assert_eq!(d1.volume_id, d2.volume_id);
    }

    #[test]
    fn test_pod_id_volume_id_from_deployment_id() {
        let make_server = || FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "user1".to_string(),
            "openai-community/gpt2".to_string(),
            None,
            None,
            None,
            Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            },
        );
        let uuid1 = "550e8400-e29b-41d4-a716-446655440000";
        let uuid2 = "6ba7b810-9dad-11d1-80b4-00c04fd430c8";
        let opts1 = PodDeploymentOptions {
            deployment_id: Some(uuid1.to_string()),
            ..Default::default()
        };
        let opts2 = PodDeploymentOptions {
            deployment_id: Some(uuid2.to_string()),
            ..Default::default()
        };
        let d1 = FlexServPodDeployment::with_options(make_server(), "token".to_string(), opts1);
        let d2 = FlexServPodDeployment::with_options(make_server(), "token".to_string(), opts2);
        assert_eq!(d1.pod_id, "p550e8400e29b41d4a716446655440000");
        assert_eq!(d1.volume_id, "v550e8400e29b41d4a716446655440000");
        assert_eq!(d2.pod_id, "p6ba7b8109dad11d180b400c04fd430c8");
        assert_eq!(d2.volume_id, "v6ba7b8109dad11d180b400c04fd430c8");
        assert_ne!(d1.pod_id, d2.pod_id);
        assert!(is_lowercase_alphanumeric(d1.pod_id.strip_prefix('p').unwrap()));
        assert!(is_lowercase_alphanumeric(d1.volume_id.strip_prefix('v').unwrap()));
    }

    #[test]
    fn test_pod_deployment_new_optional_fields_none() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "u".to_string(),
            "m".to_string(),
            None,
            None,
            None,
            Backend::Transformers {
                command_prefix: vec!["python".to_string()],
            },
        );
        let d = FlexServPodDeployment::new(server, "token".to_string());
        assert!(d.volume_info.is_none());
        assert!(d.pod_info.is_none());
    }

}
