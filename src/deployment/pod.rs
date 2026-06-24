use std::collections::HashMap;

use super::{DeploymentError, DeploymentResult, FlexServDeployment};
use crate::backend::Backend;
use crate::server::{FlexServInstance, ModelConfig, TapisConfig, ValidationError};

use tapis_pods::apis;
use tapis_pods::client::TapisPods;
use tapis_pods::models;

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

        let options = PodDeploymentOptions {
            deployment_id,
            ..Default::default()
        };

        Ok(Self::with_options(server, tapis_token, options))
    }

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

    /// Implement ID selection here: explicit deployment_id first, otherwise a
    /// deterministic hash from tenant/user/model/backend.
    /// Returns Pod id, Volume id.
    fn ids_from_options(
        server: &FlexServInstance,
        options: &PodDeploymentOptions,
    ) -> (String, String) {
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

    /// Build the high-level Tapis Pods client from tenant URL + token.
    fn pods_client(&self) -> Result<TapisPods, DeploymentError> {
        let base = self.server.tenant_url.trim_end_matches('/');
        let api_base = format!("{}/v3", base);

        let client = TapisPods::new(&api_base, Some(&self.tapis_token))
            .map_err(|e| DeploymentError::TapisAuthFailed(e.to_string()))?;

        Ok(client)
    }

    /// Convert a model id into the directory name expected on the mounted volume.
    fn model_dir_name(&self) -> String {
        self.server.default_model.clone()
    }

    /// Build the token clients will use to call the FlexServ pod.
    /// Example: `mysecret-` + `openai-community_gpt2`
    fn flexserv_token(&self, model_dir_name: &str) -> String {
        let secret = self
            .options
            .flexserv_secret
            .clone()
            .unwrap_or_else(|| std::env::var("FLEXSERV_SECRET").unwrap_or_default());
        format!("{}{}", secret, model_dir_name)
    }

    /// Build the Tapis `NewVolume` request body.
    fn build_volume_request(&self) -> Result<models::NewVolume, DeploymentError> {
        let mut new_volume = models::NewVolume::new(self.volume_id.clone());
        new_volume.description = Some(format!(
            "Volume for {}@{}",
            self.server.tapis_user, self.server.default_model
        ));
        new_volume.size_limit = Some(self.options.volume_size_mb.unwrap_or(10 * 1024)); // default to 10 GB

        Ok(new_volume)
    }

    /// Build the full Tapis `NewPod` request body in one place.
    fn build_pod_request(
        &self,
        model_dir_name: &str,
        flexserv_token: &str,
    ) -> Result<models::NewPod, DeploymentError> {
        const MODEL_REPO_PATH: &str = "/app/models";
        const PRIVATE_MODEL_REPO_PATH: &str = "/app/models/private";
        const PUBLIC_MODEL_REPO_PATH: &str = "/app/models/public";
        const FLEXSERV_PORT: &str = "8000";
        const BACKEND_PORT: &str = "8001";
        const TRANSFORMERS_BACKEND_SERVER: &str = "/app/flexserv/backend/backend_server.py";

        // Create New Pod
        let mut new_pod = models::NewPod::new(self.pod_id.clone());
        new_pod.description = Some(format!(
            "FlexServ pod for {}@{}",
            self.server.tapis_user, self.server.default_model
        ));

        // Set new pods image
        new_pod.image = self
            .options
            .image
            .clone()
            .or_else(|| Some("zhangwei217245/flexserv-transformers:1.4.6".to_string()));

        // Create new Mount value
        let mut mount =
            models::VolumeMountsValue::new(models::volume_mounts_value::Type::Tapisvolume);
        mount.source_id = Some(Some(self.volume_id.clone()));
        mount.sub_path = Some(String::new());
        mount.read_only = Some(Some(false));

        // Mount the volume at /app/models/private
        let mut volume_mounts = HashMap::new();
        volume_mounts.insert(MODEL_REPO_PATH.to_string(), mount);

        // Add volume mounts to new pod
        new_pod.volume_mounts = Some(volume_mounts);

        // Defining params
        let pod_params = self
            .server
            .backend
            .parameter_set_builder()
            .build_params_for_pod(&self.server);

        // Setting ENV variables
        let flexserv_secret = self
            .options
            .flexserv_secret
            .clone()
            .unwrap_or_else(|| std::env::var("FLEXSERV_SECRET").unwrap_or_default());
        let hf_token = self
            .server
            .hf_token
            .clone()
            .or_else(|| std::env::var("HF_TOKEN").ok());

        let mut env_vars = pod_params.environment_variables.unwrap_or_default();
        env_vars.extend([
            ("MODEL_REPO".to_string(), serde_json::json!(MODEL_REPO_PATH)),
            (
                "PRI_MODEL_REPO".to_string(),
                serde_json::json!(PRIVATE_MODEL_REPO_PATH),
            ),
            (
                "PUB_MODEL_REPO".to_string(),
                serde_json::json!(PUBLIC_MODEL_REPO_PATH),
            ),
            (
                "FLEXSERV_PORT".to_string(),
                serde_json::json!(FLEXSERV_PORT),
            ),
            (
                "GATEWAY_BACKEND_PORT".to_string(),
                serde_json::json!(BACKEND_PORT),
            ),
            ("MODEL_NAME".to_string(), serde_json::json!(model_dir_name)),
            (
                "FLEXSERV_SECRET".to_string(),
                serde_json::json!(flexserv_secret),
            ),
            (
                "FLEXSERV_TOKEN".to_string(),
                serde_json::json!(flexserv_token),
            ),
        ]);

        if let Some(ref token) = hf_token {
            env_vars.insert("HF_TOKEN".to_string(), serde_json::json!(token));
        }

        let model_path = format!("FLEX:PRI:{}", model_dir_name);

        //  Main cmd to run
        let (command, arguments) = match &self.server.backend {
            Backend::Transformers { .. } => (
                Some(vec!["/app/flexserv/bin/flexserv-gateway".to_string()]),
                vec![
                    "--manage-backend".to_string(),
                    "--backend-kind".to_string(),
                    "transformers".to_string(),
                    "--backend-default-model".to_string(),
                    model_path,
                    "--port".to_string(),
                    FLEXSERV_PORT.to_string(),
                    "--backend-port".to_string(),
                    BACKEND_PORT.to_string(),
                    "--backend-host".to_string(),
                    "0.0.0.0".to_string(),
                    "--backend-server".to_string(),
                    TRANSFORMERS_BACKEND_SERVER.to_string(),
                    "--flexserv-token".to_string(),
                    flexserv_token.to_string(),
                    "--backend-device".to_string(),
                    "auto".to_string(),
                    "--backend-dtype".to_string(),
                    "bfloat16".to_string(),
                    "--backend-model-timeout".to_string(),
                    "86400".to_string(),
                    "--backend-quantization".to_string(),
                    "none".to_string(),
                    "--backend-attn-implementation".to_string(),
                    "sdpa".to_string(),
                ],
            ),
            _ => (
                pod_params.command.clone(),
                pod_params.arguments.unwrap_or_default(),
            ),
        };

        // Networking
        let mut networking = HashMap::new();
        let mut net = models::Networking::new();
        net.protocol = Some("http".to_string());
        net.port = Some(8000);
        networking.insert("default".to_string(), net);

        // Resources
        let mut resources = models::ModelsPodsResources::new();
        resources.cpu_request = Some(self.options.cpu_request.unwrap_or(1000));
        resources.cpu_limit = Some(self.options.cpu_limit.unwrap_or(2000));
        resources.mem_request = Some(self.options.mem_request_mb.unwrap_or(4096));
        resources.mem_limit = Some(self.options.mem_limit_mb.unwrap_or(8192));
        resources.gpus = Some(self.options.gpus.unwrap_or(0));

        // Adding all components to pod
        new_pod.command = command.map(Some);
        new_pod.arguments = Some(Some(arguments));
        new_pod.environment_variables = Some(env_vars);
        new_pod.status_requested = Some("ON".to_string());
        new_pod.time_to_stop_default = Some(-1);
        new_pod.time_to_stop_instance = Some(Some(-1));
        new_pod.networking = Some(networking);
        new_pod.resources = Some(Box::new(resources));

        Ok(new_pod)
    }

    /// Convert a successful pod response model into this crate's public result type.
    fn pod_result_from_model(&self, pod: &models::PodResponseModel) -> DeploymentResult {
        self.pod_result_from_model_with_info(
            pod,
            format!("{:#?}", pod),
            self.volume_info.clone().unwrap_or_default(),
        )
    }

    /// Convert a successful pod response model plus explicit info strings into this crate's public result type.
    fn pod_result_from_model_with_info(
        &self,
        pod: &models::PodResponseModel,
        pod_info: String,
        volume_info: String,
    ) -> DeploymentResult {
        DeploymentResult::PodResult {
            pod_id: self.pod_id.clone(),
            volume_id: self.volume_id.clone(),
            pod_url: Self::pod_url_from_result(pod),
            status: pod.status.clone(),
            pod_info,
            volume_info,
            tapis_user: self.server.tapis_user.clone(),
            tapis_tenant: self.server.tenant_url.clone(),
            model_id: self.server.default_model.clone(),
        }
    }

    /// Extract the public URL from the Tapis pod response.
    fn pod_url_from_result(pod: &models::PodResponseModel) -> Option<String> {
        pod.networking
            .as_ref()
            .and_then(|networking| networking.get("default"))
            .and_then(|default_net| default_net.url.clone())
    }

    /// Detect the stale-resource case so create can delete and retry.
    fn is_already_exists_error<E: std::fmt::Debug>(err: &apis::Error<E>) -> bool {
        matches!(
            err,
            apis::Error::ResponseError(resp)
                if resp.content.contains("already exists")
                    || resp.content.contains("UniqueViolation")
        )
    }

    /// Convert Tapis SDK errors into this crate's deployment errors.
    fn map_pods_error<E: std::fmt::Debug>(err: apis::Error<E>) -> DeploymentError {
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
        let client = self.pods_client()?;

        // Remove stale resources with these deterministic IDs before creating new ones.
        // The delete calls are intentionally best-effort: "not found" is fine here.
        let _ = client.pods.delete_pod(&self.pod_id).await;
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        let _ = client.volumes.delete_volume(&self.volume_id).await;
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        let new_volume = self.build_volume_request()?;

        match client.volumes.create_volume(new_volume.clone()).await {
            Ok(resp) => {
                self.volume_info = Some(format!("{:#?}", resp.result));
            }
            Err(e) if Self::is_already_exists_error(&e) => {
                let _ = client.volumes.delete_volume(&self.volume_id).await;
                tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
                let resp = client
                    .volumes
                    .create_volume(new_volume)
                    .await
                    .map_err(Self::map_pods_error)?;
                self.volume_info = Some(format!("{:#?}", resp.result));
            }
            Err(e) => return Err(Self::map_pods_error(e)),
        }

        let model_dir_name = self.model_dir_name();
        let flexserv_token = self.flexserv_token(&model_dir_name);
        let new_pod = self.build_pod_request(&model_dir_name, &flexserv_token)?;

        if let Ok(body) = serde_json::to_string_pretty(&new_pod) {
            log::info!("TapisPods create_pod request body:\n{}", body);
        }

        let pod_resp = match client.pods.create_pod(new_pod).await {
            Ok(resp) => resp,
            Err(e) => {
                let _ = client.volumes.delete_volume(&self.volume_id).await;
                return Err(Self::map_pods_error(e));
            }
        };

        self.pod_info = Some(format!("{:#?}", pod_resp.result));
        Ok(self.pod_result_from_model(&pod_resp.result))
    }

    async fn start(&self) -> Result<DeploymentResult, DeploymentError> {
        let pods_client = self.pods_client()?;

        let start_response = match pods_client.pods.start_pod(&self.pod_id).await {
            Ok(resp) => resp,
            Err(e) => return Err(Self::map_pods_error(e)),
        };

        Ok(self.pod_result_from_model(&start_response.result))
    }

    async fn stop(&self) -> Result<DeploymentResult, DeploymentError> {
        let pods_client = self.pods_client()?;

        let stop_response = match pods_client.pods.stop_pod(&self.pod_id).await {
            Ok(resp) => resp,
            Err(e) => return Err(Self::map_pods_error(e)),
        };

        Ok(self.pod_result_from_model(&stop_response.result))
    }

    async fn terminate(&self) -> Result<DeploymentResult, DeploymentError> {
        // Implement:
        // - call `pods_client`
        // - delete pod first
        // - wait briefly
        // - delete volume second
        // - return a `DeploymentResult::PodResult` with no pod_url/status

        let pods_client = self.pods_client()?;

        let _ = match pods_client.pods.delete_pod(&self.pod_id).await {
            Ok(_resp) => {}
            Err(e) => {
                log::error!(
                    "Error deleting pod {}: {:?}",
                    self.pod_id,
                    Self::map_pods_error(e)
                );

                // Continue to attempt volume deletion regardless of pod deletion result
            }
        };

        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        if !self.volume_id.is_empty() {
            if let Err(e) = pods_client.volumes.delete_volume(&self.volume_id).await {
                log::error!(
                    "Error deleting volume {}: {:?}",
                    self.volume_id,
                    Self::map_pods_error(e)
                );
            }
        }

        let pod_result = DeploymentResult::PodResult {
            pod_id: self.pod_id.clone(),
            volume_id: self.volume_id.clone(),
            pod_url: None,
            status: Some("TERMINATED".to_string()),
            pod_info: self.pod_info.clone().unwrap_or_default(),
            volume_info: self.volume_info.clone().unwrap_or_default(),
            tapis_user: self.server.tapis_user.clone(),
            tapis_tenant: self.server.tenant_url.clone(),
            model_id: self.server.default_model.clone(),
        };

        Ok(pod_result)
    }

    async fn monitor(&self) -> Result<DeploymentResult, DeploymentError> {
        let pods_client = self.pods_client()?;

        let pods_result = match pods_client.pods.get_pod(&self.pod_id, None, None).await {
            Ok(resp) => resp,
            Err(e) => return Err(Self::map_pods_error(e)),
        };

        let volume_info = if self.volume_id.is_empty() {
            String::new()
        } else {
            match pods_client.volumes.get_volume(&self.volume_id).await {
                Ok(resp) => format!("{:#?}", resp.result),
                Err(_) => String::new(),
            }
        };

        Ok(self.pod_result_from_model_with_info(
            &pods_result.result,
            format!("{:#?}", pods_result.result),
            volume_info,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::server::{FlexServInstance, ModelConfig, TapisConfig};

    fn is_lowercase_alphanumeric(s: &str) -> bool {
        s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
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
            Backend::Transformers { command: vec![] },
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
            Backend::Transformers { command: vec![] },
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
        assert_eq!(
            deployment.options.image.as_deref(),
            Some("myregistry/flexserv:2.0")
        );
        assert_eq!(deployment.options.cpu_request, Some(2000));
        assert_eq!(
            deployment.options.flexserv_secret.as_deref(),
            Some("mysecret")
        );
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
            Backend::Transformers { command: vec![] },
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
            Backend::Transformers { command: vec![] },
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
            Backend::Transformers { command: vec![] },
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
            Backend::Transformers { command: vec![] },
        );
        let deployment = FlexServPodDeployment::new(server, "dummy-token".to_string());
        assert!(
            deployment.pod_id.starts_with('p'),
            "pod_id should start with p"
        );
        assert!(
            deployment.volume_id.starts_with('v'),
            "volume_id should start with v"
        );
        assert!(
            is_lowercase_alphanumeric(&deployment.pod_id),
            "pod_id must be lowercase alphanumeric"
        );
        assert!(
            is_lowercase_alphanumeric(&deployment.volume_id),
            "volume_id must be lowercase alphanumeric"
        );
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
            Backend::Transformers { command: vec![] },
        );
        let d1 = FlexServPodDeployment::new(server, "token".to_string());
        let server2 = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "user1".to_string(),
            "model-a".to_string(),
            None,
            None,
            None,
            Backend::Transformers { command: vec![] },
        );
        let d2 = FlexServPodDeployment::new(server2, "token".to_string());
        assert_eq!(d1.pod_id, d2.pod_id);
        assert_eq!(d1.volume_id, d2.volume_id);
    }

    #[test]
    fn test_pod_id_volume_id_from_deployment_id() {
        let make_server = || {
            FlexServInstance::new(
                "https://tacc.tapis.io".to_string(),
                "user1".to_string(),
                "openai-community/gpt2".to_string(),
                None,
                None,
                None,
                Backend::Transformers { command: vec![] },
            )
        };
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
        assert!(is_lowercase_alphanumeric(
            d1.pod_id.strip_prefix('p').unwrap()
        ));
        assert!(is_lowercase_alphanumeric(
            d1.volume_id.strip_prefix('v').unwrap()
        ));
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
            Backend::Transformers { command: vec![] },
        );
        let d = FlexServPodDeployment::new(server, "token".to_string());
        assert!(d.volume_info.is_none());
        assert!(d.pod_info.is_none());
    }
}
