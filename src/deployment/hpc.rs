use super::{DeploymentError, DeploymentResult, FlexServDeployment};
use crate::server::FlexServInstance;
use tokio::time::{sleep, Duration};
use tapis_sdk::jobs::apis;
use tapis_sdk::jobs::apis::configuration;
use tapis_sdk::jobs::apis::jobs_api;
use tapis_sdk::jobs::models;

#[derive(Debug, Clone)]
pub struct HpcDeploymentOptions {
    pub app_id: String,
    pub app_version: String,
    pub exec_system_id: String,
    pub exec_system_logical_queue: String,
    pub max_minutes: i32,
    pub allocation: String,
    // reservation optional
}

impl HpcDeploymentOptions {
    pub fn new(
        app_id: impl Into<String>,
        app_version: impl Into<String>,
        exec_system_id: impl Into<String>,
        exec_system_logical_queue: impl Into<String>,
        max_minutes: i32,
        allocation: impl Into<String>,
    ) -> Self {
        Self {
            app_id: app_id.into(),
            app_version: app_version.into(),
            exec_system_id: exec_system_id.into(),
            exec_system_logical_queue: exec_system_logical_queue.into(),
            max_minutes,
            allocation: allocation.into(),
        }
    }
}

/// HPC-based deployment using TAPIS Jobs.
pub struct FlexServHPCDeployment {
    pub server: Option<FlexServInstance>,
    pub tenant_url: Option<String>,
    pub tapis_token: String,
    pub options: Option<HpcDeploymentOptions>,
    pub job_uuid: Option<String>,
}

impl FlexServHPCDeployment {
    pub fn new(
        server: FlexServInstance,
        tapis_token: String,
        options: HpcDeploymentOptions,
    ) -> Self {
        Self {
            tenant_url: Some(server.tenant_url.clone()),
            server: Some(server),
            tapis_token,
            options: Some(options),
            job_uuid: None,
        }
    }

    /// Build deployment handle from an existing job UUID.
    pub fn from_existing(tapis_token: String, job_uuid: String) -> Self {
        Self {
            server: None,
            tenant_url: None,
            tapis_token,
            options: None,
            job_uuid: Some(job_uuid),
        }
    }

    fn jobs_config(&self) -> Result<configuration::Configuration, DeploymentError> {
        let mut config = configuration::Configuration::default();
        if let Some(server) = self.server.as_ref() {
            let base = server.tenant_url.trim_end_matches('/');
            config.base_path = format!("{}/v3", base);
        } else if let Some(tenant_url) = self.tenant_url.as_ref() {
            let base = tenant_url.trim_end_matches('/');
            config.base_path = format!("{}/v3", base);
        } else {
            return Err(DeploymentError::InvalidConfiguration(
                "missing tenant URL; pass server in new() or set tenant_url for from_existing()"
                    .to_string(),
            ));
        }
        config.api_key = Some(configuration::ApiKey {
            prefix: None,
            key: self.tapis_token.clone(),
        });
        Ok(config)
    }

    fn require_job_uuid(&self) -> Result<&str, DeploymentError> {
        self.job_uuid.as_deref().ok_or_else(|| {
            DeploymentError::JobCreationFailed(
                "job_uuid is not set; call create() first".to_string(),
            )
        })
    }

    fn map_jobs_error<E: std::fmt::Debug>(err: apis::Error<E>) -> DeploymentError {
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

    /// `Job.status` from submit/resubmit responses (`Option<Status>`), as TAPIS API strings.
    fn job_status_from_record(job: &models::Job) -> Option<String> {
        job.status.map(|s| match s {
            models::job::Status::Pending => "PENDING",
            models::job::Status::ProcessingInputs => "PROCESSING_INPUTS",
            models::job::Status::StagingInputs => "STAGING_INPUTS",
            models::job::Status::StagingJob => "STAGING_JOB",
            models::job::Status::SubmittingJob => "SUBMITTING_JOB",
            models::job::Status::Queued => "QUEUED",
            models::job::Status::Running => "RUNNING",
            models::job::Status::Archiving => "ARCHIVING",
            models::job::Status::Blocked => "BLOCKED",
            models::job::Status::Paused => "PAUSED",
            models::job::Status::Finished => "FINISHED",
            models::job::Status::Cancelled => "CANCELLED",
            models::job::Status::Failed => "FAILED",
        }
        .to_string())
    }

    fn build_submit_request(&self) -> Result<models::ReqSubmitJob, DeploymentError> {
        let server = self.server.as_ref().ok_or_else(|| {
            DeploymentError::JobCreationFailed(
                "missing server context; create() requires full server metadata".to_string(),
            )
        })?;
        let options = self.options.as_ref().ok_or_else(|| {
            DeploymentError::JobCreationFailed(
                "missing HPC deployment options; pass HpcDeploymentOptions from the call site"
                    .to_string(),
            )
        })?;

        let deployment_hash = server.deployment_hash().to_lowercase();
        let mut req = models::ReqSubmitJob::new(
            format!("flexserv-{}", deployment_hash),
            options.app_id.clone(),
            options.app_version.clone(),
        );
        req.exec_system_id = Some(options.exec_system_id.clone());
        req.exec_system_logical_queue = Some(options.exec_system_logical_queue.clone());
        req.max_minutes = Some(options.max_minutes);

        let mut parameter_set = self
            .server
            .as_ref()
            .expect("server checked above")
            .backend
            .parameter_set_builder()
            .build_params_for_hpc(server);

        // FlexServ app defines scheduler option "TACC Resource Allocation" with placeholder.
        // Enforce explicit allocation so submissions don't accidentally run with a placeholder.
        let alloc = options.allocation.trim();
        if alloc.is_empty() {
            return Err(DeploymentError::JobCreationFailed(
                "HPC allocation is required and cannot be empty".to_string(),
            ));
        }
        let mut sched = parameter_set.scheduler_options.unwrap_or_default();
        sched.push(models::JobArgSpec {
            name: Some("TACC Resource Allocation".to_string()),
            description: None,
            include: Some(true),
            arg: Some(format!("-A {}", alloc)),
            notes: None,
        });
        parameter_set.scheduler_options = Some(sched);

        req.parameter_set = Some(Box::new(parameter_set));
        Ok(req)
    }

    /// Current job status from `GET /jobs/{uuid}/status` (e.g. `PENDING`, `RUNNING`, `FINISHED`).
    pub async fn job_status(&self) -> Result<String, DeploymentError> {
        let config = self.jobs_config()?;
        let job_uuid = self.require_job_uuid()?;
        let resp = jobs_api::get_job_status(&config, job_uuid)
            .await
            .map_err(Self::map_jobs_error)?;
        Ok(resp
            .result
            .as_ref()
            .and_then(|r| r.status.clone())
            .unwrap_or_else(|| "unknown".to_string()))
    }

    fn parse_access_information(log_text: &str) -> Option<(String, String)> {
        for line in log_text.lines() {
            if !line.contains("FlexServ address:") || !line.contains("TAP token:") {
                continue;
            }
            let rest = line.split_once("FlexServ address:")?.1.trim();
            let (addr_part, token_part) = rest.split_once("TAP token:")?;
            let hpc_url = addr_part.trim().to_string();
            let flexserv_token = token_part.trim().to_string();
            if !hpc_url.is_empty() && !flexserv_token.is_empty() {
                return Some((hpc_url, flexserv_token));
            }
        }
        None
    }

    async fn fetch_running_access_from_logs(
        &self,
        config: &configuration::Configuration,
        job: &models::Job,
    ) -> Option<(String, String)> {
        let exec_system_id = job.exec_system_id.as_deref()?;
        let exec_output_dir = job.exec_system_output_dir.as_deref()?;
        let log_path = format!("{}/tapisjob.out", exec_output_dir.trim_end_matches('/'));
        let normalized_path = log_path.trim_start_matches('/');
        let base = config.base_path.trim_end_matches('/');
        let endpoint = format!("{}/files/content/{}/{}", base, exec_system_id, normalized_path);
        let server_ready_marker = "Server ready to accept requests";

        for page in 1..=5 {
            let mut req_builder = config.client.request(reqwest::Method::GET, endpoint.as_str());
            req_builder = req_builder.header("more", page.to_string());
            if let Some(ref api_key) = config.api_key {
                let token = match api_key.prefix {
                    Some(ref prefix) => format!("{} {}", prefix, api_key.key),
                    None => api_key.key.clone(),
                };
                req_builder = req_builder.header("X-Tapis-Token", token);
            }

            let request = req_builder.build().ok()?;
            let response = config.client.execute(request).await.ok()?;
            if !response.status().is_success() {
                return None;
            }
            let log_text = response.text().await.ok()?;

            if let Some(access) = Self::parse_access_information(&log_text) {
                return Some(access);
            }
            if log_text.contains(server_ready_marker) {
                return None;
            }
            sleep(Duration::from_secs(5)).await;
        }

        None
    }

    async fn running_connection_fields(
        &self,
        config: &configuration::Configuration,
        status: Option<&str>,
        job_uuid: &str,
        job: Option<&models::Job>,
    ) -> (Option<String>, Option<String>) {
        if status != Some("RUNNING") {
            return (None, None);
        }

        let mut full_job = job.cloned();
        let needs_full_job = full_job
            .as_ref()
            .map(|j| j.exec_system_id.is_none() || j.exec_system_output_dir.is_none())
            .unwrap_or(true);

        if needs_full_job {
            if let Ok(resp) = jobs_api::get_job(config, job_uuid).await {
                full_job = resp.result.map(|j| *j);
            }
        }

        if let Some(job_for_logs) = full_job.as_ref() {
            if let Some((hpc_url, flexserv_token)) =
                self.fetch_running_access_from_logs(config, job_for_logs).await
            {
                return (Some(hpc_url), Some(flexserv_token));
            }
        }

        (None, None)
    }
}

impl FlexServDeployment for FlexServHPCDeployment {
    async fn create(&mut self) -> Result<DeploymentResult, DeploymentError> {
        let config = self.jobs_config()?;
        let req = self.build_submit_request()?;
        let resp = jobs_api::submit_job(&config, req)
            .await
            .map_err(Self::map_jobs_error)?;
        let job = resp.result.map(|j| *j);
        let job_uuid = job
            .as_ref()
            .and_then(|j| j.uuid.clone())
            .unwrap_or_default();
        self.job_uuid = Some(job_uuid.clone());
        let status = job.as_ref().and_then(|j| Self::job_status_from_record(j));
        let (hpc_url, flexserv_token) = self
            .running_connection_fields(&config, status.as_deref(), &job_uuid, job.as_ref())
            .await;

        Ok(DeploymentResult::HPCResult {
            job_uuid: job_uuid.clone(),
            status,
            job,
            hpc_url,
            flexserv_token,
        })
    }

    async fn start(&self) -> Result<DeploymentResult, DeploymentError> {
        let config = self.jobs_config()?;
        let prior_uuid = self.require_job_uuid()?;
        let resp = jobs_api::resubmit_job(&config, prior_uuid, None)
            .await
            .map_err(Self::map_jobs_error)?;
        let job = resp.result.map(|j| *j);
        let job_uuid = job
            .as_ref()
            .and_then(|j| j.uuid.clone())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| prior_uuid.to_string());
        let status = job.as_ref().and_then(|j| Self::job_status_from_record(j));
        let (hpc_url, flexserv_token) = self
            .running_connection_fields(&config, status.as_deref(), &job_uuid, job.as_ref())
            .await;
        Ok(DeploymentResult::HPCResult {
            job_uuid: job_uuid.clone(),
            status,
            job,
            hpc_url,
            flexserv_token,
        })
    }

    async fn stop(&self) -> Result<DeploymentResult, DeploymentError> {
        let config = self.jobs_config()?;
        let job_uuid = self.require_job_uuid()?;
        let _resp = jobs_api::cancel_job(&config, job_uuid, None)
            .await
            .map_err(Self::map_jobs_error)?;
        Ok(DeploymentResult::HPCResult {
            job_uuid: job_uuid.to_string(),
            // `JobCancelDisplay` has no lifecycle status; avoid a follow-up GET that could fail silently.
            status: None,
            job: None,
            hpc_url: None,
            flexserv_token: None,
        })
    }

    async fn terminate(&self) -> Result<DeploymentResult, DeploymentError> {
        self.stop().await
    }

    async fn monitor(&self) -> Result<DeploymentResult, DeploymentError> {
        let config = self.jobs_config()?;
        let job_uuid = self.require_job_uuid()?;
        let status_resp = jobs_api::get_job_status(&config, job_uuid)
            .await
            .map_err(Self::map_jobs_error)?;
        let full_resp = jobs_api::get_job(&config, job_uuid)
            .await
            .map_err(Self::map_jobs_error)?;
        let status = status_resp.result.as_ref().and_then(|r| r.status.clone());
        let (hpc_url, flexserv_token) = self
            .running_connection_fields(
                &config,
                status.as_deref(),
                job_uuid,
                full_resp.result.as_deref(),
            )
            .await;
        Ok(DeploymentResult::HPCResult {
            job_uuid: job_uuid.to_string(),
            status,
            job: full_resp.result.map(|j| *j),
            hpc_url,
            flexserv_token,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;

    #[test]
    fn test_hpc_deployment_creation() {
        let server = FlexServInstance::new(
            "https://tacc.tapis.io".to_string(),
            "testuser".to_string(),
            "meta-llama/Llama-2-7b".to_string(),
            None,
            None,
            None,
            Backend::VLlm { command: vec![] },
        );

        let deployment = FlexServHPCDeployment::new(
            server,
            "test-token".to_string(),
            HpcDeploymentOptions::new(
                "FlexServ-1.4.0",
                "1.4.0",
                "vista-tapis",
                "gh",
                60,
                "TACC-ACI-CIC",
            ),
        );
        assert_eq!(
            deployment.server.as_ref().map(|s| s.tapis_user.as_str()),
            Some("testuser")
        );
        assert_eq!(deployment.tapis_token, "test-token");
        assert_eq!(
            deployment.options.as_ref().unwrap().exec_system_id,
            "vista-tapis"
        );
    }

    #[test]
    fn test_build_submit_request_adds_generic_hpc_args_and_envs() {
        let server = FlexServInstance::new(
            "https://public.tapis.io".to_string(),
            "testuser".to_string(),
            "Qwen/Qwen3.5-0.8B".to_string(),
            None,
            Some("hf_test_token".to_string()),
            None,
            Backend::Transformers { command: vec![] },
        );

        let deployment = FlexServHPCDeployment::new(
            server,
            "test-token".to_string(),
            HpcDeploymentOptions::new(
                "FlexServ-1.4.0",
                "1.4.0",
                "vista-tapis",
                "gh",
                60,
                "TACC-ACI-CIC",
            ),
        );
        let req = deployment.build_submit_request().unwrap();
        let parameter_set = req.parameter_set.unwrap();
        let app_args = parameter_set.app_args.as_ref().unwrap();
        let env_vars = parameter_set.env_variables.as_ref().unwrap();

        assert!(app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--flexserv-port 8000")));
        assert!(app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--model-name Qwen/Qwen3.5-0.8B")));
        assert!(app_args
            .iter()
            .any(|arg| arg.arg.as_deref() == Some("--enable-https")));
        assert!(env_vars.iter().any(|env| {
            env.key.as_deref() == Some("FLEXSERV_BACKEND_TYPE")
                && env.value.as_deref() == Some("transformers")
        }));
        assert!(env_vars.iter().any(|env| {
            env.key.as_deref() == Some("HUGGINGFACE_TOKEN")
                && env.value.as_deref() == Some("hf_test_token")
        }));
    }
}
