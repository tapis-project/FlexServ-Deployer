use super::{DeploymentError, DeploymentResult, FlexServDeployment};
use crate::server::FlexServInstance;
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

fn hpc_job_arg(name: impl Into<String>, arg: impl Into<String>) -> models::JobArgSpec {
    models::JobArgSpec {
        name: Some(name.into()),
        description: None,
        include: Some(true),
        arg: Some(arg.into()),
        notes: None,
    }
}

fn hpc_env_var(key: impl Into<String>, value: impl Into<String>) -> models::KeyValuePair {
    models::KeyValuePair {
        key: Some(key.into()),
        value: Some(value.into()),
        description: None,
        include: Some(true),
        notes: None,
    }
}

fn push_or_replace_job_arg(
    entries: &mut Vec<models::JobArgSpec>,
    name: impl Into<String>,
    arg: impl Into<String>,
) {
    let name = name.into();
    let arg = arg.into();
    if let Some(existing) = entries
        .iter_mut()
        .find(|entry| entry.name.as_deref() == Some(name.as_str()))
    {
        existing.arg = Some(arg);
        existing.include = Some(true);
    } else {
        entries.push(hpc_job_arg(name, arg));
    }
}

fn push_or_replace_env_var(
    entries: &mut Vec<models::KeyValuePair>,
    key: impl Into<String>,
    value: impl Into<String>,
) {
    let key = key.into();
    let value = value.into();
    if let Some(existing) = entries
        .iter_mut()
        .find(|entry| entry.key.as_deref() == Some(key.as_str()))
    {
        existing.value = Some(value);
        existing.include = Some(true);
    } else {
        entries.push(hpc_env_var(key, value));
    }
}

/// HPC-based deployment
pub struct FlexServHPCDeployment {
    pub server: FlexServInstance,
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
            server,
            tapis_token,
            options: Some(options),
            job_uuid: None,
        }
    }

    /// Build deployment handle from an existing job UUID.
    pub fn from_existing(server: FlexServInstance, tapis_token: String, job_uuid: String) -> Self {
        Self {
            server,
            tapis_token,
            options: None,
            job_uuid: Some(job_uuid),
        }
    }

    fn jobs_config(&self) -> Result<configuration::Configuration, DeploymentError> {
        let base = self.server.tenant_url.trim_end_matches('/');
        let api_base = format!("{}/v3", base);
        let mut config = configuration::Configuration::default();
        config.base_path = api_base;
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

    fn build_submit_request(&self) -> Result<models::ReqSubmitJob, DeploymentError> {
        let options = self.options.as_ref().ok_or_else(|| {
            DeploymentError::JobCreationFailed(
                "missing HPC deployment options; pass HpcDeploymentOptions from the call site"
                    .to_string(),
            )
        })?;

        let deployment_hash = self.server.deployment_hash().to_lowercase();
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
            .backend
            .parameter_set_builder()
            .build_params_for_hpc(&self.server);

        let mut app_args = parameter_set.app_args.unwrap_or_default();
        push_or_replace_job_arg(&mut app_args, "flexServPort", "--flexserv-port 8000");
        push_or_replace_job_arg(
            &mut app_args,
            "modelName",
            format!("--model-name {}", self.server.default_model),
        );
        push_or_replace_job_arg(&mut app_args, "enableHttps", "--enable-https");
        push_or_replace_job_arg(&mut app_args, "isDistributed", "--is-distributed 0");
        parameter_set.app_args = Some(app_args);

        let mut env_vars = parameter_set.env_variables.unwrap_or_default();
        push_or_replace_env_var(
            &mut env_vars,
            "FLEXSERV_BACKEND_TYPE",
            self.server.backend.as_str().to_string(),
        );
        if let Some(hf_token) = &self.server.hf_token {
            push_or_replace_env_var(&mut env_vars, "HUGGINGFACE_TOKEN", hf_token.clone());
        }
        parameter_set.env_variables = Some(env_vars);

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
}

impl FlexServDeployment for FlexServHPCDeployment {
    async fn create(&mut self) -> Result<DeploymentResult, DeploymentError> {
        let config = self.jobs_config()?;
        let req = self.build_submit_request()?;
        let resp = jobs_api::submit_job(&config, req)
            .await
            .map_err(Self::map_jobs_error)?;
        let job = resp.result;
        let job_uuid = job
            .as_ref()
            .and_then(|j| j.uuid.clone())
            .unwrap_or_default();
        self.job_uuid = Some(job_uuid.clone());

        Ok(DeploymentResult::HPCResult {
            job_info: format!("job_uuid={}; response={:#?}", job_uuid, job),
            tapis_user: self.server.tapis_user.clone(),
            tapis_tenant: self.server.tenant_url.clone(),
            model_id: self.server.default_model.clone(),
        })
    }

    async fn start(&self) -> Result<DeploymentResult, DeploymentError> {
        let config = self.jobs_config()?;
        let job_uuid = self.require_job_uuid()?;
        let resp = jobs_api::resubmit_job(&config, job_uuid, None)
            .await
            .map_err(Self::map_jobs_error)?;
        Ok(DeploymentResult::HPCResult {
            job_info: format!("resubmitted from {}; response={:#?}", job_uuid, resp.result),
            tapis_user: self.server.tapis_user.clone(),
            tapis_tenant: self.server.tenant_url.clone(),
            model_id: self.server.default_model.clone(),
        })
    }

    async fn stop(&self) -> Result<DeploymentResult, DeploymentError> {
        let config = self.jobs_config()?;
        let job_uuid = self.require_job_uuid()?;
        let resp = jobs_api::cancel_job(&config, job_uuid, None)
            .await
            .map_err(Self::map_jobs_error)?;
        Ok(DeploymentResult::HPCResult {
            job_info: format!("canceled {}; response={:#?}", job_uuid, resp.result),
            tapis_user: self.server.tapis_user.clone(),
            tapis_tenant: self.server.tenant_url.clone(),
            model_id: self.server.default_model.clone(),
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
        Ok(DeploymentResult::HPCResult {
            job_info: format!(
                "status={:#?}\njob={:#?}",
                status_resp.result, full_resp.result
            ),
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
        assert_eq!(deployment.server.tapis_user, "testuser");
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
