//! FlexServ Deployer CLI.
//!

use anyhow::{Context, Result, anyhow};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use clap::{Args, Parser, Subcommand, ValueEnum};
use flexserv_deployer::{
    Backend, DeploymentError, DeploymentResult, FlexServDeployment, FlexServHPCDeployment,
    FlexServInstance, FlexServPodDeployment, HpcDeploymentOptions, PodDeploymentOptions,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tapis_jobs::apis::configuration;
use tapis_jobs::apis::jobs_api;
use tapis_jobs::models::JobListDto;
use tapis_pods::client::TapisPods;
use tapis_pods::models::PodResponseModel;

type CliResult<T> = Result<T>;

// -----------------------------------------------------------------------------
// Top-level parser
// -----------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(
    name = "flexserv-deployer",
    version,
    about = "Create and manage FlexServ pod and HPC deployments",
    long_about = "Create and manage FlexServ deployments on Tapis Pods and Tapis Jobs."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ValueEnum)]
enum BackendKind {
    Transformers,
    Vllm,
    Sglang,
    Trtllm,
}

impl BackendKind {
    fn into_backend(self) -> Backend {
        match self {
            BackendKind::Transformers => Backend::Transformers { command: vec![] },
            BackendKind::Vllm => Backend::VLlm { command: vec![] },
            BackendKind::Sglang => Backend::SGLang { command: vec![] },
            BackendKind::Trtllm => Backend::TrtLlm { command: vec![] },
        }
    }
}

// -----------------------------------------------------------------------------
// Command tree
// -----------------------------------------------------------------------------

#[derive(Debug, Subcommand)]
enum Command {
    /// Save a Tapis JWT token for future commands.
    #[command(alias = "m")]
    Manage(ManageCommand),
    /// Manage Tapis Pod deployments.
    #[command(alias = "p", alias = "pods", alias = "ps")]
    Pod(PodCommand),
    /// Manage Tapis Jobs deployments.
    #[command(alias = "j", alias = "jobs", alias = "js")]
    Job(JobCommand),
}

#[derive(Debug, Args)]
struct ManageCommand {
    #[command(subcommand)]
    command: ManageSubcommand,
}

#[derive(Debug, Subcommand)]
enum ManageSubcommand {
    /// Save a Tapis JWT token for future commands.
    #[command(alias = "auth", alias = "login")]
    Authenticate(AuthenticateArgs),
    /// Save a HuggingFace token for future commands.
    #[command(alias = "hf")]
    HFToken(HFTokenArgs),
    /// Set the output format for command results.
    Output(OutputFormatArgs),
    /// Show saved and effective CLI configuration.
    #[command(alias = "config", alias = "whoami")]
    Show,
}

#[derive(Debug, Args)]
struct AuthenticateArgs {
    /// Tapis JWT token to save for later CLI calls.
    #[arg(value_name = "JWT_TOKEN", hide_env_values = true)]
    token: String,
}

#[derive(Debug, Args)]
struct HFTokenArgs {
    /// HuggingFace token to save for later CLI calls.
    #[arg(value_name = "HF_TOKEN", hide_env_values = true)]
    token: String,
}

#[derive(Debug, Args)]
struct OutputFormatArgs {
    /// Output format to use for command results.
    #[arg(
        value_name = "OUTPUT_FORMAT",
        default_value = "text",
        hide_env_values = true
    )]
    output: String,
}

#[derive(Debug, Args)]
struct PodCommand {
    #[command(subcommand)]
    command: PodSubcommand,
}

#[derive(Debug, Subcommand)]
enum PodSubcommand {
    /// Create a FlexServ pod and backing volume.
    Create(PodCreateArgs),
    /// List pods visible to the authenticated Tapis user.
    List(ListArgs),
    /// Show current pod state.
    Monitor(PodIdArgs),
    /// Request pod start.
    Start(PodIdArgs),
    /// Request pod stop.
    Stop(PodIdArgs),
    /// Delete the pod.
    #[command(alias = "delete", alias = "rm")]
    Terminate(PodIdArgs),
}

#[derive(Debug, Args, Default)]
struct PodCreateArgs {
    /// Hugging Face model id to serve.
    #[arg(long, default_value = "openai-community/gpt2")]
    model: Option<String>,
    /// FlexServ backend to run.
    #[arg(long, default_value = "transformers")]
    backend: Option<BackendKind>,
    /// Stable external id used to derive pod and volume ids.
    #[arg(long)]
    deployment_id: Option<String>,
    /// Backing volume size in MB.
    #[arg(long, default_value = "10240")]
    volume_size_mb: Option<i32>,
    /// Container image to run.
    #[arg(long, default_value = "zhangwei217245/flexserv-transformers:1.4.6")]
    image: Option<String>,
    /// CPU request in millicpus.
    #[arg(long, default_value = "1000")]
    cpu_request: Option<i32>,
    /// CPU limit in millicpus.
    #[arg(long, default_value = "2000")]
    cpu_limit: Option<i32>,
    /// Memory request in MB.
    #[arg(long, default_value = "4096")]
    mem_request_mb: Option<i32>,
    /// Memory limit in MB.
    #[arg(long, default_value = "8192")]
    mem_limit_mb: Option<i32>,
    /// Number of GPUs to request.
    #[arg(long, default_value = "0")]
    gpus: Option<i32>,
    /// FlexServ auth token.
    #[arg(
        long,
        hide_env_values = true,
        help = "FlexServ auth token (default: empty)"
    )]
    flexserv_secret: Option<String>,
}

#[derive(Debug, Args, Clone)]
struct PodIdArgs {
    /// Full pod id or a unique prefix from `pod list`.
    id: String,
    /// Full volume id. Defaults to the matching v... id for p... FlexServ pods.
    #[arg(long)]
    volume_id: Option<String>,
}

#[derive(Debug, Args)]
struct JobCommand {
    #[command(subcommand)]
    command: JobSubcommand,
}

#[derive(Debug, Subcommand)]
enum JobSubcommand {
    /// Submit a FlexServ HPC job.
    Create(JobCreateArgs),
    /// List jobs visible to the authenticated Tapis user.
    List(ListArgs),
    /// Show current job state.
    #[command(alias = "status", alias = "view")]
    Monitor(JobIdArgs),
    /// Resubmit a previous job.
    #[command(alias = "resubmit")]
    Start(JobIdArgs),
    /// Cancel a job.
    #[command(alias = "stop", alias = "terminate")]
    Cancel(JobIdArgs),
}

#[derive(Debug, Args)]
struct JobCreateArgs {
    /// Hugging Face model id to serve.
    #[arg(long, default_value = "openai-community/gpt2")]
    model: Option<String>,
    /// FlexServ backend to run.
    #[arg(long, default_value = "transformers")]
    backend: Option<BackendKind>,
    /// Tapis app id to submit.
    #[arg(value_name = "APP_ID")]
    app_id: String,
    /// Tapis app version to submit.
    #[arg(value_name = "APP_VERSION")]
    app_version: String,
    /// Execution system id.
    #[arg(value_name = "EXEC_SYSTEM_ID")]
    exec_system_id: String,
    /// Execution system logical queue.
    #[arg(value_name = "QUEUE")]
    exec_system_logical_queue: String,
    /// Max job runtime in minutes.
    #[arg(value_name = "MAX_MINUTES")]
    max_minutes: i32,
    /// HPC allocation to charge.
    #[arg(value_name = "ALLOCATION")]
    allocation: String,
}

#[derive(Debug, Args, Clone)]
struct JobIdArgs {
    /// Full job UUID or a unique prefix from `job list`.
    id: String,
}

#[derive(Debug, Args, Clone)]
struct ListArgs {
    #[arg(long, default_value_t = 25)]
    limit: i32,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct SavedConfig {
    tapis_token: Option<String>,
    hf_token: Option<String>,
    output: Option<OutputFormat>,
}

// -----------------------------------------------------------------------------
// Entrypoint and top-level error handling
// -----------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    env_logger::init();

    if let Err(err) = run_cli().await {
        print_cli_error(&err);
        std::process::exit(1);
    }
}

async fn run_cli() -> CliResult<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Manage(command) => manage(command),
        Command::Pod(command) => run_pod(command).await,
        Command::Job(command) => run_job(command).await,
    }
}

// -----------------------------------------------------------------------------
// Command dispatchers
// -----------------------------------------------------------------------------

fn manage(command: ManageCommand) -> CliResult<()> {
    match command.command {
        ManageSubcommand::Authenticate(_args) => {
            // Read the existing SavedConfig, or start with SavedConfig::default().
            // Put `_args.token` into `config.tapis_token`.
            // Write the updated config to disk.
            // Print a short success message and the next useful command.
            let mut saved_config = read_saved_config()?.unwrap_or_default();
            saved_config.tapis_token = Some(_args.token);

            write_saved_config(&saved_config)?;

            print_next_steps()
        }
        ManageSubcommand::HFToken(_args) => {
            // Read the existing SavedConfig, or start with SavedConfig::default().
            // Store `_args.token` in `config.hf_token`.
            // Write the updated config back to disk.
            let mut saved_config = read_saved_config()?.unwrap_or_default();
            saved_config.hf_token = Some(_args.token);

            write_saved_config(&saved_config)
        }
        ManageSubcommand::Output(_args) => {
            // Parse `_args.output` with `OutputFormat::from_str`.
            // Store the parsed value in `config.output`.
            // Persist the config.
            let mut saved_config = read_saved_config()?.unwrap_or_default();
            saved_config.output = Some(
                OutputFormat::from_str(&_args.output, true)
                    .map_err(|e| anyhow!("Invalid output format: {e}"))?,
            );

            write_saved_config(&saved_config)
        }
        ManageSubcommand::Show => print_config(),
    }
}

async fn run_pod(command: PodCommand) -> CliResult<()> {
    match command.command {
        PodSubcommand::Create(args) => {
            // Build `FlexServInstance` from create args and saved config.
            // Convert `PodCreateArgs` into `PodDeploymentOptions`.
            // Build `FlexServPodDeployment::with_options`.
            // Call `create().await`.
            // Print the result.

            let server = flexserv_server(args.model, args.backend)?;

            let deployment_options = PodDeploymentOptions {
                deployment_id: args.deployment_id,
                volume_size_mb: args.volume_size_mb,
                image: args.image,
                cpu_request: args.cpu_request,
                cpu_limit: args.cpu_limit,
                mem_request_mb: args.mem_request_mb,
                mem_limit_mb: args.mem_limit_mb,
                gpus: args.gpus,
                flexserv_secret: args.flexserv_secret,
            };

            let mut deployment =
                FlexServPodDeployment::with_options(server, tapis_token()?, deployment_options);

            let pod_result = deployment.create().await?;

            print_deployment_result(effective_output()?, &pod_result)
        }
        PodSubcommand::List(args) => {
            // Call `list_pods(args.limit).await`.
            list_pods(args.limit).await
        }
        PodSubcommand::Monitor(args) => {
            // Build an existing pod deployment handle.
            // Call `monitor().await`.
            // Print the result.

            let deployment = existing_pod_deployment(args).await?;
            let pod_result = deployment.monitor().await?;

            print_deployment_result(effective_output()?, &pod_result)
        }
        PodSubcommand::Start(args) => {
            // Resolve/build existing pod deployment, then call `start()`.
            let deployment = existing_pod_deployment(args).await?;
            let pod_result = deployment.start().await?;

            print_deployment_result(effective_output()?, &pod_result)
        }
        PodSubcommand::Stop(args) => {
            // Resolve/build existing pod deployment, then call `stop()`.
            let deployment = existing_pod_deployment(args).await?;
            let pod_result = deployment.stop().await?;

            print_deployment_result(effective_output()?, &pod_result)
        }
        PodSubcommand::Terminate(args) => {
            // Resolve/build existing pod deployment, then call `terminate()`.
            let deployment = existing_pod_deployment(args).await?;
            let pod_result = deployment.terminate().await?;

            print_deployment_result(effective_output()?, &pod_result)
        }
    }
}

async fn run_job(command: JobCommand) -> CliResult<()> {
    match command.command {
        JobSubcommand::Create(args) => {
            // Build `FlexServInstance` from create args and saved config.
            // Convert `JobCreateArgs` into `HpcDeploymentOptions`.
            // Build `FlexServHPCDeployment::new`.
            // Call `create().await`.
            // Print the result.

            let deployment_options = HpcDeploymentOptions {
                app_id: args.app_id,
                app_version: args.app_version,
                exec_system_id: args.exec_system_id,
                exec_system_logical_queue: args.exec_system_logical_queue,
                max_minutes: args.max_minutes,
                allocation: args.allocation,
            };

            let mut deployment = FlexServHPCDeployment::new(
                flexserv_server(args.model, args.backend)?,
                tapis_token()?,
                deployment_options,
            );

            let result = deployment.create().await?;

            print_deployment_result(effective_output()?, &result)
        }
        JobSubcommand::List(args) => {
            // Call `list_jobs(args.limit).await`.
            list_jobs(args.limit).await
        }
        JobSubcommand::Monitor(args) => {
            // Resolve/build existing HPC deployment, then call `monitor()`.
            let deployment = existing_hpc_deployment(args).await?;
            let hpc_result = deployment.monitor().await?;

            print_deployment_result(effective_output()?, &hpc_result)
        }
        JobSubcommand::Start(args) => {
            // Resolve/build existing HPC deployment, then call `start()`.
            let deployment = existing_hpc_deployment(args).await?;
            let hpc_result = deployment.start().await?;

            print_deployment_result(effective_output()?, &hpc_result)
        }
        JobSubcommand::Cancel(args) => {
            // Resolve/build existing HPC deployment, then call `stop()`.
            let deployment = existing_hpc_deployment(args).await?;
            let hpc_result = deployment.stop().await?;

            print_deployment_result(effective_output()?, &hpc_result)
        }
    }
}
// -----------------------------------------------------------------------------
// Deployment builders
// -----------------------------------------------------------------------------

fn flexserv_server(
    model: Option<String>,
    backend: Option<BackendKind>,
) -> CliResult<FlexServInstance> {
    // Convert CLI settings and saved config into the library server context.
    // `FlexServInstance::new(...)` wants tenant URL, user, model id,
    // optional model revision, optional HF token, optional embedding model, and
    // the backend.

    Ok(FlexServInstance::new(
        effective_tenant_url()?,
        effective_user()?,
        effective_model(model)?,
        None,
        effective_hf_token()?,
        None,
        effective_backend(backend)?.into_backend(),
    ))
}

async fn existing_pod_deployment(args: PodIdArgs) -> CliResult<FlexServPodDeployment> {
    // Resolve the pod id with `resolve_pod_id`.
    // Derive or accept the volume id.
    // Return `FlexServPodDeployment::from_existing(...)`.
    let pod_id = resolve_pod_id(&args.id).await?;
    let volume_id = args
        .volume_id
        .unwrap_or_else(|| inferred_volume_id(&pod_id));

    let (model, backend) = resolve_model_backend_from_pod_id(&pod_id).await?;

    Ok(FlexServPodDeployment::from_existing(
        flexserv_server(Some(model), Some(backend))?,
        tapis_token()?,
        pod_id,
        volume_id,
    ))
}

async fn existing_hpc_deployment(args: JobIdArgs) -> CliResult<FlexServHPCDeployment> {
    // Resolve the job UUID with `resolve_job_id`.
    // Build `FlexServHPCDeployment::from_existing(...)`.
    // Set the tenant URL on the deployment.
    let job_uuid = resolve_job_id(&args.id).await?;
    let mut hpc_deployment = FlexServHPCDeployment::from_existing(tapis_token()?, job_uuid);

    hpc_deployment.tenant_url = Some(effective_tenant_url()?);

    Ok(hpc_deployment)
}

// -----------------------------------------------------------------------------
// Tapis clients and auth
// -----------------------------------------------------------------------------

fn pods_client() -> CliResult<TapisPods> {
    // Build the base URL as `<tenant_url>/v3`.
    // Load the token using `tapis_token`.
    // Construct `TapisPods::new(...)`.
    let tenant_url = effective_tenant_url()?;
    let api_base = format!("{}/v3", tenant_url.trim_end_matches('/'));
    let token = tapis_token()?;

    TapisPods::new(&api_base, Some(&token))
        .map_err(|e| anyhow!("failed to configure Tapis Pods client: {e}"))
}

fn jobs_config() -> CliResult<configuration::Configuration> {
    // Build a `configuration::Configuration`.
    // Set base_path to `<tenant_url>/v3`.
    // Set api_key with the token from `tapis_token`.
    let mut config = configuration::Configuration::default();
    let tenant_url = effective_tenant_url()?;
    config.base_path = format!("{}/v3", tenant_url.trim_end_matches('/'));
    config.api_key = Some(configuration::ApiKey {
        prefix: None,
        key: tapis_token()?,
    });

    Ok(config)
}

fn tapis_token() -> CliResult<String> {
    // Prefer TAPIS_TOKEN from the environment, then the saved config.
    // Otherwise return an auth-specific error.
    if let Some(token) = env_var_if_set("TAPIS_TOKEN") {
        return Ok(token);
    }

    if let Some(saved_config) = read_saved_config()? {
        if let Some(token) = saved_config.tapis_token {
            return Ok(token);
        }
    }

    Err(anyhow!("Missing token; please authenticate first"))
}

fn read_saved_config() -> CliResult<Option<SavedConfig>> {
    // Use `auth_config_path()` to find the config path.
    // If the file does not exist, return `Ok(None)`.
    // Read the file contents as a string.
    // Deserialize JSON into `SavedConfig` with `serde_json`.
    // Attach path-specific context to read and parse failures.
    let config_path = auth_config_path()?;

    if !config_path.exists() {
        return Ok(None);
    }

    let config_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read config file: {}", config_path.display()))?;

    let config: SavedConfig = serde_json::from_str(&config_str)
        .with_context(|| format!("failed to parse config file: {}", config_path.display()))?;

    Ok(Some(config))
}

fn write_saved_config(_config: &SavedConfig) -> CliResult<()> {
    // Use `auth_config_path()` to choose the file path.
    // Create the parent directory when it is missing.
    // Serialize `_config` as pretty JSON.
    // Write the JSON to disk.
    let config_path = auth_config_path()?;

    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config directory: {}", parent.display()))?;
    }

    let config_str = serde_json::to_string_pretty(_config)
        .with_context(|| format!("failed to serialize config to JSON"))?;

    std::fs::write(&config_path, config_str)
        .with_context(|| format!("failed to write config file: {}", config_path.display()))?;

    Ok(())
}

fn auth_config_path() -> CliResult<PathBuf> {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("could not find HOME to store flexserv-deployer config"))?;
    Ok(home.join(".flexserv-deployer").join("config.json"))
}

fn effective_tenant_url() -> CliResult<String> {
    // Resolve the tenant URL used by all Tapis clients.
    // Call `tapis_token()` to get the saved token.
    // Pass it to `tenant_url_from_jwt`.
    // Return the decoded tenant URL or a clear auth error.
    let token = tapis_token()?;
    tenant_url_from_jwt(&token)
}

fn effective_user() -> CliResult<String> {
    // Call `tapis_token()` to get the saved token.
    // Pass it to `user_from_jwt`.
    // Return the decoded username or a clear auth error.
    let token = tapis_token()?;
    user_from_jwt(&token)
}

fn jwt_claims(jwt_token: &str) -> CliResult<serde_json::Value> {
    // Split the JWT on '.' and take the second segment, the payload.
    // Base64url-decode that payload without padding.
    // Parse the decoded bytes as JSON.

    let payload = jwt_token
        .split('.')
        .nth(1)
        .ok_or_else(|| anyhow!("Invalid JWT token: missing payload segment"))?;

    let decoded_bytes = URL_SAFE_NO_PAD
        .decode(payload)
        .map_err(|e| anyhow!("Failed to decode JWT payload: {e}"))?;

    serde_json::from_slice(&decoded_bytes).map_err(|e| anyhow!("Failed to parse JWT claims: {e}"))
}

fn user_from_jwt(_jwt_token: &str) -> CliResult<String> {
    // Inspect real Tapis JWT claims and choose the canonical username field.
    // Candidate is `tapis/username`
    // Return a clear auth error if the token is malformed, expired, or lacks
    // the expected username claim.
    let claims = jwt_claims(_jwt_token)?;

    let username = claims
        .get("tapis/username")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("JWT token missing required username claim"))?;

    Ok(username.to_string())
}

fn tenant_url_from_jwt(jwt_token: &str) -> CliResult<String> {
    // Inspect real Tapis JWT claims and choose the canonical tenant field.
    // Candidate is `tapis/tenant_id`
    // Return a clear auth error if the token is malformed, expired, or lacks
    // the expected tenant claim.
    let claims = jwt_claims(jwt_token)?;

    let tenant_id = claims
        .get("tapis/tenant_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("JWT token missing required tenant id claim"))?
        .trim();

    if tenant_id.is_empty() {
        Err(anyhow!("JWT token has empty tenant id claim"))
    } else {
        Ok(format!("https://{}.tapis.io", tenant_id.to_lowercase()))
    }
}

fn token_expiration_status(jwt_token: &str) -> CliResult<String> {
    let claims = jwt_claims(jwt_token)?;
    let expires_at = claims
        .get("exp")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| anyhow!("JWT token missing required exp claim"))?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| anyhow!("system clock is before Unix epoch: {e}"))?
        .as_secs() as i64;

    if expires_at <= now {
        Ok("Expired".to_string())
    } else {
        Ok(format!("Expires in {}", format_duration(expires_at - now)))
    }
}

fn format_duration(seconds: i64) -> String {
    let days = seconds / 86_400;
    let hours = (seconds % 86_400) / 3_600;
    let minutes = (seconds % 3_600) / 60;
    let seconds = seconds % 60;

    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    }
}

fn effective_model(model: Option<String>) -> CliResult<String> {
    // Resolve the model id used by create commands.
    // Prefer command input, otherwise use the project default.
    Ok(model.unwrap_or_else(|| "openai-community/gpt2".to_string()))
}

fn effective_backend(backend: Option<BackendKind>) -> CliResult<BackendKind> {
    // Resolve the backend used by create commands.
    // Prefer command input, otherwise use `BackendKind::Transformers`.
    Ok(backend.unwrap_or(BackendKind::Transformers))
}

fn effective_hf_token() -> CliResult<Option<String>> {
    // Resolve the Hugging Face token.
    // Prefer HF_TOKEN from the environment, then the saved config.
    // Otherwise return `None`, since HF token can be optional for some deployments.
    if let Some(token) = env_var_if_set("HF_TOKEN") {
        return Ok(Some(token));
    }

    if let Some(saved_config) = read_saved_config()? {
        Ok(saved_config.hf_token)
    } else {
        Ok(None)
    }
}

fn env_var_if_set(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.trim().is_empty())
}

fn effective_output() -> CliResult<OutputFormat> {
    // Resolve the output format for command results.
    // Use `SavedConfig.output`.
    // Otherwise use `OutputFormat::Text`.
    if let Some(saved_config) = read_saved_config()? {
        if let Some(output) = saved_config.output {
            return Ok(output);
        }
    }
    Ok(OutputFormat::Text)
}

fn output_name(output: OutputFormat) -> &'static str {
    match output {
        OutputFormat::Text => "text",
        OutputFormat::Json => "json",
    }
}

fn print_config() -> CliResult<()> {
    // Print a clean user-facing view of effective configuration.
    // Include tenant_url, tapis_user, model, backend, output, token status,
    // Hugging Face token status, and config file path.
    // Never print raw token values.
    let tenant_url = effective_tenant_url().unwrap_or_else(|_| "unavailable".to_string());
    let tapis_user = effective_user().unwrap_or_else(|_| "unavailable".to_string());
    let output = effective_output()
        .map(|o| output_name(o).to_string())
        .unwrap_or_else(|_| "unavailable".to_string());

    let tapis_token_value = tapis_token();
    let token_expiration = tapis_token_value
        .as_ref()
        .ok()
        .and_then(|token| token_expiration_status(token).ok())
        .unwrap_or_else(|| "unavailable".to_string());
    let tapis_tkn = tapis_token_value.unwrap_or_else(|_| "not set".to_string());
    let hf_tkn = effective_hf_token()?.unwrap_or_else(|| "not set".to_string());

    let config_path = auth_config_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unavailable".to_string());

    println!("FlexServ Deployer configuration");
    println!("  tenant_url:   {tenant_url}");
    println!("  tapis_user:   {tapis_user}");
    println!("  output:       {output}");
    println!("  tapis_token:  {tapis_tkn}");
    println!("  hf_token:     {hf_tkn}");
    println!("  config_file:  {config_path}");
    println!("  expires:      {token_expiration}");

    Ok(())
}

fn print_next_steps() -> CliResult<()> {
    // Print the exact command a user should run next.
    println!("Great! Here are some next steps you can take:");
    println!("  flexserv-deployer manage show     Display current configuration");
    println!("");
    println!("  flexserv-deployer pod list        List your FlexServ pods");
    println!("  flexserv-deployer pod create      Create a new FlexServ pod");
    println!("");
    println!("  flexserv-deployer job list        List your FlexServ jobs");
    println!("  flexserv-deployer job create      Submit a new FlexServ job");
    Ok(())
}

// -----------------------------------------------------------------------------
// List commands and ID-prefix resolution
// -----------------------------------------------------------------------------

async fn get_all_pods() -> CliResult<Vec<PodResponseModel>> {
    // Build Tapis Pods client.
    // Call `client.pods.list_pods().await`.

    let client = pods_client()?;
    let response = client
        .pods
        .list_pods()
        .await
        .map_err(|e| anyhow!("{e:?}"))?;

    Ok(response.result)
}

async fn get_jobs(api_limit: Option<i32>) -> CliResult<Vec<JobListDto>> {
    // Build Tapis Jobs config.
    // Call `jobs_api::get_job_list`.

    let config = jobs_config()?;
    let response = jobs_api::get_job_list(
        &config,
        api_limit,
        None,
        None,
        Some("created(desc)"),
        None,
        None,
    )
    .await
    .map_err(|e| anyhow!("{e:?}"))?;

    Ok(response.result.unwrap_or_default())
}

async fn list_pods(limit: i32) -> CliResult<()> {
    // Limit rows.
    // Print JSON or a compact table.
    let pods = limited(get_all_pods().await?, limit);

    if matches!(effective_output()?, OutputFormat::Json) {
        let body = serde_json::to_string_pretty(&pods).context("failed to serialize pods list")?;
        println!("{body}");
        return Ok(());
    }

    print_pod_row(&["POD ID", "MODEL_NAME", "STATUS", "URL"]);
    for pod in pods {
        print_pod_row(&[
            &pod.pod_id,
            pod.environment_variables
                .as_ref()
                .and_then(|hashmap| hashmap.get("MODEL_NAME"))
                .and_then(|s| s.as_str())
                .unwrap_or_else(|| "-".into()),
            pod.status.as_deref().unwrap_or("-"),
            pod_url(&pod).as_deref().unwrap_or("-"),
        ]);
    }

    Ok(())
}

async fn list_jobs(limit: i32) -> CliResult<()> {
    // Build Tapis Jobs config.
    // Call `jobs_api::get_job_list`.
    // Limit/order rows as needed.
    // Print JSON or a compact table.
    let api_limit = if limit < 0 { None } else { Some(limit) };
    let jobs = limited(get_jobs(api_limit).await?, limit);

    if matches!(effective_output()?, OutputFormat::Json) {
        let body = serde_json::to_string_pretty(&jobs).context("failed to serialize jobs list")?;
        println!("{body}");
        return Ok(());
    }

    print_job_row(&["JOB UUID", "NAME", "STATUS"]);
    for job in jobs {
        print_job_row(&[
            job.uuid.as_deref().unwrap_or("-"),
            job.name.as_deref().unwrap_or("-"),
            job_status(&job).as_deref().unwrap_or("-"),
        ]);
    }

    Ok(())
}

async fn resolve_pod_id(id_or_prefix: &str) -> CliResult<String> {
    // List pods visible to the user.
    // Feed pod ids into `resolve_unique_prefix`.
    let pods = get_all_pods().await?;

    resolve_unique_prefix(
        id_or_prefix,
        pods.iter().map(|pod| pod.pod_id.as_str()),
        "pod",
    )
}

async fn resolve_job_id(id_or_prefix: &str) -> CliResult<String> {
    // List recent jobs visible to the user.
    // Feed job UUIDs into `resolve_unique_prefix`.
    let jobs = get_jobs(Some(-1)).await?;

    resolve_unique_prefix(
        id_or_prefix,
        jobs.iter().filter_map(|job| job.uuid.as_deref()),
        "job",
    )
}

async fn resolve_model_backend_from_pod_id(pod_id: &str) -> CliResult<(String, BackendKind)> {
    // Find the pod with the given pod_id.
    // Return its model_id and backend.
    let client = pods_client()?;
    let pods_result = match client.pods.get_pod(pod_id, None, None).await {
        Ok(resp) => resp,
        Err(e) => {
            return Err(anyhow!("Failed to get pod with id '{pod_id}': {e}"));
        }
    };

    let resp = *pods_result.result;

    let model_name = resp
        .environment_variables
        .as_ref()
        .and_then(|hashmap| hashmap.get("MODEL_NAME"))
        .and_then(|value| value.as_str())
        .unwrap_or("-")
        .to_string();

    let args: Vec<_> = resp
        .arguments
        .and_then(|inner_option| inner_option)
        .ok_or_else(|| anyhow!("pod with id '{pod_id}' has no arguments"))?
        .into_iter()
        .collect();

    let backend_str = args
        .iter()
        .position(|arg| arg == "---backend-kind")
        .and_then(|index| args.get(index + 1))
        .map(|s| s.as_str())
        .unwrap_or("transformers");

    let backend = match backend_str {
        "transformers" => BackendKind::Transformers,
        "vllm" => BackendKind::Vllm,
        "sglang" => BackendKind::Sglang,
        "trtllm" => BackendKind::Trtllm,
        _ => {
            return Err(anyhow!(
                "pod with id '{pod_id}' has unknown BACKEND '{backend_str}'"
            ));
        }
    };

    Ok((model_name, backend))
}

fn resolve_unique_prefix<'a>(
    id_or_prefix: &str,
    candidates: impl Iterator<Item = &'a str>,
    resource: &str,
) -> CliResult<String> {
    // Return the full id when exactly one prefix matches.
    // Return a not-found error when zero prefixes match.
    // Return an ambiguity error with the matches when multiple prefixes match.
    let matches: Vec<&str> = candidates
        .filter(|candidate| candidate == &id_or_prefix || candidate.starts_with(id_or_prefix))
        .collect::<Vec<_>>();

    match matches.as_slice() {
        [] => Err(anyhow!("no {resource} id or prefix '{id_or_prefix}' found")),
        [single] => Ok((*single).to_string()),
        multiple => {
            let match_list = multiple.join(", ");
            Err(anyhow!(
                "ambiguous {resource} id prefix '{id_or_prefix}' matched: {match_list}"
            ))
        }
    }
}

fn inferred_volume_id(pod_id: &str) -> String {
    // If pod id starts with `p`, replace that first character with `v`.
    // Otherwise return an empty string.
    pod_id
        .strip_prefix('p')
        .map(|suffix| format!("v{suffix}"))
        .unwrap_or_else(|| "".to_string())
}

// -----------------------------------------------------------------------------
// Formatting helpers
// -----------------------------------------------------------------------------

fn pod_url(pod: &PodResponseModel) -> Option<String> {
    pod.networking
        .as_ref()
        .and_then(|networking| networking.get("default"))
        .and_then(|default_net| default_net.url.clone())
}

fn job_status(job: &JobListDto) -> Option<String> {
    job.status.map(|s| {
        match s {
            tapis_jobs::models::job_list_dto::Status::Pending => "PENDING",
            tapis_jobs::models::job_list_dto::Status::ProcessingInputs => "PROCESSING_INPUTS",
            tapis_jobs::models::job_list_dto::Status::StagingInputs => "STAGING_INPUTS",
            tapis_jobs::models::job_list_dto::Status::StagingJob => "STAGING_JOB",
            tapis_jobs::models::job_list_dto::Status::SubmittingJob => "SUBMITTING_JOB",
            tapis_jobs::models::job_list_dto::Status::Queued => "QUEUED",
            tapis_jobs::models::job_list_dto::Status::Running => "RUNNING",
            tapis_jobs::models::job_list_dto::Status::Archiving => "ARCHIVING",
            tapis_jobs::models::job_list_dto::Status::Blocked => "BLOCKED",
            tapis_jobs::models::job_list_dto::Status::Paused => "PAUSED",
            tapis_jobs::models::job_list_dto::Status::Finished => "FINISHED",
            tapis_jobs::models::job_list_dto::Status::Cancelled => "CANCELLED",
            tapis_jobs::models::job_list_dto::Status::Failed => "FAILED",
        }
        .to_string()
    })
}

fn limited<T>(items: Vec<T>, limit: i32) -> Vec<T> {
    // Return at most `limit` items. Treat negative limits as unbounded.
    if limit < 0 {
        items
    } else {
        items.into_iter().take(limit as usize).collect()
    }
}

fn print_deployment_result(format: OutputFormat, result: &DeploymentResult) -> CliResult<()> {
    // If JSON, pretty-print the DeploymentResult.
    // If text and pod result, print pod_id, volume_id, status, url, user, tenant, and model.
    // If text and HPC result, print job_uuid, status, url, and token.
    // When JSON serialization fails, attach context with `.context(...)`.
    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(result)
                .context("Failed to serialize DeploymentResult to JSON")?;
            println!("{json}");
        }
        OutputFormat::Text => match result {
            DeploymentResult::PodResult {
                pod_id,
                volume_id,
                pod_url,
                status,
                pod_info: _,
                tapis_user,
                tapis_tenant,
                model_id,
                ..
            } => {
                println!("Pod deployment result:");
                println!("  pod_id:     {pod_id}");
                println!("  volume_id:  {volume_id}");
                println!("  status:     {}", status.as_deref().unwrap_or("-"));
                println!("  pod_url:    {}", pod_url.as_deref().unwrap_or("-"));
                println!("  tapis_user: {tapis_user}");
                println!("  tapis_tenant: {tapis_tenant}");
                // println!("  flexserv_secret: {flexserv_secret}");
                println!("  model_id:   {model_id}");
            }
            DeploymentResult::HPCResult {
                job_uuid,
                status,
                hpc_url,
                flexserv_token,
                ..
            } => {
                println!("HPC deployment result:");
                println!("  job_uuid:   {job_uuid}");
                println!("  status:     {}", status.as_deref().unwrap_or("-"));
                println!("  hpc_url:    {}", hpc_url.as_deref().unwrap_or("-"));
                println!(
                    "  flexserv_secret: {}",
                    flexserv_token.as_deref().unwrap_or("-")
                );
            }
        },
    }

    Ok(())
}

fn print_job_row(columns: &[&str; 3]) {
    println!("{:<46}  {:<24}  {:<12}", columns[0], columns[1], columns[2]);
}

fn print_pod_row(columns: &[&str; 4]) {
    println!(
        "{:<16}  {:<24}  {:<12}  {:<30}",
        columns[0], columns[1], columns[2], columns[3]
    );
}

// -----------------------------------------------------------------------------
// Friendly errors
// -----------------------------------------------------------------------------

fn print_cli_error(err: &anyhow::Error) {
    // Print the original error.
    // If the chain contains a DeploymentError, format it as a deployment error.
    // If `is_auth_error(err)` is true, print exact next-step commands:
    // `flexserv-deployer authenticate <JWT_TOKEN>`.

    if is_auth_error(err) {
        eprintln!("Authentication error: {err}");
        eprintln!("--------------------------------");
        print_auth_next_steps();
    } else if let Some(deployment_error) = deployment_error_from_chain(err) {
        eprintln!("Deployment error: {deployment_error}");
    } else {
        eprintln!("Error: {err}");
    }
}

fn print_auth_next_steps() {
    let env_token_is_set = env_var_if_set("TAPIS_TOKEN").is_some();
    let saved_token_is_set = read_saved_config()
        .ok()
        .flatten()
        .and_then(|saved_config| saved_config.tapis_token)
        .is_some();

    if env_token_is_set {
        eprintln!("TAPIS_TOKEN is set, but Tapis rejected it or it is not a valid Tapis JWT.");
        eprintln!("Next steps: replace TAPIS_TOKEN with a fresh Tapis JWT and retry.");
    } else if saved_token_is_set {
        eprintln!("The saved manager Tapis token is set, but Tapis rejected it or it is not a valid Tapis JWT.");
        eprintln!("Next steps: refresh the saved token and retry:");
        eprintln!("  flexserv-deployer manage authenticate <JWT_TOKEN>");
    } else {
        eprintln!("No Tapis token is configured.");
        eprintln!("Next steps: set one of the following and retry:");
        eprintln!("  export TAPIS_TOKEN=<JWT_TOKEN>");
        eprintln!("  flexserv-deployer manage authenticate <JWT_TOKEN>");
    }
}

fn deployment_error_from_chain(err: &anyhow::Error) -> Option<&DeploymentError> {
    // Walk the anyhow error chain and return the first DeploymentError.
    err.chain()
        .find_map(|cause| cause.downcast_ref::<DeploymentError>())
}

fn is_auth_error(err: &anyhow::Error) -> bool {
    // Decide whether an error is authentication-related.
    // First check for DeploymentError::TapisAuthFailed in the error chain.
    // Then inspect the full error text for confirmed token-auth messages.
    if let Some(deployment_error) = deployment_error_from_chain(err) {
        if matches!(deployment_error, DeploymentError::TapisAuthFailed(_)) {
            return true;
        }
    }

    let err_str = error_chain_text(err).to_lowercase();
    err_str.contains("missing token; please authenticate first")
        || err_str.contains("invalid jwt token")
        || err_str.contains("failed to decode jwt payload")
        || err_str.contains("failed to parse jwt claims")
        || err_str.contains("jwt token missing required")
        || err_str.contains("jwt token has empty")
        || err_str.contains("invalid tapis token")
        || err_str.contains("could not parse the tapis access token")
        || err_str.contains("missing json web token")
        || err_str.contains("rejected due to missing json web token")
        || err_str.contains("token was expected to have")
        || err_str.contains("unauthorized")
        || err_str.contains("401")
        || err_str.contains("403")
        || (err_str.contains("token") && err_str.contains("expired"))
}

fn error_chain_text(err: &anyhow::Error) -> String {
    let mut text = String::new();
    for cause in err.chain() {
        if !text.is_empty() {
            text.push_str(" | ");
        }
        text.push_str(&cause.to_string());
    }
    text
}
