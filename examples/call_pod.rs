//! Example: call a running FlexServ pod (health, models, completion).
//!
//! --- Working curl commands (copy-paste; set POD_URL first) ---
//!
//!   export POD_URL="https://YOUR_POD.pods.tacc.tapis.io"   # from create_pod (pod_url)
//!   export TOKEN="openai-community_gpt2"                   # from create_pod (auth_token)
//!
//!   # Health
//!   curl -s -H "Authorization: Bearer $TOKEN" "$POD_URL/v1/flexserv/health"
//!
//!   # Models
//!   curl -s -H "Authorization: Bearer $TOKEN" "$POD_URL/v1/models"
//!
//!   # Completions (GPT-2 has no chat template; use /v1/completions with "prompt")
//!   curl -s -X POST "$POD_URL/v1/completions" \
//!     -H "Content-Type: application/json" \
//!     -H "Authorization: Bearer $TOKEN" \
//!     -d '{"model":"/app/models/openai-community_gpt2","prompt":"The capital of France is","max_tokens":20}'
//!
//! Use https (no port) for POD_URL. If 401: ingress may strip headers (ask TAPIS to forward Authorization).
//!
//! --- Rust example ---
//!
//!   export POD_URL=https://<your-pod-url>
//!   export FLEXSERV_TOKEN=openai-community_gpt2
//!   cargo run --example call_pod
//!
//! Or: cargo run --example call_pod -- <POD_URL> <FLEXSERV_TOKEN>

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let (base_url, token) = if let (Ok(url), Ok(t)) = (
        std::env::var("POD_URL"),
        std::env::var("FLEXSERV_TOKEN"),
    ) {
        (url, t)
    } else if let [url, token, ..] = std::env::args().collect::<Vec<_>>().as_slice() {
        (url.clone(), token.clone())
    } else {
        eprintln!("Usage: POD_URL=... FLEXSERV_TOKEN=... cargo run --example call_pod");
        eprintln!("   or: cargo run --example call_pod -- <POD_URL> <FLEXSERV_TOKEN>");
        eprintln!("Example token for GPT-2 (no FLEXSERV_SECRET): openai-community_gpt2");
        std::process::exit(1);
    };

    let base_url = base_url.trim_end_matches('/');
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    // Send both: X-FlexServ-Secret and Authorization Bearer (proxies often strip custom headers)
    let token_val = HeaderValue::from_str(&token).map_err(|e| format!("invalid token: {}", e))?;
    let bearer = HeaderValue::from_str(&format!("Bearer {}", token))
        .map_err(|e| format!("invalid token: {}", e))?;
    let mut headers = HeaderMap::new();
    headers.insert("X-FlexServ-Secret", token_val.clone());
    headers.insert(AUTHORIZATION, bearer);

    println!("POD_URL: {}", base_url);
    println!("Token:   {}...\n", if token.len() > 12 { &token[..12] } else { &token });

    // 1. Health
    println!("--- GET /v1/flexserv/health ---");
    let url = format!("{}/v1/flexserv/health", base_url);
    let resp = client.get(&url).headers(headers.clone()).send()?;
    let status = resp.status();
    let body = resp.text()?;
    println!("Status: {}", status);
    println!("Body:   {}", body);
    println!();

    // 2. Models
    println!("--- GET /v1/models ---");
    let url = format!("{}/v1/models", base_url);
    let resp = client.get(&url).headers(headers.clone()).send()?;
    let status = resp.status();
    let body = resp.text()?;
    println!("Status: {}", status);
    println!("Body:   {}", body);
    println!();

    // 3. Completions (GPT-2 has no chat template; use /v1/completions with "prompt")
    let model_path = format!("/app/models/{}", token);
    println!("--- POST /v1/completions ---");
    let url = format!("{}/v1/completions", base_url);
    let body_json = serde_json::json!({
        "model": model_path,
        "prompt": "The capital of France is",
        "max_tokens": 20
    });
    let resp = client
        .post(&url)
        .headers(headers.clone())
        .json(&body_json)
        .send()?;
    let status = resp.status();
    let body = resp.text()?;
    println!("Status: {}", status);
    println!("Body:   {}", body);

    Ok(())
}
