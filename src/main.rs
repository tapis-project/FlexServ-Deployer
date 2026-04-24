use actix_web::{middleware, web, App, HttpResponse, HttpServer, Responder};

async fn health() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "service": "flexserv-deployer"
    }))
}

async fn get_models() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "models": ["transformers", "vllm", "sglang", "trtllm"]
    }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    log::info!("Starting FlexServ Deployer Server...");

    HttpServer::new(|| {
        App::new()
            .wrap(middleware::Logger::default())
            .route("/health", web::get().to(health))
            .route("/models", web::get().to(get_models))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
