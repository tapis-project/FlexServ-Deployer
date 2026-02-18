use actix_web::{test, web, App};
use flexserv_deployer::{Backend, FlexServInstance};

async fn health() -> actix_web::HttpResponse {
    actix_web::HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "service": "flexserv-deployer"
    }))
}

async fn get_models() -> actix_web::HttpResponse {
    actix_web::HttpResponse::Ok().json(serde_json::json!({
        "models": ["transformers", "vllm", "sglang", "trtllm"]
    }))
}

#[actix_rt::test]
async fn test_health_endpoint() {
    let app = test::init_service(App::new().route("/health", web::get().to(health))).await;

    let req = test::TestRequest::get().uri("/health").to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
}

#[actix_rt::test]
async fn test_models_endpoint() {
    let app = test::init_service(App::new().route("/models", web::get().to(get_models))).await;

    let req = test::TestRequest::get().uri("/models").to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
}

#[test]
fn test_flexserv_library() {
    let server = FlexServInstance::new(
        "https://tacc.tapis.io".to_string(),
        "testuser".to_string(),
        "meta-llama/Llama-2-7b".to_string(),
        None,
        Backend::Transformers {
            command: vec!["python".to_string()],
        },
    );

    assert_eq!(server.tenant_url, "https://tacc.tapis.io");
    assert_eq!(server.tapis_user, "testuser");
    assert!(!server.deployment_hash().is_empty());
}
