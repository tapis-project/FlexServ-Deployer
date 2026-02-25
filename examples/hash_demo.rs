use flexserv_deployer::{Backend, FlexServInstance};

fn main() {
    // Create a FlexServ instance
    let server = FlexServInstance::new(
        "https://tacc.tapis.io".to_string(),
        "testuser".to_string(),
        "meta-llama/Llama-3-70b-hf".to_string(),
        None,
        None,
        None,
        Backend::Transformers {
            command_prefix: vec![
                "/app/venvs/transformers/bin/python".to_string(),
                "/app/python/flexserv.py".to_string(),
            ],
        },
    );

    // Generate deployment hash
    let hash = server.deployment_hash();
    println!(
        "Deployment hash (first 12 chars of base62-encoded SHA256): {}",
        hash
    );
    println!("Hash length: {}", hash.len());

    // Demonstrate that the same configuration always produces the same hash
    let hash2 = server.deployment_hash();
    println!("Hash is deterministic: {}", hash == hash2);
}
