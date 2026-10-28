# Gateway validates these shared contracts again before publishing any readiness.
{schema_version:1,job_id:$job,deployment_id:("slurm-"+$job),run_id:$run,runtime_dir:$runtime,
 tls_dir:($root+"/tmp/tls/"+$run),nodes_file:($runtime+"/nodes.txt"),run_not_after_utc:$until,
 fcs_arguments:(if $packs==null then ["--pack",($root+"/runtime/pack"),"--approved-checksums",$digest]
                else [$packs[] | "--pack",.path,"--approved-checksums",.checksums_sha256,"--approved-zip-sha256",.zip_sha256] end),
 startup_timeout_seconds:$timeout,observer_principal:"gateway-observer",
 observer_scopes:["mcp:read","node:read","services:read"],observer_owners:[$principal],
 delegated_principals:[$principal],delegated_scopes:["mcp:read","mcp:call"],
 archive:{root:($root+"/reports")},mcp:{schema_version:1,operations:{},callers:[{
 principal:$principal,token_env:"FCS_MCP_TOKEN",scopes:["mcp:read","mcp:call"],allowed_tools:$tools}]}}
+ if $model=="" then {} else {routing:{schema_version:1,rules:[{
 id:"local-vllm",owner:$principal,profile:"local_vllm",
 source:{name:"io.flexserv.deployments.vllm",version:"1.0.0"},model_id:$model,
 backend_key:"vllm",members:{},endpoints:$endpoints,
 controls:{inspect:{name:"io.flexserv.deployments.inspect",version:"1.0.0"},
 output:{name:"io.flexserv.deployments.output",version:"1.0.0"},
 stop:{name:"io.flexserv.deployments.stop",version:"1.0.0"}}
 } + if $credential then {credential:{binding_id:"vllm-key",value_env:"GATEWAY_VLLM_API_KEY"}} else {} end]}} end
