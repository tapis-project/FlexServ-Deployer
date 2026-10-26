# Closed input to the existing Rust host policy generator; no secrets are embedded.
{pack:$pack,output_dir:$output,principal:$principal,runtime_root:$root,
 existing_venv_path:$venv,private_repo:$private,public_repo:$public,
 work_dir:($output+"/work"),uv:$uv,python:($venv+"/bin/python"),hardware_env:$hardware,
 backend:$backend,gpu_devices:($devices|split(",")|map(select(length>0))),
 bind:$bind,advertise_host:$host,ports:[$start,$end],approved_checksums:$digest}
+ if $credential then {gateway_credential_binding:"vllm-key"} else {} end
