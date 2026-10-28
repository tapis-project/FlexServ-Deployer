# Closed input to the existing Rust host policy generator; no secrets are embedded.
{output_dir:$output,principal:$principal,runtime_root:$root,
 existing_venv_path:$venv,private_repo:$private,public_repo:$public,
 work_dir:($output+"/work"),uv:$uv,python:($venv+"/bin/python"),hardware_env:$hardware,
 backend:$backend,gpu_devices:($devices|split(",")|map(select(length>0))),
 bind:$bind,advertise_host:$host,ports:[$start,$end]}
+ if $packs==null then {pack:$pack,approved_checksums:$digest}
  else {packs:($packs|map({pack:.path,approved_checksums:.checksums_sha256}))} end
+ if $credential then {gateway_credential_binding:"vllm-key"} else {} end
