#!/usr/bin/env bash
# Start one Gateway and one Slurm step; Rust owns transport identity and tool processes.
fcs_worker() {
    export FLEXSERV_JOB_DIR=$1
    select_arch
    local runtime=$2 rank=${SLURM_PROCID:?Missing Slurm rank}
    FCS_BIN="$FLEXSERV_JOB_DIR/bin/flexserv-capability-server-$ARCH.bin"
    local args=(--slurm-rank-config "$runtime/rank-launch.json")
    if [[ "$FLEXSERV_PREPARE_BACKENDS" == 1 ]]; then
        local policy="$runtime/config/host-$rank"
        mkdir -p "$policy"; chmod 700 "$policy"
        python3 - "$policy/request.json" "$policy" <<'PY'
import hashlib,ipaddress,json,os,pathlib,socket,sys
from urllib.parse import urlsplit
e=os.environ; root=pathlib.Path(e['FLEXSERV_JOB_DIR']); policy=pathlib.Path(sys.argv[2])
node=e.get('SLURMD_NODENAME',socket.getfqdn())
addresses=socket.getaddrinfo(node,None,socket.AF_INET,socket.SOCK_STREAM)
usable=[r[4][0] for r in addresses if not (ipaddress.ip_address(r[4][0]).is_loopback or ipaddress.ip_address(r[4][0]).is_unspecified or ipaddress.ip_address(r[4][0]).is_multicast)]
bind=e.get('FLEXSERV_SERVICE_BIND') or (usable[0] if usable else None)
if not bind: raise SystemExit('No routable service address; set FLEXSERV_SERVICE_BIND explicitly')
raw=e.get('CUDA_VISIBLE_DEVICES') or e.get('ROCR_VISIBLE_DEVICES') or e.get('ZE_AFFINITY_MASK') or ''
gpus=e.get('FLEXSERV_GPU_DEVICES',raw).split(',')
if not raw and not e.get('FLEXSERV_GPU_DEVICES'):
    raise SystemExit('Set FLEXSERV_GPU_DEVICES to the Slurm-approved device IDs before deployment')
request=dict(pack=str(root/'runtime/pack'),output_dir=str(policy),principal=e['MCP_PRINCIPAL'],
 runtime_root=e['FLEXSERV_VENV_ROOT'],existing_venv_path=e['FLEXSERV_VLLM_VENV_PATH'],
 private_repo=e.get('PRI_MODEL_REPO',str(root/'runtime/private-models')),public_repo=e.get('PUB_MODEL_REPO'),
 work_dir=str(policy/'work'),uv=str(pathlib.Path.home()/'.local/bin/uv'),python=e['FLEXSERV_VLLM_VENV_PATH']+'/bin/python',
 hardware_env=e['HARDWARE_ENV'],backend=e['FLEXSERV_ACCELERATOR_BACKEND'],gpu_devices=[v for v in gpus if v],
 bind=bind,advertise_host=node,ports=[int(e.get('FLEXSERV_SERVICE_PORT_START','18000')),int(e.get('FLEXSERV_SERVICE_PORT_END','18100'))],
 approved_checksums=hashlib.sha256((root/'runtime/pack/checksums.sha256').read_bytes()).hexdigest())
if e.get('FCS_VLLM_API_KEY'): request['gateway_credential_binding']='vllm-key'
for p in [pathlib.Path(request['private_repo']),pathlib.Path(request['work_dir'])]: p.mkdir(parents=True,exist_ok=True,mode=0o700)
pathlib.Path(sys.argv[1]).write_text(json.dumps(request));pathlib.Path(sys.argv[1]).chmod(0o600)
PY
        "$FCS_BIN" --prepare-host-config "$policy/request.json" > "$policy/generated.json"
        args+=(--deploy-policy "$policy/deployment.json" --connections "$policy/connections.json" --workflow-policy "$policy/workflow.json")
    fi
    # Public Gateway credentials do not enter FCS workers or tool environments.
    unset FLEXSERV_ADMIN_SECRET FCS_MCP_TOKEN FLEXSERV_TOKEN RAND_TOKEN GATEWAY_VLLM_API_KEY
    exec "$FCS_BIN" ${args[@]+"${args[@]}"}
}
wait_marker() {
    local marker=$1 end=$((SECONDS+STARTUP_TIMEOUT_SECONDS))
    while [[ ! -f "$marker" ]]; do
        kill -0 "$GATEWAY_PID" 2>/dev/null || { wait "$GATEWAY_PID"; return 1; }
        [[ ! -f "$FLEXSERV_RUNTIME/failed.json" ]] || { echo 'Gateway job initialization failed; inspect logs.' >&2; return 1; }
        (( SECONDS < end )) || { echo "Startup timed out waiting for ${marker##*/}." >&2; return 1; }
        sleep 0.2
    done
}
launch_flexserv() {
    export FLEXSERV_RUN_ID="run-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
    export FLEXSERV_RUNTIME="$FLEXSERV_JOB_DIR/runtime/$FLEXSERV_RUN_ID"
    mkdir -p "$FLEXSERV_RUNTIME"; chmod 700 "$FLEXSERV_RUNTIME"
    scontrol show hostnames "$SLURM_JOB_NODELIST" > "$FLEXSERV_RUNTIME/nodes.txt"
    chmod 600 "$FLEXSERV_RUNTIME/nodes.txt"
    python3 - "$FLEXSERV_RUNTIME/job-launch.json" <<'PY'
import datetime as dt,hashlib,json,os,pathlib,socket,subprocess,sys
e=os.environ; root=pathlib.Path(e['FLEXSERV_JOB_DIR']); runtime=pathlib.Path(e['FLEXSERV_RUNTIME'])
nodes=(runtime/'nodes.txt').read_text().splitlines()
if not nodes or len(nodes)!=len(set(nodes)): raise SystemExit('Invalid Slurm node list')
job=subprocess.check_output(['scontrol','show','job','-o',e['SLURM_JOB_ID']],text=True,env=dict(e,TZ='UTC'))
fields=dict(w.split('=',1) for w in job.split() if '=' in w)
end=dt.datetime.fromisoformat(fields['EndTime']).replace(tzinfo=dt.timezone.utc)-dt.timedelta(seconds=10)
if e.get('RUN_SECONDS'): end=min(end,dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=int(e['RUN_SECONDS'])))
if end<=dt.datetime.now(dt.timezone.utc): raise SystemExit('No job walltime remains')
pack=root/'runtime/pack'; checksum=hashlib.sha256((pack/'checksums.sha256').read_bytes()).hexdigest()
# Grant only explicitly public wrappers; hidden/native tools remain internal.
tools=[]
for p in pack.glob('fcs_tools/**/tool.yaml'):
    lines=p.read_text().splitlines()
    if any(line.strip()=='visibility: true' for line in lines):
        tools.extend(line.split(':',1)[1].strip() for line in lines if line.startswith('  name:'))
tools+=['gateway.operations.'+x for x in ['status','result','cancel','list']]
config=dict(schema_version=1,job_id=e['SLURM_JOB_ID'],deployment_id='slurm-'+e['SLURM_JOB_ID'],run_id=e['FLEXSERV_RUN_ID'],
 runtime_dir=str(runtime),tls_dir=str(root/'tmp/tls'/e['FLEXSERV_RUN_ID']),nodes_file=str(runtime/'nodes.txt'),
 run_not_after_utc=end.isoformat().replace('+00:00','Z'),
 fcs_arguments=['--pack',str(pack),'--approved-checksums',checksum],
 startup_timeout_seconds=int(e['STARTUP_TIMEOUT_SECONDS']),observer_principal='gateway-observer',
 observer_scopes=['mcp:read','node:read','services:read'],observer_owners=[e['MCP_PRINCIPAL']],
 delegated_principals=[e['MCP_PRINCIPAL']],delegated_scopes=['mcp:read','mcp:call'],
 archive={'root':str(root/'reports')},mcp=dict(schema_version=1,operations={},callers=[dict(principal=e['MCP_PRINCIPAL'],token_env='FCS_MCP_TOKEN',scopes=['mcp:read','mcp:call'],allowed_tools=sorted(set(tools)))]))
if e.get('FLEXSERV_MODEL_ID'):
    endpoints={}
    for rank,node in enumerate(nodes):
        addresses=sorted({r[4][0] for r in socket.getaddrinfo(node,None,socket.AF_INET,socket.SOCK_STREAM)})
        endpoints['fcs-'+str(rank)]={'host':node,'addresses':addresses,'scheme':'http','port_start':int(e.get('FLEXSERV_SERVICE_PORT_START','18000')),'port_end':int(e.get('FLEXSERV_SERVICE_PORT_END','18100'))}
    rule=dict(id='local-vllm',owner=e['MCP_PRINCIPAL'],profile='local_vllm',source={'name':'io.flexserv.deployments.vllm','version':'1.0.0'},model_id=e['FLEXSERV_MODEL_ID'],backend_key='vllm',members={},endpoints=endpoints,
      controls={k:{'name':'io.flexserv.deployments.'+k,'version':'1.0.0'} for k in ['inspect','output','stop']})
    if e.get('FCS_VLLM_API_KEY'): rule['credential']={'binding_id':'vllm-key','value_env':'GATEWAY_VLLM_API_KEY'}
    config['routing']={'schema_version':1,'rules':[rule]}
pathlib.Path(sys.argv[1]).write_text(json.dumps(config,indent=2)+'\n');pathlib.Path(sys.argv[1]).chmod(0o600)
PY
    local bind=${GATEWAY_BIND_HOST:-127.0.0.1} args=() network origin
    for network in ${FLEXSERV_TRUSTED_PROXY_NETWORKS:-}; do args+=(--trusted-proxy-network "$network"); done
    for network in ${FLEXSERV_LOGIN_PUBLIC_HOSTS:-}; do args+=(--mcp-allowed-host "$network:${LOGIN_PORT:-$LOCAL_PORT}"); done
    for origin in ${FLEXSERV_MCP_ALLOWED_ORIGINS:-}; do args+=(--mcp-allowed-origin "$origin"); done
    if [[ -n "${FCS_VLLM_API_KEY:-}" ]]; then export GATEWAY_VLLM_API_KEY="$FCS_VLLM_API_KEY"; fi
    export FCS_MCP_TOKEN="$RAND_TOKEN" FLEXSERV_TOKEN="$RAND_TOKEN"
    FLEXSERV_ADMIN_SECRET=${FLEXSERV_ADMIN_SECRET:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}
    export FLEXSERV_ADMIN_SECRET
    mkdir -p "$FLEXSERV_JOB_DIR/runtime/private-models" "$FLEXSERV_JOB_DIR/runtime/public-models"
    "$GATEWAY_BIN" --slurm-job-config "$FLEXSERV_RUNTIME/job-launch.json" \
        --host "$bind" --port "$LOCAL_PORT" --workers 2 \
        --mcp-allowed-host "localhost:$LOCAL_PORT" --mcp-allowed-host "127.0.0.1:$LOCAL_PORT" \
        --mcp-allowed-host "$NODE_HOSTNAME_DOMAIN:$LOCAL_PORT" \
        --mcp-allowed-host "${OSC_ONDEMAND_HOST:-ondemand.osc.edu}" \
        --mcp-allowed-host "${FLEXSERV_PUBLIC_HOST:-$NODE_HOSTNAME_DOMAIN}:${LOGIN_PORT:-$LOCAL_PORT}" \
        --log-file "$FLEXSERV_JOB_DIR/logs/gateway.log" --trace-log-file "$FLEXSERV_JOB_DIR/logs/trace.log" \
        --startup-lock-file "$FLEXSERV_RUNTIME/start.lock" --readiness-lock-file "$FLEXSERV_RUNTIME/ready.lock" \
        --runtime-config-dir "$FLEXSERV_RUNTIME/gateway" --fetch-state-dir "$FLEXSERV_RUNTIME/fetch" \
        --pub-model-repo "${PUB_MODEL_REPO:-$FLEXSERV_JOB_DIR/runtime/public-models}" \
        --pri-model-repo "${PRI_MODEL_REPO:-$FLEXSERV_JOB_DIR/runtime/private-models}" \
        ${args[@]+"${args[@]}"} >> "$FLEXSERV_JOB_DIR/logs/gateway-startup.log" 2>&1 &
    GATEWAY_PID=$!
    wait_marker "$FLEXSERV_RUNTIME/config-ready.json"
    local count
    count=$(wc -l < "$FLEXSERV_RUNTIME/nodes.txt")
    srun --overlap --exact --nodes="$count" --ntasks="$count" --ntasks-per-node=1 \
        --cpus-per-task="$FCS_CPUS_PER_TASK" --kill-on-bad-exit=1 --export=ALL \
        bash "$FLEXSERV_JOB_DIR/app/run_flexserv.sh" --fcs-worker "$FLEXSERV_RUNTIME" \
        >> "$FLEXSERV_JOB_DIR/logs/fcs.log" 2>&1 &
    SRUN_PID=$!
    wait_marker "$FLEXSERV_RUNTIME/ready.json"
}
wait_for_services() {
    while kill -0 "$GATEWAY_PID" 2>/dev/null && kill -0 "$SRUN_PID" 2>/dev/null; do
        sleep 1
    done
    echo 'A main service exited; shutting down this job run.' >&2
    if kill -0 "$GATEWAY_PID" 2>/dev/null; then wait "$SRUN_PID"; else wait "$GATEWAY_PID"; fi
}
cleanup_flexserv() {
    local pid end=$((SECONDS+5))
    for pid in ${TUNNEL_PIDS[@]+"${TUNNEL_PIDS[@]}"} "${SRUN_PID:-}" "${GATEWAY_PID:-}"; do
        [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in ${TUNNEL_PIDS[@]+"${TUNNEL_PIDS[@]}"} "${SRUN_PID:-}" "${GATEWAY_PID:-}"; do
        [[ -n "$pid" ]] || continue
        while kill -0 "$pid" 2>/dev/null && (( SECONDS < end )); do sleep 0.1; done
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    # Only this invocation's staging leaf is owned; persistent evidence is retained.
    if [[ -n "${FLEXSERV_RUN_ID:-}" ]]; then
        python3 - "$FLEXSERV_JOB_DIR" "$FLEXSERV_RUN_ID" <<'PY'
import pathlib,re,shutil,sys
root=pathlib.Path(sys.argv[1]).resolve(); run=sys.argv[2]
if not re.fullmatch(r'run-[0-9a-f]{32}',run): raise SystemExit('Invalid owned run ID')
parent=root/'tmp/tls'; leaf=parent/run
if parent.is_symlink() or leaf.is_symlink(): raise SystemExit('Refusing linked TLS cleanup path')
if leaf.exists(): shutil.rmtree(leaf)
if parent.exists() and not any(parent.iterdir()): parent.rmdir()
PY
    fi
    concrete_cleanup_site_resources
}
