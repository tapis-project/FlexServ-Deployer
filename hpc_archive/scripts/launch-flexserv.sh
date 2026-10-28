#!/usr/bin/env bash
# Bash owns Slurm orchestration; Gateway/FCS own identities, policies and processes.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

fcs_worker() {
    export FLEXSERV_JOB_DIR=$1
    select_arch
    local runtime=$2 rank=${SLURM_PROCID:?Missing Slurm rank}
    FCS_BIN="$FLEXSERV_JOB_DIR/bin/flexserv-capability-server-$ARCH.bin"
    local args=(--slurm-rank-config "$runtime/rank-launch.json")
    if [[ "$FLEXSERV_PREPARE_BACKENDS" == 1 ]]; then
        local policy="$runtime/config/host-$rank" node=${SLURMD_NODENAME:-$(hostname -f)} bind raw packs=null digest=
        if [[ ${FLEXSERV_PACK_MODE:-legacy} == multi ]]; then
            packs=$(jq -c '.packs' "$FLEXSERV_JOB_DIR/runtime/packs.json")
        else
            digest=$(file_sha256 "$FLEXSERV_JOB_DIR/runtime/pack/checksums.sha256")
        fi
        mkdir -p "$policy/work" "$PRI_MODEL_REPO"; chmod 700 "$policy"
        bind=${FLEXSERV_SERVICE_BIND:-}
        if [[ -z "$bind" ]]; then
            bind=$(node_ipv4_addresses "$node" | awk '$0 !~ /^127\./ && $0 !~ /^0\./ && $0 !~ /^(22[4-9]|23[0-9])\./ {print; exit}')
        fi
        [[ -n "$bind" ]] || { echo 'No service address; set FLEXSERV_SERVICE_BIND.' >&2; return 1; }
        raw=${FLEXSERV_GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-${ZE_AFFINITY_MASK:-}}}}
        [[ -n "$raw" ]] || { echo 'Set FLEXSERV_GPU_DEVICES to the Slurm-approved device IDs.' >&2; return 1; }
        valid_port "$FLEXSERV_SERVICE_PORT_START" && valid_port "$FLEXSERV_SERVICE_PORT_END" || return 1
        jq -n --arg pack "$FLEXSERV_JOB_DIR/runtime/pack" --arg output "$policy" \
            --arg principal "$MCP_PRINCIPAL" --arg root "$FLEXSERV_VENV_ROOT" \
            --arg venv "$FLEXSERV_VLLM_VENV_PATH" --arg private "$PRI_MODEL_REPO" --arg public "$PUB_MODEL_REPO" \
            --arg uv "$HOME/.local/bin/uv" --arg hardware "$HARDWARE_ENV" --arg backend "$FLEXSERV_ACCELERATOR_BACKEND" \
            --arg devices "$raw" --arg bind "$bind" --arg host "$node" \
            --arg digest "$digest" --argjson packs "$packs" \
            --argjson start "$FLEXSERV_SERVICE_PORT_START" --argjson end "$FLEXSERV_SERVICE_PORT_END" \
            --argjson credential "$([[ -n ${FCS_VLLM_API_KEY:-} ]] && echo true || echo false)" \
            -f "$FLEXSERV_JOB_DIR/scripts/host-request.jq" > "$policy/request.json"
        chmod 600 "$policy/request.json"
        "$FCS_BIN" --prepare-host-config "$policy/request.json" > "$policy/generated.json"
        args+=(--deploy-policy "$policy/deployment.json" --connections "$policy/connections.json" --workflow-policy "$policy/workflow.json")
    fi
    # FLEXSERV_TRUST_REMOTE_CODE is only inherited metadata in this iteration.
    unset FLEXSERV_ADMIN_SECRET FCS_MCP_TOKEN FLEXSERV_TOKEN RAND_TOKEN GATEWAY_VLLM_API_KEY PACK_TO_LOAD
    exec "$FCS_BIN" ${args[@]+"${args[@]}"}
}
wait_marker() {
    local marker=$1 end=$((SECONDS+STARTUP_TIMEOUT_SECONDS))
    while [[ ! -f "$marker" ]]; do
        kill -0 "$GATEWAY_PID" 2>/dev/null || { wait "$GATEWAY_PID"; return 1; }
        [[ ! -f "$FLEXSERV_RUNTIME/failed.json" ]] || { echo 'Gateway initialization failed; inspect logs.' >&2; return 1; }
        (( SECONDS < end )) || { echo "Startup timed out waiting for ${marker##*/}." >&2; return 1; }
        sleep 0.2
    done
}
prepare_job_config() {
    local job end_text end now node rank=0 addresses endpoints='{}' tools packs=null digest=
    job=$(TZ=UTC scontrol show job -o "$SLURM_JOB_ID")
    end_text=$(printf '%s\n' "$job" | tr ' ' '\n' | sed -n 's/^EndTime=//p')
    end=$(utc_epoch "$end_text") || return
    end=$((end-10)); now=$(date -u +%s)
    if [[ -n ${RUN_SECONDS:-} ]]; then
        [[ "$RUN_SECONDS" =~ ^[1-9][0-9]{0,6}$ ]] || { echo 'Invalid RUN_SECONDS.' >&2; return 1; }
        (( now+RUN_SECONDS >= end )) || end=$((now+RUN_SECONDS))
    fi
    (( end > now )) || { echo 'No job walltime remains.' >&2; return 1; }
    while IFS= read -r node; do
        [[ "$node" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo 'Invalid Slurm node name.' >&2; return 1; }
        if [[ -n ${FLEXSERV_MODEL_ID:-} ]]; then
            addresses=$(node_ipv4_addresses "$node" | jq -Rsc 'split("\n")|map(select(length>0))')
            [[ "$addresses" != '[]' ]] || { echo "No IPv4 address for $node" >&2; return 1; }
            endpoints=$(jq --arg id "fcs-$rank" --arg host "$node" --argjson addresses "$addresses" \
                --argjson start "$FLEXSERV_SERVICE_PORT_START" --argjson end "$FLEXSERV_SERVICE_PORT_END" \
                '.+{($id):{host:$host,addresses:$addresses,scheme:"http",port_start:$start,port_end:$end}}' <<< "$endpoints")
        fi
        rank=$((rank+1))
    done < "$FLEXSERV_RUNTIME/nodes.txt"
    (( rank > 0 )) && [[ $(LC_ALL=C sort -u "$FLEXSERV_RUNTIME/nodes.txt" | wc -l) -eq "$rank" ]] || {
        echo 'Empty or duplicate Slurm node list.' >&2; return 1;
    }
    if [[ ${FLEXSERV_PACK_MODE:-legacy} == multi ]]; then
        packs=$(jq -c '.packs' "$FLEXSERV_JOB_DIR/runtime/packs.json")
        tools=$(jq -c '.visible_tools + ["gateway.operations.status","gateway.operations.result","gateway.operations.cancel","gateway.operations.list"] | unique' "$FLEXSERV_JOB_DIR/runtime/packs.json")
    else
        # Legacy bundles expose identity lines from their checked extracted catalog.
        digest=$(file_sha256 "$FLEXSERV_JOB_DIR/runtime/pack/checksums.sha256")
        tools=$({
        while IFS= read -r -d '' node; do
            awk '/^  name: / {name=$2} /^  visibility: true$/ {visible=1} END {if(visible) print name}' "$node"
        done < <(find "$FLEXSERV_JOB_DIR/runtime/pack/fcs_tools" -name tool.yaml -type f -print0)
        printf '%s\n' gateway.operations.status gateway.operations.result gateway.operations.cancel gateway.operations.list
        } | LC_ALL=C sort -u | jq -Rsc 'split("\n")|map(select(length>0))')
    fi
    jq -n --arg job "$SLURM_JOB_ID" --arg run "$FLEXSERV_RUN_ID" --arg runtime "$FLEXSERV_RUNTIME" \
        --arg root "$FLEXSERV_JOB_DIR" --arg until "$(utc_timestamp "$end")" --arg principal "$MCP_PRINCIPAL" \
        --arg digest "$digest" --argjson packs "$packs" \
        --argjson timeout "$STARTUP_TIMEOUT_SECONDS" --argjson tools "$tools" --argjson endpoints "$endpoints" \
        --arg model "${FLEXSERV_MODEL_ID:-}" --argjson credential "$([[ -n ${FCS_VLLM_API_KEY:-} ]] && echo true || echo false)" \
        -f "$FLEXSERV_JOB_DIR/scripts/job-launch.jq" > "$FLEXSERV_RUNTIME/job-launch.json"
    chmod 600 "$FLEXSERV_RUNTIME/job-launch.json"
}
launch_flexserv() {
    export FLEXSERV_RUN_ID="run-$(secure_random_hex 16)"
    export FLEXSERV_RUNTIME="$FLEXSERV_JOB_DIR/runtime/$FLEXSERV_RUN_ID"
    mkdir -p "$FLEXSERV_RUNTIME"; chmod 700 "$FLEXSERV_RUNTIME"
    scontrol show hostnames "$SLURM_JOB_NODELIST" > "$FLEXSERV_RUNTIME/nodes.txt"
    chmod 600 "$FLEXSERV_RUNTIME/nodes.txt"
    prepare_job_config
    local bind=${GATEWAY_BIND_HOST:-0.0.0.0} args=() network origin
    for network in ${FLEXSERV_TRUSTED_PROXY_NETWORKS:-}; do args+=(--trusted-proxy-network "$network"); done
    for network in ${FLEXSERV_LOGIN_PUBLIC_HOSTS:-}; do args+=(--mcp-allowed-host "$network:${LOGIN_PORT:-$LOCAL_PORT}"); done
    for origin in ${FLEXSERV_MCP_ALLOWED_ORIGINS:-}; do args+=(--mcp-allowed-origin "$origin"); done
    if [[ "$ENABLE_HTTPS" == 1 ]]; then args+=(--cert "$TLS_CERT" --key "$TLS_KEY")
    else args+=(--allow-insecure-public-http); fi
    if [[ -n ${FCS_VLLM_API_KEY:-} ]]; then export GATEWAY_VLLM_API_KEY="$FCS_VLLM_API_KEY"; fi
    export FCS_MCP_TOKEN="$RAND_TOKEN" FLEXSERV_TOKEN="$RAND_TOKEN"
    export FLEXSERV_ADMIN_SECRET=${FLEXSERV_ADMIN_SECRET:-$(secure_random_hex)}
    mkdir -p "$PRI_MODEL_REPO"
    "$GATEWAY_BIN" --slurm-job-config "$FLEXSERV_RUNTIME/job-launch.json" \
        --host "$bind" --port "$LOCAL_PORT" --workers 2 \
        --mcp-allowed-host "localhost:$LOCAL_PORT" --mcp-allowed-host "127.0.0.1:$LOCAL_PORT" \
        --mcp-allowed-host "$NODE_HOSTNAME_DOMAIN:$LOCAL_PORT" \
        --mcp-allowed-host "${OSC_ONDEMAND_HOST:-ondemand.osc.edu}" \
        --mcp-allowed-host "${FLEXSERV_PUBLIC_HOST:-$NODE_HOSTNAME_DOMAIN}:${LOGIN_PORT:-$LOCAL_PORT}" \
        --log-file "$FLEXSERV_JOB_DIR/logs/gateway.log" --trace-log-file "$FLEXSERV_JOB_DIR/logs/trace.log" \
        --startup-lock-file "$FLEXSERV_RUNTIME/start.lock" --readiness-lock-file "$FLEXSERV_RUNTIME/ready.lock" \
        --runtime-config-dir "$FLEXSERV_RUNTIME/gateway" --fetch-state-dir "$FLEXSERV_RUNTIME/fetch" \
        --pub-model-repo "$PUB_MODEL_REPO" --pri-model-repo "$PRI_MODEL_REPO" \
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
    while kill -0 "$GATEWAY_PID" 2>/dev/null && kill -0 "$SRUN_PID" 2>/dev/null; do sleep 1; done
    echo 'A main service exited; shutting down this job run.' >&2
    if kill -0 "$GATEWAY_PID" 2>/dev/null; then wait "$SRUN_PID"; else wait "$GATEWAY_PID"; fi
}
cleanup_flexserv() {
    local pid end=$((SECONDS+5)) parent leaf
    for pid in ${TUNNEL_PIDS[@]+"${TUNNEL_PIDS[@]}"} "${SRUN_PID:-}" "${GATEWAY_PID:-}"; do
        [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in ${TUNNEL_PIDS[@]+"${TUNNEL_PIDS[@]}"} "${SRUN_PID:-}" "${GATEWAY_PID:-}"; do
        [[ -n "$pid" ]] || continue
        while kill -0 "$pid" 2>/dev/null && (( SECONDS < end )); do sleep 0.1; done
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    if [[ -n ${FLEXSERV_RUN_ID:-} ]]; then
        [[ "$FLEXSERV_RUN_ID" =~ ^run-[a-f0-9]{32}$ ]] || { echo 'Invalid owned run ID.' >&2; return 1; }
        parent="$FLEXSERV_JOB_DIR/tmp/tls"; leaf="$parent/$FLEXSERV_RUN_ID"
        [[ ! -L "$FLEXSERV_JOB_DIR/tmp" && ! -L "$parent" && ! -L "$leaf" ]] || {
            echo 'Refusing linked TLS cleanup path.' >&2; return 1;
        }
        rm -rf -- "$leaf"
        rmdir "$parent" 2>/dev/null || true
    fi
    concrete_cleanup_site_resources
}
