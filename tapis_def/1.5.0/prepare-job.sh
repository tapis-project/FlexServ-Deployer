#!/usr/bin/env bash
# Produce a job request for an existing TACC execution system; submit nothing.
set -euo pipefail
command -v jq >/dev/null || { echo 'jq is required.' >&2; exit 1; }
: "${TAPIS_EXEC_SYSTEM_ID:?Set your existing TACC execution system ID}"
: "${TAPIS_LOGICAL_QUEUE:?Set a queue on that system}"
: "${TAPIS_ALLOCATION:?Set your TACC allocation}"
[[ "$TAPIS_ALLOCATION" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid TACC allocation.' >&2; exit 2; }
for flag in "${TAPIS_ENABLE_HTTPS:-0}" "${TAPIS_TRUST_REMOTE_CODE:-0}" "${TAPIS_CONTROL_PLANE_ONLY:-1}"; do
    [[ "$flag" == 0 || "$flag" == 1 ]] || { echo 'Job flags must be 0 or 1.' >&2; exit 2; }
done
jq -n --arg name "${TAPIS_JOB_NAME:-flexserv-1.5.0-check}" \
    --arg system "$TAPIS_EXEC_SYSTEM_ID" --arg queue "$TAPIS_LOGICAL_QUEUE" --arg allocation "$TAPIS_ALLOCATION" \
    --arg private "${PRI_MODEL_REPO:-}" --arg public "${PUB_MODEL_REPO:-}" \
    --argjson https "${TAPIS_ENABLE_HTTPS:-0}" --argjson trust "${TAPIS_TRUST_REMOTE_CODE:-0}" \
    --argjson control "${TAPIS_CONTROL_PLANE_ONLY:-1}" \
    '{name:$name,appId:"FlexServ-1.5.0",appVersion:"1.5.0",execSystemId:$system,
      execSystemLogicalQueue:$queue,maxMinutes:30,parameterSet:{
        appArgs:([{name:"enableHttps",arg:"--enable-https",include:($https==1)},
          {name:"trustRemoteCode",arg:"--trust-remote-code",include:($trust==1)},
          {name:"controlPlaneOnly",arg:"--control-plane-only",include:($control==1)}]),
        schedulerOptions:[{name:"allocation",arg:("-A "+$allocation)}],
        envVariables:([{key:"PRI_MODEL_REPO",value:$private},{key:"PUB_MODEL_REPO",value:$public}]|map(select(.value!="")))}}'
