#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-unit}"

echo "[flexserv-deployer] Running tests (mode=${MODE})"

case "${MODE}" in
  unit)
    echo "-> cargo test --lib"
    cargo test --lib
    ;;

  integration)
    echo "-> cargo test --tests -- --nocapture"
    echo "   (integration tests will SKIP themselves if TAPIS_TENANT_URL/TAPIS_TOKEN etc. are not set)"
    cargo test --tests -- --nocapture
    ;;

  all)
    echo "-> cargo test -- --nocapture"
    echo "   (integration tests will SKIP themselves if required env vars are missing)"
    cargo test -- --nocapture
    ;;

  create)
    if [ "$#" -lt 3 ]; then
      echo "Usage: $0 create <TAPIS_TENANT_URL> <TAPIS_TOKEN>" >&2
      exit 1
    fi
    export TAPIS_TENANT_URL="$2"
    export TAPIS_TOKEN="$3"
    echo "-> cargo test --test pod_create_integration -- --nocapture"
    cargo test --test pod_create_integration -- --nocapture
    ;;

  start)
    if [ "$#" -lt 5 ]; then
      echo "Usage: $0 start <TAPIS_TENANT_URL> <TAPIS_TOKEN> <POD_ID> <VOLUME_ID>" >&2
      exit 1
    fi
    export TAPIS_TENANT_URL="$2"
    export TAPIS_TOKEN="$3"
    export POD_ID="$4"
    export VOLUME_ID="$5"
    echo "-> cargo test --test pod_start_integration -- --nocapture"
    cargo test --test pod_start_integration -- --nocapture
    ;;

  stop)
    if [ "$#" -lt 5 ]; then
      echo "Usage: $0 stop <TAPIS_TENANT_URL> <TAPIS_TOKEN> <POD_ID> <VOLUME_ID>" >&2
      exit 1
    fi
    export TAPIS_TENANT_URL="$2"
    export TAPIS_TOKEN="$3"
    export POD_ID="$4"
    export VOLUME_ID="$5"
    echo "-> cargo test --test pod_stop_integration -- --nocapture"
    cargo test --test pod_stop_integration -- --nocapture
    ;;

  monitor)
    if [ "$#" -lt 4 ]; then
      echo "Usage: $0 monitor <TAPIS_TENANT_URL> <TAPIS_TOKEN> <POD_ID> [VOLUME_ID]" >&2
      exit 1
    fi
    export TAPIS_TENANT_URL="$2"
    export TAPIS_TOKEN="$3"
    export POD_ID="$4"
    export VOLUME_ID="${5-}"
    echo "-> cargo test --test pod_monitor_integration -- --nocapture"
    cargo test --test pod_monitor_integration -- --nocapture
    ;;

  terminate)
    if [ "$#" -lt 5 ]; then
      echo "Usage: $0 terminate <TAPIS_TENANT_URL> <TAPIS_TOKEN> <POD_ID> <VOLUME_ID>" >&2
      exit 1
    fi
    export TAPIS_TENANT_URL="$2"
    export TAPIS_TOKEN="$3"
    export POD_ID="$4"
    export VOLUME_ID="$5"
    echo "-> cargo test --test pod_terminate_integration -- --nocapture"
    cargo test --test pod_terminate_integration -- --nocapture
    ;;

  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: $0 [unit|integration|all|create|start|stop|monitor|terminate]" >&2
    echo "  create    <TAPIS_TENANT_URL> <TAPIS_TOKEN>" >&2
    echo "  start     <TAPIS_TENANT_URL> <TAPIS_TOKEN> <POD_ID> <VOLUME_ID>" >&2
    echo "  stop      <TAPIS_TENANT_URL> <TAPIS_TOKEN> <POD_ID> <VOLUME_ID>" >&2
    echo "  monitor   <TAPIS_TENANT_URL> <TAPIS_TOKEN> <POD_ID> [VOLUME_ID]" >&2
    echo "  terminate <TAPIS_TENANT_URL> <TAPIS_TOKEN> <POD_ID> <VOLUME_ID>" >&2
    exit 1
    ;;
esac

echo "[flexserv-deployer] Tests completed (mode=${MODE})"

