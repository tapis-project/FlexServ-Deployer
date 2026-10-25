#!/usr/bin/env bash
# Reuse the same extracted Tapis bundle inside an interactive allocation.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
    echo 'Set FLEXSERV_BUNDLE_DIR to an extracted Tapis-FlexServ.zip directory.'
    echo 'Usage: launch.sh [--control-plane-only]'
    exit 0
fi
root=${FLEXSERV_BUNDLE_DIR:?Set FLEXSERV_BUNDLE_DIR to the extracted Alpha bundle}
[[ -r "$root/app/run_flexserv.sh" ]] || { echo 'Not an extracted Alpha bundle.' >&2; exit 2; }
exec bash "$root/app/run_flexserv.sh" "$@"
