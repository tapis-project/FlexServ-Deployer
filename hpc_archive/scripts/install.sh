#!/usr/bin/env bash
# Verify release bytes and prepare userspace runtimes without interpreter glue.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/common.sh"

select_arch() {
    case "$(uname -m)" in
        x86_64|amd64|x64) ARCH=amd64 ;;
        aarch64|arm64) ARCH=arm64 ;;
        *) echo 'Only Linux x86_64 and aarch64 are supported.' >&2; return 1 ;;
    esac
    [[ $(uname -s) == Linux ]] || { echo 'Linux is required.' >&2; return 1; }
    export ARCH
}
check_pack_tree() {
    local pack=$1 inventory=$2 scratch=$3 path
    [[ -d "$pack" && ! -L "$pack" ]] && cmp -s "$pack/checksums.sha256" "$inventory" || {
        echo 'Existing extracted Pack differs from the bundled release.' >&2; return 1;
    }
    [[ -z $(find "$pack" -type l -print -quit) ]] || { echo 'Linked Pack entry.' >&2; return 1; }
    verify_inventory "$pack" "$inventory" || return
    { awk '{print substr($0,67)}' "$inventory"; echo checksums.sha256; } | LC_ALL=C sort > "$scratch/expected"
    while IFS= read -r -d '' path; do printf '%s\n' "${path#"$pack/"}"; done \
        < <(find "$pack" -type f -print0) | LC_ALL=C sort > "$scratch/actual"
    cmp -s "$scratch/actual" "$scratch/expected" || { echo 'Pack file inventory mismatch.' >&2; return 1; }
}
prepare_selected_packs() (
    # The FCS snapshots and validates every ZIP before publishing a manifest.
    set -e
    local root=$FLEXSERV_JOB_DIR tag selection cache="$FLEXSERV_JOB_DIR/runtime/packs"
    tag=$(cat "$root/pack-tag.txt")
    selection=${PACK_TO_LOAD-'tapis-workflow-pack.zip;vllm-pack.zip'}
    no_link_path "$root" runtime && no_link_path "$root" runtime/packs || {
        echo 'Pack cache cannot be a symbolic link.' >&2; exit 1;
    }
    mkdir -p "$cache"; chmod 700 "$cache"
    FLEXSERV_PACK_SELECTION_SCRATCH=$(mktemp -d "$root/tmp/pack-select.XXXXXX")
    trap 'rm -rf -- "$FLEXSERV_PACK_SELECTION_SCRATCH"' EXIT
    jq -n --arg tag "$tag" --arg selection "$selection" --arg cache "$cache" '
        ($selection|split(";")) as $items |
        if (($items|length)==0 or ($items|length)>16 or any($items[]; .=="" or test("[\\r\\n]")))
        then error("Invalid PACK_TO_LOAD selection")
        else {release_tag:$tag,selections:$items,output_dir:$cache} end' > "$FLEXSERV_PACK_SELECTION_SCRATCH/request.json"
    "$FCS_BIN" --prepare-packs "$FLEXSERV_PACK_SELECTION_SCRATCH/request.json" > "$FLEXSERV_PACK_SELECTION_SCRATCH/packs.json"
    jq -e --arg prefix "$cache/" --arg tag "$tag" '
        .release_tag==$tag and (.packs|type)=="array" and (.packs|length)>0 and (.packs|length)<=16 and
        all(.packs[]; (.name|type)=="string" and (.name|length)>0 and
            (.source=="github" or .source=="local") and (.path|type)=="string" and (.path|startswith($prefix)) and
            (.checksums_sha256|type)=="string" and (.checksums_sha256|test("^[a-f0-9]{64}$")) and
            (.zip_sha256|type)=="string" and (.zip_sha256|test("^[a-f0-9]{64}$"))) and
        (.visible_tools|type)=="array" and all(.visible_tools[]; type=="string")' "$FLEXSERV_PACK_SELECTION_SCRATCH/packs.json" >/dev/null || {
        echo 'Invalid prepared Pack manifest.' >&2; exit 1;
    }
    chmod 600 "$FLEXSERV_PACK_SELECTION_SCRATCH/packs.json"
    mv "$FLEXSERV_PACK_SELECTION_SCRATCH/packs.json" "$root/runtime/packs.json"
)
verify_bundle() (
    # Legacy bundles retain their extracted Pack; new bundles pin a release tag.
    set -e
    local zip="$FLEXSERV_JOB_DIR/pack/flexserv-capability-pack.zip" pack="$FLEXSERV_JOB_DIR/runtime/pack" name
    require_commands sha256sum unzip zipinfo cmp find sort awk
    verify_inventory "$FLEXSERV_JOB_DIR" "$FLEXSERV_JOB_DIR/checksums.sha256"
    mkdir -p "$FLEXSERV_JOB_DIR/tmp" "$FLEXSERV_JOB_DIR/runtime"
    if awk 'substr($0,67)=="pack-tag.txt" {found=1} END {exit !found}' "$FLEXSERV_JOB_DIR/checksums.sha256"; then
        [[ -n ${FCS_BIN:-} ]] || { echo 'Capability Server is required to prepare Packs.' >&2; exit 1; }
        prepare_selected_packs
        exit
    fi
    awk 'substr($0,67)=="pack/flexserv-capability-pack.zip" {found=1} END {exit !found}' "$FLEXSERV_JOB_DIR/checksums.sha256" || {
        echo 'Bundle has no verified Pack source.' >&2; exit 1;
    }
    FLEXSERV_PACK_CHECK_SCRATCH=$(mktemp -d "$FLEXSERV_JOB_DIR/tmp/pack-check.XXXXXX")
    trap 'rm -rf -- "$FLEXSERV_PACK_CHECK_SCRATCH"' EXIT
    zipinfo -1 "$zip" > "$FLEXSERV_PACK_CHECK_SCRATCH/members"
    [[ -z $(LC_ALL=C sort "$FLEXSERV_PACK_CHECK_SCRATCH/members" | uniq -d) ]] || { echo 'Duplicate Pack ZIP member.' >&2; exit 1; }
    while IFS= read -r name; do
        safe_relative_path "${name%/}" || { echo 'Unsafe Pack ZIP member.' >&2; exit 1; }
    done < "$FLEXSERV_PACK_CHECK_SCRATCH/members"
    zipinfo -l "$zip" | awk '$1 ~ /^l/ {bad=1} END {exit bad}' || { echo 'Linked Pack ZIP member.' >&2; exit 1; }
    unzip -l "$zip" | awk '$2 ~ /-/ && $3 ~ /:/ {count++; bytes+=$1} END {exit(count==0 || count>4096 || bytes>67108864)}' || {
        echo 'Pack exceeds its file or byte limit.' >&2; exit 1;
    }
    unzip -p "$zip" checksums.sha256 > "$FLEXSERV_PACK_CHECK_SCRATCH/inventory"
    if [[ ! -e "$pack" && ! -L "$pack" ]]; then
        mkdir "$FLEXSERV_PACK_CHECK_SCRATCH/pack"
        unzip -q "$zip" -d "$FLEXSERV_PACK_CHECK_SCRATCH/pack"
        check_pack_tree "$FLEXSERV_PACK_CHECK_SCRATCH/pack" "$FLEXSERV_PACK_CHECK_SCRATCH/inventory" "$FLEXSERV_PACK_CHECK_SCRATCH"
        mv "$FLEXSERV_PACK_CHECK_SCRATCH/pack" "$pack"
    else
        check_pack_tree "$pack" "$FLEXSERV_PACK_CHECK_SCRATCH/inventory" "$FLEXSERV_PACK_CHECK_SCRATCH"
    fi
)
ensure_uv() {
    export PATH="$HOME/.local/bin:$PATH"
    export UV_BIN="$HOME/.local/bin/uv"
    if [[ ! -x "$UV_BIN" ]]; then
        local installer
        installer=$(mktemp "$FLEXSERV_JOB_DIR/tmp/uv-install.XXXXXX")
        if command -v curl >/dev/null; then curl -fLsS https://astral.sh/uv/install.sh -o "$installer"
        elif command -v wget >/dev/null; then wget -qO "$installer" https://astral.sh/uv/install.sh
        else echo 'curl or wget is required to install uv.' >&2; return 1; fi
        UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh "$installer"
        rm -f -- "$installer"
    fi
    "$UV_BIN" --version
}
venv_ready() {
    local executable=$1 packages requirement name version
    shift
    [[ -x "$executable" && $("$executable" --version 2>&1) == 'Python 3.13.'* ]] || return 1
    packages=$("$UV_BIN" pip list --python "$executable" --format json) || return 1
    for requirement in "$@"; do
        name=${requirement%%==*}; version=
        if [[ "$requirement" == *==* ]]; then version=${requirement#*==}; fi
        jq -e --arg name "$name" --arg version "$version" \
            'any(.[]; .name==$name and ($version=="" or .version==$version))' <<< "$packages" >/dev/null || return 1
    done
}
inspect_accelerator_backend() {
    "$FCS_BIN" --internal-host-probe | jq -er '
        [.gpus[]? | ((.device_type//"")+" "+(.name//"")|ascii_downcase) |
          if test("nvidia|cuda") then "cuda" elif test("amd|radeon|rocm") then "rocm"
          elif test("intel|xpu") then "xpu" else empty end] | unique |
        if length==1 then .[0] else error("Set FLEXSERV_ACCELERATOR_BACKEND explicitly") end'
}
ensure_backend_environments() {
    require_commands flock timeout stat
    mkdir -p "$FLEXSERV_VENV_ROOT" || { echo 'Set a writable FLEXSERV_VENV_ROOT.' >&2; return 1; }
    local identity owner mode canonical
    identity=$(stat -c '%u %a' "$FLEXSERV_VENV_ROOT" 2>/dev/null || stat -f '%u %Lp' "$FLEXSERV_VENV_ROOT")
    read -r owner mode <<< "$identity"
    canonical=$(cd -- "$FLEXSERV_VENV_ROOT" && pwd -P)
    [[ "$canonical" == "$FLEXSERV_VENV_ROOT" && "$owner" == "$(id -u)" ]] && (( (8#$mode & 8#022)==0 )) || {
        echo 'Use a canonical VENV root owned by this account and not writable by other accounts.' >&2; return 1;
    }
    exec 9>"$FLEXSERV_VENV_ROOT/.install-$HARDWARE_ENV.lock"; flock 9
    export FLEXSERV_ACCELERATOR_BACKEND=${FLEXSERV_ACCELERATOR_BACKEND:-$(inspect_accelerator_backend)}
    case "$FLEXSERV_ACCELERATOR_BACKEND" in cuda|xpu|rocm) ;; *) echo 'No supported accelerator backend.' >&2; return 1 ;; esac
    ensure_uv
    export UV_PYTHON_INSTALL_DIR="$FLEXSERV_VENV_PYTHON_INSTALL_DIR"
    "$UV_BIN" python install 3.13
    local vllm="vllm${VLLM_VERSION:+==$VLLM_VERSION}" aiperf="aiperf${AIPERF_VERSION:+==$AIPERF_VERSION}"
    if ! venv_ready "$FLEXSERV_VLLM_VENV_PATH/bin/python" "$vllm" "$aiperf" ||
       ! "$UV_BIN" pip check --python "$FLEXSERV_VLLM_VENV_PATH/bin/python"; then
        if [[ ! -x "$FLEXSERV_VLLM_VENV_PATH/bin/python" ]]; then "$UV_BIN" venv --python 3.13 "$FLEXSERV_VLLM_VENV_PATH"; fi
        local index_args=()
        if [[ -n ${FLEXSERV_VLLM_INDEX:-} ]]; then index_args=(--index "$FLEXSERV_VLLM_INDEX"); fi
        case "$FLEXSERV_ACCELERATOR_BACKEND" in
            xpu|rocm) [[ ${#index_args[@]} != 0 ]] || { echo 'Set FLEXSERV_VLLM_INDEX for XPU/ROCm.' >&2; return 1; } ;;
        esac
        "$UV_BIN" pip install --only-binary=:all: --python "$FLEXSERV_VLLM_VENV_PATH/bin/python" \
            ${index_args[@]+"${index_args[@]}"} "$vllm" "$aiperf"
        venv_ready "$FLEXSERV_VLLM_VENV_PATH/bin/python" "$vllm" "$aiperf"
        "$UV_BIN" pip check --python "$FLEXSERV_VLLM_VENV_PATH/bin/python"
    fi
    # Package CLIs perform their normal imports; detailed runtime checks stay in FCS.
    timeout 60 "$FLEXSERV_VLLM_VENV_PATH/bin/vllm" --version
    timeout 60 "$FLEXSERV_VLLM_VENV_PATH/bin/aiperf" --version
    ensure_max
    local packages python_version max_version=
    packages=$("$UV_BIN" pip list --python "$FLEXSERV_VLLM_VENV_PATH/bin/python" --format json)
    python_version=$("$FLEXSERV_VLLM_VENV_PATH/bin/python" --version)
    if [[ "$FLEXSERV_MAX_STATUS" == ready ]]; then
        max_version=$("$UV_BIN" pip list --python "$FLEXSERV_MAX_VENV_PATH/bin/python" --format json | jq -r '.[]|select(.name=="max")|.version')
    fi
    jq -n --arg python "$python_version" --arg uv "$("$UV_BIN" --version)" --argjson packages "$packages" \
        --arg backend "$FLEXSERV_ACCELERATOR_BACKEND" --arg status "$FLEXSERV_MAX_STATUS" --arg max "$max_version" \
        '{python:$python,uv:$uv,accelerator_backend:$backend,max_status:$status,
          vllm:($packages|map(select(.name=="vllm"))[0].version),aiperf:($packages|map(select(.name=="aiperf"))[0].version)}
          + if $status=="ready" then {max:$max} else {} end' > "$FLEXSERV_JOB_DIR/runtime/environment-versions.json"
    flock -u 9; exec 9>&-
}
max_supported() {
    [[ ${FLEXSERV_ACCELERATOR_BACKEND:-} != xpu && $(uname -s) == Linux ]] && command -v cc >/dev/null || return 1
    local libc major minor flags feature
    libc=$(getconf GNU_LIBC_VERSION) || return 1
    libc=${libc##* }; major=${libc%%.*}; minor=${libc#*.}; minor=${minor%%.*}
    [[ "$major.$minor" =~ ^[0-9]+\.[0-9]+$ ]] && (( major>2 || (major==2 && minor>=35) )) || return 1
    if [[ $(uname -m) == x86_64 ]]; then
        flags=" $(awk -F: '/^flags/ {print $2; exit}' /proc/cpuinfo) "
        for feature in avx avx2 bmi1 bmi2 f16c fma movbe xsave; do [[ "$flags" == *" $feature "* ]] || return 1; done
        [[ "$flags" == *' abm '* || "$flags" == *' lzcnt '* ]] || return 1
    fi
}
ensure_max() {
    export FLEXSERV_MAX_STATUS=not_checked
    case "$FLEXSERV_INSTALL_MAX" in
        0|false|off) export FLEXSERV_MAX_STATUS=disabled; echo 'MAX installation disabled.'; return ;;
        auto|1|true|on) ;;
        *) echo 'FLEXSERV_INSTALL_MAX must be auto, 0 or 1.' >&2; return 1 ;;
    esac
    if ! max_supported; then
        [[ "$FLEXSERV_INSTALL_MAX" == auto ]] || { echo 'MAX is required but unsupported on this node.' >&2; return 1; }
        export FLEXSERV_MAX_STATUS=unsupported
        echo 'MAX skipped: unsupported platform/backend or missing runtime linker.'; return
    fi
    local requirement="max${MAX_VERSION:+==$MAX_VERSION}"
    if ! venv_ready "$FLEXSERV_MAX_VENV_PATH/bin/python" "$requirement" ||
       ! "$UV_BIN" pip check --python "$FLEXSERV_MAX_VENV_PATH/bin/python" ||
       ! grep -Fq 'max[all]' "$FLEXSERV_MAX_VENV_ROOT/pyproject.toml" 2>/dev/null ||
       [[ ! -x "$FLEXSERV_MAX_VENV_PATH/bin/max" ]]; then
        if [[ ! -f "$FLEXSERV_MAX_VENV_ROOT/pyproject.toml" ]]; then "$UV_BIN" init --bare --python 3.13 "$FLEXSERV_MAX_VENV_ROOT"; fi
        if [[ ! -x "$FLEXSERV_MAX_VENV_PATH/bin/python" ]]; then "$UV_BIN" venv --python 3.13 "$FLEXSERV_MAX_VENV_PATH"; fi
        (cd "$FLEXSERV_MAX_VENV_ROOT" && UV_PROJECT_ENVIRONMENT="$FLEXSERV_MAX_VENV_PATH" "$UV_BIN" add --no-build "max[all]${MAX_VERSION:+==$MAX_VERSION}")
        venv_ready "$FLEXSERV_MAX_VENV_PATH/bin/python" "$requirement"
        "$UV_BIN" pip check --python "$FLEXSERV_MAX_VENV_PATH/bin/python"
    fi
    timeout 60 "$FLEXSERV_MAX_VENV_PATH/bin/max" --help >/dev/null
    export FLEXSERV_MAX_STATUS=ready
}
install_flexserv() {
    require_commands jq od sha256sum unzip zipinfo getent date
    valid_port "$LOCAL_PORT" || { echo 'Invalid LOCAL_PORT.' >&2; return 1; }
    [[ "$STARTUP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]{0,4}$ ]] || { echo 'Invalid startup timeout.' >&2; return 1; }
    mkdir -p "$FLEXSERV_JOB_DIR/tmp" "$FLEXSERV_JOB_DIR/runtime" "$FLEXSERV_JOB_DIR/logs" "$FLEXSERV_JOB_DIR/reports"
    mkdir -p "$PRI_MODEL_REPO"
    [[ -d "$PUB_MODEL_REPO" ]] || mkdir -p "$PUB_MODEL_REPO"
    select_arch
    # Verify the FCS executable before using it to validate external Pack ZIPs.
    verify_inventory "$FLEXSERV_JOB_DIR" "$FLEXSERV_JOB_DIR/checksums.sha256"
    GATEWAY_BIN="$FLEXSERV_JOB_DIR/bin/flexserv-gateway-$ARCH.bin"
    FCS_BIN="$FLEXSERV_JOB_DIR/bin/flexserv-capability-server-$ARCH.bin"
    awk -v gateway="bin/flexserv-gateway-$ARCH.bin" -v fcs="bin/flexserv-capability-server-$ARCH.bin" '
        substr($0,67)==gateway {a=1} substr($0,67)==fcs {b=1} END {exit !(a && b)}' "$FLEXSERV_JOB_DIR/checksums.sha256" || {
        echo 'Bundle inventory must cover both selected executables.' >&2; return 1;
    }
    chmod u+x "$GATEWAY_BIN" "$FCS_BIN"
    verify_bundle
    if awk 'substr($0,67)=="pack-tag.txt" {found=1} END {exit !found}' "$FLEXSERV_JOB_DIR/checksums.sha256"; then
        export FLEXSERV_PACK_MODE=multi
    else
        export FLEXSERV_PACK_MODE=legacy
    fi
    "$GATEWAY_BIN" --version; "$FCS_BIN" --export-native-contracts >/dev/null
    export GATEWAY_BIN FCS_BIN
    if [[ "$FLEXSERV_PREPARE_BACKENDS" == 1 ]]; then ensure_backend_environments; fi
}
