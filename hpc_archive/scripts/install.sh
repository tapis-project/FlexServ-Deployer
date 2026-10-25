#!/usr/bin/env bash
# Validate the bundled binaries/Pack and prepare missing userspace environments.
select_arch() {
    case "$(uname -m)" in
        x86_64|amd64|x64) ARCH=amd64 ;;
        aarch64|arm64) ARCH=arm64 ;;
        *) echo 'Only Linux x86_64 and aarch64 are supported.' >&2; return 1 ;;
    esac
    [[ $(uname -s) == Linux ]] || { echo 'Linux is required.' >&2; return 1; }
    export ARCH
}
verify_bundle() {
    python3 - "$FLEXSERV_JOB_DIR" <<'PY'
import hashlib, pathlib, sys, zipfile
root=pathlib.Path(sys.argv[1]).resolve()
for line in (root/'checksums.sha256').read_text().splitlines():
    digest,name=line.split('  ',1); p=root/name
    if pathlib.PurePosixPath(name).is_absolute() or '..' in pathlib.PurePosixPath(name).parts or p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
        raise SystemExit('Bundle checksum mismatch: '+name)
pack=root/'runtime/pack'
with zipfile.ZipFile(root/'pack/flexserv-capability-pack.zip') as z:
    seen=set()
    for info in z.infolist():
        name=pathlib.PurePosixPath(info.filename)
        if info.filename in seen or name.is_absolute() or '..' in name.parts or '\\' in info.filename or (info.external_attr>>16)&0o170000==0o120000:
            raise SystemExit('Unsafe Pack ZIP member')
        seen.add(info.filename)
    if len(z.infolist())>4096 or sum(i.file_size for i in z.infolist())>64*1024*1024: raise SystemExit('Pack exceeds 64 MiB')
    # Pack is immutable runtime input: never overwrite an existing extracted Pack.
    if pack.is_symlink(): raise SystemExit('Pack directory must not be a link')
    expected_inventory=z.read('checksums.sha256')
    if not pack.exists():
        pack.mkdir(parents=True,mode=0o700)
        z.extractall(pack)
if (pack/'checksums.sha256').read_bytes()!=expected_inventory:
    raise SystemExit('Existing extracted Pack differs from the bundled release')
expected_names={'checksums.sha256'}
for line in (pack/'checksums.sha256').read_text().splitlines():
    digest,name=line.split('  ',1); p=pack/name
    try: p.resolve().relative_to(pack.resolve())
    except ValueError: raise SystemExit('Pack path escapes root')
    expected_names.add(name)
    if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
        raise SystemExit('Pack checksum mismatch')
actual_names={p.relative_to(pack).as_posix() for p in pack.rglob('*') if p.is_file() or p.is_symlink()}
if actual_names!=expected_names: raise SystemExit('Pack file inventory mismatch')
PY
}
ensure_uv() {
    export PATH="$HOME/.local/bin:$PATH"
    UV_BIN="$HOME/.local/bin/uv"
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
    export UV_BIN
}
venv_ready() {
    local python=$1; shift
    [[ -x "$python" ]] || return 1
    "$python" - "$@" <<'PY'
import importlib, importlib.metadata as metadata, sys
if sys.version_info[:2] != (3,13): raise SystemExit(1)
for requirement in sys.argv[1:]:
    name, _, version=requirement.partition('==')
    if version and metadata.version(name)!=version: raise SystemExit(1)
    importlib.import_module(name)
PY
}
ensure_backend_environments() {
    mkdir -p "$FLEXSERV_VENV_ROOT" || { echo 'Set a writable FLEXSERV_VENV_ROOT.' >&2; return 1; }
    python3 - "$FLEXSERV_VENV_ROOT" <<'PY'
import os,pathlib,sys
p=pathlib.Path(sys.argv[1]); st=p.stat()
if p.resolve()!=p or st.st_uid!=os.geteuid() or st.st_mode&0o022:
    raise SystemExit('Set FLEXSERV_VENV_ROOT to a canonical directory owned by this account and not writable by other accounts; shared site roots are never chmod-ed automatically')
PY
    command -v flock >/dev/null || { echo 'flock is required for shared environment installation.' >&2; return 1; }
    # One per-profile lock covers Python and both environments across concurrent jobs.
    exec 9>"$FLEXSERV_VENV_ROOT/.install-$HARDWARE_ENV.lock"
    flock 9
    if [[ -z "${FLEXSERV_ACCELERATOR_BACKEND:-}" ]]; then
        FLEXSERV_ACCELERATOR_BACKEND=$("$FCS_BIN" --internal-host-probe | python3 -c '
import json,sys
facts=json.load(sys.stdin); names=" ".join(str(g.get("device_type",""))+" "+str(g.get("name","")) for g in facts.get("gpus",[])).lower()
backend="cuda" if "nvidia" in names else "rocm" if "amd" in names else "xpu" if "intel" in names else ""
if not backend: raise SystemExit("Cannot detect accelerator; set FLEXSERV_ACCELERATOR_BACKEND explicitly")
print(backend)')
        export FLEXSERV_ACCELERATOR_BACKEND
    fi
    ensure_uv
    export UV_PYTHON_INSTALL_DIR="$FLEXSERV_VENV_PYTHON_INSTALL_DIR"
    "$UV_BIN" python install 3.13
    local vllm="vllm${VLLM_VERSION:+==$VLLM_VERSION}" aiperf="aiperf${AIPERF_VERSION:+==$AIPERF_VERSION}"
    if ! venv_ready "$FLEXSERV_VLLM_VENV_PATH/bin/python" "$vllm" "$aiperf" || ! "$UV_BIN" pip check --python "$FLEXSERV_VLLM_VENV_PATH/bin/python"; then
        if [[ ! -x "$FLEXSERV_VLLM_VENV_PATH/bin/python" ]]; then "$UV_BIN" venv --python 3.13 "$FLEXSERV_VLLM_VENV_PATH"; fi
        local index_args=()
        if [[ -n "${FLEXSERV_VLLM_INDEX:-}" ]]; then index_args=(--index "$FLEXSERV_VLLM_INDEX"); fi
        # XPU/ROCm package sources must be approved explicitly rather than using CUDA defaults.
        case "${FLEXSERV_ACCELERATOR_BACKEND:-}" in
            xpu|rocm) [[ ${#index_args[@]} != 0 ]] || { echo 'Set FLEXSERV_VLLM_INDEX for XPU/ROCm.' >&2; return 1; } ;;
        esac
        "$UV_BIN" pip install --only-binary=:all: --python "$FLEXSERV_VLLM_VENV_PATH/bin/python" ${index_args[@]+"${index_args[@]}"} "$vllm" "$aiperf"
        venv_ready "$FLEXSERV_VLLM_VENV_PATH/bin/python" "$vllm" "$aiperf"
        "$UV_BIN" pip check --python "$FLEXSERV_VLLM_VENV_PATH/bin/python"
    fi
    FLEXSERV_ACCELERATOR_BACKEND=$("$FLEXSERV_VLLM_VENV_PATH/bin/python" -c 'import torch; print("rocm" if torch.version.hip else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")')
    export FLEXSERV_ACCELERATOR_BACKEND
    case "$FLEXSERV_ACCELERATOR_BACKEND" in cuda|xpu|rocm) ;; *) echo 'No supported accelerator runtime is available; use --control-plane-only for control-plane validation.' >&2; return 1 ;; esac
    ensure_max
    "$FLEXSERV_VLLM_VENV_PATH/bin/python" -c 'import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ("vllm","aiperf")}))' > "$FLEXSERV_JOB_DIR/runtime/environment-versions.json"
    flock -u 9; exec 9>&-
}
max_supported() {
    # Wheel resolution validates the selected stable version against this Python/ABI.
    [[ ${FLEXSERV_ACCELERATOR_BACKEND:-} != xpu ]] && command -v cc >/dev/null || return 1
    python3 - <<'PYTHON'
import platform
if platform.system() != 'Linux': raise SystemExit(1)
libc,version=platform.libc_ver()
if libc != 'glibc' or tuple(map(int,version.split('.')[:2])) < (2,35): raise SystemExit(1)
if platform.machine()=='x86_64':
    flags=open('/proc/cpuinfo').read().split('flags',1)[-1].split('\n',1)[0].split()
    if not {'avx','avx2','bmi1','bmi2','f16c','fma','lzcnt','movbe','xsave'} <= set(flags): raise SystemExit(1)
PYTHON
}
ensure_max() {
    case "$FLEXSERV_INSTALL_MAX" in
        0|false|off) echo 'MAX installation disabled.'; return ;;
        auto|1|true|on) ;;
        *) echo 'FLEXSERV_INSTALL_MAX must be auto, 0 or 1.' >&2; return 1 ;;
    esac
    if ! max_supported; then
        [[ "$FLEXSERV_INSTALL_MAX" == auto ]] || { echo 'MAX is required but unsupported on this node.' >&2; return 1; }
        echo 'MAX skipped: unsupported backend or missing runtime linker.'; return
    fi
    local requirement="max${MAX_VERSION:+==$MAX_VERSION}"
    if ! venv_ready "$FLEXSERV_MAX_VENV_PATH/bin/python" "$requirement" ||
       ! "$UV_BIN" pip check --python "$FLEXSERV_MAX_VENV_PATH/bin/python" ||
       ! grep -Fq 'max[all]' "$FLEXSERV_MAX_VENV_ROOT/pyproject.toml" 2>/dev/null ||
       [[ ! -x "$FLEXSERV_MAX_VENV_PATH/bin/max" ]]; then
        if [[ ! -f "$FLEXSERV_MAX_VENV_ROOT/pyproject.toml" ]]; then "$UV_BIN" init --bare --python 3.13 "$FLEXSERV_MAX_VENV_ROOT"; fi
        if [[ ! -x "$FLEXSERV_MAX_VENV_PATH/bin/python" ]]; then "$UV_BIN" venv --python 3.13 "$FLEXSERV_MAX_VENV_PATH"; fi
        (cd "$FLEXSERV_MAX_VENV_ROOT" && "$UV_BIN" add --no-build "max[all]${MAX_VERSION:+==$MAX_VERSION}")
        venv_ready "$FLEXSERV_MAX_VENV_PATH/bin/python" "$requirement"
        "$UV_BIN" pip check --python "$FLEXSERV_MAX_VENV_PATH/bin/python"
        "$FLEXSERV_MAX_VENV_PATH/bin/max" --help >/dev/null
    fi
}
install_flexserv() {
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,8) else "Python 3.8+ is required for the launcher")'
    mkdir -p "$FLEXSERV_JOB_DIR/tmp" "$FLEXSERV_JOB_DIR/runtime" "$FLEXSERV_JOB_DIR/logs" "$FLEXSERV_JOB_DIR/reports"
    select_arch; verify_bundle
    GATEWAY_BIN="$FLEXSERV_JOB_DIR/bin/flexserv-gateway-$ARCH.bin"
    FCS_BIN="$FLEXSERV_JOB_DIR/bin/flexserv-capability-server-$ARCH.bin"
    chmod u+x "$GATEWAY_BIN" "$FCS_BIN"
    "$GATEWAY_BIN" --version; "$FCS_BIN" --export-native-contracts >/dev/null
    export GATEWAY_BIN FCS_BIN
    if [[ "$FLEXSERV_PREPARE_BACKENDS" == 1 ]]; then ensure_backend_environments; fi
}
