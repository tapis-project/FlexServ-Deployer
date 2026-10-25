#!/usr/bin/env python3
"""Run the actual Tapis Bash entry and real control servers with fake Slurm only.

This optional local integration test is not a release ABI test. On macOS it
labels native test executables with the bundle's Linux suffixes and overrides
uname solely in its private PATH. No production test mode, package install,
model download, real allocation or external resource is used.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import aiohttp
import yaml

ROOT = Path(__file__).resolve().parents[2]
FCS = ROOT.parent / 'FlexServ-FCS'
sys.path.insert(0, str(FCS / 'python'))
from client import connect, discover_tools

TOKEN = 'alpha-script-fixture-access-0123456789abcdef'
ADMIN = 'alpha-script-fixture-admin-0123456789abcdef'


def digest(path):
    """Hash executable and bundle evidence incrementally."""
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for data in iter(lambda: source.read(1024 * 1024), b''):
            result.update(data)
    return result.hexdigest()


def port():
    """Choose only a fixture candidate; Gateway still has to bind it successfully."""
    with socket.socket() as stream:
        stream.bind(('127.0.0.1', 0))
        return stream.getsockname()[1]


def executable(path, text):
    """Create a private PATH fixture, never replace host scheduler commands."""
    path.write_text(text)
    path.chmod(0o700)


def assemble_fixture(root, args):
    """Assemble actual scripts and native bytes, explicitly labeling the ABI substitution."""
    bundle = root / 'job'
    shutil.copytree(ROOT / 'hpc_archive', bundle)
    (bundle / 'bin').mkdir()
    (bundle / 'pack').mkdir()
    for arch in ('amd64', 'arm64'):
        for component, source in [('flexserv-gateway', args.gateway), ('flexserv-capability-server', args.server)]:
            shutil.copyfile(source, bundle / f'bin/{component}-{arch}.bin')
    shutil.copyfile(args.pack, bundle / 'pack/flexserv-capability-pack.zip')
    metadata = {'fixture_only': True, 'actual_platform': platform.platform(),
        'linux_labels_are_fixture_aliases': platform.system() != 'Linux',
        'gateway_sha256': digest(args.gateway), 'fcs_sha256': digest(args.server),
        'pack_sha256': digest(args.pack)}
    (bundle / 'BUILD.json').write_text(json.dumps(metadata, indent=2) + '\n')
    inventory = ''.join(f'{digest(path)}  {path.relative_to(bundle).as_posix()}\n'
        for path in sorted(bundle.rglob('*')) if path.is_file())
    (bundle / 'checksums.sha256').write_text(inventory)
    return bundle


def scheduler_fixtures(root):
    """Stub scheduler discovery/launch only; Gateway and FCS are never mocked."""
    binaries = root / 'fixture-bin'
    binaries.mkdir(mode=0o700)
    python = str(Path(sys.executable).absolute())
    arch = 'aarch64' if platform.machine() in ('arm64', 'aarch64') else 'x86_64'
    executable(binaries / 'uname', '#!/usr/bin/env bash\ncase "$1" in\n-m) echo ' + arch + ';;\n-s) echo Linux;;\n*) exec /usr/bin/uname "$@";;\nesac\n')
    executable(binaries / 'scontrol', f'''#!{python}
import datetime as dt,sys
if sys.argv[1:3]==['show','hostnames']:
    print('localhost')
elif sys.argv[1:3]==['show','job']:
    end=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%S')
    print('JobId=4242 Partition=fixture EndTime='+end)
else:
    raise SystemExit('Unexpected fixture scontrol invocation')
''')
    executable(binaries / 'srun', f'''#!{python}
import json,os,pathlib,sys
arguments=sys.argv[1:]
while arguments and arguments[0].startswith('--'):
    arguments.pop(0)
if not arguments or arguments[0]!='bash': raise SystemExit('Unexpected fixture srun command')
env=dict(os.environ,SLURM_PROCID='0',SLURMD_NODENAME='localhost')
pathlib.Path(env['ALPHA_SCRIPT_FCS_WITNESS']).write_text(json.dumps({{'pid':os.getpid()}}))
os.execvpe(arguments[0],arguments,env)
''')
    (binaries / 'python3').symlink_to(python)
    return binaries


def direct_children(pid):
    """Read only a live fixture parent's child identities; never adopt arbitrary PIDs."""
    result = subprocess.run(['pgrep', '-P', str(pid)], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError('Cannot inspect owned fixture children')
    return [int(value) for value in result.stdout.split()]


def process_identity(pid):
    """PID plus start time and command guards against mistaking a reused PID for a survivor."""
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart=', '-o', 'command='], capture_output=True, text=True)
    return result.stdout.strip() or None


async def closed(value):
    """Check actual listener closure after shell traps complete."""
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection('127.0.0.1', value), 1)
        writer.close()
        await writer.wait_closed()
        return False
    except (OSError, TimeoutError):
        return True


async def stop_entry(child):
    """Exercise the public entry's normal SIGTERM trap before any emergency kill."""
    if child.returncode is None:
        child.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(child.wait(), 15)
        except TimeoutError:
            child.kill()
            await child.wait()


async def scenario(args):
    """Run unmodified deployment scripts and verify actual protocol/cleanup results."""
    checks = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='flexserv-alpha-script-') as temporary:
        root = Path(temporary).resolve()
        root.chmod(0o700)
        bundle = assemble_fixture(root, args)
        fixtures = scheduler_fixtures(root)
        selected_port = port()
        environment = dict(os.environ)
        for name in ('FCS_IDENTITIES_JSON', 'FCS_IDENTITIES_FILE', 'FCS_DEMO_TOKEN', 'FCS_ALLOWED_TOOLS',
                     'MCP_ALLOWED_HOSTS', 'GATEWAY_DISTRIBUTED_CONFIG', 'FLEXSERV_JOB_LAUNCH_CONFIG',
                     'FCS_GATEWAY_SERVICE_KEY', 'FCS_GATEWAY_SUBJECTS', 'TLS_CERT', 'TLS_KEY',
                     'FLEXSERV_MODEL_ID', 'FLEXSERV_ON_HEAD', 'PRI_MODEL_REPO', 'PUB_MODEL_REPO'):
            environment.pop(name, None)
        witness = root / 'fcs-witness.json'
        environment.update(PATH=str(fixtures) + ':' + environment['PATH'], FLEXSERV_SITE='template',
            HOST_TYPE='fixture', SLURM_JOB_ID='4242', SLURM_JOB_NODELIST='localhost',
            SLURMD_NODENAME='localhost', SLURM_CLUSTER_NAME='fixture', SLURM_JOB_PARTITION='fixture',
            FLEXSERV_VENV_ROOT=str(root / 'unused-venvs'), MCP_PRINCIPAL='alpha-user',
            LOCAL_PORT=str(selected_port), GATEWAY_BIND_HOST='127.0.0.1',
            RAND_TOKEN=TOKEN, FLEXSERV_ADMIN_SECRET=ADMIN, ALPHA_SCRIPT_FCS_WITNESS=str(witness),
            STARTUP_TIMEOUT_SECONDS='35', RUN_SECONDS='180')
        output = root / 'entry.log'
        identities = {}
        with output.open('wb') as log:
            child = await asyncio.create_subprocess_exec('bash', str(bundle / 'app/run_flexserv.sh'),
                '--control-plane-only', env=environment, cwd=bundle, stdout=log, stderr=log)
            try:
                access = bundle / 'flexserv_access_info.txt'
                async with asyncio.timeout(45):
                    while not access.exists():
                        for pid in direct_children(child.pid):
                            identity = process_identity(pid)
                            if identity and str(bundle / 'bin/flexserv-') in identity:
                                identities[pid] = identity
                        assert child.returncode is None, 'Bash entry exited before access readiness'
                        await asyncio.sleep(.1)
                runtime = next(path.parent for path in (bundle / 'runtime').glob('run-*/ready.json'))
                manifest = json.loads((runtime / 'manifest.json').read_text())
                assert len(manifest['instances']) == 1
                assert not (bundle / 'tmp/tls').exists()
                assert access.stat().st_mode & 0o777 == 0o600
                lines = access.read_text().splitlines()
                assert f'FlexServ token: {TOKEN}' in lines
                url = next(line.split(': ', 1)[1] for line in lines if line.startswith('FlexServ URL: '))
                assert url == f'http://127.0.0.1:{selected_port}'
                checks.append('actual run_flexserv entry invokes bundled install, one-node launch and template access functions')
                checks.append('private access info and retained manifest are published only after TLS staging deletion')
                fcs_pid = json.loads(witness.read_text())['pid']
                candidates = direct_children(child.pid)
                identities = {pid: process_identity(pid) for pid in candidates}
                gateway_ids = [pid for pid, identity in identities.items()
                               if identity and str(bundle / 'bin/flexserv-gateway-') in identity]
                assert len(gateway_ids) == 1 and fcs_pid in candidates, identities
                identities = {pid: identities[pid] for pid in [gateway_ids[0], fcs_pid]}
                import zipfile
                with zipfile.ZipFile(args.pack) as archive:
                    tools = [yaml.safe_load(archive.read(name)) for name in archive.namelist() if name.endswith('/tool.yaml')]
                assert len(tools) == 65, len(tools)
                async with connect(url + '/mcp', token=TOKEN) as client:
                    names = {tool.name for tool in await discover_tools(client)}
                    assert {'io.flexserv.deployments.vllm', 'io.tapis.jobs.submit',
                            'io.tapis.workflows.provision_and_run'} <= names, names
                    assert not any(name.startswith('local.') for name in names)
                    result = await client.call_tool('gateway.operations.list', {})
                    assert not result.is_error and result.structured_content['operations'] == [], result
                checks.append('full 65-tool merged Pack loads without Tapis credentials and public MCP authenticates/discovers both namespaces')
                async with aiohttp.ClientSession(trust_env=False) as http:
                    async with http.get(url + '/healthz/ready') as response:
                        assert response.status == 200, await response.text()
                for path in [output, *list((bundle / 'logs').glob('*'))]:
                    if path.is_file():
                        text = path.read_text(errors='replace')
                        assert TOKEN not in text and ADMIN not in text, 'Credential leaked into ordinary logs'
                checks.append('public readiness is reachable and ordinary logs contain neither access nor administrator secret')
                await stop_entry(child)
                assert child.returncode == 143, child.returncode
                for pid, identity in identities.items():
                    assert process_identity(pid) != identity, f'Owned role survived shutdown: {pid}'
                ports = [selected_port]
                from urllib.parse import urlsplit
                ports += [urlsplit(member['endpoint']).port for member in manifest['instances']]
                assert all(await asyncio.gather(*(closed(value) for value in ports)))
                assert (runtime / 'manifest.json').is_file() and (bundle / 'reports').is_dir()
                assert not (bundle / 'tmp/tls').exists()
                checks.append('SIGTERM executes shell cleanup, reaps both owned roles and closes all ports while retaining evidence')
                return {'checks': checks, 'passed': len(checks), 'seconds': round(time.monotonic() - started, 3),
                    'entry_pid': child.pid, 'owned_roles': sorted(identities), 'all_owned_roles_gone': True,
                    'temporary_job_removed_on_return': str(root), 'discovered_tools': len(names), 'pack_tools': len(tools)}
            except BaseException:
                for path in [output, *list((bundle / 'logs').glob('*'))]:
                    if path.is_file():
                        text = path.read_text(errors='replace')[-12000:]
                        print(path.name + ':\n' + text.replace(TOKEN, '<redacted>').replace(ADMIN, '<redacted>'), file=sys.stderr)
                raise
            finally:
                await stop_entry(child)
                # Emergency cleanup remains limited to a witnessed direct child
                # whose exact start time and executable command still match.
                for pid, identity in identities.items():
                    if identity and process_identity(pid) == identity:
                        os.kill(pid, signal.SIGTERM)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and any(identity and process_identity(pid) == identity for pid, identity in identities.items()):
                    await asyncio.sleep(.1)
                for pid, identity in identities.items():
                    if identity and process_identity(pid) == identity:
                        os.kill(pid, signal.SIGKILL)


async def main(args):
    """Record host-native script evidence separately from Linux release validation."""
    result = {'stage': 'Alpha actual Bash entry', 'platform': platform.platform(),
        'fixture_only_linux_labels': platform.system() != 'Linux', 'real_slurm': 'NOT RUN',
        'real_gpu_vllm': 'NOT RUN', 'linux_release_abi': 'NOT TESTED BY THIS FIXTURE',
        'binaries': {name: {'path': str(getattr(args, name)), 'sha256': digest(getattr(args, name))}
                     for name in ('gateway', 'server')}, 'pack_sha256': digest(args.pack), 'scenario': await scenario(args)}
    assert not Path(result['scenario']['temporary_job_removed_on_return']).exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Alpha actual Bash entry: PASS ({result["scenario"]["passed"]} checks)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway', type=Path, required=True)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    for name in ('gateway', 'server', 'pack'):
        setattr(options, name, getattr(options, name).resolve(strict=True))
    asyncio.run(main(options))
