"""Verify Slurm consumes the FCS-prepared Pack snapshot without shell parsing YAML."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
INSTALL = ROOT / 'hpc_archive/scripts/install.sh'
LAUNCH = ROOT / 'hpc_archive/scripts/launch-flexserv.sh'


class PackRuntime(unittest.TestCase):
    def prepare_fixture(self, root):
        (root / 'tmp').mkdir(); (root / 'runtime').mkdir(); (root / 'bin').mkdir()
        tag = root / 'pack-tag.txt'; tag.write_text('pack-v1.5.0-alpha-2\n')
        (root / 'checksums.sha256').write_text(hashlib.sha256(tag.read_bytes()).hexdigest() + '  pack-tag.txt\n')
        server = root / 'bin/flexserv-capability-server-amd64.bin'
        server.write_text(r'''#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == --prepare-packs ]] || exit 2
cp "$2" "$TRACE"
[[ ${FAIL_FCS:-0} == 0 ]] || { echo 'fixture preparation failed' >&2; exit 1; }
jq -n --slurpfile request "$2" '
  $request[0] as $r | {release_tag:$r.release_tag,
  packs:[range(0;($r.selections|length)) as $i | {name:("item-"+($i|tostring)+".zip"),
    path:($r.output_dir+"/item-"+($i|tostring)+".zip"),source:"local",
    checksums_sha256:("a"*64),zip_sha256:("b"*64)}],visible_tools:["io.example.visible"]}'
'''); server.chmod(0o700)
        return server

    def call_prepare(self, root, server, selection=None, fail=False):
        env = dict(os.environ, FLEXSERV_JOB_DIR=str(root), FCS_BIN=str(server), TRACE=str(root / 'trace'),
                   FAIL_FCS='1' if fail else '0')
        env.pop('PACK_TO_LOAD', None)
        if selection is not None:
            env['PACK_TO_LOAD'] = selection
        return subprocess.run(['bash', '-euc', 'source "$1"; verify_bundle', 'test', str(INSTALL)],
                              env=env, capture_output=True, text=True)

    def test_defaults_and_explicit_local_paths_are_preserved(self):
        for selection, expected in [(None, ['tapis-workflow-pack.zip', 'vllm-pack.zip']),
                                    ('./local path.zip;other.zip', ['./local path.zip', 'other.zip'])]:
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); server = self.prepare_fixture(root)
                result = self.call_prepare(root, server, selection)
                self.assertEqual(result.returncode, 0, result.stderr)
                request = json.loads((root / 'trace').read_text())
                manifest = json.loads((root / 'runtime/packs.json').read_text())
                self.assertEqual(request['selections'], expected)
                self.assertEqual(manifest['release_tag'], 'pack-v1.5.0-alpha-2')
                self.assertEqual(len(manifest['packs']), 2)
                self.assertEqual(manifest['visible_tools'], ['io.example.visible'])
                self.assertEqual((root / 'runtime/packs.json').stat().st_mode & 0o777, 0o600)
                self.assertFalse(list((root / 'tmp').glob('pack-select.*')))

    def test_invalid_selection_and_failed_preparation_do_not_publish_manifest(self):
        for selection, fail in [('a.zip;;b.zip', False), ('a.zip', True)]:
            with self.subTest(selection=selection, fail=fail), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); server = self.prepare_fixture(root)
                result = self.call_prepare(root, server, selection, fail)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / 'runtime/packs.json').exists())
                self.assertFalse(list((root / 'tmp').glob('pack-select.*')))

    def test_protected_job_config_preserves_pack_order_and_release_visibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); server = self.prepare_fixture(root)
            result = self.call_prepare(root, server)
            self.assertEqual(result.returncode, 0, result.stderr)
            (root / 'scripts').mkdir()
            shutil.copyfile(ROOT / 'hpc_archive/scripts/job-launch.jq', root / 'scripts/job-launch.jq')
            run = root / 'runtime/run-fixture'; run.mkdir()
            (run / 'nodes.txt').write_text('node-a\nnode-b\n')
            shell = '''source "$1"; scontrol(){ printf 'EndTime=%s\\n' "$END_TIME"; }; prepare_job_config'''
            import datetime
            end = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%S')
            env = dict(os.environ, FLEXSERV_JOB_DIR=str(root), FLEXSERV_RUNTIME=str(run),
                       FLEXSERV_RUN_ID='run-fixture', SLURM_JOB_ID='123', END_TIME=end,
                       STARTUP_TIMEOUT_SECONDS='30', MCP_PRINCIPAL='fixture', FLEXSERV_PACK_MODE='multi')
            env.pop('FLEXSERV_MODEL_ID', None); env.pop('FCS_VLLM_API_KEY', None)
            result = subprocess.run(['bash', '-euc', shell, 'test', str(LAUNCH)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads((run / 'job-launch.json').read_text())
            packs = json.loads((root / 'runtime/packs.json').read_text())['packs']
            self.assertEqual(config['fcs_arguments'], sum((['--pack', item['path'], '--approved-checksums', item['checksums_sha256'], '--approved-zip-sha256', item['zip_sha256']] for item in packs), []))
            allowed = config['mcp']['callers'][0]['allowed_tools']
            self.assertIn('io.example.visible', allowed)
            self.assertIn('gateway.operations.status', allowed)
            self.assertNotIn('io.example.hidden', allowed)

    def test_worker_passes_host_pins_and_removes_ambient_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); server = self.prepare_fixture(root)
            result = self.call_prepare(root, server)
            self.assertEqual(result.returncode, 0, result.stderr)
            (root / 'scripts').mkdir()
            shutil.copyfile(ROOT / 'hpc_archive/scripts/host-request.jq', root / 'scripts/host-request.jq')
            server.write_text(r'''#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == --prepare-host-config ]]; then cp "$2" "$TRACE_HOST"; echo '{}'
else printf '%s' "${PACK_TO_LOAD-unset}" > "$TRACE_ENV"; fi
''');server.chmod(0o700)
            shell = 'source "$1"; source "$2"; select_arch(){ ARCH=amd64; }; node_ipv4_addresses(){ echo 192.0.2.1; }; fcs_worker "$3" "$3/runtime/run-fixture"'
            env = dict(os.environ, SLURM_PROCID='0', SLURMD_NODENAME='node-a', FLEXSERV_PREPARE_BACKENDS='1',
                FLEXSERV_VENV_ROOT=str(root/'venvs'), FLEXSERV_VLLM_VENV_PATH=str(root/'venvs/vllm'),
                PRI_MODEL_REPO=str(root/'private'), PUB_MODEL_REPO=str(root/'public'), MCP_PRINCIPAL='fixture',
                HARDWARE_ENV='fixture', FLEXSERV_ACCELERATOR_BACKEND='cuda', FLEXSERV_GPU_DEVICES='0',
                FLEXSERV_SERVICE_PORT_START='18000', FLEXSERV_SERVICE_PORT_END='18100', PACK_TO_LOAD='untrusted.zip',
                FLEXSERV_PACK_MODE='multi',
                TRACE_HOST=str(root/'host.json'), TRACE_ENV=str(root/'environment'))
            result = subprocess.run(['bash','-euc',shell,'test',str(INSTALL),str(LAUNCH),str(root)],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            request = json.loads((root/'host.json').read_text())
            self.assertEqual(len(request['packs']), 2)
            self.assertNotIn('pack', request)
            self.assertEqual((root/'environment').read_text(), 'unset')


if __name__ == '__main__':
    unittest.main()
