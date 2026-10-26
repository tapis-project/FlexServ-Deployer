"""Public ZIP layout and simple shell contracts; no site changes or package installs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('assemble',ROOT/'scripts/assemble_bundle.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Bundle(unittest.TestCase):
    def test_complete_reproducible_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); binaries=root/'bin';binaries.mkdir()
            for component in ('flexserv-gateway','flexserv-capability-server'):
                for arch in ('amd64','arm64'):
                    (binaries/f'{component}-{arch}.bin').write_bytes(b'controlled-fixture-not-a-release-binary')
            pack=root/'pack.zip'
            with zipfile.ZipFile(pack,'w') as archive:
                archive.writestr('manifest.json','{}');archive.writestr('checksums.sha256','')
            a,b=root/'a.zip',root/'b.zip'
            for output in [a,b]:module.assemble(ROOT/'hpc_archive',binaries,pack,output,{'gateway':{'commit':'a'*40}})
            self.assertEqual(a.read_bytes(),b.read_bytes())
            with zipfile.ZipFile(a) as archive:
                names=archive.namelist()
                self.assertIn('app/run_flexserv.sh',names)
                self.assertIn('scripts/site/osc.sh',names)
                self.assertIn('bin/flexserv-capability-server-arm64.bin',names)
                self.assertNotIn('bin/flexserv-bootstrap',names)
                self.assertEqual(archive.read('tapisjob.manifest').decode().strip(),'tapisjob_executable=app/run_flexserv.sh\nmy_project=FlexServ')
                for line in archive.read('checksums.sha256').decode().splitlines():
                    digest,name=line.split('  ',1)
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),digest)

    def test_missing_architecture_fails_before_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaisesRegex(ValueError,'Missing architecture'):
                module.assemble(ROOT/'hpc_archive',root,root/'unused',root/'out.zip',{})

    def test_release_assets_must_match_before_draft_can_be_published(self):
        """Exercise the workflow's actual validation against authenticated list fixtures."""
        workflow=(ROOT/'.github/workflows/publish-flexserv-bundle.yml').read_text()
        blocks=re.findall(r"python3 - <<'PY'\n(.*?)\n          PY",workflow,re.S)
        validation=next(textwrap.dedent(body) for body in blocks if 'import hashlib,json,os,pathlib' in body)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dist=root/'dist';dist.mkdir()
            assets=[]
            for name,data in [('Tapis-FlexServ.zip',b'archive-fixture'),('Tapis-FlexServ.zip.sha256',b'checksum-fixture')]:
                (dist/name).write_bytes(data)
                assets.append(dict(name=name,size=len(data),digest='sha256:'+hashlib.sha256(data).hexdigest()))
            release=dict(id=42,tag_name='fixture',draft=True,assets=assets)
            cases=[('draft',[release],True,'draft'),
                   ('published',[dict(release,draft=False)],True,'published'),
                   ('missing',[dict(release,assets=assets[:1])],False,None),
                   ('corrupt',[dict(release,assets=[dict(assets[0],digest='sha256:'+'0'*64),assets[1]])],False,None),
                   ('missing_digest',[dict(release,assets=[dict(assets[0],digest=None),assets[1]])],False,None),
                   ('ambiguous',[release,release],False,None)]
            for name,records,success,state in cases:
                with self.subTest(name=name):
                    for path in [dist/'release-id.txt',dist/'release-state.txt']:
                        path.unlink(missing_ok=True)
                    (dist/'uploaded-releases.json').write_text(json.dumps([records]))
                    result=subprocess.run([sys.executable,'-c',validation],cwd=root,
                        env=dict(os.environ,RELEASE_TAG='fixture'),capture_output=True,text=True)
                    self.assertEqual(result.returncode==0,success,result.stderr)
                    if success:
                        self.assertEqual((dist/'release-id.txt').read_text(),'42')
                        self.assertEqual((dist/'release-state.txt').read_text(),state)
                    else:
                        self.assertFalse((dist/'release-id.txt').exists())

    def test_bash_syntax_and_help(self):
        for path in (ROOT/'hpc_archive').rglob('*.sh'):
            subprocess.run(['bash','-n',str(path)],check=True)
        output=subprocess.check_output(['bash',str(ROOT/'hpc_archive/app/run_flexserv.sh'),'--help'],text=True)
        self.assertIn('--control-plane-only',output)

    def test_site_paths_and_no_source_side_effects(self):
        for host,expected in [('vista','/work/projects/aci/cic/apps/flexserv/virt_envs'),('dgx','/opt/share/virt_envs'),('horizon',str(Path.home()/'tapis-app/flexserv/virt_envs'))]:
            command='source "$1"; HOST_TYPE="$2"; get_flexserv_venv_root'
            output=subprocess.check_output(['bash','-c',command,'test',str(ROOT/'hpc_archive/scripts/site/tacc.sh'),host],text=True)
            self.assertEqual(output.strip(),expected)
        for path in (ROOT/'hpc_archive/scripts').glob('*.sh'):
            result=subprocess.run(['bash','-c','set -euo pipefail; source "$1"','test',str(path)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout,'')

    def test_max_skip_and_explicit_requirement(self):
        source=str(ROOT/'hpc_archive/scripts/install.sh')
        command='source "$1"; max_supported(){ return 1; }; FLEXSERV_INSTALL_MAX=auto; ensure_max'
        result=subprocess.run(['bash','-ec',command,'test',source],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('MAX skipped',result.stdout)
        result=subprocess.run(['bash','-ec',command.replace('=auto','=1'),'test',source],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('required but unsupported',result.stderr)

    def test_existing_pack_is_bound_to_bundled_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'pack').mkdir(); (root/'runtime/pack').mkdir(parents=True)
            old=b'{"name":"old"}'; new=b'{"name":"new"}'
            def inventory(data): return (hashlib.sha256(data).hexdigest()+'  manifest.json\n').encode()
            (root/'runtime/pack/manifest.json').write_bytes(old)
            (root/'runtime/pack/checksums.sha256').write_bytes(inventory(old))
            pack=root/'pack/flexserv-capability-pack.zip'
            with zipfile.ZipFile(pack,'w') as archive:
                archive.writestr('manifest.json',new); archive.writestr('checksums.sha256',inventory(new))
            (root/'checksums.sha256').write_text(hashlib.sha256(pack.read_bytes()).hexdigest()+'  pack/flexserv-capability-pack.zip\n')
            command='source "$1"; export FLEXSERV_JOB_DIR="$2"; verify_bundle'
            result=subprocess.run(['bash','-ec',command,'test',str(ROOT/'hpc_archive/scripts/install.sh'),str(root)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('differs from the bundled release',result.stderr)

    def test_pack_zip_traversal_and_symlinks_are_rejected_before_extraction(self):
        for kind in ['traversal','symlink']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);(root/'pack').mkdir()
                path=root/'pack/flexserv-capability-pack.zip'
                with zipfile.ZipFile(path,'w') as archive:
                    if kind=='traversal':archive.writestr('../escape','must-not-extract')
                    else:
                        info=zipfile.ZipInfo('linked');info.create_system=3;info.external_attr=0o120777<<16
                        archive.writestr(info,'../escape')
                (root/'checksums.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'  pack/flexserv-capability-pack.zip\n')
                command='source "$1"; FLEXSERV_JOB_DIR="$2"; verify_bundle'
                result=subprocess.run(['bash','-ec',command,'test',str(ROOT/'hpc_archive/scripts/install.sh'),str(root)],capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0);self.assertFalse((root/'runtime/escape').exists())
                self.assertFalse((root/'runtime/pack').exists())

    def test_owned_tls_cleanup_preserves_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); run='run-'+'a'*32
            (root/'tmp/tls'/run).mkdir(parents=True); (root/'tmp/tls'/run/'key').write_text('fixture')
            (root/'runtime').mkdir(); (root/'runtime/status').write_text('retained')
            command='source "$1"; FLEXSERV_JOB_DIR="$2"; FLEXSERV_RUN_ID="$3"; TUNNEL_PIDS=(); concrete_cleanup_site_resources(){ :; }; cleanup_flexserv'
            subprocess.run(['/bin/bash','-euc',command,'test',str(ROOT/'hpc_archive/scripts/launch-flexserv.sh'),str(root),run],check=True)
            self.assertFalse((root/'tmp/tls').exists());self.assertEqual((root/'runtime/status').read_text(),'retained')

    def test_plaintext_tunnel_can_bind_publicly_when_http_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'logs').mkdir();(root/'bin').mkdir()
            ssh=root/'bin/ssh';ssh.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$TRACE"\nexec sleep 30\n');ssh.chmod(0o700)
            command='source "$1"; PROTOCOL=http; FLEXSERV_JOB_DIR="$2"; LOGIN_PORT=55000; LOCAL_PORT=8000; FLEXSERV_LOGIN_NODES=fixture; FLEXSERV_LOGIN_PUBLIC_HOSTS=login.example; TUNNEL_PIDS=(); trap \'kill "${TUNNEL_PIDS[@]}" 2>/dev/null || true; wait\' EXIT; concrete_setup_reverse_tunnels; concrete_setup_access_url; echo "$FLEXSERV_ACCESS_URL"'
            result=subprocess.run(['/bin/bash','-ec',command,'test',str(ROOT/'hpc_archive/scripts/site/tacc.sh'),str(root)],
                env=dict(os.environ,PATH=str(root/'bin')+':'+os.environ['PATH'],TRACE=str(root/'trace')),capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('-R 0.0.0.0:55000:127.0.0.1:8000',(root/'trace').read_text())
            self.assertIn('http://login.example:55000',result.stdout)

    def test_shell_files_never_embed_or_dispatch_python_glue(self):
        for path in [*(ROOT/'hpc_archive').rglob('*.sh'),*(ROOT/'interactive-slurm').rglob('*.sh'),*(ROOT/'tapis_def').rglob('*.sh')]:
            text=path.read_text()
            self.assertNotRegex(text,r'\bpython[0-9.]*\s+(?:-c|-\s|[^\s]+\.py\b)',str(path))
            self.assertNotRegex(text,r'(?m)^\s*(?:import|from)\s+\w+',str(path))
        help=subprocess.check_output(['bash',str(ROOT/'hpc_archive/app/run_flexserv.sh'),'--help'],text=True)
        self.assertIn('--enable-https',help);self.assertIn('--trust-remote-code',help)

    def test_missing_tap_falls_back_to_random_token_and_own_login_port(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'bin').mkdir()
            ssh=root/'bin/ssh';ssh.write_text('#!/usr/bin/env bash\nif [[ "$1" == -G ]]; then echo "hostname login.example"; else echo "LISTEN 0 128 0.0.0.0:22000 0.0.0.0:*"; fi\n');ssh.chmod(0o700)
            command='source "$1"; source "$2"; TAP_FUNCTIONS="$3/missing"; ENABLE_HTTPS=0; unset RAND_TOKEN LOGIN_PORT; concrete_setup_environment; concrete_setup_random_token; concrete_setup_login_port; TLS_CERT=stale; TLS_KEY=stale; concrete_setup_cert_tls; [[ -z ${TLS_CERT+x} && -z ${TLS_KEY+x} ]]; printf "%s %s %s\\n" "$TAP_AVAILABLE" "$RAND_TOKEN" "$LOGIN_PORT"'
            result=subprocess.run(['/bin/bash','-ec',command,'test',str(ROOT/'hpc_archive/scripts/common.sh'),str(ROOT/'hpc_archive/scripts/site/tacc.sh'),str(root)],
                env=dict(os.environ,PATH=str(root/'bin')+':'+os.environ['PATH'],FLEXSERV_LOGIN_NODES='fixture'),capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            available,token,port=result.stdout.split()
            self.assertEqual(available,'0');self.assertRegex(token,r'^[a-f0-9]{64}$')
            self.assertTrue(20000<=int(port)<60000);self.assertNotEqual(port,'22000')

    def test_https_requires_tap_certificate_and_never_silently_uses_http(self):
        command='source "$1"; ENABLE_HTTPS=1; TAP_AVAILABLE=0; SLURM_JOB_ID=fixture; setup_public_tls'
        result=subprocess.run(['/bin/bash','-ec',command,'test',str(ROOT/'hpc_archive/scripts/common.sh')],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('TAP certificate/key',result.stderr)

    def test_tacc_app_and_job_generator_use_current_native_contract(self):
        app=json.loads((ROOT/'tapis_def/1.5.0/app.json').read_text())
        self.assertEqual((app['id'],app['version']),('FlexServ-1.5.0','1.5.0'))
        self.assertEqual((app['runtime'],app['jobType']),('ZIP','BATCH'))
        self.assertNotIn('runtimeOptions',app);self.assertNotIn('name',app)
        parameters=app['jobAttributes']['parameterSet']
        self.assertEqual({v['key'] for v in parameters['envVariables']} & {'PRI_MODEL_REPO','PUB_MODEL_REPO'}, {'PRI_MODEL_REPO','PUB_MODEL_REPO'})
        self.assertEqual({v['arg'] for v in parameters['appArgs']},{'--enable-https','--trust-remote-code','--control-plane-only'})
        result=subprocess.check_output(['bash',str(ROOT/'tapis_def/1.5.0/prepare-job.sh')],
            env=dict(os.environ,TAPIS_EXEC_SYSTEM_ID='fixture-system',TAPIS_LOGICAL_QUEUE='fixture',TAPIS_ALLOCATION='fixture123'),text=True)
        job=json.loads(result);self.assertEqual(job['appId'],app['id']);self.assertEqual(job['appVersion'],app['version'])
        flags={arg['name']:arg['include'] for arg in job['parameterSet']['appArgs']}
        self.assertEqual(flags,dict(enableHttps=False,trustRemoteCode=False,controlPlaneOnly=True))
        self.assertEqual(job['parameterSet']['schedulerOptions'][0]['arg'],'-A fixture123')

    def test_worker_json_preserves_paths_and_does_not_enable_remote_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'bin').mkdir();(root/'runtime/pack').mkdir(parents=True)
            (root/'scripts').mkdir();shutil.copyfile(ROOT/'hpc_archive/scripts/host-request.jq',root/'scripts/host-request.jq')
            (root/'runtime/pack/checksums.sha256').write_text('fixture')
            server=root/'bin/flexserv-capability-server-amd64.bin'
            server.write_text('#!/usr/bin/env bash\nif [[ "$1" == --prepare-host-config ]]; then cp -- "$2" "$TRACE"; echo "{}"; else printf "%s" "$FLEXSERV_TRUST_REMOTE_CODE" > "$TRACE.env"; fi\n');server.chmod(0o700)
            private=str(root/'private space "quoted"')
            command='source "$1"; source "$2"; select_arch(){ ARCH=amd64; }; node_ipv4_addresses(){ echo 192.0.2.1; }; fcs_worker "$3" "$3/runtime/run-fixture"'
            result=subprocess.run(['bash','-euc',command,'test',str(ROOT/'hpc_archive/scripts/install.sh'),str(ROOT/'hpc_archive/scripts/launch-flexserv.sh'),str(root)],
                env=dict(os.environ,SLURM_PROCID='0',SLURMD_NODENAME='fixture',FLEXSERV_PREPARE_BACKENDS='1',
                    FLEXSERV_VENV_ROOT=str(root/'venvs'),FLEXSERV_VLLM_VENV_PATH=str(root/'venvs/vllm'),
                    PRI_MODEL_REPO=private,PUB_MODEL_REPO=str(root/'public'),MCP_PRINCIPAL='fixture',HARDWARE_ENV='fixture',
                    FLEXSERV_ACCELERATOR_BACKEND='cuda',FLEXSERV_GPU_DEVICES='0,1',FLEXSERV_SERVICE_PORT_START='18000',FLEXSERV_SERVICE_PORT_END='18100',
                    FLEXSERV_TRUST_REMOTE_CODE='true',TRACE=str(root/'request')),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            request=json.loads((root/'request').read_text());self.assertEqual(request['private_repo'],private)
            self.assertEqual(request['gpu_devices'],['0','1']);self.assertNotIn('trust_remote_code',request)
            self.assertEqual((root/'request.env').read_text(),'true')

    def test_no_allocation_is_rejected(self):
        env=dict(os.environ);env.pop('SLURM_JOB_ID',None);env.pop('SLURM_JOB_NODELIST',None)
        result=subprocess.run(['bash',str(ROOT/'hpc_archive/app/run_flexserv.sh'),'--control-plane-only'],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('Slurm allocation',result.stderr)


if __name__=='__main__':unittest.main()
