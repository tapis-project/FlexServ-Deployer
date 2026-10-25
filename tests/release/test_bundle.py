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

    def test_owned_tls_cleanup_preserves_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); run='run-'+'a'*32
            (root/'tmp/tls'/run).mkdir(parents=True); (root/'tmp/tls'/run/'key').write_text('fixture')
            (root/'runtime').mkdir(); (root/'runtime/status').write_text('retained')
            command='source "$1"; FLEXSERV_JOB_DIR="$2"; FLEXSERV_RUN_ID="$3"; TUNNEL_PIDS=(); concrete_cleanup_site_resources(){ :; }; cleanup_flexserv'
            subprocess.run(['/bin/bash','-euc',command,'test',str(ROOT/'hpc_archive/scripts/launch-flexserv.sh'),str(root),run],check=True)
            self.assertFalse((root/'tmp/tls').exists());self.assertEqual((root/'runtime/status').read_text(),'retained')

    def test_plaintext_tunnel_cannot_bind_publicly(self):
        command='source "$1"; PROTOCOL=http; FLEXSERV_TUNNEL_BIND=0.0.0.0; concrete_setup_reverse_tunnels'
        result=subprocess.run(['/bin/bash','-ec',command,'test',str(ROOT/'hpc_archive/scripts/site/tacc.sh')],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('requires HTTPS',result.stderr)

    def test_no_allocation_is_rejected(self):
        env=dict(os.environ);env.pop('SLURM_JOB_ID',None);env.pop('SLURM_JOB_NODELIST',None)
        result=subprocess.run(['bash',str(ROOT/'hpc_archive/app/run_flexserv.sh'),'--control-plane-only'],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('Slurm allocation',result.stderr)


if __name__=='__main__':unittest.main()
