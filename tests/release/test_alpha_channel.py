"""Exercise rolling-channel decisions without changing real GitHub releases."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
CHANNEL='tapis-flexserv-1.5.0-alpha-44'


class AlphaChannel(unittest.TestCase):
    def test_verified_promotion_and_rejection_boundaries(self):
        cases=['promote','beta_candidate','beta_published','immutable','bad_digest','initial_channel','already_current']
        for case in cases:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);(root/'bin').mkdir();(root/'dist').mkdir()
                archive=root/'dist/Tapis-FlexServ.zip';archive.write_bytes(b'verified-candidate-fixture')
                checksum=root/'dist/Tapis-FlexServ.zip.sha256'
                checksum.write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  Tapis-FlexServ.zip\n')
                assets=[dict(name=p.name,digest='sha256:'+hashlib.sha256(p.read_bytes()).hexdigest()) for p in [archive,checksum]]
                candidate='tapis-flexserv-1.5.0-alpha-45'
                if case=='beta_candidate':candidate='tapis-flexserv-1.5.0-beta-1'
                if case=='initial_channel':candidate=CHANNEL
                source=dict(id=45,tag_name=candidate,draft=False,assets=assets)
                if case=='bad_digest':source=dict(source,assets=[dict(assets[0],digest='sha256:'+'0'*64),assets[1]])
                channel=dict(id=44,tag_name=CHANNEL,draft=False,immutable=(case=='immutable'),
                    assets=assets if case=='already_current' else [dict(a,digest='sha256:'+'1'*64) for a in assets])
                releases=[[dict(tag_name='tapis-flexserv-1.5.0-beta-1',draft=False)]] if case=='beta_published' else [[]]
                for name,data in [('source',source),('channel',channel),('updated',dict(channel,assets=assets)),('releases',releases)]:
                    (root/(name+'.json')).write_text(json.dumps(data))
                gh=root/'bin/gh'
                gh.write_text(r'''#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FIXTURE/trace"
if [[ "$1" == release && "$2" == upload ]]; then touch "$FIXTURE/uploaded"; exit; fi
if [[ "$1" == api && "$2" == --method ]]; then cat "$FIXTURE/updated.json"; exit; fi
case "$2" in
  *releases\?*) cat "$FIXTURE/releases.json" ;;
  *tags/tapis-flexserv-1.5.0-alpha-44) cat "$FIXTURE/channel.json" ;;
  *tags/*) cat "$FIXTURE/source.json" ;;
  *) exit 2 ;;
esac
''');gh.chmod(0o700)
                result=subprocess.run(['bash',str(ROOT/'scripts/update-alpha-channel.sh'),candidate,str(root/'dist')],
                    env=dict(os.environ,PATH=str(root/'bin')+':'+os.environ['PATH'],FIXTURE=str(root),GITHUB_REPOSITORY='fixture/repo'),capture_output=True,text=True)
                self.assertEqual(result.returncode==0,case not in ['immutable','bad_digest'],result.stderr)
                self.assertEqual((root/'uploaded').exists(),case=='promote')
                if case=='promote':
                    calls=(root/'trace').read_text()
                    self.assertIn('release upload '+CHANNEL,calls)
                    self.assertIn('--clobber',calls)
                    self.assertIn('PATCH repos/fixture/repo/releases/44',calls)


if __name__=='__main__':unittest.main()
