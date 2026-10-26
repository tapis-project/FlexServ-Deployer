"""Exercise fixed-channel publication without creating or moving real Git tags."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CHANNEL = 'tapis-flexserv-1.5.0-alpha-44'
COMMIT = 'a' * 40


class AlphaChannel(unittest.TestCase):
    def test_verified_assembly_and_rejection_boundaries(self):
        cases = ['promote', 'beta_published', 'immutable', 'bad_checksum', 'wrong_commit',
                 'wrong_branch', 'already_current', 'bad_uploaded_digest', 'extra_asset', 'superseded']
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'bin').mkdir(); (root / 'dist').mkdir()
                components = {name: dict(repository='tapis-project/' + repository, release_id=index,
                    commit=COMMIT, assets={'asset.zip': '1' * 64}) for index, (name, repository) in enumerate(
                        [('gateway', 'FlexServ'), ('fcs', 'FlexServ-FCS'), ('pack', 'FlexServ-FCS-Pack')])}
                components['deployer'] = dict(repository='fixture/repo', commit='b' * 40 if case == 'wrong_commit' else COMMIT)
                archive = root / 'dist/Tapis-FlexServ.zip'
                with zipfile.ZipFile(archive, 'w') as packed:
                    packed.writestr('BUILD.json', json.dumps(dict(schema_version=1, components=components)))
                checksum = root / 'dist/Tapis-FlexServ.zip.sha256'
                checksum.write_text(('0' * 64 if case == 'bad_checksum' else hashlib.sha256(archive.read_bytes()).hexdigest())
                                    + '  Tapis-FlexServ.zip\n')
                assets = [dict(name=p.name, digest='sha256:' + hashlib.sha256(p.read_bytes()).hexdigest()) for p in [archive, checksum]]
                old_assets = [dict(a, digest='sha256:' + '1' * 64) for a in assets]
                channel = dict(id=44, tag_name=CHANNEL, draft=False, immutable=(case == 'immutable'),
                               assets=assets if case == 'already_current' else old_assets)
                if case == 'extra_asset':
                    channel['assets'].append(dict(name='unexpected', digest='sha256:' + '0' * 64))
                releases = [[dict(tag_name='tapis-flexserv-1.5.0-beta-1', draft=False)]] if case == 'beta_published' else [[]]
                for name, data in [('channel', channel), ('updated', dict(channel, assets=old_assets if case == 'bad_uploaded_digest' else assets)), ('releases', releases)]:
                    (root / (name + '.json')).write_text(json.dumps(data))
                gh = root / 'bin/gh'
                gh.write_text(r'''#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FIXTURE/trace"
if [[ "$1" == release && "$2" == upload ]]; then touch "$FIXTURE/uploaded"; exit; fi
if [[ "$1" == api && "$2" == --method ]]; then cat "$FIXTURE/updated.json"; exit; fi
case "$2" in
  */commits/*) printf '%s\n' "$FIXTURE_HEAD" ;;
  *releases\?*) cat "$FIXTURE/releases.json" ;;
  *tags/tapis-flexserv-1.5.0-alpha-44) cat "$FIXTURE/channel.json" ;;
  *) exit 2 ;;
esac
''')
                gh.chmod(0o700)
                result = subprocess.run(['bash', str(ROOT / 'scripts/update-alpha-channel.sh'), str(root / 'dist')],
                    env=dict(os.environ, PATH=str(root / 'bin') + ':' + os.environ['PATH'], FIXTURE=str(root),
                        GITHUB_REPOSITORY='fixture/repo', GITHUB_SHA=COMMIT, GITHUB_RUN_ID='1234',
                        FIXTURE_HEAD='b' * 40 if case == 'superseded' else COMMIT,
                        GITHUB_REF='refs/heads/wrong' if case == 'wrong_branch' else 'refs/heads/feature/1.5.0-alpha'),
                    capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, case in ['promote', 'beta_published', 'already_current', 'superseded'], result.stderr)
                self.assertEqual((root / 'uploaded').exists(), case in ['promote', 'bad_uploaded_digest'])
                if case == 'promote':
                    calls = (root / 'trace').read_text()
                    self.assertIn('release upload ' + CHANNEL, calls)
                    self.assertIn('--clobber', calls)
                    self.assertIn('PATCH repos/fixture/repo/releases/44', calls)
                    self.assertNotIn('release create', calls)
                    self.assertNotIn('/git/refs', calls)


if __name__ == '__main__':
    unittest.main()
