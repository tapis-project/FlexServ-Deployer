"""Verify pinned release selection and downloaded inventory boundaries offline."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('download', ROOT / 'scripts/download_component_releases.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class ComponentDownload(unittest.TestCase):
    def test_deployer_uses_only_released_components(self):
        workflow = (ROOT / '.github/workflows/publish-flexserv-bundle.yml').read_text()
        self.assertNotIn('build-component.yml', workflow)
        self.assertNotIn('cargo ', workflow)
        self.assertNotIn('rustup', workflow)
        self.assertNotIn('source-ref', workflow)
        self.assertIn('download_component_releases.py', workflow)
        self.assertIn('prepare_release_inputs.py', workflow)
        self.assertEqual(download.COMPONENTS['gateway'], ('tapis-project/FlexServ', 'feature/1.5.0'))
        self.assertEqual(download.COMPONENTS['pack'][1], 'feature/new_fcs')

    def test_latest_published_on_approved_branch_and_explicit_pin(self):
        def request(path, paginate=False):
            if '/compare/' in path:
                return {'status': 'behind' if 'b' * 40 in path else 'ahead'}
            if '/commits/' in path:
                return {'sha': 'b' * 40 if path.endswith('wrong-branch') else 'a' * 40}
            releases = [dict(id=index, tag_name=tag, draft=draft, published_at=date,
                             assets=[{'name': 'release-assets.sha256'}])
                        for index, tag, draft, date in [(1, 'good', False, '2026-10-24'),
                            (2, 'draft', True, '2026-10-26'), (3, 'wrong-branch', False, '2026-10-25')]]
            if '/tags/' in path:
                return next(r for r in releases if path.endswith(r['tag_name']))
            return [releases]
        result = download.resolve('gateway', request=request)
        self.assertEqual(result['tag'], 'good')
        self.assertEqual(result['repository'], 'tapis-project/FlexServ')
        self.assertEqual(download.resolve('gateway', 'good', request)['release_id'], 1)
        with self.assertRaises(ValueError):
            download.resolve('gateway', 'wrong-branch', request)

    def test_full_inventory_and_api_hashes_are_both_required(self):
        for case in ('valid', 'api_digest', 'inventory_digest', 'extra_asset', 'duplicate', 'traversal'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                data = {'binary.bin': b'binary-fixture'}
                digest = hashlib.sha256(data['binary.bin']).hexdigest()
                data['release-assets.sha256'] = ((('0' * 64 if case == 'inventory_digest' else digest)
                                                + '  binary.bin\n')).encode()
                if case == 'extra_asset':
                    data['unexpected.txt'] = b'unlisted'
                assets = [dict(name=name, id=index, size=len(content), digest='sha256:' + hashlib.sha256(content).hexdigest())
                          for index, (name, content) in enumerate(data.items())]
                if case == 'api_digest':
                    assets[0]['digest'] = 'sha256:' + '0' * 64
                if case == 'duplicate':
                    assets.append(assets[0])
                if case == 'traversal':
                    assets[0] = dict(assets[0], name='../outside')
                def fetch(repo, asset, path):
                    path.write_bytes(data[asset['name']])
                selected = dict(repository='fixture/repo', commit='a' * 40, release={'assets': assets})
                if case == 'valid':
                    identity, hashes = download.download(selected, Path(directory) / 'out', fetch)
                    self.assertEqual(hashes['binary.bin'], digest)
                    self.assertNotIn('release', identity)
                else:
                    with self.assertRaises(ValueError):
                        download.download(selected, Path(directory) / 'out', fetch)
