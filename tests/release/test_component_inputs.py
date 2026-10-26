"""Exercise standalone Pack merge and binary provenance without native builds."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('inputs', ROOT / 'scripts/prepare_release_inputs.py')
inputs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inputs)


def write_pack(path, name, content=b'shared', version='1.0.0'):
    """Construct a fully inventoried fixture with one shared toolset."""
    manifest = dict(schema_version='flexserv.fcs-pack.v1alpha2', name=name, version=version,
                    target='any', toolsets=[dict(name='io.shared', version='1.0.0', path='fcs_tools/io/shared')])
    data = {'manifest.json': (json.dumps(manifest) + '\n').encode(), 'fcs_tools/io/shared/tool.yaml': content}
    data['checksums.sha256'] = ''.join(f'{hashlib.sha256(value).hexdigest()}  {key}\n'
                                     for key, value in sorted(data.items())).encode()
    with zipfile.ZipFile(path, 'w') as archive:
        for key, value in data.items():
            archive.writestr(key, value)


class ComponentInputs(unittest.TestCase):
    def test_merge_identical_helpers_is_deterministic_and_conflicts_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'a.zip', root / 'b.zip'
            write_pack(a, 'a'); write_pack(b, 'b')
            one, two = root / 'one.zip', root / 'two.zip'
            inputs.merge_packs([a, b], one); inputs.merge_packs([b, a], two)
            self.assertEqual(one.read_bytes(), two.read_bytes())
            manifest, payload = inputs.read_pack(one)
            self.assertEqual(len(manifest['toolsets']), 1)
            self.assertEqual(payload['fcs_tools/io/shared/tool.yaml'], b'shared')
            write_pack(b, 'b', b'conflicting')
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                inputs.merge_packs([a, b], root / 'bad.zip')
            write_pack(b, 'b', version='2.0.0')
            with self.assertRaisesRegex(ValueError, 'versions'):
                inputs.merge_packs([a, b], root / 'bad.zip')

    def test_pack_tampering_and_unsafe_member_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pack.zip'
            for member in ('extra.txt', '../outside', 'a//b'):
                with self.subTest(member=member):
                    write_pack(path, 'fixture')
                    with zipfile.ZipFile(path, 'a') as archive:
                        archive.writestr(member, b'not-in-inventory')
                    with self.assertRaises(ValueError):
                        inputs.read_pack(path)

    def test_release_provenance_and_cross_architecture_identity(self):
        for case in ('valid', 'source', 'version', 'abi', 'pack_source'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root / 'downloads'; source.mkdir()
                metadata = {name: dict(commit='a' * 40, repository='tapis-project/' + name)
                            for name in ('gateway', 'fcs', 'pack')}
                (source / 'resolved.json').write_text(json.dumps(metadata))
                for component, prefix in (('gateway', 'flexserv-gateway'), ('fcs', 'flexserv-capability-server')):
                    target = source / component; target.mkdir()
                    for arch, platform in (('amd64', 'linux-x86_64'), ('arm64', 'linux-aarch64')):
                        stem = prefix + '-' + platform
                        (target / (prefix + '-' + arch + '.bin')).write_bytes(b'controlled-binary-fixture')
                        (target / (stem + '-source.txt')).write_text('b' * 40 if case == 'source' else 'a' * 40)
                        (target / (stem + '-version.txt')).write_text('2' if case == 'version' and arch == 'arm64' else '1')
                        (target / (stem + '-rust.txt')).write_text('rustc fixture')
                        (target / (stem + '-abi.json')).write_text(json.dumps({'architecture': 'wrong' if case == 'abi' else arch}))
                pack = source / 'pack'; pack.mkdir()
                names = ['tapis-workflow-pack', 'vllm-pack']
                for name in names:
                    write_pack(pack / (name + '.zip'), name)
                (pack / 'pack-source.json').write_text(json.dumps(dict(repository=metadata['pack']['repository'],
                    commit='b' * 40 if case == 'pack_source' else 'a' * 40, packs={n: '1.0.0' for n in names})))
                if case == 'valid':
                    inputs.prepare(source, root / 'dist', 'c' * 40)
                    result = json.loads((root / 'dist/revisions.json').read_text())
                    self.assertEqual(result['gateway']['version'], '1')
                    self.assertTrue((root / 'dist/binaries/flexserv-capability-server-arm64.bin').exists())
                else:
                    with self.assertRaises(ValueError):
                        inputs.prepare(source, root / 'dist', 'c' * 40)
