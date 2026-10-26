#!/usr/bin/env python3
"""Verify component identities and merge Pack data without executing source hooks."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import zipfile


def read_pack(path):
    """Read a bounded regular-file ZIP and validate its complete payload inventory."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 4096 or sum(e.file_size for e in entries) > 67108864:
            raise ValueError('Pack exceeds size/file limits')
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate Pack member')
        for entry in entries:
            parts = PurePosixPath(entry.filename)
            mode = entry.external_attr >> 16
            if (not re.fullmatch(r'[A-Za-z0-9_./-]+', entry.filename)
                    or parts.is_absolute() or any(p in ('', '.', '..') for p in entry.filename.split('/'))
                    or entry.is_dir() or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))):
                raise ValueError('Unsafe Pack member: ' + entry.filename)
        data = {entry.filename: archive.read(entry) for entry in entries}
    if not {'manifest.json', 'checksums.sha256'} <= data.keys():
        raise ValueError('Missing Pack root')
    expected = {}
    for line in data['checksums.sha256'].decode().splitlines():
        match = re.fullmatch(r'([a-f0-9]{64})  (.+)', line)
        if not match or match[2] in expected:
            raise ValueError('Malformed Pack inventory')
        expected[match[2]] = match[1]
    actual = {name: hashlib.sha256(content).hexdigest() for name, content in data.items()
              if name != 'checksums.sha256'}
    if actual != expected:
        raise ValueError('Pack payload checksum mismatch')
    manifest = json.loads(data['manifest.json'])
    if manifest['schema_version'] != 'flexserv.fcs-pack.v1alpha2':
        raise ValueError('Only v0.13 Pack definitions are supported')
    return manifest, data


def merge_packs(paths, output):
    """Deduplicate identical helpers and reject all identity or content conflicts."""
    files, toolsets, locations, base = {}, {}, {}, None
    for path in paths:
        manifest, payload = read_pack(path)
        shared = {k: v for k, v in manifest.items() if k not in ('name', 'toolsets')}
        if base is not None and shared != base:
            raise ValueError('Pack manifest versions/targets conflict')
        base = shared
        for item in manifest['toolsets']:
            key = (item['name'], item['version'])
            if key in toolsets and toolsets[key] != item:
                raise ValueError('Conflicting toolset identity')
            if item['path'] in locations and locations[item['path']] != key:
                raise ValueError('Conflicting toolset location')
            toolsets[key], locations[item['path']] = item, key
        for name, content in payload.items():
            if name in ('manifest.json', 'checksums.sha256'):
                continue
            if name in files and files[name] != content:
                raise ValueError('Conflicting shared Pack file: ' + name)
            files[name] = content
    manifest = dict(base, name='flexserv-capability-pack', toolsets=[toolsets[k] for k in sorted(toolsets)])
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    files['checksums.sha256'] = ''.join(f'{hashlib.sha256(content).hexdigest()}  {name}\n'
                                      for name, content in sorted(files.items())).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w') as archive:
        for name, content in sorted(files.items()):
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, content)
    return manifest


def prepare(downloads, output, deployer_commit):
    """Cross-check source sidecars before making files eligible for final assembly."""
    metadata = json.loads((downloads / 'resolved.json').read_text())
    binaries = output / 'binaries'
    binaries.mkdir(parents=True, exist_ok=False)
    for component, prefix in (('gateway', 'flexserv-gateway'), ('fcs', 'flexserv-capability-server')):
        directory, selected = downloads / component, metadata[component]
        versions, abi, toolchains = set(), {}, {}
        for arch, platform in (('amd64', 'linux-x86_64'), ('arm64', 'linux-aarch64')):
            stem = directory / (prefix + '-' + platform)
            if Path(str(stem) + '-source.txt').read_text().strip() != selected['commit']:
                raise ValueError('Binary source revision mismatch: ' + component)
            versions.add(Path(str(stem) + '-version.txt').read_text().strip())
            abi[platform] = json.loads(Path(str(stem) + '-abi.json').read_text())
            toolchains[platform] = Path(str(stem) + '-rust.txt').read_text().strip()
            binary = directory / (prefix + '-' + arch + '.bin')
            if abi[platform]['architecture'] != arch:
                raise ValueError('Binary ABI architecture mismatch: ' + component)
            abi[platform]['sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
            shutil.copyfile(binary, binaries / binary.name)
        if len(versions) != 1 or not next(iter(versions)):
            raise ValueError('Architecture versions disagree: ' + component)
        selected.update(version=versions.pop(), abi=abi, toolchains=toolchains)
    source = json.loads((downloads / 'pack/pack-source.json').read_text())
    if source['commit'] != metadata['pack']['commit'] or source['repository'] != metadata['pack']['repository']:
        raise ValueError('Pack source revision mismatch')
    paths = [downloads / 'pack' / (name + '.zip') for name in ('tapis-workflow-pack', 'vllm-pack')]
    for path in paths:
        manifest, _ = read_pack(path)
        if manifest['name'] != path.stem or source['packs'].get(path.stem) != manifest['version']:
            raise ValueError('Pack release name/version mismatch')
    merged = merge_packs(paths, output / 'flexserv-capability-pack.zip')
    metadata['pack'].update(version=merged['version'], packs=source['packs'])
    metadata['deployer'] = dict(repository='tapis-project/FlexServ-Deployer', commit=deployer_commit)
    (output / 'revisions.json').write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    """Consume only verified release files; preserve provenance for BUILD.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    prepare(args.downloads, args.output, commit)


if __name__ == '__main__':
    main()
