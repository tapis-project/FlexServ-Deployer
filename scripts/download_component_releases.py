#!/usr/bin/env python3
"""Resolve published component assets once and verify bytes before assembly."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

COMPONENTS = {
    'gateway': ('tapis-project/FlexServ', 'feature/1.5.0'),
    'fcs': ('tapis-project/FlexServ-FCS', 'feature/new_fcs'),
    'pack': ('tapis-project/FlexServ-FCS-Pack', 'feature/new_fcs'),
}


def api(path, paginate=False):
    """Use gh's authenticated API and redirect handling without exposing tokens."""
    command = ['gh', 'api', path]
    if paginate:
        command += ['--paginate', '--slurp']
    return json.loads(subprocess.check_output(command, text=True))


def resolve(component, tag='', request=api):
    """Select a published release whose exact tag commit belongs to its source branch."""
    repository, branch = COMPONENTS[component]
    base = 'repos/' + repository
    branch_commit = request(base + '/commits/' + quote(branch, safe=''))['sha']
    if tag:
        candidates = [request(base + '/releases/tags/' + quote(tag, safe=''))]
    else:
        candidates = [release for page in request(base + '/releases?per_page=100', True)
                      for release in page]
        candidates.sort(key=lambda item: item.get('published_at') or '', reverse=True)
    for release in candidates:
        if release.get('draft') or not release.get('published_at'):
            continue
        if 'release-assets.sha256' not in {a['name'] for a in release['assets']}:
            continue
        commit = request(base + '/commits/' + quote(release['tag_name'], safe=''))['sha']
        if not re.fullmatch(r'[a-f0-9]{40}', commit):
            raise ValueError('Invalid release commit: ' + component)
        comparison = request(base + '/compare/' + commit + '...' + branch_commit)
        if comparison['status'] not in ('ahead', 'identical'):
            continue
        return dict(repository=repository, branch=branch, tag=release['tag_name'],
                    commit=commit, release_id=release['id'], release=release)
    raise ValueError('No published component release on approved branch: ' + component)


def fetch_asset(repository, asset, destination):
    """Download by immutable asset ID; gh handles private release redirects."""
    with destination.open('wb') as output:
        subprocess.run(['gh', 'api', 'repos/' + repository + '/releases/assets/' + str(asset['id']),
                        '-H', 'Accept: application/octet-stream'], stdout=output, check=True)


def download(resolved, output, fetch=fetch_asset):
    """Require both API and full-inventory SHA-256 checks, with flat safe names."""
    assets = resolved['release']['assets']
    names = [asset['name'] for asset in assets]
    if len(names) != len(set(names)) or 'release-assets.sha256' not in names:
        raise ValueError('Ambiguous or missing release inventory')
    if not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,200}', name) for name in names):
        raise ValueError('Unsafe release asset name')
    if sum(a['size'] for a in assets) > 1073741824:
        raise ValueError('Release exceeds 1 GiB download limit')
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for asset in assets:
        destination = output / asset['name']
        fetch(resolved['repository'], asset, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if destination.stat().st_size != asset['size'] or asset.get('digest') != 'sha256:' + digest:
            raise ValueError('Release asset digest/size mismatch: ' + asset['name'])
        hashes[asset['name']] = digest
    expected = {}
    for line in (output / 'release-assets.sha256').read_text().splitlines():
        match = re.fullmatch(r'([a-f0-9]{64})  ([A-Za-z0-9][A-Za-z0-9._-]{0,200})', line)
        if not match or match[2] in expected:
            raise ValueError('Invalid or duplicate release inventory line')
        expected[match[2]] = match[1]
    actual = {name: digest for name, digest in hashes.items() if name != 'release-assets.sha256'}
    if actual != expected:
        raise ValueError('Release inventory does not match the complete asset set')
    return dict((k, v) for k, v in resolved.items() if k != 'release'), hashes


def main():
    """Resolve all inputs before downloading; emit source and asset identity JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    for component in COMPONENTS:
        parser.add_argument('--' + component + '-tag', default='')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    chosen = {name: resolve(name, getattr(args, name + '_tag')) for name in COMPONENTS}
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for name, selected in chosen.items():
        identity, hashes = download(selected, args.output / name)
        metadata[name] = dict(identity, assets=hashes)
        print(name + ': ' + identity['tag'] + ' @ ' + identity['commit'])
    (args.output / 'resolved.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
