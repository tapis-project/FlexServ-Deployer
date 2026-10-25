#!/usr/bin/env python3
"""Assemble only declared runtime files; never execute downloaded source hooks."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import zipfile


def assemble(source, binaries, pack, output, revisions):
    """Build a deterministic Tapis ZIP and external archive checksum."""
    files = {'tapisjob.manifest': (source / 'tapisjob.manifest').read_bytes()}
    files.update({p.relative_to(source).as_posix(): p.read_bytes()
                  for prefix in ('app', 'scripts') for p in sorted((source / prefix).rglob('*')) if p.is_file()})
    expected={f'{name}-{arch}.bin' for name in ('flexserv-gateway','flexserv-capability-server') for arch in ('amd64','arm64')}
    for name in sorted(expected):
        path=binaries/name
        if not path.is_file(): raise ValueError('Missing architecture binary: '+name)
        files['bin/'+name]=path.read_bytes()
    with zipfile.ZipFile(pack) as archive:
        if not {'manifest.json','checksums.sha256'} <= set(archive.namelist()): raise ValueError('Pack root is not v0.13')
    files['pack/flexserv-capability-pack.zip']=pack.read_bytes()
    metadata=dict(schema_version=1,components=revisions,platforms={
        'amd64':{'platform':'linux-x86_64','target':'x86_64-unknown-linux-gnu','glibc_min':'2.28'},
        'arm64':{'platform':'linux-aarch64','target':'aarch64-unknown-linux-gnu','glibc_min':'2.28'}},
        rust_toolchain='1.99.1',pack_sha256=hashlib.sha256(pack.read_bytes()).hexdigest())
    files['BUILD.json']=(json.dumps(metadata,indent=2)+'\n').encode()
    files['checksums.sha256']=''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n' for name,data in sorted(files.items())).encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            entry=zipfile.ZipInfo(name,(1980,1,1,0,0,0));entry.create_system=3
            entry.external_attr=(stat.S_IFREG | (0o755 if name.endswith(('.sh','.bin')) else 0o644))<<16
            entry.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(entry,data)
    output.with_name(output.name+'.sha256').write_text(f'{hashlib.sha256(output.read_bytes()).hexdigest()}  {output.name}\n')


def main():
    """Consume binaries built from the already resolved immutable source revisions."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','binaries','pack','output','revisions'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    assemble(args.source,args.binaries,args.pack,args.output,json.loads(args.revisions.read_text()))


if __name__=='__main__': main()
