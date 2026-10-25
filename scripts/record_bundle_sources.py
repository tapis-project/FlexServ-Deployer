#!/usr/bin/env python3
"""Verify immutable cross-repository build identities before bundle publication."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binaries',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); data={}
for component,prefix,repo in [('gateway','flexserv-gateway','zhangwei217245/FlexServ'),('fcs','flexserv-capability-server','tapis-project/FlexServ-FCS')]:
    expected=os.environ[component.upper()+'_SHA']
    abi={};versions=set()
    for platform in ['linux-x86_64','linux-aarch64']:
        # Component artifacts have distinct source filenames to survive merging.
        source=a.binaries/(prefix+'-'+platform+'-source.txt')
        if source.read_text().strip()!=expected: raise SystemExit('Binary source revision mismatch: '+component)
        versions.add((a.binaries/(prefix+'-'+platform+'-version.txt')).read_text().strip())
        abi[platform]=json.loads((a.binaries/(prefix+'-'+platform+'-abi.json')).read_text())
    if len(versions)!=1: raise SystemExit('Architecture version mismatch: '+component)
    data[component]={'repository':repo,'commit':expected,'version':versions.pop(),'abi':abi}
data['pack']={'repository':'tapis-project/FlexServ-FCS-Pack','commit':os.environ['PACK_SHA'],'version':json.loads(Path('selected-pack/pack/manifest.json').read_text())['version']}
data['deployer']={'repository':os.environ.get('GITHUB_REPOSITORY','tapis-project/FlexServ-Deployer'),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
a.output.write_text(json.dumps(data,indent=2)+'\n')
