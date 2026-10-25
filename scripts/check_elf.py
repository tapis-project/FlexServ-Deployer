#!/usr/bin/env python3
"""Reject non-portable control-plane executables before publishing a release."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def inspect(binary, architecture):
    """Check the actual ELF architecture, loader and versioned dynamic ABI requirements."""
    header=subprocess.check_output(['readelf','-h',str(binary)],text=True)
    expected={'amd64':'Advanced Micro Devices X86-64','arm64':'AArch64'}[architecture]
    if expected not in header: raise ValueError('ELF architecture mismatch')
    program=subprocess.check_output(['readelf','-l',str(binary)],text=True)
    loader={'amd64':'/lib64/ld-linux-x86-64.so.2','arm64':'/lib/ld-linux-aarch64.so.1'}[architecture]
    if loader not in program: raise ValueError('Unexpected ELF interpreter')
    versions=subprocess.check_output(['readelf','--version-info',str(binary)],text=True)
    glibc=[tuple(map(int,v.split('.'))) for v in re.findall(r'\bGLIBC_(\d+(?:\.\d+)+)',versions)]
    if any(v>(2,28) for v in glibc): raise ValueError('GLIBC requirement exceeds 2.28')
    cpp=[tuple(map(int,v.split('.'))) for v in re.findall(r'\bGLIBCXX_(\d+(?:\.\d+)+)',versions)]
    if any(v>(3,4,25) for v in cpp): raise ValueError('C++ ABI requirement exceeds GCC 8 runtime')
    dynamic=subprocess.check_output(['readelf','-d',str(binary)],text=True)
    libraries=re.findall(r'\(NEEDED\).*?\[(.*?)\]',dynamic)
    allowed={'libc.so.6','libm.so.6','libdl.so.2','librt.so.1','libpthread.so.0','libgcc_s.so.1','libstdc++.so.6','ld-linux-x86-64.so.2','ld-linux-aarch64.so.1'}
    if set(libraries)-allowed: raise ValueError('Undeclared dynamic dependency: '+str(set(libraries)-allowed))
    return dict(architecture=architecture,interpreter=loader,glibc_max='.'.join(map(str,max(glibc,default=(0,)))),libraries=libraries)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('binary',type=Path);p.add_argument('--arch',choices=['amd64','arm64'],required=True)
    a=p.parse_args();print(json.dumps(inspect(a.binary,a.arch),indent=2))
