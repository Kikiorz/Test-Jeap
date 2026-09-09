#!/usr/bin/env python3
"""Install only the public allowlist on this host; reject unknown local edits."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

SOURCE = Path(__file__).resolve().parents[1]
DESTINATION = Path('/workspace/ts_JEPA_con')
RUNTIME = Path('/workspace/artifacts/research_reports/con/paper_con1_mean_from5k_20260909/runtime')
# Exactly the superseded preparation files written by this task, not user edits.
LEGACY = {
    'ops/prepare_mean_con1.py':('7fad6cb217adb6a681040f00bdc6de6c4c024a89880009611a92d2c89f7828ce',
                                'b9363a6ec550b8e84fb3923d92f3289e5c9b49bfe2ce048ac15db03adea46eeb'),
    'ops/supervisor/con1-mean-preparation.conf':('42c5ec2575b93b397e4d144d1dbc9f9fac812cddc86d4a8199506dd8a0c36bd0',),
    'ops/install_mean_public_code.py':('8f16cf0403e0ba5e9f1ef2373eb492f27ffcc8526e8e3bea4625a50f4709c798',),
    'ops/mean_publication_files.txt':('57d17b17a10397fca43c387662cf401f41f4bda164df4f07124e6dd9431c7bd5',),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if SOURCE == DESTINATION:
        raise RuntimeError('Run from the fresh GitHub checkout, not the dirty destination')
    names = (SOURCE/'ops/mean_publication_files.txt').read_text().splitlines()
    revision = subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip():
        raise RuntimeError('GitHub source checkout is not clean')
    expected = {}
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe allowlisted path')
        expected[name] = sha(SOURCE/name)
        current = DESTINATION/name
        if current.exists() and sha(current) not in (expected[name], *LEGACY.get(name, ())):
            raise RuntimeError('Preserving unexpected local modification: '+name)
    preserved = []
    for name in names:
        current = DESTINATION/name
        if current.exists() and sha(current) == expected[name]:
            continue
        if current.exists():
            backup = RUNTIME/'pre-public-code-backup'/revision/name
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists() and sha(backup) != sha(current):
                raise RuntimeError('Earlier backup differs: '+name)
            shutil.copy2(current, backup)
            preserved.append(str(backup))
        current.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE/name, current)
    actual = {name:sha(DESTINATION/name) for name in names}
    if actual != expected:
        raise RuntimeError('Installed code hashes differ')
    value = {'passed':True,'revision':revision,'source':'https://github.com/Kikiorz/Test-Jeap',
             'source_sha256':actual,'unix_time':time.time(),'preserved_files':preserved,
             'note':'Allowlisted files verified; unrelated dirty files and git HEAD intentionally preserved.'}
    report = RUNTIME/'github_mean_code_verified.json'
    temporary = report.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(report)
    print(json.dumps({'passed':True,'revision':revision,'verified_files':len(actual)}))


if __name__ == '__main__':
    main()
