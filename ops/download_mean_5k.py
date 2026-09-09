#!/usr/bin/env python3
"""Fetch the explicitly public 5k model, verify every byte, preserve partials."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
RUNTIME = Path('/workspace/artifacts/research_reports/con/paper_con1_mean_from5k_20260909/runtime')
ANCHOR = Path('/workspace/artifacts/checkpoints/pi05_libero_paper_con1_direct_delta_stage1/all4_direct_delta_late2_fixed005_5k/4999')


def hashes(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            with path.open('rb') as handle:
                result[str(path.relative_to(root))] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return result


def main():
    manifest = json.loads((REPO/'ops/mean_5k_manifest.json').read_text())
    expected = manifest['sha256']
    if len(expected) != 33:
        raise RuntimeError('Expected the verified complete checkpoint with 33 files')
    report = RUNTIME/'hf_5k_complete.json'
    if report.exists():
        value = json.loads(report.read_text())
        if not value.get('passed') or value['revision'] != manifest['revision'] or hashes(ANCHOR) != expected:
            raise RuntimeError('Existing checkpoint/report no longer matches the pinned source')
        return
    destination = Path('/workspace/artifacts/downloads/con1_5k_hf_20260909')
    env = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_OFFLINE='0',
               UV_NO_CACHE='0', UV_CACHE_DIR='/workspace/.uv-cache', HF_HUB_DOWNLOAD_TIMEOUT='120')
    command = ['/usr/local/bin/uv','tool','run','--from','huggingface-hub==1.30.0','hf',
               'download',manifest['repo_id'],'--revision',manifest['revision'],
               '--include',manifest['prefix']+'/*','--local-dir',str(destination),'--max-workers','8']
    subprocess.run(command, env=env, check=True)
    source = destination/manifest['prefix']
    if hashes(source) != expected:
        raise RuntimeError('Downloaded checkpoint does not match the original 33 SHA-256 hashes')
    backup = None
    if ANCHOR.exists() and hashes(ANCHOR) != expected:
        backup = ANCHOR.with_name('4999.partial-direct-20260909')
        if backup.exists():
            raise RuntimeError('Refusing to overwrite an earlier partial checkpoint backup')
        ANCHOR.rename(backup)
    if not ANCHOR.exists():
        ANCHOR.parent.mkdir(parents=True, exist_ok=True)
        source.rename(ANCHOR)
    value = {'passed':True,'repo_id':manifest['repo_id'],'revision':manifest['revision'],
             'path':str(ANCHOR),'files':expected,'unix_time':time.time(),
             'preserved_partial':str(backup) if backup else None}
    temporary = report.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(report)
    print(json.dumps({'event':'public_5k_verified','files':len(expected),'revision':manifest['revision']}),flush=True)


if __name__ == '__main__':
    main()
