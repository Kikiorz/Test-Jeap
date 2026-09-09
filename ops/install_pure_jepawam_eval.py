#!/usr/bin/env python3
"""Install new baseline-evaluation files from a clean GitHub checkout only."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

FILES = ["ops/run_con1_dynamic_eval.py","ops/con1_dynamic_worker.py","ops/con1_dynamic_queue.py","ops/run_con1_full_plus.py","ops/con1_eval_dashboard.py","ops/con1_full_plus_test.py","ops/run_pure_jepawam_full_plus.py","ops/pure_jepawam_full_plus_test.py","ops/supervisor/pure-jepawam-full-plus-4x16.conf"]
SOURCE = Path(__file__).resolve().parents[1]
DEST = Path('/workspace/ts_JEPA_con')
ROOT = Path('/workspace/artifacts/research_reports/con/pure_jepawam_full_plus_4x16_20260909')


def main():
    if SOURCE == DEST:
        raise RuntimeError('Use the separate GitHub checkout')
    if subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Source checkout is not clean')
    revision = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    copies = [(SOURCE/name, DEST/name) for name in FILES]
    conf = 'ops/supervisor/pure-jepawam-full-plus-4x16.conf'
    copies.append((SOURCE/conf, Path('/etc/supervisor/conf.d/pure-jepawam-full-plus-4x16.conf')))
    for src, dst in copies:
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            prior = subprocess.check_output(['git', '-C', str(SOURCE), 'show',
                     '32f08c4bda808c82a1f83cedaafa145e0a34a07c:'+str(src.relative_to(SOURCE))])
            if dst.read_bytes() != prior:
                raise RuntimeError('Preserving unexpected local file: '+str(dst))
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            backup = ROOT/'code_backups'/revision/str(dst).lstrip('/')
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists() and backup.read_bytes() != dst.read_bytes():
                raise RuntimeError('Earlier code backup differs: '+str(backup))
            shutil.copy2(dst, backup)
            shutil.copy2(src, dst)
        elif not dst.exists():
            shutil.copy2(src, dst)
        if dst.read_bytes() != src.read_bytes():
            raise RuntimeError('Installed file differs: '+str(dst))
    ROOT.mkdir(parents=True, exist_ok=True)
    report = dict(passed=True, revision=revision, source='https://github.com/Kikiorz/Test-Jeap',
                  installed_sha256={str(dst):hashlib.sha256(dst.read_bytes()).hexdigest() for _,dst in copies})
    (ROOT/('code_install_'+revision+'.json')).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, revision=revision, installed=len(copies))))


if __name__ == '__main__':
    main()
