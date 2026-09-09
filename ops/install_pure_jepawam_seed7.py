#!/usr/bin/env python3
"""Install only the reviewed seed-isolated wrapper and its new service/test."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    source = Path(__file__).resolve().parents[1]
    target = Path('/workspace/ts_JEPA_con')
    root = Path('/workspace/artifacts/research_reports/con/pure_jepawam_60k_robot_seed7_20260909')
    if source==target or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Use the separate clean GitHub checkout')
    revision = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    files = ['ops/run_pure_jepawam_full_plus.py','ops/pure_jepawam_seed7_test.py',
             'ops/supervisor/pure-jepawam-60k-robot-seed7.conf']
    copies = [(source/f,target/f) for f in files]
    copies.append((source/files[-1],Path('/etc/supervisor/conf.d/pure-jepawam-60k-robot-seed7.conf')))
    for src,dst in copies:
        if dst.exists() and dst.read_bytes()!=src.read_bytes():
            if src.name!='run_pure_jepawam_full_plus.py':
                raise RuntimeError('Unexpected existing file: '+str(dst))
            old = subprocess.check_output(['git','-C',str(source),'show',
                'a97b0ff8aa8a4230158aea5921c8758a3a47a868:'+str(src.relative_to(source))])
            if dst.read_bytes()!=old:
                raise RuntimeError('Preserving unexpected wrapper modification')
    root.mkdir(parents=True,exist_ok=True)
    for src,dst in copies:
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists() and dst.read_bytes()!=src.read_bytes():
            backup = root/'code_backups'/revision/str(dst).lstrip('/')
            backup.parent.mkdir(parents=True,exist_ok=True)
            if backup.exists() and backup.read_bytes()!=dst.read_bytes():
                raise RuntimeError('Earlier backup differs')
            shutil.copy2(dst,backup)
            shutil.copy2(src,dst)
        elif not dst.exists():
            shutil.copy2(src,dst)
        if dst.read_bytes()!=src.read_bytes():
            raise RuntimeError('Install verification failed')
    report = dict(revision=revision,sha256={str(dst):hashlib.sha256(dst.read_bytes()).hexdigest() for _,dst in copies})
    (root/'code_install.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
