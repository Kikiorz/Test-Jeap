#!/usr/bin/env python3
"""Add only new Robot sweep files from the authorized clean GitHub checkout."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    source = Path(__file__).resolve().parents[1]
    target = Path('/workspace/ts_JEPA_con')
    root = Path('/workspace/artifacts/research_reports/con/pure_jepawam_robot_checkpoints_20260909')
    if source==target or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Use the separate, clean GitHub checkout')
    files = ['ops/run_pure_jepawam_robot_sweep.py','ops/pure_jepawam_robot_sweep_test.py',
             'ops/supervisor/pure-jepawam-robot-sweep.conf']
    copies = [(source/f,target/f) for f in files]
    copies.append((source/files[-1],Path('/etc/supervisor/conf.d/pure-jepawam-robot-sweep.conf')))
    for src,dst in copies:
        if dst.exists() and dst.read_bytes()!=src.read_bytes():
            raise RuntimeError('Preserving unexpected existing file: '+str(dst))
    root.mkdir(parents=True,exist_ok=True)
    for src,dst in copies:
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            shutil.copy2(src,dst)
        if dst.read_bytes()!=src.read_bytes():
            raise RuntimeError('Install verification failed')
    report = dict(revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                  source='https://github.com/Kikiorz/Test-Jeap',
                  sha256={str(dst):hashlib.sha256(dst.read_bytes()).hexdigest() for _,dst in copies})
    (root/'code_install.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
