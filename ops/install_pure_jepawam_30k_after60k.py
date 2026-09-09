#!/usr/bin/env python3
"""Install new 30k continuation files only; never alter the running 60k job."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    source=Path(__file__).resolve().parents[1]
    dest=Path('/workspace/ts_JEPA_con')
    root=Path('/workspace/artifacts/research_reports/con/pure_jepawam_30k_robot_seed7_20260909')
    if source==dest or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Use the clean separate GitHub checkout')
    files=['ops/run_pure_jepawam_30k_after60k.py','ops/pure_jepawam_30k_after60k_test.py',
           'ops/supervisor/pure-jepawam-30k-after60k-seed7.conf']
    pairs=[(source/f,dest/f) for f in files]
    pairs.append((source/files[-1],Path('/etc/supervisor/conf.d/pure-jepawam-30k-after60k-seed7.conf')))
    for src,dst in pairs:
        if dst.exists() and dst.read_bytes()!=src.read_bytes():
            raise RuntimeError('Preserving unexpected existing file: '+str(dst))
    root.mkdir(parents=True,exist_ok=True)
    for src,dst in pairs:
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            shutil.copy2(src,dst)
        if dst.read_bytes()!=src.read_bytes():
            raise RuntimeError('Install verification failed')
    report=dict(revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                sha256={str(dst):hashlib.sha256(dst.read_bytes()).hexdigest() for _,dst in pairs})
    (root/'code_install.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
