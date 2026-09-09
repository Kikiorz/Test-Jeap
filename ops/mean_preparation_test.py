"""Regression tests for the preparation failure and restart-safe audits."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

import prepare_mean_con1 as preparation
from smoke_mujoco_egl import SCENE_XML


def test_renderer_is_a_script_with_valid_xml():
    source = (Path(__file__).parent/'smoke_mujoco_egl.py').read_text()
    compile(source, 'smoke_mujoco_egl.py', 'exec')
    assert ET.fromstring(SCENE_XML).find('worldbody/geom').attrib == {'type': 'sphere', 'size': '.1'}


def test_audit_restart_preserves_previous_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, 'ROOT', tmp_path)
    (tmp_path/'runtime').mkdir()
    previous = tmp_path/'runtime/direct_target_audit.json'
    previous.write_text('{"passed":true,"old_evidence":true}')
    original = previous.read_bytes()
    def run(name, args):
        path = Path(args[-1])
        assert not path.exists()
        path.write_text(json.dumps({'passed': True}))
    monkeypatch.setattr(preparation, 'run', run)
    first = preparation.fresh_audit('direct_target_audit', 'audit.py')
    second = preparation.fresh_audit('direct_target_audit', 'audit.py')
    assert first != second and first.exists() and second.exists()
    assert previous.read_bytes() == original


def test_failed_audit_does_not_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, 'ROOT', tmp_path)
    (tmp_path/'runtime').mkdir()
    monkeypatch.setattr(preparation, 'run', lambda name, args: Path(args[-1]).write_text('{"passed":false}'))
    with pytest.raises(RuntimeError, match='Audit did not pass'):
        preparation.fresh_audit('direct_target_audit', 'audit.py')
