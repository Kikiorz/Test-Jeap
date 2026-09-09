"""CPU-only contract tests: no model or GPU allocation."""
import json

import numpy as np
import pytest

import cache_con1_features as cache


def test_atomic_array_keeps_teacher_bytes(tmp_path):
    z = np.arange(40,dtype=np.float16).reshape(5,8)
    source=tmp_path/'source.npy'
    np.save(source,z)
    dest=tmp_path/'dest.npy'
    cache.atomic_npy(dest,np.load(source))
    assert cache.digest(source)==cache.digest(dest)


def test_metadata_identity_detects_change(tmp_path):
    (tmp_path/'meta').mkdir()
    for name in ('info.json','episodes.jsonl','tasks.jsonl'):
        (tmp_path/'meta'/name).write_text('{}')
    first=cache.source_identity(tmp_path)
    (tmp_path/'meta/tasks.jsonl').write_text('{"task":"different"}')
    assert cache.source_identity(tmp_path)!=first


def test_frame_tail_uses_same_anchor_no_episode_crossing():
    from openpi.con1.data import anchored_example
    r=np.zeros((4,2,3));z=np.arange(20).reshape(4,5)
    example=anchored_example(r,z,2,10)
    assert example['valid'].sum()==1
    np.testing.assert_array_equal(example['anchor'],z[2])
    np.testing.assert_array_equal(example['future_target'][0],z[3])
    np.testing.assert_array_equal(example['future_target'][1],z[2])
    assert set(example)=={'r_tokens','anchor','future_target','valid'}
