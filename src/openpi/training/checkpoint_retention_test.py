"""Exercise real Orbax async retention on tiny temporary checkpoints."""

import numpy as np
import pytest

from openpi.training.checkpoints import initialize_checkpoint_dir
from openpi.training.config import paper_con1_config


@pytest.mark.parametrize("protected,expected", [
    ((), [5, 10, 14]),
    ((4, 9, 14), [4, 5, 9, 10, 14]),
])
def test_explicit_ids_preserve_evaluation_boundaries_and_periodic_checkpoints(tmp_path, protected, expected):
    folder = tmp_path / "checkpoints"
    manager, resumed = initialize_checkpoint_dir(folder, keep_period=5, keep_steps=protected,
                                                 overwrite=False, resume=False)
    assert not resumed
    try:
        for step in (1, 4, 5, 6, 9, 10, 11, 14):
            manager.save(step, {"params": {"value": np.array([step], dtype=np.int32)}})
            manager.wait_until_finished()
        assert sorted(manager.all_steps()) == expected
        assert all((folder / str(step) / "params/_METADATA").is_file() for step in expected)
    finally:
        manager.close()
    manager, resumed = initialize_checkpoint_dir(folder, keep_period=5, keep_steps=protected,
                                                 overwrite=False, resume=True)
    try:
        assert resumed and sorted(manager.all_steps()) == expected
    finally:
        manager.close()


def test_direct_training_protects_the_exact_evaluated_checkpoint_ids(tmp_path):
    fixed = paper_con1_config(joint=False, late2=True, direct_delta=True)
    joint = paper_con1_config(joint=True, late2=True, direct_delta=True)
    assert fixed.keep_steps == (4999,)
    assert joint.keep_steps == (4999, 9999, 14999)
    assert paper_con1_config(joint=True, late2=True).keep_steps == ()
    with pytest.raises(ValueError, match="nonnegative integers"):
        initialize_checkpoint_dir(tmp_path / "invalid", keep_period=5, keep_steps=(-1,),
                                  overwrite=False, resume=False)
