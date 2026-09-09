import itertools

import numpy as np
import pytest

from openpi.training.seekable_sampler import SeekableBatchSampler


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("cursor", [0, 1, 6, 7, 21, 50])
def test_cursor_matches_uninterrupted_batches(shuffle, cursor):
    sampler = SeekableBatchSampler(31, 4, seed=42, shuffle=shuffle)
    uninterrupted = list(itertools.islice(iter(sampler), cursor, cursor + 4))
    sampler.seek(cursor)
    assert list(itertools.islice(iter(sampler), 4)) == uninterrupted


def test_epoch_is_a_permutation_with_only_drop_last_tail_missing():
    sampler = SeekableBatchSampler(31, 4, seed=42, shuffle=True)
    batches = list(itertools.islice(iter(sampler), 14))
    assert len(set(np.array(batches[:7]).ravel())) == 28
    assert len(set(np.array(batches[7:]).ravel())) == 28
    assert batches[:7] != batches[7:]
    with pytest.raises(ValueError):
        sampler.seek(-1)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_torch_loader_reseeks_without_replaying_prior_batches(num_workers):
    from openpi.training.data_loader import TorchDataLoader

    rows = [{"value": np.int32(i)} for i in range(31)]
    loader = TorchDataLoader(rows, 4, seed=42, shuffle=True, seekable_batches=True,
                            framework="pytorch", num_batches=14, num_workers=num_workers)
    expected = [batch["value"].tolist() for batch in loader]
    loader.set_start_batch(5)
    actual = [batch["value"].tolist() for batch in itertools.islice(iter(loader), 9)]
    assert actual == expected[5:14]
