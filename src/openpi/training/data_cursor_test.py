import itertools
import random
import unittest

import numpy as np
import torch

from openpi.training.data_cursor import replay_batches


class RandomSampleDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 31

    def __getitem__(self, index):
        return torch.tensor([index, random.random(), np.random.random(), torch.rand(()).item()], dtype=torch.float64)


def batches():
    # Match the production loader's seeded shuffle, spawn, persistence and
    # drop-last behavior. Random transforms exercise worker RNG restoration.
    loader = torch.utils.data.DataLoader(
        RandomSampleDataset(), batch_size=4, shuffle=True, num_workers=2,
        multiprocessing_context="spawn", persistent_workers=True, drop_last=True,
        generator=torch.Generator().manual_seed(42),
    )
    while True:
        yield from loader


class DataCursorTest(unittest.TestCase):
    def test_already_fetched_first_batch(self):
        for consumed in (0, 1, 7, 1001):
            with self.subTest(consumed=consumed):
                stream = itertools.count()
                current = replay_batches(stream, next(stream), consumed)
                self.assertEqual(current, consumed)
                self.assertEqual(next(stream), consumed + 1)

    def test_invalid_or_exhausted_cursor(self):
        for count in (-1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                replay_batches(iter(()), None, count)
        with self.assertRaisesRegex(RuntimeError, "batch 1/2"):
            replay_batches(iter(()), None, 2)

    def test_seeded_worker_rng_across_epoch_boundary(self):
        # Seven batches per epoch: checkpoint after nine updates is in epoch 2.
        continuous = batches()
        restarted = batches()
        try:
            expected = list(itertools.islice(continuous, 13))
            current = replay_batches(restarted, next(restarted), 9)
            torch.testing.assert_close(current, expected[9], rtol=0, atol=0)
            for batch in expected[10:]:
                torch.testing.assert_close(next(restarted), batch, rtol=0, atol=0)
        finally:
            continuous.close()
            restarted.close()


if __name__ == "__main__":
    unittest.main()
