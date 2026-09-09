"""Deterministic sample-index cursor for paper Con1, without decoding skipped images.

Worker-side transforms for this protocol are deterministic; random image
augmentation happens later in JAX, seeded by the completed optimizer step.
This sampler is opt-in and does not change historical checkpoint ordering.
"""

import numpy as np


class SeekableBatchSampler:
    def __init__(self, length, batch_size, *, seed, shuffle):
        if length < batch_size or batch_size < 1:
            raise ValueError("Dataset must contain at least one full batch")
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.batches_per_epoch = self.length // self.batch_size
        self.start_batch = 0

    def seek(self, completed_batches):
        if isinstance(completed_batches, bool) or not isinstance(completed_batches, int) or completed_batches < 0:
            raise ValueError("Completed batch count must be a nonnegative integer")
        self.start_batch = completed_batches

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        epoch, batch = divmod(self.start_batch, self.batches_per_epoch)
        while True:
            if self.shuffle:
                indices = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(self.length)
            else:
                indices = np.arange(self.length)
            for index in range(batch, self.batches_per_epoch):
                start = index * self.batch_size
                yield indices[start:start + self.batch_size].tolist()
            epoch += 1
            batch = 0
