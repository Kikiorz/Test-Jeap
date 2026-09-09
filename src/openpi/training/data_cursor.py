"""Replay a deterministic loader when a checkpoint has no serialized cursor."""

from collections.abc import Iterator
import logging
import time
from typing import TypeVar

Batch = TypeVar("Batch")


def replay_batches(data_iter: Iterator[Batch], first_batch: Batch, consumed_batches: int) -> Batch:
    """Return batch ``consumed_batches``, given an already-fetched batch zero.

    Re-execute the original loader, including worker-side transforms, rather
    than seeking just the sampler. Callers must preserve dataset, seed, batch
    size, worker count, transforms and loader implementation across the restart.
    This does not make nondeterministic transforms deterministic and is not a
    claim of bitwise-identical model updates across different GPU runtimes.
    """
    if isinstance(consumed_batches, bool) or not isinstance(consumed_batches, int) or consumed_batches < 0:
        raise ValueError("consumed_batches must be a nonnegative integer")
    started = time.monotonic()
    logging.info("Data cursor replay: skipping %d consumed batches", consumed_batches)
    batch = first_batch
    for count in range(1, consumed_batches + 1):
        try:
            batch = next(data_iter)
        except StopIteration as exc:
            raise RuntimeError(f"Loader ended while replaying batch {count}/{consumed_batches}") from exc
        if count % 100 == 0:
            logging.info("Data cursor replay: %d/%d", count, consumed_batches)
    logging.info("Data cursor replay complete: next batch index %d in %.2fs", consumed_batches, time.monotonic() - started)
    return batch
