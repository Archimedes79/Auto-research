"""A training loader that `fit` can resume exactly: the order of a pass over the data is a function of (seed, epoch), not of the global random state.

`fit` counts the micro-batches it takes from the data as (epoch, pos), saves that in every checkpoint (extra["data"]) and, on a resume, asks the data for the same pass from the
same place: `batches.iter_from(epoch, pos)`. Data without that method (a list, a plain DataLoader) starts a pass afresh on a resume (README.md)."""
import numpy as np
import torch


class EpochSampler(torch.utils.data.Sampler):
    """the indices 0 .. n - 1 in a random order per pass: pass `epoch` is a numpy permutation from a generator seeded with the pair (seed, epoch) (not torch's: its CPU generator keeps only
    32 bits of a seed, so two pairs could share a stream), the same in every process and after every resume. `start(epoch, skip)` chooses the pass that the next iter() makes and drops
    its first `skip` indices; every iter() leaves the sampler at the next pass"""

    def __init__(self, n, seed):
        super().__init__()
        self.n, self.seed, self.epoch, self.skip = n, seed, 0, 0

    def start(self, epoch, skip=0):
        self.epoch, self.skip = epoch, skip

    def __iter__(self):
        order = np.random.default_rng([self.seed, self.epoch]).permutation(self.n)[self.skip:]
        self.epoch, self.skip = self.epoch + 1, 0
        return iter(order.tolist())

    def __len__(self):
        return self.n


class EpochLoader(torch.utils.data.DataLoader):
    """a shuffling DataLoader (every DataLoader argument but `shuffle`, `sampler` and `generator`) with the position protocol of `fit`. It owns a generator, which draws the workers'
    base seed at every iter(): building a pass does not touch the global random streams, so a resumed run draws what the uninterrupted one drew. Randomness inside a dataset's
    __getitem__ in a worker process is not made resumable here."""

    def __init__(self, dataset, batch_size, seed, **kw):
        super().__init__(dataset, batch_size=batch_size, sampler=EpochSampler(len(dataset), seed), generator=torch.Generator().manual_seed(seed), **kw)

    def position_id(self):
        """what a saved position is valid for: the number of samples (the order of a pass) and the batch size (the batches in it). `fit` saves it with the position"""
        return {"n": len(self.dataset), "batch_size": self.batch_size}

    def iter_from(self, epoch, pos):
        """the iterator over pass `epoch` without its first `pos` batches"""
        self.sampler.start(epoch, pos * self.batch_size)
        return iter(self)
