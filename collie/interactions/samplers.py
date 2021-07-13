import math
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from collie.interactions.datasets import HDF5Interactions, Interactions


class ApproximateNegativeSampler(torch.utils.data.sampler.Sampler):
    """
    Custom ``Sampler`` for bulk-sampling approximate negative items in ``Interactions`` data.

    Parameters
    ----------
    interactions: Interactions
    batch_size: int
        Number of samples per batch to load
    shuffle: bool
        Whether to shuffle the order of batches returned or not. This is especially useful for
        training data to ensure the model does not overfit to a specific order of data
    seed: int
        Seed for shuffling if ``shuffle is True``

    """
    def __init__(self,
                 interactions: Interactions,
                 batch_size: int = 1024,
                 shuffle: bool = False,
                 seed: Optional[int] = None):
        self.interactions = interactions
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.iteration_order = np.arange(len(self.interactions))

        np.random.seed(self.seed)

    def __iter__(self) -> 'ApproximateNegativeSampler':
        """Setup iteration through ``ApproximateNegativeSamplingInteractionsDataLoader`` data."""
        if self.shuffle:
            np.random.shuffle(self.iteration_order)

        # reset pointer
        self._pointer = 0

        return self

    def __next__(self) -> np.array:
        """Get the indices for the next batch of data."""
        if self._pointer >= len(self.interactions):
            raise StopIteration

        idxs = self.iteration_order[self._pointer:(self._pointer + self.batch_size)]

        self._pointer += self.batch_size

        return idxs

    def __len__(self) -> int:
        """Number of batches returned in the sampler."""
        return math.ceil(len(self.interactions) / self.batch_size)


class HDF5Sampler(torch.utils.data.sampler.Sampler):
    """
    Custom ``Sampler`` for HDF5 data, with each sampled item being a start index and a batch size
    to use in ``HDF5Interactions.__getitem__``.

    Parameters
    ----------
    hdf5_interactions: HDF5Interactions
    batch_size: int
        Number of samples per batch to load
    shuffle: bool
        Whether to shuffle the order of batches returned or not. This is especially useful for
        training data to ensure the model does not overfit to a specific order of data. Note that
        this will not perform a true shuffle of the data, but will instead shuffle the order of
        batches. While this is an approximation of true sampling, it allows us a greater speed up
        during model training for a negligible effect on model performance
    seed: int
        Seed for shuffling if ``shuffle is True``

    """
    def __init__(self,
                 hdf5_interactions: HDF5Interactions,
                 batch_size: int = 1024,
                 shuffle: bool = False,
                 seed: Optional[int] = None):
        self.hdf5_interactions = hdf5_interactions
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.data_to_iterate_through = [
            (start_idx, self.batch_size)
            for start_idx in range(0, len(self.hdf5_interactions), self.batch_size)
        ]

        random.seed(self.seed)

    def __iter__(self) -> 'HDF5Sampler':
        """Setup iteration through ``HDF5Sampler`` data."""
        if self.shuffle:
            random.shuffle(self.data_to_iterate_through)

        # reset pointer
        self._pointer = 0

        return self

    def __next__(self) -> List[Tuple[int]]:
        """Get the indices for the next batch of data."""
        if self._pointer >= len(self.data_to_iterate_through):
            raise StopIteration

        idx = self.data_to_iterate_through[self._pointer]

        self._pointer += 1

        return idx

    def __len__(self) -> int:
        """Number of batches returned in the sampler."""
        return math.ceil(len(self.hdf5_interactions) / self.batch_size)
