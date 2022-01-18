import textwrap
from typing import Iterable, Optional, Union

import numpy as np
from scipy.sparse import coo_matrix
import torch

from collie.interactions.datasets import (BaseInteractions,
                                          ExplicitInteractions,
                                          HDF5Interactions,
                                          Interactions)
from collie.interactions.samplers import ApproximateNegativeSampler, HDF5Sampler


class BaseInteractionsDataLoader(torch.utils.data.DataLoader):
    """
    A base class acting as a wrapper around a ``torch.utils.data.DataLoader`` for
    ``Interactions``-type datasets. This class should only be inherited from and not used for model
    training.

    Parameters
    ----------
    interactions: Interactions or HDF5Interactions
    **kwargs: keyword arguments
        Keyword arguments passed into ``torch.utils.data.DataLoader.__init__``:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

    Original ``torch.utils.data.DataLoader`` docstring as follows:
    ########
    """
    def __init__(self,
                 interactions: Union[Interactions, HDF5Interactions] = None,
                 **kwargs):

        super().__init__(
            dataset=interactions,
            **kwargs,
        )

        self.interactions = interactions

    __doc__ += torch.utils.data.DataLoader.__doc__

    @property
    def num_users(self) -> int:
        """Number of users in ``interactions``."""
        return self.interactions.num_users

    @property
    def num_items(self) -> int:
        """Number of items in ``interactions``."""
        return self.interactions.num_items

    @property
    def num_negative_samples(self) -> int:
        """Number of negative samples in ``interactions``."""
        return self.interactions.num_negative_samples

    @property
    def num_interactions(self) -> int:
        """Number of interactions in ``interactions``."""
        return self.interactions.num_interactions

    @property
    def mat(self) -> coo_matrix:
        """Sparse COO matrix of ``interactions``."""
        return self.interactions.mat


class InteractionsDataLoader(BaseInteractionsDataLoader):
    """
    A light wrapper around a ``torch.utils.data.DataLoader`` for ``Interactions``-type datasets.

    For implicit data, batches will be created one-point-at-a-time using exact negative sampling
    (unless configured not to in ``interactions``), which is optimal when datasets are smaller
    (< 1M+ interactions) and model training speed is not a concern. This is the default
    ``DataLoader`` for ``Interactions`` datasets.

    For explicit data, negative sampling is not used, but batches will still be created
    one-point-at-a-time.

    Parameters
    ----------
    interactions: BaseInteractions
        If not provided, an ``Interactions`` object will be created with ``mat`` or all of
        ``users``, ``items``, and ``ratings``
    mat: scipy.sparse.coo_matrix or numpy.array, 2-dimensional
        If ``interactions is None``, will be used instead of ``users``, ``items``, and ``ratings``
        arguments to create an ``Interactions`` object
    users: Iterable[int], 1-d
        If ``interactions is None and mat is None``, array of user IDs, starting at 0
    items: Iterable[int], 1-d
        If ``interactions is None and mat is None``, array of corresponding item IDs to ``users``,
        starting at 0
    ratings: Iterable[int], 1-d
        If ``interactions is None and mat is None``, array of corresponding ratings to both
        ``users`` and ``items``. If ``None``, will default to each user in ``user`` interacting with
        an item with a rating value of 1
    batch_size: int
        Number of samples per batch to load
    shuffle: bool
        Whether to shuffle the order of data returned or not. This is especially useful for training
        data to ensure the model does not overfit to a specific order of data
    **kwargs: keyword arguments
        Relevant keyword arguments will be passed into ``Interactions`` object creation, if
        ``interactions is None`` and the keyword argument matches one of
        ``Interactions.__init__.__code__.co_varnames``. All other keyword arguments will be passed
        into ``torch.utils.data.DataLoader``:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

    Attributes
    ----------
    interactions: Interactions (default) or ExplicitInteractions

    Original ``torch.utils.data.DataLoader`` docstring as follows:
    ########
    """
    def __init__(self,
                 interactions: BaseInteractions = None,
                 mat: Optional[Union[coo_matrix, np.array]] = None,
                 users: Optional[Iterable[int]] = None,
                 items: Optional[Iterable[int]] = None,
                 ratings: Optional[Iterable[int]] = None,
                 batch_size: int = 1024,
                 shuffle: bool = False,
                 **kwargs):
        if interactions is None:
            # find all kwargs in the ``__init__`` for a ``Interactions`` object
            interactions_only_kwargs = {
                k: v for k, v in kwargs.items() if k in Interactions.__init__.__code__.co_varnames
            }
            # find all kwargs not in the ``__init__`` for a ``Interactions`` object OR all kwargs
            # that are used in the ``__init__`` for a ``torch.utils.data.DataLoader`` object
            kwargs = {
                k: v for k, v in kwargs.items()
                if k not in Interactions.__init__.__code__.co_varnames
                or k in torch.utils.data.DataLoader.__init__.__code__.co_varnames
            }

            interactions = Interactions(mat=mat,
                                        users=users,
                                        items=items,
                                        ratings=ratings,
                                        **interactions_only_kwargs)

        super().__init__(
            interactions=interactions,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )

        self.shuffle = shuffle

    __doc__ += torch.utils.data.DataLoader.__doc__

    def __repr__(self) -> str:
        """String representation of ``InteractionsDataLoader`` class."""
        if hasattr(self.interactions, 'num_negative_samples'):
            extra_repr_str = (
                f'{self.num_negative_samples} negative samples per implicit interaction in'
            )
        else:
            extra_repr_str = 'explicit,'

        return textwrap.dedent(
            f'''
            InteractionsDataLoader object with {self.num_interactions} interactions between
            {self.num_users} users and {self.num_items} items, returning
            {extra_repr_str}
            {'shuffled' if self.shuffle else 'non-shuffled'} batches of size {self.batch_size}.
            '''
        ).replace('\n', ' ').strip()


class ApproximateNegativeSamplingInteractionsDataLoader(BaseInteractionsDataLoader):
    """
    A computationally more efficient ``DataLoader`` for ``Interactions`` data using approximate
    negative sampling for negative items.

    This DataLoader groups ``__getitem__`` calls together into a single operation, which
    dramatically speeds up a traditional DataLoader's process of calling ``__getitem__`` one index
    at a time, then concatenating them together before returning. In an effort to batch operations
    together, all negative samples returned will be approximate, meaning this does not check if a
    user has previously interacted with the item. With a sufficient number of interactions (1M+), we
    have found a speed increase of 2x at the cost of a 1% reduction in MAP @ 10 performance
    compared to ``InteractionsDataLoader``.

    For greater efficiency, we disable automated batching by setting the DataLoader's
    ``batch_size`` attribute to ``None``. Thus,
    to access the "true" batch size that the sampler uses, access
    ``ApproximateNegativeSamplingInteractionsDataLoader.approximate_negative_sampler.batch_size``.

    Parameters
    ----------
    interactions: Interactions
        If not provided, an ``Interactions`` object will be created with ``mat`` or all of
        ``users``, ``items``, and ``ratings`` with ``max_number_of_samples_to_consider=0``
    mat: scipy.sparse.coo_matrix or numpy.array, 2-dimensional
        If ``interactions is None``, will be used instead of ``users``, ``items``, and ``ratings``
        arguments to create an ``Interactions`` object
    users: Iterable[int], 1-d
        If ``interactions is None and mat is None``, array of user IDs, starting at 0
    items: Iterable[int], 1-d
        If ``interactions is None and mat is None``, array of corresponding item IDs to ``users``,
        starting at 0
    ratings: Iterable[int], 1-d
        If ``interactions is None and mat is None``, array of corresponding ratings to both
        ``users`` and ``items``. If ``None``, will default to each user in ``user`` interacting with
        an item with a rating value of 1
    batch_size: int
        Number of samples per batch to load
    shuffle: bool
        Whether to shuffle the order of data returned or not. This is especially useful for training
        data to ensure the model does not overfit to a specific order of data
    **kwargs: keyword arguments
        Relevant keyword arguments will be passed into ``Interactions`` object creation, if
        ``interactions is None`` and the keyword argument matches one of
        ``Interactions.__init__.__code__.co_varnames``. All other keyword arguments will be passed
        into ``torch.utils.data.DataLoader``:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

    Attributes
    ----------
    interactions: Interactions

    Original ``torch.utils.data.DataLoader`` docstring as follows:
    ########
    """
    def __init__(self,
                 interactions: Interactions = None,
                 mat: Optional[Union[coo_matrix, np.array]] = None,
                 users: Optional[Iterable[int]] = None,
                 items: Optional[Iterable[int]] = None,
                 ratings: Optional[Iterable[int]] = None,
                 batch_size: int = 1024,
                 shuffle: bool = False,
                 **kwargs):
        if isinstance(interactions, ExplicitInteractions):
            raise ValueError(
                '``ApproximateNegativeSamplingInteractionsDataLoader`` does not support explicit'
                ' data types!'
            )

        if interactions is None:
            interactions_only_kwargs = {
                k: v for k, v in kwargs.items()
                if k in Interactions.__init__.__code__.co_varnames
            }
            kwargs = {
                k: v for k, v in kwargs.items()
                if k not in Interactions.__init__.__code__.co_varnames
                or k in torch.utils.data.DataLoader.__init__.__code__.co_varnames
            }

            interactions = Interactions(mat=mat,
                                        users=users,
                                        items=items,
                                        ratings=ratings,
                                        max_number_of_samples_to_consider=0,
                                        **interactions_only_kwargs)
        else:
            # we need ``max_number_of_samples_to_consider`` to be 0 in order to do approximate
            # negative sampling
            interactions.max_number_of_samples_to_consider = 0

        approximate_negative_sampler = ApproximateNegativeSampler(interactions=interactions,
                                                                  batch_size=batch_size,
                                                                  shuffle=shuffle,
                                                                  seed=interactions.seed)

        super().__init__(
            interactions=interactions,
            sampler=approximate_negative_sampler,
            batch_size=None,  # Disable automated batching
            **kwargs,
        )

        self.approximate_negative_sampler = approximate_negative_sampler
        self.shuffle = shuffle

    __doc__ += torch.utils.data.DataLoader.__doc__

    def __repr__(self) -> str:
        """String representation of ``ApproximateNegativeSamplingInteractionsDataLoader`` class."""
        return textwrap.dedent(
            f'''
            ApproximateNegativeSamplingInteractionsDataLoader object with {self.num_interactions}
            interactions between {self.num_users} users and {self.num_items} items, returning
            {self.num_negative_samples} negative samples per implicit interaction in
            {'shuffled' if self.shuffle else 'non-shuffled'} batches of size
            {self.approximate_negative_sampler.batch_size}.
            '''
        ).replace('\n', ' ').strip()


class HDF5InteractionsDataLoader(BaseInteractionsDataLoader):
    """
    A light wrapper around a ``torch.utils.data.DataLoader`` for HDF5 data, with behavior very
    similar to ``ApproximateNegativeSamplingInteractionsDataLoader``.

    If not provided, a ``HDF5Interactions`` dataset will be created as the data for the
    ``DataLoader``. A custom sampler, ``HDF5Sampler``, will also be instantiated for the
    ``DataLoader`` to use that allows sampling in batches that make for faster HDF5 data reads from
    disk.

    While similar to a standard ``DataLoader``, note that when ``shuffle is True``, this will only
    shuffle the order of batches and the data within batches to still make for efficient reading
    of HDF5 data from disk, rather than shuffling across the entire dataset.

    For greater efficiency, we disable automated batching by setting the DataLoader's
    ``batch_size`` attribute to ``None``. Thus,
    to access the "true" batch size that the sampler uses, access
    ``HDF5InteractionsDataLoader.hdf5_sampler.batch_size``.

    Parameters
    ----------
    hdf5_interactions: HDF5Interactions
        If provided, will override input argument for ``hdf5_path``
    hdf5_path: str
        If ``hdf5_interactions is None``, the path to the HDF5 dataset
    batch_size: int
        Number of samples per batch to load
    shuffle: bool
        Whether to shuffle the order of batches returned or not. This is especially useful for
        training data to ensure the model does not overfit to a specific order of data. Note that
        this will not perform a true shuffle of the data, but shuffle the order of batches. While
        this is an approximation of true sampling, it allows us a greater speed up during model
        training for a negligible effect on model performance
    **kwargs: keyword arguments
        Relevant keyword arguments will be passed into ``HDF5Interactions`` object creation, if
        ``hdf5_interactions is None`` and the keyword argument matches one of
        ``HDF5Interactions.__init__.__code__.co_varnames``. All other keyword arguments will be
        passed into ``torch.utils.data.DataLoader``:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

    Original ``torch.utils.data.DataLoader`` docstring as follows:
    ########
    """
    def __init__(self,
                 hdf5_interactions: HDF5Interactions = None,
                 hdf5_path: Optional[str] = None,
                 batch_size: int = 1024,
                 shuffle: bool = False,
                 **kwargs):
        if hdf5_interactions is None:
            # find all kwargs in the ``__init__`` for a ``HDF5Interactions`` object
            interactions_only_kwargs = {
                k: v for k, v in kwargs.items()
                if k in HDF5Interactions.__init__.__code__.co_varnames
            }
            # find all kwargs not in the ``__init__`` for a ``HDF5Interactions`` object OR all
            # kwargs that are used in the ``__init__`` for a ``torch.utils.data.DataLoader`` object
            kwargs = {
                k: v for k, v in kwargs.items()
                if k not in HDF5Interactions.__init__.__code__.co_varnames
                or k in torch.utils.data.DataLoader.__init__.__code__.co_varnames
            }

            hdf5_interactions = HDF5Interactions(hdf5_path=hdf5_path,
                                                 **interactions_only_kwargs)

        hdf5_sampler = HDF5Sampler(hdf5_interactions=hdf5_interactions,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   seed=hdf5_interactions.seed)

        super().__init__(
            interactions=hdf5_interactions,
            sampler=hdf5_sampler,
            batch_size=None,  # Disable automated batching
            **kwargs,
        )

        self.hdf5_sampler = hdf5_sampler
        self.hdf5_path = hdf5_path
        self.shuffle = shuffle

    __doc__ += torch.utils.data.DataLoader.__doc__

    @property
    def mat(self) -> None:
        """``mat`` attribute is not possible to access in ``HDF5InteractionsDataLoader``."""
        raise AttributeError('``HDF5InteractionsDataLoader`` cannot support ``mat`` attribute since'
                             ' data is read in from disk dynamically.')

    def __repr__(self) -> str:
        """String representation of ``HDF5InteractionsDataLoader`` class."""
        return textwrap.dedent(
            f'''
            HDF5InteractionsDataLoader object with {self.interactions.num_interactions}
            interactions between {self.interactions.num_users} users and
            {self.interactions.num_items} items, returning {self.num_negative_samples} negative
            samples per implicit interaction in {'shuffled' if self.shuffle else 'non-shuffled'}
            batches of size {self.hdf5_sampler.batch_size}.
            '''
        ).replace('\n', ' ').strip()
