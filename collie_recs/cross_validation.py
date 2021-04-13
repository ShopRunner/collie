from collections import defaultdict
import functools
import multiprocessing as mp
import operator
from typing import Any, Iterable, Optional, Tuple

from joblib import delayed, Parallel
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

from collie_recs.interactions import Interactions
from collie_recs.utils import get_random_seed


def random_split(interactions: Interactions,
                 val_p: float = 0.0,
                 test_p: float = 0.2,
                 seed: Optional[int] = None,
                 **kwargs) -> Tuple[Interactions, ...]:
    """
    Randomly split interactions into training, validation, and testing sets.

    This split does NOT guarantee that every user will be represented in both the training and
    testing datasets. While much faster than ``stratified_split``, it is not the most representative
    data split because of this.

    Note that this function is not supported for ``HDF5Interactions`` objects, since this data split
    implementation requires all data to fit in memory. A data split for large datasets should be
    done using a big data processing technology, like Spark.

    Parameters
    ----------
    interactions: collie_recs.interactions.Interactions
    val_p: float
        Proportion of data used for validation
    test_p: float
        Proportion of data used for testing
    seed: int
        Random seed for splits
    kwargs: keyword arguments
        Ignored, included only for compatibility with ``stratified_split`` API

    Returns
    ----------
    train_interactions: collie_recs.interactions.Interactions
        Training data of size proportional to ``1 - val_p - test_p``
    validate_interactions: collie_recs.interactions.Interactions
        Validation data of size proportional to ``val_p``, returned only if ``val_p > 0``
    test_interactions: collie_recs.interactions.Interactions
        Testing data of size proportional to ``test_p``

    Examples
    -------------
    .. code-block:: python

        >>> interactions = Interactions(...)
        >>> len(interactions)
        100000
        >>> train, test = random_split(interactions)
        >>> len(train), len(test)
        (80000, 20000)

    """
    if len(kwargs) > 0 and [kwargs_key for kwargs_key in kwargs] != ['processes']:
        raise ValueError(f'Unexpected ``kwargs``: {kwargs}')

    _validate_val_p_and_test_p(val_p=val_p, test_p=test_p)

    if seed is None:
        seed = get_random_seed()

    np.random.seed(seed)

    shuffle_indices = np.arange(len(interactions))
    np.random.shuffle(shuffle_indices)

    interactions = _subset_interactions(interactions=interactions,
                                        idxs=shuffle_indices)

    validate_and_test_p = val_p + test_p
    validate_cutoff = int((1.0 - validate_and_test_p) * len(interactions))
    test_cutoff = int((1.0 - test_p) * len(interactions))

    train_idxs = np.arange(validate_cutoff)
    validate_idxs = np.arange(validate_cutoff, test_cutoff)
    test_idxs = np.arange(test_cutoff, len(interactions))

    train_interactions = _subset_interactions(interactions=interactions,
                                              idxs=train_idxs)
    test_interactions = _subset_interactions(interactions=interactions,
                                             idxs=test_idxs)

    if val_p > 0:
        validate_interactions = _subset_interactions(interactions=interactions,
                                                     idxs=validate_idxs)

        return train_interactions, validate_interactions, test_interactions
    else:
        return train_interactions, test_interactions


def _subset_interactions(interactions: Interactions, idxs: Iterable[int]) -> Interactions:
    idxs = np.array(idxs)

    coo_mat = coo_matrix(
        (interactions.mat.data[idxs], (interactions.mat.row[idxs],
                                       interactions.mat.col[idxs])),
        shape=(interactions.num_users, interactions.num_items)
    )

    return Interactions(
        mat=coo_mat,
        num_negative_samples=interactions.num_negative_samples,
        allow_missing_ids=True,
        num_users=interactions.num_users,
        num_items=interactions.num_items,
        check_num_negative_samples_is_valid=False,
        max_number_of_samples_to_consider=interactions.max_number_of_samples_to_consider,
        seed=interactions.seed,
    )


def stratified_split(interactions: Interactions,
                     val_p: float = 0.0,
                     test_p: float = 0.2,
                     processes: int = mp.cpu_count(),
                     seed: Optional[int] = None) -> Tuple[Interactions, ...]:
    """
    Split an ``Interactions`` instance into train, validate, and test datasets in a stratified
    manner such that each user appears at least once in each of the datasets.

    This split guarantees that every user will be represented in the training, validation, and
    testing datasets given they appear in ``interactions`` at least three times. If ``val_p ==
    0``, they will appear in the training and testing datasets given they appear at least two times.
    If a user appears fewer than this number of times, a ``ValueError`` will
    be raised. To filter users with fewer than ``n`` points out, use
    ``collie_recs.utils.remove_users_with_fewer_than_n_interactions``.

    This is computationally more complex than ``random_split``, but produces a more representative
    data split. Note that when ``val_p > 0``, the algorithm will perform the data split twice,
    once to create the test set and another to create the validation set, essentially doubling the
    computational time.

    Note that this function is not supported for ``HDF5Interactions`` objects, since this data split
    implementation requires all data to fit in memory. A data split for large datasets should be
    done using a big data processing technology, like Spark.

    Parameters
    -------------
    interactions: collie_recs.interactions.Interactions
        ``Interactions`` instance containing the data to split
    val_p: float
        Proportion of data used for validation
    test_p: float
        Proportion of data used for testing
    processes: int
        Number of CPUs to use for parallelization
    seed: int
        Random seed for splits

    Returns
    -------------
    train_interactions: collie_recs.interactions.Interactions
        Training data of size proportional to ``1 - val_p - test_p``
    validate_interactions: collie_recs.interactions.Interactions
        Validation data of size proportional to ``val_p``, returned only if ``val_p > 0``
    test_interactions: collie_recs.interactions.Interactions
        Testing data of size proportional to ``test_p``

    Examples
    -------------
    .. code-block:: python

        >>> interactions = Interactions(...)
        >>> len(interactions)
        100000
        >>> train, test = stratified_split(interactions)
        >>> len(train), len(test)
        (80000, 20000)

    """
    _validate_val_p_and_test_p(val_p=val_p, test_p=test_p)

    if seed is None:
        seed = get_random_seed()

    train, test = _stratified_split(interactions=interactions,
                                    test_p=test_p,
                                    processes=processes,
                                    seed=seed)

    if val_p > 0:
        train, validate = _stratified_split(interactions=train,
                                            test_p=val_p / (1 - test_p),
                                            processes=processes,
                                            seed=seed)

        return train, validate, test
    else:
        return train, test


def _stratified_split(interactions: Interactions,
                      test_p: float,
                      processes: int,
                      seed: int) -> Tuple[Interactions, Interactions]:
    users = interactions.mat.row
    unique_users = set(users)

    # while we should be able to run ``np.where(users == user)[0]`` to find all items each user
    # interacted with, by building up a dictionary to get these values instead, we can achieve the
    # same result in O(N) complexity rather than O(M * N), a nice timesave to have when working with
    # larger datasets
    all_idxs_for_users_dict = defaultdict(list)
    for idx, user in enumerate(users):
        all_idxs_for_users_dict[user].append(idx)

    # run the function below in parallel for each user
    # by setting the seed to ``seed + user``, we get a balance between reproducability and actual
    # randomness so users with the same number of interactions are not split the exact same way
    test_idxs = Parallel(n_jobs=processes)(
        delayed(_stratified_split_parallel_worker)(all_idxs_for_users_dict[user],
                                                   test_p,
                                                   seed + user)
        for user in unique_users
    )

    # reduce the list of lists down to a 1-d list
    test_idxs = functools.reduce(operator.iconcat, test_idxs, [])
    # find all indices not in test set - they are now train
    train_idxs = list(set(range(len(users))) - set(test_idxs))

    train_interactions = _subset_interactions(interactions=interactions,
                                              idxs=train_idxs)
    test_interactions = _subset_interactions(interactions=interactions,
                                             idxs=test_idxs)

    return train_interactions, test_interactions


def _stratified_split_parallel_worker(idxs_to_split: Iterable[Any],
                                      test_p: float, seed: int) -> np.array:
    _, test_idxs = train_test_split(idxs_to_split,
                                    test_size=test_p,
                                    random_state=seed,
                                    shuffle=True,
                                    stratify=np.ones_like(idxs_to_split))

    return test_idxs


def _validate_val_p_and_test_p(val_p: float, test_p: float):
    validate_and_test_p = val_p + test_p

    if val_p >= 1 or val_p < 0:
        raise ValueError('``val_p`` must be in the range [0, 1).')
    if test_p >= 1 or test_p < 0:
        raise ValueError('``val_p`` must be in the range [0, 1).')
    if validate_and_test_p >= 1 or validate_and_test_p <= 0:
        raise ValueError('The sum of ``val_p`` and ``test_p`` must be in the range (0, 1).')
