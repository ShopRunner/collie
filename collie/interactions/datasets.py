from abc import ABCMeta, abstractmethod
import collections
import random
import textwrap
from typing import Any, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, dok_matrix
import torch
from tqdm.auto import tqdm

import collie


class BaseInteractions(torch.utils.data.Dataset, metaclass=ABCMeta):
    """
    PyTorch ``Dataset`` for implicit user-item interactions data.

    If ``mat`` is provided, the ``Interactions`` instance will act as a wrapper for a sparse matrix
    in COOrdinate format, typically looking like:

    * Users comprising the rows

    * Items comprising the columns

    * Ratings given by that user for that item comprising the elements of the matrix

    ``Interactions`` can be instantiated instead by passing in single arrays with corresponding
    user_ids, item_ids, and ratings (by default, set to 1 for implicit recommenders) values with
    the same functionality as a matrix. Note that with this approach, the number of users and items
    will be the maximum values in those two columns, respectively, and it is expected that all
    integers between 0 and the maximum ID should appear somewhere in the data.

    By default, exact negative sampling will be used during each ``__getitem__`` call. To use
    approximate negative sampling, set ``max_number_of_samples_to_consider = 0``. This will avoid
    building a positive item lookup dictionary during initialization.

    Parameters
    ----------
    mat: scipy.sparse.coo_matrix or numpy.array, 2-dimensional
        Interactions matrix, which, if provided, will be used instead of ``users``, ``items``, and
        ``ratings`` arguments
    users: Iterable[int], 1-d
        Array of user IDs, starting at 0
    items: Iterable[int], 1-d
        Array of corresponding item IDs to ``users``, starting at 0
    ratings: Iterable[int], 1-d
        Array of corresponding ratings to both ``users`` and ``items``. If ``None``, will default to
        each user in ``user`` interacting with an item with a rating value of 1
    num_negative_samples: int
        Number of negative samples to return with each ``__getitem__`` call
    allow_missing_ids: bool
        If ``False``, will check that both ``users`` and ``items`` contain each integer from 0 to
        the maximum value in the array. This check only applies when initializing an
        ``Interactions`` instance using 1-dimensional arrays ``users`` and ``items``
    remove_duplicate_user_item_pairs: bool
        Will check for and remove any duplicate user, item ID pairs from the ``Interactions`` matrix
        during initialization. Note that this will create a second sparse matrix held in memory
        to efficiently check, which could cause memory concerns for larger data. If you are sure
        that there are no duplicated, user, item ID pairs, set to ``False``
    num_users: int
        Number of users in the dataset. If ``num_users == 'infer'``, this will be set to the
        ``mat.shape[0]`` or ``max(users) + 1``, depending on the input
    num_items: int
        Number of items in the dataset. If ``num_items == 'infer'``, this will be set to the
        ``mat.shape[1]`` or ``max(items) + 1``, depending on the input
    check_num_negative_samples_is_valid: bool
        Check that ``num_negative_samples`` is less than the maximum number of items a user has
        interacted with. If it is not, then for all users who have fewer than
        ``num_negative_samples`` items not interacted with, a random sample including positive items
        will be returned as negative
    max_number_of_samples_to_consider: int
        Number of samples to try for a given user before returning an approximate negative sample.
        This should be greater than ``num_negative_samples``. If set to ``0``, approximate negative
        sampling will be used by default in ``__getitem__`` and a positive item lookup dictionary
        will NOT be built

    """
    def __init__(self,
                 mat: Optional[Union[coo_matrix, np.array]] = None,
                 users: Optional[Iterable[int]] = None,
                 items: Optional[Iterable[int]] = None,
                 ratings: Optional[Iterable[int]] = None,
                 allow_missing_ids: bool = False,
                 remove_duplicate_user_item_pairs: bool = True,
                 num_users: int = 'infer',
                 num_items: int = 'infer'):
        if mat is None:
            assert users is not None and items is not None, (
                'Either 1) ``mat`` or 2) both ``users`` or ``items`` must be non-null!'
            )

            if len(users) != len(items):
                raise ValueError('Lengths of ``users`` and ``items`` must be equal.')

            num_users = collie.utils._infer_num_if_needed_for_1d_array(num_users, users)
            num_items = collie.utils._infer_num_if_needed_for_1d_array(num_items, items)

            if allow_missing_ids is False:
                _check_array_contains_all_integers(array=users,
                                                   array_max_value=num_users,
                                                   array_name='users')
                _check_array_contains_all_integers(array=items,
                                                   array_max_value=num_items,
                                                   array_name='items')

            if ratings is not None:
                if len(users) != len(ratings):
                    raise ValueError(
                        'Length of ``ratings`` must be equal to lengths of ``users``, ``items``.'
                    )

            mat = collie.utils._create_sparse_ratings_matrix_helper(users=users,
                                                                    items=items,
                                                                    ratings=ratings,
                                                                    num_users=num_users,
                                                                    num_items=num_items)
        else:
            mat = coo_matrix(mat)

            if num_users == 'infer':
                num_users = mat.shape[0]
            if num_items == 'infer':
                num_items = mat.shape[1]

            if allow_missing_ids is False:
                _check_array_contains_all_integers(array=mat.row,
                                                   array_max_value=num_users,
                                                   array_name='mat.shape[0]')
                _check_array_contains_all_integers(array=mat.col,
                                                   array_max_value=num_items,
                                                   array_name='mat.shape[1]')

        if remove_duplicate_user_item_pairs:
            print('Checking for and removing duplicate user, item ID pairs...')

            # remove duplicate entires in the COO matrix
            dok_mat = dok_matrix((mat.shape), dtype=mat.dtype)
            dok_mat._update(zip(zip(mat.row, mat.col), mat.data))
            mat = dok_mat.tocoo()

            # trigger garbage collection early
            del dok_mat

        self.mat = mat
        self.allow_missing_ids = allow_missing_ids
        self.remove_duplicate_user_item_pairs = remove_duplicate_user_item_pairs
        self.num_users = num_users
        self.num_items = num_items

        self.num_interactions = self.mat.nnz
        self.min_rating = self.mat.data.min()
        self.max_rating = self.mat.data.max()

    @abstractmethod
    def __getitem__(self, index: Union[int, Iterable[int]]) -> (
        Union[Tuple[Tuple[int, int], np.array], Tuple[Tuple[np.array, np.array], np.array]]
    ):
        """Access item in the ``BaseInteractions`` instance."""
        pass

    def __len__(self) -> int:
        """Number of non-zero interactions in the ``BaseInteractions`` instance."""
        return self.num_interactions

    def todense(self) -> np.matrix:
        """Transforms ``BaseInteractions`` instance sparse matrix to np.matrix, 2-d."""
        return self.mat.todense()

    def toarray(self) -> np.array:
        """Transforms ``BaseInteractions`` instance sparse matrix to np.array, 2-d."""
        return self.mat.toarray()

    def head(self, n: int = 5) -> np.array:
        """Return the first ``n`` rows of the dense matrix as a np.array, 2-d."""
        n = self._prep_head_tail_n(n=n)
        return self.mat.tocsr()[range(n), :].toarray()

    def tail(self, n: int = 5) -> np.array:
        """Return the last ``n`` rows of the dense matrix as a np.array, 2-d."""
        n = self._prep_head_tail_n(n=n)
        return self.mat.tocsr()[range(-n, 0), :].toarray()

    def _prep_head_tail_n(self, n: int) -> int:
        """Ensure we don't run into an ``IndexError`` when using ``head`` or ``tail`` methods."""
        if n < 0:
            n = self.num_users + n
        if n > self.num_users:
            n = self.num_users

        return n


class Interactions(BaseInteractions):
    """
    PyTorch ``Dataset`` for implicit user-item interactions data.

    If ``mat`` is provided, the ``Interactions`` instance will act as a wrapper for a sparse matrix
    in COOrdinate format, typically looking like:

    * Users comprising the rows

    * Items comprising the columns

    * Ratings given by that user for that item comprising the elements of the matrix

    ``Interactions`` can be instantiated instead by passing in single arrays with corresponding
    user_ids, item_ids, and ratings (by default, set to 1 for implicit recommenders) values with
    the same functionality as a matrix. Note that with this approach, the number of users and items
    will be the maximum values in those two columns, respectively, and it is expected that all
    integers between 0 and the maximum ID should appear somewhere in the data.

    By default, exact negative sampling will be used during each ``__getitem__`` call. To use
    approximate negative sampling, set ``max_number_of_samples_to_consider = 0``. This will avoid
    building a positive item lookup dictionary during initialization.

    Unlike in ``ExplicitInteractions``, we rely on negative sampling for implicit data. Each
    ``__getitem__`` call will thus return a nested tuple containing user IDs, item IDs, and
    sampled negative item IDs. This nested vs. non-nested structure is key for the model to
    determine where it should be implicit or explicit. Use the table below for reference:

    .. list-table::
        :header-rows: 1

        * - ``__getitem__`` Format
          - Expected Meaning
          - Model Type
        * - ``((X, Y), Z)``
          - ``((user IDs, item IDs), negative item IDs)``
          - **Implicit**
        * - ``(X, Y, Z)``
          - ``(user IDs, item IDs, ratings)``
          - **Explicit**

    Parameters
    -------------
    mat: scipy.sparse.coo_matrix or numpy.array, 2-dimensional
        Interactions matrix, which, if provided, will be used instead of ``users``, ``items``, and
        ``ratings`` arguments
    users: Iterable[int], 1-d
        Array of user IDs, starting at 0
    items: Iterable[int], 1-d
        Array of corresponding item IDs to ``users``, starting at 0
    ratings: Iterable[int], 1-d
        Array of corresponding ratings to both ``users`` and ``items``. If ``None``, will default to
        each user in ``user`` interacting with an item with a rating value of 1
    num_negative_samples: int
        Number of negative samples to return with each ``__getitem__`` call
    allow_missing_ids: bool
        If ``False``, will check that both ``users`` and ``items`` contain each integer from 0 to
        the maximum value in the array. This check only applies when initializing an
        ``Interactions`` instance using 1-dimensional arrays ``users`` and ``items``
    remove_duplicate_user_item_pairs: bool
        Will check for and remove any duplicate user, item ID pairs from the ``Interactions`` matrix
        during initialization. Note that this will create a second sparse matrix held in memory
        to efficiently check, which could cause memory concerns for larger data. If you are sure
        that there are no duplicated, user, item ID pairs, set to ``False``
    num_users: int
        Number of users in the dataset. If ``num_users == 'infer'``, this will be set to the
        ``mat.shape[0]`` or ``max(users) + 1``, depending on the input
    num_items: int
        Number of items in the dataset. If ``num_items == 'infer'``, this will be set to the
        ``mat.shape[1]`` or ``max(items) + 1``, depending on the input
    check_num_negative_samples_is_valid: bool
        Check that ``num_negative_samples`` is less than the maximum number of items a user has
        interacted with. If it is not, then for all users who have fewer than
        ``num_negative_samples`` items not interacted with, a random sample including positive items
        will be returned as negative
    max_number_of_samples_to_consider: int
        Number of samples to try for a given user before returning an approximate negative sample.
        This should be greater than ``num_negative_samples``. If set to ``0``, approximate negative
        sampling will be used by default in ``__getitem__`` and a positive item lookup dictionary
        will NOT be built
    seed: int
        Seed for random sampling

    """
    def __init__(self,
                 mat: Optional[Union[coo_matrix, np.array]] = None,
                 users: Optional[Iterable[int]] = None,
                 items: Optional[Iterable[int]] = None,
                 ratings: Optional[Iterable[int]] = None,
                 num_negative_samples: int = 10,
                 allow_missing_ids: bool = False,
                 remove_duplicate_user_item_pairs: bool = True,
                 num_users: int = 'infer',
                 num_items: int = 'infer',
                 check_num_negative_samples_is_valid: bool = True,
                 max_number_of_samples_to_consider: int = 200,
                 seed: Optional[int] = None):
        if mat is None and ratings is not None and 0 in set(ratings):
            warnings.warn(
                '``ratings`` contain ``0``s, which are ignored for implicit data.'
                ' Filtering these rows out.'
            )
            indices_to_drop = [idx for idx, rating in enumerate(ratings) if rating == 0]

            users = _drop_array_values_by_idx(array=users, indices_to_drop=indices_to_drop)
            items = _drop_array_values_by_idx(array=items, indices_to_drop=indices_to_drop)
            ratings = _drop_array_values_by_idx(array=ratings, indices_to_drop=indices_to_drop)

        super().__init__(mat=mat,
                         users=users,
                         items=items,
                         ratings=ratings,
                         allow_missing_ids=allow_missing_ids,
                         remove_duplicate_user_item_pairs=remove_duplicate_user_item_pairs,
                         num_users=num_users,
                         num_items=num_items)

        if seed is None:
            seed = collie.utils.get_random_seed()

        self.num_negative_samples = num_negative_samples
        self.max_number_of_samples_to_consider = max_number_of_samples_to_consider
        self.check_num_negative_samples_is_valid = check_num_negative_samples_is_valid
        self.seed = seed

        random.seed(self.seed)

        assert self.num_negative_samples >= 1

        if (
            self.num_negative_samples >= self.max_number_of_samples_to_consider
            and self.max_number_of_samples_to_consider > 0
        ):
            # no warning for ``max_number_of_samples_to_consider==0`` since it is likely intentional
            warnings.warn(
                '``num_negative_samples > max_number_of_samples_to_consider``. Approximate negative'
                ' sampling will be used.'
            )

        # When an ``Interactions`` is instantiated with exact negative sampling, a
        # ``positive_items`` attribute is created, a ``set`` of the ``mat`` object that enables
        # fast, O(1), ``(row, col)`` lookup. When ``__getitem__`` is called, negative item IDs are
        # sampled one-at-a-time from all possible values in ``self.num_items``, we check if that
        # user ID, item ID pair is in ``self.positive_items``, and sample continuously until we
        # have a negative match or reach a limit of ``max_number_of_samples_to_consider`` tries
        if self.check_num_negative_samples_is_valid:
            print('Checking ``num_negative_samples`` is valid...')
            counter = collections.Counter(self.mat.row)
            max_number_of_items_interacted_with = counter.most_common(1)[0][1]
            print('Maximum number of items a user has interacted with: {}'.format(
                max_number_of_items_interacted_with
            ))

            del counter

            is_valid = (
                self.num_negative_samples
                < (self.num_items - max_number_of_items_interacted_with)
            )
            assert is_valid, '``num_negative_samples`` must be less than {}!'.format(
                (self.num_items - max_number_of_items_interacted_with)
            )

        self.positive_items = {}
        if self.max_number_of_samples_to_consider > 0:
            print('Generating positive items set...')
            self._generate_positive_item_set()

    def _generate_positive_item_set(self) -> None:
        """Build positive item dictionary lookup for exact negative sampling."""
        self.positive_items = set(zip(self.mat.row, self.mat.col))

    def __repr__(self) -> str:
        """String representation of ``Interactions`` class."""
        return textwrap.dedent(
            f'''
            Interactions object with {self.num_interactions} interactions between {self.num_users}
            users and {self.num_items} items, returning {self.num_negative_samples} negative
            samples per interaction.
            '''
        ).replace('\n', ' ').strip()

    def __getitem__(self, index: Union[int, Iterable[int]]) -> (
        Union[Tuple[Tuple[int, int], np.array], Tuple[Tuple[np.array, np.array], np.array]]
    ):
        """Access item in the ``Interactions`` instance, returning negative samples as well."""
        user_id = self.mat.row[index]
        item_id = self.mat.col[index]
        # rating = self.mat.data[index]  # not needed for any loss currently implemented

        negative_item_ids_array = self._negative_sample(user_id)

        return (user_id, item_id), negative_item_ids_array

    def _negative_sample(self, user_id: Union[int, np.array]) -> np.array:
        """Generate negative samples for a ``user_id``."""
        if self.max_number_of_samples_to_consider > 0:
            # if we are here, we are doing true negative sampling
            negative_item_ids_list = list()

            if not isinstance(user_id, collections.abc.Iterable):
                user_id = [user_id]

            for specific_user_id in user_id:
                # generate true negative samples for the ``user_id``
                samples_checked = 0
                temp_negative_item_ids_list = list()

                while len(temp_negative_item_ids_list) < self.num_negative_samples:
                    negative_item_id = random.choice(range(self.num_items))
                    # we have a negative sample, make sure the user has not interacted with it
                    # before, else we resample and try again
                    while (
                        (specific_user_id, negative_item_id) in self.positive_items
                        or negative_item_id in temp_negative_item_ids_list
                    ):
                        if samples_checked >= self.max_number_of_samples_to_consider:
                            num_samples_left_to_generate = (
                                self.num_negative_samples - len(temp_negative_item_ids_list) - 1
                            )
                            temp_negative_item_ids_list += random.choices(
                                range(self.num_items), k=num_samples_left_to_generate
                            )
                            break

                        negative_item_id = random.choice(range(self.num_items))
                        samples_checked += 1

                    temp_negative_item_ids_list.append(negative_item_id)

                negative_item_ids_list += [np.array(temp_negative_item_ids_list)]

            if len(user_id) > 1:
                negative_item_ids_array = np.stack(negative_item_ids_list)
            else:
                negative_item_ids_array = negative_item_ids_list[0]
        else:
            # if we are here, we are doing approximate negative sampling
            if isinstance(user_id, collections.abc.Iterable):
                size = (len(user_id), self.num_negative_samples)
            else:
                size = (self.num_negative_samples,)

            negative_item_ids_array = np.random.randint(
                low=0,
                high=self.num_items,
                size=size,
            )

        return negative_item_ids_array


class ExplicitInteractions(BaseInteractions):
    """
    PyTorch ``Dataset`` for explicit user-item interactions data.

    If ``mat`` is provided, the ``Interactions`` instance will act as a wrapper for a sparse matrix
    in COOrdinate format, typically looking like:

    * Users comprising the rows

    * Items comprising the columns

    * Ratings given by that user for that item comprising the elements of the matrix

    ``Interactions`` can be instantiated instead by passing in single arrays with corresponding
    user_ids, item_ids, and ratings values with the same functionality as a matrix. Note that with
    this approach, the number of users and items will be the maximum values in those two columns,
    respectively, and it is expected that all integers between 0 and the maximum ID should appear
    somewhere in the user or item ID data.

    Unlike in ``Interactions``, there is no need for negative sampling for explicit data. Each
    ``__getitem__`` call will thus return a single, non-nested tuple containing user IDs, item IDs,
    and ratings. This nested vs. non-nested structure is key for the model to determine where it
    should be implicit or explicit. Use the table below for reference:

    .. list-table::
        :header-rows: 1

        * - ``__getitem__`` Format
          - Expected Meaning
          - Model Type
        * - ``((X, Y), Z)``
          - ``((user IDs, item IDs), negative item IDs)``
          - **Implicit**
        * - ``(X, Y, Z)``
          - ``(user IDs, item IDs, ratings)``
          - **Explicit**

    Parameters
    -------------
    mat: scipy.sparse.coo_matrix or numpy.array, 2-dimensional
        Interactions matrix, which, if provided, will be used instead of ``users``, ``items``, and
        ``ratings`` arguments
    users: Iterable[int], 1-d
        Array of user IDs, starting at 0
    items: Iterable[int], 1-d
        Array of corresponding item IDs to ``users``, starting at 0
    ratings: Iterable[int], 1-d
        Array of corresponding ratings to both ``users`` and ``items``. If ``None``, will default to
        each user in ``user`` interacting with an item with a rating value of 1
    allow_missing_ids: bool
        If ``False``, will check that both ``users`` and ``items`` contain each integer from 0 to
        the maximum value in the array. This check only applies when initializing an
        ``ExplicitInteractions`` instance using 1-dimensional arrays ``users`` and ``items``
    remove_duplicate_user_item_pairs: bool
        Will check for and remove any duplicate user, item ID pairs from the
        ``ExplicitInteractions`` matrix during initialization. Note that this will create a second
        sparse matrix held in memory to efficiently check, which could cause memory concerns for
        larger data. If you are sure that there are no duplicated, user, item ID pairs, set to
        ``False``
    num_users: int
        Number of users in the dataset. If ``num_users == 'infer'``, this will be set to the
        ``mat.shape[0]`` or ``max(users) + 1``, depending on the input
    num_items: int
        Number of items in the dataset. If ``num_items == 'infer'``, this will be set to the
        ``mat.shape[1]`` or ``max(items) + 1``, depending on the input

    """
    def __init__(self,
                 mat: Optional[Union[coo_matrix, np.array]] = None,
                 users: Optional[Iterable[int]] = None,
                 items: Optional[Iterable[int]] = None,
                 ratings: Optional[Iterable[int]] = None,
                 allow_missing_ids: bool = False,
                 remove_duplicate_user_item_pairs: bool = True,
                 num_users: int = 'infer',
                 num_items: int = 'infer'):
        if mat is None and ratings is None:
            raise ValueError(
                'Ratings must be provided to ``ExplicitInteractions`` with ``mat`` or ``ratings``'
                ' - both cannot be ``None``!'
            )

        super().__init__(mat=mat,
                         users=users,
                         items=items,
                         ratings=ratings,
                         allow_missing_ids=allow_missing_ids,
                         remove_duplicate_user_item_pairs=remove_duplicate_user_item_pairs,
                         num_users=num_users,
                         num_items=num_items)

    @property
    def num_negative_samples(self) -> int:
        """Does not exist for explicit data."""
        raise AttributeError('``num_negative_samples`` does not exist for explicit datasets.')

    def __repr__(self) -> str:
        """String representation of ``ExplicitInteractions`` class."""
        return textwrap.dedent(
            f'''
            ExplicitInteractions object with {self.num_interactions} interactions between
            {self.num_users} users and {self.num_items} items, with minimum rating of
            {self.min_rating} and maximum rating of {self.max_rating}.
            '''
        ).replace('\n', ' ').strip()

    def __getitem__(self, index: Union[int, Iterable[int]]) -> (
        Union[Tuple[int, int, np.array], Tuple[np.array, np.array, np.array]]
    ):
        """Access item in the ``ExplicitInteractions`` instance."""
        user_id = self.mat.row[index]
        item_id = self.mat.col[index]
        rating = self.mat.data[index]

        return user_id, item_id, rating


class HDF5Interactions(torch.utils.data.Dataset):
    """
    Create an ``Interactions``-like object for data in the HDF5 format that might be too large to
    fit in memory.

    Many of the same features of ``Interactions`` are implemented here, with the exception that
    approximate negative sampling will always be used.

    Parameters
    ----------
    hdf5_path: str
    user_col: str
        Column in HDF5 file with user IDs. IDs must begin at 0
    item_col: str
        Column in HDF5 file with item IDs. IDs must begin at 0
    num_negative_samples: int
        Number of negative samples to return with each ``__getitem__`` call
    num_users: int
        Number of users in the dataset. If ``num_users == 'infer'`` and there is not a ``meta`` key
        in ``hdf5_path``'s HDF5 dataset, this will be set to the the maximum value in
        ``user_col`` + 1, found by iterating through the entire dataset
    num_items: int
        Number of items in the dataset. If ``num_items == 'infer'`` and there is not an ``meta`` key
        in ``hdf5_path``'s HDF5 dataset, this will be set to the the maximum value in
        ``item_col`` + 1, found by iterating through the entire dataset
    seed: int
        Seed for random sampling and shuffling if ``shuffle is True``
    shuffle: bool
        Shuffle data in a batch. For example, if one calls ``__getitem__`` with
        ``start_idx_and_batch_size = (0, 4)`` and ``shuffle is False``, this will always return the
        data at indices 0, 1, 2, 3 in order. However, the same call with ``shuffle = True`` will
        return a random shuffle of 0, 1, 2, 3 each call. This is recommended for use in a
        ``HDF5InteractionsDataLoader`` for training data in lieu of true data shuffling

    """
    def __init__(self,
                 hdf5_path: str,
                 user_col: str = 'users',
                 item_col: str = 'items',
                 num_negative_samples: int = 10,
                 num_users: int = 'infer',
                 num_items: int = 'infer',
                 seed: Optional[int] = None,
                 shuffle: bool = False):
        self.hdf5_path = hdf5_path
        self.user_col = user_col
        self.item_col = item_col
        self.num_negative_samples = num_negative_samples
        self.seed = seed
        self.shuffle = shuffle

        with pd.HDFStore(self.hdf5_path, mode='r', complib='blosc') as store:
            self.num_interactions = store.get_storer('interactions').shape

            if isinstance(num_users, int) and isinstance(num_items, int):
                self.num_users = num_users
                self.num_items = num_items
            else:
                try:
                    chunk = store.select('meta')
                    self.num_users = chunk['num_users'].item()
                    self.num_items = chunk['num_items'].item()
                except KeyError:
                    print('``meta`` key not found - generating ``num_users`` and ``num_items``.')

                    self.num_users = 0
                    self.num_items = 0
                    # while we are here, we can also check minimum IDs are 0 for free
                    # TODO: is there a more efficient way to check this? should we always check?
                    min_user_id = 1
                    min_item_id = 1

                    # default Pandas ``chunksize`` is 100000, so we will use that too
                    chunksize = 100000
                    for idx in tqdm(range(0, self.num_interactions, chunksize)):
                        chunk = store.select('interactions', start=idx, stop=(idx + chunksize))
                        self.num_users = max(chunk[self.user_col].max(), self.num_users)
                        self.num_items = max(chunk[self.item_col].max(), self.num_items)
                        min_user_id = min(chunk[self.user_col].min(), min_user_id)
                        min_item_id = min(chunk[self.item_col].min(), min_item_id)

                    if min_user_id != 0 or min_item_id != 0:
                        raise ValueError(
                            f'Minimum values of {user_col} and {item_col} in HDF5 data must both be'
                            f' 0, not {min_user_id} and {min_item_id}, respectively.'
                        )

                    # add one here since ``users`` and ``items`` are both zero-indexed
                    self.num_users += 1
                    self.num_items += 1

        assert self.num_users > 1
        assert self.num_items > 1

        if self.seed is None:
            self.seed = collie.utils.get_random_seed()

        np.random.seed(seed=self.seed)

    def __getitem__(self, start_idx_and_batch_size: Tuple[int, int]) -> (
        Tuple[Tuple[np.array, np.array], np.array]
    ):
        """Get a batch of data."""
        if isinstance(start_idx_and_batch_size, tuple):
            start_idx, batch_size = start_idx_and_batch_size
        else:
            start_idx = start_idx_and_batch_size
            batch_size = 1

        chunk = self._get_data_chunk(start_idx, batch_size)

        if len(chunk) == 0:
            raise IndexError(f'Index {start_idx} out of range for HDF5 data.')

        user_ids = chunk[self.user_col].to_numpy()
        item_ids = chunk[self.item_col].to_numpy()
        # ratings = chunk[self.ratings_col].to_numpy()  # not needed for any implemented loss yet

        if self.shuffle:
            idxs = np.random.permutation(len(user_ids))
            user_ids = user_ids[idxs]
            item_ids = item_ids[idxs]

        negative_item_ids = np.random.randint(
            low=0,
            high=self.num_items,
            size=(len(user_ids), self.num_negative_samples)
        )

        return (user_ids, item_ids), negative_item_ids

    def _get_data_chunk(self, start_idx: int, batch_size: int) -> pd.DataFrame:
        with pd.HDFStore(self.hdf5_path, mode='r', complib='blosc') as store:
            return store.select('interactions',
                                start=start_idx,
                                stop=(start_idx + batch_size))

    def __len__(self) -> int:
        """Get number of batches."""
        return self.num_interactions

    def __repr__(self) -> str:
        """String representation of ``HDF5Interactions`` class."""
        return textwrap.dedent(
            f'''
            HDF5Interactions object with {self.num_interactions} interactions between
            {self.num_users} users and {self.num_items} items, returning
            {self.num_negative_samples} negative samples per interaction.
            '''
        ).replace('\n', ' ').strip()

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first ``n`` rows of the underlying pd.DataFrame."""
        n = self._prep_head_tail_n(n=n)
        return self._get_data_chunk(0, n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """Return the last ``n`` rows of the underlying pd.DataFrame."""
        n = self._prep_head_tail_n(n=n)
        return self._get_data_chunk(self.num_interactions - n, n)

    def _prep_head_tail_n(self, n: int) -> int:
        """Ensure we don't run into an ``IndexError`` when using ``head`` or ``tail`` methods."""
        if n < 0:
            n = self.num_interactions + n
        if n > self.num_interactions:
            n = self.num_interactions

        return n


def _check_array_contains_all_integers(array: Iterable[int],
                                       array_max_value: int,
                                       array_name: str = 'Array') -> None:
    """Check that an array has all numbers between 0 and ``array_max``."""
    if set(array) != set(range(array_max_value)):
        raise ValueError(
            f'``{array_name}`` must contain every integer between 0 and {array_max_value - 1}. '
            + 'To override this error, set ``allow_missing_ids`` to True.'
        )


def _drop_array_values_by_idx(array: Iterable[Any], indices_to_drop: Iterable[int]) -> List[Any]:
    return [element for idx, element in enumerate(array) if idx not in indices_to_drop]
