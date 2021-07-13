import os

import numpy as np
import pandas as pd
import pytest

from collie.interactions import ExplicitInteractions, HDF5Interactions, Interactions
from collie.utils import create_ratings_matrix, pandas_df_to_hdf5


@pytest.fixture()
def df_for_interactions():
    # this should exactly match ``ratings_matrix_for_interactions`` below
    return pd.DataFrame(data={
        'user_id': [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
        'item_id': [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 0, 3],
        'ratings': [1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 5],
    })


@pytest.fixture()
def ratings_matrix_for_interactions():
    # this should exactly match ``df_for_interactions`` above
    return np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 2, 3, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 4, 5, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 2, 3, 4],
                     [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 5, 0, 0, 0, 0, 0, 0]])


@pytest.fixture()
def sparse_ratings_matrix_for_interactions(df_for_interactions):
    return create_ratings_matrix(df=df_for_interactions,
                                 user_col='user_id',
                                 item_col='item_id',
                                 ratings_col='ratings',
                                 sparse=True)


@pytest.fixture()
def df_for_interactions_with_missing_ids():
    # we are missing item ID 7
    # this should exactly match ``ratings_matrix_for_interactions_with_missing_ids`` below
    return pd.DataFrame(data={
        'user_id': [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
        'item_id': [1, 2, 2, 3, 4, 5, 6, 0, 8, 9, 0, 3],
        'ratings': [1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 5],
    })


@pytest.fixture()
def ratings_matrix_for_interactions_with_missing_ids():
    # we are missing item ID 7
    # this should exactly match ``df_for_interactions_with_missing_ids`` above
    return np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 2, 3, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 4, 5, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 5, 0, 0, 0, 0, 0, 0]])


@pytest.fixture()
def sparse_ratings_matrix_for_interactions_with_missing_ids(df_for_interactions_with_missing_ids):
    return create_ratings_matrix(df=df_for_interactions_with_missing_ids,
                                 user_col='user_id',
                                 item_col='item_id',
                                 ratings_col='ratings',
                                 sparse=True)


@pytest.fixture()
def df_for_interactions_with_0_ratings():
    # ``df_for_interactions`` but with three extra interactions with ratings of 0
    return pd.DataFrame(data={
        'user_id': [0, 0, 1, 1, 2, 2, 3, 1, 2, 3, 3, 3, 4, 5, 5],
        'item_id': [1, 2, 2, 3, 4, 5, 2, 4, 6, 7, 8, 9, 0, 3, 4],
        'ratings': [1, 1, 2, 3, 4, 5, 0, 0, 1, 2, 3, 4, 5, 5, 0],
    })


@pytest.fixture()
def df_for_interactions_with_duplicates():
    # this should match ``df_for_interactions`` with duplicate user/item pairs added at the
    # following indices: ``0 & 1`` and ``12 & 13``
    return pd.DataFrame(data={
        'user_id': [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5],
        'item_id': [1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 0, 3, 3],
        'ratings': [1, 2, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 5, 4],
    })


@pytest.fixture()
def interactions_pandas(df_for_interactions):
    return Interactions(users=df_for_interactions['user_id'],
                        items=df_for_interactions['item_id'],
                        ratings=df_for_interactions['ratings'],
                        check_num_negative_samples_is_valid=False)


@pytest.fixture()
def interactions_matrix(ratings_matrix_for_interactions):
    return Interactions(mat=ratings_matrix_for_interactions,
                        check_num_negative_samples_is_valid=False)


@pytest.fixture()
def interactions_sparse_matrix(sparse_ratings_matrix_for_interactions):
    return Interactions(mat=sparse_ratings_matrix_for_interactions,
                        check_num_negative_samples_is_valid=False)


@pytest.fixture()
def explicit_interactions_pandas(df_for_interactions):
    return ExplicitInteractions(users=df_for_interactions['user_id'],
                                items=df_for_interactions['item_id'],
                                ratings=df_for_interactions['ratings'])


@pytest.fixture()
def explicit_interactions_matrix(ratings_matrix_for_interactions):
    return ExplicitInteractions(mat=ratings_matrix_for_interactions)


@pytest.fixture()
def explicit_interactions_sparse_matrix(sparse_ratings_matrix_for_interactions):
    return ExplicitInteractions(mat=sparse_ratings_matrix_for_interactions)


@pytest.fixture()
def hdf5_pandas_df_path(df_for_interactions, tmpdir):
    hdf5_path = os.path.join(str(tmpdir), 'df_for_interactions.h5')
    pandas_df_to_hdf5(df=df_for_interactions, out_path=hdf5_path, key='interactions')

    return hdf5_path


@pytest.fixture()
def hdf5_pandas_df_path_with_meta(df_for_interactions, tmpdir):
    hdf5_path = os.path.join(str(tmpdir), 'df_for_interactions_meta.h5')
    pandas_df_to_hdf5(df=df_for_interactions, out_path=hdf5_path, key='interactions')

    additional_info_df = pd.DataFrame({
        'num_users': [df_for_interactions['user_id'].max() + 1],
        'num_items': [df_for_interactions['item_id'].max() + 1],
    })
    pandas_df_to_hdf5(df=additional_info_df, out_path=hdf5_path, key='meta')

    return hdf5_path


@pytest.fixture(params=['users', 'items', 'both_users_and_items'])
def hdf5_pandas_df_path_ids_start_at_1(request, df_for_interactions, tmpdir):
    incremented_df_for_interactions = df_for_interactions

    if 'users' in request.param:
        incremented_df_for_interactions['user_id'] += 1
    if 'items' in request.param:
        incremented_df_for_interactions['item_id'] += 1

    hdf5_path = os.path.join(str(tmpdir), 'df_for_interactions_incremented.h5')
    pandas_df_to_hdf5(df=incremented_df_for_interactions, out_path=hdf5_path, key='interactions')

    return hdf5_path


@pytest.fixture()
def hdf5_interactions(hdf5_pandas_df_path_with_meta):
    return HDF5Interactions(hdf5_path=hdf5_pandas_df_path_with_meta,
                            user_col='user_id',
                            item_col='item_id')
