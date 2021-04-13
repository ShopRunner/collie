import os

import numpy as np
import pandas as pd
import pytest

from collie_recs.utils import create_ratings_matrix, pandas_df_to_hdf5


@pytest.fixture()
def df_for_interactions():
    # this should exactly match ``ratings_matrix_for_interactions`` below
    return pd.DataFrame(data={
        'user_id': [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
        'item_id': [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 0, 3],
        'rating': [1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 5],
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
                                 ratings_col='rating',
                                 sparse=True)


@pytest.fixture()
def df_for_interactions_with_missing_ids():
    # we are missing item ID 7
    # this should exactly match ``ratings_matrix_for_interactions_with_missing_ids`` below
    return pd.DataFrame(data={
        'user_id': [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
        'item_id': [1, 2, 2, 3, 4, 5, 6, 0, 8, 9, 0, 3],
        'rating': [1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 5],
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
                                 ratings_col='rating',
                                 sparse=True)


@pytest.fixture()
def hdf5_pandas_df_path(df_for_interactions, tmpdir):
    hdf5_path = os.path.join(str(tmpdir), 'df_for_interactions.h5')
    pandas_df_to_hdf5(df=df_for_interactions, out_path=hdf5_path, key='interactions')

    return hdf5_path
