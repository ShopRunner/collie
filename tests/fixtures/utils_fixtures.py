import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture()
def explicit_df():
    return pd.DataFrame(data={
        'userId': [2, 0, 1, 0, 1, 0, 3],
        'itemId': [0, 1, 1, 0, 3, 2, 1],
        'rating': [1, 2, 3, 1, 5, 3, 4]
    })


@pytest.fixture()
def explicit_df_with_duplicate_user_item_pairs():
    # the first and last user-item pairs have been repeated with modified ratings.
    # other than that, this is exactly the same as ``explicit_df`` above
    return pd.DataFrame(data={
        'userId': [2, 2, 2, 0, 1, 0, 1, 0, 3, 3, 3],
        'itemId': [0, 0, 0, 1, 1, 0, 3, 2, 1, 1, 1],
        'rating': [1, 3, 5, 2, 3, 1, 5, 3, 1, 4, 2]
    })


@pytest.fixture()
def df_to_turn_to_interactions():
    return pd.DataFrame(data={
        'userId': [0, 3, 1, 2, 1, 2, 1, 4],
        'itemId': [2, 0, 2, 2, 1, 4, 3, 2],
    })


@pytest.fixture()
def df_with_users_interacting_only_once():
    return pd.DataFrame(data={
        'userId': [0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 0, 5],
        'itemId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3],
    })


@pytest.fixture()
def item_embeddings():
    return torch.Tensor([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])


@pytest.fixture()
def item_biases():
    return np.array([[10.1],
                     [11],
                     [12]])


@pytest.fixture()
def iid_to_item_dict():
    return {0: 'item_1', 1: 'item_2', 2: 'item_3'}


@pytest.fixture()
def df_html_test():
    d = {'title': ['Greg',
                   'Real Greg'],
         'description': ['some text here',
                         'more text here'],
         'link': ['https://madeupsite.com',
                  'https://anothermadeupsite.com'],
         'image': ['https://avatars0.githubusercontent.com/u/13399445',
                   'https://avatars3.githubusercontent.com/u/31417712']}

    df_html_test = pd.DataFrame(data=d)
    df_html_test = df_html_test[['title', 'description', 'link', 'image']]

    return df_html_test
