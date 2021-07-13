import pytest
from sklearn.model_selection import train_test_split

from collie.cross_validation import random_split, stratified_split
from collie.interactions import ExplicitInteractions, Interactions
from collie.movielens import (get_movielens_metadata,
                              read_movielens_df,
                              read_movielens_df_item,
                              read_movielens_posters_df)
from collie.utils import convert_to_implicit


@pytest.fixture(scope='session')
def movielens_explicit_df():
    return read_movielens_df(decrement_ids=True)


@pytest.fixture(scope='session')
def movielens_explicit_df_not_decremented():
    return read_movielens_df(decrement_ids=False)


@pytest.fixture(scope='session')
def movielens_implicit_df(movielens_explicit_df):
    return convert_to_implicit(movielens_explicit_df)


@pytest.fixture(scope='session')
def movielens_df_item():
    return read_movielens_df_item()


@pytest.fixture(scope='session')
def movielens_posters_df():
    return read_movielens_posters_df()


@pytest.fixture(scope='session')
def movielens_metadata_df():
    return get_movielens_metadata()


@pytest.fixture(scope='session')
def movielens_implicit_interactions(movielens_implicit_df):
    return Interactions(users=movielens_implicit_df['user_id'],
                        items=movielens_implicit_df['item_id'],
                        ratings=movielens_implicit_df['rating'],
                        num_negative_samples=10,
                        max_number_of_samples_to_consider=200,
                        allow_missing_ids=True)


@pytest.fixture(scope='session')
def movielens_explicit_interactions(movielens_explicit_df):
    return ExplicitInteractions(users=movielens_explicit_df['user_id'],
                                items=movielens_explicit_df['item_id'],
                                ratings=movielens_explicit_df['rating'],
                                allow_missing_ids=True)


@pytest.fixture(scope='session')
def train_val_implicit_pandas_data(movielens_implicit_df):
    return train_test_split(movielens_implicit_df, test_size=0.2)


@pytest.fixture(scope='session')
def train_val_implicit_data(movielens_implicit_interactions):
    return stratified_split(
        interactions=movielens_implicit_interactions,
        val_p=0.,
        test_p=0.2,
        seed=42,
    )


@pytest.fixture(scope='session')
def train_val_explicit_data(movielens_explicit_interactions):
    return stratified_split(
        interactions=movielens_explicit_interactions,
        val_p=0.,
        test_p=0.2,
        seed=42,
    )


@pytest.fixture(scope='session')
def train_val_implicit_sample_data(movielens_implicit_interactions):
    _, train, val = random_split(
        interactions=movielens_implicit_interactions,
        val_p=0.05,
        test_p=0.01,
        seed=42,
    )

    return train, val


@pytest.fixture(scope='session')
def train_val_explicit_sample_data(movielens_explicit_interactions):
    _, train, val = random_split(
        interactions=movielens_explicit_interactions,
        val_p=0.05,
        test_p=0.01,
        seed=42,
    )

    return train, val
