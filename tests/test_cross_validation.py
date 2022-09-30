import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix

from collie.cross_validation import random_split, stratified_split
from collie.interactions import ExplicitInteractions, Interactions


def test_bad_random_split_HDF5Interactions(hdf5_interactions):
    with pytest.raises(AssertionError):
        random_split(
            interactions=hdf5_interactions,
        )


def test_bad_stratified_split_HDF5Interactions(hdf5_interactions):
    with pytest.raises(AssertionError):
        stratified_split(
            interactions=hdf5_interactions,
        )


@pytest.mark.parametrize('data_type', ['implicit', 'explicit'])
def test_random_split(implicit_interactions_to_split,
                      explicit_interactions_to_split,
                      data_type):
    if data_type == 'implicit':
        interactions_class = Interactions
        interactions_kwargs = {
            'check_num_negative_samples_is_valid': False,
        }
        interactions_to_split = implicit_interactions_to_split
    else:
        interactions_class = ExplicitInteractions
        interactions_kwargs = {}
        interactions_to_split = explicit_interactions_to_split

    train_expected_df = pd.DataFrame(
        data={
            'user_id': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 4],
            'item_id': [0, 1, 2, 3, 4, 5, 8, 1, 3, 4, 1, 3, 4, 2, 2, 4],
            'rating': [1, 2, 3, 4, 5, 4, 1, 1, 3, 4, 2, 4, 5, 5, 3, 5],
        }
    )
    train_expected = interactions_class(
        mat=coo_matrix(
            (
                train_expected_df['rating'],
                (train_expected_df['user_id'], train_expected_df['item_id']),
            ),
            shape=(interactions_to_split.num_users, interactions_to_split.num_items),
        ),
        allow_missing_ids=True,
        **interactions_kwargs,
    )

    validate_expected_df = pd.DataFrame(
        data={'user_id': [3, 4, 4], 'item_id': [1, 1, 5], 'rating': [1, 2, 4]}
    )
    validate_expected = interactions_class(
        mat=coo_matrix(
            (
                validate_expected_df['rating'],
                (validate_expected_df['user_id'], validate_expected_df['item_id']),
            ),
            shape=(interactions_to_split.num_users, interactions_to_split.num_items),
        ),
        allow_missing_ids=True,
        **interactions_kwargs,
    )

    test_expected_df = pd.DataFrame(
        data={
            'user_id': [0, 0, 1, 2, 3],
            'item_id': [6, 7, 2, 2, 4],
            'rating': [3, 2, 2, 3, 4],
        }
    )
    test_expected = interactions_class(
        mat=coo_matrix(
            (
                test_expected_df['rating'],
                (test_expected_df['user_id'], test_expected_df['item_id']),
            ),
            shape=(interactions_to_split.num_users, interactions_to_split.num_items),
        ),
        allow_missing_ids=True,
        **interactions_kwargs,
    )

    (train_actual, validate_actual, test_actual) = random_split(
        interactions=interactions_to_split, val_p=0.1, test_p=0.2, seed=42
    )

    np.testing.assert_array_equal(train_actual.toarray(), train_expected.toarray())
    np.testing.assert_array_equal(
        validate_actual.toarray(), validate_expected.toarray()
    )
    np.testing.assert_array_equal(test_actual.toarray(), test_expected.toarray())

    assert (
        train_actual.num_users
        == train_expected.num_users
        == validate_actual.num_users
        == validate_expected.num_users
        == test_actual.num_users
        == test_expected.num_users
    )

    assert (
        train_actual.num_items
        == train_expected.num_items
        == validate_actual.num_items
        == validate_expected.num_items
        == test_actual.num_items
        == test_expected.num_items
    )

    assert (
        type(train_actual)
        == type(train_expected)
        == type(validate_actual)
        == type(validate_expected)
        == type(test_actual)
        == type(test_expected)
        == interactions_class
    )


def test_random_split_with_users_with_only_one_interaction(
    interactions_to_split_with_users_with_only_one_interaction,
):
    # unlike for ``stratified_split``, this should work without error
    random_split(
        interactions=interactions_to_split_with_users_with_only_one_interaction,
    )


@pytest.mark.parametrize('data_type', ['implicit', 'explicit'])
def test_stratified_split(implicit_interactions_to_split,
                          explicit_interactions_to_split,
                          data_type):
    if data_type == 'implicit':
        interactions_class = Interactions
        interactions_kwargs = {
            'check_num_negative_samples_is_valid': False,
        }
        interactions_to_split = implicit_interactions_to_split
    else:
        interactions_class = ExplicitInteractions
        interactions_kwargs = {}
        interactions_to_split = explicit_interactions_to_split

    train_expected_df = pd.DataFrame(
        data={
            'user_id': [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 4],
            'item_id': [1, 2, 3, 4, 6, 8, 1, 2, 3, 4, 2, 4, 5],
            'rating': [2, 3, 4, 5, 3, 1, 1, 2, 4, 5, 5, 5, 4],
        }
    )
    train_expected = interactions_class(
        mat=coo_matrix(
            (
                train_expected_df['rating'],
                (train_expected_df['user_id'], train_expected_df['item_id']),
            ),
            shape=(interactions_to_split.num_users, interactions_to_split.num_items),
        ),
        allow_missing_ids=True,
        **interactions_kwargs,
    )

    validate_expected_df = pd.DataFrame(
        data={
            'user_id': [0, 1, 2, 3, 4],
            'item_id': [7, 3, 2, 1, 2],
            'rating': [2, 3, 3, 1, 3],
        }
    )
    validate_expected = interactions_class(
        mat=coo_matrix(
            (
                validate_expected_df['rating'],
                (validate_expected_df['user_id'], validate_expected_df['item_id']),
            ),
            shape=(interactions_to_split.num_users, interactions_to_split.num_items),
        ),
        allow_missing_ids=True,
        **interactions_kwargs,
    )

    test_expected_df = pd.DataFrame(
        data={
            'user_id': [0, 0, 1, 2, 3, 4],
            'item_id': [0, 5, 4, 1, 4, 1],
            'rating': [1, 4, 4, 2, 4, 2],
        }
    )
    test_expected = interactions_class(
        mat=coo_matrix(
            (
                test_expected_df['rating'],
                (test_expected_df['user_id'], test_expected_df['item_id']),
            ),
            shape=(interactions_to_split.num_users, interactions_to_split.num_items),
        ),
        allow_missing_ids=True,
        **interactions_kwargs,
    )

    (train_actual, validate_actual, test_actual) = stratified_split(
        interactions=interactions_to_split, val_p=0.1, test_p=0.2, seed=46
    )

    np.testing.assert_array_equal(train_actual.toarray(), train_expected.toarray())
    np.testing.assert_array_equal(
        validate_actual.toarray(), validate_expected.toarray()
    )
    np.testing.assert_array_equal(test_actual.toarray(), test_expected.toarray())

    assert (
        train_actual.num_users
        == train_expected.num_users
        == validate_actual.num_users
        == validate_expected.num_users
        == test_actual.num_users
        == test_expected.num_users
    )

    assert (
        train_actual.num_items
        == train_expected.num_items
        == validate_actual.num_items
        == validate_expected.num_items
        == test_actual.num_items
        == test_expected.num_items
    )

    assert (
        type(train_actual)
        == type(train_expected)
        == type(validate_actual)
        == type(validate_expected)
        == type(test_actual)
        == type(test_expected)
        == interactions_class
    )


@pytest.mark.parametrize('processes', [0, -1])
def test_stratified_split_with_users_with_only_one_interaction_raises_error(
    interactions_to_split_with_users_with_only_one_interaction,
    processes
):
    with pytest.raises(
        ValueError,
        match='Unable to stratify split on users - the ``interactions`` object contains users '
              'with a single interaction. Either set ``force_split = True`` to put all users '
              'with a single interaction in the training set or run '
              '``collie.utils.remove_users_with_fewer_than_n_interactions`` first.'
    ):
        stratified_split(
            interactions=interactions_to_split_with_users_with_only_one_interaction,
            test_p=0.2,
            seed=42,
            processes=processes,
        )


@pytest.mark.parametrize('processes', [0, -1])
def test_stratified_split_with_users_with_only_one_interaction_force_split(
    interactions_to_split_with_users_with_only_one_interaction,
    processes
):
    users_with_only_one_interaction = [0, 5, 6]

    (train_actual, _, _) = stratified_split(
        interactions=interactions_to_split_with_users_with_only_one_interaction,
        val_p=0.1,
        test_p=0.2,
        seed=42,
        processes=processes,
        force_split=True
    )

    assert all(user in train_actual[:][0][0].tolist() for user in users_with_only_one_interaction)


class TestSplitsWithWrongP:
    def test_combined_too_large_random(self, implicit_interactions_to_split):
        with pytest.raises(ValueError):
            random_split(interactions=implicit_interactions_to_split, val_p=0.9, test_p=0.2)

    def test_combined_too_large_stratified(self, implicit_interactions_to_split):
        with pytest.raises(ValueError):
            stratified_split(interactions=implicit_interactions_to_split, val_p=0.9, test_p=0.2)

    def test_combined_equal_one_random(self, implicit_interactions_to_split):
        with pytest.raises(ValueError):
            random_split(interactions=implicit_interactions_to_split, val_p=0.7, test_p=0.3)

    def test_combined_equal_one_stratified(self, implicit_interactions_to_split):
        with pytest.raises(ValueError):
            stratified_split(interactions=implicit_interactions_to_split, val_p=0.7, test_p=0.3)

    def test_val_negative_but_combined_good_random(self, implicit_interactions_to_split):
        with pytest.raises(ValueError):
            random_split(interactions=implicit_interactions_to_split, val_p=-0.1, test_p=0.3)

    def test_val_negative_but_combined_good_stratified(self, implicit_interactions_to_split):
        with pytest.raises(ValueError):
            stratified_split(interactions=implicit_interactions_to_split, val_p=-0.1, test_p=0.3)

    def test_test_p_too_large_random(self, implicit_interactions_to_split):
        with pytest.raises(ValueError):
            random_split(interactions=implicit_interactions_to_split, test_p=1.1)

    def test_test_p_too_large_stratified(self, implicit_interactions_to_split):
        with pytest.raises(ValueError):
            stratified_split(interactions=implicit_interactions_to_split, test_p=1.1)

    def test_test_p_equal_one_random(self, implicit_interactions_to_split):
        with pytest.raises(ValueError):
            random_split(interactions=implicit_interactions_to_split, test_p=1)

    def test_test_p_equal_one_stratified(self, implicit_interactions_to_split):
        with pytest.raises(ValueError):
            stratified_split(interactions=implicit_interactions_to_split, test_p=1)

    def test_test_p_negative_random(self, implicit_interactions_to_split):
        with pytest.raises(ValueError):
            random_split(interactions=implicit_interactions_to_split, test_p=-0.7)

    def test_test_p_negative_stratified(self, implicit_interactions_to_split):
        with pytest.raises(ValueError):
            stratified_split(interactions=implicit_interactions_to_split, test_p=-0.7)


def test_splits_vary_number_of_processes(implicit_interactions_to_split):
    train_1, test_1 = stratified_split(interactions=implicit_interactions_to_split,
                                       seed=42,
                                       processes=-1)
    train_2, test_2 = stratified_split(interactions=implicit_interactions_to_split,
                                       seed=42,
                                       processes=0)
    train_3, test_3 = stratified_split(interactions=implicit_interactions_to_split,
                                       seed=42,
                                       processes=1)
    train_4, test_4 = stratified_split(interactions=implicit_interactions_to_split,
                                       seed=42,
                                       processes=2)

    # transitive property in action here
    np.testing.assert_array_equal(train_1.toarray(), train_2.toarray())
    np.testing.assert_array_equal(train_2.toarray(), train_3.toarray())
    np.testing.assert_array_equal(train_3.toarray(), train_4.toarray())

    np.testing.assert_array_equal(test_1.toarray(), test_2.toarray())
    np.testing.assert_array_equal(test_2.toarray(), test_3.toarray())
    np.testing.assert_array_equal(test_3.toarray(), test_4.toarray())
