import sys

import numpy as np
import pandas as pd
import pytest

from collie.interactions import (ApproximateNegativeSamplingInteractionsDataLoader,
                                 ExplicitInteractions,
                                 HDF5Interactions,
                                 HDF5InteractionsDataLoader,
                                 Interactions,
                                 InteractionsDataLoader,
                                 SequentialInteractions)


NUM_NEGATIVE_SAMPLES = 3
NUM_USERS_TO_GENERATE = 10


def test_Interactions(interactions_matrix,
                      interactions_sparse_matrix,
                      interactions_pandas):
    np.testing.assert_equal(interactions_matrix.toarray(), interactions_sparse_matrix.toarray())
    np.testing.assert_equal(interactions_matrix.toarray(), interactions_pandas.toarray())
    assert (
        interactions_matrix.num_users
        == interactions_sparse_matrix.num_users
        == interactions_pandas.num_users
    )
    assert (
        interactions_matrix.num_items
        == interactions_sparse_matrix.num_items
        == interactions_pandas.num_items
    )
    assert (
        interactions_matrix.num_interactions
        == interactions_sparse_matrix.num_interactions
        == interactions_pandas.num_interactions
    )

    expected_repr = (
        'Interactions object with 12 interactions between 6 users and 10 items,'
        ' returning 10 negative samples per interaction.'
    )
    assert (
        str(interactions_matrix)
        == str(interactions_sparse_matrix)
        == str(interactions_pandas)
        == expected_repr
    )


def test_ExplicitInteractions(explicit_interactions_matrix,
                              explicit_interactions_sparse_matrix,
                              explicit_interactions_pandas):
    np.testing.assert_equal(explicit_interactions_matrix.toarray(),
                            explicit_interactions_sparse_matrix.toarray())
    np.testing.assert_equal(explicit_interactions_matrix.toarray(),
                            explicit_interactions_pandas.toarray())
    assert (
        explicit_interactions_matrix.num_users
        == explicit_interactions_sparse_matrix.num_users
        == explicit_interactions_pandas.num_users
    )
    assert (
        explicit_interactions_matrix.num_items
        == explicit_interactions_sparse_matrix.num_items
        == explicit_interactions_pandas.num_items
    )
    assert (
        explicit_interactions_matrix.num_interactions
        == explicit_interactions_sparse_matrix.num_interactions
        == explicit_interactions_pandas.num_interactions
    )

    expected_repr = (
        'ExplicitInteractions object with 12 interactions between 6 users and 10 items,'
        ' with minimum rating of 1 and maximum rating of 5.'
    )
    assert (
        str(explicit_interactions_matrix)
        == str(explicit_interactions_sparse_matrix)
        == str(explicit_interactions_pandas)
        == expected_repr
    )


class TestInteractionsWithMissingIDs:
    def test_Interactions_with_missing_ids_raises_error(
        self,
        df_for_interactions_with_missing_ids,
        ratings_matrix_for_interactions_with_missing_ids,
        sparse_ratings_matrix_for_interactions_with_missing_ids
    ):
        with pytest.raises(ValueError):
            Interactions(mat=ratings_matrix_for_interactions_with_missing_ids,
                         check_num_negative_samples_is_valid=False)

        with pytest.raises(ValueError):
            Interactions(mat=sparse_ratings_matrix_for_interactions_with_missing_ids,
                         check_num_negative_samples_is_valid=False)

        with pytest.raises(ValueError):
            Interactions(users=df_for_interactions_with_missing_ids['user_id'],
                         items=df_for_interactions_with_missing_ids['item_id'],
                         ratings=df_for_interactions_with_missing_ids['ratings'],
                         check_num_negative_samples_is_valid=False)

    def test_Interactions_with_missing_ids(
        self,
        df_for_interactions_with_missing_ids,
        ratings_matrix_for_interactions_with_missing_ids,
        sparse_ratings_matrix_for_interactions_with_missing_ids
    ):
        Interactions(mat=ratings_matrix_for_interactions_with_missing_ids,
                     allow_missing_ids=True,
                     check_num_negative_samples_is_valid=False)

        Interactions(mat=sparse_ratings_matrix_for_interactions_with_missing_ids,
                     allow_missing_ids=True,
                     check_num_negative_samples_is_valid=False)

        Interactions(users=df_for_interactions_with_missing_ids['user_id'],
                     items=df_for_interactions_with_missing_ids['item_id'],
                     ratings=df_for_interactions_with_missing_ids['ratings'],
                     allow_missing_ids=True,
                     check_num_negative_samples_is_valid=False)


class TestInteractionsWithInvalidLengths:
    def test_items_length_bad(self, df_for_interactions_with_missing_ids):
        with pytest.raises(ValueError):
            Interactions(users=df_for_interactions_with_missing_ids['user_id'],
                         items=df_for_interactions_with_missing_ids['item_id'][:-1],
                         ratings=df_for_interactions_with_missing_ids['ratings'],
                         allow_missing_ids=True,
                         check_num_negative_samples_is_valid=False)

    def test_users_length_bad(self, df_for_interactions_with_missing_ids):
        with pytest.raises(ValueError):
            Interactions(users=df_for_interactions_with_missing_ids['user_id'][:-1],
                         items=df_for_interactions_with_missing_ids['item_id'],
                         ratings=df_for_interactions_with_missing_ids['ratings'],
                         allow_missing_ids=True,
                         check_num_negative_samples_is_valid=False)

    def test_ratings_length_bad(self, df_for_interactions_with_missing_ids):
        with pytest.raises(ValueError):
            Interactions(users=df_for_interactions_with_missing_ids['user_id'],
                         items=df_for_interactions_with_missing_ids['item_id'],
                         ratings=df_for_interactions_with_missing_ids['ratings'][:-1],
                         allow_missing_ids=True,
                         check_num_negative_samples_is_valid=False)

    def test_all_lengths_bad(self, df_for_interactions_with_missing_ids):
        Interactions(users=df_for_interactions_with_missing_ids['user_id'][:-1],
                     items=df_for_interactions_with_missing_ids['item_id'][:-1],
                     ratings=df_for_interactions_with_missing_ids['ratings'][:-1],
                     allow_missing_ids=True,
                     check_num_negative_samples_is_valid=False)


def test_Interactions_with_0_ratings(interactions_pandas, df_for_interactions_with_0_ratings):
    with pytest.warns(UserWarning):
        interactions_with_0s = Interactions(users=df_for_interactions_with_0_ratings['user_id'],
                                            items=df_for_interactions_with_0_ratings['item_id'],
                                            ratings=df_for_interactions_with_0_ratings['ratings'],
                                            check_num_negative_samples_is_valid=False)

    assert np.array_equal(interactions_pandas.toarray(), interactions_with_0s.toarray())


def test_ExplicitInteractions_with_0_ratings(explicit_interactions_pandas,
                                             df_for_interactions_with_0_ratings):
    interactions_with_0s = ExplicitInteractions(
        users=df_for_interactions_with_0_ratings['user_id'],
        items=df_for_interactions_with_0_ratings['item_id'],
        ratings=df_for_interactions_with_0_ratings['ratings'],
    )

    assert np.array_equal(explicit_interactions_pandas.toarray(), interactions_with_0s.toarray())

    assert interactions_with_0s.min_rating == 0
    assert explicit_interactions_pandas.num_interactions < interactions_with_0s.num_interactions


class TestBadInteractionsInstantiation:
    def test_items_None(self, df_for_interactions):
        with pytest.raises(AssertionError):
            Interactions(users=df_for_interactions['user_id'],
                         items=None,
                         ratings=df_for_interactions['ratings'],
                         check_num_negative_samples_is_valid=False)

    def test_users_None(self, df_for_interactions):
        with pytest.raises(AssertionError):
            Interactions(users=None,
                         items=df_for_interactions['item_id'],
                         ratings=df_for_interactions['ratings'],
                         check_num_negative_samples_is_valid=False)

    def test_ratings_None_but_its_okay(self, df_for_interactions):
        Interactions(users=df_for_interactions['user_id'],
                     items=df_for_interactions['item_id'],
                     ratings=None,
                     check_num_negative_samples_is_valid=False)

    def test_ratings_None_but_its_explicit_so_not_okay(self, df_for_interactions):
        with pytest.raises(ValueError):
            ExplicitInteractions(users=df_for_interactions['user_id'],
                                 items=df_for_interactions['item_id'],
                                 ratings=None)

    def test_duplicate_user_item_pairs(self,
                                       interactions_pandas,
                                       df_for_interactions_with_duplicates):
        duplicated_interactions = Interactions(users=df_for_interactions_with_duplicates['user_id'],
                                               items=df_for_interactions_with_duplicates['item_id'],
                                               check_num_negative_samples_is_valid=False,
                                               remove_duplicate_user_item_pairs=False)

        assert duplicated_interactions.mat.getnnz() != interactions_pandas.mat.getnnz()

        non_duplicated_interactions = (
            Interactions(users=df_for_interactions_with_duplicates['user_id'],
                         items=df_for_interactions_with_duplicates['item_id'],
                         check_num_negative_samples_is_valid=False,
                         remove_duplicate_user_item_pairs=True)
        )

        assert non_duplicated_interactions.mat.getnnz() == interactions_pandas.mat.getnnz()

    def test_duplicate_user_item_pairs_explicit(self,
                                                explicit_interactions_pandas,
                                                df_for_interactions_with_duplicates):
        duplicated_interactions = ExplicitInteractions(
            users=df_for_interactions_with_duplicates['user_id'],
            items=df_for_interactions_with_duplicates['item_id'],
            ratings=df_for_interactions_with_duplicates['ratings'],
            remove_duplicate_user_item_pairs=False,
        )

        assert duplicated_interactions.mat.getnnz() != explicit_interactions_pandas.mat.getnnz()

        non_duplicated_interactions = (
            ExplicitInteractions(users=df_for_interactions_with_duplicates['user_id'],
                                 items=df_for_interactions_with_duplicates['item_id'],
                                 ratings=df_for_interactions_with_duplicates['ratings'],
                                 remove_duplicate_user_item_pairs=True)
        )

        assert non_duplicated_interactions.mat.getnnz() == explicit_interactions_pandas.mat.getnnz()


class TestInteractionsDataMethods:
    def test_to_dense(self,
                      interactions_matrix,
                      interactions_pandas,
                      sparse_ratings_matrix_for_interactions):
        assert np.array_equal(interactions_matrix.todense(), interactions_pandas.todense())
        assert np.array_equal(interactions_matrix.todense(),
                              sparse_ratings_matrix_for_interactions.todense())

    def test_to_array(self,
                      interactions_matrix,
                      interactions_pandas,
                      sparse_ratings_matrix_for_interactions):
        assert np.array_equal(interactions_matrix.toarray(), interactions_pandas.toarray())
        assert np.array_equal(interactions_matrix.toarray(),
                              sparse_ratings_matrix_for_interactions.toarray())

    def test_head_default(self,
                          interactions_matrix,
                          interactions_pandas,
                          sparse_ratings_matrix_for_interactions):
        assert np.array_equal(interactions_matrix.head(), interactions_pandas.head())
        assert np.array_equal(interactions_matrix.head(),
                              sparse_ratings_matrix_for_interactions.toarray()[:5])

    def test_tail_default(self,
                          interactions_matrix,
                          interactions_pandas,
                          sparse_ratings_matrix_for_interactions):
        assert np.array_equal(interactions_matrix.tail(), interactions_pandas.tail())
        assert np.array_equal(interactions_matrix.tail(),
                              sparse_ratings_matrix_for_interactions.toarray()[-5:])

    def test_head_3(self,
                    interactions_matrix,
                    interactions_pandas,
                    sparse_ratings_matrix_for_interactions):
        assert np.array_equal(interactions_matrix.head(3), interactions_pandas.head(3))
        assert np.array_equal(interactions_matrix.head(3),
                              sparse_ratings_matrix_for_interactions.toarray()[:3])

    def test_tail_3(self,
                    interactions_matrix,
                    interactions_pandas,
                    sparse_ratings_matrix_for_interactions):
        assert np.array_equal(interactions_matrix.tail(3), interactions_pandas.tail(3))
        assert np.array_equal(interactions_matrix.tail(3),
                              sparse_ratings_matrix_for_interactions.toarray()[3:])

    def test_head_negative_3(self,
                             interactions_matrix,
                             interactions_pandas,
                             sparse_ratings_matrix_for_interactions):
        assert np.array_equal(interactions_matrix.head(-3), interactions_pandas.head(-3))
        assert np.array_equal(interactions_matrix.head(-3),
                              sparse_ratings_matrix_for_interactions.toarray()[:-3])

    def test_tail_negative_3(self,
                             interactions_matrix,
                             interactions_pandas,
                             sparse_ratings_matrix_for_interactions):
        assert np.array_equal(interactions_matrix.tail(-3), interactions_pandas.tail(-3))
        assert np.array_equal(interactions_matrix.tail(-3),
                              sparse_ratings_matrix_for_interactions.toarray()[-3:])

    def test_head_large_positive(self,
                                 interactions_matrix,
                                 interactions_pandas,
                                 sparse_ratings_matrix_for_interactions):
        assert np.array_equal(interactions_matrix.head(sys.maxsize),
                              interactions_pandas.head(sys.maxsize))
        assert np.array_equal(interactions_matrix.head(sys.maxsize),
                              sparse_ratings_matrix_for_interactions.toarray()[:sys.maxsize])

    def test_tail_large_positive(self,
                                 interactions_matrix,
                                 interactions_pandas,
                                 sparse_ratings_matrix_for_interactions):
        assert np.array_equal(interactions_matrix.tail(sys.maxsize),
                              interactions_pandas.tail(sys.maxsize))
        assert np.array_equal(interactions_matrix.tail(sys.maxsize),
                              sparse_ratings_matrix_for_interactions.toarray()[-sys.maxsize:])

    def test_head_large_negative(self,
                                 interactions_matrix,
                                 interactions_pandas,
                                 sparse_ratings_matrix_for_interactions):
        assert np.array_equal(interactions_matrix.head(-sys.maxsize),
                              interactions_pandas.head(-sys.maxsize))
        assert np.array_equal(interactions_matrix.head(-sys.maxsize),
                              sparse_ratings_matrix_for_interactions.toarray()[:-sys.maxsize])

    def test_tail_large_negative(self,
                                 interactions_matrix,
                                 interactions_pandas,
                                 sparse_ratings_matrix_for_interactions):
        assert np.array_equal(interactions_matrix.tail(-sys.maxsize),
                              interactions_pandas.tail(-sys.maxsize))
        assert np.array_equal(interactions_matrix.tail(-sys.maxsize),
                              sparse_ratings_matrix_for_interactions.toarray()[sys.maxsize:])


class TestInteractionsNegativeSampling:
    def test_Interactions_approximate_negative_samples(self, ratings_matrix_for_interactions):
        interactions = Interactions(mat=ratings_matrix_for_interactions,
                                    num_negative_samples=NUM_NEGATIVE_SAMPLES,
                                    max_number_of_samples_to_consider=0,
                                    seed=42)

        assert interactions.positive_items == {}

        for _ in range(3):
            _, negative_samples = interactions[0]

            assert len(negative_samples) == 3

    def test_Interactions_approximate_negative_samples_many_users(
        self,
        ratings_matrix_for_interactions,
    ):
        interactions = Interactions(mat=ratings_matrix_for_interactions,
                                    num_negative_samples=NUM_NEGATIVE_SAMPLES,
                                    max_number_of_samples_to_consider=0,
                                    seed=42)

        assert interactions.positive_items == {}

        for _ in range(3):
            _, negative_samples = interactions[list(range(NUM_USERS_TO_GENERATE))]

            assert len(negative_samples) == NUM_USERS_TO_GENERATE

            for negative_sample in negative_samples:
                assert len(negative_sample) == NUM_NEGATIVE_SAMPLES

    def test_Interactions_approximate_negative_samples_partway_through(
        self,
        ratings_matrix_for_interactions,
    ):
        with pytest.warns(UserWarning):
            interactions = Interactions(mat=ratings_matrix_for_interactions,
                                        num_negative_samples=NUM_NEGATIVE_SAMPLES,
                                        max_number_of_samples_to_consider=1,
                                        seed=42)

        assert interactions.positive_items != {}

        for _ in range(3):
            _, negative_samples = interactions[0]

            assert len(negative_samples) == 3

    def test_Interactions_exact_negative_samples(self, ratings_matrix_for_interactions):
        interactions = Interactions(mat=ratings_matrix_for_interactions,
                                    num_negative_samples=NUM_NEGATIVE_SAMPLES,
                                    max_number_of_samples_to_consider=200,
                                    seed=42)

        assert interactions.positive_items != {}

        all_negative_samples = list()
        for _ in range(10):
            _, negative_samples = interactions[0]

            assert len(negative_samples) == NUM_NEGATIVE_SAMPLES

            for negative_sample in negative_samples:
                assert negative_sample.item() not in ratings_matrix_for_interactions[0].nonzero()[0]

            all_negative_samples += negative_samples.tolist()

        assert len(set(all_negative_samples)) > NUM_NEGATIVE_SAMPLES

    def test_Interactions_exact_negative_samples_many_users(self, ratings_matrix_for_interactions):
        interactions = Interactions(mat=ratings_matrix_for_interactions,
                                    num_negative_samples=NUM_NEGATIVE_SAMPLES,
                                    max_number_of_samples_to_consider=200,
                                    seed=42)

        assert interactions.positive_items != {}

        for _ in range(10):
            (user_ids, _), negative_samples = interactions[list(range(NUM_USERS_TO_GENERATE))]

            assert len(negative_samples) == NUM_USERS_TO_GENERATE

            for idx, user_id in enumerate(user_ids):
                assert len(negative_samples[idx]) == NUM_NEGATIVE_SAMPLES

                for negative_sample in negative_samples[idx]:
                    assert (
                        negative_sample.item()
                        not in ratings_matrix_for_interactions[user_id].nonzero()[0]
                    )

    def test_Interactions_exact_negative_samples_num_negative_samples_too_large(
        self,
        ratings_matrix_for_interactions,
    ):
        with pytest.raises(AssertionError):
            Interactions(mat=ratings_matrix_for_interactions,
                         max_number_of_samples_to_consider=200,
                         num_negative_samples=8)


class TestSequentialInteractions():
    def test_sequences_one_sequence(self, users_items_timestamps_sequential):
        expected = np.array([
            [-1, -1, 0, 1, 2, 3, 4, 5, 6, 7],
        ])
        users, items, timestamps = users_items_timestamps_sequential

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps)

        assert len(test_interactions) == len(expected)
        assert np.array_equal(test_interactions.sequences, expected)

    @pytest.mark.parametrize('sequence_method', ['none', 'todense', 'toarray'])
    def test_sequences_one_sequence_shorter_max_sequence(
        self, sequence_method, users_items_timestamps_sequential,
    ):
        expected = np.array([
            [5, 6, 7],
            [2, 3, 4],
            [-1, 0, 1],
        ])
        users, items, timestamps = users_items_timestamps_sequential

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_sequence_length=3)

        if sequence_method == 'none':
            actual = test_interactions.sequences
        elif sequence_method == 'todense':
            actual = test_interactions.todense()
        elif sequence_method == 'toarray':
            actual = test_interactions.toarray()

        assert len(test_interactions) == len(expected)
        assert np.array_equal(actual, expected)

    def test_sequences_one_sequence_shorter_max_sequence_step_size(
        self, users_items_timestamps_sequential,
    ):
        expected = np.array([
            [5, 6, 7],
            [3, 4, 5],
            [1, 2, 3],
            [-1, 0, 1],
        ])
        users, items, timestamps = users_items_timestamps_sequential

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_sequence_length=3,
                                                   step_size=2)

        assert len(test_interactions) == len(expected)
        assert np.array_equal(test_interactions.sequences, expected)

    def test_sequences_one_sequence_shorter_min_sequence(self, users_items_timestamps_sequential):
        expected = np.array([
            [5, 6, 7],
            [2, 3, 4],
        ])
        users, items, timestamps = users_items_timestamps_sequential

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   min_sequence_length=3,
                                                   max_sequence_length=3)

        assert len(test_interactions) == len(expected)
        assert np.array_equal(test_interactions.sequences, expected)

    def test_sequences_one_sequence_shorter_max_time(self, users_items_timestamps_sequential):
        expected = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 7],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 6],
            [-1, -1, -1, -1, -1, -1, -1, -1, 4, 5],
            [-1, -1, -1, -1, -1, -1, 0, 1, 2, 3],
        ])
        users, items, timestamps = users_items_timestamps_sequential

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_time=2)

        assert len(test_interactions) == len(expected)
        assert np.array_equal(test_interactions.sequences, expected)

    def test_sequences_one_sequence_max_time_1(self, users_items_timestamps_sequential):
        expected = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 7],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 6],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 4],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 2],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        ])
        users, items, timestamps = users_items_timestamps_sequential

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_time=1)

        assert len(test_interactions) == len(expected)
        assert np.array_equal(test_interactions.sequences, expected)

    def test_sequences_one_sequence_max_time_1_uneven_time(self, users_items_timestamps_sequential):
        expected = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 7],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 6],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 4],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 2],
            [-1, -1, -1, -1, -1, -1, -1, -1, 0, 1],
        ])
        users, items, _ = users_items_timestamps_sequential
        timestamps = [0, 0.9, 4, 6, 9, 11, 14, 17]

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_time=1)

        assert len(test_interactions) == len(expected)
        assert np.array_equal(test_interactions.sequences, expected)

    def test_sequences_one_sequence_shorter_max_time_min_length_2(
        self, users_items_timestamps_sequential,
    ):
        expected = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, 4, 5],
            [-1, -1, -1, -1, -1, -1, 0, 1, 2, 3],
        ])
        users, items, timestamps = users_items_timestamps_sequential

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_time=2,
                                                   min_sequence_length=2)

        assert len(test_interactions) == len(expected)
        assert np.array_equal(test_interactions.sequences, expected)

    def test_sequences_one_sequence_shorter_max_time_min_length_2_step_size(
        self, users_items_timestamps_sequential,
    ):
        expected = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, 4, 5],
            [-1, -1, -1, -1, -1, -1, 0, 1, 2, 3],
        ])
        users, items, timestamps = users_items_timestamps_sequential

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_time=2,
                                                   min_sequence_length=2,
                                                   step_size=3)

        assert len(test_interactions) == len(expected)
        assert np.array_equal(test_interactions.sequences, expected)

    def test_sequences_one_sequence_shorter_max_time_step_size(
        self, users_items_timestamps_sequential,
    ):
        expected = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 7],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 6],
            [-1, -1, -1, -1, -1, -1, -1, -1, 4, 5],
            [-1, -1, -1, -1, -1, -1, 0, 1, 2, 3],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        ])
        users, items, timestamps = users_items_timestamps_sequential

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_time=2,
                                                   step_size=3)

        assert len(test_interactions) == len(expected)
        assert np.array_equal(test_interactions.sequences, expected)

    def test_sequences_one_sequence_shorter_max_time_min_length_3(
        self, users_items_timestamps_sequential,
    ):
        expected = np.array([
            [-1, -1, -1, -1, -1, -1, 0, 1, 2, 3],
        ])
        users, items, timestamps = users_items_timestamps_sequential

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_time=2,
                                                   min_sequence_length=3)

        assert len(test_interactions) == len(expected)
        assert np.array_equal(test_interactions.sequences, expected)

    def test_sequences_one_sequence_shorter_max_time_max_length_min_length_3(
        self, users_items_timestamps_sequential,
    ):
        expected = np.array([
            [1, 2, 3],
        ])
        users, items, timestamps = users_items_timestamps_sequential

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_time=2,
                                                   max_sequence_length=3,
                                                   min_sequence_length=3)

        assert len(test_interactions) == len(expected)
        assert np.array_equal(test_interactions.sequences, expected)

    def test_sequences_multiple_users(self, users_items_timestamps_sequential):
        expected = np.array([
            [-1, -1, -1, -1, -1, -1, -1, 0, 1, 2],
            [-1, -1, -1, -1, -1, -1, -1, -1, 3, 4],
            [-1, -1, -1, -1, -1, -1, -1, -1, 5, 6],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 7],
        ])
        _, items, timestamps = users_items_timestamps_sequential
        users = [0, 0, 0, 1, 1, 3, 3, 4]

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps)

        print(test_interactions.sequences)
        assert len(test_interactions) == len(expected)
        assert np.array_equal(test_interactions.sequences, expected)

    def test_sequences_multiple_users_time_difference(self, users_items_timestamps_sequential):
        expected = np.array([
            [-1, -1, -1, -1, -1, -1, -1, 0, 1, 2],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 7],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 6],
            [-1, -1, -1, -1, -1, -1, -1, -1, 4, 5],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 3],
        ])

        _, items, timestamps = users_items_timestamps_sequential
        users = [0, 0, 0, 1, 1, 1, 1, 1]

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_time=2)

        print(test_interactions.sequences)
        assert len(test_interactions) == len(expected)
        assert np.array_equal(test_interactions.sequences, expected)

    def test_sequences_with_missing_ids(self):
        with pytest.raises(ValueError):
            SequentialInteractions(
                users=[0, 0, 0, 0, 0, 0, 0, 0],
                # note the lack of an item ID of ``3``
                items=[0, 1, 2, 2, 4, 5, 6, 7],
                timestamps=[0, 2, 4, 6, 9, 11, 14, 17],
            )

        SequentialInteractions(
            users=[0, 0, 0, 0, 0, 2, 2, 2],
            # again, note the lack of an item ID of ``3``
            items=[0, 1, 2, 2, 4, 5, 6, 7],
            timestamps=[0, 2, 4, 6, 9, 11, 14, 17],
            allow_missing_ids=True,
        )

    def test_sequences_repr_with_min_sequence_length(self, users_items_timestamps_sequential):
        _, items, timestamps = users_items_timestamps_sequential
        users = [0, 0, 0, 1, 1, 3, 3, 4]

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_sequence_length=4,
                                                   min_sequence_length=2,
                                                   step_size=3,
                                                   num_negative_samples=5)

        assert str(test_interactions) == (
            'SequentialInteractions object with 3 sequences between 4 users and 8 items, returning'
            ' 5 negative samples per interaction with sequence length between 2 and 4, and step'
            ' size 3.'
        )

    def test_sequences_repr_without_min_sequence_length(self, users_items_timestamps_sequential):
        _, items, timestamps = users_items_timestamps_sequential
        users = [0, 0, 0, 1, 1, 3, 3, 4]

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_sequence_length=4,
                                                   step_size=3,
                                                   num_negative_samples=5)

        assert str(test_interactions) == (
            'SequentialInteractions object with 4 sequences between 4 users and 8 items, returning'
            ' 5 negative samples per interaction with sequence length less than 4, and step size 3.'
        )

    def test_sequences__getitem__single(
        self, users_items_timestamps_sequential,
    ):
        expected = np.array([
            [5, 6, 7],
            [3, 4, 5],
            [1, 2, 3],
            [-1, 0, 1],
        ])
        users, items, timestamps = users_items_timestamps_sequential
        num_negative_samples = 7

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_sequence_length=3,
                                                   step_size=2,
                                                   num_negative_samples=num_negative_samples)

        actual = test_interactions[0]

        assert len(actual) == 2

        assert np.array_equal(actual[0], expected[0])
        assert len(actual[1]) == num_negative_samples

    def test_sequences__getitem__multiple(
        self, users_items_timestamps_sequential,
    ):
        expected = np.array([
            [5, 6, 7],
            [3, 4, 5],
            [1, 2, 3],
            [-1, 0, 1],
        ])
        users, items, timestamps = users_items_timestamps_sequential
        num_negative_samples = 7

        test_interactions = SequentialInteractions(users=users,
                                                   items=items,
                                                   timestamps=timestamps,
                                                   max_sequence_length=3,
                                                   step_size=2,
                                                   num_negative_samples=num_negative_samples)

        actual = test_interactions[[0, 1, 2]]

        assert len(actual) == 2

        assert np.array_equal(actual[0], expected[[0, 1, 2]])
        assert actual[1].shape == (3, num_negative_samples)

    # def test_sequences_head(self, users_items_timestamps_sequential):
    #     expected = np.array([
    #         [-1, -1, -1, -1, -1, -1, -1, -1, -1, 7],
    #         [-1, -1, -1, -1, -1, -1, -1, -1, -1, 6],
    #         [-1, -1, -1, -1, -1, -1, -1, -1, -1, 5],
    #         [-1, -1, -1, -1, -1, -1, -1, -1, -1, 4],
    #         [-1, -1, -1, -1, -1, -1, -1, -1, -1, 3],
    #         [-1, -1, -1, -1, -1, -1, -1, -1, -1, 2],
    #         [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
    #         [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
    #     ])
    #     users, items, timestamps = users_items_timestamps_sequential
    #
    #     test_interactions = SequentialInteractions(users=users,
    #                                                items=items,
    #                                                timestamps=timestamps,
    #                                                max_time=1)
    #
    #     assert np.array_equal(test_interactions.head(5), expected)


def test_HDF5Interactions_meta_instantiation(hdf5_pandas_df_path,
                                             hdf5_pandas_df_path_with_meta,
                                             capfd):
    interactions_with_meta = HDF5Interactions(hdf5_path=hdf5_pandas_df_path_with_meta,
                                              user_col='user_id',
                                              item_col='item_id')

    out, _ = capfd.readouterr()
    assert '``meta`` key not found - generating ``num_users`` and ``num_items``' not in out

    interactions_no_meta = HDF5Interactions(hdf5_path=hdf5_pandas_df_path,
                                            user_col='user_id',
                                            item_col='item_id')

    out, _ = capfd.readouterr()
    assert '``meta`` key not found - generating ``num_users`` and ``num_items``' in out

    assert interactions_with_meta.num_users == interactions_no_meta.num_users
    assert interactions_with_meta.num_items == interactions_no_meta.num_items

    expected_repr = (
        'HDF5Interactions object with 12 interactions between 6 users and 10 items, returning 10'
        ' negative samples per interaction.'
    )
    assert str(interactions_with_meta) == str(interactions_no_meta) == expected_repr


def test_bad_HDF5Interactions_instantiation_incremented(hdf5_pandas_df_path_ids_start_at_1):
    with pytest.raises(ValueError):
        HDF5Interactions(hdf5_path=hdf5_pandas_df_path_ids_start_at_1,
                         user_col='user_id',
                         item_col='item_id')


def test_HDF5Interactions__getitem__(hdf5_interactions, hdf5_pandas_df_path_with_meta):
    shuffled_interactions = HDF5Interactions(hdf5_path=hdf5_pandas_df_path_with_meta,
                                             user_col='user_id',
                                             item_col='item_id',
                                             shuffle=True,
                                             seed=42)

    # ensure both methods of indexing the ``__getitem__`` method are equal
    assert np.array_equal(hdf5_interactions[(0, 1)][0][0], hdf5_interactions[0][0][0])
    assert np.array_equal(hdf5_interactions[(0, 1)][0][1], hdf5_interactions[0][0][1])

    # ensure when we shuffle data, the outputs of ``__getitem__`` are not equal
    assert not np.array_equal(hdf5_interactions[(0, 5)][0][0], shuffled_interactions[(0, 5)][0][0])
    assert not np.array_equal(hdf5_interactions[(0, 5)][0][1], shuffled_interactions[(0, 5)][0][1])

    # ensure we handle an out-of-bounds ``__getitem__`` properly...
    with pytest.raises(IndexError):
        hdf5_interactions[hdf5_interactions.num_interactions]

    # ... but we can request more than is available and still get a data chunk
    out_of_bounds_request = hdf5_interactions[(hdf5_interactions.num_interactions - 1, 1024)]
    assert len(out_of_bounds_request) < 1024


class TestHDF5InteractionsHeadTail:
    def test_head_3(self, hdf5_interactions, df_for_interactions):
        pd.testing.assert_frame_equal(hdf5_interactions.head(3), df_for_interactions.head(3))

    def test_tail_3(self, hdf5_interactions, df_for_interactions):
        pd.testing.assert_frame_equal(hdf5_interactions.tail(3), df_for_interactions.tail(3))

    def test_head_42(self, hdf5_interactions, df_for_interactions):
        pd.testing.assert_frame_equal(hdf5_interactions.head(42), df_for_interactions.head(42))

    def test_tail_42(self, hdf5_interactions, df_for_interactions):
        pd.testing.assert_frame_equal(hdf5_interactions.tail(42), df_for_interactions.tail(42))

    def test_head_negative_1(self, hdf5_interactions, df_for_interactions):
        pd.testing.assert_frame_equal(hdf5_interactions.head(-1), df_for_interactions.head(-1))

    def test_tail_negative_1(self, hdf5_interactions, df_for_interactions):
        pd.testing.assert_frame_equal(hdf5_interactions.tail(-1), df_for_interactions.tail(-1))

    def test_head_large_positive(self, hdf5_interactions, df_for_interactions):
        pd.testing.assert_frame_equal(hdf5_interactions.head(sys.maxsize),
                                      df_for_interactions.head(sys.maxsize))

    def test_tail_large_positive(self, hdf5_interactions, df_for_interactions):
        pd.testing.assert_frame_equal(hdf5_interactions.tail(sys.maxsize),
                                      df_for_interactions.tail(sys.maxsize))

    def test_head_large_negative(self, hdf5_interactions, df_for_interactions):
        pd.testing.assert_frame_equal(hdf5_interactions.head(-sys.maxsize),
                                      df_for_interactions.head(-sys.maxsize))

    def test_tail_large_negative(self, hdf5_interactions, df_for_interactions):
        pd.testing.assert_frame_equal(hdf5_interactions.tail(-sys.maxsize),
                                      df_for_interactions.tail(-sys.maxsize))


@pytest.mark.parametrize('data_loader_class', [InteractionsDataLoader,
                                               ApproximateNegativeSamplingInteractionsDataLoader])
def test_instantiate_data_loaders(ratings_matrix_for_interactions,
                                  sparse_ratings_matrix_for_interactions,
                                  df_for_interactions,
                                  data_loader_class):
    interactions_kwargs = {
        'num_negative_samples': 4,
    }
    common_data_loader_kwargs = {
        'batch_size': 5,
        'shuffle': False,
        'drop_last': False,
    }

    data_loader_class_1 = data_loader_class(mat=ratings_matrix_for_interactions,
                                            **interactions_kwargs,
                                            **common_data_loader_kwargs)
    data_loader_class_2 = data_loader_class(users=df_for_interactions['user_id'],
                                            items=df_for_interactions['item_id'],
                                            ratings=df_for_interactions['ratings'],
                                            **interactions_kwargs,
                                            **common_data_loader_kwargs)
    data_loader_class_3 = data_loader_class(interactions=data_loader_class_1.interactions,
                                            **common_data_loader_kwargs)

    np.testing.assert_equal(data_loader_class_1.interactions.toarray(),
                            data_loader_class_2.interactions.toarray())
    np.testing.assert_equal(data_loader_class_2.interactions.toarray(),
                            data_loader_class_3.interactions.toarray())

    assert (
        data_loader_class_1.interactions.num_users
        == data_loader_class_2.interactions.num_users
        == data_loader_class_3.interactions.num_users
        == data_loader_class_1.num_users
        == data_loader_class_2.num_users
        == data_loader_class_3.num_users
    )
    assert (
        data_loader_class_1.interactions.num_items
        == data_loader_class_2.interactions.num_items
        == data_loader_class_3.interactions.num_items
        == data_loader_class_1.num_items
        == data_loader_class_2.num_items
        == data_loader_class_3.num_items
    )
    assert (
        data_loader_class_1.interactions.num_interactions
        == data_loader_class_2.interactions.num_interactions
        == data_loader_class_3.interactions.num_interactions
        == data_loader_class_1.num_interactions
        == data_loader_class_2.num_interactions
        == data_loader_class_3.num_interactions
    )

    data_loader_class_1_first_batch = next(iter(data_loader_class_1))
    data_loader_class_2_first_batch = next(iter(data_loader_class_2))
    data_loader_class_3_first_batch = next(iter(data_loader_class_3))

    for idx in range(len(data_loader_class_1_first_batch[0])):
        assert (
            len(data_loader_class_1_first_batch[0][idx])
            == len(data_loader_class_2_first_batch[0][idx])
            == len(data_loader_class_3_first_batch[0][idx])
            == common_data_loader_kwargs['batch_size']
        )

        assert (
            data_loader_class_1_first_batch[0][idx].tolist()
            == data_loader_class_2_first_batch[0][idx].tolist()
            == data_loader_class_3_first_batch[0][idx].tolist()
        )

    assert (
        data_loader_class_1_first_batch[1].shape
        == data_loader_class_2_first_batch[1].shape
        == data_loader_class_3_first_batch[1].shape
        == (common_data_loader_kwargs['batch_size'],
            interactions_kwargs['num_negative_samples'])
    )


def test_explicit_interactions_does_not_work_with_approximate_dataloader(
    explicit_interactions_pandas,
):
    with pytest.raises(ValueError):
        ApproximateNegativeSamplingInteractionsDataLoader(interactions=explicit_interactions_pandas)


def test_instantiate_data_loaders_explicit(explicit_interactions_pandas):
    common_data_loader_kwargs = {
        'batch_size': 5,
        'shuffle': False,
        'drop_last': False,
    }

    data_loader_class = InteractionsDataLoader(interactions=explicit_interactions_pandas,
                                               **common_data_loader_kwargs)

    assert data_loader_class.interactions.num_users == data_loader_class.num_users
    assert data_loader_class.interactions.num_items == data_loader_class.num_items

    data_loader_class_first_batch = next(iter(data_loader_class))

    # ensure that the format for implicit data is:
    # ``(X, Y, Z) = (user IDs, item IDs, ratings)``
    assert len(data_loader_class_first_batch) == 3
    assert len(data_loader_class_first_batch[0]) == common_data_loader_kwargs['batch_size']
    assert len(data_loader_class_first_batch[1]) == common_data_loader_kwargs['batch_size']
    assert len(data_loader_class_first_batch[2]) == common_data_loader_kwargs['batch_size']

    expected_repr = (
        'InteractionsDataLoader object with 12 interactions between 6 users and 10 items, returning'
        ' explicit, non-shuffled batches of size 5.'
    )
    assert str(data_loader_class) == expected_repr


def test_hdf5_interactions_dataloader_attributes(df_for_interactions, hdf5_pandas_df_path):
    interactions_dl = InteractionsDataLoader(users=df_for_interactions['user_id'],
                                             items=df_for_interactions['item_id'],
                                             num_negative_samples=5)

    hdf5_interactions_dl = HDF5InteractionsDataLoader(hdf5_path=hdf5_pandas_df_path,
                                                      user_col='user_id',
                                                      item_col='item_id',
                                                      num_negative_samples=5)

    assert hdf5_interactions_dl.num_users == interactions_dl.num_users
    assert hdf5_interactions_dl.num_items == interactions_dl.num_items
    assert hdf5_interactions_dl.num_negative_samples == interactions_dl.num_negative_samples
    assert hdf5_interactions_dl.num_interactions == interactions_dl.num_interactions

    with pytest.raises(AttributeError):
        hdf5_interactions_dl.mat


def test_all_data_loaders_output_equal(df_for_interactions, hdf5_pandas_df_path, tmpdir, capfd):
    common_data_loader_kwargs = {
        'num_negative_samples': 4,
        'batch_size': 5,
        'shuffle': False,
        'drop_last': False,
    }

    interactions_dl = InteractionsDataLoader(users=df_for_interactions['user_id'],
                                             items=df_for_interactions['item_id'],
                                             **common_data_loader_kwargs)
    approx_dl = (
        ApproximateNegativeSamplingInteractionsDataLoader(users=df_for_interactions['user_id'],
                                                          items=df_for_interactions['item_id'],
                                                          **common_data_loader_kwargs)
    )
    hdf5_interactions_dl = HDF5InteractionsDataLoader(hdf5_path=hdf5_pandas_df_path,
                                                      user_col='user_id',
                                                      item_col='item_id',
                                                      **common_data_loader_kwargs)

    expected_repr = (
        '{} object with 12 interactions between 6 users and 10 items, returning 4 negative samples'
        ' per implicit interaction in non-shuffled batches of size 5.'
    )

    assert str(interactions_dl) == expected_repr.format(str(type(interactions_dl).__name__))
    assert str(approx_dl) == expected_repr.format(str(type(approx_dl).__name__))
    assert (
        str(hdf5_interactions_dl) == expected_repr.format(str(type(hdf5_interactions_dl).__name__))
    )

    assert interactions_dl.num_users == approx_dl.num_users == hdf5_interactions_dl.num_users
    assert interactions_dl.num_items == approx_dl.num_items == hdf5_interactions_dl.num_items
    assert (
        interactions_dl.num_interactions
        == approx_dl.num_interactions
        == hdf5_interactions_dl.num_interactions
    )

    # get all batches from every DataLoader, add them to a list for comparison below
    def get_all_batches_from_DataLoader(dataloader, batch_size):
        all_batches = list()
        for idx, batch in enumerate(dataloader):
            assert len(batch[0][0]) == len(batch[0][1]) == len(batch[1])

            if idx < len(dataloader) - 1:
                assert len(batch[0][0]) == batch_size

            all_batches.append(batch)

        return all_batches

    interactions_batches = get_all_batches_from_DataLoader(interactions_dl,
                                                           batch_size=interactions_dl.batch_size)
    approximate_batches = get_all_batches_from_DataLoader(
        approx_dl,
        batch_size=approx_dl.approximate_negative_sampler.batch_size,
    )
    hdf5_batches = get_all_batches_from_DataLoader(
        hdf5_interactions_dl,
        batch_size=hdf5_interactions_dl.hdf5_sampler.batch_size,
    )

    for idx in range(len(interactions_batches)):
        assert (
            interactions_batches[idx][0][0].tolist()
            == approximate_batches[idx][0][0].tolist()
            == hdf5_batches[idx][0][0].tolist()
        )
        assert (
            interactions_batches[idx][0][1].tolist()
            == approximate_batches[idx][0][1].tolist()
            == hdf5_batches[idx][0][1].tolist()
        )
        # random negative samples will never be exactly equal
        assert (
            interactions_batches[idx][1].shape
            == approximate_batches[idx][1].shape
            == hdf5_batches[idx][1].shape
        )

    # test that our last batch is less than the specified batch size and that is okay for HDF5 data
    assert len(interactions_batches[-1][0][0]) < interactions_dl.batch_size
    assert len(approximate_batches[-1][0][0]) < approx_dl.approximate_negative_sampler.batch_size
    assert len(hdf5_batches[-1][0][0]) < hdf5_interactions_dl.hdf5_sampler.batch_size

    # ensure that the format for implicit data is:
    # ``((X, Y), Z) = ((user IDs, item IDs), negative item IDs)``
    assert (
        len(interactions_batches[0][0])
        == len(approximate_batches[0][0])
        == len(hdf5_batches[0][0])
        == 2
    )
    assert (
        len(interactions_batches[0])
        == len(approximate_batches[0])
        == len(hdf5_batches[0])
        == 2
    )
