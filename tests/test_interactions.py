import numpy as np
import pandas as pd
import pytest

from collie_recs.interactions import (ApproximateNegativeSamplingInteractionsDataLoader,
                                      HDF5Interactions,
                                      HDF5InteractionsDataLoader,
                                      Interactions,
                                      InteractionsDataLoader)


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


def test_Interactions_with_missing_ids(df_for_interactions_with_missing_ids,
                                       ratings_matrix_for_interactions_with_missing_ids,
                                       sparse_ratings_matrix_for_interactions_with_missing_ids):
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


def test_Interactions_with_invalid_lengths(df_for_interactions_with_missing_ids):
    with pytest.raises(ValueError):
        Interactions(users=df_for_interactions_with_missing_ids['user_id'],
                     items=df_for_interactions_with_missing_ids['item_id'][:-1],
                     ratings=df_for_interactions_with_missing_ids['ratings'],
                     allow_missing_ids=True,
                     check_num_negative_samples_is_valid=False)

    with pytest.raises(ValueError):
        Interactions(users=df_for_interactions_with_missing_ids['user_id'][:-1],
                     items=df_for_interactions_with_missing_ids['item_id'],
                     ratings=df_for_interactions_with_missing_ids['ratings'],
                     allow_missing_ids=True,
                     check_num_negative_samples_is_valid=False)

    with pytest.raises(ValueError):
        Interactions(users=df_for_interactions_with_missing_ids['user_id'],
                     items=df_for_interactions_with_missing_ids['item_id'],
                     ratings=df_for_interactions_with_missing_ids['ratings'][:-1],
                     allow_missing_ids=True,
                     check_num_negative_samples_is_valid=False)

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


def test_bad_Interactions_instantiation(df_for_interactions):
    with pytest.raises(AssertionError):
        Interactions(users=df_for_interactions['user_id'],
                     items=None,
                     ratings=df_for_interactions['ratings'],
                     check_num_negative_samples_is_valid=False)

    with pytest.raises(AssertionError):
        Interactions(users=None,
                     items=df_for_interactions['item_id'],
                     ratings=df_for_interactions['ratings'],
                     check_num_negative_samples_is_valid=False)

    Interactions(users=df_for_interactions['user_id'],
                 items=df_for_interactions['item_id'],
                 ratings=None,
                 check_num_negative_samples_is_valid=False)


def test_Interactions_data_methods(interactions_matrix,
                                   interactions_pandas,
                                   sparse_ratings_matrix_for_interactions):
    assert np.array_equal(interactions_matrix.todense(), interactions_pandas.todense())
    assert np.array_equal(interactions_matrix.todense(),
                          sparse_ratings_matrix_for_interactions.todense())

    assert np.array_equal(interactions_matrix.toarray(), interactions_pandas.toarray())
    assert np.array_equal(interactions_matrix.toarray(),
                          sparse_ratings_matrix_for_interactions.toarray())

    assert np.array_equal(interactions_matrix.head(), interactions_pandas.head())
    assert np.array_equal(interactions_matrix.head(),
                          sparse_ratings_matrix_for_interactions.toarray()[:5])

    assert np.array_equal(interactions_matrix.tail(), interactions_pandas.tail())
    assert np.array_equal(interactions_matrix.tail(),
                          sparse_ratings_matrix_for_interactions.toarray()[-5:])

    assert len(interactions_matrix.head(3)) == len(interactions_pandas.head(3)) == 3
    assert len(interactions_matrix.tail(3)) == len(interactions_pandas.tail(3)) == 3

    assert (
        len(interactions_matrix.head(42))
        == len(interactions_pandas.head(42))
        == interactions_matrix.num_users
    )
    assert (
        len(interactions_matrix.tail(42))
        == len(interactions_pandas.tail(42))
        == interactions_matrix.num_users
    )

    assert (
        len(interactions_matrix.head(-1))
        == len(interactions_pandas.head(-1))
        == interactions_matrix.num_users
    )
    assert (
        len(interactions_matrix.tail(-1))
        == len(interactions_pandas.tail(-1))
        == interactions_matrix.num_users
    )


def test_Interactions_approximate_negative_samples(ratings_matrix_for_interactions):
    interactions = Interactions(mat=ratings_matrix_for_interactions,
                                num_negative_samples=NUM_NEGATIVE_SAMPLES,
                                max_number_of_samples_to_consider=0,
                                seed=42)

    assert interactions.positive_items == {}

    for _ in range(3):
        _, negative_samples = interactions[0]

        assert len(negative_samples) == 3


def test_Interactions_approximate_negative_samples_many_users(ratings_matrix_for_interactions):
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


def test_Interactions_approximate_negative_samples_partway_through(ratings_matrix_for_interactions):
    with pytest.warns(UserWarning):
        interactions = Interactions(mat=ratings_matrix_for_interactions,
                                    num_negative_samples=NUM_NEGATIVE_SAMPLES,
                                    max_number_of_samples_to_consider=1,
                                    seed=42)

    assert interactions.positive_items != {}

    for _ in range(3):
        _, negative_samples = interactions[0]

        assert len(negative_samples) == 3


def test_Interactions_exact_negative_samples(ratings_matrix_for_interactions):
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


def test_Interactions_exact_negative_samples_many_users(ratings_matrix_for_interactions):
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
    ratings_matrix_for_interactions
):
    with pytest.raises(AssertionError):
        Interactions(mat=ratings_matrix_for_interactions,
                     max_number_of_samples_to_consider=200,
                     num_negative_samples=8)


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


def test_HDF5Interactions_head_tail(hdf5_interactions, df_for_interactions):
    pd.testing.assert_frame_equal(hdf5_interactions.head(3), df_for_interactions.head(3))
    pd.testing.assert_frame_equal(hdf5_interactions.tail(3), df_for_interactions.tail(3))

    pd.testing.assert_frame_equal(hdf5_interactions.head(42), df_for_interactions.head(42))
    pd.testing.assert_frame_equal(hdf5_interactions.tail(42), df_for_interactions.tail(42))

    pd.testing.assert_frame_equal(hdf5_interactions.head(-1), df_for_interactions.head(-1))
    pd.testing.assert_frame_equal(hdf5_interactions.tail(-1), df_for_interactions.tail(-1))


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
        ' per interaction in non-shuffled batches of size 5.'
    )

    assert str(interactions_dl) == expected_repr.format(str(type(interactions_dl).__name__))
    assert str(approx_dl) == expected_repr.format(str(type(approx_dl).__name__))
    assert (
        str(hdf5_interactions_dl) == expected_repr.format(str(type(hdf5_interactions_dl).__name__))
    )

    out, _ = capfd.readouterr()
    assert '``meta`` key not found - generating ``num_users`` and ``num_items``' in out

    assert interactions_dl.num_users == approx_dl.num_users == hdf5_interactions_dl.num_users
    assert interactions_dl.num_items == approx_dl.num_items == hdf5_interactions_dl.num_items
    assert (
        interactions_dl.num_interactions
        == approx_dl.num_interactions
        == hdf5_interactions_dl.num_interactions
    )

    interactions_dl_first_batch = next(iter(interactions_dl))
    approx_dl_first_batch = next(iter(approx_dl))
    hdf5_interactions_dl_first_batch = next(iter(hdf5_interactions_dl))

    # test the shapes of outputs are equal
    for idx in range(len(interactions_dl_first_batch[0])):
        assert (
            len(interactions_dl_first_batch[0][idx])
            == len(approx_dl_first_batch[0][idx])
            == len(hdf5_interactions_dl_first_batch[0][idx])
            == common_data_loader_kwargs['batch_size']
        )

    assert (
        interactions_dl_first_batch[1].shape
        == approx_dl_first_batch[1].shape
        == hdf5_interactions_dl_first_batch[1].shape
        == (common_data_loader_kwargs['batch_size'],
            common_data_loader_kwargs['num_negative_samples'])
    )

    # get all batches from every DataLoader, add them to a list for comparison below
    interactions_batches = list()
    for idx, batch in enumerate(interactions_dl):
        assert len(batch[0][0]) == len(batch[0][1]) == len(batch[1])
        if idx < len(interactions_dl) - 1:
            assert len(batch[0][0]) == interactions_dl.batch_size

        interactions_batches.append(batch)

    approximate_batches = list()
    for idx, batch in enumerate(approx_dl):
        assert len(batch[0][0]) == len(batch[0][1]) == len(batch[1])
        if idx < len(approx_dl) - 1:
            assert len(batch[0][0]) == approx_dl.approximate_negative_sampler.batch_size

        approximate_batches.append(batch)

    hdf5_batches = list()
    for idx, batch in enumerate(hdf5_interactions_dl):
        assert len(batch[0][0]) == len(batch[0][1]) == len(batch[1])
        if idx < len(hdf5_interactions_dl) - 1:
            assert len(batch[0][0]) == hdf5_interactions_dl.hdf5_sampler.batch_size

        hdf5_batches.append(batch)

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

    interactions_last_batch = interactions_batches[-1]
    approximate_last_batch = approximate_batches[-1]
    hdf5_last_batch = hdf5_batches[-1]

    # test that our last batch is less than the specified batch size and that is okay for HDF5 data
    assert len(interactions_last_batch[0][0]) < interactions_dl.batch_size
    assert len(approximate_last_batch[0][0]) < approx_dl.approximate_negative_sampler.batch_size
    assert len(hdf5_last_batch[0][0]) < hdf5_interactions_dl.hdf5_sampler.batch_size
