from unittest import mock
import urllib

import pandas as pd
import pytest

from collie.movielens import (get_recommendation_visualizations,
                              read_movielens_posters_df,
                              run_movielens_example)


def test_read_movielens_df(movielens_explicit_df, movielens_explicit_df_not_decremented):
    assert movielens_explicit_df.shape == (100000, 4)
    assert movielens_explicit_df_not_decremented.shape == (100000, 4)

    assert movielens_explicit_df['user_id'].min() == movielens_explicit_df['item_id'].min() == 0
    assert (
        movielens_explicit_df_not_decremented['user_id'].min()
        == movielens_explicit_df_not_decremented['item_id'].min()
        == 1
    )


def test_read_movielens_df_item(movielens_df_item):
    assert movielens_df_item.shape == (1682, 23)


def test_read_movielens_df_user(movielens_df_user):
    assert movielens_df_user.shape == (943, 5)


def test_read_movielens_posters_df(movielens_posters_df):
    assert movielens_posters_df.shape == (1682, 2)


@mock.patch('os.path.exists')
def test_read_movielens_posters_df_no_local_file(os_path_exists_mock, movielens_posters_df):
    os_path_exists_mock.return_value = False

    expected = movielens_posters_df

    try:
        actual = read_movielens_posters_df()
    except urllib.error.URLError:
        pytest.skip('Internet connection not available - cannot run this test!')

    pd.testing.assert_frame_equal(actual, expected)


def test_get_movielens_metadata(movielens_metadata_df):
    assert len(movielens_metadata_df) == 1682

    expected_columns = [
        'genre_action',
        'genre_adventure',
        'genre_animation',
        'genre_children',
        'genre_comedy',
        'genre_crime',
        'genre_documentary',
        'genre_drama',
        'genre_fantasy',
        'genre_film_noir',
        'genre_horror',
        'genre_musical',
        'genre_mystery',
        'genre_romance',
        'genre_sci_fi',
        'genre_thriller',
        'genre_war',
        'genre_western',
        'genre_unknown',
        'decade_unknown',
        'decade_20',
        'decade_30',
        'decade_40',
        'decade_50',
        'decade_60',
        'decade_70',
        'decade_80',
        'decade_90',
    ]

    assert movielens_metadata_df.columns.tolist() == expected_columns


def test_get_user_metadata(user_metadata_df):
    assert len(user_metadata_df) == 943

    expected_columns = [
        'age',
        'female',
        'occupation_administrator',
        'occupation_artist',
        'occupation_doctor',
        'occupation_educator',
        'occupation_engineer',
        'occupation_entertainment',
        'occupation_executive',
        'occupation_healthcare',
        'occupation_homemaker',
        'occupation_lawyer',
        'occupation_librarian',
        'occupation_marketing',
        'occupation_none',
        'occupation_other',
        'occupation_programmer',
        'occupation_retired',
        'occupation_salesman',
        'occupation_scientist',
        'occupation_student',
        'occupation_technician',
        'occupation_writer',
    ]

    assert user_metadata_df.columns.tolist() == expected_columns


@mock.patch('collie.model.MatrixFactorizationModel')
def test_run_movielens_example(save_model_mock, gpu_count):
    save_model_mock.save_model.side_effect = print('Saved.')

    run_movielens_example(epochs=1, gpus=gpu_count)


@pytest.mark.parametrize('supply_df_user', [False, True])
@pytest.mark.parametrize('supply_df_item', [False, True])
@pytest.mark.parametrize('supply_movielens_posters_df', [False, True])
def test_run_get_recommendation_visualizations(implicit_model,
                                               movielens_explicit_df_not_decremented,
                                               movielens_df_item,
                                               movielens_posters_df,
                                               supply_df_user,
                                               supply_df_item,
                                               supply_movielens_posters_df):
    html_not_detailed = get_recommendation_visualizations(
        model=implicit_model,
        user_id=42,
        df_user=movielens_explicit_df_not_decremented if supply_df_user else None,
        df_item=movielens_df_item if supply_df_item else None,
        movielens_posters_df=movielens_posters_df if supply_movielens_posters_df else None,
        filter_films=False,
        shuffle=False,
        detailed=False,
    )
    assert len(html_not_detailed) > 0

    html_detailed = get_recommendation_visualizations(
        model=implicit_model,
        user_id=42,
        df_user=movielens_explicit_df_not_decremented if supply_df_user else None,
        df_item=movielens_df_item if supply_df_item else None,
        movielens_posters_df=movielens_posters_df if supply_movielens_posters_df else None,
        filter_films=True,
        shuffle=True,
        detailed=True,
    )
    assert len(html_detailed) > 0

    assert len(html_not_detailed) < len(html_detailed)


def test_run_get_recommendation_visualizations_not_decremented(
    implicit_model, movielens_explicit_df, movielens_df_item
):
    with pytest.raises(ValueError):
        get_recommendation_visualizations(
            model=implicit_model,
            user_id=42,
            df_user=movielens_explicit_df,
        )

    movielens_df_item_not_decremented = movielens_df_item.copy()
    movielens_df_item_not_decremented['item_id'] -= 1

    with pytest.raises(ValueError):
        get_recommendation_visualizations(
            model=implicit_model,
            user_id=42,
            df_item=movielens_df_item_not_decremented,
        )


def test_run_get_recommendation_visualizations_invalid_user(
    implicit_model, movielens_explicit_df, movielens_df_item
):
    def bad_get_item_predictions_function(*args, **kwargs):
        return []

    implicit_model.get_item_predictions = bad_get_item_predictions_function

    with pytest.raises(ValueError) as error_1:
        get_recommendation_visualizations(model=implicit_model, user_id=42, filter_films=True)

        assert 'User 42 cannot have rated every movie.' in str(error_1.value)

    with pytest.raises(ValueError) as error_2:
        get_recommendation_visualizations(model=implicit_model, user_id=42, filter_films=False)

        assert 'User 42 has no top rated films.' in str(error_2.value)
