from unittest import mock

import numpy as np
import pandas as pd
import pytest

from collie_recs.utils import (convert_to_implicit,
                               create_ratings_matrix,
                               df_to_html,
                               df_to_interactions,
                               get_init_arguments,
                               remove_users_with_fewer_than_n_interactions,
                               Timer)


EXPECTED_RATINGS_MATRIX = np.array([[1, 2, 3, 0],
                                    [0, 3, 0, 5],
                                    [1, 0, 0, 0],
                                    [0, 4, 0, 0]])


def test_create_ratings_matrix_not_sparse(explicit_df):
    actual = create_ratings_matrix(df=explicit_df,
                                   user_col='userId',
                                   item_col='itemId',
                                   ratings_col='rating',
                                   sparse=False)

    np.testing.assert_equal(actual, EXPECTED_RATINGS_MATRIX)


def test_create_ratings_matrix_sparse(explicit_df):
    actual = create_ratings_matrix(df=explicit_df,
                                   user_col='userId',
                                   item_col='itemId',
                                   ratings_col='rating',
                                   sparse=True)

    np.testing.assert_equal(actual.toarray(), EXPECTED_RATINGS_MATRIX)


def test_create_ratings_matrix_not_sparse_starting_at_one(explicit_df):
    explicit_df['userId'] += 1
    explicit_df['itemId'] += 1

    with pytest.raises(ValueError):
        create_ratings_matrix(df=explicit_df,
                              user_col='userId',
                              item_col='itemId',
                              ratings_col='rating',
                              sparse=False)


def test_create_ratings_matrix_sparse_starting_at_one(explicit_df):
    explicit_df['userId'] += 1
    explicit_df['itemId'] += 1

    with pytest.raises(ValueError):
        create_ratings_matrix(df=explicit_df,
                              user_col='userId',
                              item_col='itemId',
                              ratings_col='rating',
                              sparse=True)


def test_df_to_interactions(df_to_turn_to_interactions):
    expected = np.array([[0, 0, 1, 0, 0],
                         [0, 1, 1, 1, 0],
                         [0, 0, 1, 0, 1],
                         [1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0]])

    actual = df_to_interactions(df=df_to_turn_to_interactions,
                                user_col='userId',
                                item_col='itemId',
                                ratings_col=None,
                                check_num_negative_samples_is_valid=False)

    np.testing.assert_equal(actual.toarray(), expected)


def test_explicit_to_implicit(explicit_df):
    expected = pd.DataFrame(data={
        'userId': [1, 3],
        'itemId': [3, 1],
        'rating': [1, 1]
    })

    actual = convert_to_implicit(explicit_df)

    pd.testing.assert_frame_equal(actual, expected)


def test_remove_users_with_fewer_than_n_interactions(df_with_users_interacting_only_once):
    expected = pd.DataFrame(data={
        'userId': [0, 1, 1, 2, 2, 2, 3, 3, 3, 0],
        'itemId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 2],
    })

    actual = remove_users_with_fewer_than_n_interactions(df=df_with_users_interacting_only_once,
                                                         min_num_of_interactions=2,
                                                         user_col='userId')

    pd.testing.assert_frame_equal(actual, expected)


def test_get_init_arguments_just_args():
    expected = {
        'var_1': 'hey',
        'var_2': 54321,
    }

    class TestClass():
        def __init__(self, var_1, var_2=54321):
            super().__init__()

            self.actual = get_init_arguments()

    actual = TestClass('hey').actual

    assert actual == expected


def test_get_init_arguments_just_kwargs():
    expected = {
        'var_1': 'hello',
        'var_2': 12345,
    }

    class TestClass():
        def __init__(self, **kwargs):
            super().__init__()

            self.actual = get_init_arguments()

    actual = TestClass(var_1='hello', var_2=12345).actual

    assert actual == expected


def test_get_init_arguments_both_args_and_kwargs():
    expected = {
        'var_1': 'greetings',
        'var_2': 2468,
        'var_3': 'yes'
    }

    class TestClass():
        def __init__(self, var_1, var_2=12345, **kwargs):
            super().__init__()

            self.actual = get_init_arguments()

    actual = TestClass('greetings', 2468, var_3='yes').actual

    assert actual == expected


def test_df_to_html(df_html_test):
    expected = """
        <table border="1" class="dataframe">
            <thead>
            <tr style="text-align: right;">

                <th></th>
                <th>title</th>
                <th>description</th>
                <th>link</th>

                <th>image</th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <th>0</th>

                <td><h2>Greg</h2></td>
                <td>some text here</td>

                <td><a target="_blank" href=" https://madeupsite.com">
                    https://madeupsite.com</a></td>
                <td><img src="https://avatars0.githubusercontent.co
                        m/u/13399445"></td>
            </tr>
            <tr>
                <th>1</th>
                <td><h2>Real Greg</h2>
                </td>
                <td>more text here</td>
                <td><a target="_blank" href=" https://anotherm
                        adeupsite.com">https://anothermadeupsite.com</a></td>

                <td><img src="https://avatars3.githubusercontent.com/u/31417712"></td>
            </tr>

            </tbody>
        </table>
        """
    actual = df_to_html(
        df_html_test,
        image_cols='image',
        hyperlink_cols=['link', 'image'],
        html_tags={'title': 'h2'},
        max_num_rows=None,
    )

    assert expected.replace('\n', '').replace('  ', '') == (
        actual.replace('\n', '').replace('  ', '')
    )


def test_df_to_html_with_kwargs(df_html_test):
    expected = """
        <table border="1" class="dataframe for_css">
            <thead>
            <tr style="text-align: right;">

                <th>title</th>
                <th>description</th>
                <th>link</th>

                <th>image</th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <td><h2>Greg</h2></td>
                <td>some text here</td>

                <td><a target="_blank" href=" https://madeupsite.com">
                    https://madeupsite.com</a></td>
                <td><img src="https://avatars0.githubusercontent.co
                        m/u/13399445"></td>
            </tr>
            <tr>
                <td><h2>Real Greg</h2>
                </td>
                <td>more text here</td>
                <td><a target="_blank" href=" https://anotherm
                        adeupsite.com">https://anothermadeupsite.com</a></td>

                <td><img src="https://avatars3.githubusercontent.com/u/31417712"></td>
            </tr>

            </tbody>
        </table>
        """
    actual = df_to_html(
        df_html_test,
        image_cols='image',
        hyperlink_cols=['link', 'image'],
        html_tags={'title': 'h2'},
        max_num_rows=None,
        index=False,
        classes=['for_css'],
    )

    assert expected.replace('\n', '').replace('  ', '') == (
        actual.replace('\n', '').replace('  ', '')
    )


def test_df_to_html_with_max_num_rows(df_html_test):
    expected = """
        <table border="1" class="dataframe">
            <thead>
            <tr style="text-align: right;">
                <th></th>
                <th>title</th>
                <th>description</th>
                <th>link</th>
                <th>image</th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <th>0</th>
                <td><h2>Greg</h2></td>
                <td>some text here</td>
                <td>
                    <a target="_blank" href=" https://madeupsite.com">https://madeupsite.com</a>
                </td>
                <td>
                    <img src="https://avatars0.githubusercontent.com/u/13399445">
                </td>
            </tr>
            </tbody>
        </table>
        """
    actual = df_to_html(
        df_html_test,
        image_cols='image',
        hyperlink_cols=['link', 'image'],
        html_tags={'title': 'h2'},
        max_num_rows=1,
    )

    assert expected.replace('\n', '').replace('  ', '') == (
        actual.replace('\n', '').replace('  ', '')
    )


def test_df_to_html_with_number_colname(df_html_test):
    expected = """
        <table border="1" class="dataframe">
            <thead>
            <tr style="text-align: right;">

                <th></th>
                <th>title</th>
                <th>description</th>
                <th>link</th>

                <th>0</th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <th>0</th>

                <td><h2>Greg</h2></td>
                <td>some text here</td>

                <td><a target="_blank" href=" https://madeupsite.com">
                    https://madeupsite.com</a></td>
                <td><img src="https://avatars0.githubusercontent.co
                        m/u/13399445"></td>
            </tr>
            <tr>
                <th>1</th>
                <td><h2>Real Greg</h2>
                </td>
                <td>more text here</td>
                <td><a target="_blank" href=" https://anotherm
                        adeupsite.com">https://anothermadeupsite.com</a></td>

                <td><img src="https://avatars3.githubusercontent.com/u/31417712"></td>
            </tr>

            </tbody>
        </table>
        """
    df_html_test = df_html_test.rename(columns={'image': 0})
    actual = df_to_html(
        df_html_test,
        image_cols=0,
        hyperlink_cols=['link', 0],
        html_tags={'title': 'h2'},
        max_num_rows=None,
    )

    assert expected.replace('\n', '').replace('  ', '') == (
        actual.replace('\n', '').replace('  ', '')
    )


def test_timecheck_default_message(capsys):
    with mock.patch('time.time', mock.MagicMock(side_effect=[0, 60])):
        timer = Timer()
        actual = timer.timecheck()

    out, _ = capsys.readouterr()

    assert out == 'finished (1.00 min)\n'
    assert actual == 1


def test_timecheck_custom_message(capsys):
    with mock.patch('time.time', mock.MagicMock(side_effect=[120, 150])):
        timer = Timer()
        actual = timer.timecheck('I am a test!')

    out, _ = capsys.readouterr()

    assert out == 'I am a test! (0.50 min)\n'
    assert actual == 0.5


def test_time_since_start_default_message(capsys):
    with mock.patch('time.time', mock.MagicMock(side_effect=[120, 150, 180])):
        timer = Timer()
        timer.timecheck('Ignore me.')
        actual = timer.time_since_start()

    out, _ = capsys.readouterr()
    message = out.split('\n')[1] + '\n'

    assert message == 'total time: 1.00 min\n'
    assert actual == 1


def test_time_since_start_custom_message(capsys):
    with mock.patch('time.time', mock.MagicMock(side_effect=[120, 150, 180])):
        timer = Timer()
        timer.timecheck('Ignore me.')
        actual = timer.time_since_start('I am a test!')

    out, _ = capsys.readouterr()
    message = out.split('\n')[1] + '\n'

    assert message == 'I am a test!: 1.00 min\n'
    assert actual == 1
