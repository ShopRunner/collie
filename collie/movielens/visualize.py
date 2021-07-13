import collections
from contextlib import suppress
import random
from typing import Any, Iterable, Optional, Union

import pandas as pd

import collie
from collie.movielens import read_movielens_df, read_movielens_df_item, read_movielens_posters_df
from collie.utils import df_to_html


def get_recommendation_visualizations(
    model: collie.model.BasePipeline,
    user_id: int,
    df_user: Optional[pd.DataFrame] = None,
    df_item: Optional[pd.DataFrame] = None,
    movielens_posters_df: Optional[pd.DataFrame] = None,
    num_user_movies_to_display: int = 10,
    num_similar_movies: int = 10,
    filter_films: bool = True,
    shuffle: bool = True,
    detailed: bool = False,
    image_width: int = 500,
) -> str:
    """
    Visualize Movielens 100K recommendations for a given user.

    Parameters
    ----------
    model: collie.model.BasePipeline
    user_id: int
        User ID to retrieve recommendations for
    df_user: DataFrame
        ``u.data`` from MovieLens data. This DataFrame must have columns:

        * ``user_id`` (starting at ``1``)

        * ``item_id`` (starting at ``1``)

        * ``rating`` (explicit ratings)

        If ``None``, will set to the output of ``read_movielens_df(decrement_ids=False)``.
    df_item: DataFrame
        ``u.item`` from MovieLens data. This DataFrame must have columns:

        * ``item_id`` (starting at ``1``)

        * ``movie_title``

        If ``None``, will set to the output of ``read_movielens_df_item()``
    movielens_posters_df: DataFrame
        DataFrame containing item_ids from MovieLens data and the poster url. This DataFrame must
        have columns:

        * ``item_id`` (starting at ``1``)

        * ``url``

        If ``None``, will set to the output of ``read_movielens_posters_df()``
    num_user_movies_to_display: int
        Number of movies rated ``4`` or ``5`` to display for the user
    num_similar_movies: int
        Number of movies recommendations to display
    filter_films: bool
        Filter films out of recommendations if the user has already interacted with them
    shuffle: bool
        Shuffle order of ``num_user_movies_to_display`` films
    detailed: bool
        Of the top ``N`` unfiltered recommendations, display how many movies the user gave a
        positive and negative rating to
    image_width: int
        Image width for HTML images

    Returns
    -------
    html: str
        HTML string of movies a user loved and the model recommended for a given user, ready for
        displaying

    """
    assert num_similar_movies > 0, 'Number of similar movies returned must be 1 or greater.'

    if df_user is None:
        df_user = read_movielens_df(decrement_ids=False)

    if df_item is None:
        df_item = read_movielens_df_item()

    if movielens_posters_df is None:
        movielens_posters_df = read_movielens_posters_df()

    if df_user['user_id'].min() != 1 or df_user['item_id'].min() != 1:
        raise ValueError(
            'Both user and item IDs must start at ``1`` for MovieLens 100K ``df_user`` data.'
        )
    if df_item['item_id'].min() != 1:
        raise ValueError(
            'Item IDs must start at ``1`` for MovieLens 100K ``df_item`` data.'
        )

    user_df = df_user.query(f'user_id=={user_id}')
    user_liked_movies = sorted(user_df[user_df['rating'] >= 4]['item_id'].tolist())

    if shuffle:
        random.shuffle(user_liked_movies)

    user_liked_movies = user_liked_movies[:num_user_movies_to_display]

    top_movies = model.get_item_predictions(user_id - 1,
                                            unseen_items_only=filter_films,
                                            sort_values=True)
    top_movies_k = top_movies[:num_similar_movies]

    if len(top_movies_k) == 0:
        if filter_films:
            raise ValueError(f'User {user_id} cannot have rated every movie.')
        else:
            raise ValueError(f'User {user_id} has no top rated films.')

    html = f'<h3>User {user_id}:</h3>'
    html += _get_posters_html(movielens_posters_df=movielens_posters_df,
                              df_item=df_item,
                              item_ids=user_liked_movies,
                              col_description='Some loved films:',
                              image_width=image_width)
    html += _get_posters_html(movielens_posters_df=movielens_posters_df,
                              df_item=df_item,
                              item_ids=(top_movies_k.index + 1),
                              col_description='Recommended films:',
                              image_width=image_width)

    if detailed:
        loved_movies = df_user.query(f'user_id == {user_id} and (rating >= 4)')
        loved_movies = loved_movies.item_id.tolist()
        hated_movies = df_user.query(f'user_id == {user_id} and (rating < 4)')
        hated_movies = hated_movies.item_id.tolist()

        unfiltered_top_movies = model.get_item_predictions(user_id - 1,
                                                           unseen_items_only=False,
                                                           sort_values=True)
        unfiltered_top_movies_k = (unfiltered_top_movies[:num_similar_movies].index + 1).tolist()

        percent_captured = round(
            len(set(loved_movies) & set(unfiltered_top_movies_k)) / num_similar_movies * 100, 3
        )
        percent_bad = round(
            len(set(hated_movies) & set(unfiltered_top_movies_k)) / num_similar_movies * 100, 3
        )

        html += (
            '-----'
            f'<p style="margin:0">User {user_id} has rated <strong>{len(loved_movies)}'
            '</strong> films with a 4 or 5</p>'
            f'<p style="margin:0">User {user_id} has rated <strong>{len(hated_movies)}'
            '</strong> films with a 1, 2, or 3</p>'
            '<p style="margin:0">% of these films rated 5 or 4 appearing in the '
            f'first {num_similar_movies} recommendations:'
            f'<strong style="color:green">{percent_captured}%</strong></p>'
            '<p style="margin:0">% of these films rated 1, 2, or 3 appearing in the '
            f'first {num_similar_movies} recommendations: '
            f'<strong style="color:red">{percent_bad}%</strong></p>'
        )

    return html


def _get_posters_html(movielens_posters_df: pd.DataFrame,
                      df_item: pd.DataFrame,
                      item_ids: Union[int, Iterable[Any]],
                      col_description: str = 'Recommended films:',
                      image_width: Optional[int] = 500) -> str:
    if not isinstance(item_ids, collections.abc.Iterable):
        item_ids = [item_ids]

    top_movies_titles = [
        df_item[df_item['item_id'] == x]['movie_title'].iloc[0] for x in item_ids
    ]
    final_urls = []

    for item_id in item_ids:
        url = ''
        with suppress((ValueError, TypeError)):
            url = movielens_posters_df.query(f'item_id == {item_id}')['url'].item()

        final_urls.append(url)

    final_df = pd.DataFrame(final_urls)

    final_df.index = top_movies_titles
    final_df.columns = [col_description]

    return df_to_html(df=final_df,
                      image_cols=[col_description],
                      transpose=True,
                      image_width=image_width)
