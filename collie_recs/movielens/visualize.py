import random
from typing import Any, Iterable, Optional

import pandas as pd

import collie_recs
from collie_recs.movielens import (read_movielens_df,
                                   read_movielens_df_item,
                                   read_movielens_posters_df)
from collie_recs.utils import df_to_html


def get_recommendation_visualizations(
    model: collie_recs.model.BasePipeline,
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
    Get visual recommendations through Movielens posters for a given user.

    Parameters
    -------------
    model: collie_recs.model.BasePipeline
    user_id: int
        User ID to pull recommendations for
    df_user: DataFrame
        ``u.data`` from MovieLens data. If ``None``, will set to the output of
        ``read_movielens_df(decrement_ids=False)``
        Note: User and item IDs should start at 1 for this, which is the default for MovieLens 100K
    df_item: DataFrame
        ``u.item`` from MovieLens data. If ``None``, will set to the output of
        ``read_movielens_df_item()``
    movielens_posters_df: DataFrame
        DataFrame containing item_ids from MovieLens data and the poster url. If ``None``, will set
        to the output of ``read_movielens_posters_df()``
    num_user_movies_to_display: int
        Number of movies rated 4 or 5 to display for the user
    num_similar_movies: int
        Number of movies recommendations to display
    filter_films: bool
        Filter already-seen films from recommendations
    shuffle: bool
        Shuffle user-loved movie order
    detailed: bool
        Of the top N unfiltered recommendations, displays how many movies the user gave a positive
        and negative rating to
    image_width: int
        Image width for HTML images

    Returns
    -------------
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
    html += _get_posters_html(movielens_posters_df, df_item, user_liked_movies,
                              col_description='Some loved films:', image_width=image_width)
    html += _get_posters_html(movielens_posters_df, df_item, top_movies_k.index + 1)

    if detailed:
        loved_movies = df_user.query(f'user_id=={user_id} and (rating >= 4)')
        loved_movies = loved_movies.item_id.tolist()
        hated_movies = df_user.query(f'user_id=={user_id} and (rating < 4)')
        hated_movies = hated_movies.item_id.tolist()

        unfiltered_top_movies = model.get_item_predictions(user_id - 1,
                                                           unseen_items_only=False,
                                                           sort_values=True)
        unfiltered_top_movies_k = (
            (unfiltered_top_movies[:num_similar_movies].index + 1).tolist()
        )

        percent_captured = round(len(set(loved_movies) & set(unfiltered_top_movies_k))
                                 / num_similar_movies * 100, 3)
        percent_bad = round(len(set(hated_movies) & set(unfiltered_top_movies_k))
                            / num_similar_movies * 100, 3)

        html += '-----'
        html += f'<p style="margin:0">User {user_id} has rated <strong>{len(loved_movies)}'
        html += '</strong> films with a 4 or 5</p>'
        html += f'<p style="margin:0">User {user_id} has rated <strong>{len(hated_movies)}'
        html += '</strong> films with a 1, 2, or 3</p>'
        html += '<p style="margin:0">% of these films rated 5 or 4 appearing in the '
        html += f'first {num_similar_movies} recommendations:'
        html += f'<strong style="color:green">{percent_captured}%</strong></p>'
        html += '<p style="margin:0">% of these films rated 1, 2, or 3 appearing in the '
        html += f'first {num_similar_movies} recommendations: '
        html += f'<strong style="color:red">{percent_bad}%</strong></p>'

    return html


def _get_posters_html(movielens_posters_df: pd.DataFrame,
                      df_item: pd.DataFrame,
                      item_ids: Iterable[Any],
                      col_description: str = 'Recommended films:',
                      titles_provided: bool = False,
                      image_width: Optional[int] = 500) -> str:
    if type(item_ids) == int:
        item_ids = [item_ids]

    final = pd.Series(dtype='object')
    if not titles_provided:
        try:
            top_movies_titles = [df_item[df_item['item_id'] == x]
                                 ['movie_title'].iloc[0] for x in item_ids]
        except IndexError as e:
            raise IndexError('Ensure user and item IDs start at 1 for MovieLens 100K data:', e)
    else:
        top_movies_titles = item_ids
    for x in item_ids:
        url = _get_single_poster_html(movielens_posters_df, x)
        final = final.append(pd.Series(url))

    final_df = final.to_frame()

    final_df.index = top_movies_titles
    final_df.columns = [col_description]

    return df_to_html(final_df,
                      image_cols=[col_description],
                      transpose=True,
                      image_width=image_width)


def _get_single_poster_html(movielens_posters_df: pd.DataFrame, item_id: Any) -> str:
    try:
        url = movielens_posters_df.query(f'item_id == {item_id}').values[0][1]
    except (ValueError, TypeError):
        url = ' '

    return url
