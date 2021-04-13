from datetime import datetime
import inspect
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning
from scipy.sparse import coo_matrix
import torch

from collie_recs.interactions import Interactions


def get_random_seed() -> int:
    """Generate a random seed based on the current datetime."""
    return int(datetime.now().replace(microsecond=0).timestamp())


def create_ratings_matrix(df: pd.DataFrame,
                          user_col: str = 'user_id',
                          item_col: str = 'item_id',
                          ratings_col: str = 'rating',
                          sparse: bool = False) -> (coo_matrix, np.array):
    """
    Helper function to convert a Pandas DataFrame to 2-dimensional matrix.

    Parameters
    -------------
    df: pd.DataFrame
        Dataframe with columns for user IDs, item IDs, and ratings
    user_col: str
        Column name for the user IDs, starting at 0
    item_col: str
        Column name for the item IDs, starting at 0
    ratings_col: str
        Column name for the ratings column
    sparse: bool
        Whether to return data as a sparse ``coo_matrix`` (True) or np.array (False)

    Returns
    -------------
    ratings_matrix: np.array or scipy.sparse.coo_matrix, 2-d
        Data with users as rows, items as columns, and ratings as values

    """
    if sparse:
        ratings_matrix = _create_sparse_ratings_matrix_helper(users=df[user_col],
                                                              items=df[item_col],
                                                              ratings=df[ratings_col])
    else:
        if df[user_col].min() != 0 or df[item_col].min() != 0:
            raise ValueError(
                'Minimum values of ``df[user_col]`` and ``df[item_col]`` must both be 0.'
            )

        ratings_df = df.pivot(index=user_col, columns=item_col, values=ratings_col).fillna(0)
        ratings_matrix = ratings_df.to_numpy()

    return ratings_matrix


def _create_sparse_ratings_matrix_helper(users: Iterable[int],
                                         items: Iterable[int],
                                         ratings: Optional[Iterable[int]] = None,
                                         num_users: Union[int, str] = 'infer',
                                         num_items: Union[int, str] = 'infer') -> coo_matrix:
    """Create a sparse matrix from a series of arrays."""
    # we found that putting this function in ``utils.py`` gave an error since ``utils.py`` imports
    # ``interactions.py`` which imports ``utils.py`` and so on.
    if min(users) != 0 or min(items) != 0:
        raise ValueError('Minimum values of ``users`` and ``items`` must both be 0.')

    num_users = _infer_num_if_needed_for_1d_array(num_users, users)
    num_items = _infer_num_if_needed_for_1d_array(num_items, items)

    if ratings is None:
        ratings = np.ones_like(users)

    return coo_matrix(
        (np.array(ratings), (np.array(users), np.array(items))), shape=(num_users, num_items)
    )


def _infer_num_if_needed_for_1d_array(num: Union[int, str], array: Iterable[int]):
    """Return ``num`` or, if ``None``, the maximum value of ``array`` + 1."""
    if num == 'infer':
        num = max(array) + 1

    return num


def df_to_interactions(df: pd.DataFrame,
                       user_col: str = 'user_id',
                       item_col: str = 'item_id',
                       ratings_col: Optional[str] = 'rating',
                       **kwargs) -> Interactions:
    """
    Helper function to convert a DataFrame to an ``Interactions`` object.

    Parameters
    -------------
    df: pd.DataFrame
        Dataframe with columns for user IDs, item IDs, and (optionally) ratings
    user_col: str
        Column name for the user IDs
    item_col: str
        Column name for the item IDs
    ratings_col: str
        Column name for the ratings column. If ``None``, will default to ratings of all 1s
    **kwargs
        Keyword arguments to pass to ``Interactions``

    Returns
    -------------
    interactions: collie_recs.interactions.Interactions

    """
    ratings = df[ratings_col] if ratings_col is not None else None

    return Interactions(users=df[user_col], items=df[item_col], ratings=ratings, **kwargs)


def convert_to_implicit(explicit_df: pd.DataFrame,
                        min_rating_to_keep: Optional[float] = 4,
                        ratings_col: Optional[str] = 'rating') -> pd.DataFrame:
    """
    Convert explicit interactions data to implicit data.

    All scores that are ``>= min_rating_to_keep`` are kept and converted to an interaction score of
    1. Every other data point is removed.

    Parameters
    -------------
    explicit_df: pd.DataFrame
        Dataframe with explicit ratings in the rating column
    min_rating_to_keep: int
        Minimum rating to be considered a valid interaction
    ratings_col: str
        Column name for the ratings column

    Returns
    -------------
    implicit_df: pd.DataFrame
        Dataframe that converts all ``ratings >= min_rating_to_keep`` to 1 and drops the rest with a
        reset index

    """
    implicit_df = explicit_df.copy()
    implicit_df = implicit_df.drop(implicit_df[implicit_df[ratings_col] < min_rating_to_keep].index)
    implicit_df[ratings_col] = 1

    return implicit_df.reset_index(drop=True)


def remove_users_with_fewer_than_n_interactions(df: pd.DataFrame,
                                                min_num_of_interactions: int = 3,
                                                user_col: str = 'user_id') -> pd.DataFrame:
    """
    Remove DataFrame rows with users who appear fewer than ``min_num_of_interactions`` times.

    Parameters
    -------------
    df: pd.DataFrame
    min_num_of_interactions: int
        Minimum number of interactions a user can have while remaining in ``filtered_df``
    user_col: str
        Column name for the user IDs

    Returns
    -------------
    filtered_df: pd.DataFrame

    """
    value_counts_df = df[user_col].value_counts()

    return (
        df[~df[user_col].isin(
            value_counts_df[value_counts_df < min_num_of_interactions].index
        )].reset_index(drop=True)
    )


def trunc_normal(embedding_weight: torch.tensor,
                 mean: float = 0.0,
                 std: float = 1.0) -> torch.tensor:
    """
    Truncated normal initialization (approximation).

    Taken from FastAI: https://github.com/fastai/fastai/blob/master/fastai/layers.py

    """
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return embedding_weight.normal_().fmod_(2).mul_(std).add_(mean)


def get_init_arguments(exclude: Optional[Iterable] = [], verbose: bool = False) -> Dict[str, Any]:
    """
    Get all input arguments (*args and **kwargs) sent to the most-recently called method, given it
    is an ``__init__`` of a class.

    Parameters
    ----------
    exclude: list
        Arguments to exclude from ``init_args``. If an argument is not found in ``init_args``, it
        will be ignored
    verbose: bool
        Print keys in ``exclude`` not found in ``init_args``

    Returns
    ----------
    init_args: dict
        Argument dictionary with keys being argument names and values being arguments

    Notes
    ----------
    If the most-recently called method is not an ``__init__`` of a class, this function will return
    an empty dictionary.

    """
    frame = inspect.currentframe().f_back
    init_args = pytorch_lightning.utilities.parsing.get_init_args(frame)

    if exclude:
        for exclude_arg in exclude:
            try:
                del init_args[exclude_arg]
            except KeyError:
                if verbose:
                    print(f'Key {exclude_arg} not found in ``init_args`` and will be ignored.')
                continue

    return init_args


def pandas_df_to_hdf5(df: pd.DataFrame, out_path: Union[str, Path], key: str = 'interactions'):
    """Append a Pandas DataFrame to HDF5 using a ``table`` format and ``blosc`` compression."""
    df.to_hdf(str(out_path),
              key=key,
              mode='a',
              append=True,
              format='table',
              complib='blosc')


def df_to_html(df: pd.DataFrame,
               image_cols: List[str] = [],
               hyperlink_cols: List[str] = [],
               html_tags: Dict[str, Union[str, List[str]]] = dict(),
               transpose: bool = False,
               image_width: Optional[int] = None,
               max_num_rows: int = 200,
               **kwargs) -> str:
    """
    Convert a Pandas DataFrame to HTML.

    Parameters
    -------------
    df: DataFrame
        DataFrame to convert to HTML
    image_cols: str or list
        Column names that contain image urls or file paths. Columns specified as images will make
        all other transformations to those columns be ignored. Local files will display correctly in
        Jupyter if specified using relative paths but not if specified using absolute paths (see
        https://github.com/jupyter/notebook/issues/3810).
    hyperlink_cols: str or list
        Column names that contain hyperlinks to open in a new tab
    html_tags: dictionary
        A transformation to be inserted directly into the HTML tag.

        Ex: ``{'col_name_1': 'strong'}`` becomes ``<strong>col_name_1</strong>``

        Ex: ``{'col_name_2': 'mark'}`` becomes ``<mark>col_name_2</mark>``

        Ex: ``{'col_name_3': 'h2'}`` becomes ``<h2>col_name_3</h2>``

        Ex: ``{'col_name_4': ['em', 'strong']}`` becomes ``<em><strong>col_name_4</strong></em>``

    transpose: bool
        Transpose the DataFrame before converting to HTML
    image_width: int
        Set image width for each image generated
    max_num_rows: int
        Maximum number of rows to display
    **kwargs: keyword arguments
        Additional arguments sent to ``pandas.DataFrame.to_html``, as listed in:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_html.html

    Returns
    -------------
    df_html: HTML
        DataFrame converted to a HTML string, ready for displaying

    Examples
    -------------
    In a Jupyter notebook:

    .. code-block:: python

        from IPython.core.display import display, HTML
        import pandas as pd


        df = pd.DataFrame({
            'item': ['Beefy Fritos® Burrito'],
            'price': ['1.00'],
            'image_url': ['https://www.tacobell.com/images/22480_beefy_fritos_burrito_269x269.jpg'],
        })
        display(
            HTML(
                df_to_html(
                    df,
                    image_cols='image_url',
                    html_tags={'item': 'strong', 'price': 'em'},
                    image_width=200,
                )
            )
        )

    Notes
    -------------
    Converted table will have CSS class 'dataframe', unless otherwise specified.

    """
    def _wrap_cols_if_needed(cols: [str, List[str]]) -> List[str]:
        """Necessary for columns named with integers."""
        try:
            iter(cols)
        except TypeError:
            cols = [cols]
        if isinstance(cols, str):
            cols = [cols]

        return cols

    if max_num_rows is None or len(df) <= max_num_rows:
        df = df.copy()  # copy the dataframe so we don't edit the original!
    else:
        # explicit copy eliminates a warning we don't need
        df = df.head(max_num_rows).copy()

    image_cols = _wrap_cols_if_needed(image_cols)
    for image_col in image_cols:
        if image_col not in df.columns:
            raise ValueError('{} not a column in df!'.format(image_col))
        if not image_width:
            df[image_col] = df[image_col].map(lambda x: '<img src="{}">'.format(x))
        else:
            df[image_col] = df[image_col].map(
                lambda x: '<img src="{}" width={}>'.format(x, image_width)
            )

    hyperlink_cols = _wrap_cols_if_needed(hyperlink_cols)
    for hyperlink_col in hyperlink_cols:
        if hyperlink_col not in df.columns:
            raise ValueError('{} not a column in df!'.format(hyperlink_col))
        if hyperlink_col in image_cols:
            continue
        df[hyperlink_col] = df[hyperlink_col].map(
            lambda x: '<a target="_blank" href=" {}">{}</a>'.format(x, x)
        )

    for col, transformations in html_tags.items():
        if col not in df.columns:
            raise ValueError('{} not a column in df!'.format(col))
        if col in image_cols:
            continue

        if isinstance(transformations, str):
            transformations = [transformations]

        for extra in transformations:
            df[col] = df[col].map(lambda x: '<{0}>{1}</{0}>'.format(extra, x))

    max_colwidth = pd.get_option('display.max_colwidth')
    if pd.__version__ != '0':
        # this option is not backwards compatible with Pandas v1.0.0
        pd.set_option('display.max_colwidth', None)
    else:
        pd.set_option('display.max_colwidth', -1)

    if transpose:
        df = df.T
    df_html = df.to_html(escape=False, **kwargs)

    pd.set_option('display.max_colwidth', max_colwidth)

    return df_html


class Timer(object):
    """Class to manage timing different sections of a job."""
    def __init__(self):
        self.start_time = time.time()
        self.current_time = self.start_time

    def timecheck(self, message: str = 'finished') -> float:
        """Get time since last timecheck."""
        tmp_time = time.time()
        elapsed_time = (tmp_time-self.current_time)/60.0
        print('{0} ({1:.2f} min)'.format(message, elapsed_time))
        self.current_time = tmp_time

        return elapsed_time

    def time_since_start(self, message: str = 'total time') -> float:
        """Get time since timer was instantiated."""
        total_time = (time.time()-self.start_time)/60.0
        print('{0}: {1:.2f} min'.format(message, total_time))

        return total_time
