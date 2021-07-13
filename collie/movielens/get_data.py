import os
from pathlib import Path
import re
import zipfile

import pandas as pd
import requests

from collie.config import DATA_PATH


def read_movielens_df(decrement_ids: bool = True) -> pd.DataFrame:
    """
    Read ``u.data`` from the MovieLens 100K dataset.

    If there is not a directory at ``$DATA_PATH/ml-100k``, this function creates that directory and
    downloads the entire dataset there.

    See the MovieLens 100K README for additional information on the dataset:
    https://files.grouplens.org/datasets/movielens/ml-100k-README.txt

    Parameters
    ----------
    decrement_ids: bool
        Decrement user and item IDs by 1 before returning, which is required for Collie's
        ``Interactions`` dataset

    Returns
    -------
    df: pd.DataFrame
        MovieLens 100K ``u.data`` comprising of columns:

            * user_id

            * item_id

            * rating

            * timestamp

    Side Effects
    ------------
    Creates directory at ``$DATA_PATH/ml-100k`` and downloads data files if data does not exist.

    """
    _make_data_path_dirs_if_not_exist()

    df_path = os.path.join(DATA_PATH, 'ml-100k', 'u.data')
    if not Path(df_path).exists():
        _download_movielens_100k()

    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(df_path, sep='\t', names=column_names)

    if decrement_ids:
        df.loc[:, 'user_id'] = df['user_id'] - 1
        df.loc[:, 'item_id'] = df['item_id'] - 1

    return df


def read_movielens_df_item() -> pd.DataFrame:
    """
    Read ``u.item`` from the MovieLens 100K dataset.

    If there is not a directory at ``$DATA_PATH/ml-100k``, this function creates that directory and
    downloads the entire dataset there.

    See the MovieLens 100K README for additional information on the dataset:
    https://files.grouplens.org/datasets/movielens/ml-100k-README.txt

    Returns
    -------
    df_item: pd.DataFrame
        MovieLens 100K ``u.item`` containing columns:

            * item_id

            * movie_title

            * release_date

            * video_release_date

            * IMDb_URL

            * unknown

            * Action

            * Adventure

            * Animation

            * Children

            * Comedy

            * Crime

            * Documentary

            * Drama

            * Fantasy

            * Film_Noir

            * Horror

            * Musical

            * Mystery

            * Romance', 'Sci_Fi

            * Thriller

            * War

            * Wester

    Side Effects
    ------------
    Creates directory at ``$DATA_PATH/ml-100k`` and downloads data files if data does not exist.

    """
    _make_data_path_dirs_if_not_exist()

    df_item_path = os.path.join(DATA_PATH, 'ml-100k', 'u.item')
    if not Path(df_item_path).exists():
        _download_movielens_100k()

    column_names = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
                    'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                    'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery',
                    'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']
    df_item = pd.read_csv(df_item_path, sep='|', encoding='latin-1', names=column_names)

    df_item['release_date'] = pd.to_datetime(df_item['release_date'])
    df_item = df_item.drop(columns=['video_release_date'])

    return df_item


def _make_data_path_dirs_if_not_exist() -> None:
    """Get path to the movielens dataset file."""
    if not DATA_PATH.exists():
        print(f'Making data path at ``{DATA_PATH}``...')
        DATA_PATH.mkdir(parents=True, exist_ok=True)


def _download_movielens_100k() -> None:
    """Download the MovieLens 100K data."""
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    req = requests.get(url, stream=True)

    print('Downloading MovieLens 100K data...')

    with open(os.path.join(DATA_PATH, 'ml-100k.zip'), 'wb') as f:
        f.write(req.content)

    with zipfile.ZipFile(os.path.join(DATA_PATH / 'ml-100k.zip'), 'r') as z:
        z.extractall(DATA_PATH)


def read_movielens_posters_df() -> pd.DataFrame:
    """
    Read in data containing the item ID and poster URL for visualization purposes of MovieLens 100K
    data.

    This function will attempt to read the file at ``data/movielens_posters.csv`` if it exists and,
    if not, will read the CSV from the origin GitHub repo at
    https://raw.githubusercontent.com/ShopRunner/collie/main/data/movielens_posters.csv.

    Returns
    -------
    posters_df: pd.DataFrame
        DataFrame comprising columns:

            * item_id

            * url

    """
    # attempt to first load from a local file
    absolute_data_path = Path(__file__).parent.absolute().parent.parent / 'data'
    movielens_posters_csv_filepath = os.path.join(absolute_data_path, 'movielens_posters.csv')

    # be prepared to read the CSV from the origin GitHub repo as well
    movielens_posters_csv_url = (
        'https://raw.githubusercontent.com/ShopRunner/collie/main/data/movielens_posters.csv'
    )

    posters_df = pd.read_csv(
        movielens_posters_csv_filepath
        if os.path.exists(movielens_posters_csv_filepath)
        else movielens_posters_csv_url
    )

    return posters_df


def get_movielens_metadata(df_item: pd.DataFrame = None) -> pd.DataFrame:
    """
    Return MovieLens 100K metadata as a DataFrame.

    DataFrame returned has the following column order:

    .. code-block:: python

        [
            'genre_action', 'genre_adventure', 'genre_animation', 'genre_children', 'genre_comedy',
            'genre_crime', 'genre_documentary', 'genre_drama', 'genre_fantasy', 'genre_film_noir',
            'genre_horror', 'genre_musical', 'genre_mystery', 'genre_romance', 'genre_sci_fi',
            'genre_thriller', 'genre_war', 'genre_western', 'genre_unknown', 'decade_unknown',
            'decade_20', 'decade_30', 'decade_40', 'decade_50', 'decade_60',
            'decade_70', 'decade_80', 'decade_90',
        ]

    See the MovieLens 100K README for additional information on the dataset:
    https://files.grouplens.org/datasets/movielens/ml-100k-README.txt

    Parameters
    ----------
    df_item: pd.DataFrame
        DataFrame of MovieLens 100K ``u.item`` containing binary columns of movie names and
        metadata. If ``None``, will automatically read the output of ``read_movielens_df_item()``

    Returns
    -------
    metadata_df: pd.DataFrame

    """
    if df_item is None:
        df_item = read_movielens_df_item()

    # format movies decade
    df_item_date = df_item.iloc[:, [2]].copy()
    df_item_date.loc[:, 'year'] = df_item_date['release_date'].dt.year.fillna(1900)
    df_item_date.loc[:, 'decade'] = ((df_item_date['year'] - 1900) / 10).astype('int64') * 10
    df_decades = pd.get_dummies(df_item_date.decade, prefix='decade')
    df_decades.columns = ['decade_unknown'] + df_decades.columns[1:].tolist()

    # format movie genre
    df_item_genre = df_item.iloc[:, list(range(4, 23))].copy()
    df_item_genre.columns = 'genre_' + df_item_genre.columns.str.lower()

    # format final metadata structure
    metadata_df = pd.merge(df_item_genre, df_decades, left_index=True, right_index=True)

    # find and swap genre_unknown to end of genre list
    cols = metadata_df.columns.values.tolist()
    last_genre_element = list(filter(re.compile('genre*').match, cols))[-1]
    last_genre_index = cols.index(last_genre_element)
    cols.insert(last_genre_index + 1, 'genre_unknown')
    cols.remove('genre_unknown')
    metadata_df = metadata_df[cols]

    return metadata_df
