import os
from pathlib import Path
import warnings


DATA_PATH = Path(os.environ.get('DATA_PATH', 'data'))


def warn_rename():
    """Raise a warning that the library will be renamed soon."""
    warnings.warn(
        '``collie_recs`` has been renamed to ``collie``. All future developments and additions to'
        ' the library will happen in the newly-named library. Please install and use the new'
        ' package name from PyPI: https://pypi.org/project/collie/'
    )
