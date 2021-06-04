import inspect
import re

import docstring_parser


NEWLINE_CHARACTER = '\n'
FOUR_SPACES = '    '


def merge_docstrings(parent_class, child_docstring, child_class__init__):
    """
    Merge docstrings for Collie models to reduce the amount of repeated, shared docstrings.

    This method notes the arguments of the ``child_class``'s ``__init__`` function, and searches
    the docstrings of both the child and parent (in order) to construct the docstring for the child
    class.

    Specifically, the final docstring returned will be, in order:

    ```
    CHILD SHORT DESCRIPTION

    CHILD LONG DESCRIPTION

    Parameters
    ----------
    for each ``arg`` in CHILD ``__init__`` ARGUMENTS:
        CHILD ARGUMENT DOCSTRING (if it exists in child docstring), else PARENT ARGUMENT DOCSTRING
    ...

    POST ``Parameters`` CHILD DOCSTRING

    ```

    Note that ``Parameters`` must exist in both the parent and child docstring for this function
    to properly work.

    TODO:
    * Flesh out docstring here more.
    * Add test.

    """
    parent_docstring = parent_class.__doc__

    child_docstring_list = child_docstring.split(NEWLINE_CHARACTER)

    try:
        child_parameters_idx = [
            idx for idx, arg in enumerate(child_docstring_list)
            if re.search('\\sParameters\\s?$', arg)
        ]

        if len(child_parameters_idx) > 1:
            # two ``Parameters`` sections is bad, fail early
            return
        elif len(child_parameters_idx) == 0:
            # no ``Parameters`` sections is bad, fail early
            return

        child_parameters_idx = child_parameters_idx[0]
    except IndexError:
        # no ``Parameters`` section, abort
        return

    try:
        child_post_parameters_separator_idx = [
            idx for idx, arg in enumerate(child_docstring_list)
            if re.search(r'(-)\1{2,}$', arg)
            and idx > (child_parameters_idx + 1)
        ]

        if len(child_post_parameters_separator_idx) > 0:
            # we have a post-``Parameters`` section!
            child_post_parameters_separator_idx = min(child_post_parameters_separator_idx)
            rest_of_child_docstring = NEWLINE_CHARACTER.join(
                child_docstring_list[(child_post_parameters_separator_idx - 1):]
            )
        else:
            # we don't have anything past the ``Parameters`` section, and that's okay
            rest_of_child_docstring = ''
    except IndexError:
        # we don't have anything past the ``Parameters`` section, and that's okay
        rest_of_child_docstring = ''

    parent_parse = docstring_parser.numpydoc.NumpydocParser().parse(parent_docstring)
    child_parse = docstring_parser.numpydoc.NumpydocParser().parse(child_docstring)

    parent_arg_name_idx_dict = {
        param.arg_name: idx for idx, param in enumerate(parent_parse.params)
    }
    child_arg_name_idx_dict = {
        param.arg_name: idx for idx, param in enumerate(child_parse.params)
    }

    child_class__init__args = inspect.getfullargspec(child_class__init__).args

    # initial description
    final_docstring = (
        NEWLINE_CHARACTER
        + FOUR_SPACES
        + child_parse.short_description.replace(NEWLINE_CHARACTER, NEWLINE_CHARACTER + FOUR_SPACES)
        + NEWLINE_CHARACTER * 2
        + FOUR_SPACES
        + child_parse.long_description.replace(NEWLINE_CHARACTER, NEWLINE_CHARACTER + FOUR_SPACES)
        + NEWLINE_CHARACTER * 2
    )

    # parameters
    if len(child_class__init__args) > 0:
        final_docstring += (
            FOUR_SPACES
            + 'Parameters'
            + NEWLINE_CHARACTER
            + FOUR_SPACES
            + '----------'
            + NEWLINE_CHARACTER
        )

        for arg in child_class__init__args:
            if arg in child_arg_name_idx_dict:
                param_idx = child_arg_name_idx_dict[arg]
                param = child_parse.params[param_idx]
            elif arg in parent_arg_name_idx_dict:
                param_idx = parent_arg_name_idx_dict[arg]
                param = parent_parse.params[param_idx]
            else:
                # argument isn't in the docstring, we can skip it
                continue

            final_docstring += (
                f'{FOUR_SPACES}{param.arg_name.strip()}: {param.type_name.strip()}'
                f'{NEWLINE_CHARACTER}{FOUR_SPACES}{FOUR_SPACES}'
                f'{param.description.strip().replace(NEWLINE_CHARACTER, NEWLINE_CHARACTER + FOUR_SPACES + FOUR_SPACES)}'  # noqa: E501
                f'{NEWLINE_CHARACTER}'
            )

    final_docstring += NEWLINE_CHARACTER + rest_of_child_docstring

    return final_docstring
