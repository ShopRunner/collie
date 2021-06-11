from collie_recs.utils import merge_docstrings


# START: test class definitions
# NOTE: we include classes here since pytest does not accept classes as fixtures
class BaseClass:
    """
    This is the short description.

    This is a longer description. It contains many lines.
    With line breaks, like this.

    You can also have new paragraphs!

    NOTE: This is an important note!

    Parameters
    ----------
    arg1: str
        The first argument
    arg2: int
        This argument's description is longer.
        See how it is on a new line:
            * Even with a bullet list now!
    arg3: np.array
    **kwargs

    References
    ----------
    arg8: list
    arg9: int
        No description above, and that is okay!

    """
    def __init__(self, arg1, arg2, arg3, **kwargs):
        pass


class ChildClass(BaseClass):
    """
    This is the short description for the child.

    This is a longer description for the child. It also contains many lines.
    With line breaks, like this.

    You can also have new paragraphs!

    NOTE: This is an important note!

    Look, a new line of documentation after the note!

    Parameters
    ----------
    arg1: str
        The first argument
    arg4: int
        An important argument!

    References
    ----------
    arg8: list
    arg9: int
        No description above, and that is okay!
    arg10: str
        This one is new.

    """
    def __init__(self, arg1, arg2, arg3, arg4):
        pass


class ChildClassWithArgs(BaseClass):
    """
    This is the short description for the child.

    This is a longer description for the child. It also contains many lines.
    With line breaks, like this.

    You can also have new paragraphs!

    NOTE: This is an important note!

    Look, a new line of documentation after the note!

    Parameters
    ----------
    arg1: str
        The first argument
    arg4: int
        An important argument!
    *args: arguments
        A description for these args here.

    References
    ----------
    arg8: list
    arg9: int
        No description above, and that is okay!
    arg10: str
        This one is new.

    """
    def __init__(self, arg1, arg2, arg3, arg4, *args):
        pass


class ChildClassWithKwargs(BaseClass):
    """
    This is the short description for the child.

    This is a longer description for the child. It also contains many lines.
    With line breaks, like this.

    You can also have new paragraphs!

    NOTE: This is an important note!

    Look, a new line of documentation after the note!

    Parameters
    ----------
    arg1: str
        The first argument
    arg4: int
        An important argument!
    **kwargs: keyword argument
        Additional keyword arguments to pass into ``BaseClass``

    References
    ----------
    arg8: list
    arg9: int
        No description above, and that is okay!
    arg10: str
        This one is new.

    """
    def __init__(self, arg1, arg2, arg3, arg4, **kwargs):
        pass


class ChildClassWithArgsAndKwargs(BaseClass):
    """
    This is the short description for the child.

    This is a longer description for the child. It also contains many lines.
    With line breaks, like this.

    You can also have new paragraphs!

    NOTE: This is an important note!

    Look, a new line of documentation after the note!

    Parameters
    ----------
    arg1: str
        The first argument
    arg4: int
        An important argument!
    *args: arguments
    **kwargs: keyword argument
        Additional keyword arguments to pass into ``BaseClass``

    References
    ----------
    arg8: list
    arg9: int
        No description above, and that is okay!
    arg10: str
        This one is new.

    """
    def __init__(self, arg1, arg2, arg3, arg4, *args, **kwargs):
        pass


class ChildClassNoParamaters(BaseClass):
    """
    No ``Parameters`` section at all here!

    References
    ----------
    arg8

    """
    def __init__(self):
        pass


class ChildClassParamatersOnly(BaseClass):
    """
    Note that nothing is after the ``Parameters`` section here.

    Parameters
    ----------
    arg1: str
        The first argument
    arg4: int
        An important argument!
    *args: arguments
    **kwargs: keyword argument
        Additional keyword arguments to pass into ``BaseClass``

    """
    def __init__(self, arg1, arg2, arg3, arg4, *args, **kwargs):
        pass


class ChildClassExtraParamatersNoDoc(BaseClass):
    """
    Note that nothing is after the ``Parameters`` section here.

    Parameters
    ----------
    arg1: str
        The first argument
    arg4: int
        An important argument!
    *args: arguments
    **kwargs: keyword argument
        Additional keyword arguments to pass into ``BaseClass``

    """
    def __init__(self, arg1, arg2, arg3, arg4, extra, *args, **kwargs):
        pass


class ChildClassWithTwoExtraSections(BaseClass):
    """
    This is the short description for the child.

    This is a longer description for the child. It also contains many lines.
    With line breaks, like this.

    You can also have new paragraphs!

    NOTE: This is an important note!

    Look, a new line of documentation after the note!

    Parameters
    ----------
    arg1: str
        The first argument
    arg4: int
        An important argument!
    *args: arguments
    **kwargs: keyword argument
        Additional keyword arguments to pass into ``BaseClass``

    References
    ----------
    arg8: list
    arg9: int
        No description above, and that is okay!
    arg10: str
        This one is new.

    Notes
    -----
    This is a note. The above ``References`` section used to say ``Returns``, but classes do not
    return anything and I did not feel inclined to change the description.

    """
    def __init__(self, arg1, arg2, arg3, arg4, *args, **kwargs):
        pass


# START: tests
def test_merge_docstrings():
    expected = """
    This is the short description for the child.

    This is a longer description for the child. It also contains many lines.
    With line breaks, like this.

    You can also have new paragraphs!

    NOTE: This is an important note!

    Look, a new line of documentation after the note!

    Parameters
    ----------
    arg1: str
        The first argument
    arg2: int
        This argument's description is longer.
        See how it is on a new line:
            * Even with a bullet list now!
    arg3: np.array
    arg4: int
        An important argument!

    References
    ----------
    arg8: list
    arg9: int
        No description above, and that is okay!
    arg10: str
        This one is new.

    """

    actual = merge_docstrings(BaseClass, ChildClass.__doc__, ChildClass.__init__)

    print(expected)
    print(actual)

    assert actual == expected


def test_merge_docstrings_with_args():
    expected = """
    This is the short description for the child.

    This is a longer description for the child. It also contains many lines.
    With line breaks, like this.

    You can also have new paragraphs!

    NOTE: This is an important note!

    Look, a new line of documentation after the note!

    Parameters
    ----------
    arg1: str
        The first argument
    arg2: int
        This argument's description is longer.
        See how it is on a new line:
            * Even with a bullet list now!
    arg3: np.array
    arg4: int
        An important argument!
    *args: arguments
        A description for these args here.

    References
    ----------
    arg8: list
    arg9: int
        No description above, and that is okay!
    arg10: str
        This one is new.

    """

    actual = merge_docstrings(BaseClass, ChildClassWithArgs.__doc__, ChildClassWithArgs.__init__)

    assert actual == expected


def test_merge_docstrings_with_kwargs():
    expected = """
    This is the short description for the child.

    This is a longer description for the child. It also contains many lines.
    With line breaks, like this.

    You can also have new paragraphs!

    NOTE: This is an important note!

    Look, a new line of documentation after the note!

    Parameters
    ----------
    arg1: str
        The first argument
    arg2: int
        This argument's description is longer.
        See how it is on a new line:
            * Even with a bullet list now!
    arg3: np.array
    arg4: int
        An important argument!
    **kwargs: keyword argument
        Additional keyword arguments to pass into ``BaseClass``

    References
    ----------
    arg8: list
    arg9: int
        No description above, and that is okay!
    arg10: str
        This one is new.

    """

    actual = merge_docstrings(BaseClass,
                              ChildClassWithKwargs.__doc__,
                              ChildClassWithKwargs.__init__)

    assert actual == expected


def test_merge_docstrings_with_args_and_kwargs():
    expected = """
    This is the short description for the child.

    This is a longer description for the child. It also contains many lines.
    With line breaks, like this.

    You can also have new paragraphs!

    NOTE: This is an important note!

    Look, a new line of documentation after the note!

    Parameters
    ----------
    arg1: str
        The first argument
    arg2: int
        This argument's description is longer.
        See how it is on a new line:
            * Even with a bullet list now!
    arg3: np.array
    arg4: int
        An important argument!
    *args: arguments
    **kwargs: keyword argument
        Additional keyword arguments to pass into ``BaseClass``

    References
    ----------
    arg8: list
    arg9: int
        No description above, and that is okay!
    arg10: str
        This one is new.

    """

    actual = merge_docstrings(BaseClass,
                              ChildClassWithArgsAndKwargs.__doc__,
                              ChildClassWithArgsAndKwargs.__init__)

    assert actual == expected


def test_merge_docstrings_no_paramaters_section():
    expected = """
    No ``Parameters`` section at all here!

    References
    ----------
    arg8

    """

    actual = merge_docstrings(BaseClass,
                              ChildClassNoParamaters.__doc__,
                              ChildClassNoParamaters.__init__)

    assert actual == expected


def test_merge_docstrings_parameters_section_nothing_after():
    expected = """
    Note that nothing is after the ``Parameters`` section here.

    Parameters
    ----------
    arg1: str
        The first argument
    arg2: int
        This argument's description is longer.
        See how it is on a new line:
            * Even with a bullet list now!
    arg3: np.array
    arg4: int
        An important argument!
    *args: arguments
    **kwargs: keyword argument
        Additional keyword arguments to pass into ``BaseClass``

    """

    actual = merge_docstrings(BaseClass,
                              ChildClassParamatersOnly.__doc__,
                              ChildClassParamatersOnly.__init__)

    assert actual == expected


def test_merge_docstrings_extra_parameter_included_with_no_documentation():
    expected = """
    Note that nothing is after the ``Parameters`` section here.

    Parameters
    ----------
    arg1: str
        The first argument
    arg2: int
        This argument's description is longer.
        See how it is on a new line:
            * Even with a bullet list now!
    arg3: np.array
    arg4: int
        An important argument!
    *args: arguments
    **kwargs: keyword argument
        Additional keyword arguments to pass into ``BaseClass``

    """

    actual = merge_docstrings(BaseClass,
                              ChildClassExtraParamatersNoDoc.__doc__,
                              ChildClassExtraParamatersNoDoc.__init__)

    assert actual == expected


def test_merge_docstrings_with_two_extra_sections():
    expected = """
    This is the short description for the child.

    This is a longer description for the child. It also contains many lines.
    With line breaks, like this.

    You can also have new paragraphs!

    NOTE: This is an important note!

    Look, a new line of documentation after the note!

    Parameters
    ----------
    arg1: str
        The first argument
    arg2: int
        This argument's description is longer.
        See how it is on a new line:
            * Even with a bullet list now!
    arg3: np.array
    arg4: int
        An important argument!
    *args: arguments
    **kwargs: keyword argument
        Additional keyword arguments to pass into ``BaseClass``

    References
    ----------
    arg8: list
    arg9: int
        No description above, and that is okay!
    arg10: str
        This one is new.

    Notes
    -----
    This is a note. The above ``References`` section used to say ``Returns``, but classes do not
    return anything and I did not feel inclined to change the description.

    """

    actual = merge_docstrings(BaseClass,
                              ChildClassWithTwoExtraSections.__doc__,
                              ChildClassWithTwoExtraSections.__init__)

    assert actual == expected
