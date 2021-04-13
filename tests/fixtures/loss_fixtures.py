import pytest
import torch


SCORES = torch.tensor([
    1.1, 1.3, 4.7, -7.234,
    -1.3, 0.7, 4.7, -2.2468,
    -4.7, 1.3, 3.56, -0.8924,
    4.01, 2.7, 3.7, -5.2468,
    3.89, 12.8, -1.7, -0.8143,
])


@pytest.fixture()
def positive_items():
    return torch.tensor([0, 1, 2, 3])


@pytest.fixture()
def negative_items():
    return torch.tensor([4, 5, 6, 7])


@pytest.fixture()
def many_negative_items():
    return torch.tensor([
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
    ])


@pytest.fixture()
def positive_scores(positive_items):
    return SCORES[positive_items]


@pytest.fixture()
def negative_scores(negative_items):
    return SCORES[negative_items]


@pytest.fixture()
def many_negative_scores(many_negative_items):
    return SCORES[many_negative_items]


@pytest.fixture()
def metadata_a():
    return torch.tensor([
        0, 1, 1, 2,
        0, 0, 0, 1,
        2, 1, 1, 2,
        0, 0, 1, 2,
        0, 0, 1, 2,
    ])


@pytest.fixture()
def metadata_a_diff():
    return torch.tensor([.8, 1, 1, 1])


@pytest.fixture()
def metadata_b():
    return torch.tensor([
        1, 2, 2, 3,
        3, 2, 1, 3,
        3, 2, 2, 2,
        1, 1, 2, 2,
        1, 1, 2, 2,
    ])


@pytest.fixture()
def metadata_a_and_2_diff():
    return torch.tensor([
        [.8, .7, 1, .7],
        [1, .5, .5, .8],
        [.5, 1, .5, .8],
        [.5, 1, .5, .8],
    ])
