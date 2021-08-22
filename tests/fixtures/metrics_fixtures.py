import pytest
import torch

from collie.interactions import ExplicitInteractions, Interactions


@pytest.fixture()
def test_implicit_interactions():
    return Interactions(
        users=[0, 0, 0, 1, 1, 1, 2, 2],
        items=[0, 1, 2, 1, 2, 3, 0, 2],
        ratings=[1, 1, 1, 1, 1, 1, 1, 1],
        check_num_negative_samples_is_valid=False,
    )


@pytest.fixture()
def test_implicit_predicted_scores():
    return torch.tensor([
        [0.9, 0.4, 0.5, 0.7],
        [0.1, 1.2, 1.7, 0.12],
        [-1, -2, -3, 4],
    ])


@pytest.fixture()
def test_explicit_interactions():
    return ExplicitInteractions(
        users=[0, 0, 0, 1, 1, 1, 2, 2],
        items=[0, 1, 2, 1, 2, 3, 0, 2],
        ratings=[1, 2, 3, 4, 5, 4, 3, 2],
    )


@pytest.fixture()
def test_explicit_predicted_scores():
    return torch.tensor([
        [0.9, 1.4, 2.5, -0.7],
        [5.1, 4.2, 3.7, 2.12],
        [-1, 2, -3, 4],
    ])


@pytest.fixture()
def test_implicit_recs():
    return torch.tensor([
        [0, 3, 2, 1],
        [2, 1, 3, 0],
        [3, 0, 1, 2],
    ])


@pytest.fixture()
def test_implicit_labels():
    return torch.tensor([
        [1, 0, 1, 1],
        [1, 1, 1, 0],
        [0, 1, 0, 1],
    ], dtype=float)


@pytest.fixture()
def targets(test_implicit_interactions):
    return test_implicit_interactions.mat.tocsr()


@pytest.fixture()
def test_sequential_recs():
    return torch.tensor([
        [0, 3, 2, 4, 5, 1],
        [2, 1, 3, 0, 5, 4],
        [5, 3, 4, 0, 1, 2],
        [0, 1, 2, 3, 4, 5],
        [4, 2, 1, 0, 3, 5],
    ])


@pytest.fixture()
def test_sequential_labels():
    return torch.tensor([
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ], dtype=float)


@pytest.fixture()
def sequential_targets():
    return torch.tensor([1, 3, 5, 0, 1])


@pytest.fixture()
def test_sequences():
    return torch.tensor([[2, 3, 4, 7, 5, 6, 0],
                         [0, 7, 1, 6, 2, 5, 4],
                         [3, 4, 2, 5, 1, 0, 7],
                         [2, 1, 7, 6, 4, 3, 5],
                         [4, 6, 2, 3, 7, 0, 5]])


@pytest.fixture()
def test_sequential_predicted_scores():
    return torch.tensor([[1.2, -1.3, 2.4, 1.1, 3.7, 5, 0.7, -0.4],
                         [0.0, -0.7, 1.1, 2.6, 3.2, 1.5, 1.7223, -9],
                         [0.3, -2.7, -0.5, 1.6, 3.8, 1.7, -1.5, -0.47],
                         [7.9, -1.7, 0.3, 2.7, 3.2, 1.5, 1.8, 0.23],
                         [-0.6, 1.4, 2.2, 3.3, 1.7, 0.0, 0.01, -0.3]])


@pytest.fixture()
def metrics():
    return {
        'mapk': 0.7685185,
        'mrr': 0.8333333,
        'auc': 0.5277777,
        'mse': 5.86055,
        'mae': 1.69750,
    }
