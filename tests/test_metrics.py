from functools import partial
from unittest import mock

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
import torch

from collie_recs.metrics import (
    _get_labels,
    _get_user_item_pairs,
    auc,
    evaluate_in_batches,
    get_preds,
    mapk,
    mrr,
)


def get_model_scores(user, item, scores):
    return scores[user, item]


def test_get_user_item_pairs_example(device):
    user_ids = np.array([10, 11, 12])
    n_items = 4

    expected_users = torch.tensor([10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12], device=device)
    expected_items = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], device=device)

    actual_users, actual_items = _get_user_item_pairs(user_ids, n_items, device)

    assert torch.equal(actual_users, expected_users)
    assert torch.equal(actual_items, expected_items)


def test_get_user_item_pairs_unordered(device):
    user_ids = np.array([1, 16, 33, 22])
    n_items = 2

    expected_users = torch.tensor([1, 1, 16, 16, 33, 33, 22, 22], device=device)
    expected_items = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device)

    actual_users, actual_items = _get_user_item_pairs(user_ids, n_items, device)

    assert torch.equal(actual_users, expected_users)
    assert torch.equal(actual_items, expected_items)


@mock.patch('collie_recs.model.MatrixFactorizationModel')
def test_get_preds_implicit(model, test_implicit_predicted_scores, device):
    n_users, n_items = test_implicit_predicted_scores.shape
    user_ids = np.arange(n_users)
    model.return_value = test_implicit_predicted_scores.view(-1)
    actual_preds = get_preds(model, user_ids, n_items, device)

    assert torch.equal(actual_preds, test_implicit_predicted_scores)


def test_get_labels(targets, test_implicit_recs, test_implicit_labels, device):
    user_ids = np.array([1, 2])
    actual_labels = _get_labels(targets, user_ids, test_implicit_recs[user_ids, :], device)
    expected_labels = test_implicit_labels[user_ids, :].to(device)

    assert torch.equal(actual_labels, expected_labels)


def test_get_labels_k(targets, test_implicit_recs, test_implicit_labels, device):
    user_ids = np.arange(test_implicit_recs.shape[0])
    k = 2
    actual_labels = _get_labels(targets, user_ids, test_implicit_recs[:, :k], device)
    expected_labels = test_implicit_labels[:, :k].to(device)

    assert torch.equal(actual_labels, expected_labels)


def test_map(targets, test_implicit_predicted_scores):
    user_ids = np.array([1, 2])
    actual_score = mapk(targets, user_ids, test_implicit_predicted_scores[user_ids, :], k=4)

    assert actual_score == .75


def test_map_1(targets, test_implicit_predicted_scores):
    user_ids = np.arange(test_implicit_predicted_scores.shape[0])
    actual_score = mapk(targets, user_ids, test_implicit_predicted_scores[user_ids, :], k=1)

    np.testing.assert_almost_equal(actual_score, 2/3)


def test_mrr(targets, test_implicit_predicted_scores):
    user_ids = np.arange(test_implicit_predicted_scores.shape[0])
    actual_score = mrr(targets, user_ids, test_implicit_predicted_scores[user_ids, :])

    np.testing.assert_almost_equal(actual_score, (1 + 1 + 1/2) / 3)


def test_auc(targets, test_implicit_predicted_scores):
    user_ids = np.arange(test_implicit_predicted_scores.shape[0])
    actual_score = auc(targets, user_ids, test_implicit_predicted_scores[user_ids, :])

    expected_score = 0
    for i in user_ids:
        expected_score += roc_auc_score(
            targets[i].toarray()[0],
            test_implicit_predicted_scores[i, :],
        )
    expected_score = expected_score/len(user_ids)

    np.testing.assert_almost_equal(actual_score, expected_score)


@pytest.mark.parametrize('batch_size', [20, 2, 1])  # default, uneven, single
@mock.patch('collie_recs.model.MatrixFactorizationModel')
def test_evaluate_in_batches(
    model,
    test_implicit_interactions,
    test_implicit_predicted_scores,
    metrics,
    batch_size,
):
    model.side_effect = partial(get_model_scores, scores=test_implicit_predicted_scores)

    mapk_score, mrr_score, auc_score = evaluate_in_batches(
        [mapk, mrr, auc],
        test_implicit_interactions,
        model,
        k=4,
        batch_size=batch_size,
    )

    np.testing.assert_almost_equal(mapk_score, metrics['mapk'])
    np.testing.assert_almost_equal(mrr_score, metrics['mrr'])
    np.testing.assert_almost_equal(auc_score, metrics['auc'])


def test_evaluate_in_batches_logger(
    implicit_model,
    test_implicit_interactions,
    test_implicit_predicted_scores,
):
    class LightningLoggerFixture():
        """A simple logger base class with a method ``log_metrics``."""
        def __init__(self):
            pass

        def log_metrics(self, metrics, step):
            """Save ``metrics`` and ``step`` as class-level attributes for testing."""
            self.metrics = metrics
            self.step = step

    logger = LightningLoggerFixture()

    mapk_score, mrr_score, auc_score = evaluate_in_batches(
        [mapk, mrr, auc],
        test_implicit_interactions,
        implicit_model,
        k=4,
        logger=logger,
    )

    assert mapk_score == logger.metrics['mapk']
    assert mrr_score == logger.metrics['mrr']
    assert auc_score == logger.metrics['auc']

    assert logger.step == implicit_model.hparams.n_epochs_completed_
