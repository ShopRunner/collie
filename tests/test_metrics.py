import copy
from functools import partial
from unittest import mock

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
import torch
import torchmetrics

from collie.metrics import (auc,
                            evaluate_in_batches,
                            explicit_evaluate_in_batches,
                            get_labels,
                            get_preds,
                            get_user_item_pairs,
                            get_sequential_item_pairs,
                            mapk,
                            mrr,
                            sequential_get_labels,
                            sequential_mapk,
                            sequential_mrr)
from collie.metrics.metrics_utils import _get_evaluate_in_batches_device


def get_model_scores(user, item, scores):
    return scores[user.long(), item.long()]


def get_model_scores(user, item, scores):
    return scores[user.long(), item.long()]


@pytest.mark.parametrize('n_items_type', ['int', 'np.int64'])
def test_get_user_item_pairs_example(device, n_items_type):
    user_ids = np.array([10, 11, 12])

    n_items = 4
    if n_items_type == 'np.int64':
        n_items = np.int64(n_items)

    expected_users = torch.tensor([10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12], device=device)
    expected_items = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], device=device)

    actual_users, actual_items = get_user_item_pairs(user_ids=user_ids,
                                                     n_items=n_items,
                                                     device=device)

    assert torch.equal(actual_users, expected_users)
    assert torch.equal(actual_items, expected_items)


@pytest.mark.parametrize('n_items_type', ['int', 'np.int64'])
def test_get_user_item_pairs_unordered(device, n_items_type):
    user_ids = np.array([1, 16, 33, 22])

    n_items = 2
    if n_items_type == 'np.int64':
        n_items = np.int64(n_items)

    expected_users = torch.tensor([1, 1, 16, 16, 33, 33, 22, 22], device=device)
    expected_items = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device)

    actual_users, actual_items = get_user_item_pairs(user_ids=user_ids,
                                                     n_items=n_items,
                                                     device=device)

    assert torch.equal(actual_users, expected_users)
    assert torch.equal(actual_items, expected_items)


@mock.patch('collie.model.MatrixFactorizationModel')
def test_get_preds_implicit(model, test_implicit_predicted_scores, device):
    n_users, n_items = test_implicit_predicted_scores.shape
    user_ids = np.arange(n_users)
    model.return_value = test_implicit_predicted_scores.view(-1)
    actual_preds = get_preds(model=model,
                             user_ids=user_ids,
                             n_items=n_items,
                             device=device)

    assert torch.equal(actual_preds, test_implicit_predicted_scores)


def test_get_labels(targets, test_implicit_recs, test_implicit_labels, device):
    user_ids = np.array([1, 2])
    actual_labels = get_labels(targets=targets,
                               user_ids=user_ids,
                               preds=test_implicit_recs[user_ids, :],
                               device=device)
    expected_labels = test_implicit_labels[user_ids, :].to(device)

    assert torch.equal(actual_labels, expected_labels)


def test_get_labels_k(targets, test_implicit_recs, test_implicit_labels, device):
    user_ids = np.arange(test_implicit_recs.shape[0])
    k = 2
    actual_labels = get_labels(targets=targets,
                               user_ids=user_ids,
                               preds=test_implicit_recs[:, :k],
                               device=device)
    expected_labels = test_implicit_labels[:, :k].to(device)

    assert torch.equal(actual_labels, expected_labels)


def test_map(targets, test_implicit_predicted_scores):
    user_ids = np.array([1, 2])
    actual_score = mapk(targets=targets,
                        user_ids=user_ids,
                        preds=test_implicit_predicted_scores[user_ids, :],
                        k=4)

    assert actual_score == .75


def test_map_1(targets, test_implicit_predicted_scores):
    user_ids = np.arange(test_implicit_predicted_scores.shape[0])
    actual_score = mapk(targets=targets,
                        user_ids=user_ids,
                        preds=test_implicit_predicted_scores[user_ids, :],
                        k=1)

    np.testing.assert_almost_equal(actual_score, 2/3)


def test_map_k_too_big(targets, test_implicit_predicted_scores):
    user_ids = np.arange(test_implicit_predicted_scores.shape[0])

    with pytest.raises(ValueError):
        mapk(targets=targets,
             user_ids=user_ids,
             preds=test_implicit_predicted_scores[user_ids, :],
             k=(targets.shape[1] + 1))


def test_mrr(targets, test_implicit_predicted_scores):
    user_ids = np.arange(test_implicit_predicted_scores.shape[0])
    actual_score = mrr(targets=targets,
                       user_ids=user_ids,
                       preds=test_implicit_predicted_scores[user_ids, :])

    np.testing.assert_almost_equal(actual_score, (1 + 1 + 1/2) / 3)


def test_auc(targets, test_implicit_predicted_scores):
    user_ids = np.arange(test_implicit_predicted_scores.shape[0])
    actual_score = auc(targets=targets,
                       user_ids=user_ids,
                       preds=test_implicit_predicted_scores[user_ids, :])

    expected_score = 0
    for i in user_ids:
        expected_score += roc_auc_score(
            targets[i].toarray()[0],
            test_implicit_predicted_scores[i, :],
        )
    expected_score = expected_score/len(user_ids)

    np.testing.assert_almost_equal(actual_score, expected_score)


def test_bad_evaluate_in_batches_with_explicit_data(test_explicit_interactions):
    with pytest.raises(ValueError):
        evaluate_in_batches(
            metric_list=[mapk],
            test_interactions=test_explicit_interactions,
            model='test_model',
        )


def test_bad_explicit_evaluate_in_batches_with_implicit_data(test_implicit_interactions):
    with pytest.raises(ValueError):
        explicit_evaluate_in_batches(
            metric_list=[torchmetrics.MeanSquaredError()],
            test_interactions=test_implicit_interactions,
            model='test_model',
        )


class TestEvaluateInBatchesDevice:
    @mock.patch('torch.cuda.is_available')
    @mock.patch('collie.model.MatrixFactorizationModel')
    def test_cuda_available_model_cpu(self, model, is_available_mock):
        is_available_mock.return_value = True
        model.device = 'cpu'

        with pytest.warns(UserWarning):
            device = _get_evaluate_in_batches_device(model=model)

        assert device == 'cpu'

    @mock.patch('torch.cuda.is_available')
    @mock.patch('collie.model.MatrixFactorizationModel')
    def test_cuda_not_available_model_cuda(self, model, is_available_mock):
        is_available_mock.return_value = False
        model.device = 'cuda:0'

        with pytest.warns(None):  # assert no warning is raised here
            device = _get_evaluate_in_batches_device(model=model)

        assert device == 'cuda:0'

    @mock.patch('torch.cuda.is_available')
    @mock.patch('collie.model.MatrixFactorizationModel')
    def test_cuda_available_model_no_device(self, model, is_available_mock):
        is_available_mock.return_value = True
        model.device = None

        with pytest.warns(None):  # assert no warning is raised here
            device = _get_evaluate_in_batches_device(model=model)

        assert device == 'cuda:0'

    @mock.patch('torch.cuda.is_available')
    @mock.patch('collie.model.MatrixFactorizationModel')
    def test_cuda_not_available_model_no_device(self, model, is_available_mock):
        is_available_mock.return_value = False
        model.device = None

        with pytest.warns(None):  # assert no warning is raised here
            device = _get_evaluate_in_batches_device(model=model)

        assert device == 'cpu'


@pytest.mark.parametrize('batch_size', [20, 2, 1])  # default, uneven, single
@mock.patch('collie.model.MatrixFactorizationModel')
def test_evaluate_in_batches(
    model,
    test_implicit_interactions,
    test_implicit_predicted_scores,
    metrics,
    batch_size,
):
    model.side_effect = partial(get_model_scores, scores=test_implicit_predicted_scores)

    # need to do this for the Mock in order for the metrics to be on the right device
    model.device = 'cpu'

    mapk_score, mrr_score, auc_score = evaluate_in_batches(
        metric_list=[mapk, mrr, auc],
        test_interactions=test_implicit_interactions,
        model=model,
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
    model = copy.deepcopy(implicit_model)

    mapk_score, mrr_score, auc_score = evaluate_in_batches(
        metric_list=[mapk, mrr, auc],
        test_interactions=test_implicit_interactions,
        model=model,
        k=4,
        logger=logger,
    )

    assert mapk_score == logger.metrics['mapk']
    assert mrr_score == logger.metrics['mrr']
    assert auc_score == logger.metrics['auc']

    assert logger.step == model.hparams.num_epochs_completed


@mock.patch('collie.model.MatrixFactorizationModel')
def test_explicit_evaluate_in_batches(
    model,
    test_explicit_interactions,
    test_explicit_predicted_scores,
    metrics,
):
    model.side_effect = partial(get_model_scores, scores=test_explicit_predicted_scores)

    # need to do this for the Mock in order for the metrics to be on the right device
    model.device = 'cpu'

    mse_score, mae_score = explicit_evaluate_in_batches(
        metric_list=[torchmetrics.MeanSquaredError(), torchmetrics.MeanAbsoluteError()],
        test_interactions=test_explicit_interactions,
        model=model,
        num_workers=0,
    )

    np.testing.assert_almost_equal(mse_score, metrics['mse'])
    np.testing.assert_almost_equal(mae_score, metrics['mae'])


def test_explicit_evaluate_in_batches_logger(
    explicit_model,
    test_explicit_interactions,
    test_explicit_predicted_scores,
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

    mse_score, mae_score = explicit_evaluate_in_batches(
        metric_list=[torchmetrics.MeanSquaredError(), torchmetrics.MeanAbsoluteError()],
        test_interactions=test_explicit_interactions,
        model=explicit_model,
        logger=logger,
        num_workers=0,
    )

    assert mse_score == logger.metrics['MeanSquaredError']
    assert mae_score == logger.metrics['MeanAbsoluteError']

    assert logger.step == explicit_model.hparams.num_epochs_completed


@pytest.mark.parametrize('n_items_type', ['int', 'np.int64'])
def test_get_sequential_item_pairs(device, n_items_type):
    sequences = np.array([[208, 31, 188, 241, 170, 110, 255, 4, 73, 101],
                          [254, 271, 270, 19, 128, 220, 5, 243, 17, 269],
                          [99, 153, 8, 168, 177, 86, 15, 221, 257, 265],
                          [231, 62, 138, 74, 141, 50, 53, 227, 43, 85],
                          [2, 137, 36, 65, 37, 139, 240, 75, 77, 225]])

    n_items = 4
    if n_items_type == 'np.int64':
        n_items = np.int64(n_items)

    expected_sequences = torch.tensor([[208, 31, 188, 241, 170, 110, 255, 4, 73, 101],
                                       [208, 31, 188, 241, 170, 110, 255, 4, 73, 101],
                                       [208, 31, 188, 241, 170, 110, 255, 4, 73, 101],
                                       [208, 31, 188, 241, 170, 110, 255, 4, 73, 101],
                                       [254, 271, 270, 19, 128, 220, 5, 243, 17, 269],
                                       [254, 271, 270, 19, 128, 220, 5, 243, 17, 269],
                                       [254, 271, 270, 19, 128, 220, 5, 243, 17, 269],
                                       [254, 271, 270, 19, 128, 220, 5, 243, 17, 269],
                                       [99, 153, 8, 168, 177, 86, 15, 221, 257, 265],
                                       [99, 153, 8, 168, 177, 86, 15, 221, 257, 265],
                                       [99, 153, 8, 168, 177, 86, 15, 221, 257, 265],
                                       [99, 153, 8, 168, 177, 86, 15, 221, 257, 265],
                                       [231, 62, 138, 74, 141, 50, 53, 227, 43, 85],
                                       [231, 62, 138, 74, 141, 50, 53, 227, 43, 85],
                                       [231, 62, 138, 74, 141, 50, 53, 227, 43, 85],
                                       [231, 62, 138, 74, 141, 50, 53, 227, 43, 85],
                                       [2, 137, 36, 65, 37, 139, 240, 75, 77, 225],
                                       [2, 137, 36, 65, 37, 139, 240, 75, 77, 225],
                                       [2, 137, 36, 65, 37, 139, 240, 75, 77, 225],
                                       [2, 137, 36, 65, 37, 139, 240, 75, 77, 225]])
    expected_items = torch.tensor(
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], device=device,
    )

    actual_sequences, actual_items = get_sequential_item_pairs(sequences=sequences,
                                                               n_items=n_items,
                                                               device=device)

    assert torch.equal(actual_sequences, expected_sequences)
    assert torch.equal(actual_items, expected_items)


# @mock.patch('collie.model.MatrixFactorizationModel')
# def test_get_preds_implicit(model, test_implicit_predicted_scores, device):
#     n_users, n_items = test_implicit_predicted_scores.shape
#     user_ids = np.arange(n_users)
#     model.return_value = test_implicit_predicted_scores.view(-1)
#     actual_preds = get_preds(model=model,
#                              user_ids=user_ids,
#                              n_items=n_items,
#                              device=device)
#
#     assert torch.equal(actual_preds, test_implicit_predicted_scores)


def test_sequential_get_labels(sequential_targets,
                               test_sequential_labels,
                               test_sequential_recs,
                               device):
    actual_labels = sequential_get_labels(targets=sequential_targets,
                                          preds=test_sequential_recs,
                                          device=device)

    assert torch.equal(actual_labels, test_sequential_labels)


def test_sequential_get_labels_k(sequential_targets,
                                 test_sequential_labels,
                                 test_sequential_recs,
                                 device):
    k = 2

    actual_labels = sequential_get_labels(targets=sequential_targets,
                                          preds=test_sequential_recs[:, :k],
                                          device=device)

    expected_labels = test_sequential_labels[:, :k]

    assert torch.equal(actual_labels, expected_labels)


def test_sequential_mapk(sequential_targets, test_sequences, test_sequential_predicted_scores):
    actual_score = sequential_mapk(targets=sequential_targets,
                                   sequences=test_sequences,
                                   preds=test_sequential_predicted_scores,
                                   k=4)

    assert actual_score == 0.45


def test_sequential_map_1(sequential_targets, test_sequences, test_sequential_predicted_scores):
    actual_score = sequential_mapk(targets=sequential_targets,
                                   sequences=test_sequences,
                                   preds=test_sequential_predicted_scores,
                                   k=1)

    assert actual_score == (1/5)


def test_sequential_map_k_too_big(sequential_targets,
                                  test_sequences,
                                  test_sequential_predicted_scores):
    with pytest.raises(ValueError):
        sequential_mapk(targets=sequential_targets,
                        sequences=test_sequences,
                        preds=test_sequential_predicted_scores,
                        k=(test_sequential_predicted_scores.shape[1] + 1))


def test_sequential_mrr(sequential_targets, test_sequences, test_sequential_predicted_scores):
    actual_score = sequential_mrr(targets=sequential_targets,
                                  sequences=test_sequences,
                                  preds=test_sequential_predicted_scores)

    np.testing.assert_almost_equal(actual_score, 0.475)


# @pytest.mark.parametrize('batch_size', [20, 2, 1])  # default, uneven, single
# @mock.patch('collie.model.MatrixFactorizationModel')
# def test_evaluate_in_batches(
#     model,
#     test_implicit_interactions,
#     test_implicit_predicted_scores,
#     metrics,
#     batch_size,
# ):
#     model.side_effect = partial(get_model_scores, scores=test_implicit_predicted_scores)
#
#     # need to do this for the Mock in order for the metrics to be on the right device
#     model.device = 'cpu'
#
#     mapk_score, mrr_score, auc_score = evaluate_in_batches(
#         metric_list=[mapk, mrr, auc],
#         test_interactions=test_implicit_interactions,
#         model=model,
#         k=4,
#         batch_size=batch_size,
#     )
#
#     np.testing.assert_almost_equal(mapk_score, metrics['mapk'])
#     np.testing.assert_almost_equal(mrr_score, metrics['mrr'])
#     np.testing.assert_almost_equal(auc_score, metrics['auc'])
#
#
# def test_evaluate_in_batches_logger(
#     implicit_model,
#     test_implicit_interactions,
#     test_implicit_predicted_scores,
# ):
#     class LightningLoggerFixture():
#         """A simple logger base class with a method ``log_metrics``."""
#         def __init__(self):
#             pass
#
#         def log_metrics(self, metrics, step):
#             """Save ``metrics`` and ``step`` as class-level attributes for testing."""
#             self.metrics = metrics
#             self.step = step
#
#     logger = LightningLoggerFixture()
#     model = copy.deepcopy(implicit_model)
#
#     mapk_score, mrr_score, auc_score = evaluate_in_batches(
#         metric_list=[mapk, mrr, auc],
#         test_interactions=test_implicit_interactions,
#         model=model,
#         k=4,
#         logger=logger,
#     )
#
#     assert mapk_score == logger.metrics['mapk']
#     assert mrr_score == logger.metrics['mrr']
#     assert auc_score == logger.metrics['auc']
#
#     assert logger.step == model.hparams.num_epochs_completed
