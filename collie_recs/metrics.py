from typing import Callable, Iterable, List, Tuple

import numpy as np
import pytorch_lightning
from scipy.sparse import csr_matrix
import torch
from torchmetrics.functional import auroc
from tqdm.auto import tqdm

import collie_recs
from collie_recs.model import BasePipeline


def _get_user_item_pairs(user_ids: (np.array, torch.tensor),
                         n_items: int, device: str) -> Tuple[torch.tensor, torch.tensor]:
    """
    Create tensors pairing each input user ID with each item ID.

    Parameters
    ----------
    user_ids: np.array or torch.tensor, 1-d
        Iterable[int] of users to score
    n_items: int
        Number of items in the training data
    device: string
        Device to store tensors on

    Returns
    ----------
    users: torch.tensor, 1-d
        Tensor with ``n_items`` copies of each user ID
    items: torch.tensor, 1-d
        Tensor with ``len(user_ids)`` copies of each item ID

    Example
    ----------
    .. code-block:: python

        >>> user_ids = np.array([10, 11, 12])
        >>> n_items = 4
        >>> user, item = _get_user_item_pairs(user_ids: user_ids, n_items: 4, device: 'cpu'):
        >>> user
        np.array([10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12])
        >>> item
        np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

    """
    # Added because sometimes we call this function with n_items
    # as np.int64 type which breaks repeat_interleave.
    if isinstance(n_items, np.int64):
        n_items = n_items.item()

    users = torch.tensor(
        user_ids,
        dtype=torch.int64,
        requires_grad=False,
        device=device,
    ).repeat_interleave(n_items)

    items = torch.arange(
        start=0,
        end=n_items,
        requires_grad=False,
        device=device,
    ).repeat(len(user_ids))

    return users, items


def get_preds(model: BasePipeline,
              user_ids: (np.array, torch.tensor),
              n_items: int,
              device: str) -> torch.tensor:
    """
    Returns a ``n_users x n_items`` tensor with the item IDs of recommended products for each user
    ID.

    Parameters
    ----------
    model: collie_recs.model.BasePipeline
        Model that can take a (user_id, item_id) pair as input and return a recommendation score
    user_ids: np.array or torch.tensor
        Iterable[int] of users to score
    n_items: int
        Number of items in the training data
    device: string
        Device torch should use

    Returns
    ----------
    predicted_scores: torch.tensor
        Tensor of shape ``n_users x n_items``

    """
    user, item = _get_user_item_pairs(user_ids, n_items, device)

    with torch.no_grad():
        predicted_scores = model(user, item)

    return predicted_scores.view(-1, n_items)


def _get_labels(targets: csr_matrix,
                user_ids: (np.array, torch.tensor),
                preds: (np.array, torch.tensor),
                device: str) -> torch.tensor:
    """
    Returns a binary array indicating which of the recommended products are in each user's target
    set.

    Parameters
    ----------
    targets: scipy.sparse.csr_matrix
        Interaction matrix containing user and item IDs
    user_ids: np.array or torch.tensor
        Users corresponding to the recommendations in the top k predictions
    preds: torch.tensor
        Top ``k`` item IDs to recommend to each user of shape (n_users x k)
    device: string
        Device torch should use

    Returns
    ----------
    labels: torch.tensor
        Tensor with the same dimensions as input ``preds``

    """
    return torch.tensor(
        (targets[user_ids[:, None], np.array(preds.detach().cpu())] > 0)
        .astype('double')
        .toarray(),
        requires_grad=False,
        device=device,
    )


def mapk(targets: csr_matrix,
         user_ids: (np.array, torch.tensor),
         preds: (np.array, torch.tensor),
         k: int) -> float:
    """
    Calculate the mean average precision at K (MAP@K) score for each user.

    Parameters
    ----------
    targets: scipy.sparse.csr_matrix
        Interaction matrix containing user and item IDs
    user_ids: np.array or torch.tensor
        Users corresponding to the recommendations in the top k predictions
    preds: torch.tensor
        Tensor of shape (n_users x n_items) with each user's scores for each item
    k: int
        Number of recommendations to consider per user

    Returns
    ----------
    mapk_score: float

    """
    device = preds.device
    n_users = preds.shape[0]

    try:
        predicted_items = preds.topk(k, dim=1).indices
    except RuntimeError as e:
        raise ValueError(
            f'Ensure ``k`` ({k}) is less than the number of items ({preds.shape[1]}):', str(e)
        )

    topk_labeled = _get_labels(targets, user_ids, predicted_items, device)
    accuracy = topk_labeled.int()

    weights = (
        1.0 / torch.arange(
            start=1,
            end=k+1,
            dtype=torch.float64,
            requires_grad=False,
            device=device
        )
    ).repeat(n_users, 1)

    denominator = torch.min(
        torch.tensor(k, device=device, dtype=torch.int).repeat(len(user_ids)),
        torch.tensor(targets[user_ids].getnnz(axis=1), device=device)
    )

    res = ((accuracy * accuracy.cumsum(axis=1) * weights).sum(axis=1)) / denominator
    res[torch.isnan(res)] = 0

    return res.mean().item()


def mrr(targets: csr_matrix,
        user_ids: (np.array, torch.tensor),
        preds: (np.array, torch.tensor),
        **kwargs) -> float:
    """
    Calculate the mean reciprocal rank (MRR) of the input predictions.

    Parameters
    ----------
    targets: scipy.sparse.csr_matrix
        Interaction matrix containing user and item IDs
    user_ids: np.array or torch.tensor
        Users corresponding to the recommendations in the top k predictions
    preds: torch.tensor
        Tensor of shape (n_users x n_items) with each user's scores for each item
    kwargs: keyword arguments
        Ignored, included only for compatibility with ``mapk``

    Returns
    ----------
    mrr_score: float

    """
    if len(kwargs) > 0 and [kwargs_key for kwargs_key in kwargs] != ['k']:
        raise ValueError(f'Unexpected ``kwargs``: {kwargs}')

    predicted_items = preds.topk(preds.shape[1], dim=1).indices
    labeled = _get_labels(targets, user_ids, predicted_items, device=preds.device)

    # weighting each 0/1 by position so that topk returns index of *first* postive result
    position_weight = 1.0/(
        torch.arange(1, targets.shape[1] + 1, device=preds.device)
        .repeat(len(user_ids), 1)
        .float()
    )
    labeled_weighted = (labeled.float() * position_weight)

    highest_score, rank = labeled_weighted.topk(k=1)

    reciprocal_rank = 1.0/(rank.float() + 1)
    reciprocal_rank[highest_score == 0] = 0

    return reciprocal_rank.mean().item()


def auc(targets: csr_matrix,
        user_ids: (np.array, torch.tensor),
        preds: (np.array, torch.tensor),
        **kwargs) -> float:
    """
    Calculate the area under the ROC curve (AUC) for each user and average the results.

    Parameters
    ----------
    targets: scipy.sparse.csr_matrix
        Interaction matrix containing user and item IDs
    user_ids: np.array or torch.tensor
        Users corresponding to the recommendations in the top k predictions
    preds: torch.tensor
        Tensor of shape (n_users x n_items) with each user's scores for each item
    kwargs: keyword arguments
        Ignored, included only for compatibility with ``mapk``

    Returns
    ----------
    auc_score: float

    """
    if len(kwargs) > 0 and [kwargs_key for kwargs_key in kwargs] != ['k']:
        raise ValueError(f'Unexpected ``kwargs``: {kwargs}')

    agg = 0
    for i, user_id in enumerate(user_ids):
        target_tensor = torch.tensor(
            targets[user_id].toarray(),
            device=preds.device,
            dtype=torch.long
        ).view(-1)
        # many models' ``preds`` may be unbounded if a final activation layer is not applied
        # we have to normalize ``preds`` here to avoid a ``ValueError`` stating that ``preds``
        # should be probabilities, but values were detected outside of [0,1] range
        auc = auroc(torch.sigmoid(preds[i, :]), target=target_tensor, pos_label=1)
        agg += auc

    return (agg/len(user_ids)).item()


def evaluate_in_batches(
    metric_list: Iterable[Callable],
    test_interactions: collie_recs.interactions.Interactions,
    model: collie_recs.model.BasePipeline,
    k: int = 10,
    batch_size: int = 20,
    logger: pytorch_lightning.loggers.base.LightningLoggerBase = None,
    verbose: bool = True,
) -> List[float]:
    """
    Evaluate a model with potentially several different metrics.

    Memory constraints require that most test sets will need to be evaluated in batches. This
    function handles the looping and batching boilerplate needed to properly evaluate the model
    without running out of memory.

    Parameters
    ----------
    metric_list: list of functions
        List of evaluation functions to apply. Each function must accept keyword arguments:

        * ``targets``

        * ``user_ids``

        * ``preds``

        * ``k``

    test_interactions: collie_recs.interactions.Interactions
        Interactions to use as labels
    model: collie_recs.model.BasePipeline
        Model that can take a (user_id, item_id) pair as input and return a recommendation score
    k: int
        Number of recommendations to consider per user. This is ignored by some metrics
    batch_size: int
        Number of users to score in a single batch. For best efficiency, this number should be as
        high as possible without running out of memory
    logger: pytorch_lightning.loggers.base.LightningLoggerBase
        If provided, will log outputted metrics dictionary using the ``log_metrics`` method with
        keys being the string representation of ``metric_list`` and values being
        ``evaluation_results``. Additionally, if ``model.hparams.n_epochs_completed_`` exists, this
        will be logged as well, making it possible to track metrics progress over the course of
        model training
    verbose: bool
        Display progress bar and print statements during function execution

    Returns
    ----------
    evaluation_results: list
        List of floats, with each metric value corresponding to the respective function passed in
        ``metric_list``

    Examples
    -------------
    .. code-block:: python

        from collie_recs.metrics import auc, evaluate_in_batches, mapk, mrr


        map_10_score, mrr_score, auc_score = evaluate_in_batches(
            metric_list=[mapk, mrr, auc],
            test_interactions=test,
            model=model,
        )

        print(map_10_score, mrr_score, auc_score)

    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    test_users = np.unique(test_interactions.mat.row)
    targets = test_interactions.mat.tocsr()

    if len(test_users) < batch_size:
        batch_size = len(test_users)

    accumulators = [0] * len(metric_list)

    data_to_iterate_over = range(int(np.ceil(len(test_users) / batch_size)))
    if verbose:
        data_to_iterate_over = tqdm(data_to_iterate_over)

    for i in data_to_iterate_over:
        user_range = test_users[i * batch_size:(i + 1) * batch_size]
        preds = get_preds(model, user_range, test_interactions.num_items, device)
        for metric_ind, metric in enumerate(metric_list):
            score = metric(targets=targets, user_ids=user_range, preds=preds, k=k)
            accumulators[metric_ind] += (score * len(user_range))

    all_scores = [acc_score / len(test_users) for acc_score in accumulators]

    if logger is not None:
        try:
            step = model.hparams.get('n_epochs_completed_')
        except torch.nn.modules.module.ModuleAttributeError:
            step = None

        metrics_dict = dict(zip([x.__name__ for x in metric_list], all_scores))

        if verbose:
            print(f'Logging metrics {metrics_dict} to ``logger``...')

        logger.log_metrics(metrics=metrics_dict, step=step)

    return all_scores[0] if len(all_scores) == 1 else all_scores
