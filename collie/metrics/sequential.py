from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning
import torch
from tqdm.auto import tqdm

import collie
from collie.interactions import SequentialInteractions
from collie.metrics.metrics_utils import _get_evaluate_in_batches_device, _log_metrics
from collie.model import BasePipeline


def get_sequential_item_pairs(
    sequences: Union[np.array, torch.tensor],
    n_items: int,
    device: Union[str, torch.device],
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Create tensors pairing each input sequence with each item ID.

    Parameters
    ----------
    sequences: np.array or torch.tensor, 2-d
        Iterable[int] of sequences to run through the model
    n_items: int
        Number of items in the training data
    device: string
        Device to store tensors on

    Returns
    -------
    sequences_repeated: torch.tensor, 2-d
        ``sequences`` with each row repeated for every item in ``items``
    items: torch.tensor, 1-d
        Tensor with ``len(sequences_repeated)`` copies of each item ID

    Example
    -------
    .. code-block:: python

        >>> sequences = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> n_items = 4
        >>> sequences_repeated, item = get_sequential_item_pairs(sequences=sequences,
        >>>                                                       n_items=4,
        >>>                                                       device='cpu')
        >>> sequences_repeated
        np.array([[1, 2, 3],
                  [1, 2, 3],
                  [1, 2, 3],
                  [1, 2, 3],
                  [4, 5, 6],
                  [4, 5, 6],
                  [4, 5, 6],
                  [4, 5, 6],
                  [7, 8, 9],
                  [7, 8, 9],
                  [7, 8, 9],
                  [7, 8, 9]])
        >>> item
        np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

    """
    # Added because sometimes we call this function with ``n_items`` as ``np.int64`` type which
    # breaks ``repeat_interleave``.
    if isinstance(n_items, np.int64):
        n_items = n_items.item()

    sequences_repeated = torch.tensor(
        sequences,
        dtype=torch.int64,
        requires_grad=False,
        device=device,
    ).repeat_interleave(n_items, dim=0)

    items = torch.arange(
        start=0,
        end=n_items,
        requires_grad=False,
        device=device,
    ).repeat(len(sequences))

    return sequences_repeated, items


def sequential_get_preds(model: BasePipeline,
                         sequences: Union[np.array, torch.tensor],
                         n_items: int,
                         device: Union[str, torch.device]) -> torch.tensor:
    """
    Returns a ``n_users x n_items`` tensor with the item IDs of recommended products for each user
    ID.

    Parameters
    ----------
    model: collie.model.BasePipeline
        Model that can take a (user_id, item_id) pair as input and return a recommendation score
    user_ids: np.array or torch.tensor
        Iterable[int] of users to score
    n_items: int
        Number of items in the training data
    device: string
        Device torch should use

    Returns
    -------
    predicted_scores: torch.tensor
        Tensor of shape ``n_users x n_items``

    """
    sequences_repeated, item = get_sequential_item_pairs(sequences=sequences,
                                                         n_items=n_items,
                                                         device=device)
    item = item.reshape(-1, 1)

    with torch.no_grad():
        sequences_repeated = model.compute_user_representation(sequences_repeated, predicting=True)
        size = (len(item),) + sequences_repeated.size()[1:]
        predicted_scores = model(sequences_repeated.expand(*size), item)

    return predicted_scores.view(-1, n_items)


def sequential_get_labels(targets: Union[np.array, torch.tensor],
                          predicted_items: Union[np.array, torch.tensor],
                          device: str) -> torch.tensor:
    """
    Returns a binary array indicating which of the recommended items are in each sequences's target
    set.

    Parameters
    ----------
    targets: np.array or torch.tensor, 1-d
        Target item IDs for each sequence
    predicted_items: torch.tensor
        Top ``k`` item IDs to recommend as the final item of each sequence, of shape
        (n_sequences x k)
    device: string
        Device torch should use

    Returns
    -------
    labels: torch.tensor
        Tensor with the same dimensions as input ``preds``

    """
    return torch.tensor(
        (predicted_items == torch.tensor(targets)[:, None])
        .double(),
        # .toarray(),
        requires_grad=False,
        device=device,
    )


def sequential_mapk(targets: Union[np.array, torch.tensor],
                    sequences: Union[np.array, torch.tensor],
                    preds: (np.array, torch.tensor),
                    k: int = 10) -> float:
    """
    Calculate the mean average precision at K (MAP@K) score for each user.

    Parameters
    ----------
    targets: np.array or torch.tensor, 1-d
        Target item IDs for each sequence
    sequences: np.array or torch.tensor, 2-d
        Iterable[int] of sequences to run through the model
    preds: torch.tensor
        Tensor of shape (n_users x n_items) with each user's scores for each item
    k: int
        Number of recommendations to consider per user

    Returns
    -------
    mapk_score: float

    """
    device = preds.device
    n_sequences = preds.shape[0]

    try:
        predicted_items = preds.topk(k, dim=1).indices
    except RuntimeError as e:
        raise ValueError(
            f'Ensure ``k`` ({k}) is less than the number of items ({preds.shape[1]}):', str(e)
        )

    topk_labeled = sequential_get_labels(targets=targets,
                                         predicted_items=predicted_items,
                                         device=preds.device)
    accuracy = topk_labeled.int()

    weights = (
        1.0 / torch.arange(
            start=1,
            end=k+1,
            dtype=torch.float64,
            requires_grad=False,
            device=device
        )
    ).repeat(n_sequences, 1)

    # setting ``denominator`` to ``1`` since the maximum target for a given sequence is only a
    # single item
    denominator = 1

    res = ((accuracy * accuracy.cumsum(axis=1) * weights).sum(axis=1)) / denominator
    res[torch.isnan(res)] = 0

    return res.mean().item()


def sequential_mrr(targets: Union[np.array, torch.tensor],
                   sequences: Union[np.array, torch.tensor],
                   preds: Union[np.array, torch.tensor],
                   k: Optional[Any] = None) -> float:
    """
    Calculate the mean reciprocal rank (MRR) of the input predictions.

    Parameters
    ----------
    targets: np.array or torch.tensor, 1-d
        Target item IDs for each sequence
    sequences: np.array or torch.tensor, 2-d
        Iterable[int] of sequences to run through the model
    preds: torch.tensor
        Tensor of shape (n_users x n_items) with each user's scores for each item
    k: Any
        Ignored, included only for compatibility with ``mapk``

    Returns
    -------
    mrr_score: float

    """
    predicted_items = preds.topk(preds.shape[1], dim=1).indices
    labeled = sequential_get_labels(targets=targets,
                                    predicted_items=predicted_items,
                                    device=preds.device)

    # weighting each 0/1 by position so that topk returns index of *first* postive result
    position_weight = 1.0/(
        torch.arange(1, preds.shape[1] + 1, device=preds.device)
        .repeat(len(sequences), 1)
        .float()
    )
    labeled_weighted = (labeled.float() * position_weight)

    highest_score, rank = labeled_weighted.topk(k=1)

    reciprocal_rank = 1.0/(rank.float() + 1)
    reciprocal_rank[highest_score == 0] = 0

    return reciprocal_rank.mean().item()


def sequential_evaluate_in_batches(
    metric_list: Iterable[Callable],
    test_interactions: collie.interactions.SequentialInteractions,
    model: collie.model.BasePipeline,
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

    test_interactions: collie.interactions.Interactions
        Interactions to use as labels
    model: collie.model.BasePipeline
        Model that can take a (user_id, item_id) pair as input and return a recommendation score
    k: int
        Number of recommendations to consider per user. This is ignored by some metrics
    batch_size: int
        Number of users to score in a single batch. For best efficiency, this number should be as
        high as possible without running out of memory
    logger: pytorch_lightning.loggers.base.LightningLoggerBase
        If provided, will log outputted metrics dictionary using the ``log_metrics`` method with
        keys being the string representation of ``metric_list`` and values being
        ``evaluation_results``. Additionally, if ``model.hparams.num_epochs_completed`` exists, this
        will be logged as well, making it possible to track metrics progress over the course of
        model training
    verbose: bool
        Display progress bar and print statements during function execution

    Returns
    -------
    evaluation_results: list
        List of floats, with each metric value corresponding to the respective function passed in
        ``metric_list``

    Examples
    --------
    .. code-block:: python

        from collie.metrics import sequential_evaluate_in_batches, sequential_mapk, sequential_mrr


        map_10_score, mrr_score = evaluate_in_batches(
            metric_list=[sequential_mapk, sequential_mrr],
            test_interactions=test,
            model=model,
        )

        print(map_10_score, mrr_score, auc_score)

    """
    if not isinstance(test_interactions, SequentialInteractions):
        # TODO: fix this
        raise ValueError(
            '``test_interactions`` must be of type ``SequentialInteractions``, not '
            f'{type(test_interactions)}. Try using ``evaluate_in_batches`` or '
            '``explicit_evaluate_in_batches`` instead.'
        )

    device = _get_evaluate_in_batches_device(model=model)
    model.to(device)

    test_sequences = test_interactions.sequences[:, :-1]
    targets = test_interactions.sequences[:, -1]

    if len(test_sequences) < batch_size:
        batch_size = len(test_sequences)

    accumulators = [0] * len(metric_list)

    data_to_iterate_over = range(int(np.ceil(len(test_sequences) / batch_size)))
    if verbose:
        data_to_iterate_over = tqdm(data_to_iterate_over)

    for i in data_to_iterate_over:
        sequence_range = test_sequences[(i * batch_size):((i + 1) * batch_size)]
        targets_range = targets[(i * batch_size):((i + 1) * batch_size)]

        preds = sequential_get_preds(model=model,
                                     sequences=sequence_range,
                                     n_items=test_interactions.num_items,
                                     device=device)

        for metric_ind, metric in enumerate(metric_list):
            score = metric(targets=targets_range, sequences=sequence_range, preds=preds, k=k)
            accumulators[metric_ind] += (score * len(sequence_range))

    all_scores = [acc_score / len(test_sequences) for acc_score in accumulators]

    if logger is not None:
        _log_metrics(model=model,
                     logger=logger,
                     metric_list=metric_list,
                     all_scores=all_scores,
                     verbose=verbose)

    return all_scores[0] if len(all_scores) == 1 else all_scores
