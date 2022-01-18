from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
import warnings

import numpy as np
import pytorch_lightning
from scipy.sparse import csr_matrix
import torch
from torchmetrics import Metric
from torchmetrics.functional import auroc
from tqdm.auto import tqdm

import collie
from collie.interactions import ExplicitInteractions, Interactions, InteractionsDataLoader
from collie.model import BasePipeline


def _get_user_item_pairs(user_ids: Union[np.array, torch.tensor],
                         n_items: int,
                         device: Union[str, torch.device]) -> Tuple[torch.tensor, torch.tensor]:
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
    -------
    users: torch.tensor, 1-d
        Tensor with ``n_items`` copies of each user ID
    items: torch.tensor, 1-d
        Tensor with ``len(user_ids)`` copies of each item ID

    Example
    -------
    .. code-block:: python

        >>> user_ids = np.array([10, 11, 12])
        >>> n_items = 4
        >>> user, item = _get_user_item_pairs(user_ids: user_ids, n_items: 4, device: 'cpu'):
        >>> user
        np.array([10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12])
        >>> item
        np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

    """
    # Added because sometimes we call this function with ``n_items`` as ``np.int64`` type which
    # breaks ``repeat_interleave``.
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
              user_ids: Union[np.array, torch.tensor],
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
    user, item = _get_user_item_pairs(user_ids, n_items, device)

    with torch.no_grad():
        predicted_scores = model(user, item)

    return predicted_scores.view(-1, n_items)


def _get_labels(targets: csr_matrix,
                user_ids: Union[np.array, torch.tensor],
                preds: Union[np.array, torch.tensor],
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
    -------
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
         user_ids: Union[np.array, torch.tensor],
         preds: Union[np.array, torch.tensor],
         k: int = 10) -> float:
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
    -------
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
        user_ids: Union[np.array, torch.tensor],
        preds: Union[np.array, torch.tensor],
        k: Optional[Any] = None) -> float:
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
    k: Any
        Ignored, included only for compatibility with ``mapk``

    Returns
    -------
    mrr_score: float

    """
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
        user_ids: Union[np.array, torch.tensor],
        preds: Union[np.array, torch.tensor],
        k: Optional[Any] = None) -> float:
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
    k: Any
        Ignored, included only for compatibility with ``mapk``

    Returns
    -------
    auc_score: float

    """
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
    metric_list: Iterable[Callable[
        [csr_matrix, Union[np.array, torch.tensor], Union[np.array, torch.tensor], Optional[int]],
        float
    ]],
    test_interactions: collie.interactions.Interactions,
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

        from collie.metrics import auc, evaluate_in_batches, mapk, mrr


        map_10_score, mrr_score, auc_score = evaluate_in_batches(
            metric_list=[mapk, mrr, auc],
            test_interactions=test,
            model=model,
        )

        print(map_10_score, mrr_score, auc_score)

    """
    if not isinstance(test_interactions, Interactions):
        raise ValueError(
            '``test_interactions`` must be of type ``Interactions``, not '
            f'{type(test_interactions)}. Try using ``explicit_evaluate_in_batches`` instead.'
        )

    device = _get_evaluate_in_batches_device(model=model)
    model.to(device)
    model._move_any_external_data_to_device()

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
        _log_metrics(model=model,
                     logger=logger,
                     metric_list=metric_list,
                     all_scores=all_scores,
                     verbose=verbose)

    return all_scores[0] if len(all_scores) == 1 else all_scores


def explicit_evaluate_in_batches(
    metric_list: Iterable[Metric],
    test_interactions: collie.interactions.ExplicitInteractions,
    model: collie.model.BasePipeline,
    logger: pytorch_lightning.loggers.base.LightningLoggerBase = None,
    verbose: bool = True,
    **kwargs,
) -> List[float]:
    """
    Evaluate a model with potentially several different metrics.

    Memory constraints require that most test sets will need to be evaluated in batches. This
    function handles the looping and batching boilerplate needed to properly evaluate the model
    without running out of memory.

    Parameters
    ----------
    metric_list: list of ``torchmetrics.Metric``
        List of evaluation functions to apply. Each function must accept arguments for predictions
        and targets, in order
    test_interactions: collie.interactions.ExplicitInteractions
    model: collie.model.BasePipeline
        Model that can take a (user_id, item_id) pair as input and return a recommendation score
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
    **kwargs: keyword arguments
        Additional arguments sent to the ``InteractionsDataLoader``

    Returns
    ----------
    evaluation_results: list
        List of floats, with each metric value corresponding to the respective function passed in
        ``metric_list``

    Examples
    -------------
    .. code-block:: python

        import torchmetrics

        from collie.metrics import explicit_evaluate_in_batches


        mse_score, mae_score = evaluate_in_batches(
            metric_list=[torchmetrics.MeanSquaredError(), torchmetrics.MeanAbsoluteError()],
            test_interactions=test,
            model=model,
        )

        print(mse_score, mae_score)

    """
    if not isinstance(test_interactions, ExplicitInteractions):
        raise ValueError(
            '``test_interactions`` must be of type ``ExplicitInteractions``, not '
            f'{type(test_interactions)}. Try using ``evaluate_in_batches`` instead.'
        )

    try:
        device = _get_evaluate_in_batches_device(model=model)
        model.to(device)
        model._move_any_external_data_to_device()

        test_loader = InteractionsDataLoader(interactions=test_interactions,
                                             **kwargs)

        data_to_iterate_over = test_loader
        if verbose:
            data_to_iterate_over = tqdm(test_loader)

        for batch in data_to_iterate_over:
            users, items, ratings = batch

            # move data to batch before sending to model
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.cpu()

            preds = model(users, items)

            for metric in metric_list:
                metric(preds.cpu(), ratings)

        all_scores = [metric.compute() for metric in metric_list]

        if logger is not None:
            _log_metrics(model=model,
                         logger=logger,
                         metric_list=metric_list,
                         all_scores=all_scores,
                         verbose=verbose)

        return all_scores[0] if len(all_scores) == 1 else all_scores
    finally:
        for metric in metric_list:
            metric.reset()


def _get_evaluate_in_batches_device(model: BasePipeline):
    device = getattr(model, 'device', None)

    if torch.cuda.is_available() and str(device) == 'cpu':
        warnings.warn('CUDA available but model device is set to CPU - is this desired?')

    if device is None:
        if torch.cuda.is_available():
            warnings.warn(
                '``model.device`` attribute is ``None``. Since GPU is available, putting model on '
                'GPU.'
            )
            device = 'cuda:0'
        else:
            device = 'cpu'

    return device


def _log_metrics(model: BasePipeline,
                 logger: pytorch_lightning.loggers.base.LightningLoggerBase,
                 metric_list: List[Union[Callable[..., Any], Metric]],
                 all_scores: List[float],
                 verbose: bool):
    try:
        step = model.hparams.get('num_epochs_completed')
    except torch.nn.modules.module.ModuleAttributeError:
        # if, somehow, there is no ``model.hparams`` attribute, this shouldn't fail
        step = None

    try:
        metrics_dict = dict(zip([x.__name__ for x in metric_list], all_scores))
    except AttributeError:
        metrics_dict = dict(zip([type(x).__name__ for x in metric_list], all_scores))

    if verbose:
        print(f'Logging metrics {metrics_dict} to ``logger``...')

    logger.log_metrics(metrics=metrics_dict, step=step)
