from typing import Iterable, List

import pytorch_lightning
from torchmetrics import Metric
from tqdm.auto import tqdm

import collie
from collie.interactions import ExplicitInteractions, InteractionsDataLoader
from collie.metrics.metrics_utils import _get_evaluate_in_batches_device, _log_metrics


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
    kwargs: keyword arguments
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
