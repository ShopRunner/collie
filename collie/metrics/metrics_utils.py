from typing import Callable, List, Union
import warnings

import pytorch_lightning
import torch
from torchmetrics import Metric

from collie.model import BasePipeline


def _get_evaluate_in_batches_device(model: BasePipeline):
    device = getattr(model, 'device') or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available() and getattr(model, 'device') == 'cpu':
        warnings.warn('CUDA available but model device is set to CPU - is this desired?')

    return device


def _log_metrics(model: BasePipeline,
                 logger: pytorch_lightning.loggers.base.LightningLoggerBase,
                 metric_list: List[Union[Callable, Metric]],
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
