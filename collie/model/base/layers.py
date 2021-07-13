from typing import Dict, List

import torch


class ScaledEmbedding(torch.nn.Embedding):
    """Embedding layer that initializes its values to use a truncated normal distribution."""
    def reset_parameters(self) -> None:
        """Overriding default ``reset_parameters`` method."""
        self.weight.data.normal_(0, 1.0 / (self.embedding_dim * 2.5))


class ZeroEmbedding(torch.nn.Embedding):
    """Embedding layer with weights zeroed-out."""
    def reset_parameters(self) -> None:
        """Overriding default ``reset_parameters`` method."""
        self.weight.data.zero_()


class MultiOptimizer(object):
    """
    A simple class that allows us to wrap multiple optimizers into a single API typical of a single
    optimizer, with ``zero_grad`` and ``step`` methods.

    Parameters
    ----------
    optimizers: List of ``torch.optim.Optimizer``s

    """
    def __init__(self, optimizers: List[torch.optim.Optimizer]):
        assert isinstance(optimizers, list), f'Expected list, got {type(optimizers)}!'

        self.optimizers = optimizers

    def zero_grad(self) -> None:
        """Apply ``zero_grad`` to all optimizers."""
        for optimizer in self.optimizers:
            optimizer.zero_grad()


class MultiLRScheduler(object):
    """
    A simple class that allows us to wrap multiple learning rate schedulers into a single API
    typical of a single learning rate scheduler with a ``step`` method.

    Parameters
    ----------
    lr_schedulers: List of dictionaries
        Each dictionary must have key ``scheduler``, which contains the actual learning rate
        scheduler

    """
    def __init__(self, lr_schedulers: List[Dict[str, torch.optim.lr_scheduler._LRScheduler]]):
        assert isinstance(lr_schedulers, list), f'Expected list, got {type(lr_schedulers)}!'

        lr_schedulers_dicts_removed = [lr_scheduler['scheduler'] for lr_scheduler in lr_schedulers]

        self.lr_schedulers = lr_schedulers_dicts_removed

    def step(self, *args, **kwargs) -> None:
        """Apply ``step`` to all optimizers."""
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step(*args, **kwargs)
