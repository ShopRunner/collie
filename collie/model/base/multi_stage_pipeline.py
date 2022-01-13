from abc import ABCMeta
from collections import OrderedDict
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from collie.interactions import (ApproximateNegativeSamplingInteractionsDataLoader,
                                 Interactions,
                                 InteractionsDataLoader)
from collie.model.base import BasePipeline
from collie.utils import get_init_arguments, merge_docstrings


INTERACTIONS_LIKE_INPUT = Union[ApproximateNegativeSamplingInteractionsDataLoader,
                                Interactions,
                                InteractionsDataLoader]


class MultiStagePipeline(BasePipeline, metaclass=ABCMeta):
    """
    Multi-stage pipeline model architectures to inherit from.

    This model template is intended for models that train in distinct stages, with a different
    optimizer optimizing each step. This allows model components to be optimized with a set
    order in mind, rather than all at once, such as with the ``BasePipeline``.

    Generally, multi-stage models will have a training protocol like:

    .. code-block:: python

        from collie.model import CollieTrainer, SomeMultiStageModel


        model = SomeMultiStageModel(train=train)
        trainer = CollieTrainer(model)

        # fit stage 1
        trainer.fit(model)

        # fit stage 2
        trainer.max_epochs += 10
        model.advance_stage()
        trainer.fit(model)

        # fit stage 3
        trainer.max_epochs += 10
        model.advance_stage()
        trainer.fit(model)

        # ... and so on, until...

        model.eval()

    Just like with ``BasePipeline``, all subclasses MUST at least override the following methods:

    * ``_setup_model`` - Set up the model architecture

    * ``forward`` - Forward pass through a model

    For ``item_item_similarity`` to work properly, all subclasses are should also implement:

    * ``_get_item_embeddings`` - Returns item embeddings from the model

    Notes
    -----
    * With each call of ``trainer.fit``, the optimizer and learning rate scheduler state will reset.
    * When loading a multi-stage model in, the state will be set to the last possible state. This
      state may have a different ``forward`` calculation than other states.

    Parameters
    ----------
    optimizer_config_list: list of dict
        List of dictionaries containing the optimizer configurations for each stage's
        optimizer(s). Each dictionary must contain the following keys:

        * ``lr``: str
            Learning rate for the optimizer

        * ``optimizer``: ``torch.optim`` or ``str``

        * ``parameter_prefix_list``: List[str]
            List of string prefixes corressponding to the model components that should be
            optimized with this optimizer

        * ``stage``: str
            Name of stage

        This must be ordered with the intended progression of stages.

    """
    def __init__(self,
                 train: INTERACTIONS_LIKE_INPUT = None,
                 val: INTERACTIONS_LIKE_INPUT = None,
                 lr_scheduler_func: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 weight_decay: float = 0.0,
                 optimizer_config_list: List[Dict[str, Union[float, List[str], str]]] = None,
                 loss: Union[str, Callable[..., torch.tensor]] = 'hinge',
                 metadata_for_loss: Optional[Dict[str, torch.tensor]] = None,
                 metadata_for_loss_weights: Optional[Dict[str, float]] = None,
                 load_model_path: Optional[str] = None,
                 map_location: Optional[str] = None,
                 **kwargs):
        stage_list = None

        if load_model_path is None:
            if optimizer_config_list is None:
                raise ValueError(
                    'Must provide ``optimizer_config_list`` when initializing a new multi-stage '
                    'model!'
                )
            else:
                stage_list = list(
                    OrderedDict.fromkeys(
                        [optimizer_config['stage'] for optimizer_config in optimizer_config_list]
                    )
                )

        super().__init__(stage_list=stage_list,
                         **get_init_arguments())

        if load_model_path is None:
            # set stage if we have not already loaded it in and set it there
            self.hparams.stage = self.hparams.stage_list[0]
            self.set_stage(self.hparams.stage)

    __doc__ = merge_docstrings(BasePipeline, __doc__, __init__)

    def _load_model_init_helper(self, *args, **kwargs) -> None:
        super()._load_model_init_helper(*args, **kwargs)

        # set the stage to the last stage
        self.hparams.stage = self.hparams.stage_list[-1]
        print(f'Set ``self.hparams.stage`` to "{self.hparams.stage}"')

    def advance_stage(self) -> None:
        """Advance the stage to the next one in ``self.hparams.stage_list``."""
        stage = self.hparams.stage

        if stage in self.hparams.stage_list:
            stage_idx = self.hparams.stage_list.index(stage)
            if (stage_idx + 1) >= len(self.hparams.stage_list):
                raise ValueError(f'Cannot advance stage past {stage} - it is the final stage!')

            self.set_stage(stage=self.hparams.stage_list[stage_idx + 1])

    def set_stage(self, stage: str) -> None:
        """Set the model to the desired stage."""
        if stage in self.hparams.stage_list:
            self.hparams.stage = stage
            print(f'Set ``self.hparams.stage`` to "{self.hparams.stage}"')
        else:
            raise ValueError(
                f'{stage} is not a valid stage, please choose one of {self.hparams.stage_list}'
            )

    def _get_optimizer_parameters(
            self,
            optimizer_config: List[Dict[str, Union[float, List[str], str]]],
            include_weight_decay: bool = True,
            **kwargs
    ) -> List[Dict[str, Union[torch.tensor, float]]]:
        optimizer_parameters = [
            {
                'params': (
                    param for (name, param) in self.named_parameters()
                    if reduce(
                        lambda x, y: x or y,
                        [
                            name.startswith(prefix) for prefix in
                            optimizer_config['parameter_prefix_list']
                        ],
                        False,
                    )
                ),
                'lr': optimizer_config['lr'],
            }
        ]

        if include_weight_decay:
            weight_decay_dict = {'weight_decay': self.hparams.weight_decay}
            [d.update(weight_decay_dict) for d in optimizer_parameters]

        return optimizer_parameters

    def configure_optimizers(self) -> (
        Union[
            Tuple[List[torch.optim.Optimizer], List[torch.optim.Optimizer]],
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
            torch.optim.Optimizer
        ]
    ):
        """
        Configure optimizers and learning rate schedulers to use in optimization.

        This method will be called after `setup`.

        Creates an optimizer and learning rate scheduler for each configuration dictionary in
        ``self.hparams.optimizer_config_list``.

        """
        # since this is the only function that is called before each ``trainer.fit`` call, we will
        # also take this time to ensure any external data a model might rely on has been properly
        # moved to the device before training
        self._move_any_external_data_to_device()

        optimizer_config_list = [
            self._get_optimizer(self.optimizer, optimizer_config=optimizer_config)
            for optimizer_config in self.hparams.optimizer_config_list
        ]

        if self.lr_scheduler_func is not None:
            monitor = 'val_loss_epoch'
            if self.val_loader is None:
                monitor = 'train_loss_epoch'

            # add in optimizer to scheduler function
            scheduler_list = [
                {
                    'scheduler': self.lr_scheduler_func(optimizer_config),
                    'monitor': monitor,
                }
                for optimizer_config in optimizer_config_list
            ]

            return optimizer_config_list, scheduler_list

        else:
            return optimizer_config_list

    def optimizer_step(self,
                       epoch: int = None,
                       batch_idx: int = None,
                       optimizer: torch.optim.Optimizer = None,
                       optimizer_idx: int = None,
                       optimizer_closure: Optional[Callable[..., Any]] = None,
                       **kwargs) -> None:
        """
        Overriding Lightning's optimizer step function to only step the optimizer associated with
        the relevant stage.

        See here for more details:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#optimizer-step

        Parameters
        ----------
        epoch: int
            Current epoch
        batch_idx: int
            Index of current batch
        optimizer: torch.optim.Optimizer
            A PyTorch optimizer
        optimizer_idx: int
            If you used multiple optimizers, this indexes into that list
        optimizer_closure: Callable
            Closure for all optimizers

        """
        if self.hparams.optimizer_config_list[optimizer_idx]['stage'] == self.hparams.stage:
            optimizer.step(closure=optimizer_closure)
        elif optimizer_closure is not None:
            optimizer_closure()
