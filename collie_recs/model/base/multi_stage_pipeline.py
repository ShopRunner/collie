from functools import reduce
import multiprocessing
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from collie_recs.interactions import (ApproximateNegativeSamplingInteractionsDataLoader,
                                      Interactions,
                                      InteractionsDataLoader)
from collie_recs.model import BasePipeline


INTERACTIONS_LIKE_INPUT = Union[ApproximateNegativeSamplingInteractionsDataLoader,
                                Interactions,
                                InteractionsDataLoader]


# TODO: make sure this works for minimal trainer as well

class MultiStagePipeline(BasePipeline):  # TODO: check all types
    """Pipeline which allows for multiple stages in the training process"""
    def __init__(
        self,
        train: INTERACTIONS_LIKE_INPUT = None,
        val: INTERACTIONS_LIKE_INPUT = None,
        batch_size: int = 1024,
        optimizer_config_list: List = None,
        stage: str = None,
        lr_scheduler_func: Optional[Callable] = None,
        weight_decay: float = 0.0,
        loss: Union[str, Callable] = 'hinge',
        metadata_for_loss: Optional[Dict[str, torch.tensor]] = None,
        metadata_for_loss_weights: Optional[Dict[str, float]] = None,
        approximate_negative_sampling: bool = False,
        num_workers: int = multiprocessing.cpu_count(),
        load_model_path: Optional[str] = None,
        map_location: Optional[str] = None,
        **kwargs,
    ):
        """  # TODO: format docstring using the merge helpers
        Currently supports a single optimizer per stage.
        Each stage has it's own instance of the lr scheduler, but they are all the same type.
        You can then define different architectures for different stages in `forward`.


        Parameters
        ----------
        with keys 'lr, 'parameter_filter',
        parameter_filter should be a function that takes the name of a parameter and
        returns a boolean indicating whether or not it is a part of that stage
        """
        super().__init__(
            train=train,
            val=val,
            batch_size=batch_size,
            optimizer_config_list=optimizer_config_list,
            stage_list=list(dict.fromkeys([c['stage'] for c in optimizer_config_list])),
            stage=stage,
            lr_scheduler_func=lr_scheduler_func,
            weight_decay=weight_decay,
            loss=loss,
            metadata_for_loss=metadata_for_loss,
            metadata_for_loss_weights=metadata_for_loss_weights,
            approximate_negative_sampling=approximate_negative_sampling,
            num_workers=num_workers,
            load_model_path=load_model_path,
            map_location=map_location,
            **kwargs,
        )
        self.hparams.optimizer_config_list = optimizer_config_list
        self.set_stage(stage)

    def set_stage(self, stage):
        """Set the model to the desired stage"""
        if stage in self.hparams.stage_list:
            self.stage = stage
        else:
            raise ValueError(
                f'{stage} is not a valid stage, please choose one of {self.hparams.stage_list}'
            )

    def _get_optimizer_parameters(
            self,
            opt_config,
            include_weight_decay: bool = True,
            **kwargs
    ) -> Dict[str, Union[torch.tensor, float]]:
        optimizer_parameters = [
            {
                'params': (
                    param for (name, param) in self.named_parameters()
                    if reduce(
                        lambda x, y: x or y,
                        [
                            name.startswith(prefix) for prefix in
                            opt_config['param_prefix_list']
                        ],
                        False,
                    )
                ),
                'lr': opt_config['lr'],
            }
        ]
        if include_weight_decay:
            weight_decay_dict = {'weight_decay': self.hparams.weight_decay}
            [d.update(weight_decay_dict) for d in optimizer_parameters]
        return optimizer_parameters

    def configure_optimizers(self) -> (
        Union[Tuple[List[Callable], List[Callable]], Tuple[Callable, Callable], Callable]
    ):
        """
        Configure optimizers and learning rate schedulers to use in optimization.

        This method will be called after `setup`.

        If `self.bias_optimizer` is None, only a single optimizer will be returned. If there is a
        non-None class attribute for `bias_optimizer`, two optimizers will be created: one for all
        layers with the name 'bias' in it, and another for all other model parameters. The bias
        optimizer will be set with the same parameters as `optimizer` with the exception of the
        learning rate, which will be set to `self.hparams.bias_lr`.

        """
        optimizer_config_list = [
            self._get_optimizer(self.optimizer, opt_config=opt_config)
            for opt_config in self.hparams.optimizer_config_list
        ]

        if self.lr_scheduler_func is not None:
            monitor = 'val_loss_epoch'
            if self.val_loader is None:
                monitor = 'train_loss_epoch'

            # add in optimizer to scheduler function
            scheduler_list = [
                {
                    'scheduler': self.lr_scheduler_func(opt_i),
                    'monitor': monitor,
                }
                for opt_i in optimizer_config_list
            ]

            return optimizer_config_list, scheduler_list

        else:
            return optimizer_config_list

    def optimizer_step(
        self, current_epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False
    ):
        """
        Overriding step function to only step the optimizer associated with the relevant stage.

        More details:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#optimizer-step
        """
        if self.hparams.optimizer_config_list[optimizer_idx]['stage'] == self.stage:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
