from functools import partial
import multiprocessing
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collie_recs.interactions import (ApproximateNegativeSamplingInteractionsDataLoader,
                                      Interactions,
                                      InteractionsDataLoader)
from collie_recs.model import MultiStagePipeline, ScaledEmbedding, ZeroEmbedding
from collie_recs.utils import get_init_arguments, merge_docstrings


INTERACTIONS_LIKE_INPUT = Union[ApproximateNegativeSamplingInteractionsDataLoader,
                                Interactions,
                                InteractionsDataLoader]


class ColdStartModel(MultiStagePipeline):
    """TODO - Add docstring."""
    # TODO: set up advance stage to be correct
    def __init__(
        self,
        train: INTERACTIONS_LIKE_INPUT = None,
        val: INTERACTIONS_LIKE_INPUT = None,
        item_buckets: torch.tensor = None,
        item_buckets_lr: float = 1e-3,  # TODO: standardize LR naming schemes
        no_buckets_lr: float = 1e-3,
        item_buckets_optimizer: Union[str, Callable] = 'adam',
        no_buckets_optimizer: Union[str, Callable] = 'adam',
        # old arguments
        embedding_dim: int = 30,
        sparse: bool = False,
        batch_size: int = 1024,
        dropout_p: float = 0.0,
        lr_scheduler_func: Optional[Callable] = partial(
            ReduceLROnPlateau,
            patience=1,
            verbose=True,
        ),
        weight_decay: float = 0.0,
        loss: Union[str, Callable] = 'hinge',
        metadata_for_loss: Optional[Dict[str, torch.tensor]] = None,
        metadata_for_loss_weights: Optional[Dict[str, float]] = None,
        approximate_negative_sampling: bool = False,
        num_workers: int = multiprocessing.cpu_count(),
        load_model_path: Optional[str] = None,
        map_location: Optional[str] = None,
    ):
        if load_model_path is None:
            optimizer_config_list = [
                {
                    'lr': item_buckets_lr,
                    'optimizer': item_buckets_optimizer,
                    'param_prefix_list': [
                        'user_embed', 'user_bias',
                        'item_bucket_embed', 'item_bucket_bias',
                    ],
                    'stage': 'item_buckets',
                },
                {
                    'lr': no_buckets_lr,
                    'optimizer': no_buckets_optimizer,
                    'param_prefix_list': [
                        'user_embed', 'user_bias',
                        'item_embed', 'item_bias',
                    ],
                    'stage': 'no_buckets',
                },
            ]

            self.stage = 'item_buckets'

        num_item_buckets = item_buckets.max().item() + 1

        super().__init__(**get_init_arguments(),
                         optimizer_config_list=optimizer_config_list,
                         stage='item_buckets',  # initial stage
                         num_item_buckets=num_item_buckets)

    __doc__ = merge_docstrings(MultiStagePipeline, __doc__, __init__)

    def _copy_weights(self, old, new, buckets):
        new.weight.data.copy_(old.weight.data[buckets])

    def set_stage(self, stage):
        """TODO."""
        current_stage = self.hparams.stage

        if stage in self.hparams.stage_list:
            if current_stage == 'item_buckets' and stage == 'no_buckets':
                print('item embeddings initialized')
                self._copy_weights(self.item_bucket_biases,
                                   self.item_biases,
                                   self.hparams.item_buckets)
                self._copy_weights(self.item_bucket_embeddings,
                                   self.item_embeddings,
                                   self.hparams.item_buckets)
        else:
            raise ValueError(
                f'{stage} is not a valid stage, please choose one of {self.hparams.stage_list}'
            )

        self.hparams.stage = stage
        print(f'set to stage {stage}')

    def _setup_model(self, **kwargs) -> None:
        """
        Method for building model internals that rely on the data passed in.

        This method will be called after `prepare_data`.

        """
        # define initial embedding groups
        self.item_bucket_biases = ZeroEmbedding(
            num_embeddings=self.hparams.num_item_buckets,
            embedding_dim=1,
            sparse=self.hparams.sparse,
        )
        self.item_bucket_embeddings = ScaledEmbedding(
            num_embeddings=self.hparams.num_item_buckets,
            embedding_dim=self.hparams.embedding_dim,
            sparse=self.hparams.sparse,
        )

        # define fine-tuned embedding groups
        self.user_biases = ZeroEmbedding(
            num_embeddings=self.hparams.num_users,
            embedding_dim=1,
            sparse=self.hparams.sparse
        )
        self.item_biases = ZeroEmbedding(
            num_embeddings=self.hparams.num_items,
            embedding_dim=1,
            sparse=self.hparams.sparse,
        )
        self.user_embeddings = ScaledEmbedding(
            num_embeddings=self.hparams.num_users,
            embedding_dim=self.hparams.embedding_dim,
            sparse=self.hparams.sparse
        )
        self.item_embeddings = ScaledEmbedding(
            num_embeddings=self.hparams.num_items,
            embedding_dim=self.hparams.embedding_dim,
            sparse=self.hparams.sparse,
        )

        self.dropout = nn.Dropout(p=self.hparams.dropout_p)

    def forward(self, users: torch.tensor, items: torch.tensor) -> torch.tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        users: tensor, 1-d
            Array of user indices
        items: tensor, 1-d
            Array of item indices

        Returns
        ----------
        preds: tensor, 1-d
            Predicted ratings or rankings

        """
        user_embeddings = self.user_embeddings(users)
        user_biases = self.user_biases(users)

        if self.hparams.stage == 'item_buckets':
            # transform item IDs to item bucket IDs
            items = self.hparams.item_buckets[items]

            item_embeddings = self.item_bucket_embeddings(items)
            item_biases = self.item_bucket_biases(items)
        elif self.hparams.stage == 'no_buckets':
            item_embeddings = self.item_embeddings(items)
            item_biases = self.item_biases(items)

        pred_scores = (
            torch.mul(self.dropout(user_embeddings), self.dropout(item_embeddings)).sum(axis=1)
            + user_biases.squeeze(1)
            + item_biases.squeeze(1)
        )

        return pred_scores.squeeze()

    def _get_item_embeddings(self) -> np.array:
        """Get item embeddings."""
        # TODO: update this to get the embeddings post-MLP
        return self.item_embeddings(
            torch.arange(self.hparams.num_items, device=self.device)
        ).detach().cpu()
