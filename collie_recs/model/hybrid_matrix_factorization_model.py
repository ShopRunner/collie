from functools import partial
import multiprocessing
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collie_recs.config import DATA_PATH
from collie_recs.interactions import (ApproximateNegativeSamplingInteractionsDataLoader,
                                      Interactions,
                                      InteractionsDataLoader)
from collie_recs.model.base import MultiStagePipeline, ScaledEmbedding, ZeroEmbedding
from collie_recs.utils import get_init_arguments, merge_docstrings


INTERACTIONS_LIKE_INPUT = Union[ApproximateNegativeSamplingInteractionsDataLoader,
                                Interactions,
                                InteractionsDataLoader]


class HybridModel(MultiStagePipeline):
    """TODO - Add docstring."""
    def __init__(
        self,
        train: INTERACTIONS_LIKE_INPUT = None,
        val: INTERACTIONS_LIKE_INPUT = None,
        embedding_dim: int = 30,
        sparse: bool = False,  # TODO: remove?
        item_metadata: Union[torch.tensor, pd.DataFrame, np.array] = None,
        metadata_layers_dims: Optional[List[int]] = None,
        combined_layers_dims: List[int] = [128, 64, 32],
        dropout_p: float = 0.0,
        embeddings_lr: float = 1e-3,
        bias_lr: float = 1e-2,
        metadata_only_stage_lr: float = 1e-3,
        all_stage_lr: float = 1e-4,
        embeddings_optimizer: Union[str, Callable] = 'adam',
        bias_optimizer: Union[str, Callable] = 'sgd',
        metadata_only_stage_optimizer: Union[str, Callable] = 'adam',
        all_stage_optimizer: Union[str, Callable] = 'adam',
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
        load_model_path: Optional[str] = None,  # TODO: check model loading interaction w/ stages
        map_location: Optional[str] = None,
    ):
        item_metadata_num_cols = None
        optimizer_config_list = None

        if load_model_path is None:
            if item_metadata is None:
                raise ValueError('Must provide item metadata for ``HybridPretrainedModel``.')
            elif isinstance(item_metadata, pd.DataFrame):
                item_metadata = torch.from_numpy(item_metadata.to_numpy())
            elif isinstance(item_metadata, np.ndarray):
                item_metadata = torch.from_numpy(item_metadata)

            item_metadata = item_metadata.to(self.device).float()

            item_metadata_num_cols = item_metadata.shape[1]

            optimizer_config_list = [
                # TODO: allow those to be a single optimizer for both embeddings and bias terms
                {
                    'lr': embeddings_lr,
                    'optimizer': embeddings_optimizer,
                    # optimize embeddings...
                    'param_prefix_list': ['user_embed', 'item_embed'],
                    'stage': 'matrix_factorization',
                },
                {
                    'lr': bias_lr,
                    'optimizer': bias_optimizer,
                    # ... and optimize bias terms too
                    'param_prefix_list': ['user_bias', 'item_bias'],
                    'stage': 'matrix_factorization',
                },
                {
                    'lr': metadata_only_stage_lr,
                    'optimizer': metadata_only_stage_optimizer,
                    # optimize metadata layers only
                    'param_prefix_list': ['metadata', 'combined'],
                    'stage': 'metadata_only',
                },
                {
                    'lr': all_stage_lr,
                    'optimizer': all_stage_optimizer,
                    # optimize everything
                    'param_prefix_list': ['user', 'item', 'metadata', 'combined'],
                    'stage': 'all',
                },
            ]

        super().__init__(**get_init_arguments(),
                         optimizer_config_list=optimizer_config_list,
                         stage='matrix_factorization',
                         item_metadata_num_cols=item_metadata_num_cols)

    __doc__ = merge_docstrings(MultiStagePipeline, __doc__, __init__)

    def _load_model_init_helper(self, load_model_path: str, map_location: str, **kwargs) -> None:
        self.item_metadata = (
            joblib.load(os.path.join(load_model_path, 'metadata.pkl')).to(self.device)
        )
        super()._load_model_init_helper(load_model_path=os.path.join(load_model_path, 'model.pth'),
                                        map_location=map_location,
                                        **kwargs)
        self.hparams.stage = 'all'
        # TODO: print out current stage

    def _setup_model(self, **kwargs) -> None:
        """
        Method for building model internals that rely on the data passed in.

        This method will be called after `prepare_data`.

        """
        if self.hparams.load_model_path is None:
            if not hasattr(self, 'item_metadata'):
                self.item_metadata = kwargs.pop('item_metadata')

        self.user_biases = ZeroEmbedding(num_embeddings=self.hparams.num_users,
                                         embedding_dim=1,
                                         sparse=self.hparams.sparse)
        self.item_biases = ZeroEmbedding(num_embeddings=self.hparams.num_items,
                                         embedding_dim=1,
                                         sparse=self.hparams.sparse)
        self.user_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_users,
                                               embedding_dim=self.hparams.embedding_dim,
                                               sparse=self.hparams.sparse)
        self.item_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_items,
                                               embedding_dim=self.hparams.embedding_dim,
                                               sparse=self.hparams.sparse)
        self.dropout = nn.Dropout(p=self.hparams.dropout_p)

        # set up metadata-only layers
        metadata_output_dim = self.hparams.item_metadata_num_cols
        self.metadata_layers = None
        if self.hparams.metadata_layers_dims is not None:
            metadata_layers_dims = (
                [self.hparams.item_metadata_num_cols] + self.hparams.metadata_layers_dims
            )
            self.metadata_layers = [
                nn.Linear(metadata_layers_dims[idx - 1], metadata_layers_dims[idx])
                for idx in range(1, len(metadata_layers_dims))
            ]
            for i, layer in enumerate(self.metadata_layers):
                nn.init.xavier_normal_(self.metadata_layers[i].weight)
                self.add_module('metadata_layer_{}'.format(i), layer)

            metadata_output_dim = metadata_layers_dims[-1]

        # set up combined layers
        combined_dimension_input = (
            self.user_embeddings.embedding_dim
            + self.item_embeddings.embedding_dim
            + metadata_output_dim
        )
        combined_layers_dims = [combined_dimension_input] + self.hparams.combined_layers_dims + [1]
        self.combined_layers = [
            nn.Linear(combined_layers_dims[idx - 1], combined_layers_dims[idx])
            for idx in range(1, len(combined_layers_dims))
        ]
        for i, layer in enumerate(self.combined_layers):
            nn.init.xavier_normal_(self.combined_layers[i].weight)
            self.add_module('combined_layer_{}'.format(i), layer)

    def forward(self, users: torch.tensor, items: torch.tensor) -> torch.tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        users: tensor, 1-d
            Array of user indices
        items: tensor, 1-d
            Array of item indices

        Retuprns
        ----------
        preds: tensor, 1-d
            Predicted ratings or rankings

        """
        if self.hparams.stage == 'matrix_factorization':
            pred_scores = (
                torch.mul(
                    self.dropout(self.user_embeddings(users)),
                    self.dropout(self.item_embeddings(items))
                ).sum(axis=1)
                + self.user_biases(users).squeeze(1)
                + self.item_biases(items).squeeze(1)
            )
        else:
            # TODO: remove self.device and let lightning do it
            metadata_output = self.item_metadata[items, :].to(self.device)
            if self.metadata_layers is not None:
                for metadata_nn_layer in self.metadata_layers:
                    metadata_output = self.dropout(
                        F.leaky_relu(
                            metadata_nn_layer(metadata_output)
                        )
                    )

            # TODO: make this matrix factorization instead of only a MLP
            combined_output = torch.cat((self.user_embeddings(users),
                                         self.item_embeddings(items),
                                         metadata_output), 1)
            for combined_nn_layer in self.combined_layers[:-1]:
                combined_output = self.dropout(
                    F.leaky_relu(
                        combined_nn_layer(combined_output)
                    )
                )

            pred_scores = self.combined_layers[-1](combined_output)

        return pred_scores.squeeze()

    def _get_item_embeddings(self) -> np.array:
        """Get item embeddings."""
        # TODO: update this to get the embeddings post-MLP
        return self.item_embeddings(
            torch.arange(self.hparams.num_items, device=self.device)
        ).detach().cpu()

    def save_model(self,
                   path: Union[str, Path] = os.path.join(DATA_PATH / 'model'),
                   overwrite: bool = False) -> None:
        """
        Save the model's state dictionary, hyperparameters, and item metadata.

        While PyTorch Lightning offers a way to save and load models, there are two main reasons
        for overriding these:
        1) To properly save and load a model requires the `Trainer` object, meaning that all
           deployed models will require on Lightning to run the model, which is not needed.
        2) In the v0.8.4 release, loading a model back in leads to a `RuntimeError` unable to load
           in weights.

        Parameters
        ----------
        path: str or Path
            Directory path to save model and data files
        overwrite: bool
            Whether or not to overwrite existing data

        """
        path = str(path)

        if os.path.exists(path):
            if os.listdir(path) and overwrite is False:
                raise ValueError(f'Data exists in ``path`` at {path} and ``overwrite`` is False.')

        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.item_metadata, os.path.join(path, 'metadata.pkl'))

        super().save_model(filename=os.path.join(path, 'model.pth'))
