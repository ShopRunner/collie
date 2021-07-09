from collections import OrderedDict
from functools import partial
import multiprocessing
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

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
from collie_recs.model import MultiStagePipeline, ScaledEmbedding, ZeroEmbedding


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
        sparse: bool = False,
        item_metadata: Union[torch.tensor, pd.DataFrame, np.array] = None,
        metadata_layers_dims: Optional[List[int]] = None,
        combined_layers_dims: List[int] = [128, 64, 32],
        batch_size: int = 1024,
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
        stage: Optional[str] = 'matrix_factorization',
    ):
        item_metadata_num_cols = None
        if load_model_path is None:
            if item_metadata is None:
                raise ValueError('Must provide item metadata for `HybridModel`.')
            elif isinstance(item_metadata, pd.DataFrame):
                item_metadata = torch.tensor(item_metadata.to_numpy()).float()
            elif isinstance(item_metadata, np.ndarray):
                item_metadata = torch.tensor(item_metadata).float()
            else:
                item_metadata = item_metadata.float()

            item_metadata_num_cols = item_metadata.shape[1]

            optimizer_config_list = [
                {
                    'lr': embeddings_lr,
                    'optimizer': embeddings_optimizer,
                    'param_prefix_list': ['user_embed', 'item_embed'],
                    'stage': 'matrix_factorization',
                },
                {
                    'lr': bias_lr,
                    'optimizer': bias_optimizer,
                    'param_prefix_list': ['user_bias', 'item_bias'],
                    'stage': 'matrix_factorization',
                },
                {
                    'lr': metadata_only_stage_lr,
                    'optimizer': metadata_only_stage_optimizer,
                    'param_prefix_list': ['metadata'],
                    'stage': 'metadata_only',
                },
                {
                    'lr': all_stage_lr,
                    'optimizer': all_stage_optimizer,
                    'param_prefix_list': ['user', 'item', 'metadata', 'combined'],
                    'stage': 'all',
                },
            ]
        else:
            raise NotImplementedError('load saved model')

        super().__init__(
            train=train,
            val=val,
            optimizer_config_list=optimizer_config_list,
            stage='matrix_factorization',
            embedding_dim=embedding_dim,
            sparse=sparse,
            item_metadata=item_metadata,
            item_metadata_num_cols=item_metadata_num_cols,
            metadata_layers_dims=metadata_layers_dims,
            combined_layers_dims=combined_layers_dims,
            dropout_p=dropout_p,
            batch_size=batch_size,
            lr_scheduler_func=lr_scheduler_func,
            weight_decay=weight_decay,
            loss=loss,
            metadata_for_loss=metadata_for_loss,
            metadata_for_loss_weights=metadata_for_loss_weights,
            approximate_negative_sampling=approximate_negative_sampling,
            num_workers=num_workers,
            load_model_path=load_model_path,
            map_location=map_location,
        )

        self.item_metadata = item_metadata.to(self.device)

    def _load_model_init_helper(self, load_model_path: str, map_location: str) -> None:
        self.item_metadata = joblib.load(os.path.join(load_model_path, 'metadata.pkl'))
        super()._load_model_init_helper(load_model_path=os.path.join(load_model_path, 'model.pth'),
                                        map_location=map_location)

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
        if self.stage == 'matrix_factorization':
            pred_scores = (
                (
                    self.dropout(self.user_embeddings(users))
                    * self.dropout(self.item_embeddings(items))
                ).sum(1)
                + self.user_biases(users).squeeze(1)
                + self.item_biases(items).squeeze(1)
            )
        else:
            metadata_output = self.item_metadata[items, :].to(self.device)
            if self.metadata_layers is not None:
                for metadata_nn_layer in self.metadata_layers:
                    metadata_output = self.dropout(
                        F.leaky_relu(
                            metadata_nn_layer(metadata_output)
                        )
                    )

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

        if os.path.isfile(path):
            raise ValueError(f'`path` must be a directory path! {path} not valid.')
        elif os.path.exists(path):
            if os.listdir(path) and overwrite is False:
                raise ValueError(f'Data exists in `path` at {path} and `overwrite` is False.')
        elif os.path.isfile(path):
            print(f'The path {path} is for a file, not directory.')

        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.item_metadata, os.path.join(path, 'metadata.pkl'))

        # preserve ordering while extracting the state dictionary without the `_trained_model`
        # component
        state_dict_keys_to_save = [
            k for k, _ in self.state_dict().items() if '_trained_model' not in k
        ]
        state_dict_vals_to_save = [
            v for k, v in self.state_dict().items() if '_trained_model' not in k
        ]
        state_dict_to_save = OrderedDict(zip(state_dict_keys_to_save, state_dict_vals_to_save))

        dict_to_save = {'state_dict': state_dict_to_save, 'hparams': self.hparams}
        torch.save(dict_to_save, os.path.join(path, 'model.pth'))


class ColdStartModel(MultiStagePipeline):
    """TODO - Add docstring."""
    def __init__(
        self,
        train: INTERACTIONS_LIKE_INPUT = None,
        val: INTERACTIONS_LIKE_INPUT = None,
        item_buckets: torch.tensor = None,
        user_buckets: torch.tensor = None,
        both_buckets_lr,
        item_buckets_lr,
        no_buckets_lr,
        both_buckets_optimizer,
        item_buckets_optimizer,
        no_buckets_optimizer,
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
                    'lr': both_buckets_lr,
                    'optimizer': both_buckets_optimizer,
                    'param_prefix_list': [
                        'user_bucket_embed', 'user_bucket_bias',
                        'item_bucket_embed', 'item_bucket_bias',
                    ],
                    'stage': 'both_buckets',
                },
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

            self.stage = 'both_buckets'

        self.hparams.num_user_buckets = user_buckets.max().item() + 1
        self.hparams.num_item_buckets = item_buckets.max().item() + 1
        self.item_buckets = item_buckets.long()
        self.user_buckets = user_buckets.long()

        super().__init__(
            load_model_path=load_model_path,
            map_location=map_location,
            train=train,
            val=val,
            embedding_dim=embedding_dim,
            sparse=sparse,
            batch_size=batch_size,
            dropout_p=dropout_p,
            optimizer_config_list=optimizer_config_list,
            stage='both_buckets',  # initial stage
            lr_scheduler_func=lr_scheduler_func,
            weight_decay=weight_decay,
            optimizer=optimizer,
            loss=loss,
            approximate_negative_sampling=approximate_negative_sampling,
            num_workers=num_workers,
            metadata_for_loss=metadata_for_loss,
            metadata_for_loss_weights=metadata_for_loss_weights
        )

    def _copy_weights(self, old, new, buckets):
        new.weight.data.copy_(old.weight.data[buckets])

    def set_eval_stage(self, stage):
        print(f'set to stage {stage}')
        self.stage = stage

        # TODO: this method needs to get added to the interactions objects
        # the k9academy branch linked in the handoff doc has one way of doing this,
        # Nate's suggestion to do the bucketing on the fly is a better approach
        self.train_interactions.set_buckets(stage)
        if self.val_loader is not None:
            self.val_loader.set_buckets(stage)

    def set_stage(self, stage):
        if stage in self.hparams.stage_list:
            print(f'set to stage {stage}')
            self.stage = stage
            self.train_interactions.set_buckets(stage)
            # TODO: it would be awesome to disentangle val + training groups
            # since we're currently setting things for both training + eval
            if self.val_loader is not None:
                self.val_loader.set_buckets(stage)

            if stage == 'item_buckets':
                print('item embeddings initialized')
                self._copy_weights(self.user_bucket_bias, self.user_bias, self.user_buckets)
                self._copy_weights(self.user_bucket_embed, self.user_embed, self.user_buckets)
            elif stage == 'no_buckets':
                print('user embeddings initialized')
                self._copy_weights(self.item_bucket_bias, self.item_bias, self.item_buckets)
                self._copy_weights(self.item_bucket_embed, self.item_embed, self.item_buckets)

        else:
            raise ValueError(
                f'{stage} is not a valid stage, please choose one of {self.hparams.stage_list}'
            )

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
        user_embeddings = self.user_embed(users)
        user_biases = self.user_bias(users)
        item_embeddings = self.item_embed(items)
        item_biases = self.item_bias(items)

        if 'both' in self.stage:
            user_embeddings = self.user_bucket_embed(users)
            user_biases = self.user_bucket_bias(users)
            item_embeddings = self.item_bucket_embed(items)
            item_biases = self.item_bucket_bias(items)

        elif 'item' in self.stage:
            item_embeddings = self.item_bucket_embed(items)
            item_biases = self.item_bucket_bias(items)

        elif 'user' in self.stage:
            user_embeddings = self.user_bucket_embed(users)
            user_biases = self.user_bucket_bias(users)

        pred_scores = (
            (self.dropout(user_embeddings) * self.dropout(item_embeddings)).sum(1)
            + user_biases.squeeze(1)
            + item_biases.squeeze(1)
        )

        return pred_scores.squeeze()

    def _setup_model(self, **kwargs) -> None:
        """
        Method for building model internals that rely on the data passed in.

        This method will be called after `prepare_data`.

        """
        # define initial embedding groups
        self.user_bucket_bias = ZeroEmbedding(
            num_embeddings=self.hparams.num_user_buckets,
            embedding_dim=1,
            sparse=self.hparams.sparse
        )
        self.item_bucket_bias = ZeroEmbedding(
            num_embeddings=self.hparams.num_item_buckets,
            embedding_dim=1,
            sparse=self.hparams.sparse,
        )
        self.user_bucket_embed = ScaledEmbedding(
            num_embeddings=self.hparams.num_user_buckets,
            embedding_dim=self.hparams.embedding_dim,
            sparse=self.hparams.sparse
        )
        self.item_bucket_embed = ScaledEmbedding(
            num_embeddings=self.hparams.num_item_buckets,
            embedding_dim=self.hparams.embedding_dim,
            sparse=self.hparams.sparse,
        )

        # define fine-tuned embedding groups
        self.user_bias = ZeroEmbedding(
            num_embeddings=self.hparams.num_users,
            embedding_dim=1,
            sparse=self.hparams.sparse
        )
        self.item_bias = ZeroEmbedding(
            num_embeddings=self.hparams.num_items,
            embedding_dim=1,
            sparse=self.hparams.sparse,
        )
        self.user_embed = ScaledEmbedding(
            num_embeddings=self.hparams.num_users,
            embedding_dim=self.hparams.embedding_dim,
            sparse=self.hparams.sparse
        )
        self.item_embed = ScaledEmbedding(
            num_embeddings=self.hparams.num_items,
            embedding_dim=self.hparams.embedding_dim,
            sparse=self.hparams.sparse,
        )
        self.dropout = nn.Dropout(p=self.hparams.dropout_p)

    def _calculate_loss(
        self,
        batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.tensor]
    ) -> torch.tensor:
        ((users, pos_items), neg_items) = batch

        users = users.long()
        pos_items = pos_items.long()
        # TODO: see if there is a way to not have to transpose each time - probably a bit costly
        neg_items = torch.transpose(neg_items, 0, 1).long()

        # get positive item predictions from model
        pos_preds = self(users, pos_items)

        # get negative item predictions from model
        users_repeated = users.repeat(neg_items.shape[0])
        neg_items_flattened = neg_items.flatten()
        neg_preds = self(users_repeated, neg_items_flattened).view(
            neg_items.shape[0], len(users)
        )

        # implicit loss function expected here
        if 'bucket' in self.stage:
            loss = self.loss_function(
                pos_preds,
                neg_preds,
                num_items=self.hparams.num_items,
                positive_items=pos_items,
                negative_items=neg_items,
            )
        else:
            loss = self.loss_function(
                pos_preds,
                neg_preds,
                num_items=self.hparams.num_items,
                positive_items=pos_items,
                negative_items=neg_items,
                metadata=self.hparams.metadata_for_loss,
                metadata_weights=self.hparams.metadata_for_loss_weights,
            )

        if torch.isnan(loss).any():
            print(pos_preds)
            print(neg_preds)
            raise ValueError('nan in loss')

        return loss

    def _get_item_embeddings(self) -> np.array:
        """Get item embeddings."""
        # TODO: update this to get the embeddings post-MLP
        return self.item_embed(
            torch.arange(self.hparams.num_items, device=self.device)
        ).detach().cpu()
