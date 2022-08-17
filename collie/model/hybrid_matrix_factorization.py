from functools import partial
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collie.config import DATA_PATH
from collie.interactions import (ApproximateNegativeSamplingInteractionsDataLoader,
                                 Interactions,
                                 InteractionsDataLoader)
from collie.model.base import MultiStagePipeline, ScaledEmbedding, ZeroEmbedding
from collie.utils import get_init_arguments, merge_docstrings


INTERACTIONS_LIKE_INPUT = Union[ApproximateNegativeSamplingInteractionsDataLoader,
                                Interactions,
                                InteractionsDataLoader]


class HybridModel(MultiStagePipeline):
    # NOTE: the full docstring is merged in with ``MultiStagePipeline``'s using
    # ``merge_docstrings``. Only the description of new or changed parameters are included in this
    # docstring
    """
    Training pipeline for a multi-stage hybrid recommendation model.

    ``HybridModel`` models contain dense layers that process item and/or user metadata, concatenate
    this embedding with user and item embeddings, sending this concatenated embedding through more
    dense layers to output a single float ranking / rating. We add both user and item biases to
    this score before returning. This is the same architecture as the ``HybridPretrainedModel``,
    but we are training the embeddings ourselves rather than relying on pulling this from a
    pre-trained model.

    The stages in a ``HybridModel`` depend on whether both item and user metadata is used.
    For the full model, they are, in order:

    1. ``matrix_factorization``
        Matrix factorization exactly as we do in ``MatrixFactorizationModel``. In this stage,
        metadata is NOT incorporated into the model.
    2. ``metadata_only``
        User and item embeddings terms are frozen, and the MLP layers for the metadata (if
        specified) and combined embedding-metadata data are optimized.
    3. ``all``
        Embedding and MLP layers are all optimized together, including those for metadata.

    All ``HybridModel`` instances are subclasses of the ``LightningModule`` class provided by
    PyTorch Lightning. This means to train a model, you will need a
    ``collie.model.CollieTrainer`` object, but the model can be saved and loaded without this
    ``Trainer`` instance. Example usage may look like:

    .. code-block:: python

        from collie.model import CollieTrainer, HybridModel


        # instantiate and fit a ``HybridModel`` as expected
        model = HybridModel(train=train,
                            item_metadata=item_metadata,
                            user_metadata=user_metadata)
        trainer = CollieTrainer(model)
        trainer.fit(model)

        # train for X more epochs on the next stage, ``metadata_only``
        trainer.max_epochs += X
        model.advance_stage()
        trainer.fit(model)

        # train for Y more epochs on the next stage, ``all``
        trainer.max_epochs += Y
        model.advance_stage()
        trainer.fit(model)

        model.eval()

        # do evaluation as normal with ``model``

        model.save_model(path='model')
        new_model = HybridModel(load_model_path='model')

        # do evaluation as normal with ``new_model``

    Note
    ----
    The ``forward`` calculation will be different depending on the stage that is set. Note this
    when evaluating / saving and loading models in.

    Parameters
    ----------
    item_metadata: torch.tensor, pd.DataFrame, or np.array, 2-dimensional
        The shape of the item metadata should be (num_items x metadata_features), and each item's
        metadata should be available when indexing a row by an item ID
    user_metadata: torch.tensor, pd.DataFrame, or np.array, 2-dimensional
        The shape of the user metadata should be (num_users x metadata_features), and each user's
        metadata should be available when indexing a row by a user ID
    embedding_dim: int
        Number of latent factors to use for user and item embeddings
    item_metadata_layers_dims: list
        List of linear layer dimensions to apply to the item metadata only, starting with
        the dimension directly following ``item_metadata_features`` and ending with the
        dimension to concatenate with the item embeddings
    user_metadata_layers_dims: list
        List of linear layer dimensions to apply to the user metadata only, starting with
        the dimension directly following ``user_metadata_features`` and ending with the
        dimension to concatenate with the user embeddings
    combined_layers_dims: list
        List of linear layer dimensions to apply to the concatenated item embeddings and item
        metadata, starting with the dimension directly following the shape of
        ``item_embeddings + metadata_features`` and ending with the dimension before the final
        linear layer to dimension 1
    dropout_p: float
        Probability of dropout
    metadata_only_stage_lr: float
        Learning rate for metadata and combined layers optimized during the ``metadata_only`` stage
    all_stage_lr: float
        Learning rate for all model parameters optimized during the ``all`` stage
    optimizer: torch.optim or str
        Optimizer used for embeddings and bias terms (if ``bias_optimizer`` is ``None``) during the
        ``matrix_factorization`` stage. If a string, one of the following supported optimizers:

        * ``'sgd'`` (for ``torch.optim.SGD``)

        * ``'adam'`` (for ``torch.optim.Adam``)

    metadata_only_stage_optimizer: torch.optim or str
        Optimizer used for metadata and combined layers during the ``metadata_only`` stage. If a
        string, one of the following supported optimizers:

        * ``'sgd'`` (for ``torch.optim.SGD``)

        * ``'adam'`` (for ``torch.optim.Adam``)

    all_stage_optimizer: torch.optim or str
        Optimizer used for all model parameters during the ``all`` stage. If a string, one of the
        following supported optimizers:

        * ``'sgd'`` (for ``torch.optim.SGD``)

        * ``'adam'`` (for ``torch.optim.Adam``)

    """
    def __init__(self,
                 train: INTERACTIONS_LIKE_INPUT = None,
                 val: INTERACTIONS_LIKE_INPUT = None,
                 item_metadata: Union[torch.tensor, pd.DataFrame, np.array] = None,
                 user_metadata: Union[torch.tensor, pd.DataFrame, np.array] = None,
                 embedding_dim: int = 30,
                 item_metadata_layers_dims: Optional[List[int]] = None,
                 user_metadata_layers_dims: Optional[List[int]] = None,
                 combined_layers_dims: List[int] = [128, 64, 32],
                 dropout_p: float = 0.0,
                 lr: float = 1e-3,
                 bias_lr: Optional[Union[float, str]] = 1e-2,
                 metadata_only_stage_lr: float = 1e-3,
                 all_stage_lr: float = 1e-4,
                 lr_scheduler_func: Optional[torch.optim.lr_scheduler._LRScheduler] = partial(
                     ReduceLROnPlateau,
                     patience=1,
                     verbose=False,
                 ),
                 weight_decay: float = 0.0,
                 optimizer: Union[str, torch.optim.Optimizer] = 'adam',
                 bias_optimizer: Optional[Union[str, torch.optim.Optimizer]] = 'sgd',
                 metadata_only_stage_optimizer: Union[str, torch.optim.Optimizer] = 'adam',
                 all_stage_optimizer: Union[str, torch.optim.Optimizer] = 'adam',
                 loss: Union[str, Callable[..., torch.tensor]] = 'hinge',
                 metadata_for_loss: Optional[Dict[str, torch.tensor]] = None,
                 metadata_for_loss_weights: Optional[Dict[str, float]] = None,
                 load_model_path: Optional[str] = None,
                 map_location: Optional[str] = None):
        item_metadata_num_cols = None
        user_metadata_num_cols = None
        optimizer_config_list = None

        if load_model_path is None:
            if item_metadata is None and user_metadata is None:
                raise ValueError(
                    'Must provide item metadata and/or user metadata for ``HybridModel``.'
                )

            if item_metadata is not None:
                if isinstance(item_metadata, pd.DataFrame):
                    item_metadata = torch.from_numpy(item_metadata.to_numpy())
                elif isinstance(item_metadata, np.ndarray):
                    item_metadata = torch.from_numpy(item_metadata)
                item_metadata = item_metadata.float()
                item_metadata_num_cols = item_metadata.shape[1]

            if user_metadata is not None:
                if isinstance(user_metadata, pd.DataFrame):
                    user_metadata = torch.from_numpy(user_metadata.to_numpy())
                elif isinstance(user_metadata, np.ndarray):
                    user_metadata = torch.from_numpy(user_metadata)
                user_metadata = user_metadata.float()
                user_metadata_num_cols = user_metadata.shape[1]

            if bias_optimizer is not None:
                initial_optimizer_block = [
                    {
                        'lr': lr,
                        'optimizer': optimizer,
                        # optimize embeddings...
                        'parameter_prefix_list': ['user_embedding', 'item_embedding'],
                        'stage': 'matrix_factorization',
                    },
                    {
                        'lr': lr if bias_lr == 'infer' else bias_lr,
                        'optimizer': optimizer if bias_optimizer == 'infer' else bias_optimizer,
                        # ... and optimize bias terms too
                        'parameter_prefix_list': ['user_bias', 'item_bias'],
                        'stage': 'matrix_factorization',
                    },
                ]
            else:
                initial_optimizer_block = [
                    {
                        'lr': lr,
                        'optimizer': optimizer,
                        # optimize embeddings and bias terms all together
                        'parameter_prefix_list': [
                            'user_embedding',
                            'item_embedding',
                            'user_bias',
                            'item_bias'],
                        'stage': 'matrix_factorization',
                    },
                ]

            optimizer_config_list = initial_optimizer_block + [
                {
                    'lr': metadata_only_stage_lr,
                    'optimizer': metadata_only_stage_optimizer,
                    # optimize metadata layers only
                    'parameter_prefix_list': [
                        'item_metadata', 'user_metadata', 'combined', 'user_bias', 'item_bias'
                    ],
                    'stage': 'metadata_only',
                },
                {
                    'lr': all_stage_lr,
                    'optimizer': all_stage_optimizer,
                    # optimize everything
                    'parameter_prefix_list': [
                        'user', 'item', 'combined'
                    ],
                    'stage': 'all',
                },
            ]

        super().__init__(optimizer_config_list=optimizer_config_list,
                         item_metadata_num_cols=item_metadata_num_cols,
                         user_metadata_num_cols=user_metadata_num_cols,
                         **get_init_arguments())

    __doc__ = merge_docstrings(MultiStagePipeline, __doc__, __init__)

    def _move_any_external_data_to_device(self):
        """Move item and user metadata to the device before training."""
        if self.item_metadata is not None:
            self.item_metadata = self.item_metadata.to(self.device)
        if self.user_metadata is not None:
            self.user_metadata = self.user_metadata.to(self.device)

    def _load_model_init_helper(self, load_model_path: str, map_location: str, **kwargs) -> None:
        try:
            self.item_metadata = (
                joblib.load(os.path.join(load_model_path, 'item_metadata.pkl'))
            )
        except FileNotFoundError:
            warnings.warn('``item_metadata.pkl`` not found')

        try:
            self.user_metadata = (
                joblib.load(os.path.join(load_model_path, 'user_metadata.pkl'))
            )
        except FileNotFoundError:
            warnings.warn('``user_metadata.pkl`` not found')

        super()._load_model_init_helper(load_model_path=os.path.join(load_model_path, 'model.pth'),
                                        map_location=map_location,
                                        **kwargs)

    def _configure_metadata_layers(
        self,
        metadata_type: str,
        metadata_layers_dims: Optional[Iterable[int]],
        num_metadata_cols: Optional[int],
    ) -> None:
        """
        Configure metadata layers for either item or user data.

        Parameters
        ----------
        metadata_type: str
            Metadata type, one of ``user`` or ``item``. It is used to set
            the attributes ``{metadata_type}_metadata_layers`` and
            ``{metadata_type}_metadata_layers_dims``
        metadata_layers_dims: list
            List of dimensions for the hidden state of the metadata layers
        num_metadata_cols: int
            Number of columns in the metadata dataset

        """
        if metadata_layers_dims is not None:
            full_metadata_layers_dims = (
                [num_metadata_cols] + metadata_layers_dims
            )

            full_metadata_layers = [
                nn.Linear(full_metadata_layers_dims[idx - 1], full_metadata_layers_dims[idx])
                for idx in range(1, len(full_metadata_layers_dims))
            ]

            setattr(self, f'{metadata_type}_metadata_layers', full_metadata_layers)

            for i, layer in enumerate(getattr(self, f'{metadata_type}_metadata_layers')):
                nn.init.xavier_normal_(
                    getattr(self, f'{metadata_type}_metadata_layers')[i].weight
                )
                self.add_module(f'{metadata_type}_metadata_layer_{i}', layer)

            setattr(self,
                    f'{metadata_type}_metadata_layers_dims',
                    full_metadata_layers_dims)

    def _setup_model(self, **kwargs) -> None:
        """
        Method for building model internals that rely on the data passed in.

        This method will be called after `prepare_data`.

        """
        if self.hparams.load_model_path is None:
            if 'item_metadata' in kwargs:
                self.item_metadata = kwargs.pop('item_metadata')
            if 'user_metadata' in kwargs:
                self.user_metadata = kwargs.pop('user_metadata')

        self.user_biases = ZeroEmbedding(num_embeddings=self.hparams.num_users,
                                         embedding_dim=1)
        self.item_biases = ZeroEmbedding(num_embeddings=self.hparams.num_items,
                                         embedding_dim=1)
        self.user_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_users,
                                               embedding_dim=self.hparams.embedding_dim)
        self.item_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_items,
                                               embedding_dim=self.hparams.embedding_dim)
        self.dropout = nn.Dropout(p=self.hparams.dropout_p)

        # set up item metadata-only layers
        item_metadata_output_dim = self.hparams.item_metadata_num_cols
        self.item_metadata_layers = None
        if self.hparams.item_metadata_layers_dims is not None:
            self._configure_metadata_layers(
                metadata_type='item',
                metadata_layers_dims=self.hparams.item_metadata_layers_dims,
                num_metadata_cols=self.hparams.item_metadata_num_cols,
            )
            item_metadata_output_dim = self.item_metadata_layers_dims[-1]

        # set up user metadata-only layers
        user_metadata_output_dim = self.hparams.user_metadata_num_cols
        self.user_metadata_layers = None
        if self.hparams.user_metadata_layers_dims is not None:
            self._configure_metadata_layers(
                metadata_type='user',
                metadata_layers_dims=self.hparams.user_metadata_layers_dims,
                num_metadata_cols=self.hparams.user_metadata_num_cols,
            )
            user_metadata_output_dim = self.user_metadata_layers_dims[-1]

        # set up combined layers depending on metadata inputs
        if item_metadata_output_dim is not None and user_metadata_output_dim is not None:
            combined_dimension_input = (
                user_metadata_output_dim
                + self.user_embeddings.embedding_dim
                + self.item_embeddings.embedding_dim
                + item_metadata_output_dim
            )
        elif item_metadata_output_dim is not None:
            combined_dimension_input = (
                self.user_embeddings.embedding_dim
                + self.item_embeddings.embedding_dim
                + item_metadata_output_dim
            )
        elif user_metadata_output_dim is not None:
            combined_dimension_input = (
                user_metadata_output_dim
                + self.user_embeddings.embedding_dim
                + self.item_embeddings.embedding_dim
            )
        combined_layers_dims = [combined_dimension_input] + self.hparams.combined_layers_dims + [1]
        self.combined_layers = [
            nn.Linear(combined_layers_dims[idx - 1], combined_layers_dims[idx])
            for idx in range(1, len(combined_layers_dims))
        ]
        for i, layer in enumerate(self.combined_layers):
            nn.init.xavier_normal_(self.combined_layers[i].weight)
            self.add_module('combined_layer_{}'.format(i), layer)

    def _compute_metadata_output(
        self,
        metadata_type: str,
        ids: torch.tensor,
    ) -> torch.tensor:
        """
        Calculate metadata output for either item or user data.

        Parameters
        ----------
        metadata_type: str
            Metadata type, one of ``user`` or ``item``
        ids: tensor, 1-d
            Array of user indices or item indices

        Returns
        -------
        metadata_output: tensor, 1-d
            Metadata output
        """
        # TODO: remove self.device and let lightning do it

        metadata = getattr(self, f'{metadata_type}_metadata')
        metadata_layers = getattr(self, f'{metadata_type}_metadata_layers')
        metadata_output = metadata[ids, :].to(self.device)

        if metadata_layers is not None:
            for metadata_nn_layer in metadata_layers:
                metadata_output = self.dropout(
                    F.leaky_relu(
                        metadata_nn_layer(metadata_output)
                    )
                )
        return metadata_output

    def _compute_prediction(
        self,
        combined_output: torch.tensor,
        users: torch.tensor,
        items: torch.tensor,
    ) -> torch.tensor:
        """
        Calculate prediction output

        Parameters
        ----------
        combined_output: tensor, 2-d
            Array of user and item embeddings concatenated with item and/or user metadata
        users: tensor, 1-d
            Array of user indices
        items: tensor, 1-d
            Array of item indices

        Returns
        -------
        pred_scores: tensor, 2-d
            Predicted scores
        """

        for combined_nn_layer in self.combined_layers[:-1]:
            combined_output = self.dropout(
                F.leaky_relu(
                    combined_nn_layer(combined_output)
                )
            )

        pred_scores = (
            self.combined_layers[-1](combined_output)
            + self.user_biases(users)
            + self.item_biases(items)
        )
        return pred_scores

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
        -------
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
        elif self.hparams.stage in ('metadata_only', 'all') and self.user_metadata is None:
            item_metadata_output = self._compute_metadata_output(
                metadata_type='item',
                ids=items
            )

            # TODO: make this matrix factorization instead of only a MLP
            combined_output = torch.cat((self.user_embeddings(users),
                                         self.item_embeddings(items),
                                         item_metadata_output), 1)
            pred_scores = self._compute_prediction(combined_output, users, items)

        elif self.hparams.stage in ('metadata_only', 'all') and self.item_metadata is None:
            user_metadata_output = self._compute_metadata_output(
                metadata_type='user',
                ids=users
            )

            # TODO: make this matrix factorization instead of only a MLP
            combined_output = torch.cat((user_metadata_output,
                                         self.user_embeddings(users),
                                         self.item_embeddings(items)), 1)
            pred_scores = self._compute_prediction(combined_output, users, items)

        else:
            user_metadata_output = self._compute_metadata_output(
                metadata_type='user',
                ids=users
            )
            item_metadata_output = self._compute_metadata_output(
                metadata_type='item',
                ids=items
            )

            # TODO: make this matrix factorization instead of only a MLP
            combined_output = torch.cat((user_metadata_output,
                                         self.user_embeddings(users),
                                         self.item_embeddings(items),
                                         item_metadata_output), 1)
            pred_scores = self._compute_prediction(combined_output, users, items)

        return pred_scores.squeeze()

    def _get_item_embeddings(self) -> torch.tensor:
        """Get item embeddings on device."""
        # TODO: update this to get the embeddings post-MLP
        return self.item_embeddings.weight.data

    def _get_user_embeddings(self) -> torch.tensor:
        """Get user embeddings on device."""
        # TODO: update this to get the embeddings post-MLP
        return self.user_embeddings.weight.data

    def save_model(self,
                   path: Union[str, Path] = os.path.join(DATA_PATH / 'model'),
                   overwrite: bool = False) -> None:
        """
        Save the model's state dictionary, hyperparameters, and user and/or item metadata.

        While PyTorch Lightning offers a way to save and load models, there are two main reasons
        for overriding these:

        1) To properly save and load a model requires the ``Trainer`` object, meaning that all
           deployed models will require Lightning to run the model, which is not actually needed
           for inference.

        2) In the v0.8.4 release, loading a model back in leads to a ``RuntimeError`` unable to load
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
        if self.item_metadata is not None:
            joblib.dump(self.item_metadata, os.path.join(path, 'item_metadata.pkl'))

        if self.user_metadata is not None:
            joblib.dump(self.user_metadata, os.path.join(path, 'user_metadata.pkl'))

        super().save_model(filename=os.path.join(path, 'model.pth'))
