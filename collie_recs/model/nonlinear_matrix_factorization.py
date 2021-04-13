from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collie_recs.model.base import (BasePipeline,
                                    interactions_like_input,
                                    ScaledEmbedding,
                                    ZeroEmbedding)
from collie_recs.utils import get_init_arguments


class NonlinearMatrixFactorizationModel(BasePipeline):
    """
    Training pipeline for a nonlinear matrix factorization model.

    ``NonlinearMatrixFactorizationModel`` models have an embedding layer for users and items. These
    are sent through separate dense networks, which output more refined embeddings, which are then
    dot producted for a single float ranking / rating.

    Collie adds a twist on to this novel framework by allowing separate optimizers for embeddings
    and bias terms. With larger datasets and multiple epochs of training, a model might incorrectly
    learn to only optimize the bias terms for a quicker path towards a local loss minimum,
    essentially memorizing how popular each item is. By using a separate, slower optimizer for the
    bias terms (like Stochastic Gradient Descent), the model must prioritize optimizing the
    embeddings for meaningful, more varied recommendations, leading to a model that is able to
    achieve a much lower loss. See the documentation below for ``bias_lr`` and ``bias_optimizer``
    input arguments for implementation details.

    All ``NonlinearMatrixFactorizationModel`` instances are subclasses of the ``LightningModule``
    class provided by PyTorch Lightning. This means to train a model, you will need a
    ``collie_recs.model.CollieTrainer`` object, but the model can be saved and loaded without this
    ``Trainer`` instance. Example usage may look like:

    .. code-block:: python

        from collie_recs.model import CollieTrainer, NonlinearMatrixFactorizationModel


        model = NonlinearMatrixFactorizationModel(train=train)
        trainer = CollieTrainer(model)
        trainer.fit(model)
        model.freeze()

        # do evaluation as normal with ``model``

        model.save_model(filename='model.pth')
        new_model = NonlinearMatrixFactorizationModel(load_model_path='model.pth')

        # do evaluation as normal with ``new_model``

    Parameters
    ----------
    train: ``collie_recs.interactions`` object
        Data loader for training data. If an ``Interactions`` object is supplied, an
        ``InteractionsDataLoader`` will automatically be instantiated with ``shuffle=True``. Note
        that when the model class is saved, datasets will NOT be saved as well
    val: ``collie_recs.interactions`` object
        Data loader for validation data. If an ``Interactions`` object is supplied, an
        ``InteractionsDataLoader`` will automatically be instantiated with ``shuffle=False``. Note
        that when the model class is saved, datasets will NOT be saved as well
    user_embedding_dim: int
        Number of latent factors to use for user embeddings
    item_embedding_dim: int
        Number of latent factors to use for item embeddings
    user_dense_layers_dims: list
        List of linear layer dimensions to apply to the user embedding, starting with the dimension
        directly following ``user_embedding_dim``
    item_dense_layers_dims: list
        List of linear layer dimensions to apply to the item embedding, starting with the dimension
        directly following ``item_embedding_dim``
    embedding_dropout_p: float
        Probability of dropout on the embedding layers
    dense_dropout_p: float
        Probability of dropout on the dense layers
    lr: float
        Embedding layer learning rate
    bias_lr: float
        Bias terms learning rate. If 'infer', will set equal to ``lr``
    lr_scheduler_func: torch.optim.lr_scheduler
        Learning rate scheduler to use during fitting
    weight_decay: float
        Weight decay passed to the optimizer, if optimizer permits
    optimizer: torch.optim or str
        If a string, one of the following supported optimizers:

        * ``'sgd'`` (for ``torch.optim.SGD``)

        * ``'adam'`` (for ``torch.optim.Adam``)

    bias_optimizer: torch.optim or str
        Optimizer for the bias terms. This supports the same string options as ``optimizer``, with
        the addition of ``infer``, which will set the optimizer equal to ``optimizer``. If
        ``bias_optimizer`` is ``None``, only a single optimizer will be created for all model
        parameters
    loss: function or str
        If a string, one of the following implemented losses:

        * ``'bpr'`` / ``'adaptive_bpr'``

        * ``'hinge'`` / ``'adaptive_hinge'``

        * ``'warp'``

        If ``train.num_negative_samples > 1``, the adaptive loss version will automatically be used
    metadata_for_loss: dict
        Keys should be strings identifying each metadata type that match keys in
        ``metadata_weights``. Values should be a ``torch.tensor`` of shape (num_items x 1). Each
        tensor should contain categorical metadata information about items (e.g. a number
        representing the genre of the item)
    metadata_for_loss_weights: dict
        Keys should be strings identifying each metadata type that match keys in ``metadata``.
        Values should be the amount of weight to place on a match of that type of metadata, with
        the sum of all values ``<= 1``.ights = {'genre': .3, 'director': .2}``, then an item is:

        * a 100% match if it's the same item,

        * a 50% match if it's a different item with the same genre and same director,

        * a 30% match if it's a different item with the same genre and different director,

        * a 20% match if it's a different item with a different genre and same director,

        * a 0% match if it's a different item with a different genre and different director,
          which is equivalent to the loss without any partial credit
    y_range: tuple
        Specify as ``(min, max)`` to apply a sigmoid layer to the output score of the model to get
        predicted ratings within the range of ``min`` and ``max``
    load_model_path: str or Path
        To load a previously-saved model, pass in path to output of ``model.save_model()`` method.
        If ``None``, will initialize model as normal
    map_location: str or torch.device
        If ``load_model_path`` is provided, device specifying how to remap storage locations when
        ``torch.load``-ing the state dictionary

    """
    def __init__(self,
                 train: interactions_like_input = None,
                 val: interactions_like_input = None,
                 user_embedding_dim: int = 60,
                 item_embedding_dim: int = 60,
                 user_dense_layers_dims: List[float] = [48, 32],
                 item_dense_layers_dims: List[float] = [48, 32],
                 embedding_dropout_p: float = 0.0,
                 dense_dropout_p: float = 0.0,
                 lr: float = 1e-3,
                 bias_lr: Optional[Union[float, str]] = 1e-2,
                 lr_scheduler_func: Optional[Callable] = partial(ReduceLROnPlateau,
                                                                 patience=1,
                                                                 verbose=True),
                 weight_decay: float = 0.0,
                 optimizer: Union[str, Callable] = 'adam',
                 bias_optimizer: Optional[Union[str, Callable]] = 'sgd',
                 loss: Union[str, Callable] = 'hinge',
                 metadata_for_loss: Optional[Dict[str, torch.tensor]] = None,
                 metadata_for_loss_weights: Optional[Dict[str, float]] = None,
                 y_range: Optional[Tuple[float, float]] = None,
                 load_model_path: Optional[str] = None,
                 map_location: Optional[str] = None):
        super().__init__(**get_init_arguments())

    def _setup_model(self, **kwargs) -> None:
        """
        Method for building model internals that rely on the data passed in.

        This method will be called after ``prepare_data``.

        """
        self.user_biases = ZeroEmbedding(num_embeddings=self.hparams.num_users,
                                         embedding_dim=1)
        self.item_biases = ZeroEmbedding(num_embeddings=self.hparams.num_items,
                                         embedding_dim=1)
        self.user_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_users,
                                               embedding_dim=self.hparams.user_embedding_dim)
        self.item_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_items,
                                               embedding_dim=self.hparams.item_embedding_dim)

        self.embedding_dropout = nn.Dropout(p=self.hparams.embedding_dropout_p)
        self.dense_dropout = nn.Dropout(p=self.hparams.dense_dropout_p)

        # set up user dense layers
        user_dense_layers_dims = (
            [self.hparams.user_embedding_dim] + self.hparams.user_dense_layers_dims
        )
        self.user_dense_layers = [
            nn.Linear(user_dense_layers_dims[idx - 1], user_dense_layers_dims[idx])
            for idx in range(1, len(user_dense_layers_dims))
        ]
        for i, layer in enumerate(self.user_dense_layers):
            nn.init.xavier_normal_(self.user_dense_layers[i].weight)
            self.add_module('user_dense_layer_{}'.format(i), layer)

        # set up item dense layers
        item_dense_layers_dims = (
            [self.hparams.item_embedding_dim] + self.hparams.item_dense_layers_dims
        )
        self.item_dense_layers = [
            nn.Linear(item_dense_layers_dims[idx - 1], item_dense_layers_dims[idx])
            for idx in range(1, len(item_dense_layers_dims))
        ]
        for i, layer in enumerate(self.item_dense_layers):
            nn.init.xavier_normal_(self.item_dense_layers[i].weight)
            self.add_module('item_dense_layer_{}'.format(i), layer)

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
        user_embeddings = self.user_embeddings(users)
        item_embeddings = self.item_embeddings(items)

        for idx, user_dense_layer in enumerate(self.user_dense_layers):
            user_embeddings = F.leaky_relu(
                user_dense_layer(user_embeddings)
            )

            if idx < (len(self.user_dense_layers) - 1):
                user_embeddings = self.dense_dropout(user_embeddings)

        for idx, item_dense_layer in enumerate(self.item_dense_layers):
            item_embeddings = F.leaky_relu(
                item_dense_layer(item_embeddings)
            )

            if idx < (len(self.item_dense_layers) - 1):
                item_embeddings = self.dense_dropout(item_embeddings)

        preds = (
            (
                self.embedding_dropout(user_embeddings) * self.embedding_dropout(item_embeddings)
            ).sum(1)
            + self.user_biases(users).squeeze(1)
            + self.item_biases(items).squeeze(1)
        )

        if self.hparams.y_range is not None:
            preds = (
                torch.sigmoid(preds)
                * (self.hparams.y_range[1] - self.hparams.y_range[0])
                + self.hparams.y_range[0]
            )

        return preds

    def _get_item_embeddings(self) -> np.array:
        """Get item embeddings."""
        if not hasattr(self, 'item_embeddings_'):
            items = torch.arange(self.hparams.num_items, device=self.device)

            item_embeddings = self.item_embeddings(items)

            for item_dense_layer in self.item_dense_layers:
                item_embeddings = F.leaky_relu(
                    item_dense_layer(item_embeddings)
                )

            self.item_embeddings_ = item_embeddings.detach().cpu()

        return self.item_embeddings_
