from functools import partial
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collie_recs.model.base import (BasePipeline,
                                    interactions_like_input,
                                    ScaledEmbedding,
                                    ZeroEmbedding)
from collie_recs.utils import get_init_arguments, trunc_normal


class DeepFM(BasePipeline):
    """
    Training pipeline for a deep factorization model.

    ``DeepFM`` models combine a shallow factorization machine and a deep multilayer perceptron
    network in a single, unified model. The model consists of embedding tables for users and items,
    and model output is the sum of 1) factorization machine output of both embeddings (shallow) and
    2) MLP output for the concatenation of both embeddings (deep).

    The implementation here is meant to mimic its original implementation as specified here:
    https://arxiv.org/pdf/1703.04247.pdf

    All ``DeepFM`` instances are subclasses of the ``LightningModule`` class
    provided by PyTorch Lightning. This means to train a model, you will need a
    ``collie_recs.model.CollieTrainer`` object, but the model can be saved and loaded without this
    ``Trainer`` instance. Example usage may look like:

    .. code-block:: python

        from collie.model import CollieTrainer, DeepFM


        model = DeepFM(train=train)
        trainer = CollieTrainer(model)
        trainer.fit(model)
        model.eval()

        # do evaluation as normal with ``model``

        model.save_model(filename='model.pth')
        new_model = DeepFM(load_model_path='model.pth')

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
    embedding_dim: int
        Number of latent factors to use for the matrix factorization embedding table. For the MLP
        embedding table, the dimensionality will be calculated with the formula
        ``embedding_dim * (2 ** (num_layers - 1))``
    num_layers: int
        Number of MLP layers to apply. Each MLP layer will have its input dimension calculated with
        the formula ``embedding_dim * (2 ** (``num_layers`` - ``current_layer_number``))``
    final_layer: str or function
        Final layer activation function. Available string options include:

        * 'sigmoid'

        * 'relu'

        * 'leaky_relu'

    dropout_p: float
        Probability of dropout
    sparse: bool
        Whether or not to treat embeddings as sparse tensors. If ``True``, cannot use weight decay
        on the optimizer
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

        * ``'sparse_adam'`` (for ``torch.optim.SparseAdam``)

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
        the sum of all values ``<= 1``.
        e.g. If ``metadata_for_loss_weights = {'genre': .3, 'director': .2}``, then an item is:

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
                 embedding_dim: int = 8,
                 num_layers: int = 3,
                 final_layer: Optional[Union[str, Callable]] = None,
                 dropout_p: float = 0.0,
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
                 # y_range: Optional[Tuple[float, float]] = None,
                 load_model_path: Optional[str] = None,
                 map_location: Optional[str] = None):
        super().__init__(**get_init_arguments())

    def _setup_model(self, **kwargs) -> None:
        """
        Method for building model internals that rely on the data passed in.

        This method will be called after `prepare_data`.

        """
        self.user_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_users,
                                               embedding_dim=self.hparams.embedding_dim)
        self.item_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_items,
                                               embedding_dim=self.hparams.embedding_dim)
        self.user_biases = ZeroEmbedding(num_embeddings=self.hparams.num_users,
                                         embedding_dim=1)
        self.item_biases = ZeroEmbedding(num_embeddings=self.hparams.num_items,
                                         embedding_dim=1)
        self.user_global_bias = nn.Parameter(torch.zeros(1))
        self.item_global_bias = nn.Parameter(torch.zeros(1))

        mlp_modules = []
        input_size = self.hparams.embedding_dim * 2
        for i in range(self.hparams.num_layers):
            next_input_size = (
                int(
                    self.hparams.embedding_dim
                    * 2
                    * ((self.hparams.num_layers - i) / (self.hparams.num_layers + 1))
                )
            )
            mlp_modules.append(nn.Linear(input_size, next_input_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=self.hparams.dropout_p))
            input_size = next_input_size
        self.mlp_layers = nn.Sequential(*mlp_modules)

        self.predict_layer = nn.Linear(next_input_size, 1)

        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                # initialization taken from the official repo:
                # https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/NeuMF.py
                trunc_normal(m.weight.data, std=0.01)

        nn.init.kaiming_uniform_(self.predict_layer.weight, nonlinearity='relu')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

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

        # FM output
        embedding_sum = user_embeddings + item_embeddings
        embedding_squared_sum = torch.pow(user_embeddings, 2) + torch.pow(item_embeddings, 2)
        embeddings_difference = embedding_sum - embedding_squared_sum
        fm_output = torch.sum(embeddings_difference, dim=1)

        # MLP output
        concatenated_embeddings = torch.cat((user_embeddings, item_embeddings), -1)
        mlp_output = self.predict_layer(self.mlp_layers(concatenated_embeddings)).squeeze()

        prediction = fm_output + mlp_output

        if callable(self.hparams.final_layer):
            prediction = self.hparams.final_layer(prediction)
        elif self.hparams.final_layer == 'sigmoid':
            prediction = torch.sigmoid(prediction)
        elif self.hparams.final_layer == 'relu':
            prediction = F.relu(prediction)
        elif self.hparams.final_layer == 'leaky_relu':
            prediction = F.leaky_relu(prediction)
        elif self.hparams.final_layer is not None:
            raise ValueError(f'{self.hparams.final_layer} not valid final layer value!')

        return prediction.view(-1)

    def _get_item_embeddings(self) -> np.array:
        """Get item embeddings."""
        return self.item_embeddings(
            torch.arange(self.hparams.num_items, device=self.device)
        ).detach().cpu()