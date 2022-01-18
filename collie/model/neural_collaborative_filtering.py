from functools import partial
from typing import Callable, Dict, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collie.model.base import BasePipeline, INTERACTIONS_LIKE_INPUT, ScaledEmbedding
from collie.utils import get_init_arguments, merge_docstrings, trunc_normal


class NeuralCollaborativeFiltering(BasePipeline):
    # NOTE: the full docstring is merged in with ``BasePipeline``'s using ``merge_docstrings``.
    # Only the description of new or changed parameters are included in this docstring
    """
    Training pipeline for a neural matrix factorization model.

    ``NeuralCollaborativeFiltering`` models combine a collaborative filtering and multilayer
    perceptron network in a single, unified model. The model consists of two sections: the first
    is a simple matrix factorization that calculates a score by multiplying together user and item
    embeddings (lookups through an embedding table); the second is a MLP network that feeds
    embeddings from a second set of embedding tables (one for user, one for item). Both output
    vectors are combined and sent through a final MLP layer before returning a single recommendation
    score.

    The implementation here is meant to mimic its original implementation as specified here:
    https://arxiv.org/pdf/1708.05031.pdf [2]_

    All ``NeuralCollaborativeFiltering`` instances are subclasses of the ``LightningModule`` class
    provided by PyTorch Lightning. This means to train a model, you will need a
    ``collie.model.CollieTrainer`` object, but the model can be saved and loaded without this
    ``Trainer`` instance. Example usage may look like:

    .. code-block:: python

        from collie.model import CollieTrainer, NeuralCollaborativeFiltering


        model = NeuralCollaborativeFiltering(train=train)
        trainer = CollieTrainer(model)
        trainer.fit(model)
        model.eval()

        # do evaluation as normal with ``model``

        model.save_model(filename='model.pth')
        new_model = NeuralCollaborativeFiltering(load_model_path='model.pth')

        # do evaluation as normal with ``new_model``

    Parameters
    ----------
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
        Probability of dropout on the MLP layers
    optimizer: torch.optim or str
        If a string, one of the following supported optimizers:

        * ``'sgd'`` (for ``torch.optim.SGD``)

        * ``'adam'`` (for ``torch.optim.Adam``)

    References
    ----------
    .. [2] Xiangnan et al. "Neural Collaborative Filtering." Neural Collaborative Filtering |
        Proceedings of the 26th International Conference on World Wide Web, 1 Apr. 2017,
        dl.acm.org/doi/10.1145/3038912.3052569.

    """
    def __init__(self,
                 train: INTERACTIONS_LIKE_INPUT = None,
                 val: INTERACTIONS_LIKE_INPUT = None,
                 embedding_dim: int = 8,
                 num_layers: int = 3,
                 final_layer: Optional[Union[str, Callable[[torch.tensor], torch.tensor]]] = None,
                 dropout_p: float = 0.0,
                 lr: float = 1e-3,
                 lr_scheduler_func: Optional[torch.optim.lr_scheduler._LRScheduler] = partial(
                     ReduceLROnPlateau,
                     patience=1,
                     verbose=True
                 ),
                 weight_decay: float = 0.0,
                 optimizer: Union[str, torch.optim.Optimizer] = 'adam',
                 loss: Union[str, Callable[..., torch.tensor]] = 'hinge',
                 metadata_for_loss: Optional[Dict[str, torch.tensor]] = None,
                 metadata_for_loss_weights: Optional[Dict[str, float]] = None,
                 # y_range: Optional[Tuple[float, float]] = None,
                 load_model_path: Optional[str] = None,
                 map_location: Optional[str] = None):
        super().__init__(**get_init_arguments())

    __doc__ = merge_docstrings(BasePipeline, __doc__, __init__)

    def _setup_model(self, **kwargs) -> None:
        """
        Method for building model internals that rely on the data passed in.

        This method will be called after ``prepare_data``.

        """
        self.user_embeddings_cf = ScaledEmbedding(num_embeddings=self.hparams.num_users,
                                                  embedding_dim=self.hparams.embedding_dim)
        self.item_embeddings_cf = ScaledEmbedding(num_embeddings=self.hparams.num_items,
                                                  embedding_dim=self.hparams.embedding_dim)

        mlp_embedding_dim = self.hparams.embedding_dim * (2 ** (self.hparams.num_layers - 1))
        self.user_embeddings_mlp = ScaledEmbedding(
            num_embeddings=self.hparams.num_users,
            embedding_dim=mlp_embedding_dim,
        )
        self.item_embeddings_mlp = ScaledEmbedding(
            num_embeddings=self.hparams.num_items,
            embedding_dim=mlp_embedding_dim,
        )

        mlp_modules = []
        for i in range(self.hparams.num_layers):
            input_size = self.hparams.embedding_dim * (2 ** (self.hparams.num_layers - i))
            mlp_modules.append(nn.Dropout(p=self.hparams.dropout_p))
            mlp_modules.append(nn.Linear(input_size, input_size//2))
            mlp_modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_modules)

        self.predict_layer = nn.Linear(self.hparams.embedding_dim * 2, 1)

        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                # initialization taken from the official repo:
                # https://github.com/hexiangnan/neural_collaborative_filtering/blob/master/NeuMF.py#L63  # noqa: E501
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
        user_embedding_cf = self.user_embeddings_cf(users)
        item_embedding_cf = self.item_embeddings_cf(items)
        output_cf = user_embedding_cf * item_embedding_cf

        user_embedding_mlp = self.user_embeddings_mlp(users)
        item_embedding_mlp = self.item_embeddings_mlp(items)
        interaction = torch.cat((user_embedding_mlp, item_embedding_mlp), -1)
        output_mlp = self.mlp_layers(interaction)

        concat = torch.cat((output_cf, output_mlp), -1)

        prediction = self.predict_layer(concat)

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

    def _get_item_embeddings(self) -> torch.tensor:
        """Get item embeddings, which are the concatenated CF and MLP item embeddings, on device."""
        items = torch.arange(self.hparams.num_items, device=self.device)

        return torch.cat((
            self.item_embeddings_cf(items),
            self.item_embeddings_mlp(items),
        ), axis=1).detach()

    def _get_user_embeddings(self) -> torch.tensor:
        """Get user embeddings, which are the concatenated CF and MLP user embeddings, on device."""
        users = torch.arange(self.hparams.num_users, device=self.device)

        return torch.cat((
            self.user_embeddings_cf(users),
            self.user_embeddings_mlp(users),
        ), axis=1).detach()
