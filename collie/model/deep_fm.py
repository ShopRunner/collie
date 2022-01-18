from functools import partial
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collie.model.base import BasePipeline, INTERACTIONS_LIKE_INPUT, ScaledEmbedding, ZeroEmbedding
from collie.utils import get_init_arguments, merge_docstrings, trunc_normal


class DeepFM(BasePipeline):
    # NOTE: the full docstring is merged in with ``BasePipeline``'s using ``merge_docstrings``.
    # Only the description of new or changed parameters are included in this docstring
    """
    Training pipeline for a deep factorization model.

    ``DeepFM`` models combine a shallow factorization machine and a deep multilayer perceptron
    network in a single, unified model. The model consists of embedding tables for users and items,
    and model output is the sum of 1) factorization machine output of both embeddings (shallow) and
    2) MLP output for the concatenation of both embeddings (deep).

    The implementation here is meant to mimic its original implementation as specified here:
    https://arxiv.org/pdf/1703.04247.pdf [3]_

    All ``DeepFM`` instances are subclasses of the ``LightningModule`` class
    provided by PyTorch Lightning. This means to train a model, you will need a
    ``collie.model.CollieTrainer`` object, but the model can be saved and loaded without this
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
    bias_lr: float
        Bias terms learning rate. If 'infer', will set equal to ``lr``
    optimizer: torch.optim or str
        If a string, one of the following supported optimizers:

        * ``'sgd'`` (for ``torch.optim.SGD``)

        * ``'adam'`` (for ``torch.optim.Adam``)

    bias_optimizer: torch.optim or str
        Optimizer for the bias terms. This supports the same string options as ``optimizer``, with
        the addition of ``infer``, which will set the optimizer equal to ``optimizer``. If
        ``bias_optimizer`` is ``None``, only a single optimizer will be created for all model
        parameters
    y_range: tuple
        Specify as ``(min, max)`` to apply a sigmoid layer to the output score of the model to get
        predicted ratings within the range of ``min`` and ``max``

    References
    ----------
    .. [3] Guo, Huifeng, et al. "DeepFM: A Factorization-Machine Based Neural Network for CTR
        Prediction." ArXiv.org, 13 Mar. 2017, arxiv.org/abs/1703.04247.

    """
    def __init__(self,
                 train: INTERACTIONS_LIKE_INPUT = None,
                 val: INTERACTIONS_LIKE_INPUT = None,
                 embedding_dim: int = 8,
                 num_layers: int = 3,
                 final_layer: Optional[Union[str, Callable[..., Any]]] = None,
                 dropout_p: float = 0.0,
                 lr: float = 1e-3,
                 bias_lr: Optional[Union[float, str]] = 1e-2,
                 lr_scheduler_func: Optional[torch.optim.lr_scheduler._LRScheduler] = partial(
                     ReduceLROnPlateau,
                     patience=1,
                     verbose=True
                 ),
                 weight_decay: float = 0.0,
                 optimizer: Union[str, torch.optim.Optimizer] = 'adam',
                 bias_optimizer: Optional[Union[str, torch.optim.Optimizer]] = 'sgd',
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

    def _get_item_embeddings(self) -> torch.tensor:
        """Get item embeddings on device."""
        return self.item_embeddings.weight.data

    def _get_user_embeddings(self) -> torch.tensor:
        """Get user embeddings on device."""
        return self.user_embeddings.weight.data
