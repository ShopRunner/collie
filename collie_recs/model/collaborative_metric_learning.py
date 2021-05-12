from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collie_recs.model.base import (BasePipeline,
                                    interactions_like_input,
                                    ScaledEmbedding)
from collie_recs.utils import get_init_arguments


class CollaborativeMetricLearningModel(BasePipeline):
    """
    Training pipeline for the collaborative metric learning model.

    ``CollaborativeMetricLearningModel`` models have an embedding layer for both users and items. A
    single float, prediction is retrieved by taking the pairwise distance between the two
    embeddings.

    All ``CollaborativeMetricLearningModel`` instances are subclasses of the ``LightningModule``
    class provided by PyTorch Lightning. This means to train a model, you will need a
    ``collie_recs.model.CollieTrainer`` object, but the model can be saved and loaded without this
    ``Trainer`` instance. Example usage may look like:
    may look like:

    .. code-block:: python

        from collie.model import CollaborativeMetricLearningModel, CollieTrainer

        model = CollaborativeMetricLearningModel(train=train)
        trainer = CollieTrainer(model)
        trainer.fit(model)
        model.eval()

        # do evaluation as normal with ``model``

        model.save_model(filename='model.pth')
        new_model = CollaborativeMetricLearningModel(load_model_path='model.pth')

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
        Number of latent factors to use for user and item embeddings
    sparse: bool
        Whether or not to treat embeddings as sparse tensors. If ``True``, cannot use weight decay
        on the optimizer
    lr: float
        Embedding layer learning rate
    lr_scheduler_func: torch.optim.lr_scheduler
        Learning rate scheduler to use during fitting
    weight_decay: float
        Weight decay passed to the optimizer, if optimizer permits
    optimizer: torch.optim or str
        If a string, one of the following supported optimizers:

        * ``'sgd'`` (for ``torch.optim.SGD``)

        * ``'adam'`` (for ``torch.optim.Adam``)

        * ``'sparse_adam'`` (for ``torch.optim.SparseAdam``)
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
                 embedding_dim: int = 30,
                 sparse: bool = False,
                 lr: float = 1e-3,
                 lr_scheduler_func: Optional[Callable] = partial(ReduceLROnPlateau,
                                                                 patience=1,
                                                                 verbose=True),
                 weight_decay: float = 0.0,
                 optimizer: Union[str, Callable] = 'adam',
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

        This method will be called after `prepare_data`.

        """
        self.user_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_users,
                                               embedding_dim=self.hparams.embedding_dim,
                                               sparse=self.hparams.sparse)
        self.item_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_items,
                                               embedding_dim=self.hparams.embedding_dim,
                                               sparse=self.hparams.sparse)

    def forward(self, users: torch.tensor, items: torch.tensor) -> torch.tensor:
        """
        Forward pass through the model.

        Simple matrix factorization for a single user and item looks like:

        ```prediction = (user_embedding * item_embedding) + user_bias + item_bias```

        If dropout is added, it is applied for the two embeddings and not the biases.

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

        preds = F.pairwise_distance(user_embeddings, item_embeddings)

        return preds

    def _get_item_embeddings(self) -> np.array:
        """Get item embeddings."""
        return self.item_embeddings(
            torch.arange(self.hparams.num_items, device=self.device)
        ).detach().cpu()
