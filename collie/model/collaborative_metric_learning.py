from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collie.model.base import BasePipeline, INTERACTIONS_LIKE_INPUT, ScaledEmbedding
from collie.utils import get_init_arguments, merge_docstrings


class CollaborativeMetricLearningModel(BasePipeline):
    # NOTE: the full docstring is merged in with ``BasePipeline``'s using ``merge_docstrings``.
    # Only the description of new or changed parameters are included in this docstring
    """
    Training pipeline for the collaborative metric learning model.

    ``CollaborativeMetricLearningModel`` models have an embedding layer for both users and items. A
    single float, prediction is retrieved by taking the pairwise distance between the two
    embeddings.

    The implementation here is meant to mimic its original implementation as specified here:
    https://arxiv.org/pdf/1803.00202.pdf [1]_

    All ``CollaborativeMetricLearningModel`` instances are subclasses of the ``LightningModule``
    class provided by PyTorch Lightning. This means to train a model, you will need a
    ``collie.model.CollieTrainer`` object, but the model can be saved and loaded without this
    ``Trainer`` instance. Example usage may look like:

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
    embedding_dim: int
        Number of latent factors to use for user and item embeddings
    sparse: bool
        Whether or not to treat embeddings as sparse tensors. If ``True``, cannot use weight decay
        on the optimizer
    y_range: tuple
        Specify as ``(min, max)`` to apply a sigmoid layer to the output score of the model to get
        predicted ratings within the range of ``min`` and ``max``

    References
    ----------
    .. [1] Campo, Miguel, et al. "Collaborative Metric Learning Recommendation System: Application
        to Theatrical Movie Releases." ArXiv.org, 1 Mar. 2018, arxiv.org/abs/1803.00202.

    """
    def __init__(self,
                 train: INTERACTIONS_LIKE_INPUT = None,
                 val: INTERACTIONS_LIKE_INPUT = None,
                 embedding_dim: int = 30,
                 sparse: bool = False,
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
                 y_range: Optional[Tuple[float, float]] = None,
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
                                               embedding_dim=self.hparams.embedding_dim,
                                               sparse=self.hparams.sparse)
        self.item_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_items,
                                               embedding_dim=self.hparams.embedding_dim,
                                               sparse=self.hparams.sparse)

    def forward(self, users: torch.tensor, items: torch.tensor) -> torch.tensor:
        """
        Forward pass through the model, equivalent to:

        ```prediction = pairwise_distance(user_embedding * item_embedding)```

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

    def _get_item_embeddings(self) -> torch.tensor:
        """Get item embeddings on device."""
        return self.item_embeddings.weight.data

    def _get_user_embeddings(self) -> torch.tensor:
        """Get user embeddings on device."""
        return self.user_embeddings.weight.data
