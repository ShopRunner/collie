from functools import partial
from typing import Callable, Dict, Iterable, Optional, Union

import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collie.interactions import (ApproximateNegativeSamplingInteractionsDataLoader,
                                 Interactions,
                                 InteractionsDataLoader)
from collie.model import MultiStagePipeline, ScaledEmbedding, ZeroEmbedding
from collie.utils import get_init_arguments, merge_docstrings


INTERACTIONS_LIKE_INPUT = Union[ApproximateNegativeSamplingInteractionsDataLoader,
                                Interactions,
                                InteractionsDataLoader]


class ColdStartModel(MultiStagePipeline):
    # NOTE: the full docstring is merged in with ``MultiStagePipeline``'s using
    # ``merge_docstrings``. Only the description of new or changed parameters are included in this
    # docstring
    """
    Training pipeline for a matrix factorization model optimized for the cold-start problem.

    Many recommendation models suffer from the cold start problem, when a model is unable to provide
    adequate recommendations for a new item until enough users have interacted with it. But, if
    users only interact with recommended items, the item will never be recommended, and thus the
    model will never improve recommendations for this item.

    The ``ColdStartModel`` attempts to bypass this by limiting the item space down to "item
    buckets", training a model on this as the item space, then expanding out to all items. During
    this expansion, the learned-embeddings of each bucket is copied over to each corresponding
    item, providing a smarter initialization than a random one for both existing and new items.
    Now, when we have a new item, we can use its bucket embedding as an initialization into a model.

    The stages in a ``ColdStartModel`` are, in order:

    1. ``item_buckets``
        Matrix factorization with item embeddings and bias terms bucketed by
        ``item_buckets`` argument. Unlike in the next stage, many items may map on to a single
        bucket, and this will share the same embedding and bias representation. The model should
        learn user preference for buckets in this stage.
    2. ``no_buckets``
        Standard matrix factorization as we do in ``MatrixFactorizationModel``. However, upon
        advancing to this stage, the item embeddings are initialized with their bucketed embedding
        value (and same for biases). Not only does this provide better initialization than random,
        but allows new items to be incorporated into the model without training by using their
        item bucket embedding and bias terms at prediction time.

    Note that the cold start problem exists for new users as well, but this functionality will be
    added to this model in a future version.

    All ``ColdStartModel`` instances are subclasses of the ``LightningModule`` class provided by
    PyTorch Lightning. This means to train a model, you will need a
    ``collie.model.CollieTrainer`` object, but the model can be saved and loaded without this
    ``Trainer`` instance. Example usage may look like:

    .. code-block:: python

        from collie.model import ColdStartModel, CollieTrainer


        # instantiate and fit a ``ColdStartModel`` as expected
        model = ColdStartModel(train=train, item_buckets=item_buckets)
        trainer = CollieTrainer(model)
        trainer.fit(model)

        # train for X more epochs on the next stage, ``no_buckets``
        trainer.max_epochs += X
        model.advance_stage()
        trainer.fit(model)

        model.eval()

        # do evaluation as normal with ``model``

        # get item-item recommendations for a new item by using the bucket ID, Z
        similar_items = model.item_bucket_item_similarity(item_bucket_id=Z)

        model.save_model(filename='model.pth')
        new_model = ColdStartModel(load_model_path='model.pth')

        # do evaluation as normal with ``new_model``

    Note
    ----
    The ``forward`` calculation will be different depending on the stage that is set. Note this
    when evaluating / saving and loading models in.

    Parameters
    ----------
    item_buckets: torch.tensor, 1-d
        An ordered iterable containing the bucket ID for each item ID. For example, if you have
        five films and are going to bucket by primary genre, and your data looks like:

        * Item ID: 0, Genre ID: 1

        * Item ID: 1, Genre ID: 0

        * Item ID: 2, Genre ID: 2

        * Item ID: 3, Genre ID: 2

        * Item ID: 4, Genre ID: 1

        Then ``item_buckets`` would be: ``[1, 0, 2, 2, 1]``
    embedding_dim: int
        Number of latent factors to use for user and item embeddings
    dropout_p: float
        Probability of dropout
    item_buckets_stage_lr: float
        Learning rate for user parameters and item bucket parameters optimized during the
        ``item_buckets`` stage
    no_buckets_stage_lr: float
        Learning rate for user parameters and item parameters optimized during the ``no_buckets``
        stage
    item_buckets_stage_lr: float
        Optimizer used for user parameters and item bucket parameters optimized during the
        ``item_buckets`` stage. If a string, one of the following supported optimizers:

        * ``'sgd'`` (for ``torch.optim.SGD``)

        * ``'adam'`` (for ``torch.optim.Adam``)

    no_buckets_stage_lr: float
        Optimizer used for user parameters and item parameters optimized during the ``no_buckets``
        stage. If a string, one of the following supported optimizers:

        * ``'sgd'`` (for ``torch.optim.SGD``)

        * ``'adam'`` (for ``torch.optim.Adam``)

    """
    def __init__(self,
                 train: INTERACTIONS_LIKE_INPUT = None,
                 val: INTERACTIONS_LIKE_INPUT = None,
                 item_buckets: Iterable[int] = None,
                 embedding_dim: int = 30,
                 dropout_p: float = 0.0,
                 sparse: bool = False,
                 item_buckets_stage_lr: float = 1e-3,
                 no_buckets_stage_lr: float = 1e-3,
                 lr_scheduler_func: Optional[torch.optim.lr_scheduler._LRScheduler] = partial(
                     ReduceLROnPlateau,
                     patience=1,
                     verbose=False,
                 ),
                 weight_decay: float = 0.0,
                 item_buckets_stage_optimizer: Union[str, torch.optim.Optimizer] = 'adam',
                 no_buckets_stage_optimizer: Union[str, torch.optim.Optimizer] = 'adam',
                 loss: Union[str, Callable[..., torch.tensor]] = 'hinge',
                 metadata_for_loss: Optional[Dict[str, torch.tensor]] = None,
                 metadata_for_loss_weights: Optional[Dict[str, float]] = None,
                 load_model_path: Optional[str] = None,
                 map_location: Optional[str] = None):
        optimizer_config_list = None
        num_item_buckets = None

        if load_model_path is None:
            # TODO: separate out optimizer and bias optimizer somehow
            optimizer_config_list = [
                {
                    'lr': item_buckets_stage_lr,
                    'optimizer': item_buckets_stage_optimizer,
                    'parameter_prefix_list': [
                        'user_embed',
                        'user_bias',
                        'item_bucket_embed',
                        'item_bucket_bias',
                    ],
                    'stage': 'item_buckets',
                },
                {
                    'lr': no_buckets_stage_lr,
                    'optimizer': no_buckets_stage_optimizer,
                    'parameter_prefix_list': [
                        'user_embed',
                        'user_bias',
                        'item_embed',
                        'item_bias',
                    ],
                    'stage': 'no_buckets',
                },
            ]

            if not isinstance(item_buckets, torch.Tensor):
                item_buckets = torch.tensor(item_buckets)

            # data quality checks for ``item_buckets``
            assert item_buckets.dim() == 1, (
                f'``item_buckets`` must be 1-dimensional, not {item_buckets.dim()}-dimensional!'
            )
            if len(item_buckets) != train.num_items:
                raise ValueError(
                    'Length of ``item_buckets`` must be equal to the number of items in the '
                    f'dataset: {len(item_buckets)} != {train.num_items}.'
                )
            if min(item_buckets) != 0:
                raise ValueError(f'``item_buckets`` IDs must start at 0, not {min(item_buckets)}!')

            num_item_buckets = item_buckets.max().item() + 1

        super().__init__(optimizer_config_list=optimizer_config_list,
                         num_item_buckets=num_item_buckets,
                         **get_init_arguments())

    __doc__ = merge_docstrings(MultiStagePipeline, __doc__, __init__)

    def _move_any_external_data_to_device(self):
        """Move the item buckets to the device before training."""
        self.hparams.item_buckets = self.hparams.item_buckets.to(self.device)

    def _copy_weights(self, old: nn.Embedding, new: nn.Embedding, buckets: torch.tensor) -> None:
        new.weight.data.copy_(old.weight.data[buckets])

    def set_stage(self, stage: str) -> None:
        """Set the stage for the model."""
        current_stage = self.hparams.stage

        if stage in self.hparams.stage_list:
            if current_stage == 'item_buckets' and stage == 'no_buckets':
                print('Copying over item embeddings...')
                self._copy_weights(self.item_bucket_biases,
                                   self.item_biases,
                                   self.hparams.item_buckets)
                self._copy_weights(self.item_bucket_embeddings,
                                   self.item_embeddings,
                                   self.hparams.item_buckets)
        else:
            raise ValueError(
                f'"{stage}" is not a valid stage, please choose one of {self.hparams.stage_list}'
            )

        self.hparams.stage = stage
        print(f'Set ``self.hparams.stage`` to "{stage}"')

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
        -------
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

    def item_bucket_item_similarity(self, item_bucket_id: int) -> pd.Series:
        """
        Get most similar item indices to a item bucket by cosine similarity.

        Cosine similarity is computed with item and item bucket embeddings from a trained model.

        Parameters
        ----------
        item_id: int

        Returns
        -------
        sim_score_idxs: pd.Series
            Sorted values as cosine similarity for each item in the dataset with the index being
            the item ID

        """
        item_bucket_embeddings = self.item_bucket_embeddings.weight.data
        item_bucket_embeddings = (
            item_bucket_embeddings / item_bucket_embeddings.norm(dim=1)[:, None]
        )

        item_embeddings = self._get_item_embeddings()
        item_embeddings = item_embeddings / item_embeddings.norm(dim=1)[:, None]

        sim_score_idxs = (
            torch.matmul(item_bucket_embeddings[[item_bucket_id], :],
                         item_embeddings.transpose(1, 0))
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        sim_score_idxs_series = pd.Series(sim_score_idxs)
        sim_score_idxs_series = sim_score_idxs_series.sort_values(ascending=False)

        return sim_score_idxs_series

    def _get_item_embeddings(self) -> torch.tensor:
        """Get item embeddings on device."""
        return self.item_embeddings.weight.data

    def _get_user_embeddings(self) -> torch.tensor:
        """Get user embeddings on device."""
        return self.user_embeddings.weight.data
