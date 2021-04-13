from abc import ABCMeta, abstractmethod
from pathlib import Path
import sys
import textwrap
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
import torch
from torch import nn

from collie_recs.interactions import (ApproximateNegativeSamplingInteractionsDataLoader,
                                      Interactions,
                                      InteractionsDataLoader)
from collie_recs.loss import (adaptive_bpr_loss,
                              adaptive_hinge_loss,
                              bpr_loss,
                              hinge_loss,
                              warp_loss)
from collie_recs.utils import get_init_arguments


interactions_like_input = Union[ApproximateNegativeSamplingInteractionsDataLoader,
                                Interactions,
                                InteractionsDataLoader]


class ScaledEmbedding(nn.Embedding):
    """Embedding layer that initializes its values to use a truncated normal distribution."""
    def reset_parameters(self) -> None:
        """Overriding default ``reset_parameters`` method."""
        self.weight.data.normal_(0, 1.0 / (self.embedding_dim * 2.5))


class ZeroEmbedding(nn.Embedding):
    """Embedding layer with weights zeroed-out."""
    def reset_parameters(self) -> None:
        """Overriding default ``reset_parameters`` method."""
        self.weight.data.zero_()


class CollieTrainer(Trainer):
    """
    Helper wrapper class around PyTorch Lightning's ``Trainer`` class.

    Specifically, this wrapper:

    * Checks if a model has a validation dataset passed in (under the ``val_loader`` attribute)
      and, if not, sets ``num_sanity_val_steps`` to 0 and ``check_val_every_n_epoch`` to
      ``sys.maxint``.

    * Checks if a GPU is available and, if ``gpus is None``, sets ``gpus = -1``.

    See ``pytorch_lightning.Trainer`` documentation for more details at:
    https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api

    Parameters
    ----------
    model: collie_recs.model.BasePipeline
        Initialized Collie model
    benchmark: bool
        If set to ``True``, enables ``cudnn.benchmark``
    deterministic: bool
        If set to ``True``, enables ``cudnn.deterministic``
    kwargs: keyword arguments
        Additional keyword arguments to be sent to the ``Trainer`` class:
        https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api

    """

    def __init__(self,
                 model: torch.nn.Module,
                 benchmark: bool = True,
                 deterministic: bool = True,
                 **kwargs):
        if not hasattr(model, 'val_loader') or model.val_loader is None:
            print('Did not detect ``val_loader``. Setting ``num_sanity_val_steps`` to 0.')
            kwargs['num_sanity_val_steps'] = 0
            kwargs['check_val_every_n_epoch'] = sys.maxsize

        if kwargs.get('gpus') is None and torch.cuda.is_available():
            print('Detected GPU. Setting ``gpus`` to -1.')
            kwargs['gpus'] = -1

        kwargs['benchmark'] = benchmark
        kwargs['deterministic'] = deterministic

        super().__init__(**kwargs)


class BasePipeline(LightningModule, metaclass=ABCMeta):
    """
    Base Pipeline model architectures to inherit from.

    All subclasses MUST at least override the following methods:

    * ``_setup_model`` - Set up the model architecture

    * ``forward`` - Forward pass through a model

    For ``item_item_similarity`` to work properly, all subclasses are should also implement:

    * ``_get_item_embeddings`` - Returns item embeddings from the model

    Parameters
    ----------
    train: ``collie_recs.interactions`` object
        Data loader for training data. If an ``Interactions`` object is supplied, an
        ``InteractionsDataLoader`` will automatically be instantiated with ``shuffle=True``
    val: ``collie_recs.interactions`` object
        Data loader for validation data. If an ``Interactions`` object is supplied, an
        ``InteractionsDataLoader`` will automatically be instantiated with ``shuffle=False``
    lr: float
        Model learning rate
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
                 lr: float = 1e-3,
                 lr_scheduler_func: Optional[Callable] = None,
                 weight_decay: float = 0.0,
                 optimizer: Union[str, Callable] = 'adam',
                 loss: Union[str, Callable] = 'hinge',
                 metadata_for_loss: Optional[Dict[str, torch.tensor]] = None,
                 metadata_for_loss_weights: Optional[Dict[str, float]] = None,
                 load_model_path: Optional[str] = None,
                 map_location: Optional[str] = None,
                 **kwargs):
        if isinstance(train, Interactions):
            train = InteractionsDataLoader(interactions=train, shuffle=True)
        if isinstance(val, Interactions):
            val = InteractionsDataLoader(interactions=val, shuffle=False)

        super().__init__()

        # save datasets as class-level attributes and NOT ``hparams`` so model checkpointing /
        # saving can complete faster
        self.train_loader = train
        self.val_loader = val

        # potential issue with PyTorch Lightning is that a function cannot be saved as a
        # hyperparameter, so we will sidestep this by setting it as a class-level attribute
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2444
        self.lr_scheduler_func = lr_scheduler_func
        self.loss = loss
        self.optimizer = optimizer
        self.bias_optimizer = kwargs.get('bias_optimizer')

        if load_model_path is not None:
            # we are loading in a previously-saved model, not creating a new one
            self._load_model_init_helper(load_model_path=load_model_path,
                                         map_location=map_location,
                                         **kwargs)
        else:
            # saves all passed-in parameters
            init_args = get_init_arguments(
                exclude=['train', 'val', 'item_metadata', 'trained_model'],
                verbose=False,
            )

            self.save_hyperparameters(init_args, *kwargs.keys())

            self.hparams.num_users = self.train_loader.num_users
            self.hparams.num_items = self.train_loader.num_items
            self.hparams.n_epochs_completed_ = 0

            self._configure_loss()

            # check weight decay and sparsity
            if hasattr(self.hparams, 'sparse'):
                if self.hparams.sparse and self.hparams.weight_decay != 0:
                    warnings.warn(
                        textwrap.dedent(
                            f'''
                            ``weight_decay`` value must be 0 when ``sparse`` is flagged, not
                            {self.hparams.weight_decay}. Setting to 0.
                            '''
                        ).replace('\n', ' ').strip()
                    )
                    self.hparams.weight_decay = 0.0

            # set up the actual model
            self._setup_model(**kwargs)

    def _load_model_init_helper(self, load_model_path: str, map_location: str, **kwargs) -> None:
        loaded_dict = torch.load(str(load_model_path), map_location=map_location)

        for key, value in loaded_dict['hparams'].items():
            self.hparams[key] = value

        self.hparams['load_model_path'] = load_model_path
        self.hparams['map_location'] = map_location

        self._setup_model(**kwargs)

        self.load_state_dict(state_dict=loaded_dict['state_dict'])
        self.eval()

    @abstractmethod
    def _setup_model(self, **kwargs) -> None:
        """``_setup_model`` should be implemented in all subclasses."""
        pass

    @abstractmethod
    def forward(self, users: torch.tensor, items: torch.tensor) -> torch.tensor:
        """``forward`` should be implemented in all subclasses."""
        pass

    def _configure_loss(self) -> None:
        # set up loss function
        self.loss_function = None

        if callable(self.loss):
            self.loss_function = self.loss
        elif self.loss == 'warp':
            if self.train_loader.num_negative_samples > 1:
                self.loss_function = warp_loss
            else:
                raise ValueError('Cannot use WARP loss with a single negative sample!')
        elif 'bpr' in self.loss:
            if self.train_loader.num_negative_samples > 1:
                self.loss_function = adaptive_bpr_loss
            else:
                self.loss_function = bpr_loss
        elif 'hinge' in self.loss or 'adaptive' in self.loss:
            if self.train_loader.num_negative_samples > 1:
                self.loss_function = adaptive_hinge_loss
            else:
                self.loss_function = hinge_loss
        else:
            raise ValueError('{} is not a valid loss function.'.format(self.loss))

    def configure_optimizers(self) -> (
        Union[Tuple[List[Callable], List[Callable]], Tuple[Callable, Callable], Callable]
    ):
        """
        Configure optimizers and learning rate schedulers to use in optimization.

        This method will be called after ``setup``.

        If ``self.bias_optimizer`` is None, only a single optimizer will be returned. If there is a
        non-``None`` class attribute for ``bias_optimizer``, two optimizers will be created: one
        for all layers with the name 'bias' in them, and another for all other model parameters.
        The bias optimizer will be set with the same parameters as ``optimizer`` with the
        exception of the learning rate, which will be set to ``self.hparams.bias_lr``.

        """
        if self.bias_optimizer is not None:
            if self.bias_optimizer == 'infer':
                self.bias_optimizer = self.optimizer
            if self.hparams.bias_lr == 'infer':
                self.hparams.bias_lr = self.hparams.lr

            # create one optimizer only for layers with the term 'bias' in them
            bias_optimizer = self._get_optimizer(self.bias_optimizer, optimizer_type='bias')
            # create another optimizer for all other model layers
            optimizer = self._get_optimizer(self.optimizer, optimizer_type='all_but_bias')
        else:
            # create a single optimizer for all model layers
            optimizer = self._get_optimizer(self.optimizer, optimizer_type='all')

        if self.lr_scheduler_func is not None:
            monitor = 'val_loss_epoch'
            if self.val_loader is None:
                monitor = 'train_loss_epoch'

            # add in optimizer to scheduler function
            scheduler_dict = {
                'scheduler': self.lr_scheduler_func(optimizer),
                'monitor': monitor,
            }

            optimizer_list = [optimizer]
            scheduler_list = [scheduler_dict]

            # create a separate learning rate scheduler for our bias optimizer if it exists
            if self.bias_optimizer is not None:
                scheduler_dict_bias = {
                    'scheduler': self.lr_scheduler_func(bias_optimizer),
                    'monitor': monitor,
                }

                optimizer_list.append(bias_optimizer)
                scheduler_list.append(scheduler_dict_bias)

            return optimizer_list, scheduler_list

        if self.bias_optimizer is not None:
            return optimizer, bias_optimizer
        else:
            return optimizer

    def _get_optimizer(self, optimizer: Optional[Union[str, Callable]], **kwargs) -> Callable:
        if callable(optimizer):
            try:
                optimizer = optimizer(
                    self._get_optimizer_parameters(include_weight_decay=True, **kwargs)
                )
            except TypeError:
                optimizer = optimizer(
                    self._get_optimizer_parameters(include_weight_decay=False, **kwargs)
                )
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self._get_optimizer_parameters(include_weight_decay=True, **kwargs)
            )
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self._get_optimizer_parameters(include_weight_decay=True, **kwargs)
            )
        elif optimizer == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(
                self._get_optimizer_parameters(include_weight_decay=False, **kwargs)
            )
        else:
            raise ValueError('{} is not a valid optimizer!'.format(optimizer))

        return optimizer

    def _get_optimizer_parameters(self,
                                  include_weight_decay: bool = True,
                                  optimizer_type: str = 'all',
                                  **kwargs) -> Dict[str, Union[torch.tensor, float]]:
        """
        Set all non-bias model layers with ``lr`` learning rate and bias terms with ``bias_lr``
        learning rate.

        """
        assert optimizer_type in ['bias', 'all_but_bias', 'all'], f'{optimizer_type} not valid!'

        if optimizer_type == 'bias':
            optimizer_parameters = [
                {
                    'params': (
                        param for (name, param) in self.named_parameters() if 'bias' in name
                    ),
                    'lr': self.hparams.bias_lr,
                },
            ]
        elif optimizer_type == 'all_but_bias':
            optimizer_parameters = [
                {
                    'params': (
                        param for (name, param) in self.named_parameters() if 'bias' not in name
                    ),
                    'lr': self.hparams.lr,
                },
            ]
        else:
            optimizer_parameters = [{
                'params': self.parameters(),
                'lr': self.hparams.lr,
            }]

        if include_weight_decay:
            weight_decay_dict = {'weight_decay': self.hparams.weight_decay}
            for d in optimizer_parameters:
                d.update(weight_decay_dict)

        return optimizer_parameters

    def train_dataloader(self) -> Union[InteractionsDataLoader,
                                        ApproximateNegativeSamplingInteractionsDataLoader]:
        """
        Method that sets up training data as a PyTorch DataLoader.

        This method will be called after ``configure_optimizers``.

        """
        return self.train_loader

    def training_step(
        self,
        batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
    ) -> torch.tensor:
        """
        Method that contains logic for what happens inside the training loop.

        This method will be called after ``train_dataloader``.

        """
        loss = self._calculate_loss(batch)

        # add logging
        self.log(name='train_loss_step', value=loss)

        return loss

    def training_epoch_end(
        self,
        outputs: Union[List[float], List[List[float]]],
    ) -> None:
        """
        Method that contains a callback for logic to run after the training epoch ends.

        This method will be called after ``training_step``.

        """
        self.hparams.n_epochs_completed_ += 1

        try:
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        except TypeError:
            # with multiple optimizers, ``outputs`` has a list within a list
            avg_loss = torch.stack([x['loss'] for x in outputs[0]]).mean()

        self.log(name='train_loss_epoch', value=avg_loss)

    def val_dataloader(self) -> Union[InteractionsDataLoader,
                                      ApproximateNegativeSamplingInteractionsDataLoader]:
        """
        Method that sets up validation data as a PyTorch DataLoader.

        This method will be called after ``training_step``.

        """
        return self.val_loader

    def validation_step(
        self,
        batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
    ) -> torch.tensor:
        """
        Method that contains logic for what happens inside the validation loop.

        This method will be called after ``val_dataloader``.

        """
        loss = self._calculate_loss(batch)

        # add logging
        self.log(name='val_loss_step', value=loss)

        return loss

    def validation_epoch_end(
        self,
        outputs: List[float]
    ) -> None:
        """
        Method that contains a callback for logic to run after the validation epoch ends.

        This method will be called after ``validation_step``.

        """
        avg_loss = torch.stack([x for x in outputs]).mean()

        self.log(name='val_loss_epoch', value=avg_loss)

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

        # implicit loss function
        loss = self.loss_function(
            pos_preds,
            neg_preds,
            num_items=self.hparams.num_items,
            positive_items=pos_items,
            negative_items=neg_items,
            metadata=self.hparams.metadata_for_loss,
            metadata_weights=self.hparams.metadata_for_loss_weights,
        )

        return loss

    def get_item_predictions(self,
                             user_id: int = 0,
                             unseen_items_only: bool = True,
                             sort_values: bool = True) -> pd.Series:
        """
        Get predicted rankings/ratings for all items for a given ``user_id``.

        This method cannot be called for datasets stored in ``HDF5InteractionsDataLoader`` since
        data in this ``DataLoader`` is read in dynamically.

        Parameters
        ----------
        user_id: int
        unseen_items_only: bool
            Filter ``preds`` to only show predictions of unseen items not present in the training
            or validation datasets for that ``user_id``
        sort_values: bool
            Whether to sort recommendations by descending prediction probability or not

        Returns
        -------
        preds: pd.Series
            Sorted values as predicted ratings for each item in the dataset with the index being
            the item ID

        """
        user = torch.tensor(
            [user_id] * self.hparams.num_items,
            dtype=torch.long,
            device=self.device
        )
        item = torch.arange(self.hparams.num_items, dtype=torch.long, device=self.device)

        preds = self(user, item)
        preds = preds.detach().cpu()
        preds = pd.Series(preds)
        if sort_values:
            preds = preds.sort_values(ascending=False)

        if unseen_items_only:
            idxs_to_drop = self.train_loader.mat.tocsr()[user_id, :].nonzero()[1]
            if self.val_loader is not None:
                idxs_to_drop = np.concatenate([
                    self.train_loader.mat.tocsr()[user_id, :].nonzero()[1],
                    self.val_loader.mat.tocsr()[user_id, :].nonzero()[1]
                ])
            filtered_preds = preds.drop(idxs_to_drop)

            return filtered_preds
        else:
            return preds

    def item_item_similarity(self, item_id: int) -> pd.Series:
        """
        Get most similar item indices by cosine similarity.

        Cosine similarity is computed with item embeddings from a trained model.

        Parameters
        ----------
        item_id: int

        Returns
        -------
        sim_score_idxs: pd.Series
            Sorted values as cosine similarity for each item in the dataset with the index being
            the item ID

        Note
        ----------
        Returned array is unfiltered, so the first element, being the most similar item, will
        always be the item itself.

        """
        item_embs = self._get_item_embeddings()

        results = [
            np.inner(item_embs[item_id], item_embs[idx]) / (np.linalg.norm(item_embs[idx]) + 1e-11)
            for idx in range(self.hparams.num_items)
        ]

        sim_score_idxs = np.array(results) / np.linalg.norm(item_embs[item_id])

        sim_score_idxs_series = pd.Series(sim_score_idxs)
        sim_score_idxs_series = sim_score_idxs_series.sort_values(ascending=False)

        return sim_score_idxs_series

    def _get_item_embeddings(self) -> np.array:
        """``_get_item_embeddings`` should be implemented in all subclasses."""
        raise NotImplementedError(
            '``BasePipeline`` is meant to be inherited from, not used. '
            '``_get_item_embeddings`` is not implemented in this subclass.'
        )

    def save_model(self, filename: Union[str, Path] = 'model.pth') -> None:
        """
        Save the model's state dictionary and hyperparameters.

        While PyTorch Lightning offers a way to save and load models, there are two main reasons
        for overriding these:

        1) We only want to save the underlying PyTorch model (and not the ``Trainer`` object) so
           we don't have to require PyTorch Lightning as a dependency when deploying a model.

        2) In the v0.8.4 release, loading a model back in leads to a ``RuntimeError`` unable to
           load in weights.

        Parameters
        ----------
        filepath: str or Path
            Filepath for state dictionary to be saved at ending in '.pth'

        """
        dict_to_save = {'state_dict': self.state_dict(), 'hparams': self.hparams}
        torch.save(dict_to_save, str(filename))
