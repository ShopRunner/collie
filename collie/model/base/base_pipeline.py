from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from pathlib import Path
import textwrap
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from pytorch_lightning.core.lightning import LightningModule
import torch

from collie.interactions import (ApproximateNegativeSamplingInteractionsDataLoader,
                                 ExplicitInteractions,
                                 Interactions,
                                 InteractionsDataLoader)
from collie.loss import (adaptive_bpr_loss,
                         adaptive_hinge_loss,
                         bpr_loss,
                         hinge_loss,
                         warp_loss)
from collie.utils import get_init_arguments


INTERACTIONS_LIKE_INPUT = Union[ApproximateNegativeSamplingInteractionsDataLoader,
                                ExplicitInteractions,
                                Interactions,
                                InteractionsDataLoader]
EXPECTED_BATCH_TYPE = Union[Tuple[Tuple[torch.tensor, torch.tensor], torch.tensor],
                            Tuple[torch.tensor, torch.tensor, torch.tensor]]


class BasePipeline(LightningModule, metaclass=ABCMeta):
    """
    Base Pipeline model architectures to inherit from.

    All subclasses MUST at least override the following methods:

    * ``_setup_model`` - Set up the model architecture

    * ``forward`` - Forward pass through a model

    For ``item_item_similarity`` to work properly, all subclasses are should also implement:

    * ``_get_item_embeddings`` - Returns item embeddings from the model on the device

    For ``user_user_similarity`` to work properly, all subclasses are should also implement:

    * ``_get_user_embeddings`` - Returns user embeddings from the model on the device

    Parameters
    ----------
    train: ``collie.interactions`` object
        Data loader for training data. If an ``Interactions`` object is supplied, an
        ``InteractionsDataLoader`` will automatically be instantiated with ``shuffle=True``
    val: ``collie.interactions`` object
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

        * ``'adagrad'`` (for ``torch.optim.Adagrad``)

        * ``'adam'`` (for ``torch.optim.Adam``)

        * ``'sparse_adam'`` (for ``torch.optim.SparseAdam``)

    loss: function or str
        If a string, one of the following implemented losses:

        * ``'bpr'`` / ``'adaptive_bpr'`` (implicit data)

        * ``'hinge'`` / ``'adaptive_hinge'`` (implicit data)

        * ``'warp'`` (implicit data)

        * ``'mse'`` (explicit data)

        * ``'mae'`` (explicit data)

        For implicit data, if ``train.num_negative_samples > 1``, the adaptive loss version will
        automatically be used of the losses above (except for WARP loss, which is only adaptive by
        nature).

        If a callable is passed, that function will be used for calculating the loss. For implicit
        models, the first two arguments passed will be the positive and negative predictions,
        respectively. Additional keyword arguments passed in order are ``num_items``,
        ``positive_items``, ``negative_items``, ``metadata``, and ``metadata_weights``.
        For explicit models, the only two arguments passed in will be the prediction and actual
        rating values, in order.
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
        To load a previously-saved model for inference, pass in path to output of
        ``model.save_model()`` method. Note that datasets and optimizers will NOT be restored.
        If ``None``, will initialize model as normal
    map_location: str or torch.device
        If ``load_model_path`` is provided, device specifying how to remap storage locations when
        ``torch.load``-ing the state dictionary
    **kwargs: keyword arguments
        All keyword arguments will be saved to ``self.hparams`` by default

    """
    def __init__(self,
                 train: INTERACTIONS_LIKE_INPUT = None,
                 val: INTERACTIONS_LIKE_INPUT = None,
                 lr: float = 1e-3,
                 lr_scheduler_func: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 weight_decay: float = 0.0,
                 optimizer: Union[str, torch.optim.Optimizer] = 'adam',
                 loss: Union[str, Callable[..., torch.tensor]] = 'hinge',
                 metadata_for_loss: Optional[Dict[str, torch.tensor]] = None,
                 metadata_for_loss_weights: Optional[Dict[str, float]] = None,
                 load_model_path: Optional[str] = None,
                 map_location: Optional[str] = None,
                 **kwargs):
        if isinstance(train, Interactions) or isinstance(train, ExplicitInteractions):
            train = InteractionsDataLoader(interactions=train, shuffle=True)
        if isinstance(val, Interactions) or isinstance(val, ExplicitInteractions):
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
            if self.train_loader is None:
                raise TypeError('``train`` must be provided to all newly-instantiated models!')
            elif self.val_loader is not None:
                assert self.train_loader.num_users == self.val_loader.num_users, (
                    'Both training and val ``num_users`` must equal: '
                    f'{self.train_loader.num_users} != {self.val_loader.num_users}.'
                )
                assert self.train_loader.num_items == self.val_loader.num_items, (
                    'Both training and val ``num_items`` must equal: '
                    f'{self.train_loader.num_items} != {self.val_loader.num_items}.'
                )

                if (
                    hasattr(self.train_loader, 'num_negative_samples')
                    or hasattr(self.val_loader, 'num_negative_samples')
                ):
                    num_negative_samples_error = (
                        'Training and val ``num_negative_samples`` property must both equal ``1``'
                        f' or both be greater than ``1``, not: {self.train_loader.num_items} and'
                        f' {self.val_loader.num_items}, respectively.'
                    )
                    if self.train_loader.num_negative_samples == 1:
                        assert self.val_loader.num_negative_samples == 1, num_negative_samples_error
                    elif self.train_loader.num_negative_samples > 1:
                        assert self.val_loader.num_negative_samples > 1, num_negative_samples_error
                    else:
                        raise ValueError(
                            '``self.train_loader.num_negative_samples`` must be greater than ``0``,'
                            f' not {self.train_loader.num_negative_samples}.'
                        )

            # saves all passed-in parameters
            init_args = get_init_arguments(
                exclude=['train', 'val', 'item_metadata', 'trained_model'],
                verbose=False,
            )

            self.save_hyperparameters(init_args, *kwargs.keys())

            self.hparams.num_users = self.train_loader.num_users
            self.hparams.num_items = self.train_loader.num_items
            self.hparams.num_epochs_completed = 0

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

    def _move_any_external_data_to_device(self):
        """Code for ensuring all side-data is put onto the model's device before training."""
        pass

    def _configure_loss(self) -> None:
        # set up loss function
        self.loss_function = None

        if callable(self.loss):
            self.loss_function = self.loss
            return

        # explicit losses first
        self.hparams._is_implicit = False
        if self.loss == 'mse':
            self.loss_function = torch.nn.MSELoss(reduction='mean')
            return
        elif self.loss == 'mae':
            self.loss_function = torch.nn.L1Loss(reduction='mean')
            return

        # followed by implicit losses
        self.hparams._is_implicit = True
        if not hasattr(self.train_loader, 'num_negative_samples'):
            raise ValueError(
                '``num_negative_samples`` attribute not found in ``train_loader`` - are you using '
                'explicit data with an implicit loss function?'
            )
        elif self.loss == 'warp':
            if self.train_loader.num_negative_samples > 1:
                self.loss_function = warp_loss
                return
            else:
                raise ValueError('Cannot use WARP loss with a single negative sample!')
        elif 'bpr' in self.loss:
            if self.train_loader.num_negative_samples > 1:
                self.loss_function = adaptive_bpr_loss
            else:
                if 'adaptive' in self.loss:
                    warnings.warn(
                        textwrap.dedent(
                            '''
                            Adaptive BPR loss specified, but ``num_negative_samples`` == 1. Using
                            standard BPR loss instead.
                            '''
                        ).replace('\n', ' ').strip()
                    )

                self.loss_function = bpr_loss
            return
        elif 'hinge' in self.loss or self.loss == 'adaptive':
            if self.train_loader.num_negative_samples > 1:
                self.loss_function = adaptive_hinge_loss
            else:
                if 'adaptive' in self.loss:
                    warnings.warn(
                        textwrap.dedent(
                            '''
                            Adaptive hinge loss specified, but ``num_negative_samples`` == 1. Using
                            standard hinge loss instead.
                            '''
                        ).replace('\n', ' ').strip()
                    )

                self.loss_function = hinge_loss
            return
        else:
            raise ValueError('{} is not a valid loss function.'.format(self.loss))

    def configure_optimizers(self) -> (
        Union[
            Tuple[List[torch.optim.Optimizer], List[torch.optim.Optimizer]],
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]
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
        # since this is the only function that is called before each ``trainer.fit`` call, we will
        # also take this time to ensure any external data a model might rely on has been properly
        # moved to the device before training
        self._move_any_external_data_to_device()

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

            return (optimizer_list, scheduler_list)

        if self.bias_optimizer is not None:
            return [optimizer, bias_optimizer]
        else:
            return optimizer

    def _get_optimizer(self,
                       optimizer: Optional[Union[str, torch.optim.Optimizer]],
                       **kwargs) -> torch.optim.Optimizer:
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
        elif optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(
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

    def train_dataloader(self) -> Union[ApproximateNegativeSamplingInteractionsDataLoader,
                                        InteractionsDataLoader]:
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
        loss = self.calculate_loss(batch)

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
        self.hparams.num_epochs_completed += 1

        try:
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        except TypeError:
            # with multiple optimizers, ``outputs`` has a list within a list
            avg_loss = torch.stack([x['loss'] for x in outputs[0]]).mean()

        self.log(name='train_loss_epoch', value=avg_loss)

    def val_dataloader(self) -> Union[ApproximateNegativeSamplingInteractionsDataLoader,
                                      InteractionsDataLoader]:
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
        loss = self.calculate_loss(batch)

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

    def calculate_loss(self, batch: EXPECTED_BATCH_TYPE) -> torch.tensor:
        """
        Given a batch of data, calculate the loss value.

        Note that the data type (implicit or explicit) will be determined by the structure of the
        batch sent to this method. See the table below for expected data types:

        .. list-table::
            :header-rows: 1

            * - ``__getitem__`` Format
              - Expected Meaning
              - Model Type
            * - ``((X, Y), Z)``
              - ``((user IDs, item IDs), negative item IDs)``
              - **Implicit**
            * - ``(X, Y, Z)``
              - ``(user IDs, item IDs, ratings)``
              - **Explicit**

        """
        if len(batch) == 2 and isinstance(batch[0], Iterable) and len(batch[0]) == 2:
            if self.hparams.get('_is_implicit') is False:
                raise ValueError('Explicit loss with implicit data is invalid!')

            # implicit data
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
        elif len(batch) == 3:
            if self.hparams.get('_is_implicit') is True:
                raise ValueError('Implicit loss with explicit data is invalid!')

            # explicit data
            (users, items, ratings) = batch

            users = users.long()
            items = items.long()
            ratings = ratings.float()

            # get predictions from model
            preds = self(users, items)

            # explicit loss function
            loss = self.loss_function(preds, ratings)
        else:
            raise ValueError(f'Unexpected format for batch: {batch}. See docs for expected format.')

        return loss

    def get_item_predictions(self,
                             user_id: int = 0,
                             unseen_items_only: bool = False,
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
            or validation datasets for that ``user_id``. Note this requires both ``train_loader``
            and ``val_loader`` to be 1) class-level attributes in the model and 2) DataLoaders with
            ``Interactions`` at its core (not ``HDF5Interactions``). If you are loading in a model,
            these two attributes will need to be set manually, since datasets are NOT saved when
            saving the model
        sort_values: bool
            Whether to sort recommendations by descending prediction probability or not

        Returns
        -------
        preds: pd.Series
            Sorted values as predicted ratings for each item in the dataset with the index being
            the item ID

        """
        if user_id >= self.hparams.num_users:
            raise ValueError(
                f'``user_id`` {user_id} is not in the model. '
                'Expected ID between ``0`` and ``self.hparams.num_users - 1`` '
                f'(``{self.hparams.num_users - 1}``), not ``{user_id}``'
            )

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
            if self.val_loader is not None:
                idxs_to_drop = np.concatenate([
                    self.train_loader.mat.tocsr()[user_id, :].nonzero()[1],
                    self.val_loader.mat.tocsr()[user_id, :].nonzero()[1]
                ])
            else:
                idxs_to_drop = self.train_loader.mat.tocsr()[user_id, :].nonzero()[1]
            filtered_preds = preds.drop(idxs_to_drop)

            return filtered_preds
        else:
            return preds

    def get_user_predictions(self,
                             item_id: int = 0,
                             unseen_users_only: bool = False,
                             sort_values: bool = True) -> pd.Series:
        """
        User counterpart to ``get_item_predictions`` method.

        Get predicted rankings/ratings for all users for a given ``item_id``.

        This method cannot be called for datasets stored in ``HDF5InteractionsDataLoader`` since
        data in this ``DataLoader`` is read in dynamically.

        Parameters
        ----------
        item_id: int
        unseen_users_only: bool
            Filter ``preds`` to only show predictions of unseen users not present in the training
            or validation datasets for that ``item_id``. Note this requires both ``train_loader``
            and ``val_loader`` to be 1) class-level attributes in the model and 2) DataLoaders with
            ``Interactions`` at its core (not ``HDF5Interactions``). If you are loading in a model,
            these two attributes will need to be set manually, since datasets are NOT saved when
            saving the model
        sort_values: bool
            Whether to sort recommendations by descending prediction probability

        Returns
        -------
        preds: pd.Series
            Sorted values as predicted ratings for each user in the dataset with the index being
            the user ID
        """
        if item_id >= self.hparams.num_items:
            raise ValueError(
                f'``item_id`` {item_id} is not in the model. '
                'Expected ID between ``0`` and ``self.hparams.num_items - 1`` '
                f'(``{self.hparams.num_items - 1}``), not ``{item_id}``'
            )

        item = torch.tensor(
            [item_id] * self.hparams.num_users,
            dtype=torch.long,
            device=self.device
        )
        user = torch.arange(self.hparams.num_users, dtype=torch.long, device=self.device)

        preds = self(user, item)
        preds = preds.detach().cpu()
        preds = pd.Series(preds)
        if sort_values:
            preds = preds.sort_values(ascending=False)

        if unseen_users_only:
            if self.val_loader is not None:
                idxs_to_drop = np.concatenate([
                    self.train_loader.mat.tocsr()[:, item_id].nonzero()[1],
                    self.val_loader.mat.tocsr()[:, item_id].nonzero()[1]
                ])
            else:
                idxs_to_drop = self.train_loader.mat.tocsr()[:, item_id].nonzero()[1]
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
        ----
        Returned array is unfiltered, so the first element, being the most similar item, will
        always be the item itself.

        """
        if item_id >= self.hparams.num_items:
            raise ValueError(
                f'``item_id`` {item_id} is not in the model. '
                'Expected ID between ``0`` and ``self.hparams.num_items - 1`` '
                f'(``{self.hparams.num_items - 1}``), not ``{item_id}``'
            )

        return self._calculate_embedding_similarity(
            embeddings=self._get_item_embeddings(),
            id=item_id
        )

    def user_user_similarity(self, user_id: int) -> pd.Series:
        """
        User counterpart to ``item_item_similarity`` method.

        Get most similar user indices by cosine similarity.

        Cosine similarity is computed with user embeddings from a trained model.

        Parameters
        ----------
        user_id: int

        Returns
        -------
        sim_score_idxs: pd.Series
            Sorted values as cosine similarity for each user in the dataset with the index being
            the user ID

        Note
        ----
        Returned array is unfiltered, so the first element, being the most similar user, will
        always be the seed user themself.
        """
        if user_id >= self.hparams.num_users:
            raise ValueError(
                f'``user_id`` {user_id} is not in the model. '
                'Expected ID between ``0`` and ``self.hparams.num_users - 1`` '
                f'(``{self.hparams.num_users - 1}``), not ``{user_id}``'
            )

        return self._calculate_embedding_similarity(
            embeddings=self._get_user_embeddings(),
            id=user_id
        )

    def _calculate_embedding_similarity(self, embeddings: torch.tensor, id: int) -> pd.Series:
        """Get most similar embedding indices by cosine similarity."""
        embeddings = embeddings / embeddings.norm(dim=1)[:, None]

        return pd.Series(
            torch.matmul(embeddings[[id], :], embeddings.transpose(1, 0))
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        ).sort_values(ascending=False)

    def _get_item_embeddings(self) -> torch.tensor:
        """``_get_item_embeddings`` should be implemented in all subclasses."""
        raise NotImplementedError(
            '``BasePipeline`` is meant to be inherited from, not used. '
            '``_get_item_embeddings`` is not implemented in this subclass.'
        )

    def _get_user_embeddings(self) -> torch.tensor:
        """``_get_user_embeddings`` should be implemented in all subclasses."""
        raise NotImplementedError(
            '``BasePipeline`` is meant to be inherited from, not used. '
            '``_get_user_embeddings`` is not implemented in this subclass.'
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
