import sys
from typing import Optional, Tuple, Union
import warnings

from pytorch_lightning import Trainer
try:
    from pytorch_lightning.utilities.model_summary import ModelSummary
except ImportError:  # compatible with old ``ModelSummary`` API used in versions prior to ``1.5``
    from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.apply_func import move_data_to_device
import torch
from tqdm.auto import tqdm

from collie.model.base.base_pipeline import BasePipeline
from collie.model.base.layers import MultiLRScheduler, MultiOptimizer


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

    Compared with ``CollieMinimalTrainer``, PyTorch Lightning's ``Trainer`` offers more flexibility
    and room for exploration, at the cost of a higher training time (which is especially true for
    larger models). We recommend starting all model exploration with this ``CollieTrainer``
    (callbacks, automatic Lightning optimizations, etc.), finding a set of hyperparameters that work
    for your training job, then using this in the simpler but faster ``CollieMinimalTrainer``.

    Parameters
    ----------
    model: collie.model.BasePipeline
        Initialized Collie model
    max_epochs: int
        Stop training once this number of epochs is reached
    benchmark: bool
        If set to ``True``, enables ``cudnn.benchmark``
    deterministic: bool
        If set to ``True``, enables ``cudnn.deterministic``
    **kwargs: keyword arguments
        Additional keyword arguments to be sent to the ``Trainer`` class:
        https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api

    Original ``pytorch_lightning.Trainer`` docstring as follows:
    ########
    """
    def __init__(self,
                 model: torch.nn.Module,
                 max_epochs: int = 10,
                 benchmark: bool = True,
                 deterministic: bool = True,
                 **kwargs):
        if not hasattr(model, 'val_loader') or model.val_loader is None:
            print('Did not detect ``val_loader``. Setting ``num_sanity_val_steps`` to 0.')
            kwargs['num_sanity_val_steps'] = 0
            kwargs['check_val_every_n_epoch'] = sys.maxsize

        if kwargs.get('gpus') is None and torch.cuda.is_available():
            print('Detected GPU. Setting ``gpus`` to 1.')
            kwargs['gpus'] = 1

        kwargs['max_epochs'] = max_epochs
        kwargs['benchmark'] = benchmark
        kwargs['deterministic'] = deterministic

        super().__init__(**kwargs)

    __doc__ += Trainer.__init__.__doc__

    @property
    def max_epochs(self):
        """
        Property that just returns ``max_epochs``, included only so we can have
        a setter for it without an ``AttributeError``.

        """
        try:
            return self.fit_loop.max_epochs
        except AttributeError:
            # compatible with old Pytorch Lightning ``Trainer`` API prior to version ``1.4.0``
            return self._max_epochs

    @max_epochs.setter
    def max_epochs(self, value: int):
        """
        Set the ``max_epochs`` attribute to ``value``.

        Parameters
        ----------
        value: int
            Value to set ``max_epochs`` attribute to

        """
        try:
            self.fit_loop.max_epochs = value
        except AttributeError:
            # compatible with old Pytorch Lightning ``Trainer`` API prior to version ``1.4.0``
            self._max_epochs = value


class CollieMinimalTrainer():
    """
    A more manual implementation of PyTorch Lightning's ``Trainer`` class, attempting to port over
    the most commonly used ``Trainer`` arguments into a training loop with more transparency and
    faster training times.

    Through extensive experimentation, we found that PyTorch Lightning's ``Trainer`` was training
    Collie models about 25% slower than the more manual, typical PyTorch training loop boilerplate.
    Thus, we created the ``CollieMinimalTrainer``, which shares a similar API to PyTorch Lightning's
    ``Trainer`` object (both in instantiation and in usage), with a standard PyTorch training loop
    in its place.

    While PyTorch Lightning's ``Trainer`` offers more flexibility and customization through the
    addition of the additional ``Trainer`` arguments and ``callbacks``, we designed this class as a
    way to train a model in production, where we might be more focused on faster training times and
    less on hyperparameter tuning and R&D, where one might instead opt to use PyTorch Lightning's
    ``Trainer`` class.

    Note that the arguments the ``CollieMinimalTrainer`` trainer accepts will be slightly different
    than the ones that the ``CollieTrainer`` accept, and defaults are also not guaranteed to be
    equal as the two libraries evolve. Notable changes are:

    * If ``gpus > 1``, only a single GPU will be used and any other GPUs will remain unused. Multi-
      GPU training is not supported in ``CollieMinimalTrainer`` at this time.

    * ``logger == True`` has no meaning in ``CollieMinimalTrainer`` - a default logger will NOT be
      created if set to ``True``.

    * There is no way to pass in ``callbacks`` at this time. Instead, we will implement the most
      used ones during training here, manually, in favor of greater speed over customization.
      To use early stopping, set the ``early_stopping_patience`` to an integer other than ``None``.

    .. code-block:: python

        from collie.model import CollieMinimalTrainer, MatrixFactorizationModel


        # notice how similar the usage is to the standard ``CollieTrainer``
        model = MatrixFactorizationModel(train=train)
        trainer = CollieMinimalTrainer(model)
        trainer.fit(model)

    Model results should NOT be significantly different whether trained with ``CollieTrainer`` or
    ``CollieMinimalTrainer``.

    If there's an argument you would like to see added to ``CollieMinimalTrainer`` that is present
    in ``CollieTrainer`` used during productionalized model training, make an Issue or a PR in
    GitHub!

    Parameters
    ----------
    model: collie.model.BasePipeline
        Initialized Collie model
    max_epochs: int
        Stop training once this number of epochs is reached
    gpus: bool or int
        Whether to train on the GPU (``gpus == True`` or ``gpus > 0``) or the CPU
    logger: LightningLoggerBase
        Logger for experiment tracking. Set ``logger = None`` or ``logger = False`` to disable
        logging
    early_stopping_patience: int
        Number of epochs of patience to have without any improvement in loss before stopping
        training early. Validation epoch loss will be used if there is a validation DataLoader
        present, else training epoch loss will be used. Set ``early_stopping_patience = None`` or
        ``early_stopping_patience = False`` to disable early stopping
    log_every_n_steps: int
        How often to log within steps, if ``logger`` is enabled
    flush_logs_every_n_steps: int
        How often to flush logs to disk, if ``logger`` is enabled
    enable_model_summary: bool
        Whether to enable or disable the model summarization
    weights_summary: str
        Deprecated, replaced with ``enable_model_summary``. Prints summary of the weights when
        training begins
    detect_anomaly: bool
        Context-manager that enable anomaly detection for the autograd engine. This does two things:

        * Running the forward pass with detection enabled will allow the backward pass to print the
          traceback of the forward operation that created the failing backward function.

        * Any backward computation that generate “nan” value will raise an error.

        Warning: This mode should be enabled only for debugging as the different tests will slow
        down your program execution.
    terminate_on_nan: bool
        Deprecated, replaced with ``detect_anomaly``. If set to ``True``, will terminate training
        (by raising a ``ValueError``) at the end of each training batch, if any of the parameters
        or the loss are NaN or +/- infinity
    benchmark: bool
        If set to ``True``, enables ``cudnn.benchmark``
    deterministic: bool
        If set to ``True``, enables ``cudnn.deterministic``
    progress_bar_refresh_rate: int
        How often to refresh progress bar (in steps), if ``verbosity > 0``
    verbosity: Union[bool, int]
        How verbose to be in training.

        * ``0`` disables all printouts, including ``weights_summary``

        * ``1`` prints ``weights_summary`` (if applicable) and epoch losses

        * ``2`` prints ``weights_summary`` (if applicable), epoch losses, and progress bars

    """
    def __init__(self,
                 model: BasePipeline,
                 max_epochs: int = 10,
                 gpus: Optional[Union[bool, int]] = None,
                 logger: Optional[LightningLoggerBase] = None,
                 early_stopping_patience: Optional[int] = 3,
                 log_every_n_steps: int = 50,
                 flush_logs_every_n_steps: int = 100,
                 enable_model_summary: bool = True,
                 weights_summary: Optional[str] = None,
                 detect_anomaly: bool = False,
                 terminate_on_nan: Optional[bool] = None,
                 benchmark: bool = True,
                 deterministic: bool = True,
                 progress_bar_refresh_rate: Optional[int] = None,
                 verbosity: Union[bool, int] = True):
        # some light argument validation before saving as class-level attributes
        if gpus is None and torch.cuda.is_available():
            print('Detected GPU. Setting ``gpus`` to 1.')
            gpus = 1

        if logger is False:
            logger = None

        if early_stopping_patience is False:
            early_stopping_patience = None

        if verbosity is True:
            verbosity = 2
        elif verbosity is False:
            verbosity = 0

        if weights_summary is not None:
            warnings.warn(
                '``weights_summary`` is deprecated and is replaced with ``enable_model_summary``.',
                DeprecationWarning
            )

        if terminate_on_nan is not None:
            warnings.warn(
                '``terminate_on_nan`` is deprecated and is replaced with ``detect_anomaly``.',
                DeprecationWarning
            )
            if detect_anomaly is False:
                detect_anomaly = terminate_on_nan

        self.max_epochs = max_epochs
        self.gpus = gpus
        self.benchmark = benchmark
        self.deterministic = deterministic
        self.logger = logger
        self.early_stopping_patience = early_stopping_patience
        self.log_every_n_steps = log_every_n_steps
        self.flush_logs_every_n_steps = flush_logs_every_n_steps
        self.enable_model_summary = enable_model_summary
        self.weights_summary = weights_summary
        self.detect_anomaly = detect_anomaly
        self.terminate_on_nan = terminate_on_nan
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.verbosity = verbosity

        self.best_epoch_loss = (0, sys.maxsize)
        self.train_steps = 0
        self.val_steps = 0
        self.num_epochs_completed = 0

        if self.gpus is None or self.gpus is False or self.gpus == 0:
            self.device = 'cpu'
        else:
            self.device = 'cuda'

        torch.backends.cudnn.benchmark = self.benchmark
        torch.backends.cudnn.deterministic = self.deterministic

    @property
    def max_epochs(self):
        """
        Property that just returns ``max_epochs``, included only so we can have
        a setter for it without an ``AttributeError``.

        """
        return self._max_epochs

    @max_epochs.setter
    def max_epochs(self, value: int):
        """
        Set the ``max_epochs`` attribute to ``value``.

        Parameters
        ----------
        value: int
            Value to set ``max_epochs`` attribute to

        """
        self._max_epochs = value

    def fit(self, model: BasePipeline) -> None:
        """
        Runs the full optimization routine.

        Parameters
        ----------
        model: collie.model.BasePipeline
            Initialized Collie model

        """
        if (
            not hasattr(self, 'first_run_pre_training_setup_complete_')
            or not self.first_run_pre_training_setup_complete_
        ):
            self._pre_training_setup(model=model)
            self.first_run_pre_training_setup_complete_ = True

        self._initialize_optimizers_and_lr_schedulers(model=model)

        with torch.autograd.set_detect_anomaly(self.detect_anomaly):
            self._fit(model)

    def _fit(self, model: BasePipeline) -> None:
        # set up top-level epoch progress bar
        epoch_iterator = range(self.num_epochs_completed + 1, self.max_epochs + 1)
        if self.verbosity >= 2:
            epoch_iterator = tqdm(epoch_iterator,
                                  position=0,
                                  unit='epoch',
                                  desc='',
                                  miniters=self.progress_bar_refresh_rate)

        for epoch in epoch_iterator:
            # run the training loop
            model.train()
            train_loss = self._train_loop_single_epoch(model, epoch)
            model.eval()

            epoch_summary = f'Epoch {epoch: >5}: train loss: {train_loss :<1.5f}'
            early_stop_loss = train_loss

            # save epoch loss metrics to the logger
            if self.logger is not None:
                self.logger.log_metrics(metrics={'train_loss_epoch': train_loss}, step=epoch)

            # run the validation loop logic, if we have the ``val_dataloader`` to do so
            if self.val_dataloader is not None:
                val_loss = self._val_loop_single_epoch(model)
                epoch_summary += f', val loss: {val_loss :<1.5f}'
                early_stop_loss = val_loss

                if self.logger is not None:
                    self.logger.log_metrics(metrics={'val_loss_epoch': val_loss}, step=epoch)

            # write out to disk only a single time at the end of the epoch
            if self.logger is not None:
                self.logger.save()

            if self.verbosity >= 1:
                print(epoch_summary)

            model.hparams.num_epochs_completed += 1
            self.num_epochs_completed += 1

            # early stopping logic
            if (
                self.early_stopping_patience is not None
                and early_stop_loss >= self.best_epoch_loss[1]
                and epoch >= (self.early_stopping_patience + self.best_epoch_loss[0])
            ):
                print(f'Epoch {epoch :>5}: Early stopping activated.')
                self._finalize_training()
                return

            # save best loss stats for future early stopping logic
            if early_stop_loss < self.best_epoch_loss[1]:
                self.best_epoch_loss = (epoch, early_stop_loss)

            # learning rate scheduler stepping, if applicable
            if self.lr_scheduler is not None:
                try:
                    # used for most learning rate schedulers
                    self.lr_scheduler.step()
                except TypeError:
                    # used for ``ReduceLROnPlateau``
                    self.lr_scheduler.step(early_stop_loss)

        # run final logging things when training is complete before returning
        self._finalize_training()

    def _pre_training_setup(self, model: BasePipeline) -> None:
        """Set up DataLoaders, optimizers, learning rate schedulers, etc. before training starts."""
        self.train_dataloader = model.train_dataloader()
        self.val_dataloader = model.val_dataloader()

        if self.verbosity != 0 and (
            self.weights_summary is not None or self.enable_model_summary is True
        ):
            try:
                print(ModelSummary(model, max_depth=int(self.enable_model_summary)))
            except TypeError:
                # compatible with old ``ModelSummary`` API used in versions prior to ``1.6``
                print(ModelSummary(model, mode=self.weights_summary))

        # log model hyperparameters, if applicable
        if self.logger is not None:
            self.logger.log_hyperparams(model.hparams)
            self.logger.save()

        # move the model over to the device
        model.to(self.device)
        model._move_any_external_data_to_device()

    def _initialize_optimizers_and_lr_schedulers(self, model: BasePipeline) -> None:
        self.lr_scheduler = None
        configure_optimizers_return_value = model.configure_optimizers()
        if isinstance(configure_optimizers_return_value, tuple):
            # we have a list of optimizers and a list of lr_schedulers dictionaries
            optimizers, lr_schedulers = configure_optimizers_return_value
            self.optimizer = MultiOptimizer(optimizers)
            self.lr_scheduler = MultiLRScheduler(lr_schedulers)
        elif isinstance(configure_optimizers_return_value, list):
            # we have a list of optimizers
            self.optimizer = MultiOptimizer(configure_optimizers_return_value)
        elif isinstance(configure_optimizers_return_value, torch.optim.Optimizer):
            # we have a single optimizer
            self.optimizer = MultiOptimizer([configure_optimizers_return_value])
        else:
            # we have something we've never seen before
            raise ValueError('Unexpected output from ``model.configure_optimizers()``!')

    def _train_loop_single_epoch(self, model: torch.nn.Module, epoch: int) -> float:
        """Training loop for a single epoch, where gradients are optimized for."""
        total_loss = 0

        train_dataloader_iterator = enumerate(self.train_dataloader)
        if self.verbosity >= 2:
            train_dataloader_iterator = tqdm(train_dataloader_iterator,
                                             total=len(self.train_dataloader),
                                             unit='step',
                                             desc=f'({epoch :^5})',
                                             leave=False,
                                             miniters=self.progress_bar_refresh_rate)

        for batch_idx, batch in train_dataloader_iterator:
            self.optimizer.zero_grad()

            batch = self._move_batch_to_device(batch)
            loss = model.calculate_loss(batch)
            loss.backward()

            for optimizer_idx, optimizer in enumerate(self.optimizer.optimizers):
                model.optimizer_step(epoch=epoch,
                                     batch_idx=batch_idx,
                                     optimizer=optimizer,
                                     optimizer_idx=optimizer_idx,
                                     optimizer_closure=None)

            self.train_steps += 1

            detached_loss = loss.detach()
            total_loss += detached_loss

            if self.verbosity >= 2:
                train_dataloader_iterator.set_postfix(train_loss=detached_loss.item())

            self._log_step(name='train',
                           steps=self.train_steps,
                           total_loss=total_loss,
                           batch_idx=batch_idx)

        return (total_loss / len(self.train_dataloader)).item()

    def _val_loop_single_epoch(self, model: torch.nn.Module) -> float:
        """Validation loop for a single epoch, where gradients are NOT optimized for."""
        total_loss = 0

        for batch_idx, batch in enumerate(self.val_dataloader):
            batch = self._move_batch_to_device(batch)
            loss = model.calculate_loss(batch)

            self.val_steps += 1

            total_loss += loss.detach()

            self._log_step(name='val',
                           steps=self.val_steps,
                           total_loss=total_loss,
                           batch_idx=batch_idx)

        return (total_loss / len(self.val_dataloader)).item()

    def _move_batch_to_device(
        self, batch: Tuple[Tuple[torch.tensor, torch.tensor], torch.tensor],
    ) -> Tuple[Tuple[torch.tensor, torch.tensor], torch.tensor]:
        """Move a batch of data to the proper device."""
        # TODO: does this actually speed anything up?
        try:
            # assume we have implicit data
            ((users, pos_items), neg_items) = batch

            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            return ((users, pos_items), neg_items)
        except (AttributeError, ValueError):
            try:
                # now assume we have explicit data
                users, pos_items, ratings = batch

                users = users.to(self.device)
                pos_items = pos_items.to(self.device)
                ratings = ratings.to(self.device)

                return users, pos_items, ratings
            except (AttributeError, ValueError):
                # we have an unexpected data format, fallback to PyTorch Lightning
                return move_data_to_device(batch, self.device)

    def _log_step(self, name: str, steps: int, total_loss: torch.tensor, batch_idx: int) -> None:
        """Check if we should and, if so, log step-loss metrics to our logger."""
        if self.logger is not None:
            if steps % self.log_every_n_steps == 0:
                batch_loss = (total_loss / (batch_idx + 1)).item()
                self.logger.log_metrics(metrics={f'{name}_loss_step': batch_loss}, step=steps)
            if steps % self.flush_logs_every_n_steps == 0:
                self.logger.save()

    def _finalize_training(self) -> None:
        """Finalize logging results before returning."""
        if self.logger is not None:
            self.logger.save()
            self.logger.finalize(status='FINISHED')
