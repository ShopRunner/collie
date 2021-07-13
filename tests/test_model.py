from contextlib import suppress
import copy
from functools import partial
import os
from unittest import mock

import pandas as pd
import pytest
import pytorch_lightning
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torchmetrics

from collie_recs.interactions import HDF5InteractionsDataLoader, InteractionsDataLoader
from collie_recs.loss import (adaptive_bpr_loss,
                              adaptive_hinge_loss,
                              bpr_loss,
                              hinge_loss,
                              warp_loss)
from collie_recs.metrics import evaluate_in_batches, explicit_evaluate_in_batches, mapk
from collie_recs.model import (BasePipeline,
                               CollieMinimalTrainer,
                               CollieTrainer,
                               DeepFM,
                               HybridPretrainedModel,
                               MatrixFactorizationModel,
                               NeuralCollaborativeFiltering)


def test_CollieTrainer_no_val_data(untrained_implicit_model_no_val_data):
    trainer = CollieTrainer(model=untrained_implicit_model_no_val_data,
                            logger=False,
                            checkpoint_callback=False)

    assert trainer.num_sanity_val_steps == 0
    assert trainer.check_val_every_n_epoch > 1000


@mock.patch('torch.cuda.is_available')
def test_CollieTrainer_no_gpu(is_available_mock, untrained_implicit_model, capfd):
    is_available_mock.return_value = True

    with suppress(AttributeError, pytorch_lightning.utilities.exceptions.MisconfigurationException):
        # if we run this test on CPU, PyTorch Lightning won't let us actually use a GPU, but we
        # can still test if the ``CollieTrainer`` is doing what it should here
        CollieTrainer(model=untrained_implicit_model,
                      logger=False,
                      checkpoint_callback=False)

    out, _ = capfd.readouterr()
    assert 'Detected GPU. Setting ``gpus`` to 1.' in out


@mock.patch('torch.cuda.is_available')
def test_CollieTrainer_on_cpu(is_available_mock, untrained_implicit_model):
    is_available_mock.return_value = True

    with pytest.warns(UserWarning):
        trainer = CollieTrainer(model=untrained_implicit_model,
                                logger=False,
                                checkpoint_callback=False,
                                gpus=0)

    assert trainer.gpus == 0


def test_basepipeline_does_not_initialize(train_val_implicit_data):
    train, val = train_val_implicit_data

    with pytest.raises(TypeError):
        BasePipeline(train=train, val=val)


@pytest.mark.parametrize('change_to_make', ['num_users',
                                            'num_items',
                                            'num_negative_samples',
                                            'bad_train_num_negative_samples'])
def test_mismatched_train_and_val_loaders(train_val_implicit_data, change_to_make):
    train, val = train_val_implicit_data

    train = copy.copy(train)
    val = copy.copy(val)

    expected_error = AssertionError

    if change_to_make == 'num_users':
        train.num_users = 3
        val.num_users = 2
    elif change_to_make == 'num_items':
        train.num_items = 1
        val.num_items = 2
    elif change_to_make == 'num_negative_samples':
        train.num_negative_samples = 1
        val.num_negative_samples = 2
    elif change_to_make == 'bad_train_num_negative_samples':
        train.num_negative_samples = 0
        expected_error = ValueError

    with pytest.raises(expected_error):
        MatrixFactorizationModel(train=train, val=val)


def test_okay_mismatched_train_and_val_loaders(train_val_implicit_data):
    train, val = train_val_implicit_data

    train = copy.copy(train)
    val = copy.copy(val)

    train.num_negative_samples = 2
    val.num_negative_samples = 3

    model = MatrixFactorizationModel(train=train, val=val)
    trainer = CollieTrainer(model=model, logger=False, checkpoint_callback=False, max_steps=1)
    trainer.fit(model)


def test_instantiation_of_model_loss(train_val_implicit_data):
    train, val = train_val_implicit_data

    train = copy.copy(train)
    val = copy.copy(val)

    train.num_negative_samples = 1
    val.num_negative_samples = 1

    model_1 = MatrixFactorizationModel(train=train, val=val, loss='hinge')
    assert model_1.loss_function == hinge_loss

    model_2 = MatrixFactorizationModel(train=train, val=val, loss='bpr')
    assert model_2.loss_function == bpr_loss

    with pytest.raises(ValueError):
        MatrixFactorizationModel(train=train, val=val, loss='warp')

    train.num_negative_samples = 2
    val.num_negative_samples = 2

    model_4 = MatrixFactorizationModel(train=train, val=val, loss='hinge')
    assert model_4.loss_function == adaptive_hinge_loss

    model_5 = MatrixFactorizationModel(train=train, val=val, loss='bpr')
    assert model_5.loss_function == adaptive_bpr_loss

    model_6 = MatrixFactorizationModel(train=train, val=val, loss='warp')
    assert model_6.loss_function == warp_loss

    def custom_loss_function(*args, **kwargs):
        return 42

    model_7 = MatrixFactorizationModel(train=train, val=val, loss=custom_loss_function)
    assert model_7.loss_function == custom_loss_function

    with pytest.raises(ValueError):
        MatrixFactorizationModel(train=train, val=val, loss='nonexistent_loss')


def test_instantiation_of_model_optimizer(train_val_implicit_data):
    train, val = train_val_implicit_data

    model_1 = MatrixFactorizationModel(train=train, val=val, bias_optimizer=None)
    trainer_1 = CollieTrainer(model=model_1, logger=False, checkpoint_callback=False, max_steps=1)
    trainer_1.fit(model_1)
    assert not isinstance(model_1.optimizers(), list)
    model_1_lr_schedulers = [s['scheduler'] for s in trainer_1.lr_schedulers]
    assert len(model_1_lr_schedulers) == 1

    model_2 = MatrixFactorizationModel(train=train,
                                       val=val,
                                       bias_optimizer=None,
                                       lr_scheduler_func=None)
    trainer_2 = CollieTrainer(model=model_2, logger=False, checkpoint_callback=False, max_steps=1)
    trainer_2.fit(model_2)
    assert not isinstance(model_2.optimizers(), list)
    model_2_lr_schedulers = [s['scheduler'] for s in trainer_2.lr_schedulers]
    assert len(model_2_lr_schedulers) == 0

    model_3 = MatrixFactorizationModel(train=train,
                                       val=val,
                                       bias_optimizer='infer',
                                       bias_lr='infer')
    trainer_3 = CollieTrainer(model=model_3, logger=False, checkpoint_callback=False, max_steps=1)
    trainer_3.fit(model_3)
    assert len(model_3.optimizers()) == 2
    assert model_3.bias_optimizer == model_3.optimizer
    assert model_3.hparams.bias_lr == model_3.hparams.lr
    model_3_lr_schedulers = [s['scheduler'] for s in trainer_3.lr_schedulers]
    assert len(model_3_lr_schedulers) == 2

    model_4 = MatrixFactorizationModel(train=train,
                                       val=val,
                                       bias_optimizer='infer',
                                       bias_lr='infer',
                                       lr_scheduler_func=None)
    trainer_4 = CollieTrainer(model=model_4, logger=False, checkpoint_callback=False, max_steps=1)
    trainer_4.fit(model_4)
    assert len(model_4.optimizers()) == 2
    assert model_4.bias_optimizer == model_4.optimizer
    assert model_4.hparams.bias_lr == model_4.hparams.lr
    model_4_lr_schedulers = [s['scheduler'] for s in trainer_4.lr_schedulers]
    assert len(model_4_lr_schedulers) == 0

    model_5 = MatrixFactorizationModel(train=train,
                                       val=val,
                                       bias_optimizer='infer',
                                       bias_lr=10,
                                       lr_scheduler_func=None)
    trainer_5 = CollieTrainer(model=model_5, logger=False, checkpoint_callback=False, max_steps=1)
    trainer_5.fit(model_5)
    assert len(model_5.optimizers()) == 2
    assert model_5.bias_optimizer == model_5.optimizer
    assert model_5.hparams.bias_lr != model_5.hparams.lr
    model_5_lr_schedulers = [s['scheduler'] for s in trainer_5.lr_schedulers]
    assert len(model_5_lr_schedulers) == 0

    model_6 = MatrixFactorizationModel(train=train, val=val, optimizer='fake_optimizer')
    trainer_6 = CollieTrainer(model=model_6, logger=False, checkpoint_callback=False, max_steps=1)
    with pytest.raises(ValueError):
        trainer_6.fit(model_6)

    model_7 = MatrixFactorizationModel(train=train, val=val, bias_optimizer='fake_optimizer')
    trainer_7 = CollieTrainer(model=model_7, logger=False, checkpoint_callback=False, max_steps=1)
    with pytest.raises(ValueError):
        trainer_7.fit(model_7)

    # ``Adadelta`` accepts ``weight_decay`` parameter
    model_8 = MatrixFactorizationModel(train=train, val=val, optimizer=torch.optim.Adadelta)
    trainer_8 = CollieTrainer(model=model_8, logger=False, checkpoint_callback=False, max_steps=1)
    trainer_8.fit(model_8)

    # ``LBFGS`` does not accept ``weight_decay`` parameter
    model_9 = MatrixFactorizationModel(train=train,
                                       val=val,
                                       optimizer=torch.optim.LBFGS,
                                       sparse=True)
    trainer_9 = CollieTrainer(model=model_9, logger=False, checkpoint_callback=False, max_steps=1)
    trainer_9.fit(model_9)


class TestCollieMinimalTrainer():
    @mock.patch('torch.cuda.is_available')
    def test_no_gpu_set_but_gpu_available(is_available_mock, untrained_implicit_model, capfd):
        is_available_mock.return_value = True

        CollieMinimalTrainer(model=untrained_implicit_model, gpus=None)

        out, _ = capfd.readouterr()
        assert 'Detected GPU. Setting ``gpus`` to 1.' in out

    def test_trainer_max_epochs(self, train_val_implicit_sample_data):
        train, _ = train_val_implicit_sample_data
        model = MatrixFactorizationModel(train=train)
        trainer = CollieMinimalTrainer(model=model, max_epochs=2)

        trainer.fit(model)
        assert model.hparams.num_epochs_completed == 2
        assert model.hparams.num_epochs_completed == trainer.num_epochs_completed

        trainer.fit(model)
        assert model.hparams.num_epochs_completed == 2
        assert model.hparams.num_epochs_completed == trainer.num_epochs_completed

        trainer.max_epochs += 1
        trainer.fit(model)
        assert model.hparams.num_epochs_completed == 3
        assert model.hparams.num_epochs_completed == trainer.num_epochs_completed

    @mock.patch.object(CollieMinimalTrainer, '_train_loop_single_epoch')
    def test_early_stopping_train(self,
                                  _train_loop_single_epoch_mock,
                                  train_val_implicit_sample_data,
                                  capfd):
        # our training loss will not decrease, triggering early stopping
        _train_loop_single_epoch_mock.return_value = 100

        train, _ = train_val_implicit_sample_data
        model = MatrixFactorizationModel(train=train)
        trainer = CollieMinimalTrainer(model=model, early_stopping_patience=2)
        trainer.fit(model)

        out, _ = capfd.readouterr()
        assert 'Epoch     3: Early stopping activated.' in out

        assert model.hparams.num_epochs_completed == 3
        assert model.hparams.num_epochs_completed == trainer.num_epochs_completed

    @mock.patch.object(CollieMinimalTrainer, '_val_loop_single_epoch')
    def test_early_stopping_val(self,
                                _val_loop_single_epoch_mock,
                                train_val_implicit_sample_data,
                                capfd):
        # our validation loss will not decrease, triggering early stopping
        _val_loop_single_epoch_mock.return_value = 100

        train, val = train_val_implicit_sample_data
        model = MatrixFactorizationModel(train=train, val=val)
        trainer = CollieMinimalTrainer(model=model, early_stopping_patience=1)
        trainer.fit(model)

        out, _ = capfd.readouterr()
        assert 'Epoch     2: Early stopping activated.' in out

        assert model.hparams.num_epochs_completed == 2
        assert model.hparams.num_epochs_completed == trainer.num_epochs_completed

    def test_logging_log_every_n_steps(self, train_val_implicit_sample_data):
        class SimpleLogger(pytorch_lightning.loggers.LightningLoggerBase):
            """A simple logger that just saves all logs as class-level attributes."""
            def __init__(self):
                super().__init__()
                self.metrics = list()
                self.steps = 0
                self.saved_count = 0
                self.val_in_metrics = False

            @property
            def name(self):
                return 'MyLogger'

            @property
            @rank_zero_experiment
            def experiment(self):
                pass

            @property
            def version(self):
                return '0.0'

            @rank_zero_only
            def log_hyperparams(self, params):
                self.hparams = params

            @rank_zero_only
            def log_metrics(self, metrics, step):
                self.metrics.append(metrics)
                self.steps += 1

            @rank_zero_only
            def save(self):
                self.saved_count += 1

            @rank_zero_only
            def finalize(self, status):
                self.finalized_ = status

        simple_logger_more_verbose = SimpleLogger()
        simple_logger_less_verbose = SimpleLogger()

        train, val = train_val_implicit_sample_data

        model_more_verbose = MatrixFactorizationModel(train=train, val=val)
        trainer_more_verbose = CollieMinimalTrainer(model=model_more_verbose,
                                                    max_epochs=1,
                                                    logger=simple_logger_more_verbose,
                                                    log_every_n_steps=1)
        trainer_more_verbose.fit(model_more_verbose)

        model_less_verbose = MatrixFactorizationModel(train=train, val=val)
        trainer_less_verbose = CollieMinimalTrainer(model=model_less_verbose,
                                                    max_epochs=1,
                                                    logger=simple_logger_less_verbose,
                                                    log_every_n_steps=3)
        trainer_less_verbose.fit(model_less_verbose)

        assert (
            simple_logger_more_verbose.hparams
            == simple_logger_less_verbose.hparams
            == model_more_verbose.hparams
            == model_less_verbose.hparams
        )
        assert simple_logger_more_verbose.saved_count == simple_logger_less_verbose.saved_count
        assert simple_logger_more_verbose.finalized_ == simple_logger_less_verbose.finalized_
        assert len(simple_logger_more_verbose.metrics) == simple_logger_more_verbose.steps
        assert len(simple_logger_less_verbose.metrics) == simple_logger_less_verbose.steps

        assert len(simple_logger_more_verbose.metrics) > len(simple_logger_less_verbose.metrics)

    def test_logging_flush_logs_every_n_steps(self, train_val_implicit_sample_data):
        class SimpleLogger(pytorch_lightning.loggers.LightningLoggerBase):
            """A simple logger that just saves all logs as class-level attributes."""
            def __init__(self):
                super().__init__()
                self.metrics = list()
                self.steps = 0
                self.saved_count = 0
                self.val_in_metrics = False

            @property
            def name(self):
                return 'MyLogger'

            @property
            @rank_zero_experiment
            def experiment(self):
                pass

            @property
            def version(self):
                return '0.0'

            @rank_zero_only
            def log_hyperparams(self, params):
                self.hparams = params

            @rank_zero_only
            def log_metrics(self, metrics, step):
                self.metrics.append(metrics)
                self.steps += 1

            @rank_zero_only
            def save(self):
                self.saved_count += 1

            @rank_zero_only
            def finalize(self, status):
                self.finalized_ = status

        simple_logger_more_saves = SimpleLogger()
        simple_logger_less_saves = SimpleLogger()

        train, val = train_val_implicit_sample_data

        model_more_saves = MatrixFactorizationModel(train=train, val=val)
        trainer_more_saves = CollieMinimalTrainer(model=model_more_saves,
                                                  max_epochs=1,
                                                  logger=simple_logger_more_saves,
                                                  flush_logs_every_n_steps=1)
        trainer_more_saves.fit(model_more_saves)

        model_less_saves = MatrixFactorizationModel(train=train, val=val)
        trainer_less_saves = CollieMinimalTrainer(model=model_less_saves,
                                                  max_epochs=1,
                                                  logger=simple_logger_less_saves,
                                                  flush_logs_every_n_steps=3)
        trainer_less_saves.fit(model_less_saves)

        assert (
            simple_logger_more_saves.hparams
            == simple_logger_less_saves.hparams
            == model_more_saves.hparams
            == model_less_saves.hparams
        )
        assert simple_logger_more_saves.finalized_ == simple_logger_less_saves.finalized_
        assert (
            len(simple_logger_more_saves.metrics)
            == len(simple_logger_less_saves.metrics)
            == simple_logger_more_saves.steps
            == simple_logger_less_saves.steps
        )

        assert simple_logger_more_saves.saved_count > simple_logger_less_saves.saved_count

    @mock.patch.object(MatrixFactorizationModel, 'calculate_loss')
    def test_terminate_on_nan(self,
                              calculate_loss_mock,
                              train_val_implicit_sample_data):
        calculate_loss_mock.return_value = torch.tensor(float('nan')).requires_grad_()

        train, val = train_val_implicit_sample_data
        model = MatrixFactorizationModel(train=train, val=val)
        trainer = CollieMinimalTrainer(model=model, terminate_on_nan=True)

        with pytest.raises(ValueError):
            trainer.fit(model)

    def test_multiple_optimizers_and_lr_schedulers(self, train_val_implicit_sample_data):
        train, val = train_val_implicit_sample_data
        model = MatrixFactorizationModel(train=train,
                                         val=val,
                                         optimizer='adam',
                                         bias_optimizer='sgd',
                                         lr_scheduler_func=partial(StepLR, step_size=1))
        trainer = CollieMinimalTrainer(model=model, max_epochs=1)
        trainer.fit(model)

    def test_multiple_optimizers_only(self, train_val_implicit_sample_data):
        train, val = train_val_implicit_sample_data
        model = MatrixFactorizationModel(train=train,
                                         val=val,
                                         optimizer='adam',
                                         bias_optimizer='sgd',
                                         lr_scheduler_func=None)
        trainer = CollieMinimalTrainer(model=model, max_epochs=1)
        trainer.fit(model)

    def test_single_optimizer_and_lr_scheduler(self, train_val_implicit_sample_data):
        train, val = train_val_implicit_sample_data
        model = MatrixFactorizationModel(train=train,
                                         val=val,
                                         optimizer='adam',
                                         bias_optimizer=None,
                                         lr_scheduler_func=ReduceLROnPlateau)
        trainer = CollieMinimalTrainer(model=model, max_epochs=1)
        trainer.fit(model)

    def test_single_optimizer_only(self, train_val_implicit_sample_data):
        train, val = train_val_implicit_sample_data
        model = MatrixFactorizationModel(train=train,
                                         val=val,
                                         optimizer='adam',
                                         bias_optimizer=None,
                                         lr_scheduler_func=None)
        trainer = CollieMinimalTrainer(model=model, max_epochs=1)
        trainer.fit(model)

    @mock.patch.object(MatrixFactorizationModel, 'configure_optimizers')
    def test_unexpected_configure_optimizers_output(self,
                                                    configure_optimizers_mock,
                                                    train_val_implicit_sample_data):
        configure_optimizers_mock.return_value = 'optimizer'

        train, val = train_val_implicit_sample_data
        model = MatrixFactorizationModel(train=train,
                                         val=val,
                                         optimizer='adam',
                                         bias_optimizer=None,
                                         lr_scheduler_func=None)
        trainer = CollieMinimalTrainer(model=model, max_epochs=1)

        with pytest.raises(ValueError):
            trainer.fit(model)

    @pytest.mark.parametrize('verbosity', [0, False])
    @mock.patch('torch.cuda.is_available')
    def test_no_verbosity(self,
                          is_available_mock,
                          train_val_implicit_sample_data,
                          verbosity,
                          capfd):
        is_available_mock.return_value = False

        train, val = train_val_implicit_sample_data
        model = MatrixFactorizationModel(train=train,
                                         val=val)
        trainer = CollieMinimalTrainer(model=model,
                                       max_epochs=1,
                                       weights_summary='full',
                                       verbosity=verbosity)
        trainer.fit(model)

        out, _ = capfd.readouterr()
        assert out == ''

    def test_unexpected_batch_format(self, train_val_implicit_sample_data):
        train, val = train_val_implicit_sample_data
        model = MatrixFactorizationModel(train=train, val=val)
        trainer = CollieMinimalTrainer(model=model)

        # ensure this runs without error
        unexpected_batch_format = [
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 3]),
        ]
        actual = trainer._move_batch_to_device(unexpected_batch_format)

        for actual_batch in actual:
            if torch.cuda.is_available():
                assert str(actual_batch.device) == 'cuda:0'
            else:
                assert str(actual_batch.device) == 'cpu'


def test_model_instantiation_no_train_data():
    with pytest.raises(TypeError):
        MatrixFactorizationModel()


def test_instantiation_of_sparse_model_with_weight_decay(train_val_implicit_data, capfd):
    train, val = train_val_implicit_data

    model_1 = MatrixFactorizationModel(train=train,
                                       val=val,
                                       sparse=False,
                                       weight_decay=100)
    assert model_1.hparams.weight_decay == 100

    with pytest.warns(UserWarning):
        model_2 = MatrixFactorizationModel(train=train,
                                           val=val,
                                           sparse=True,
                                           weight_decay=100)
    assert model_2.hparams.weight_decay == 0


@pytest.mark.parametrize('adaptive_loss_type', ['adaptive_hinge', 'adaptive', 'adaptive_bpr'])
def test_bad_adaptive_implicit_loss_selected(train_val_implicit_data, adaptive_loss_type):
    train, val = train_val_implicit_data

    train = copy.copy(train)
    val = copy.copy(val)

    train.num_negative_samples = 1
    val.num_negative_samples = 1

    with pytest.warns(UserWarning):
        MatrixFactorizationModel(train=train, val=val, loss=adaptive_loss_type)


@pytest.mark.parametrize('model_type', ['with_lightning', 'no_lightning'])
def test_implicit_model(implicit_model,
                        implicit_model_no_lightning,
                        train_val_implicit_data,
                        model_type):
    if model_type == 'with_lightning':
        model = implicit_model
    elif model_type == 'no_lightning':
        model = implicit_model_no_lightning

    train, val = train_val_implicit_data

    item_preds = model.get_item_predictions(user_id=0,
                                            unseen_items_only=True,
                                            sort_values=True)

    assert isinstance(item_preds, pd.Series)
    assert len(item_preds) > 0
    assert len(item_preds) < len(train)

    item_similarities = model.item_item_similarity(item_id=42)
    assert item_similarities.index[0] == 42

    mapk_score = evaluate_in_batches([mapk], val, model)

    # The metrics used for evaluation have been determined through 30
    # trials of training the model and using the mean - 5 * std. dev.
    # as the minimum score the model must achieve to pass the test.
    assert mapk_score > 0.044


@pytest.mark.parametrize('model_type', ['with_lightning', 'no_lightning'])
def test_explicit_model(explicit_model,
                        explicit_model_no_lightning,
                        train_val_explicit_data,
                        model_type):
    if model_type == 'with_lightning':
        model = explicit_model
    elif model_type == 'no_lightning':
        model = explicit_model_no_lightning

    train, val = train_val_explicit_data

    item_preds = model.get_item_predictions(user_id=0,
                                            unseen_items_only=True,
                                            sort_values=True)

    assert isinstance(item_preds, pd.Series)
    assert len(item_preds) > 0
    assert len(item_preds) < len(train)

    item_similarities = model.item_item_similarity(item_id=42)
    assert item_similarities.index[0] == 42

    mse_score = explicit_evaluate_in_batches([torchmetrics.MeanSquaredError()],
                                             val,
                                             model,
                                             num_workers=0)

    # The metrics used for evaluation have been determined through 30
    # trials of training the model and using the mean - 5 * std. dev.
    # as the minimum score the model must achieve to pass the test.
    assert mse_score < 0.943


def test_unexpected_batch_format_calculate_loss(train_val_implicit_data):
    train, val = train_val_implicit_data
    model = MatrixFactorizationModel(train=train, val=val)

    # ensure this runs without error
    unexpected_batch_format = torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])

    with pytest.raises(ValueError):
        model.calculate_loss(batch=unexpected_batch_format)


def test_bad_final_layer_of_neucf(train_val_implicit_data):
    train, val = train_val_implicit_data

    model = NeuralCollaborativeFiltering(train=train,
                                         val=val,
                                         final_layer='nonexistent_final_layer')
    trainer = CollieTrainer(model=model, logger=False, checkpoint_callback=False, max_steps=1)

    with pytest.raises(ValueError):
        trainer.fit(model)


def test_bad_final_layer_of_deep_fm(train_val_implicit_data):
    train, val = train_val_implicit_data

    model = DeepFM(train=train,
                   val=val,
                   final_layer='nonexistent_final_layer')
    trainer = CollieTrainer(model=model, logger=False, checkpoint_callback=False, max_steps=1)

    with pytest.raises(ValueError):
        trainer.fit(model)


def test_bad_initialization_of_hybrid_pretrained_model(implicit_model,
                                                       movielens_metadata_df,
                                                       train_val_implicit_data):
    train, val = train_val_implicit_data

    with pytest.raises(ValueError):
        HybridPretrainedModel(train=train,
                              val=val,
                              item_metadata=movielens_metadata_df,
                              trained_model=None)

    with pytest.raises(ValueError):
        HybridPretrainedModel(train=train,
                              val=val,
                              item_metadata=None,
                              trained_model=implicit_model)


def test_different_item_metadata_types_for_hybrid_pretrained_model(implicit_model,
                                                                   movielens_metadata_df,
                                                                   train_val_implicit_data):
    train, val = train_val_implicit_data

    # ensure that we end up with the same ``item_metadata`` regardless of the input type
    model_1 = HybridPretrainedModel(train=train,
                                    val=val,
                                    item_metadata=movielens_metadata_df,
                                    trained_model=implicit_model)
    trainer_1 = CollieTrainer(model=model_1, logger=False, checkpoint_callback=False, max_steps=1)
    trainer_1.fit(model_1)

    model_2 = HybridPretrainedModel(train=train,
                                    val=val,
                                    item_metadata=movielens_metadata_df.to_numpy(),
                                    trained_model=implicit_model)
    trainer_2 = CollieTrainer(model=model_2, logger=False, checkpoint_callback=False, max_steps=1)
    trainer_2.fit(model_2)

    model_3 = HybridPretrainedModel(
        train=train,
        val=val,
        item_metadata=torch.from_numpy(movielens_metadata_df.to_numpy()),
        trained_model=implicit_model,
    )
    trainer_3 = CollieTrainer(model=model_3, logger=False, checkpoint_callback=False, max_steps=1)
    trainer_3.fit(model_3)

    assert model_1.item_metadata.equal(model_2.item_metadata)
    assert model_2.item_metadata.equal(model_3.item_metadata)


def test_loading_and_saving_implicit_model(implicit_model, untrained_implicit_model, tmpdir):
    expected = implicit_model.get_item_predictions(user_id=42, unseen_items_only=False)

    # set up TemporaryDirectory for writing and reading all files in this test
    temp_dir_name = str(tmpdir)

    save_model_path = os.path.join(temp_dir_name, 'test_mf_model_save.pth')
    implicit_model.save_model(save_model_path)
    loaded_implicit_model = MatrixFactorizationModel(load_model_path=save_model_path)

    actual = loaded_implicit_model.get_item_predictions(user_id=42, unseen_items_only=False)

    assert expected.equals(actual)

    # now, test that this is not equal to a randomly initialized model's output
    new_preds = untrained_implicit_model.get_item_predictions(user_id=42, unseen_items_only=False)

    assert not expected.equals(new_preds)


def test_loading_and_saving_hybrid_pretrained_model(implicit_model,
                                                    movielens_metadata_df,
                                                    train_val_implicit_data,
                                                    tmpdir):
    train, val = train_val_implicit_data

    model = HybridPretrainedModel(train=train,
                                  val=val,
                                  item_metadata=movielens_metadata_df,
                                  trained_model=implicit_model,
                                  metadata_layers_dims=[16, 8],
                                  freeze_embeddings=True)
    trainer = CollieTrainer(model=model, logger=False, checkpoint_callback=False, max_steps=10)
    trainer.fit(model)

    expected = model.get_item_predictions(user_id=42, unseen_items_only=False)

    # set up TemporaryDirectory for writing and reading the file in this test
    temp_dir_name = str(tmpdir)

    save_model_path = os.path.join(temp_dir_name, 'test_hybrid_model_save')
    model.save_model(save_model_path)
    loaded_model = HybridPretrainedModel(load_model_path=save_model_path)

    actual = loaded_model.get_item_predictions(user_id=42, unseen_items_only=False)

    assert expected.equals(actual)

    # now, test that this is not equal to a randomly initialized model's output
    implicit_preds = implicit_model.get_item_predictions(user_id=42, unseen_items_only=False)

    assert not expected.equals(implicit_preds)


def test_bad_saving_hybrid_pretrained_model(implicit_model,
                                            movielens_metadata_df,
                                            train_val_implicit_data,
                                            tmpdir):
    train, val = train_val_implicit_data

    model = HybridPretrainedModel(train=train,
                                  val=val,
                                  item_metadata=movielens_metadata_df,
                                  trained_model=implicit_model,
                                  metadata_layers_dims=[16, 8],
                                  freeze_embeddings=True)
    trainer = CollieTrainer(model=model, logger=False, checkpoint_callback=False, max_steps=10)
    trainer.fit(model)

    # set up TemporaryDirectory for writing and reading all files in this test
    temp_dir_name = str(tmpdir)

    # we shouldn't be able to overwrite a model in an existing directory unless we specifically say
    save_model_path = os.path.join(temp_dir_name, 'test_hybrid_model_save_to_overwrite')
    model.save_model(save_model_path)

    with pytest.raises(ValueError):
        model.save_model(save_model_path)

    model.save_model(save_model_path, overwrite=True)


def test_models_trained_for_one_step(models_trained_for_one_step, train_val_implicit_data):
    train, _ = train_val_implicit_data

    if not isinstance(models_trained_for_one_step.train_loader, HDF5InteractionsDataLoader):
        item_preds = models_trained_for_one_step.get_item_predictions(user_id=0,
                                                                      unseen_items_only=True,
                                                                      sort_values=True)

        assert isinstance(item_preds, pd.Series)
        assert len(item_preds) > 0
        assert len(item_preds) < len(train)

    item_similarities = models_trained_for_one_step.item_item_similarity(item_id=42)

    assert item_similarities.index[0] == 42


def test_explicit_models_trained_for_one_step(explicit_models_trained_for_one_step,
                                              train_val_explicit_data):
    train, _ = train_val_explicit_data

    item_preds = explicit_models_trained_for_one_step.get_item_predictions(user_id=0,
                                                                           unseen_items_only=True,
                                                                           sort_values=True)

    assert isinstance(item_preds, pd.Series)
    assert len(item_preds) > 0
    assert len(item_preds) < len(train)

    item_similarities = explicit_models_trained_for_one_step.item_item_similarity(item_id=42)

    assert item_similarities.index[0] == 42


def test_bad_implicit_model_explicit_data(train_val_explicit_data):
    train, val = train_val_explicit_data

    with pytest.raises(ValueError):
        MatrixFactorizationModel(train=train, val=val, loss='hinge')


def test_really_bad_implicit_model_explicit_data(train_val_explicit_data, train_val_implicit_data):
    explicit_train, explicit_val = train_val_explicit_data
    implicit_train, implicit_val = train_val_implicit_data

    model = MatrixFactorizationModel(train=implicit_train,
                                     val=implicit_val,
                                     loss='hinge')

    # if we somehow make it past the initial data quality check, ensure we still fail later on
    model.train_loader = InteractionsDataLoader(explicit_train)
    model.val_loader = InteractionsDataLoader(explicit_val)

    trainer = CollieTrainer(model=model, logger=False, checkpoint_callback=False, max_steps=1)

    with pytest.raises(ValueError):
        trainer.fit(model)


def test_bad_explicit_model_implicit_data(train_val_implicit_sample_data):
    train, val = train_val_implicit_sample_data

    model = MatrixFactorizationModel(train=train,
                                     val=val,
                                     loss='mse')
    trainer = CollieTrainer(model=model, logger=False, checkpoint_callback=False, max_steps=1)

    with pytest.raises(ValueError):
        trainer.fit(model)
