from contextlib import suppress
import os
from unittest import mock

import pandas as pd
import pytest
import pytorch_lightning
import torch

from collie_recs.interactions import HDF5InteractionsDataLoader
from collie_recs.loss import (adaptive_bpr_loss,
                              adaptive_hinge_loss,
                              bpr_loss,
                              hinge_loss,
                              warp_loss)
from collie_recs.metrics import evaluate_in_batches, mapk
from collie_recs.model import (BasePipeline,
                               CollieTrainer,
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

    with suppress(pytorch_lightning.utilities.exceptions.MisconfigurationException):
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


def test_instantiation_of_model_loss(train_val_implicit_data):
    train, val = train_val_implicit_data

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


def test_implicit_model(implicit_model, train_val_implicit_data):
    train, val = train_val_implicit_data

    item_preds = implicit_model.get_item_predictions(user_id=0,
                                                     unseen_items_only=True,
                                                     sort_values=True)

    assert isinstance(item_preds, pd.Series)
    assert len(item_preds) > 0
    assert len(item_preds) < len(train)

    item_similarities = implicit_model.item_item_similarity(item_id=42)
    assert item_similarities.index[0] == 42

    mapk_score = evaluate_in_batches([mapk], val, implicit_model)

    # The metrics used for evaluation have been determined through 30
    # trials of training the model and using the mean - 5 * std. dev.
    # as the minimum score the model must achieve to pass the test.
    assert mapk_score > 0.044


def test_bad_final_layer_of_neucf(train_val_implicit_data):
    train, val = train_val_implicit_data

    model = NeuralCollaborativeFiltering(train=train,
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
