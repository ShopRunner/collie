import os
import tempfile

import pytest
import torch

from collie_recs.interactions import (ApproximateNegativeSamplingInteractionsDataLoader,
                                      HDF5InteractionsDataLoader,
                                      InteractionsDataLoader)
from collie_recs.model import (CollieTrainer,
                               HybridPretrainedModel,
                               MatrixFactorizationModel,
                               NeuralCollaborativeFiltering,
                               NonlinearMatrixFactorizationModel)
from collie_recs.utils import pandas_df_to_hdf5


@pytest.fixture(scope='session')
def implicit_model(train_val_implicit_data, gpu_count):
    train, val = train_val_implicit_data
    model = MatrixFactorizationModel(train=train,
                                     val=val,
                                     embedding_dim=10,
                                     lr=1e-1)
    model_trainer = CollieTrainer(model=model,
                                  gpus=gpu_count,
                                  max_epochs=10,
                                  deterministic=True,
                                  logger=False,
                                  checkpoint_callback=False)
    model_trainer.fit(model)
    model.freeze()

    return model


@pytest.fixture(params=['mf_hdf5',
                        'sparse_mf',
                        'mf_non_approximate',
                        'mf_approximate',
                        'nonlinear_mf',
                        'neucf',
                        'hybrid'])
def other_models_trained_for_one_epoch(request,
                                       train_val_implicit_data,
                                       movielens_metadata_df,
                                       movielens_implicit_df,
                                       train_val_implicit_pandas_data,
                                       gpu_count):
    train, val = train_val_implicit_data

    if request.param == 'mf_hdf5':
        # create, fit, and return the model all at once so we can close the HDF5 file
        train_pandas_df, val_pandas_df = train_val_implicit_pandas_data

        with tempfile.TemporaryDirectory() as temp_dir:
            pandas_df_to_hdf5(df=train_pandas_df,
                              out_path=os.path.join(temp_dir, 'train.h5'),
                              key='interactions')
            pandas_df_to_hdf5(df=val_pandas_df,
                              out_path=os.path.join(temp_dir, 'val.h5'),
                              key='interactions')

            train_loader = HDF5InteractionsDataLoader(hdf5_path=os.path.join(temp_dir, 'train.h5'),
                                                      user_col='user_id',
                                                      item_col='item_id',
                                                      num_users=train.num_users,
                                                      num_items=train.num_items,
                                                      batch_size=1024,
                                                      shuffle=True)
            val_loader = HDF5InteractionsDataLoader(hdf5_path=os.path.join(temp_dir, 'val.h5'),
                                                    user_col='user_id',
                                                    item_col='item_id',
                                                    num_users=val.num_users,
                                                    num_items=val.num_items,
                                                    batch_size=1024,
                                                    shuffle=False)

            model = MatrixFactorizationModel(train=train_loader,
                                             val=val_loader,
                                             embedding_dim=15,
                                             dropout_p=0.1,
                                             lr=1e-1,
                                             bias_lr=1e-2,
                                             optimizer='adam',
                                             bias_optimizer='sgd',
                                             weight_decay=1e-7,
                                             loss='bpr',
                                             sparse=False)

            model_trainer = CollieTrainer(model=model,
                                          gpus=gpu_count,
                                          max_epochs=1,
                                          deterministic=True,
                                          logger=False,
                                          checkpoint_callback=False)

            model_trainer.fit(model)
            model.freeze()

            return model

    elif request.param == 'sparse_mf':
        model = MatrixFactorizationModel(train=train,
                                         val=val,
                                         embedding_dim=15,
                                         dropout_p=0.1,
                                         lr=1e-1,
                                         bias_lr=1e-2,
                                         optimizer='sparse_adam',
                                         bias_optimizer='sgd',
                                         weight_decay=0,
                                         loss='hinge',
                                         sparse=True)
    elif request.param == 'mf_non_approximate' or request.param == 'mf_approximate':
        if request.param == 'mf_non_approximate':
            train_loader = InteractionsDataLoader(interactions=train, batch_size=1024, shuffle=True)
            val_loader = InteractionsDataLoader(interactions=val, batch_size=1024, shuffle=False)
        else:
            train_loader = ApproximateNegativeSamplingInteractionsDataLoader(interactions=train,
                                                                             batch_size=1024,
                                                                             shuffle=True)
            val_loader = ApproximateNegativeSamplingInteractionsDataLoader(interactions=val,
                                                                           batch_size=1024,
                                                                           shuffle=False)

        model = MatrixFactorizationModel(train=train_loader,
                                         val=val_loader,
                                         embedding_dim=15,
                                         dropout_p=0.1,
                                         lr=1e-1,
                                         bias_lr=1e-2,
                                         optimizer='adam',
                                         bias_optimizer='sgd',
                                         weight_decay=1e-7,
                                         loss='bpr',
                                         sparse=False)
    elif request.param == 'nonlinear_mf':
        model = NonlinearMatrixFactorizationModel(train=train,
                                                  val=val,
                                                  user_embedding_dim=15,
                                                  item_embedding_dim=15,
                                                  user_dense_layers_dims=[15, 10],
                                                  item_dense_layers_dims=[15, 10],
                                                  embedding_dropout_p=0.05,
                                                  dense_dropout_p=0.1,
                                                  lr=1e-1,
                                                  bias_lr=1e-2,
                                                  optimizer='adam',
                                                  bias_optimizer='sgd',
                                                  weight_decay=1e-7,
                                                  loss='bpr')
    elif request.param == 'neucf':
        model = NeuralCollaborativeFiltering(train=train,
                                             val=val,
                                             embedding_dim=10,
                                             num_layers=1,
                                             dropout_p=0.1,
                                             lr=1e-3,
                                             weight_decay=0.,
                                             optimizer='adam',
                                             loss='adaptive')
    elif request.param == 'hybrid':
        implicit_model = MatrixFactorizationModel(train=train,
                                                  val=val,
                                                  embedding_dim=10,
                                                  lr=1e-1,
                                                  optimizer='adam')
        implicit_model_trainer = CollieTrainer(model=implicit_model,
                                               gpus=gpu_count,
                                               max_epochs=1,
                                               deterministic=True,
                                               logger=False,
                                               checkpoint_callback=False)
        implicit_model_trainer.fit(implicit_model)
        implicit_model.freeze()

        genres = (
            torch.tensor(movielens_metadata_df[
                [c for c in movielens_metadata_df.columns if 'genre' in c]
            ].values)
            .topk(1)
            .indices
            .view(-1)
        )

        model_frozen = HybridPretrainedModel(train=train,
                                             val=val,
                                             item_metadata=movielens_metadata_df,
                                             trained_model=implicit_model,
                                             freeze_embeddings=True,
                                             dropout_p=0.15,
                                             loss='warp',
                                             lr=.01,
                                             optimizer=torch.optim.Adam,
                                             metadata_for_loss={'genre': genres},
                                             metadata_for_loss_weights={'genre': .4},
                                             weight_decay=0.0)
        model_frozen_trainer = CollieTrainer(model=model_frozen,
                                             gpus=gpu_count,
                                             max_epochs=1,
                                             deterministic=True,
                                             logger=False,
                                             checkpoint_callback=False)
        model_frozen_trainer.fit(model_frozen)

        model_frozen.unfreeze_embeddings()
        model = HybridPretrainedModel(train=train,
                                      val=val,
                                      item_metadata=movielens_metadata_df,
                                      trained_model=implicit_model,
                                      freeze_embeddings=True,
                                      dropout_p=0.15,
                                      loss='bpr',
                                      lr=1e-4,
                                      optimizer=torch.optim.Adam,
                                      metadata_for_loss={'genre': genres},
                                      metadata_for_loss_weights={'genre': .4},
                                      weight_decay=0.0)
        model.load_from_hybrid_model(model_frozen)

    model_trainer = CollieTrainer(model=model,
                                  gpus=gpu_count,
                                  max_epochs=1,
                                  deterministic=True,
                                  logger=False,
                                  checkpoint_callback=False)

    model_trainer.fit(model)
    model.freeze()

    return model
