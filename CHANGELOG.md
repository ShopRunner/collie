# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project uses [Semantic Versioning](http://semver.org/).

# [1.3.1] - 2023-3-20
### Fixed
 - Versions of PyTorch Lightning between ``1.6.0`` and ``1.9.X`` are now supported

# [1.3.0] - 2022-8-20
### Added
 - ``HybridModel`` and ``HybridPretrainedModel`` now take additional optional parameters ``user_metadata`` and ``user_metadata_layers_dims``
 - ``get_data.py`` now includes ``get_user_metadata``
### Changed
 - Added ``item_metadata_layers_dims`` and ``user_metadata_layers_dims`` parameters to ``HybridPretrainedModel`` and ``HybridModel`` and removed ``metadata_layers_dims``
 - Updated notebooks and examples to include usage of ``user_metadata``

# [1.2.2] - 2022-7-14
### Fixed
 - a ``Value Error`` is now raised when ``item_metadata`` contains nulls

# [1.2.1] - 2022-7-12
### Fixed
 - Deprecated PyTorch Lightning unit test

# [1.2.0] - 2022-1-18
### Added
 - option to ``force_split`` to ``stratified_split``
 - better type hints for ``Callable``s
 - added methods ``get_user_predictions`` and ``user_user_similarity`` to the ``BasePipeline``
 - added ``_get_user_embeddings`` method to all model classes
### Changed
 - default ``Dockerfile`` image to be ``torch@1.10.0`` with CUDA 11.3
 - check if index is in-bound for ``get_item_predictions`` and ``item_item_similarity`` before calling the model
 - added ``enable_model_summary`` and ``detect_anomaly`` parameters to ``CollieMinimalTrainer`` and deprecated ``weights_summary`` and ``terminate_on_nan`` to more closely match the new ``pytorch_lightning`` API
 - clarified error message when user has a single interaction when using ``stratified_split``
 - updated all examples, tests, and notebooks with post-1.5.0 PyTorch Lightning APIs
### Fixed
 - device error when running metrics for a ``MultiStagePipeline`` models
 - ``CollieMinimalTrainer`` model summary to work with later versions of PyTorch Lightning
### Removed
 - default ``num_workers`` for ``Interactions`` DataLoaders

# [1.1.2] - 2021-8-17
### Added
 - string support for Adagrad optimizer in model pipelines

# [1.1.1] - 2021-8-17
### Changed
 - added property ``max_epochs`` to ``CollieTrainer`` and ``CollieMinimalTrainer`` with ``setter`` method
 - `CollieTrainer`'s default `max_epochs` from `1000` to `10`
### Fixed
 - used new API for setting verbosity in ``ModelSummary`` in ``CollieMinimalTrainer``

# [1.1.0] - 2021-7-15
### Added
 - multi-stage model template ``MultiStagePipeline``
 - multi-stage models ``HybridModel`` and ``ColdStartModel``
### Changed
 - optimizers and learning rate schedulers are now reset upon each call of ``fit`` in ``CollieMinimalTrainer``, matching the behavior in ``CollieTrainer``
 - ``HybridPretrainedModel`` now includes bias terms from the ``MatrixFactorizationModel`` in score calculation
 - ``item_item_similarity`` now uses a more efficient, on-device calculation for item-item similarity
 - optimizers are now stepped using the ``optimizer_step`` method for ``CollieMinimalTrainer``
 - ``_get_item_embeddings`` methods now return a ``torch.tensor`` type on device
 - ``_move_external_data_to_device`` optional method to all models
### Removed
 - ``MultiOptimizer.step`` method

# [1.0.1] - 2021-7-13
### Fixed
 - GitHub URL in ``read_movielens_posters_df`` to point to new repo name

# [1.0.0] - 2021-7-13
### Changed
 - name of library to ``collie``!

# [0.6.1] - 2021-7-13
### Added
 - name change warning from ``collie_recs -> collie``

# [0.6.0] - 2021-7-6
### Added
 - support for explicit data with ``ExplicitInteractions`` and ``explicit_evaluate_in_batches``
 - warnings for invalid adaptive loss vs. ``num_negative_samples`` combinations
### Changed
 - default ``Dockerfile`` image to be ``torch@1.9.0`` with CUDA 10.2
### Removed
 - ``hybrid_matrix_factorization_model.py`` deprecated filename

# [0.5.0] - 2021-6-11
### Added
 - new model architectures ``CollaborativeMetricLearningModel``, ``MLPMatrixFactorizationModel``, and ``DeepFM``
### Changed
 - filename for ``HybridPretrainedModel`` to ``hybrid_pretrained_matrix_factorization.py``. The former model filepath is now deprecated and will be removed in future version ``0.6.0``
 - ``collie.model.base`` is now split into its own directory with the same name
 - reduced boilerplate docstrings required for models
 - all ``model.freeze() -> model.eval()``
 - bumped version of ``sphinx-rtd-theme`` to ``0.5.2``

# [0.4.0] - 2021-5-13
### Added
 - ``CollieMinimalTrainer`` for a faster, simpler version of ``CollieTrainer``
 - ``remove_duplicate_user_item_pairs`` argument to ``Interactions``
### Changed
 - renamed `BasePipeline.hparams.n_epochs_completed_ -> BasePipeline.hparams.num_epochs_completed`
### Fixed
 - a proper ``ValueError`` is now raised if no ``train`` data is passed into a model
 - loss docstrings that incorrectly stated ``**kwargs`` would be accepted

# [0.3.0] - 2021-5-10
### Changed
 - disable automated batching in ``ApproximateNegativeSamplingInteractionsDataLoader`` and ``HDF5InteractionsDataLoader``

# [0.2.0] - 2021-4-28
### Changed
 - ``convert_to_implicit`` will now remove duplicate user/item pairs in DataFrame

# [0.1.4] - 2021-4-27
### Fixed
 - duplicate user/item pairs in ``Interactions`` are now dropped from the COO matrix during instantiation

# [0.1.3] - 2021-4-26
### Added
 - ability to run ``stratified_split`` without any ``joblib.Parallel`` parallelization
 - data quality checks to ``Interactions.__init__`` to assert ``users`` and ``items`` and ``mat`` are not ``None`` and ``ratings`` does not contain any ``0``s (if so, those rows will now automatically be filtered out)
 - increased test coverage
 - header table to all Jupyter notebooks with links to Colab and GitHub
### Changed
 - default ``processes`` for ``stratified_split`` is now ``-1``
 - default ``k`` value in ``mapk`` is now set to ``10``
 - when GPU is available but not set, ``CollieTrainer`` now sets it to ``1`` rather than ``-1``
 - all models now check that ``train_loader`` and ``val_loader`` attributes are consistent during initialization
 - default ``unseen_items_only`` in ``BasePipeline.get_item_predictions`` method is now ``False``
 - docs in ``get_recommendation_visualizations`` to be clearer
 - ``get_recommendation_visualizations`` data quality checks have been moved to the beginning of the function to potentially fail faster
 - ``create_ratings_matrix`` no longer raises ``ValueError`` if ``users`` and ``items`` do not start at ``0``
 - refactored ``adaptive_hinge_loss``
### Removed
 - ``kwargs`` option for methods that did not explicitly need them
### Fixed
 - typo in ``cross_validation.py`` error message
 - ``head`` and ``tail`` methods in ``interactions/datasets.py`` to no longer error with ``n < 1`` or large ``n`` values
 - ``num_users`` and ``num_items`` are no longer incorrectly incremented when ``meta`` key is provided in ``HDF5Interactions``
 - type hints for ``device`` now also include instances of ``torch.device``
 - the type of metadata tensors sent to ``HybridPretrainedModel`` are now consistent across all input options
 - removed ineffective quality checks in ``HybridPretrainedModel.save_model``
 - no longer use deprecated ``nn.sigmoid`` in library
 - a ``relu`` final activation layer now works in ``NeuralCollaborativeFiltering`` model
 - ``df_to_html`` now outputs proper HTML when multiple ``html_tags`` options are specified
 - tutorial notebooks now fully run on Colab without having to only rely on previously-saved data
 - add value of ``1e-11`` to ``BasePipeline.get_item_predictions`` denominator to avoid potential ``NaN``s

# [0.1.2] - 2021-4-14
### Added
 - various badges to ``README``
 - links to documentation where relevant
 - Colab links to tutorial notebooks and ``README`` quickstart
### Changed
 - ``ApproximateNegativeSamplingInteractionsDataLoader`` uses a ``sampler`` instead of a ``batch_sampler`` for greater speed increases
 - base ``Dockerfile`` image to the ``torch@1.8.1`` version
 - ``read_movielens_posters_df`` now works as expected even when ``data/movielens_posters.csv`` file does not exist
 - renamed ``LICENSE.txt -> LICENSE``
 - renamed ``pull_request_template.md -> PULL_REQUEST_TEMPLATE.md``

# [0.1.1] - 2021-4-13
### Added
 - GitHub Actions and templates in ``.github``

# [0.1.0] - 2021-4-13
### Added
 - Collie library for open sourcing

# [0.0.0] - 2021-4-12
### Added
 - Initial project scaffolding
