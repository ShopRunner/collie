# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project uses [Semantic Versioning](http://semver.org/).

# [0.5.0] - 2021-6-4
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
