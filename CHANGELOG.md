# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project uses [Semantic Versioning](http://semver.org/).

# [0.1.3] - 2021-4-22
### Added
 - ability to run ``stratified_split`` without any ``joblib.Parallel`` parallelization
 - data quality checks to ``Interactions.__init__`` to assert ``users`` and ``items`` and ``mat`` are not ``None`` and ``ratings`` does not contain any ``0``s (if so, those rows will now automatically be filtered out)
 - increased test coverage
 - header table to all Jupyter notebooks with links to Colab and GitHub
### Changed
 - error raised for unexpected ``kwargs`` is now a ``TypeError``
 - default ``processes`` for ``stratified_split`` is now ``-1``
 - default ``k`` value in ``mapk`` is now set to ``10``
 - when GPU is available but not set, ``CollieTrainer`` now sets it to ``1`` rather than ``-1``
 - all models now check that ``train_loader`` and ``val_loader`` attributes are consistent during initialization
 - default ``unseen_items_only`` in ``BasePipeline.get_item_predictions`` method is now ``False``
 - docs in ``get_recommendation_visualizations`` to be clearer
 - ``get_recommendation_visualizations`` data quality checks have been moved to the beginning of the function to potentially fail faster
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
