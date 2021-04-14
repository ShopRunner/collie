# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project uses [Semantic Versioning](http://semver.org/).

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
