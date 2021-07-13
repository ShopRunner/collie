# Contributing and Making PRs

## How to Contribute

We welcome contributions in the form of issues or pull requests!

We want this to be a place where all are welcome to discuss and contribute, so please note that this project is released with a [Contributor Code of Conduct](CODE-OF-CONDUCT.md). By participating in this project you agree to abide by its terms. Find the Code of Conduct in the ``CONDUCT.md`` file on GitHub or in the Code of Conduct section of read the docs.

If you have a problem using Collie or see a possible improvement, open an issue in the GitHub issue tracker. Please be as specific as you can.

If you see an open issue you'd like to be fixed, take a stab at it and open a PR!


Pull Requests
-------------
To create a PR against this library, please fork the project and work from there.

Steps
++++++

1. Fork the project via the ``Fork`` button on GitHub.

2. Clone the repo to your local disk.

3. Create a new branch for your PR.

    ```bash
        git checkout -b my-awesome-new-feature
    ```

4. Install requirements (either in a virtual environment like below or the Docker container).

    ```bash
        virtualenv venv
        source venv/bin/activate
        pip install -r requirements-dev.txt
        pip install -r requirements.txt
    ```

5. Develop your feature

6. Submit a PR to ``main``! Someone will review your code and merge your code into ``main`` when it is approved.

PR Checklist
+++++++++++++

- Ensure your code has followed the Style Guidelines listed below.
- Run the ``flake8`` linter on your code.

    ```bash
        source venv/bin/activate
        flake8 collie tests
    ```

- Make sure you have written unit-tests where appropriate.
- Make sure the unit-tests all pass.

    ```bash
        source venv/bin/activate
        pytest -v
    ```

- Update the docs where appropriate. You can rebuild them with the commands below.

    ```bash
        cd docs
        make html
    ```

- Update the ``CHANGELOG.md`` and ``version.py`` files.

Style Guidelines
++++++++++++++++++++++++++

For the most part, this library follows PEP8 with a couple of exceptions.

- Lines can be up to 100 characters long.
- Docstrings should be [numpy style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html) docstrings.
- Your code should be Python 3 compatible.
- We prefer single quotes for one-line strings unless using double quotes allows us to avoid escaping internal single quotes.
- When in doubt, follow the style of the existing code.
