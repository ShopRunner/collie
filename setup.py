from setuptools import find_packages, setup

with open('collie/_version.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()

with open('LICENSE') as l:
    license = l.read()

setup(
    name='collie',
    version=__version__,
    description='A PyTorch library for preparing, training, and evaluating deep learning hybrid recommender systems.',
    long_description=readme+'\n\n\nLicense\n-------\n'+license,
    long_description_content_type='text/markdown',
    author='Nathan Jones',
    url='https://github.com/ShopRunner/collie',
    download_url='https://github.com/ShopRunner/collie',
    license='BSD-3-Clause',
    data_files=[('', ['LICENSE'])],
    packages=find_packages(exclude=('tests', 'docs')),
    keywords=['deep learning', 'pytorch', 'recommender'],
    python_requires='>=3.6',
    install_requires=[
        'docstring_parser',
        'fire',
        'joblib',
        'numpy',
        'pandas',
        'pytorch-lightning>=1.0.0',  # ``collie`` library uses newer ``pytorch_lightning`` APIs
        'scikit-learn',
        'tables',
        'torch',
        'torchmetrics',
        'tqdm',
    ],
    extras_require={
        'dev': [
            'flake8',
            'flake8-docstrings',
            'flake8-import-order',
            'ipython',
            'ipywidgets',
            'jupyterlab>=3.0.0',
            'matplotlib',
            'm2r2',
            'pip-tools',
            'pydocstyle<4.0.0',
            'pytest',
            'pytest-cov<3.0.0',
            'sphinx-copybutton',
            'sphinx-rtd-theme==0.5.2',
            'widgetsnbextension',
        ]
    },
    project_urls={
        'Documentation': 'https://collie.readthedocs.io/',
        'Source Code': 'https://github.com/ShopRunner/collie',
    },
    classifiers=[
        'Environment :: Console',
        'Environment :: GPU',
        'Natural Language :: English',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
