from setuptools import find_packages, setup

with open('collie_recs/_version.py') as version_file:
    exec(version_file.read())

with open('README.md') as r:
    readme = r.read()

with open('LICENSE.txt') as l:
    license = l.read()

setup(
    name='collie_recs',
    version=__version__,
    description='deep learning recommendation system',
    long_description=readme+'\n\n\nLicense\n-------\n'+license,
    long_description_content_type='text/markdown',
    author='Nathan Jones',
    url='https://github.com/ShopRunner/collie_recs',
    license='BSD-3-Clause',
    data_files=[('', ['LICENSE.txt'])],
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'fire',
        'joblib',
        'numpy',
        'pandas',
        'pytorch-lightning>=1.0.0',  # ``collie_recs`` library uses newer ``pytorch_lightning`` APIs
        'scikit-learn',
        'tables',
        'torch',
        'torchmetrics',
        'tqdm',
    ],
)
