# collie_recs
[![Compliance OSS](https://github.com/ShopRunner/collie_recs/actions/workflows/compliance-oss.yaml/badge.svg)](https://github.com/ShopRunner/collie_recs/actions/workflows/compliance-oss.yaml)

Collie is a library for preparing, training, and evaluating implicit deep learning hybrid recommender systems, named after the Border Collie dog breed.

Collie offers a collection of simple APIs for preparing and splitting datasets, incorporating item metadata directly into a model architecture or loss, efficiently evaluating a model's performance on the GPU, and so much more. Above all else though, Collie is built with flexibility and customization in mind, allowing for faster prototyping and experimentation.

See the documentation for more details.

![](https://net-shoprunner-scratch-data-science.s3.amazonaws.com/njones/collie/collie-banner.png)
> "We adopted 2 Border Collies a year ago and they are about 3 years old. They are completely obsessed with fetch and tennis balls and it's getting out of hand. They live in the fenced back yard and when anyone goes out there they instantly run around frantically looking for a tennis ball. If there is no ball they will just keep looking and will not let you pet them. When you do have a ball, they are 100% focused on it and will not notice anything else going on around them, like it's their whole world."
>
> -- *A Reddit thread on r/DogTraining*

## Installation
```bash
pip install collie_recs
```

## Quick Start
Creating and evaluating an implicit matrix factorization model with MovieLens 100K data is simple with Collie:
```python
from collie_recs.cross_validation import stratified_split
from collie_recs.interactions import Interactions
from collie_recs.metrics import auc, evaluate_in_batches, mapk, mrr
from collie_recs.model import MatrixFactorizationModel, CollieTrainer
from collie_recs.movielens import read_movielens_df
from collie_recs.utils import convert_to_implicit


# read in MovieLens 100K data
df = read_movielens_df()

# convert the data to implicit
df_imp = convert_to_implicit(df)

# store data as ``Interactions``
interactions = Interactions(users=df_imp['user_id'],
                            items=df_imp['item_id'],
                            allow_missing_ids=True)

# perform a data split
train, val = stratified_split(interactions)

# train an implicit ``MatrixFactorization`` model
model = MatrixFactorizationModel(train=train,
                                 val=val,
                                 embedding_dim=10,
                                 lr=1e-1,
                                 loss='adaptive',
                                 optimizer='adam')
trainer = CollieTrainer(model, max_epochs=10)
trainer.fit(model)
model.freeze()

# evaluate the model
auc_score, mrr_score, mapk_score = evaluate_in_batches([auc, mrr, mapk], val, model)

print(f'AUC:          {auc_score}')
print(f'MRR:          {mrr_score}')
print(f'MAP@10:       {mapk_score}')
```

More complicated examples of pipelines can be viewed [for MovieLens 100K data here](collie_recs/movielens/run.py), [in notebooks here](collie_recs/tutorials), and documentation here.

## Comparison With Other Open-Source Recommendation Libraries

*On some smaller screens, you might have to scroll right to see the full table.* ➡️

| Aspect Included in Library | <a href="http://surpriselib.com" target="_blank">Surprise</a> | <a href="https://making.lyst.com/lightfm/docs/home.html" target="_blank">LightFM</a> | <a href="https://docs.fast.ai" target="_blank">FastAI</a> | <a href="https://maciejkula.github.io/spotlight/" target="_blank">Spotlight</a> | <a href="https://recbole.io" target="_blank">RecBole</a> | <a href="https://www.tensorflow.org/recommenders" target="_blank">TensorFlow Recommenders</a> | Collie |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Implicit data support** for when we only know when a user interacts with an item or not, not the explicit rating the user gave the item |  | ✓ |  | ✓ | ✓ | ✓ | ✓ |
| **Explicit data support** for when we know the explicit rating the user gave the item | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | * |
| Support for **side-data** incorporated directly into the models |  |  |  |  | ✓ | ✓ | ✓ |
| Support a **flexible framework for new model architectures** and experimentation |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Deep learning** libraries utilizing speed-ups with a GPU and able to implement new, cutting-edge deep learning algorithms  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Automatic support for multi-GPU training**  |  |  |  |  |  |  | ✓ |
| **Actively supported and maintained**  | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ |
| **Scalable for larger, out-of-memory datasets**  |  |  |  |  |  | ✓ | ✓ |
| Includes **model zoo** with two or model architectures implemented  |  |  |  | ✓ | ✓ |  | ✓ |
| Includes **implicit loss functions** for training and **metric functions** for model evaluation  |  | ✓ |  | ✓ | ✓ |  | ✓ |
| Includes **adaptive loss functions** for multiple negative examples  |  | ✓ |  | ✓ |  |  | ✓ |
| Includes **loss functions that account for side-data**  |  |  |  |  |  |  | ✓ |

<sup>* Coming soon!</sup>


The following table notes shows the results of an experiment training and evaluating recommendation models in some popular implicit recommendation model frameworks on a common [MovieLens 10M](https://grouplens.org/datasets/movielens/10m/) dataset. The data was split via a 90/5/5 stratified data split. Each model was trained for a maximum of 40 epochs using an embedding dimension of 32. For each model, we used default hyperparameters (unless otherwise noted below).

| Model | MAP@10 Score | Notes |
| ----- | :----------: | :---: |
| [Logistic MF](https://implicit.readthedocs.io/en/latest/lmf.html)         | 0.0128     | Using the CUDA implementation.        |
| [LightFM](https://making.lyst.com/lightfm/docs/home.html) with BPR Loss   | 0.0180     |                                       |
| [ALS](https://implicit.readthedocs.io/en/latest/als.html)                 | 0.0189     | Using the CUDA implementation.        |
| [BPR](https://implicit.readthedocs.io/en/latest/bpr.html)                 | 0.0301     | Using the CUDA implementation.        |
| [Spotlight](https://maciejkula.github.io/spotlight/index.html)            | 0.0376     | Using adaptive hinge loss.            |
| [LightFM](https://making.lyst.com/lightfm/docs/home.html) with WARP Loss  | 0.0412     |                                       |
| Collie ``MatrixFactorizationModel``                                       | **0.0425** | Using a separate SGD bias optimizer.  |

At ShopRunner, we have found Collie models outperform comparable LightFM models with up to **64% improved MAP@10 scores**.

## Development
To run locally, begin by creating a data path environment variable:

```bash
# Define where on your local hard drive you want to store data. It is best if this
# location is not inside the repo itself. An example is below
export DATA_PATH=$HOME/data/collie_recs
```

Run development from within the Docker container:
```bash
docker build -t collie_recs .

# run the container in interactive mode, leaving port ``8888`` open for Jupyter
docker run \
    -it \
    --rm \
    -v "${DATA_PATH}:/data" \
    -v "${PWD}:/collie_recs" \
    -p 8888:8888 \
    collie_recs /bin/bash
```

### Run on a GPU:
```bash
docker build -t collie_recs .

# run the container in interactive mode, leaving port ``8888`` open for Jupyter
docker run \
    -it \
    --rm \
    --gpus all \
    -v "${DATA_PATH}:/data" \
    -v "${PWD}:/collie_recs" \
    -p 8888:8888 \
    collie_recs /bin/bash
```

### Start JupyterLab
To run JupyterLab, start the container and execute the following:
```bash
jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```
Connect to JupyterLab here: [http://localhost:8888/lab](http://localhost:8888/lab)

### Unit Tests
Library unit tests in this repo are to be run in the Docker container:

```bash
# execute unit tests
pytest --cov-report term --cov=collie_recs
```

Note that a handful of tests require the [MovieLens 100K dataset](https://files.grouplens.org/datasets/movielens/ml-100k.zip) to be downloaded (~5MB in size), meaning that either before or during test time, there will need to be an internet connection. This dataset only needs to be downloaded a single time for use in both unit tests and tutorials.

## Docs
The Collie library supports Read the Docs documentation. To compile locally,
```bash
cd docs
make html

# open local docs
open build/html/index.html
```
