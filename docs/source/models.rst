Models
======

**Instantiating and Training a Collie Model**

Collie provides architectures for several state-of-the-art recommendation model architectures for both non-hybrid and hybrid models, depending on if you would like to directly incorporate metadata into the model.

Since Collie utilizes PyTorch Lightning for model training, all models, by default:

* Are compatible with CPU, GPU, multi-GPU, and TPU training
* Allow for 16-bit precision
* Integrate with common external loggers
* Allow for extensive predefined and custom training callbacks
* Are flexible with minimal boilerplate code

While each model's API differs slightly, generally, the training procedure for each model will look like:

.. code-block:: python

   from collie_recs.model import CollieTrainer, MatrixFactorizationModel


   # assume you have ``interactions`` already defined and ready-to-go

   model = MatrixFactorizationModel(interactions)

   trainer = CollieTrainer(model)
   trainer.fit(model)
   model.eval()

   # now, ``model`` is ready to be used for inference, evaluation, etc.

   model.save_model('model.pkl')

When we have side-data about items, this can be incorporated directly into the loss function of the model. For details on this, see :ref:`Losses`.

Hybrid Collie models also allow incorporating this side-data directly into the model. For an in-depth example of this, see :ref:`Tutorials`.

**Creating a Custom Architecture**

Collie not only houses incredible pre-defined architectures, but was built with customization in mind. All Collie recommendation models are built as subclasses of the ``BasePipeline`` model, inheriting common loss calculation functions and model training boilerplate. This allows for a nice balance between both flexibility and faster iteration.

While any method can be overridden with more architecture-specific implementations, at the bare minimum, each additional model *must* override:

* ``_setup_model`` - Model architecture initialization
* ``forward`` - Model step that accepts a batch of data of form ``(users, items), negative_items`` and outputs a recommendation score for each item

If we wanted to create a custom model that performed a barebones matrix factorization calculation, in Collie, this would be implemented as:

.. code-block:: python

   import torch

   from collie_recs.model import BasePipeline, CollieTrainer, ScaledEmbedding
   from collie_recs.utils import get_init_arguments


   class SimpleModel(BasePipeline):
       def __init__(self, train, val, embedding_dim):
           """
           Initialize a simple model that is a subclass of ``BasePipeline``.

           Parameters
           ----------
           train: ``collie_recs.interactions`` object
           val: ``collie_recs.interactions`` object
           embedding_dim: int
               Number of latent factors to use for user and item embeddings

           """
           super().__init__(**get_init_arguments())

       def _setup_model(self, **kwargs):
           """Method for building model internals that rely on the data passed in."""
           self.user_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_users,
                                                  embedding_dim=self.hparams.embedding_dim)
           self.item_embeddings = ScaledEmbedding(num_embeddings=self.hparams.num_items,
                                                  embedding_dim=self.hparams.embedding_dim)

       def forward(self, users, items):
           """
           Forward pass through the model.

           Parameters
           ----------
           users: tensor, 1-d
               Array of user indices
           items: tensor, 1-d
               Array of item indices

           Returns
           -------
           preds: tensor, 1-d
               Predicted scores

           """
           return torch.mul(
               self.user_embeddings(users), self.item_embeddings(items)
           ).sum(axis=1)


   # assume you have ``train`` and ``val`` already defined and ready-to-go

   model = SimpleModel(train, val, embedding_dim=10)

   trainer = CollieTrainer(model, max_epochs=10)
   trainer.fit(model)
   model.eval()

   # now, ``model`` is ready to be used for inference, evaluation, etc.

   model.save_model('model.pkl')


See the source code for the ``BasePipeline`` in :ref:`Model Templates` below for the calling order of each class method as well as initialization details for optimizers, schedulers, and more.


Standard Models
---------------

Matrix Factorization Model
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.model.MatrixFactorizationModel
    :members:
    :show-inheritance:

Nonlinear Factorization Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.model.NonlinearMatrixFactorizationModel
    :members:
    :show-inheritance:

Neural Collaborative Filtering (NeuCF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.model.NeuralCollaborativeFiltering
    :members:
    :show-inheritance:

Hybrid Models
-------------

Hybrid Pretrained Matrix Factorization Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.model.HybridPretrainedModel
    :members:
    :show-inheritance:

Trainers
--------

PyTorch Lightning Trainer
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.model.CollieTrainer
    :members:
    :inherited-members:
    :show-inheritance:

Non- PyTorch Lightning Trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.model.CollieMinimalTrainer
    :members:

Model Templates
---------------

Base Collie Pipeline Template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.model.BasePipeline
    :members:
    :show-inheritance:

Layers
------

Scaled Embedding
^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.model.ScaledEmbedding
    :members:
    :show-inheritance:

Zero Embedding
^^^^^^^^^^^^^^
.. autoclass:: collie_recs.model.ZeroEmbedding
    :members:
    :show-inheritance:
