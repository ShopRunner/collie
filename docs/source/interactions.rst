Interactions
============

**What are Interactions?**

The ``Interactions`` object is at the core of how data loading and retrieval works in Collie models.

An ``Interactions`` object is, in its simplest form, a ``torch.data.Dataset`` wrapper around a ``scipy.sparse.coo_matrix`` that supports iterating and batching data during model training. We supplement this with data consistency checks during initialization to catch potential errors sooner, a high-throughput and memory-efficient form of negative sampling, and a simple API. Indexing an ``Interactions`` object returns a user ID and an item ID that the user has interacted with, as well as an ``O(1)`` negative sample of item ID(s) a user has *not* interacted with, supporting the implicit loss functions built into Collie.

.. code-block:: python

   import pandas as pd

   from collie_recs.interactions import Interactions


   df = pd.DataFrame(data={'user_id': [0, 0, 0, 1, 1, 2],
                           'item_id': [0, 1, 2, 3, 4, 5]})
   interactions = Interactions(users=df['user_id'], items=df['item_id'], num_negative_samples=2)

   for _ in range(3):
       print(interactions[0])

.. code-block:: bash

  # output structure: ((user IDs, positive item IDs), negative items IDs)
  # notice all negative item IDs will be true negatives for user ``0``, e.g.
  ((0, 0), array([5., 3.]))
  ((0, 0), array([5., 4.]))
  ((0, 0), array([3., 5.]))

We can see this same idea holds when we instead create an ``InteractionsDataLoader``, as such:

.. code-block:: python

   import pandas as pd

   from collie_recs.interactions import InteractionsDataLoader


   df = pd.DataFrame(data={'user_id': [0, 0, 0, 1, 1, 2],
                           'item_id': [0, 1, 2, 3, 4, 5]})
   interactions_loader = InteractionsDataLoader(
       users=df['user_id'], items=df['item_id'], num_negative_samples=2
   )

   for batch in interactions_loader:
       print(batch)

.. code-block:: bash

   # output structure: [[user IDs, positive item IDs], negative items IDs]
   # users and positive items IDs is now a tensor of shape ``batch_size`` and
   # negative items IDs is now a tensor of shape ``batch_size x num_negative_samples``
   # notice all negative item IDs will still be true negatives, e.g.
   [[tensor([0, 0, 0, 1, 1, 2], dtype=torch.int32),
     tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)],
    tensor([[4., 5.],
            [3., 5.],
            [4., 5.],
            [0., 1.],
            [5., 0.],
            [3., 4.]])]

Once data is in an ``Interactions`` form, you can easily perform data splits, train and evaluate a model, and much more. See :ref:`Cross Validation` and :ref:`Models` documentation for more information on this.

**How can I speed up Interactions data loading?**

While an ``Interactions`` object works out-of-the-box with a ``torch.data.DataLoader``, such as the included ``InteractionsDataLoader``, sampling true negatives for each Interactions element can become costly as the number of items grows. In this situation, it might be desirable to *trade exact negative sampling for a faster, approximate sampler*. For these scenarios, we use the ``ApproximateNegativeSamplingInteractionsDataLoader``, an extension of the more traditional ``InteractionsDataLoader`` that samples data in batches, forgoing the expensive concatenation of individual data points an ``InteractionsDataLoader`` must do for each batch. Here, negative samples are simply returned as a collection of randomly sampled item IDs, meaning it is possible that a negative item ID returned for a user can actually be an item a user had positively interacted with. When the number of items is large, though, this scenario is increasingly rare, and the speedup benefit is worth the slight performance hit.

.. code-block:: python

   import pandas as pd

   from collie_recs.interactions import ApproximateNegativeSamplingInteractionsDataLoader


   df = pd.DataFrame(data={'user_id': [0, 0, 0, 1, 1, 2],
                           'item_id': [0, 1, 2, 3, 4, 5]})
   interactions_loader = ApproximateNegativeSamplingInteractionsDataLoader(
       users=df['user_id'], items=df['item_id'], num_negative_samples=2
   )

   for batch in interactions_loader:
       print(batch)

.. code-block:: bash

   # output structure: [[user IDs, positive item IDs], "negative" items IDs]
   # users and positive items IDs is now a tensor of shape ``batch_size`` and
   # negative items IDs is now a tensor of shape ``batch_size x num_negative_samples``
   # notice negative item IDs will *not* always be true negatives now, e.g.
   [[tensor([0, 0, 0, 1, 1, 2], dtype=torch.int32),
     tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)],
    tensor([[4, 5],
            [1, 2],
            [4, 2],
            [3, 5],
            [4, 0],
            [4, 3]])]

   interactions = interactions_loader.interactions
   # use this for cross validation, evaluation, etc.

**What if my data cannot fit in memory?**

For datasets that are too large to fit in memory, Collie includes the ``HDF5InteractionsDataLoader`` (which uses a ``HDF5Interactions`` dataset at its base, sharing many of the same features and methods as an ``Interactions`` object). A ``HDF5InteractionsDataLoader`` applies the same principles behind the ``ApproximateNegativeSamplingInteractionsDataLoader``, but for data stored on disk in a HDF5 format. The main drawback to this approach is that when ``shuffle=True``, data will only be shuffled within batches (as opposed to the true shuffle in ``ApproximateNegativeSamplingInteractionsDataLoader``). For sufficiently large enough data, this effect on model performance should be negligible.

.. code-block:: python

   import pandas as pd

   from collie_recs.interactions import HDF5InteractionsDataLoader
   from collie_recs.utils import pandas_df_to_hdf5


   # we'll write out a sample DataFrame to HDF5 format for this example
   df = pd.DataFrame(data={'user_id': [0, 0, 0, 1, 1, 2],
                           'item_id': [0, 1, 2, 3, 4, 5]})
   pandas_df_to_hdf5(df=df, out_path='sample_hdf5.h5')

   interactions_loader = HDF5InteractionsDataLoader(
       hdf5_path='sample_hdf5.h5',
       user_col='user_id',
       item_col='item_id',
       num_negative_samples=2,
   )

   for batch in interactions_loader:
       print(batch)

.. code-block:: bash

   # output structure: [[user IDs, positive item IDs], "negative" items IDs]
   # users and positive items IDs is now a tensor of shape ``batch_size`` and
   # negative items IDs is now a tensor of shape ``batch_size x num_negative_samples``
   # notice negative item IDs will *not* always be true negatives now, e.g.
   [[tensor([0, 0, 0, 1, 1, 2], dtype=torch.int32),
     tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)],
    tensor([[5, 4],
            [4, 5],
            [5, 2],
            [4, 3],
            [4, 2],
            [1, 3]])]

The table below shows the time differences to train a ``MatrixFactorizationModel`` for a single epoch on |movielens_10m_readme| data using default parameters on the GPU on a ``p3.2xlarge`` EC2 instance [#f1]_.

+-------------------------------------------------------+--------------------------------+
| DataLoader Type                                       | Time to Train a Single Epoch   |
+=======================================================+================================+
| ``InteractionsDataLoader``                            | 1min 25s                       |
+-------------------------------------------------------+--------------------------------+
| ``ApproximateNegativeSamplingInteractionsDataLoader`` | 1min 8s                        |
+-------------------------------------------------------+--------------------------------+
| ``HDF5InteractionsDataLoader``                        | 1min 10s                       |
+-------------------------------------------------------+--------------------------------+


Datasets
--------

Implicit Interactions Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.interactions.Interactions
    :members:
    :inherited-members:
    :show-inheritance:

Explicit Interactions Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.interactions.ExplicitInteractions
    :members:
    :inherited-members:
    :show-inheritance:

HDF5 Interactions Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.interactions.HDF5Interactions
    :members:
    :inherited-members:
    :show-inheritance:

DataLoaders
-----------

Interactions DataLoader
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.interactions.InteractionsDataLoader
    :members:
    :inherited-members:
    :show-inheritance:

Approximate Negative Sampling Interactions DataLoader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.interactions.ApproximateNegativeSamplingInteractionsDataLoader
    :members:
    :inherited-members:
    :show-inheritance:

HDF5 Approximate Negative Sampling Interactions DataLoader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: collie_recs.interactions.HDF5InteractionsDataLoader
    :members:
    :inherited-members:
    :show-inheritance:

.. |movielens_10m_readme| raw:: html

   <a href="http://files.grouplens.org/datasets/movielens/ml-10m-README.html" target="_blank">MovieLens 10M</a>

.. rubric:: Footnotes

.. [#f1] Welcome to a detailed footnote about this experiment!

  * The MovieLens 10M data was preprocessed using Collie utility functions in :ref:`Utility Functions` that keeps all ratings above a ``4`` and removes users with fewer than 3 interactions. This left us with ``5,005,398`` total interactions.

  * For a much faster training time, we recommend setting ``sparse=True`` (see the point below this) in the model definition and using a larger batch size with ``pin_memory=True`` in the DataLoader.

  * Since we used default parameters, the embeddings of the ``MatrixFactorizationModel`` were not sparse. Had we used sparse embeddings and a Sparse Adam optimizer, the table would show:

  +-------------------------------------------------------+--------------------------------+
  | DataLoader Type                                       | Time to Train a Single Epoch   |
  +=======================================================+================================+
  | ``InteractionsDataLoader``                            | 1min 21s                       |
  +-------------------------------------------------------+--------------------------------+
  | ``ApproximateNegativeSamplingInteractionsDataLoader`` | 1min 4s                        |
  +-------------------------------------------------------+--------------------------------+
  | ``HDF5InteractionsDataLoader``                        | 1min 7s                        |
  +-------------------------------------------------------+--------------------------------+

     These times are more dramatically different with larger datasets (1M+ items). While these options are certainly faster, having sparse settings be the default limits the optimizer options and general flexibility of customizing an architecture, since not all PyTorch operations support sparse layers. For that reason, we made the default parameters non-sparse, which works best for small-sized datasets.

  * We have also noticed drastic changes in training time depending on the version of PyTorch used. While we used ``torch@1.8.0`` here, we have noticed the fastest training times using ``torch@1.6.0``. If you understand why this is, make a PR updating these docs with that information!
