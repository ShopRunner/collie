MovieLens Functions
===================

.. image:: https://movielens.org/images/movielens-logo.svg
  :align: center

The following functions under ``collie_recs.movielens`` read and prepare MovieLens 100K data, train and evaluate a model on this data, and visualize recommendation results.

Get MovieLens 100K Data
-----------------------

Read MovieLens 100K Interactions Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: collie_recs.movielens.read_movielens_df

Read MovieLens 100K Item Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: collie_recs.movielens.read_movielens_df_item

Read MovieLens 100K Posters Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: collie_recs.movielens.read_movielens_posters_df

Format MovieLens 100K Item Metadata Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: collie_recs.movielens.get_movielens_metadata


MovieLens Model Training Pipeline
---------------------------------

.. autofunction:: collie_recs.movielens.run_movielens_example


Visualize MovieLens Predictions
-------------------------------

.. autofunction:: collie_recs.movielens.get_recommendation_visualizations
