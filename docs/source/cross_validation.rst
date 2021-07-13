Cross Validation
================

Once data is set up in an ``Interactions`` dataset, we can perform a data split to evaluate a trained model later. Collie supports the following two data splits below that share a common API, but differ in split strategy and performance.

.. code-block:: python

   from collie.cross_validation import random_split, stratified_split
   from collie.interactions import Interactions
   from collie.movielens import read_movielens_df
   from collie.utils import convert_to_implicit, Timer


   # EXPERIMENT SETUP
   # read in MovieLens 100K data
   df = read_movielens_df()

   # convert the data to implicit
   df_imp = convert_to_implicit(df)

   # store data as ``Interactions``
   interactions = Interactions(users=df_imp['user_id'],
                               items=df_imp['item_id'],
                               allow_missing_ids=True)

   t = Timer()

   # EXPERIMENT BEGIN
   train, test = random_split(interactions)
   t.timecheck(message='Random split timecheck')

   train, test = stratified_split(interactions)
   t.timecheck(message='Stratified split timecheck')


.. code-block:: bash

   # as expected, a random split is much faster than a stratified split
   Random split timecheck (0.00 min)
   Stratified split timecheck (0.04 min)

Random Split
------------
.. autofunction:: collie.cross_validation.random_split

Stratified Split
----------------
.. autofunction:: collie.cross_validation.stratified_split
