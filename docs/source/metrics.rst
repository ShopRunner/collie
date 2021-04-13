Evaluation Metrics
============================

The Collie library supports three common implicit recommendation evaluation metrics out-of-the-box. These include Area Under the ROC Curve (AUC), Mean Reciprocal Rank (MRR), and Mean Average Precision at K (MAP@K). Each metric is optimized to be as efficient as possible by having all calculations done in batch, tensor form on the GPU (if available). We provide a standard helper function, ``evaluate_in_batches``, to evaluate a model on many metrics in a single pass.

Evaluate in Batches
-------------------
.. autofunction:: collie_recs.metrics.evaluate_in_batches

Metrics
-------

AUC
^^^
.. autofunction:: collie_recs.metrics.auc

MAP@K
^^^^^
.. autofunction:: collie_recs.metrics.mapk

MRR
^^^
.. autofunction:: collie_recs.metrics.mrr
