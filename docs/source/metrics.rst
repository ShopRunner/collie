Evaluation Metrics
==================

The Collie library supports evaluating both implicit and explicit models.

Three common implicit recommendation evaluation metrics come out-of-the-box with Collie. These include Area Under the ROC Curve (AUC), Mean Reciprocal Rank (MRR), and Mean Average Precision at K (MAP@K). Each metric is optimized to be as efficient as possible by having all calculations done in batch, tensor form on the GPU (if available). We provide a standard helper function, ``evaluate_in_batches``, to evaluate a model on many metrics in a single pass.

Explicit evaluation of recommendation systems is luckily much more straightforward, allowing us to utilize the `TorchMetrics <https://torchmetrics.readthedocs.io/en/latest/>`_ library for flexible, optimized metric calculations on the GPU accessed through a standard helper function, ``explicit_evaluate_in_batches``, whose API is very similar to its implicit counterpart.

Evaluate in Batches
-------------------

Implicit Evaluate in Batches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: collie.metrics.evaluate_in_batches

Explicit Evaluate in Batches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: collie.metrics.explicit_evaluate_in_batches

Implicit Metrics
----------------

AUC
^^^
.. autofunction:: collie.metrics.auc

MAP@K
^^^^^
.. autofunction:: collie.metrics.mapk

MRR
^^^
.. autofunction:: collie.metrics.mrr
