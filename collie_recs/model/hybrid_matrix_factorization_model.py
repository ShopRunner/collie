import warnings

from collie_recs.model.hybrid_pretrained_matrix_factorization import HybridPretrainedModel


class HybridPretrainedModel(HybridPretrainedModel):
    """Filename is deprecated in favor of ``hybrid_pretrained_matrix_factorization.py``."""
    def __init__(self, *args, **kwargs):
        warning_message = (
            'This import path is deprecated in favor of the model in '
            '``hybrid_pretrained_matrix_factorization.py``. This file will be removed in v0.6.0.'
        )
        warnings.warn(warning_message, DeprecationWarning, stacklevel=2)

        super().__init__(*args, **kwargs)
