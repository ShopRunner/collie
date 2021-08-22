from functools import partial
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collie.interactions import SequentialInteractions
from collie.model.base import BasePipeline, ScaledEmbedding, ZeroEmbedding
from collie.utils import get_init_arguments  # , merge_docstrings


class SequentialCNNModel(BasePipeline):
    """
    TODO.

    All `MatrixFactorizationModel` instances are subclasses of the `LightningModule` class provided
    by PyTorch Lightning. This means to train a model, you will need a `collie.model.CollieTrainer`
    object, but the model can be saved and loaded without this `Trainer` instance. Example usage
    may look like:

    .. code-block:: python

        from collie.model import MatrixFactorizationModel, CollieTrainer

        model = MatrixFactorizationModel(train=train)
        trainer = CollieTrainer(model)
        trainer.fit(model)
        model.freeze()

        # do evaluation as normal with `model`

        model.save_model(filename='model.pth')
        new_model = MatrixFactorizationModel(load_model_path='model.pth')

        # do evaluation as normal with `new_model`

    Parameters
    ----------
    train: `collie.interactions` object
        Data loader for training data. If an `Interactions` object is supplied, an
        `InteractionsDataLoader` will automatically be instantiated with `shuffle=True`. Note that
        when the model class is saved, datasets will NOT be saved as well
    val: `collie.interactions` object
        Data loader for testing data. If an `Interactions` object is supplied, an
        `InteractionsDataLoader` will automatically be instantiated with `shuffle=False`. Note that
        when the model class is saved, datasets will NOT be saved as well
    embedding_dim: int
        Number of latent factors to use for user and item embeddings
    kernel_width: tuple or int, optional
        The kernel width of the convolutional layers. If tuple, should contain
        the kernel widths for all convolutional layers. If int, it will be
        expanded into a tuple to match the number of layers.
    dilation: tuple or int, optional
        The dilation factor for atrous convolutions. Setting this to a number
        greater than 1 inserts gaps into the convolutional layers, increasing
        their receptive field without increasing the number of parameters.
        If tuple, should contain the dilation factors for all convolutional
        layers. If int, it will be expanded into a tuple to match the number
        of layers.
    num_layers: int, optional
        Number of stacked convolutional layers.
    nonlinearity: string, optional
        One of ('tanh', 'relu', 'leaky'). Denotes the type of non-linearity to apply
        after each convolutional layer.
    residual_connections: boolean, optional
        Whether to use residual connections between convolutional layers.
    sparse: bool
        Whether or not to treat embeddings as sparse tensors. If `True`, cannot use weight decay on
        the optimizer
    lr: float
        Embedding layer learning rate
    bias_lr: float
        Bias terms learning rate. If 'infer', will set equal to `lr`
    lr_scheduler_func: torch.optim.lr_scheduler
        Learning rate scheduler to use during fitting
    weight_decay: float
        Weight decay passed to the optimizer, if optimizer permits
    optimizer: torch.optim or str
        If a string, one of the following supported optimizers:
            * 'sgd' (torch.optim.SGD)
            * 'adam' (torch.optim.Adam)
            * 'sparse_adam' (torch.optim.SparseAdam)
    bias_optimizer: torch.optim or str
        Optimizer for the bias terms. This supports the same string options as `optimizer`, with the
        addition of `infer`, which will set the optimizer equal to `optimizer`. If `bias_optimizer`
        is `None`, only a single optimizer will be created for all model parameters
    loss: function or str
        If a string, one of the following implemented losses:
            * 'bpr'
            * 'hinge'
            * 'adaptive' / 'adaptive_hinge'
            * 'adaptive_bpr'
        If `train.num_negative_samples > 1`, the adaptive loss version will automatically be used
    metadata_for_loss: dict
        Keys should be strings identifing each metadata type that match keys in `metadata_weights`.
        Values should be a `torch.tensor` of shape (num_items x 1). Each tensor should contain
        categorical metadata information about items (e.g. a number representing the genre of the
        item)
    metadata_for_loss_weights: dict
        Keys should be strings identifing each metadata type that match keys in `metadata`. Values
        should be amount of weight to place on a match of that type of metadata, which the sum of
        all values `<= 1`.
        e.g. If `metadata_weights = {'genre': .3, 'director': .2}`, then an item is:
            a 100% match if it's the same item,
            a 50% match if it's a different item with the same genre and same director,
            a 30% match if it's a different item with the same genre and different director,
            etc.
    y_range: tuple
        Specify as `(min, max)` to apply a sigmoid layer to the output score of the model to get
        predicted ratings within the range of `min` and `max`
    load_model_path: str or Path
        To load a previously-saved model, pass in path to output of `model.save_model()` method. If
        `None`, will initialize model as normal
    map_location: str or torch.device
        If `load_model_path` is provided, device specifying how to remap storage locations when
        `torch.load`ing the state dictionary

    """
    def __init__(self,
                 train: SequentialInteractions = None,
                 val: SequentialInteractions = None,
                 embedding_dim: int = 30,
                 kernel_width: int = 3,
                 dilation: int = 1,
                 num_layers: int = 1,
                 nonlinearity: Union[str, Callable] = 'tanh',
                 residual_connections: int = True,
                 sparse: bool = False,
                 lr: float = 1e-3,
                 bias_lr: Optional[Union[float, str]] = 1e-2,
                 lr_scheduler_func: Optional[Callable] = partial(ReduceLROnPlateau,
                                                                 patience=1,
                                                                 verbose=True),
                 weight_decay: float = 0.0,
                 optimizer: Union[str, Callable] = 'adam',
                 bias_optimizer: Optional[Union[str, Callable]] = 'sgd',
                 loss: Union[str, Callable] = 'hinge',
                 metadata_for_loss: Optional[Dict[str, torch.tensor]] = None,
                 metadata_for_loss_weights: Optional[Dict[str, float]] = None,
                 y_range: Optional[Tuple[float, float]] = None,
                 load_model_path: Optional[str] = None,
                 map_location: Optional[str] = None):

        kernel_width = self._to_iterable(kernel_width, num_layers)
        dilation = self._to_iterable(dilation, num_layers)

        if not callable(nonlinearity):
            if nonlinearity == 'tanh':
                nonlinearity = F.tanh
            elif nonlinearity == 'relu':
                nonlinearity = F.relu
            elif 'leaky' in nonlinearity:
                nonlinearity = F.leaky_relu
            else:
                raise ValueError(
                    f"Nonlinearity can be one of ['tanh', 'relu', 'leaky'], not f{nonlinearity}!"
                )

        super().__init__(**get_init_arguments())

    def _to_iterable(self, val, num):
        try:
            return_val = iter(val)
        except TypeError:
            return_val = (val,) * num

        return return_val

    def _setup_model(self, **kwargs) -> None:
        """
        Method for building model internals that rely on the data passed in.

        This method will be called after `prepare_data`.

        """
        self.item_embeddings = ScaledEmbedding(
            (self.hparams.num_items + 1),
            self.hparams.embedding_dim,
            padding_idx=-1,
            sparse=self.hparams.sparse,
        )

        self.item_biases = ZeroEmbedding(
            (self.hparams.num_items + 1),
            1,
            sparse=self.hparams.sparse,
            padding_idx=-1,
        )

        self.cnn_layers = [
            nn.Conv2d(
                self.hparams.embedding_dim,
                self.hparams.embedding_dim,
                (_kernel_width, 1),
                dilation=(_dilation, 1),
            )
            for (_kernel_width, _dilation) in zip(self.hparams.kernel_width, self.hparams.dilation)
        ]

        for i, layer in enumerate(self.cnn_layers):
            self.add_module('cnn_{}'.format(i), layer)

    def compute_user_representation(self, item_sequences, predicting=False):
        """
        Compute user representation from a given sequence.

        Returns
        -------
        tuple (all_representations, final_representation)
            The first element contains all representations from step
            -1 (no items seen) to t - 1 (all but the last items seen).
            The second element contains the final representation
            at step t (all items seen). This final state can be used
            for prediction or evaluation.

        """
        # replace all ``-1`` values in sequence with ``padding_idx``
        item_sequences[item_sequences == -1] = self.item_embeddings.padding_idx

        item_embeddings = self.item_embeddings(item_sequences)

        # make the embedding dimension the channel dimension
        sequence_embeddings = item_embeddings.permute(0, 2, 1)
        # add a trailing dimension of 1
        sequence_embeddings = sequence_embeddings.unsqueeze(3)

        # pad so that the CNN doesn't have the future of the sequence in its receptive field
        receptive_field_width = (
            self.hparams.kernel_width[0]
            + (self.hparams.kernel_width[0] - 1)
            * (self.hparams.dilation[0] - 1)
        )

        x = F.pad(sequence_embeddings, (0, 0, receptive_field_width, 0))
        x = self.hparams.nonlinearity(self.cnn_layers[0](x))

        if self.hparams.residual_connections:
            residual = F.pad(sequence_embeddings, (0, 0, 1, 0))
            x = x + residual

        for (cnn_layer, kernel_width, dilation) in zip(
            self.cnn_layers[1:], self.hparams.kernel_width[1:], self.hparams.dilation[1:]
        ):
            receptive_field_width = kernel_width + (kernel_width - 1) * (dilation - 1)
            residual = x
            x = F.pad(x, (0, 0, receptive_field_width - 1, 0))
            x = self.hparams.nonlinearity(cnn_layer(x))

            if self.hparams.residual_connections:
                x = x + residual

        x = x.squeeze(3)

        if predicting is False:
            # return everything up to the final item in the sequence (the target)
            return x[:, :, :-1]
        else:
            # return everything including the final item in the sequence (the target)
            return x[:, :, -1]

    # TODO: do the ``user_representation`` calculation in the forward depending on if we want
    # to include target or not, meaning we need to add another parameter here
    def forward(self, user_representations, targets):
        """
        Compute predictions for target items given user representations.

        Parameters
        ----------
        user_representations: tensor
            Result of the user_representation_method.
        targets: tensor
            Minibatch of item sequences of shape
            (minibatch_size, sequence_length).

        Returns
        -------
        predictions: tensor
            Of shape (minibatch_size, sequence_length).

        """
        item_embeddings = self.item_embeddings(targets)

        # Make the embedding dimension the channel dimension
        target_embedding = item_embeddings.permute(0, 2, 1).squeeze()
        target_bias = self.item_biases(targets).squeeze()

        dot = (user_representations * target_embedding).sum(1).squeeze()

        return target_bias + dot

    def predict(self, sequences: Iterable[int], item_ids: Iterable[int] = None):
        """
        Make predictions: given a sequence of interactions, predict
        the next item in the sequence.

        Parameters
        ----------
        sequences: array, (1 x max_sequence_length)
            Array containing the indices of the items in the sequence.
        item_ids: array (num_items x 1), optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.

        Returns
        -------
        predictions: array
            Predicted scores for all items in item_ids.

        """
        self.train(False)

        sequences = np.atleast_2d(sequences)

        if item_ids is None:
            item_ids = np.arange(self.hparams.num_items).reshape(-1, 1)

        sequences = torch.from_numpy(sequences.astype(np.int64).reshape(1, -1))
        item_ids = torch.from_numpy(item_ids.astype(np.int64))

        sequence_var = sequences.to(self.device)
        item_var = item_ids.to(self.device)

        sequence_representations = self.compute_user_representation(sequence_var, predicting=True)
        size = (len(item_var),) + sequence_representations.size()[1:]
        out = self(sequence_representations.expand(*size), item_var)

        return out.cpu().detach().numpy().flatten()

    def get_item_predictions(self, sequence, unseen_items_only=True, sorted=True):
        """TODO."""
        # get predictions from model
        padded_sequence = (
            np.pad(sequence, (len(self.train_loader.interactions) - len(sequence), 0), 'constant')
        )
        predictions = self.predict(padded_sequence)
        predictions = pd.Series(predictions)

        if sorted:
            predictions = predictions.sort_values(ascending=False)

        if unseen_items_only:
            idxs_to_drop = set(sequence)
            filtered_preds = predictions.drop(idxs_to_drop)

            return filtered_preds

        return predictions

    def _get_item_embeddings(self) -> np.array:
        """Get item embeddings."""
        return self.item_embeddings(
            torch.arange(self.hparams.num_items, device=self.device)
        ).detach().cpu()
