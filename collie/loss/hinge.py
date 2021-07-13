from typing import Any, Dict, Optional

import torch

from collie.loss.metadata_utils import ideal_difference_from_metadata


def hinge_loss(
    positive_scores: torch.tensor,
    negative_scores: torch.tensor,
    num_items: Optional[Any] = None,
    positive_items: Optional[torch.tensor] = None,
    negative_items: Optional[torch.tensor] = None,
    metadata: Optional[Dict[str, torch.tensor]] = dict(),
    metadata_weights: Optional[Dict[str, float]] = dict(),
) -> torch.tensor:
    """
    Modified hinge pairwise loss function [2]_.

    See ``ideal_difference_from_metadata`` docstring for more info on how metadata is used.

    Modified from ``Spotlight``:
    https://github.com/maciejkula/spotlight/blob/master/spotlight/losses.py

    Parameters
    ----------
    positive_scores: torch.tensor, 1-d
        Tensor containing scores for known positive items
    negative_scores: torch.tensor, 1-d
        Tensor containing scores for a single sampled negative item
    num_items: Any
        Ignored, included only for compatability with WARP loss
    positive_items: torch.tensor, 1-d
        Tensor containing ids for known positive items of shape ``1 x batch_size``. This is only
        needed if ``metadata`` is provided
    negative_items: torch.tensor, 1-d
        Tensor containing ids for randomly-sampled negative items of shape ``1 x batch_size``. This
        is only needed if ``metadata`` is provided
    metadata: dict
        Keys should be strings identifying each metadata type that match keys in
        ``metadata_weights``. Values should be a ``torch.tensor`` of shape (num_items x 1). Each
        tensor should contain categorical metadata information about items (e.g. a number
        representing the genre of the item)
    metadata_weights: dict
        Keys should be strings identifying each metadata type that match keys in ``metadata``.
        Values should be the amount of weight to place on a match of that type of metadata, with the
        sum of all values ``<= 1``.
        e.g. If ``metadata_weights = {'genre': .3, 'director': .2}``, then an item is:

        * a 100% match if it's the same item,

        * a 50% match if it's a different item with the same genre and same director,

        * a 30% match if it's a different item with the same genre and different director,

        * a 20% match if it's a different item with a different genre and same director,

        * a 0% match if it's a different item with a different genre and different director,
          which is equivalent to the loss without any partial credit

    Returns
    -------
    loss: torch.tensor

    References
    ----------
    .. [2] "Hinge Loss." Wikipedia, Wikimedia Foundation, 5 Mar. 2021, en.wikipedia.org/wiki/
        Hinge_loss.

    """
    score_difference = (positive_scores - negative_scores)

    if metadata is not None and len(metadata) > 0:
        ideal_difference = ideal_difference_from_metadata(
            positive_items=positive_items,
            negative_items=negative_items,
            metadata=metadata,
            metadata_weights=metadata_weights,
        )
    else:
        ideal_difference = 1

    loss = torch.clamp((ideal_difference - score_difference), min=0)

    return (loss.sum() + loss.pow(2).sum()) / len(positive_scores)


def adaptive_hinge_loss(
    positive_scores: torch.tensor,
    many_negative_scores: torch.tensor,
    num_items: Optional[Any] = None,
    positive_items: Optional[torch.tensor] = None,
    negative_items: Optional[torch.tensor] = None,
    metadata: Optional[Dict[str, torch.tensor]] = dict(),
    metadata_weights: Optional[Dict[str, float]] = dict(),
) -> torch.tensor:
    """
    Modified adaptive hinge pairwise loss function [3]_.

    Approximates WARP loss by taking the maximum of negative predictions for each user and sending
    this to hinge loss.

    See ``ideal_difference_from_metadata`` docstring for more info on how metadata is used.

    Modified from ``Spotlight``:
    https://github.com/maciejkula/spotlight/blob/master/spotlight/losses.py

    Parameters
    ----------
    positive_scores: torch.tensor, 1-d
        Tensor containing scores for known positive items of shape
        ``num_negative_samples x batch_size``
    many_negative_scores: torch.tensor, 2-d
        Iterable of tensors containing scores for many (n > 1) sampled negative items of shape
        ``num_negative_samples x batch_size``. More tensors increase the likelihood of finding
        ranking-violating pairs, but risk overfitting
    num_items: Any
        Ignored, included only for compatability with WARP loss
    positive_items: torch.tensor, 1-d
        Tensor containing ids for known positive items of shape
        ``num_negative_samples x batch_size``. This is only needed if ``metadata`` is provided
    negative_items: torch.tensor, 2-d
        Tensor containing ids for sampled negative items of shape
        ``num_negative_samples x batch_size``. This is only needed if ``metadata`` is provided
    metadata: dict
        Keys should be strings identifying each metadata type that match keys in
        ``metadata_weights``. Values should be a ``torch.tensor`` of shape (num_items x 1). Each
        tensor should contain categorical metadata information about items (e.g. a number
        representing the genre of the item)
    metadata_weights: dict
        Keys should be strings identifying each metadata type that match keys in ``metadata``.
        Values should be the amount of weight to place on a match of that type of metadata, with the
        sum of all values ``<= 1``.
        e.g. If ``metadata_weights = {'genre': .3, 'director': .2}``, then an item is:

        * a 100% match if it's the same item,

        * a 50% match if it's a different item with the same genre and same director,

        * a 30% match if it's a different item with the same genre and different director,

        * a 20% match if it's a different item with a different genre and same director,

        * a 0% match if it's a different item with a different genre and different director,
          which is equivalent to the loss without any partial credit

    Returns
    -------
    loss: torch.tensor

    References
    ----------
    .. [3] Kula, Maciej. "Loss Functions." Loss Functions - Spotlight Documentation,
        maciejkula.github.io/spotlight/losses.html.

    """
    highest_negative_scores, highest_negative_inds = torch.max(many_negative_scores, 0)

    if negative_items is not None and positive_items is not None:
        negative_items = (
            negative_items[highest_negative_inds, torch.arange(len(positive_items))].squeeze()
        )

    return hinge_loss(
        positive_scores,
        highest_negative_scores.squeeze(),
        positive_items=positive_items,
        negative_items=negative_items,
        metadata=metadata,
        metadata_weights=metadata_weights,
    )
