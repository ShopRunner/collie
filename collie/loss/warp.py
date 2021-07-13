from typing import Dict, Optional, Union

import torch

from collie.loss.metadata_utils import ideal_difference_from_metadata


def warp_loss(
    positive_scores: torch.tensor,
    many_negative_scores: torch.tensor,
    num_items: int,
    positive_items: Optional[torch.tensor] = None,
    negative_items: Optional[torch.tensor] = None,
    metadata: Optional[Dict[str, torch.tensor]] = dict(),
    metadata_weights: Optional[Dict[str, float]] = dict(),
) -> torch.tensor:
    """
    Modified WARP loss function [4]_.

    See http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf for loss equation.

    See ``ideal_difference_from_metadata`` docstring for more info on how metadata is used.

    Parameters
    ----------
    positive_scores: torch.tensor, 1-d
        Tensor containing scores for known positive items of shape
        ``num_negative_samples x batch_size``
    many_negative_scores: torch.tensor, 2-d
        Iterable of tensors containing scores for many (n > 1) sampled negative items of shape
        ``num_negative_samples x batch_size``. More tensors increase the likelihood of finding
        ranking-violating pairs, but risk overfitting
    num_items: int
        Total number of items in the dataset
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
    .. [4] Weston et al. WSABIE: Scaling Up To Large Vocabulary Image Annotation.
        www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf.

    """
    if negative_items is not None and positive_items is not None:
        positive_items = positive_items.repeat([many_negative_scores.shape[0], 1])

    if metadata is not None and len(metadata) > 0:
        ideal_difference = ideal_difference_from_metadata(
            positive_items=positive_items,
            negative_items=negative_items,
            metadata=metadata,
            metadata_weights=metadata_weights,
        ).transpose(1, 0)
    else:
        ideal_difference = 1

    # device to put new tensors on
    device = positive_scores.device

    # WARP loss requires a different structure for positive and negative samples
    positive_scores = positive_scores.view(len(positive_scores), 1)
    many_negative_scores = torch.transpose(many_negative_scores, 0, 1)

    batch_size, max_trials = many_negative_scores.size(0), many_negative_scores.size(1)

    flattened_new_row_indices = torch.arange(0, batch_size, 1).long().to(device) * (max_trials + 1)
    tensor_of_ones = torch.ones(batch_size, 1).float().to(device)

    # ``initial_loss`` is essentially just hinge loss for now
    hinge_loss = ideal_difference - positive_scores + many_negative_scores

    # Add column of ones to so we know when we have used all our attempts. This is used for indexing
    # and computing ``should_count_loss`` if no real value is above 0.
    initial_loss_with_ones = torch.cat([hinge_loss, tensor_of_ones], dim=1)
    # this will be modified in ``_find_first_loss_violation``
    initial_loss_with_ones_binary = torch.cat([hinge_loss, tensor_of_ones], dim=1)

    number_of_tries = _find_first_loss_violation(initial_loss_with_ones_binary, device)

    prediction_index_for_flattened_predictions = number_of_tries + flattened_new_row_indices

    number_of_tries = (number_of_tries + 1).float()

    # IMPORTANT CHANGE: normal WARP weighting has the numerator set to ``num_items - 1``, but we
    # have found this does not penalize when the last item in a negative item sequence ranks above a
    # positive item score. Adjusting the numerator as below penalizes this correctly. Additionally,
    # adding a floor function to the numerator can also have the same negative effect of not
    # not counting loss. See the original implementation as a comment below, and our modified,
    # harsher calculation implemented below.
    # loss_weights = torch.log(torch.floor((num_items - 1) / number_of_tries))
    loss_weights = torch.log((num_items / number_of_tries))

    # don't count loss if we used max number of attempts looking for a violation and didn't find one
    should_we_count_loss = (number_of_tries <= max_trials).float()

    loss = (
        loss_weights
        * (
            initial_loss_with_ones.flatten()[prediction_index_for_flattened_predictions]
        )
        * should_we_count_loss
    )

    return (loss.sum() + loss.pow(2).sum()) / len(positive_scores)


def _find_first_loss_violation(losses: torch.tensor,
                               device: Union[str, torch.device, torch.cuda.device]) -> torch.tensor:
    """
    Find the index of the first violation where ``1 - positive_score + negative_score`` is greater
    than 0.

    """
    # set all negative losses to 0 and all positive losses to 1
    losses[losses < 0] = 0
    losses[losses > 0] = 1

    # after this, maximum value will be the first non-zero (bad) loss
    reverse_indices = torch.arange(losses.shape[1], 0, -1).to(device)
    min_index_of_good_loss = losses * reverse_indices

    # report the first loss that is positive here (not 0-based indexed)
    number_of_tries = torch.argmax(min_index_of_good_loss, 1, keepdim=True).flatten()

    return number_of_tries
