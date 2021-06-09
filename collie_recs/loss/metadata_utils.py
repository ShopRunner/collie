from typing import Dict, Optional

import torch


def ideal_difference_from_metadata(
    positive_items: torch.tensor,
    negative_items: torch.tensor,
    metadata: Optional[Dict[str, torch.tensor]],
    metadata_weights: Optional[Dict[str, float]],
) -> torch.tensor:
    """
    Helper function to calculate the ideal score difference between the positive and negative items.

    Without considering metadata, the ideal score difference would be 1.0 (since the function looks
    at a pair of items, one positive item and one negative item). Taking metadata into
    consideration, the ideal score difference should be between 0 and 1 if there is a partial match
    (not the same item, but matching metadata - e.g. the same film genre). This function calculates
    that ideal difference when there is metadata available.

    Metadata passed in to this function is independent of metadata given to the model during
    training - it can be the same data or a different set. For example, one might use genre
    embeddings as metadata during training and use genre labels as metadata during loss calculation
    (since all metadata passed in to this function must be categorical).

    Parameters
    ----------
    positive_items: torch.tensor, 1-d
        Tensor containing IDs for known positive items
    negative_items: torch.tensor, 1-d
        Tensor containing IDs for sampled negative items
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
    ideal difference: torch.tensor
        Tensor with the same shape as ``positive_items``, with each element between 0 and 1

    """
    weight_sum = sum(metadata_weights.values())
    if weight_sum > 1:
        raise ValueError(f'sum of metadata weights was {weight_sum}, must be <=1')

    match_frac = torch.zeros(positive_items.shape).to(positive_items.device)
    for k, array in metadata.items():
        array = array.squeeze()
        match_frac += (
            array[positive_items.long()] == array[negative_items.long()]
        ).int().to(positive_items.device)*metadata_weights[k]

    return 1.0 - match_frac
