from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import torch

from collie.loss import (adaptive_bpr_loss,
                         adaptive_hinge_loss,
                         bpr_loss,
                         hinge_loss,
                         ideal_difference_from_metadata,
                         warp_loss)


def test_ideal_difference_from_metadata_error(
    positive_items,
    negative_items,
    metadata_a,
    metadata_b,
):
    with pytest.raises(
        ValueError,
        match='sum of metadata weights was 1.1, must be <=1'
    ):
        ideal_difference_from_metadata(
            positive_items=positive_items,
            negative_items=negative_items,
            metadata={'a': metadata_a, 'b': metadata_b},
            metadata_weights={'a': .2, 'b': .9},
        )


def test_ideal_difference_from_metadata_a(
    positive_items,
    negative_items,
    metadata_a,
    metadata_a_diff,
):
    ideal_diff = ideal_difference_from_metadata(
        positive_items=positive_items,
        negative_items=negative_items,
        metadata={'a': metadata_a},
        metadata_weights={'a': .2},
    )

    assert_array_equal(ideal_diff, metadata_a_diff)


def test_ideal_difference_from_metadata_no_matches(
    positive_items,
    negative_items,
    metadata_a,
    metadata_a_diff,
):
    ideal_diff = ideal_difference_from_metadata(
        positive_items=positive_items,
        negative_items=negative_items,
        metadata={'a': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])},
        metadata_weights={'a': .2},
    )

    assert_array_equal(ideal_diff, torch.ones(4))


def test_ideal_difference_from_metadata_a_and_b(
    positive_items,
    many_negative_items,
    metadata_a,
    metadata_b,
    metadata_a_and_2_diff,
):
    ideal_diff = ideal_difference_from_metadata(
        positive_items=positive_items.repeat(4, 1),
        negative_items=many_negative_items,
        metadata={'a': metadata_a, 'b': metadata_b},
        metadata_weights={'a': .2, 'b': .3},
    )

    assert_array_equal(ideal_diff, metadata_a_and_2_diff)


def test_bpr_loss(positive_scores, negative_scores):
    actual = bpr_loss(positive_scores, negative_scores)
    expected = (1.93074 + 1.36897) / 4

    assert_almost_equal(actual.item(), expected, decimal=3)


def test_hinge_loss(positive_scores, negative_scores):
    actual = hinge_loss(positive_scores, negative_scores)
    expected = (7.3872 + 37.00656384) / 4

    assert_almost_equal(actual.item(), expected, decimal=3)


def test_adaptive_bpr_loss(positive_scores, many_negative_scores):
    actual = adaptive_bpr_loss(positive_scores, many_negative_scores)
    expected = 6.592 / 4

    assert_almost_equal(actual.item(), expected, decimal=3)


def test_adaptive_hinge_loss(positive_scores, many_negative_scores):
    actual = adaptive_hinge_loss(positive_scores, many_negative_scores)
    expected = 252.420 / 4

    assert_almost_equal(actual.item(), expected, decimal=3)


def test_warp_loss(positive_scores, many_negative_scores):
    actual = warp_loss(positive_scores, many_negative_scores, num_items=4)
    expected = (11.366 + 72.385) / 4

    assert_almost_equal(actual.item(), expected, decimal=3)


def test_bpr_loss_metadata(
    positive_scores,
    negative_scores,
    positive_items,
    negative_items,
    metadata_a,
):
    actual = bpr_loss(
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        positive_items=positive_items,
        negative_items=negative_items,
        metadata={'a': metadata_a},
        metadata_weights={'a': 0.2}
    )
    expected = (1.73073 + 1.37570) / 4

    assert_almost_equal(actual.item(), expected, decimal=3)


def test_hinge_loss_metadata(
    positive_scores,
    negative_scores,
    positive_items,
    negative_items,
    metadata_a,
):
    actual = hinge_loss(
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        positive_items=positive_items,
        negative_items=negative_items,
        metadata={'a': metadata_a},
        metadata_weights={'a': 0.2}
    )
    expected = (7.3872 + 37.00656384) / 4

    assert_almost_equal(actual.item(), expected, decimal=3)


def test_adaptive_bpr_loss_metadata(
    positive_scores,
    many_negative_scores,
    positive_items,
    many_negative_items,
    metadata_a,
    metadata_b,
):
    actual = adaptive_bpr_loss(
        positive_scores=positive_scores,
        many_negative_scores=many_negative_scores,
        positive_items=positive_items,
        negative_items=many_negative_items,
        metadata={'a': metadata_a, 'b': metadata_b},
        metadata_weights={'a': 0.2, 'b': 0.3},
    )
    expected = (2.746 + 2.088) / 4

    assert_almost_equal(actual.item(), expected, decimal=3)


def test_adaptive_hinge_loss_metadata(
    positive_scores,
    many_negative_scores,
    positive_items,
    many_negative_items,
    metadata_a,
    metadata_b,
):
    actual = adaptive_hinge_loss(
        positive_scores=positive_scores,
        many_negative_scores=many_negative_scores,
        positive_items=positive_items,
        negative_items=many_negative_items,
        metadata={'a': metadata_a, 'b': metadata_b},
        metadata_weights={'a': 0.2, 'b': 0.3}
    )
    expected = 61.2829

    assert_almost_equal(actual.item(), expected, decimal=3)


def test_warp_loss_metadata(
    positive_scores,
    many_negative_scores,
    positive_items,
    many_negative_items,
    metadata_a,
    metadata_b,
):
    actual = warp_loss(
        positive_scores=positive_scores,
        many_negative_scores=many_negative_scores,
        num_items=4,
        positive_items=positive_items,
        negative_items=many_negative_items,
        metadata={'a': metadata_a, 'b': metadata_b},
        metadata_weights={'a': 0.2, 'b': 0.3},
    )
    expected = (10.390 + 65.063) / 4

    assert_almost_equal(actual.item(), expected, decimal=3)
