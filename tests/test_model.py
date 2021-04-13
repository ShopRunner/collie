import pandas as pd

from collie_recs.interactions import HDF5InteractionsDataLoader
from collie_recs.metrics import evaluate_in_batches, mapk


def test_implicit_model(implicit_model, train_val_implicit_data):
    train, test = train_val_implicit_data

    item_preds = implicit_model.get_item_predictions(user_id=0,
                                                     unseen_items_only=True,
                                                     sort_values=True)

    assert isinstance(item_preds, pd.Series)
    assert len(item_preds) > 0
    assert len(item_preds) < len(train)

    item_similarities = implicit_model.item_item_similarity(item_id=42)
    assert item_similarities.index[0] == 42

    mapk_score = evaluate_in_batches([mapk], test, implicit_model)

    # The metrics used for evaluation have been determined through 30
    # trials of training the model and using the mean - 5 * std. dev.
    # as the minimum score the model must achieve to pass the test.
    assert mapk_score > 0.044


def test_other_models_trained_for_one_epoch(other_models_trained_for_one_epoch,
                                            train_val_implicit_data):
    train, test = train_val_implicit_data

    if not isinstance(other_models_trained_for_one_epoch.train_loader, HDF5InteractionsDataLoader):
        item_preds = other_models_trained_for_one_epoch.get_item_predictions(user_id=0,
                                                                             unseen_items_only=True,
                                                                             sort_values=True)

        assert isinstance(item_preds, pd.Series)
        assert len(item_preds) > 0
        assert len(item_preds) < len(train)

    item_similarities = other_models_trained_for_one_epoch.item_item_similarity(item_id=42)
    assert item_similarities.index[0] == 42
