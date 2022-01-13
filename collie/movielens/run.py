import fire
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from collie.config import DATA_PATH
from collie.cross_validation import stratified_split
from collie.interactions import Interactions, InteractionsDataLoader
from collie.metrics import auc, evaluate_in_batches, mapk, mrr
from collie.model import CollieTrainer, MatrixFactorizationModel
from collie.movielens import read_movielens_df
from collie.utils import convert_to_implicit, Timer


def run_movielens_example(epochs: int = 20, gpus: int = 0) -> None:
    """
    Retrieve and split data, train and evaluate a model, and save it.

    From the terminal, you can run this script with:

    .. code-block:: bash

        python collie/movielens/run.py  --epochs 20

    Parameters
    ----------
    epochs: int
        Number of epochs for model training
    gpus: int
        Number of gpus to train on

    """
    t = Timer()

    t.timecheck('  1.0 - retrieving MovieLens 100K dataset')
    df = read_movielens_df(decrement_ids=True)
    t.timecheck('  1.0 complete')

    t.timecheck('  2.0 - splitting data')
    df_imp = convert_to_implicit(df)
    interactions = Interactions(users=df_imp['user_id'],
                                items=df_imp['item_id'],
                                allow_missing_ids=True)
    train, val, test = stratified_split(interactions, val_p=0.1, test_p=0.1)
    train_loader = InteractionsDataLoader(train, batch_size=1024, shuffle=True)
    val_loader = InteractionsDataLoader(val, batch_size=1024, shuffle=False)
    t.timecheck('  2.0 complete')

    t.timecheck('  3.0 - training the model')
    model = MatrixFactorizationModel(train=train_loader,
                                     val=val_loader,
                                     dropout_p=0.05,
                                     loss='adaptive',
                                     lr=5e-2,
                                     embedding_dim=10,
                                     optimizer='adam',
                                     weight_decay=1e-7)
    trainer = CollieTrainer(model=model,
                            gpus=gpus,
                            max_epochs=epochs,
                            deterministic=True,
                            logger=False,
                            enable_checkpointing=False,
                            callbacks=[EarlyStopping(monitor='val_loss_epoch', mode='min')])
    trainer.fit(model)
    model.eval()
    t.timecheck('\n  3.0 complete')

    t.timecheck('  4.0 - evaluating model')
    auc_score, mrr_score, mapk_score = evaluate_in_batches([auc, mrr, mapk], test, model, k=10)
    print(f'AUC:          {auc_score}')
    print(f'MRR:          {mrr_score}')
    print(f'MAP@10:       {mapk_score}')
    t.timecheck('  4.0 complete')

    t.timecheck('  5.0 - saving model')
    absolute_data_path = DATA_PATH / 'fitted_model'
    model.save_model(absolute_data_path)
    t.timecheck('  5.0 complete')


if __name__ == '__main__':
    fire.Fire(run_movielens_example)
