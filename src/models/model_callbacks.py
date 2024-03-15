import pytorch_lightning as pl
import src.constants as cst


def callback_save_model(path, metric, top_k=3):
    check_point_callback = pl.callbacks.ModelCheckpoint(
        monitor=metric,
        verbose=True,
        save_top_k=top_k,
        mode='max',
        dirpath=path,
        filename='{epoch}-{' + metric + ':.2f}'
    )
    return check_point_callback


# TODO avoid early stopping
# def early_stopping(config):
#     """ Stops if models stops improving. """
#     monitor_var = config.EARLY_STOPPING_METRIC
#     return pl.callbacks.EarlyStopping(
#         monitor=monitor_var,
#         min_delta=0.01,
#         patience=8,
#         verbose=True,
#         mode='max',
#         # |v stops when if after epoch 1, the
#         # check_on_train_epoch_end=True,
#         # divergence_threshold=1/3,
#     )
