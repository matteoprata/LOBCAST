
import pytorch_lightning as pl
import src.constants as cst


def callback_save_model(config, run_name):
    monitor_var = config.EARLY_STOPPING_METRIC
    check_point_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor_var,
        verbose=True,
        save_top_k=1,
        mode='max',
        dirpath=cst.DIR_SAVED_MODEL + config.WANDB_SWEEP_NAME,
        filename="run=" + run_name + "-{epoch}-{" + monitor_var + ':.2f}'
    )
    return check_point_callback


def early_stopping(config):
    """ Stops if models stops improving. """
    monitor_var = config.EARLY_STOPPING_METRIC
    return pl.callbacks.EarlyStopping(
        monitor=monitor_var,
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='max',
        # |v stops when if after epoch 1, the
        # check_on_train_epoch_end=True,
        # divergence_threshold=1/3,
    )
