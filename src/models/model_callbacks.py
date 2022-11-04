
import pytorch_lightning as pl
import src.config as co
from datetime import datetime


def callback_save_model(ml_model_name):
    monitor_var = co.ModelSteps.VALIDATION.value + co.Metrics.F1.value
    check_point_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor_var,
        verbose=True,
        save_top_k=3,
        mode='max',
        dirpath=co.SAVED_MODEL_DIR,
        filename=ml_model_name + '-{epoch}-{' + monitor_var + ':.2f}' + '_' + datetime.now().strftime("%d%m%Y%H%M%S")
    )
    return check_point_callback


def early_stopping():
    """ Stops if models stops improving. """
    monitor_var = co.ModelSteps.VALIDATION.value + co.Metrics.F1.value
    return pl.callbacks.EarlyStopping(
        monitor=monitor_var,
        min_delta=0.00,
        patience=25,
        verbose=True,
        mode='max'
    )
