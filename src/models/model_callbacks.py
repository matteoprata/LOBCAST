
import pytorch_lightning as pl
import src.config as co


def callback_save_model(ml_model_name):
    monitor_var = co.ModelSteps.VALIDATION.value + co.Metrics.F1.value
    check_point_callback = pl.callbacks.ModelCheckpoint(
                           monitor=monitor_var,
                           verbose=True,
                           save_top_k=3,
                           mode='max',
                           dirpath=co.SAVED_MODEL_DIR,
                           filename=ml_model_name + '-{epoch}-{' + monitor_var + ':.2f}'
    )
    return check_point_callback


def early_stopping():
    """ Stops if models stops improving. """
    monitor_var = co.ModelSteps.TRAINING.value + co.Metrics.LOSS.value
    return pl.callbacks.EarlyStopping(monitor=monitor_var,
                                      min_delta=0.00,
                                      patience=10,
                                      verbose=True,
                                      mode="min")
