
import pytorch_lightning as pl
import src.constants as cst


def fname_format(config, dataset_type, ml_model_name, monitor_var, run_name):
    """ Generates the name of a model. """
    src = "mod=" + ml_model_name + '_{epoch}_{' + monitor_var + ':.2f}' + '_run=' + run_name + "_"
    if dataset_type == cst.DatasetFamily.FI:
        return src
    elif dataset_type == cst.DatasetFamily.LOBSTER:
        return src + "wb={}_wf={}_scale={}".format(config.BACKWARD_WINDOW, config.FORWARD_WINDOW, config.LABELING_SIGMA_SCALER)
    else:
        print("Unhandled model name.")
        exit()


def callback_save_model(config, dataset_type, ml_model_name, run_name):
    monitor_var = config.SWEEP_METRIC_OPT
    check_point_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor_var,
        verbose=True,
        save_top_k=3,
        mode='max',
        dirpath=config.SAVED_MODEL_DIR + config.SWEEP_NAME,
        filename=fname_format(config, dataset_type, ml_model_name, monitor_var, run_name)
    )
    return check_point_callback


def early_stopping(config):
    """ Stops if models stops improving. """
    monitor_var = config.SWEEP_METRIC_OPT
    return pl.callbacks.EarlyStopping(
        monitor=monitor_var,
        min_delta=0.00,
        patience=15,
        verbose=True,
        mode='max'
    )
