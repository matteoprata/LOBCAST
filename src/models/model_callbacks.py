
import pytorch_lightning as pl
import src.constants as cst


# def fname_format(config, dataset_type, ml_model_name, monitor_var, run_name):
#     """ Generates the name of a model. """
#     # src = "mod=" + ml_model_name + '_{epoch}_{' + monitor_var + ':.2f}' + '_run=' + str(run_name) + "_"
#     # if dataset_type == cst.DatasetFamily.FI:
#     #     return src
#     # elif dataset_type == cst.DatasetFamily.LOBSTER:
#     #     return src + "wb={}_wf={}_scale={}".format(config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW],
#     #                                                config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW],
#     #                                                config.HYPER_PARAMETERS[cst.LearningHyperParameter.LABELING_SIGMA_SCALER])
#     # else:
#     #     print("Unhandled dataset name.")
#     #     exit()
#     return


def callback_save_model(config, run_name):
    monitor_var = config.SWEEP_METRIC['name']
    check_point_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor_var,
        verbose=True,
        save_top_k=1,
        mode='max',
        dirpath=cst.SAVED_MODEL_DIR + config.SWEEP_NAME,
        filename="run=" + run_name + "-{epoch}-{" + monitor_var + ':.2f}'
    )
    return check_point_callback


def early_stopping(config):
    """ Stops if models stops improving. """
    monitor_var = config.SWEEP_METRIC['name']
    return pl.callbacks.EarlyStopping(
        monitor=monitor_var,
        min_delta=0.00,
        patience=15,
        verbose=True,
        mode='max'
    )
