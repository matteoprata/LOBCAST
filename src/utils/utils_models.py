import numpy as np

import src.constants as cst
from src.config import LOBCASTSetupRun
from src.utils.util_training import LOBCAST_NNEngine


# MODELS
from src.utils.utils_generic import get_class_arguments


def get_tuned_parameters(config: LOBCASTSetupRun, params):
    values = [config.TUNED_H_PRAM.__getattribute__(p) for p in params]
    return values


def pick_model(config, data_module, metrics_log):
    loss_weights = None

    num_features = data_module.x_shape
    print(data_module.x_shape[0], data_module.x_shape[1])  # 10 x 40
    num_classes = data_module.num_classes

    args = get_class_arguments(config.SETTINGS.PREDICTION_MODEL.value.model)[2:]
    args_values = get_tuned_parameters(config, args)
    neural_architecture = config.SETTINGS.PREDICTION_MODEL.value.model(num_features, num_classes, *args_values)

    engine = LOBCAST_NNEngine(
        neural_architecture,
        loss_weights,
        hps=config.TUNED_H_PRAM,
        metrics_log=metrics_log,
        wandb_log=config.WANDB_INSTANCE,
    ).to(config.SETTINGS.DEVICE)

    return engine
