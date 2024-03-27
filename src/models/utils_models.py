import numpy as np

import src.constants as cst
from src.utils.util_training import LOBCAST_NNEngine


# MODELS
from src.utils.utils_generic import get_class_arguments


def get_tuned_parameters(sim, params):
    values = [sim.HP_TUNED.__getattribute__(p) for p in params]
    return values


def pick_model(sim, data_module):
    loss_weights = None

    num_features = data_module.x_shape
    num_classes = data_module.num_classes

    args = get_class_arguments(sim.SETTINGS.PREDICTION_MODEL.value.model)[2:]
    args_values = get_tuned_parameters(sim, args)
    neural_architecture = sim.SETTINGS.PREDICTION_MODEL.value.model(num_features, num_classes, *args_values)

    engine = LOBCAST_NNEngine(
        neural_architecture,
        loss_weights,
        hps=sim.HP_TUNED,
        metrics_log=sim.METRICS,
        wandb_log=sim.WANDB_INSTANCE,
    ).to(sim.SETTINGS.DEVICE)

    return engine
