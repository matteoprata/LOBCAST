import numpy as np

import src.constants as cst
from src.config import Configuration
from src.utils.util_training import LOBCAST_NNEngine


# MODELS
from src.models.mlp.mlp import MLP
from src.models.tabl.ctabl import CTABL
from src.models.translob.translob import TransLob
from src.models.cnn1.cnn1 import CNN1
from src.models.cnn2.cnn2 import CNN2
from src.models.cnnlstm.cnnlstm import CNNLSTM
from src.models.dain.dain import DAIN
from src.models.deeplob.deeplob import DeepLob
from src.models.lstm.lstm import LSTM
from src.models.binctabl.bin_tabl import BiN_CTABL
from src.models.deeplobatt.deeplobatt import DeepLobAtt
from src.models.dla.dla import DLA
from src.models.atnbof.atnbof import ATNBoF
from src.models.tlonbof.tlonbof import TLONBoF
from src.models.axial.axiallob import AxialLOB
from src.models.metalob.metalob import MetaLOB

from src.utils.utils_generic import get_class_arguments

import torch.nn as nn


def get_tuned_parameters(config: Configuration, params):
    values = [config.TUNED_H_PRAM.__getattribute__(p) for p in params]
    return values


def pick_model(config, data_module, metrics_log):
    loss_weights = None

    num_features = data_module.x_shape
    print(data_module.x_shape[0], data_module.x_shape[1])  # 10 x 40
    num_classes = data_module.num_classes

    args = get_class_arguments(config.PREDICTION_MODEL.value.model)[2:]
    args_values = get_tuned_parameters(config, args)
    neural_architecture = config.PREDICTION_MODEL.value.model(num_features, num_classes, *args_values)

    engine = LOBCAST_NNEngine(
        neural_architecture,
        loss_weights,
        hps=config.TUNED_H_PRAM,
        metrics_log=metrics_log
    ).to(cst.DEVICE_TYPE)

    return engine
