from enum import Enum
import numpy as np
import torch
from datetime import datetime


from src.constants import LearningHyperParameter, Optimizers
import src.constants as cst

np.set_printoptions(suppress=True)


class Configuration:

    def __init__(self):
        self.CLASS_NAMES = ["DOWN", "STATIONARY", "UP"]

        self.NUM_GPUS = 1 if torch.cuda.is_available() else 0
        self.DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.SAVED_MODEL_DIR = "data/saved_models/"
        self.DATA_SOURCE = "data/"
        self.DATASET_LOBSTER = "LOBSTER_6/unzipped/"
        self.DATASET_FI = "FI-2010/BenchmarkDatasets"
        self.DATA_PICKLES = "data/pickles/"

        self.SEED = 0
        self.RANDOM_GEN_DATASET = None
        self.VALIDATE_EVERY = 1

        # MODELS PARAMS
        self.EPOCHS = 100
        self.BATCH_SIZE = 32
        self.OPTIMIZER = Optimizers.ADAM.value
        self.LEARNING_RATE = 0.0001
        self.WEIGHT_DECAY = 0

        # MLP
        self.MLP_HIDDEN = 128
        self.P_DROPOUT = .1

        # LSTM
        self.LSTM_HIDDEN = 32
        self.LSTM_N_HIDDEN = 1

        # DAIN
        self.DAIN_LAYER_MODE = 'full'

        self.IS_DATA_PRELOAD = True
        self.IS_SHUFFLE_INPUT = True  # ONLY TRAIN
        self.N_LOB_LEVELS = 10
        self.NUM_SNAPSHOTS = 100
        self.INSTANCES_LOWERBOUND = 1000  # under-sampling must have at least INSTANCES_LOWERBOUND instances

        # LOBSTER way to label to measure percentage change
        self.BACKWARD_WINDOW = cst.WinSize.SEC100.value
        self.FORWARD_WINDOW = cst.WinSize.SEC50.value  # in LOBSTER = HORIZON
        self.LABELING_SIGMA_SCALER = .9   # dynamic threshold

        # K of the FI dataset
        self.HORIZON = 10  # in FI = FORWARD_WINDOW

        self.TRAIN_SPLIT_VAL = .8  # FI only

        self.CHOSEN_DATASET = cst.DatasetFamily.FI

        self.CHOSEN_PERIOD = cst.Periods.MARCH2020
        self.CHOSEN_MODEL = cst.Models.TRANSLOB

        self.CHOSEN_STOCKS = {
            cst.STK_OPEN.TRAIN: cst.Stocks.ALL,
            cst.STK_OPEN.TEST: cst.Stocks.ALL
        }

        self.IS_WANDB = None

        self.SWEEP_NAME = None

        self.SWEEP_METHOD = 'bayes'

        self.SRC_STOCK_NAME = "!SRC!"
        self.SWEEP_METRIC_OPT = cst.ModelSteps.VALIDATION.value + "_{}_".format(self.SRC_STOCK_NAME) + cst.Metrics.F1.value

        self.SWEEP_METRIC = {
            'goal': 'maximize',
            'name': self.SWEEP_METRIC_OPT
        }

        self.HYPER_PARAMETERS_SET = {lp: None for lp in LearningHyperParameter}

        self.HYPER_PARAMETERS_SET[LearningHyperParameter.OPTIMIZER] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.LEARNING_RATE] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.WEIGHT_DECAY] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.EPOCHS] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.IS_SHUFFLE] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.BATCH_SIZE] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.MLP_HIDDEN] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.LSTM_HIDDEN] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.LSTM_N_HIDDEN] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.DAIN_LAYER_MODE] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.P_DROPOUT] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.BACKWARD_WINDOW] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.FORWARD_WINDOW] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.LABELING_THRESHOLD] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.LABELING_SIGMA_SCALER] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.FI_HORIZON] = None
        self.HYPER_PARAMETERS_SET[LearningHyperParameter.NUM_SNAPSHOTS] = None

        self.IS_WANDB = None
        self.WANDB_INSTANCE = None
        self.WANDB_RUN_NAME = None
        self.SWEEP_NAME = self.sweep_name()

    def sweep_name(self):
        if self.CHOSEN_DATASET == cst.DatasetFamily.FI:
            return self.CHOSEN_DATASET.value + '_' + self.CHOSEN_MODEL.value + ''
        else:
            return self.CHOSEN_DATASET.value + '_' + self.CHOSEN_STOCKS[
                cst.STK_OPEN.TRAIN].name + '_' + self.CHOSEN_STOCKS[ cst.STK_OPEN.TEST].name + '_' + self.CHOSEN_PERIOD.name + '_' + self.CHOSEN_MODEL.value + ''
