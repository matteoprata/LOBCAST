from enum import Enum
import numpy as np
import torch
from datetime import datetime


from src.constants import LearningHyperParameter, Optimizers
import src.constants as cst

np.set_printoptions(suppress=True)


class Configuration:

    def __init__(self):

        self.SEED = 0
        self.RANDOM_GEN_DATASET = None
        self.VALIDATE_EVERY = 1

        self.IS_DATA_PRELOAD = True
        self.INSTANCES_LOWER_BOUND = 1000  # under-sampling must have at least INSTANCES_LOWER_BOUND instances

        self.TRAIN_SPLIT_VAL = .8  # FI only

        self.CHOSEN_DATASET = cst.DatasetFamily.FI
        self.CHOSEN_PERIOD = cst.Periods.JULY2021
        self.CHOSEN_MODEL = cst.Models.MLP

        self.CHOSEN_STOCKS = {
            cst.STK_OPEN.TRAIN: cst.Stocks.ALL,
            cst.STK_OPEN.TEST: cst.Stocks.ALL
        }

        self.IS_WANDB = 0
        self.SWEEP_NAME = None
        self.SWEEP_METHOD = 'bayes'

        self.WANDB_INSTANCE = None
        self.WANDB_RUN_NAME = None

        self.SWEEP_METRIC = {
            'goal': 'maximize',
            'name': None
        }

        self.HYPER_PARAMETERS = {lp: None for lp in LearningHyperParameter}

        self.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE] = 32
        self.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE] = 0.0001
        self.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS_UB] = 100
        self.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER] = cst.Optimizers.ADAM.value
        self.HYPER_PARAMETERS[LearningHyperParameter.WEIGHT_DECAY] = 0.0

        self.HYPER_PARAMETERS[LearningHyperParameter.NUM_SNAPSHOTS] = 100
        # LOBSTER way to label to measure percentage change LOBSTER = HORIZON
        self.HYPER_PARAMETERS[LearningHyperParameter.BACKWARD_WINDOW] = cst.WinSize.SEC100.value
        self.HYPER_PARAMETERS[LearningHyperParameter.FORWARD_WINDOW] = cst.WinSize.SEC50.value
        self.HYPER_PARAMETERS[LearningHyperParameter.IS_SHUFFLE_TRAIN_SET] = True
        self.HYPER_PARAMETERS[LearningHyperParameter.LABELING_SIGMA_SCALER] = .9
        self.HYPER_PARAMETERS[LearningHyperParameter.FI_HORIZON] = 10  # in FI = FORWARD_WINDOW  = k in papers

        self.HYPER_PARAMETERS[LearningHyperParameter.MLP_HIDDEN] = 128
        self.HYPER_PARAMETERS[LearningHyperParameter.LSTM_HIDDEN] = 32
        self.HYPER_PARAMETERS[LearningHyperParameter.LSTM_N_HIDDEN] = 1
        self.HYPER_PARAMETERS[LearningHyperParameter.DAIN_LAYER_MODE] = 'full'
        self.HYPER_PARAMETERS[LearningHyperParameter.P_DROPOUT] = .1

    def dynamic_config_setup(self):
        self.SWEEP_METRIC['name'] = cst.ModelSteps.VALIDATION.value + "_{}_".format(self.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name) + cst.Metrics.F1.value

        if self.CHOSEN_DATASET == cst.DatasetFamily.FI:
            self.SWEEP_NAME = self.CHOSEN_DATASET.value + '_' + self.CHOSEN_MODEL.value + ''
        else:
            self.SWEEP_NAME = self.CHOSEN_DATASET.value + '_' + self.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name + '_' +\
                   self.CHOSEN_STOCKS[cst.STK_OPEN.TEST].name + '_' + self.CHOSEN_PERIOD.name + '_' + self.CHOSEN_MODEL.value + ''

