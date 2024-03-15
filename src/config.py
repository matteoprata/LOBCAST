
import numpy as np
import os

from src.constants import LearningHyperParameter
import src.constants as cst
from src.metrics.metrics_log import Metrics
from datetime import date, datetime
import argparse
import torch
from enum import Enum
from pytorch_lightning import seed_everything

import random
np.set_printoptions(suppress=True)
import multiprocessing


class Settings:
    def __init__(self):

        # cli params have the priority

        self.SEED: int = 0
        self.IS_HPARAM_SEARCH: bool = False  # else preload from json file

        self.DATASET_NAME: cst.DatasetFamily = cst.DatasetFamily.FI
        self.STOCK_TRAIN_VAL: str = "FI"
        self.STOCK_TEST: str = "FI"

        self.N_TRENDS = 3
        self.PREDICTION_MODEL = cst.ModelsClass.MLP
        self.PREDICTION_HORIZON_UNIT: cst.UnitHorizon = cst.UnitHorizon.EVENTS
        self.PREDICTION_HORIZON_FUTURE: int = 5
        self.PREDICTION_HORIZON_PAST: int = 10
        self.OBSERVATION_PERIOD: int = 100
        self.IS_SHUFFLE_TRAIN_SET = True

        self.TRAIN_SET_PORTION = .8

        self.IS_TEST = False
        self.MODEL_PATH: str = ""

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.N_GPUs = None if self.DEVICE == 'cpu' else torch.cuda.device_count()
        self.N_CPUs = multiprocessing.cpu_count()

        self.PROJECT_NAME = ""
        self.DIR_SAVED_MODEL = ""
        self.DIR_EXPERIMENTS = ""


class ConfigHP:
    def add_hyperparameters(self, params: dict):
        for key, value in params.items():
            self.__setattr__(key, value)

    def add_hyperparameter(self, key, value):
        self.__setattr__(key, value)

    def __repr__(self):
        return self.__dict__


class ConfigHPTuned(ConfigHP):
    pass


class ConfigHPTunable(ConfigHP):
    def __init__(self):
        self.BATCH_SIZE = [64, 55]
        self.LEARNING_RATE = [0.01]
        self.EPOCHS_UB = [3]
        self.OPTIMIZER = ["SGD"]


class LOBCASTSetupRun:  # TOGLIERE questa ISA
    def __init__(self):
        super().__init__()

        self.SETTINGS = Settings()

        self.parse_cl_arguments(self.SETTINGS)
        self.seed_everything(self.SETTINGS.SEED)

        # read from
        self.TUNABLE_H_PRAM = ConfigHPTunable()
        self.TUNED_H_PRAM = ConfigHPTuned()

        self.setup_parameters()
        self.choose_parameters()  # TODO lancerà più esecuzioni, bisognerà tornare qui
        # at this point this

        self.RUN_NAME_PREFIX = self.run_name_prefix(self.SETTINGS)
        self.setup_all_directories(self.RUN_NAME_PREFIX, self.SETTINGS)

        print("RUNNING:\n>>", self.RUN_NAME_PREFIX)

    def setup_parameters(self):
        # add parameters from model
        for key, value in self.SETTINGS.PREDICTION_MODEL.value.tunable_parameters.items():
            self.TUNABLE_H_PRAM.add_hyperparameter(key, value)

        # set to default, add the same parameters in the TUNED_H_PRAM object
        for key, _ in self.TUNABLE_H_PRAM.__dict__.items():
            self.TUNED_H_PRAM.add_hyperparameter(key, None)

    def choose_parameters(self):
        # for now, it only assigned the first element in a list of possible values
        for key, value in self.TUNABLE_H_PRAM.__dict__.items():
            value = value[0] if type(value) == list else value
            self.TUNED_H_PRAM.__setattr__(key, value)

    def seed_everything(self, seed):
        """ Sets the random seed to all the random generators. """
        seed_everything(seed)
        np.random.seed(seed)
        random.seed(seed)
        # self.RANDOM_GEN_DATASET = np.random.RandomState(seed)

    @staticmethod
    def cf_name_format(ext=""):
        return "MOD={}-SEED={}-TRS={}-TES={}-DS={}-HU={}-HP={}-HF={}-OB={}" + ext

    def run_name_prefix(self, settings):
        return self.cf_name_format().format(
            settings.PREDICTION_MODEL.name,
            settings.SEED,
            settings.STOCK_TRAIN_VAL,
            settings.STOCK_TEST,
            settings.DATASET_NAME.value,
            settings.PREDICTION_HORIZON_UNIT.name,
            settings.PREDICTION_HORIZON_PAST,
            settings.PREDICTION_HORIZON_FUTURE,
            settings.OBSERVATION_PERIOD,
        )

    def parse_cl_arguments(self, settings):
        """ Parses the arguments for the command line. """

        parser = argparse.ArgumentParser(description='LOBCAST single execution arguments:')

        # every field in the settings, can be set crom cl
        for k, v in settings.__dict__.items():
            type_var = str if isinstance(v, Enum) else type(v)
            var = v.name if isinstance(v, Enum) else v
            parser.add_argument(f'--{k}', default=var, type=type_var)

        args = vars(parser.parse_args())

        # every field in the settings, is set based on the parsed values, enums are parsed by NAME
        for k, v in settings.__dict__.items():
            value = v.__class__[args[k]] if isinstance(v, Enum) else args[k]
            self.__setattr__(k, value)

    @staticmethod
    def setup_all_directories(fname, settings):
        """
        Creates two folders:
            (1) data.experiments.LOBCAST-(fname) for the jsons with the stats
            (2) data.saved_models.LOBCAST-(fname) for the models
        """

        # TODO consider using dates
        settings.PROJECT_NAME = cst.PROJECT_NAME.format(fname)
        settings.DIR_SAVED_MODEL = cst.DIR_SAVED_MODEL.format(fname) + "/"
        settings.DIR_EXPERIMENTS = cst.DIR_EXPERIMENTS.format(fname) + "/"

        # create the paths for the simulation if they do not exist already
        paths = ["data", cst.DIR_SAVED_MODEL, cst.DIR_EXPERIMENTS]
        for p in paths:
            if not os.path.exists(p):
                os.makedirs(p)


class Configuration(Settings):
    pass
    # """ Represents the configuration file of the simulation, containing all variables of the simulation. """
    # def __init__(self, run_name_prefix=None):
    #     super().__init__()
    #
    #     self.IS_DEBUG = False
    #     self.IS_TEST_ONLY = False
    #
    #     self.RUN_NAME_PREFIX = self.assign_prefix(prefix=run_name_prefix, is_debug=self.IS_DEBUG)
    #     self.setup_all_directories(self.RUN_NAME_PREFIX, self.IS_DEBUG, self.IS_TEST_ONLY)
    #
    #     self.RANDOM_GEN_DATASET = None
    #     self.VALIDATE_EVERY = 1
    #
    #     self.IS_DATA_PRELOAD = True
    #     self.INSTANCES_LOWER_BOUND = 1000  # under-sampling must have at least INSTANCES_LOWER_BOUND instances
    #
    #     self.TRAIN_SPLIT_VAL = .8  # FI only
    #     self.META_TRAIN_VAL_TEST_SPLIT = (.7, .15, .15)  # META Only
    #
    #     self.CHOSEN_PERIOD = cst.Periods.FI
    #
    #     self.CHOSEN_STOCKS = {
    #         cst.STK_OPEN.TRAIN: cst.Stocks.FI,
    #         cst.STK_OPEN.TEST: cst.Stocks.FI
    #     }
    #
    #     self.IS_WANDB = 0
    #
    #     self.SWEEP_METHOD = 'grid'  # 'bayes'
    #
    #     self.WANDB_INSTANCE = None
    #     self.WANDB_RUN_NAME = None
    #     self.WANDB_SWEEP_NAME = None
    #
    #     self.SWEEP_METRIC = {
    #         'goal': 'maximize',
    #         'name': None
    #     }
    #
    #     self.TARGET_DATASET_META_MODEL = cst.DatasetFamily.LOB
    #     self.JSON_DIRECTORY = ""
    #
    #     self.EARLY_STOPPING_METRIC = None
    #
    #     self.METRICS_JSON = Metrics(self)
    #     self.HYPER_PARAMETERS = {lp: None for lp in LearningHyperParameter}
    #
    #     self.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE] = 128
    #     self.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE] = 0.01
    #     self.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS_UB] = 100
    #     self.HYPER_PARAMETERS[LearningHyperParameter.OPTIMIZER] = cst.Optimizers.SGD.value
    #     self.HYPER_PARAMETERS[LearningHyperParameter.WEIGHT_DECAY] = 0.0
    #     self.HYPER_PARAMETERS[LearningHyperParameter.EPS] = 1e-08  # default value for ADAM
    #     self.HYPER_PARAMETERS[LearningHyperParameter.MOMENTUM] = 0.9
    #
    #     self.HYPER_PARAMETERS[LearningHyperParameter.NUM_SNAPSHOTS] = 100
    #     # LOB way to label to measure percentage change LOB = HORIZON
    #     self.HYPER_PARAMETERS[LearningHyperParameter.BACKWARD_WINDOW] = cst.WinSize.NONE.value
    #     self.HYPER_PARAMETERS[LearningHyperParameter.FORWARD_WINDOW] = cst.WinSize.NONE.value
    #     self.HYPER_PARAMETERS[LearningHyperParameter.IS_SHUFFLE_TRAIN_SET] = True
    #     self.HYPER_PARAMETERS[LearningHyperParameter.LABELING_SIGMA_SCALER] = .9
    #     self.HYPER_PARAMETERS[LearningHyperParameter.FI_HORIZON] = cst.FI_Horizons.K10.value  # in FI = FORWARD_WINDOW  = k in papers
    #
    #     self.HYPER_PARAMETERS[LearningHyperParameter.MLP_HIDDEN] = 128
    #     self.HYPER_PARAMETERS[LearningHyperParameter.RNN_HIDDEN] = 32
    #     self.HYPER_PARAMETERS[LearningHyperParameter.META_HIDDEN] = 16
    #
    #     self.HYPER_PARAMETERS[LearningHyperParameter.RNN_N_HIDDEN] = 1
    #     self.HYPER_PARAMETERS[LearningHyperParameter.DAIN_LAYER_MODE] = 'full'
    #     self.HYPER_PARAMETERS[LearningHyperParameter.P_DROPOUT] = 0
    #     self.HYPER_PARAMETERS[LearningHyperParameter.NUM_RBF_NEURONS] = 16
    #
    # def dynamic_config_setup(self):
    #     # sets the name of the metric to optimize
    #     self.SWEEP_METRIC['name'] = "{}_{}_{}".format(cst.ModelSteps.VALIDATION_MODEL.value, self.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, cst.Metrics.F1.value)
    #     self.EARLY_STOPPING_METRIC = "{}_{}_{}".format(cst.ModelSteps.VALIDATION.value, self.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, cst.Metrics.F1.value)
    #
    #     self.WANDB_SWEEP_NAME = self.cf_name_format().format(
    #         self.PREDICTION_MODEL.name,
    #         self.SEED,
    #         self.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name,
    #         self.CHOSEN_STOCKS[cst.STK_OPEN.TEST].name,
    #         self.DATASET_NAME.value,
    #         self.CHOSEN_PERIOD.name,
    #         self.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW],
    #         self.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW],
    #         self.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
    #     )
    #
    #     if not self.IS_HPARAM_SEARCH and not self.IS_WANDB:
    #         self.WANDB_RUN_NAME = self.WANDB_SWEEP_NAME
    #
    # @staticmethod
    # def cf_name_format(ext=""):
    #     return "model={}-seed={}-trst={}-test={}-data={}-peri={}-bw={}-fw={}-fiw={}" + ext
    #
    # @staticmethod
    # def setup_all_directories(prefix, is_debug, is_test):
    #     """
    #     Creates two folders:
    #         (1) data.experiments.LOB-CLASSIFIERS-(PREFIX) for the jsons with the stats
    #         (2) data.saved_models.LOB-CLASSIFIERS-(PREFIX) for the models
    #     """
    #
    #     if not is_test:
    #         cst.PROJECT_NAME = cst.PROJECT_NAME.format(prefix)
    #         cst.DIR_SAVED_MODEL = cst.DIR_SAVED_MODEL.format(prefix) + "/"
    #         cst.DIR_EXPERIMENTS = cst.DIR_EXPERIMENTS.format(prefix) + "/"
    #
    #         # create the paths for the simulation if they do not exist already
    #         paths = ["data", cst.DIR_SAVED_MODEL, cst.DIR_EXPERIMENTS]
    #         for p in paths:
    #             if not os.path.exists(p):
    #                 os.makedirs(p)
    #
    # @staticmethod
    # def assign_prefix(prefix, is_debug):
    #     if is_debug:
    #         return "debug"
    #     elif prefix is not None:
    #         return prefix
    #     else:
    #         return datetime.now().strftime("%Y-%m-%d+%H-%M-%S")
