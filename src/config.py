
import numpy as np
import os

# from src.constants import LearningHyperParameter
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
from src.utils.utils_generic import dict_to_string, str_to_bool


class Settings:
    def __init__(self):

        # cli params have the priority

        self.SEED: int = 0

        self.DATASET_NAME: cst.DatasetFamily = cst.DatasetFamily.FI

        self.N_TRENDS = 3
        self.PREDICTION_MODEL = cst.ModelsClass.MLP
        self.PREDICTION_HORIZON_UNIT: cst.UnitHorizon = cst.UnitHorizon.EVENTS
        self.PREDICTION_HORIZON_FUTURE: int = 5
        self.PREDICTION_HORIZON_PAST: int = 1
        self.OBSERVATION_PERIOD: int = 100
        self.IS_SHUFFLE_TRAIN_SET = True

        self.EPOCHS_UB = 10
        self.TRAIN_SET_PORTION = .8
        self.VALIDATION_EVERY = 5

        self.IS_TEST_ONLY = False
        self.TEST_MODEL_PATH: str = "data/saved_models/LOBCAST-(15-03-2024_20-23-49)/epoch=2-validation_f1=0.27.ckpt"

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.N_GPUs = None if self.DEVICE == 'cpu' else torch.cuda.device_count()
        self.N_CPUs = multiprocessing.cpu_count()

        self.DIR_EXPERIMENTS = ""
        self.SWEEP_METHOD = 'grid'
        self.IS_WANDB = False

    def check_parameters_validity(self):
        CONSTRAINTS = []
        c1 = (not self.IS_TEST_ONLY or os.path.exists(self.TEST_MODEL_PATH), "If IS_TEST_ONLY, then test model should exist.")

        c2 = (not self.DATASET_NAME == cst.DatasetFamily.FI or self.PREDICTION_HORIZON_UNIT == cst.UnitHorizon.EVENTS,
              f"FI-2010 Dataset can handle only event based granularity, {self.PREDICTION_HORIZON_UNIT} given.")

        c3 = (not self.DATASET_NAME == cst.DatasetFamily.FI or self.PREDICTION_HORIZON_PAST == 1,
              f"FI-2010 Dataset can handle only 1 event in the past horizon, {self.PREDICTION_HORIZON_PAST} given.")

        c4 = (not self.DATASET_NAME == cst.DatasetFamily.FI or self.PREDICTION_HORIZON_FUTURE in [1, 2, 3, 5, 10],
              f"FI-2010 Dataset can handle only {1, 2, 3, 5, 10} events in the future horizon, {self.PREDICTION_HORIZON_FUTURE} given.")

        CONSTRAINTS += [c1, c2, c3, c4]
        for constrain, description in CONSTRAINTS:
            if not constrain:
                raise ValueError(f"Constraint not met! {description} Check your parameters.")

    def __repr__(self):
        return dict_to_string(self.__dict__)


class ConfigHP:
    def add_hyperparameters(self, params: dict):
        for key, value in params.items():
            self.__setattr__(key, value)

    def add_hyperparameter(self, key, value):
        self.__setattr__(key, value)

    def __repr__(self):
        return dict_to_string(self.__dict__)


class ConfigHPTuned(ConfigHP):
    pass


class ConfigHPTunable(ConfigHP):
    def __init__(self):
        self.BATCH_SIZE = {"values": [55]}   # {"min": 0.0001, "max": 0.1} or {"values": [11]}
        self.LEARNING_RATE = {"min": 0.0001, "max": 0.1}  # {"min": 0.0001, "max": 0.1}
        self.OPTIMIZER = {"values": ["SGD"]}


class LOBCASTSetupRun:
    def __init__(self):
        super().__init__()

        self.SETTINGS = Settings()
        self.__parse_cl_arguments(self.SETTINGS)
        self.SETTINGS.check_parameters_validity()
        self.__seed_everything(self.SETTINGS.SEED)

        # TIME TO SET PARAMS
        self.TUNABLE_H_PRAM = ConfigHPTunable()
        self.TUNED_H_PRAM = ConfigHPTuned()
        self.__setup_hyper_parameters()

    def end_setup(self, tuning_parameters, wandb_instance=None):
        self.__set_tuning_parameters(tuning_parameters)

        # self.RUN_NAME_PREFIX = self.run_name_prefix(self.SETTINGS)
        self.DATE_TIME = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.__setup_all_directories(self.DATE_TIME, self.SETTINGS)
        # print("RUNNING:\n>>", self.RUN_NAME_PREFIX)

        self.METRICS = Metrics(self.SETTINGS.__dict__, self.TUNED_H_PRAM.__dict__, self.SETTINGS.DIR_EXPERIMENTS)
        self.WANDB_INSTANCE = wandb_instance

    def __set_tuning_parameters(self, tuning_parameters=None):

        for key, value in tuning_parameters.items():
            self.TUNED_H_PRAM.__setattr__(key, value)

        # def assign_first_param():
        #     # for now, it only assigned the first element in a list of possible values
        #     for key, value in self.TUNABLE_H_PRAM.__dict__.items():
        #         if "values" not in value.keys() or len(value['values']) != 1:
        #             raise ValueError(f"Hyper parameter {key} is not correctly set for the single execution." +
        #                              f"Allowed definition is {key}: {{'values': list()}}, with list of size 1.\n" +
        #                              f"Otherwise allow for WANDB execution with --IS_WANDB True.")
        #         self.TUNED_H_PRAM.__setattr__(key, value['values'][0])
        #
        # def assign_wandb_param(tuning_parameters):
            # for now, it only assigned the first element in a list of possible values

        # if tuning_parameters is None:
        #     assign_first_param()
        # else:
        #     assign_wandb_param(tuning_parameters)

        # at this point parameters are set
        print("Running with parameters:")
        print(self.TUNED_H_PRAM.__dict__)

    def __setup_hyper_parameters(self):
        # add parameters from model
        for key, value in self.SETTINGS.PREDICTION_MODEL.value.tunable_parameters.items():
            self.TUNABLE_H_PRAM.add_hyperparameter(key, value)

        # set to default, add the same parameters in the TUNED_H_PRAM object
        for key, _ in self.TUNABLE_H_PRAM.__dict__.items():
            self.TUNED_H_PRAM.add_hyperparameter(key, None)

    def __seed_everything(self, seed):
        """ Sets the random seed to all the random generators. """
        seed_everything(seed)
        np.random.seed(seed)
        random.seed(seed)
        # self.RANDOM_GEN_DATASET = np.random.RandomState(seed)

    # @staticmethod
    # def cf_name_format(ext=""):
    #     return "MOD={}-SEED={}-TRS={}-TES={}-DS={}-HU={}-HP={}-HF={}-OB={}" + ext
    #
    # def run_name_prefix(self, settings):
    #     return self.cf_name_format().format(
    #         settings.PREDICTION_MODEL.name,
    #         settings.SEED,
    #         settings.STOCK_TRAIN_VAL,
    #         settings.STOCK_TEST,
    #         settings.DATASET_NAME.value,
    #         settings.PREDICTION_HORIZON_UNIT.name,
    #         settings.PREDICTION_HORIZON_PAST,
    #         settings.PREDICTION_HORIZON_FUTURE,
    #         settings.OBSERVATION_PERIOD,
    #     )

    def __parse_cl_arguments(self, settings):
        """ Parses the arguments for the command line. """
        parser = argparse.ArgumentParser(description='LOBCAST execution arguments:')

        # every field in the settings, can be set crom cl
        for k, v in settings.__dict__.items():
            var = v.name if isinstance(v, Enum) else v
            type_var = str if isinstance(v, Enum) else type(v)
            type_var = str_to_bool if type(v) == bool else type_var  # to parse bool
            parser.add_argument(f'--{k}', default=var, type=type_var)

        args = vars(parser.parse_args())

        print("Setting CLI parameters.")
        # every field in the settings, is set based on the parsed values, enums are parsed by NAME
        for k, v in settings.__dict__.items():
            value = v.__class__[args[k]] if isinstance(v, Enum) else args[k]
            settings.__setattr__(k, value)

    @staticmethod
    def __setup_all_directories(fname, settings):
        settings.DIR_EXPERIMENTS = f"{cst.DIR_EXPERIMENTS}-({fname})/"

        # create the paths for the simulation if they do not exist already
        paths = ["data", settings.DIR_EXPERIMENTS]
        for p in paths:
            if not os.path.exists(p):
                os.makedirs(p)
