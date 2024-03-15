
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

        self.EPOCHS_UB = 5

        self.TRAIN_SET_PORTION = .8

        self.IS_TEST_ONLY = False
        self.TEST_MODEL_PATH: str = "data/saved_models/LOBCAST-(15-03-2024_20-23-49)/epoch=2-validation_f1=0.27.ckpt"

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.N_GPUs = None if self.DEVICE == 'cpu' else torch.cuda.device_count()
        self.N_CPUs = multiprocessing.cpu_count()

        self.PROJECT_NAME = ""
        self.DIR_EXPERIMENTS = ""

        self.SWEEP_METHOD = 'bayes'
        self.IS_WANDB = False

    def check_parameters_validity(self):
        CONSTRAINTS = []
        CONSTRAINTS += [not self.IS_TEST_ONLY or os.path.exists(self.TEST_MODEL_PATH)]  # if test, then test file should exist
        CONSTRAINTS += [not self.IS_TEST_ONLY or not self.IS_WANDB]  # if test, then test file should exist

        if not all(CONSTRAINTS):
            raise ValueError("Constraint not met! Check your parameters.")


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
        self.BATCH_SIZE = {"values": [64, 55]}   # {"min": 0.0001, "max": 0.1} or {"values": [11]}
        self.LEARNING_RATE = {"min": 0.0001, "max": 0.1}
        self.OPTIMIZER = {"values": ["SGD"]}


class LOBCASTSetupRun:  # TOGLIERE questa ISA
    def __init__(self):
        super().__init__()

        self.SETTINGS = Settings()
        self.__parse_cl_arguments(self.SETTINGS)
        self.SETTINGS.check_parameters_validity()
        self.__seed_everything(self.SETTINGS.SEED)

        # TIME TO SET PARAMS
        self.TUNABLE_H_PRAM = ConfigHPTunable()
        self.TUNED_H_PRAM = ConfigHPTuned()
        self.__setup_parameters()

    def end_setup(self, wandb_instance=None):
        tuning_parameters = None if wandb_instance is None else wandb_instance.config
        self.__set_tuning_parameters(tuning_parameters)

        # self.RUN_NAME_PREFIX = self.run_name_prefix(self.SETTINGS)
        self.DATE_TIME = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.__setup_all_directories(self.DATE_TIME, self.SETTINGS)
        # print("RUNNING:\n>>", self.RUN_NAME_PREFIX)

        self.METRICS = Metrics(self.SETTINGS.__dict__, self.TUNED_H_PRAM.__dict__)
        self.WANDB_INSTANCE = wandb_instance

    def __set_tuning_parameters(self, tuning_parameters=None):

        def assign_first_param():
            # for now, it only assigned the first element in a list of possible values
            for key, value in self.TUNABLE_H_PRAM.__dict__.items():
                if 'values' in value.keys():
                    value = value['values'][0]
                elif 'min' in value.keys() and 'max' in value.keys():
                    value = value['min']
                else:
                    raise ValueError("Hyper parameters are wrongly specified.")
                self.TUNED_H_PRAM.__setattr__(key, value)

        def assign_wandb_param(tuning_parameters):
            # for now, it only assigned the first element in a list of possible values
            for key, value in tuning_parameters.items():
                self.TUNED_H_PRAM.__setattr__(key, value)

        if tuning_parameters is None:
            assign_first_param()
        else:
            assign_wandb_param(tuning_parameters)

        # at this point parameters are set
        print("Running with parameters")
        print(self.TUNABLE_H_PRAM.__dict__)
        print(self.TUNED_H_PRAM.__dict__)

    def __setup_parameters(self):
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
            type_var = str if isinstance(v, Enum) else type(v)
            var = v.name if isinstance(v, Enum) else v
            parser.add_argument(f'--{k}', default=var, type=type_var)

        args = vars(parser.parse_args())  # TODO FIX BOOL

        # every field in the settings, is set based on the parsed values, enums are parsed by NAME
        for k, v in settings.__dict__.items():
            value = v.__class__[args[k]] if isinstance(v, Enum) else args[k]
            settings.__setattr__(k, value)

    @staticmethod
    def __setup_all_directories(fname, settings):
        """
        Creates two folders:
            (1) data.experiments.LOBCAST-(fname) for the jsons with the stats
            (2) data.saved_models.LOBCAST-(fname) for the models
        """

        # TODO consider using dates
        settings.PROJECT_NAME = cst.PROJECT_NAME.format(fname)
        # settings.DIR_SAVED_MODEL = cst.DIR_SAVED_MODEL.format(fname) + "/"
        settings.DIR_EXPERIMENTS = cst.DIR_EXPERIMENTS.format(fname) + "/"

        # create the paths for the simulation if they do not exist already
        paths = ["data", settings.DIR_EXPERIMENTS]
        for p in paths:
            if not os.path.exists(p):
                os.makedirs(p)
