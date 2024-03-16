

import argparse
import os
from datetime import datetime
from enum import Enum

import numpy as np
from pytorch_lightning import seed_everything

import src.constants as cst
from src.metrics.metrics_log import Metrics

np.set_printoptions(suppress=True)
from src.utils.utils_generic import str_to_bool
from src.settings import Settings
from src.hyper_parameters import ConfigHPTunable, ConfigHPTuned


class LOBCAST:
    def __init__(self):

        self.SETTINGS = Settings()
        self.TUNABLE_H_PRAM = ConfigHPTunable()
        self.TUNED_H_PRAM = ConfigHPTuned()
        self.__init_hyper_parameters()

    def update_settings(self, setting_params=None):
        if setting_params is None:
            self.__parse_cl_arguments(self.SETTINGS)
        else:
            # settings new settings
            for k, v in setting_params.items():
                self.SETTINGS.__setattr__(k, v)

        self.SETTINGS.check_parameters_validity()
        # at this point parameters are set
        print("Running with settings:\n", self.SETTINGS.__dict__)

    def update_hyper_parameters(self, tuning_parameters):
        for key, value in tuning_parameters.items():
            self.TUNED_H_PRAM.__setattr__(key, value)

        # at this point parameters are set
        print("Running with parameters:\n", self.TUNED_H_PRAM.__dict__)

    def end_setup(self, wandb_instance=None):
        self.__seed_everything(self.SETTINGS.SEED)

        # self.RUN_NAME_PREFIX = self.run_name_prefix(self.SETTINGS)
        self.DATE_TIME = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.__setup_all_directories(self.DATE_TIME, self.SETTINGS)
        # print("RUNNING:\n>>", self.RUN_NAME_PREFIX)

        self.METRICS = Metrics(self.SETTINGS.DIR_EXPERIMENTS)
        self.WANDB_INSTANCE = wandb_instance

    def __init_hyper_parameters(self):
        # add parameters from model
        for key, value in self.SETTINGS.PREDICTION_MODEL.value.tunable_parameters.items():
            self.TUNABLE_H_PRAM.add_hyperparameter(key, value)

        # set to default, add the same parameters in the TUNED_H_PRAM object
        for key, _ in self.TUNABLE_H_PRAM.__dict__.items():
            self.TUNED_H_PRAM.add_hyperparameter(key, None)

    def __seed_everything(self, seed):
        """ Sets the random seed to all the random generators. """
        seed_everything(seed)

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
        parser = argparse.ArgumentParser(description='LOBCAST arguments:')

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
