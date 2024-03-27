
import os
import numpy as np
import torch
import src.constants as cst
import multiprocessing

np.set_printoptions(suppress=True)
from src.utils.utils_generic import dict_to_string
from enum import Enum


class SettingsExp(Enum):
    SEED = "SEED"
    PREDICTION_MODEL = "PREDICTION_MODEL"
    PREDICTION_HORIZON_FUTURE = "PREDICTION_HORIZON_FUTURE"
    PREDICTION_HORIZON_PAST = "PREDICTION_HORIZON_PAST"
    OBSERVATION_PERIOD = "OBSERVATION_PERIOD"


class Settings:
    """ A class with all the settings of the simulations. Settings are set at runtime from command line. """
    def __init__(self):

        self.SEED: int = 0
        """ The random seed of the simulation. """

        self.DATASET_NAME: cst.DatasetFamily = cst.DatasetFamily.FI
        """ Name of the dataset to run tests on. """

        self.N_TRENDS = 3
        """ The number of trends to use for predictions. """

        self.PREDICTION_MODEL = cst.Models.MLP
        self.PREDICTION_HORIZON_UNIT: cst.UnitHorizon = cst.UnitHorizon.EVENTS
        """ The time unit for time series discretization. """

        self.PREDICTION_HORIZON_FUTURE: int = 5
        self.PREDICTION_HORIZON_PAST: int = 1
        self.OBSERVATION_PERIOD: int = 100
        self.IS_SHUFFLE_TRAIN_SET = True

        self.EPOCHS_UB = 30
        """ The number of training epochs. """

        self.TRAIN_SET_PORTION = .8
        self.VALIDATION_EVERY = 1

        self.IS_TEST_ONLY = False
        """ Whether or not to run the simulation in test mode. If True, no train or validation are performed. """

        self.TEST_MODEL_PATH: str = "data/saved_models/LOBCAST-(15-03-2024_20-23-49)/epoch=2-validation_f1=0.27.ckpt"
        """ The path to the model to test. """

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.N_GPUs = None if self.DEVICE == 'cpu' else torch.cuda.device_count()
        self.N_CPUs = multiprocessing.cpu_count()

        self.DIR_EXPERIMENTS = ""
        self.IS_WANDB = True
        self.WANDB_SWEEP_METHOD = 'grid'
        """ Whether or not to use wandb. """

        self.IS_SANITY_CHECK = False
        """ Whether or not to use sanity checks. """

    def check_parameters_validity(self):
        """ Checks if the parameters set at runtime are valid. """
        CONSTRAINTS = []
        c1 = (not self.IS_TEST_ONLY or os.path.exists(self.TEST_MODEL_PATH), "If IS_TEST_ONLY, then test model should exist.")

        c2 = (not self.DATASET_NAME == cst.DatasetFamily.FI or self.PREDICTION_HORIZON_UNIT == cst.UnitHorizon.EVENTS,
              f"FI-2010 Dataset can handle only event based granularity, {self.PREDICTION_HORIZON_UNIT} given.")

        c3 = (not self.DATASET_NAME == cst.DatasetFamily.FI or self.PREDICTION_HORIZON_PAST == 1,
              f"FI-2010 Dataset can handle only 1 event in the past horizon, {self.PREDICTION_HORIZON_PAST} given.")

        c4 = (not self.DATASET_NAME == cst.DatasetFamily.FI or self.PREDICTION_HORIZON_FUTURE in [1, 2, 3, 5, 10],
              f"FI-2010 Dataset can handle only {1, 2, 3, 5, 10} events in the future horizon, {self.PREDICTION_HORIZON_FUTURE} given.")

        c5 = (not self.DATASET_NAME == cst.DatasetFamily.FI or self.N_TRENDS == 3,
              f"FI-2010 Dataset can handle only 3 trends, {self.N_TRENDS} given.")

        c6 = (not self.PREDICTION_MODEL == cst.Models.BINCTABL or self.OBSERVATION_PERIOD == 10,
              f"At the moment, BINCTABL only allows OBSERVATION_PERIOD = 10, {self.OBSERVATION_PERIOD} given.")

        CONSTRAINTS += [c1, c2, c3, c4, c5, c6]
        for constrain, description in CONSTRAINTS:
            if not constrain:
                raise ValueError(f"Constraint not met! {description} Check your parameters.")

    def __repr__(self):
        return dict_to_string(self.__dict__)
