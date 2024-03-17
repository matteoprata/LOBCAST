
import os
import numpy as np
import torch
import src.constants as cst
import multiprocessing

np.set_printoptions(suppress=True)
from src.utils.utils_generic import dict_to_string



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

