from enum import Enum


class Optimizers(Enum):
    ADAM = "Adam"
    RMSPROP = "RMSprop"
    SGD = "SGD"


class Metrics(Enum):
    LOSS = 'loss'
    CM = 'cm'
    F1 = 'f1'
    F1_W = 'f1_w'

    PRECISION = 'precision'
    PRECISION_W = 'precision_w'

    RECALL = 'recall'
    RECALL_W = 'recall_w'

    ACCURACY = 'accuracy'
    MCC = 'mcc'
    COK = 'cohen-k'


class ModelSteps(Enum):
    TRAINING = "training"
    VALIDATION = "validation"  # final validation
    TESTING = "testing"


VALIDATION_METRIC = "{}_{}".format(ModelSteps.VALIDATION.value, Metrics.F1.value)


class NormalizationType(Enum):
    Z_SCORE = 0
    DYNAMIC = 1
    NONE = 2
    MINMAX = 3
    DECPRE = 4


class FI_Horizons(Enum):
    K1 = 1
    K2 = 2
    K3 = 3
    K5 = 5
    K10 = 10


class Predictions(Enum):
    DOWNWARD = 0
    STATIONARY = 1
    UPWARD = 2


from src.models.models_classes import *
# to use in the future


class DatasetFamily(str, Enum):
    FI = "FI"
    LOB = "Lobster"
    META = "Meta"


HORIZONS_MAPPINGS_FI = {
    1: -5,
    2: -4,
    3: -3,
    5: -2,
    10: -1
}

HORIZONS_MAPPINGS_LOBSTER = {
    10: -5,
    20: -4,
    30: -3,
    50: -2,
    100: -1
}


class OrderEvent(Enum):
    """ The possible kind of orders in the lob """
    SUBMISSION = 1
    CANCELLATION = 2
    DELETION = 3
    EXECUTION = 4
    HIDDEN_EXECUTION = 5
    CROSS_TRADE = 6
    TRADING_HALT = 7
    OTHER = 8


class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "val"


DOWNLOAD_FI_COMMAND = ("wget --content-disposition \"https://download.fairdata.fi:443/download?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3MTEyMzAxODksImRhdGFzZXQiOiI3M2ViNDhkNy00ZGJjLTRhMTAtYTUyYS1kYTc0NWI0N2E2NDkiLCJwYWNrYWdlIjoiNzNlYjQ4ZDctNGRiYy00YTEwLWE1MmEtZGE3NDViNDdhNjQ5X2JoeXV4aWZqLnppcCIsImdlbmVyYXRlZF9ieSI6IjlmZGRmZmVlLWY4ZDItNDZkNS1hZmIwLWQyOTM0NzdlZjg2ZiIsInJhbmRvbV9zYWx0IjoiYjVkYzQxOTAifQ.bgDP51aFumRtPMbJUtUcjhpnu-O6nI6OYZlDbc3lrfQ\"")


class ExpIndependentVariables(Enum):
    MODEL = 'model'
    K_FI = 'k'
    FORWARD_WIN = 'fw'
    BACKWARD_WIN = 'bw'


N_LOB_LEVELS = 10
NUM_CLASSES = 3

PROJECT_NAME = "LOBCAST"
VERSION = 2.0

PROJECT_NAME_VERSION = f"{PROJECT_NAME}-v{VERSION}"
DIR_EXPERIMENTS = f"data/experiments/{PROJECT_NAME_VERSION}"
DIR_SAVED_MODEL = f"data/saved_models/{PROJECT_NAME_VERSION}"
DATASET_FI = "data/datasets/FI-2010/BenchmarkDatasets/"

METRICS_RUNNING_FILE_NAME = "metrics_train.json"
METRICS_BEST_FILE_NAME = "metrics_best.json"
WANDB_SWEEP_MAX_RUNS = 20


class UnitHorizon(Enum):
    SECONDS = "seconds"
    HOURS = "hours"
    MINUTES = "minutes"
    DAYS = "days"
    EVENTS = "events"
