from enum import Enum
import numpy as np
import torch
from datetime import datetime

np.set_printoptions(suppress=True)

PROJECT_NAME = "lob-adversarial-attacks-22"

class TuningVars(Enum):
    BATCH_SIZE = "batch_size"
    LEARNING_RATE = "lr"
    EPOCHS = "epochs"
    IS_SHUFFLE = "is_shuffle"

    MLP_HIDDEN = "hidden_mlp"
    LSTM_HIDDEN = "lstm_hidden"
    LSTM_N_HIDDEN = "lstm_n_hidden"

    P_DROPOUT = "p_dropout"

    BACKWARD_WINDOW = "window_size_backward"
    FORWARD_WINDOW = "window_size_forward"
    LABELING_THRESHOLD = "labeling_threshold"
    LABELING_SIGMA_SCALER = "labeling_sigma_scaler"

    FI_HORIZON = 'fi_horizon_k'


class Metrics(Enum):
    LOSS = 'loss'
    F1 = 'f1'
    PRECISION = 'precision'
    RECALL = 'recall'
    ACCURACY = 'accuracy'


class ModelSteps(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


class NormalizationType(Enum):
    Z_SCORE = 0
    DYNAMIC = 1
    NONE = 2
    MINMAX = 3
    DECPRE = 4


class WinSize(Enum):
    SEC10 = 10
    SEC20 = 20
    SEC30 = 30
    SEC50 = 50
    SEC100 = 100

    MIN01 = 60
    MIN05 = 60 * 5
    MIN10 = 60 * 10
    MIN20 = 60 * 20

class Horizons(Enum):
    K1 = 1
    K2 = 2
    K3 = 3
    K5 = 5
    K10 = 10

class Predictions(Enum):
    UPWARD = 2
    DOWNWARD = 0
    STATIONARY = 1


# to use in the future
class Models(Enum):
    MLP = "MLP"
    CNN1 = "CNN1"
    CNN2 = "CNN2"
    LSTM = "LSTM"
    CNNLSTM = "CNNLSTM"
    DEEPLOB = "DeepLob"


class DatasetFamily(Enum):
    FI = "FI"
    LOBSTER = "Lobster"


class Granularity(Enum):
    """ The possible Granularity to build the OHLC old_data from lob """
    Sec1 = "1S"
    Sec5 = "5S"
    Sec15 = "15S"
    Sec30 = "30S"
    Min1 = "1Min"
    Min5 = "5Min"
    Min15 = "15Min"
    Min30 = "30Min"
    Hour1 = "1H"
    Hour2 = "2H"
    Hour6 = "6H"
    Hour12 = "12H"
    Day1 = "1D"
    Day2 = "2D"
    Day5 = "7D"
    Month1 = "30D"


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


CLASS_NAMES = ["DOWN", "STATIONARY", "UP"]

OHLC_DATA = "old_data/ohlc_data/"
NUM_GPUS = 1 if torch.cuda.is_available() else 0
DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_GAN = "data/GAN_models/"

SAVED_MODEL_DIR = "data/saved_models/"

SEED = 0
RANDOM_GEN_DATASET = np.random.RandomState(SEED)
DATA_SOURCE = "data/"
DATASET_LOB = "AVXL_2021-08-01_2021-08-31_10"
DATASET_FI = "FI-2010/BenchmarkDatasets"

DATA_PICKLES = "data/pickles/"

EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 0.001
VALIDATE_EVERY = 1
IS_SHUFFLE_INPUT = False

MLP_HIDDEN = 128
P_DROPOUT = .1

LSTM_HIDDEN = 32
LSTM_N_HIDDEN = 1

N_LOB_LEVELS = 10
LABELING_SIGMA_SCALER = .5  # dynamic threshold
BACKWARD_WINDOW = WinSize.SEC100.value
FORWARD_WINDOW = WinSize.SEC10.value
INSTANCES_LOWERBOUND = 1000

HORIZON = 10

TRAIN_SPLIT_VAL = .8

CHOSEN_DATASET = DatasetFamily.FI
CHOSEN_MODEL = Models.CNNLSTM

IS_WANDB = False
SWEEP_NAME = CHOSEN_DATASET.value + '_' + CHOSEN_MODEL.value + ''
SWEEP_METHOD = 'bayes'
SWEEP_METRIC = {
    'goal': 'maximize',
    'name': ModelSteps.VALIDATION.value + Metrics.F1.value
}
