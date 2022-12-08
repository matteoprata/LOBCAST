from enum import Enum
import numpy as np
import torch
from datetime import datetime

np.set_printoptions(suppress=True)

PROJECT_NAME = "lob-adversarial-attacks-22"


class TuningVars(Enum):

    OPTIMIZER = "optimizer"
    LEARNING_RATE = "lr"
    WEIGHT_DECAY = "weight_decay"

    EPOCHS = "epochs"

    IS_SHUFFLE = "is_shuffle"
    BATCH_SIZE = "batch_size"

    MLP_HIDDEN = "hidden_mlp"
    LSTM_HIDDEN = "lstm_hidden"
    LSTM_N_HIDDEN = "lstm_n_hidden"

    DAIN_LAYER_MODE = "dain_layer_mode"

    P_DROPOUT = "p_dropout"

    BACKWARD_WINDOW = "window_size_backward"
    FORWARD_WINDOW = "window_size_forward"
    LABELING_THRESHOLD = "labeling_threshold"
    LABELING_SIGMA_SCALER = "labeling_sigma_scaler"

    FI_HORIZON = 'fi_horizon_k'


class STK_OPEN(Enum):
    """ The modalities associated to a list of stocks. """
    # TODO rename
    TRAIN = "train_mod"
    TEST = "test_mod"


class Optimizers(Enum):
    ADAM = "Adam"
    RMSPROP = "RMSprop"


class Metrics(Enum):
    LOSS = 'loss'
    F1 = 'f1'
    PRECISION = 'precision'
    RECALL = 'recall'
    ACCURACY = 'accuracy'
    MCC = 'mcc'


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
    DAIN = "DAIN"
    TRANSLOB = "TransLob"
    CTABL = "CTABL"


class DatasetFamily(Enum):
    FI = "FI"
    LOBSTER = "Lobster"


class Stocks(Enum):
    AAPL = ["AAPL"]
    AMAT = ["AMAT"]
    ARVN = ["ARVN"]
    LYFT = ["LYFT"]

    ALL = ["AAPL", "TSLA", "ZM", "AAWW", "AGNC", "LYFT"]


class Periods(Enum):
    MARCH2020 = {
        'train': ('2020-03-02', '2020-03-20'),
        'val': ('2020-03-23', '2020-03-27'),
        'test': ('2020-03-30', '2020-04-03'),
    }

    JULY2021 = {
        'train': ('2021-07-01', '2021-07-22'),
        'val': ('2021-07-23', '2021-07-29'),
        'test': ('2021-07-30', '2021-08-06'),
    }


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

NUM_GPUS = 1 if torch.cuda.is_available() else 0
DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'

SAVED_MODEL_DIR = "data/saved_models/"
DATA_SOURCE = "data/"
DATASET_LOBSTER = "LOBSTER_6/unzipped/"
DATASET_FI = "FI-2010/BenchmarkDatasets"
DATA_PICKLES = "data/pickles/"

SEED = 0

VALIDATE_EVERY = 1

# MODELS PARAMS
EPOCHS = 100
BATCH_SIZE = 32
OPTIMIZER = Optimizers.ADAM.value
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0

# MLP
MLP_HIDDEN = 128
P_DROPOUT = .1

# LSTM
LSTM_HIDDEN = 32
LSTM_N_HIDDEN = 1

# DAIN
DAIN_LAYER_MODE = 'full'


IS_DATA_PRELOAD = True
IS_SHUFFLE_INPUT = True  # ONLY TRAIN
N_LOB_LEVELS = 10
NUM_SNAPSHOTS = 100
INSTANCES_LOWERBOUND = 1000  # under-sampling must have at least INSTANCES_LOWERBOUND instances

# LOBSTER way to label to measure percentage change
BACKWARD_WINDOW = WinSize.SEC100.value
FORWARD_WINDOW = WinSize.SEC50.value  # in LOBSTER = HORIZON
LABELING_SIGMA_SCALER = .9   # dynamic threshold

# K of the FI dataset
HORIZON = 10  # in FI = FORWARD_WINDOW

TRAIN_SPLIT_VAL = .8  # FI only

CHOSEN_DATASET = DatasetFamily.FI

CHOSEN_PERIOD = Periods.MARCH2020
CHOSEN_MODEL = Models.TRANSLOB

CHOSEN_STOCKS = {
    STK_OPEN.TRAIN: Stocks.LYFT,
    STK_OPEN.TEST: Stocks.NVDA
}

IS_WANDB = None

SWEEP_NAME = None

SWEEP_METRIC = {
    'goal': 'maximize',
    'name': ModelSteps.VALIDATION.value + Metrics.F1.value
}
SWEEP_METHOD = 'bayes'


#
# python -m src.main -data FI -period MARCH -model TRANSLOB -stock_train ALL -stock_test ALL -is_wandb 0
#
