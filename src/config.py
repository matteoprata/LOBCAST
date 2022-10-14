from enum import Enum
import numpy as np
import torch

np.set_printoptions(suppress=True)


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
    STATIC = 0
    DYNAMIC = 1
    NONE = 2


class WinSize(Enum):
    SEC10 = 10
    SEC20 = 20
    SEC30 = 30

    MIN01 = 60
    MIN05 = 60 * 5
    MIN10 = 60 * 10
    MIN20 = 60 * 20


class Predictions(Enum):
    UPWARD = 2
    DOWNWARD = 0
    STATIONARY = 1


# to use in the future
class Models(Enum):
    DEEPLOB = "DeepLob"
    MLP = "MLP"
    CNN = "CNN"


class DatasetFamily(Enum):
    FI = "Fi"
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


OHLC_DATA = "old_data/ohlc_data/"
DEVICE = 1 if torch.cuda.is_available() else 0

MODEL_GAN = "data/GAN_models/"

SAVE_GAN_MODEL_EVERY = 50
VALIDATE_GAN_MODEL_EVERY = 2
EPOCHS = 500

SEED = 0
RANDOM_GEN_DATASET = np.random.RandomState(SEED)

# skip the first and last *BOUNDARY_PURGE seconds of the dataframe for every day
BOUNDARY_PURGE = 60*30

HISTORIC_WIN_SIZE = 100  # time units
FUTURE_WIN_SIZE = 50     # time units

BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATE_EVERY = 1

N_LOB_LEVELS = 10
BACKWARD_WINDOW = WinSize.SEC20.value
FORWARD_WINDOW = WinSize.SEC20.value
SAVED_MODEL_DIR = "data/saved_models"

TRAIN_SPLIT_VAL = .7
DATA_DIR = "data/AVXL_010322_310322"

PROJECT_NAME = "lob-adversarial-attacks-22"
