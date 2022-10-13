from enum import Enum
import numpy as np
import torch

np.set_printoptions(suppress=True)


# to use in the future
class MLModels(Enum):
    DEEPLOB = "DeepLob"
    MLP = "MLP"
    CNN = "CNN"


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
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_GAN = "data/GAN_models/"

SAVE_GAN_MODEL_EVERY = 50
VALIDATE_GAN_MODEL_EVERY = 2
EPOCHS = 300

SEED = 0
RANDOM_GEN_DATASET = np.random.RandomState(SEED)

# skip the first and last *BOUNDARY_PURGE seconds of the dataframe for every day
BOUNDARY_PURGE = 60*30

HISTORIC_WIN_SIZE = 100  # time units
FUTURE_WIN_SIZE = 50     # time units

DATA_DIR = "data/AVXL_010322_310322"


