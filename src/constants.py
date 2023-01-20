
from enum import Enum
import torch


class LearningHyperParameter(str, Enum):
    OPTIMIZER = "optimizer"
    LEARNING_RATE = "lr"
    WEIGHT_DECAY = "weight_decay"
    EPS = "eps"
    EPOCHS_UB = "epochs"
    IS_SHUFFLE_TRAIN_SET = "is_shuffle"
    BATCH_SIZE = "batch_size"
    MLP_HIDDEN = "hidden_mlp"
    LSTM_HIDDEN = "lstm_hidden"
    LSTM_N_HIDDEN = "lstm_n_hidden"
    DAIN_LAYER_MODE = "dain_layer_mode"
    P_DROPOUT = "p_dropout"
    BACKWARD_WINDOW = "window_size_backward"
    FORWARD_WINDOW = "window_size_forward"
    # LABELING_THRESHOLD = "labeling_threshold"
    LABELING_SIGMA_SCALER = "labeling_sigma_scaler"
    FI_HORIZON = 'fi_horizon_k'
    NUM_SNAPSHOTS = 'num_snapshots'


class STK_OPEN(str, Enum):
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

    # MIN01 = 60
    # MIN05 = 60 * 5
    # MIN10 = 60 * 10
    # MIN20 = 60 * 20

    NONE = None


class FI_Horizons(Enum):
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
class Models(str, Enum):
    MLP = "MLP"
    CNN1 = "CNN1"
    CNN2 = "CNN2"
    LSTM = "LSTM"
    CNNLSTM = "CNNLSTM"
    DEEPLOB = "DeepLob"
    DAIN = "DAIN"
    TRANSLOB = "TransLob"

    CTABL = "CTABL"
    BINCTABL = "BINCTABL"
    DEEPLOBATT = "DEEPLOBATT"


class DatasetFamily(str, Enum):
    FI = "FI"
    LOBSTER = "Lobster"

FI_HORIZONS_MAPPINGS = {
    1: -5,
    2: -4,
    3: -3,
    5: -2,
    10: -1
}


# class Stocks(Enum):
#     AAPL = ["AAPL"]
#     TSLA = ["TSLA"]
#     ZM = ["ZM"]
#     AAWW = ["AAWW"]
#     AGNC = ["AGNC"]
#     LYFT = ["LYFT"]
#
#     ALL = ["AAPL", "TSLA", "ZM", "AAWW", "AGNC", "LYFT"]

class Stocks(list, Enum):
    SOFI = ["SOFI"]
    NFLX = ["NFLX"]
    CSCO = ["CSCO"]
    WING = ["WING"]
    SHLS = ["SHLS"]
    LSTR = ["LSTR"]
    FI = ["FI"]
    ALL = ["SOFI", "NFLX", "CSCO", "WING", "SHLS", "LSTR"]


class Periods(dict, Enum):
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

    FI = {}


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


class ExpIndependentVariables(Enum):
    MODEL = 'model'
    K_FI = 'k'
    FORWARD_WIN = 'fw'
    BACKWARD_WIN = 'bw'


N_LOB_LEVELS = 10

NUM_GPUS = 1 if torch.cuda.is_available() else 0
DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'

PROJECT_NAME = "LOB-CLASSIFIERS-({})"
DIR_EXPERIMENTS = "data/experiments/" + PROJECT_NAME
DIR_SAVED_MODEL = "data/saved_models/" + PROJECT_NAME

DATA_SOURCE = "data/"
DATASET_LOBSTER = "LOBSTER_6/unzipped/"
DATASET_FI = "FI-2010/BenchmarkDatasets"
DATA_PICKLES = "data/pickles/"


WANDB_SWEEP_MAX_RUNS = 15


class ServersMAC(Enum):
    ALIEN1 = 0
    # ALIEN2 = 1
    # FISSO1 = 2


ServerMACIDs = {
    '3b:5a:48:6f:c3:0e': ServersMAC.ALIEN1,
    # '91:29:87:cc:61:09': ServersMAC.ALIEN2,
    # '12:19:23:96:dd:8b': ServersMAC.FISSO1
}

