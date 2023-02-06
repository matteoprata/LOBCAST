
from enum import Enum
import torch


class LearningHyperParameter(str, Enum):
    OPTIMIZER = "optimizer"
    LEARNING_RATE = "lr"
    WEIGHT_DECAY = "weight_decay"
    EPS = "eps"
    MOMENTUM = "momentum"
    EPOCHS_UB = "epochs"
    IS_SHUFFLE_TRAIN_SET = "is_shuffle"
    BATCH_SIZE = "batch_size"
    MLP_HIDDEN = "hidden_mlp"
    RNN_HIDDEN = "rnn_hidden"
    RNN_N_HIDDEN = "rnn_n_hidden"
    DAIN_LAYER_MODE = "dain_layer_mode"
    NUM_RBF_NEURONS = "num_rbf_neurons"
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
    SGD = "SGD"


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
    DOWNWARD = 0
    STATIONARY = 1
    UPWARD = 2


# to use in the future
class Models(str, Enum):
    MLP = "MLP"
    BINCTABL = "BINCTABL"
    CTABL = "CTABL"

    CNN1 = "CNN1"
    CNNLSTM = "CNNLSTM"
    ATNBoF = "ATNBoF"

    CNN2 = "CNN2"
    TLONBoF = "TLONBoF"
    DLA = "DLA"

    LSTM = "LSTM"
    DEEPLOBATT = "DEEPLOBATT"
    DEEPLOB = "DeepLob"

    DAIN = "DAIN"
    AXIALLOB = "AXIALLOB"
    TRANSLOB = "TransLob"
    # NBoF = "NBoF"  # do not use


class DatasetFamily(str, Enum):
    FI = "FI"
    LOBSTER = "Lobster"


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

NUM_GPUS = 1 if torch.cuda.is_available() else None
DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'

PROJECT_NAME = "LOB-CLASSIFIERS-({})"
DIR_EXPERIMENTS = "data/experiments/" + PROJECT_NAME
DIR_SAVED_MODEL = "data/saved_models/" + PROJECT_NAME

DATA_SOURCE = "data/"
DATASET_LOBSTER = "LOBSTER_6/unzipped/"
DATASET_FI = "FI-2010/BenchmarkDatasets"
DATA_PICKLES = "data/pickles/"


WANDB_SWEEP_MAX_RUNS = 30


class Servers(Enum):
    ALIEN1 = 1
    ALIEN2 = 2
    FISSO1 = 3


MACIDsServer = {
    'cf:a8:49:cb:e6:fb': Servers.ALIEN1,
    '91:29:87:cc:61:09': Servers.ALIEN2,
    'ea:82:88:8a:79:89': Servers.FISSO1
}

ServerMACIDs = {v: k for k, v in MACIDsServer.items()}

metrics_name = ['F1 Score (%)', 'Precision (%)', 'Recall (%)', 'Accuracy (%)', 'MCC']

DECLARED_PERF = {
    # F1 PRECISION RECALL ACCURACY MCC

    Models.MLP: [
        # https://ieeexplore.ieee.org/abstract/document/8081663
        [48.27, 60.78, 47.81, None, None],  # k = 10
        [51.12, 65.20, 51.33, None, None],  # k = 20
        [None, None, None, None, None],     # k = 30
        [55.95, 67.14, 55.21, None, None],  # k = 50
        [None, None, None, None, None]      # k = 100
    ],

    Models.LSTM: [
        # https://ieeexplore.ieee.org/abstract/document/8081663
        [66.33, 75.92, 60.77, None, None],  # k = 10
        [62.37, 70.52, 59.60, None, None],  # k = 20
        [None, None, None, None, None],     # k = 30
        [61.43, 68.50, 60.03, None, None],  # k = 50
        [None, None, None, None, None]      # k = 100
    ],

    Models.DEEPLOB: [
        # https://ieeexplore.ieee.org/abstract/document/8673598
        [83.40, 84.00, 84.47, 84.47, None],  # k = 10
        [72.82, 74.06, 74.85, 74.85, None],  # k = 20
        [None, None, None, None, None],      # k = 30
        [80.35, 80.38, 80.51, 80.51, None],  # k = 50
        [None, None, None, None, None]       # k = 100
    ],

    Models.TRANSLOB: [
        # https://arxiv.org/pdf/2003.00130.pdf
        [88.66, 91.81, 87.66, 87.66, None],  # k = 10
        [80.65, 86.17, 78.78, 78.78, None],  # k = 20
        [None, None, None, None, None],      # k = 30
        [88.20, 88.65, 88.12, 88.12, None],  # k = 50
        [91.61, 91.63, 91.62, 91.62, None]   # k = 100
    ],

    Models.CNN1: [
        # https://ieeexplore.ieee.org/abstract/document/8010701
        [55.21, 65.54, 50.98, None, None],  # k = 10
        [59.17, 67.38, 54.79, None, None],  # k = 20
        [None, None, None, None, None],     # k = 30
        [59.44, 67.12, 55.58, None, None],  # k = 50
        [None, None, None, None, None]      # k = 100
    ],

    Models.CNN2: [
        # https://www.sciencedirect.com/science/article/pii/S1568494620303410
        [46.0, 46.0, 53.0, None, None],  # k = 10
        [None, None, None, None, None],  # k = 20
        [None, None, None, None, None],  # k = 30
        [45.0, 45.0, 53.0, None, None],  # k = 50
        [44.0, 56.0, 51.0, None, None]   # k = 100
    ],

    Models.CNNLSTM: [
        # https://www.sciencedirect.com/science/article/pii/S1568494620303410
        [47.0, 46.0, 55.0, None, None],  # k = 10
        [None, None, None, None, None],  # k = 20
        [None, None, None, None, None],  # k = 30
        [47.0, 47.0, 56.0, None, None],  # k = 50
        [47.0, 47.0, 56.0, None, None]   # k = 100
    ],

    Models.DAIN: [
        # https://arxiv.org/pdf/1902.07892.pdf
        [68.26, 65.67, 71.58, 78.83, None],   # k = 10
        [65.31, 62.10, 70.48, 78.59, None],  # k = 20
        [None, None, None, None, None],      # k = 30
        [None, None, None, None, None],      # k = 50
        [None, None, None, None, None]       # k = 100
    ],

    Models.BINCTABL: [
        # https://ieeexplore.ieee.org/abstract/document/9412547
        [81.04, 80.29, 81.84, 86.87, None],   # k = 10
        [71.22, 72.12, 70.44, 77.28,  None],  # k = 20
        [None, None, None, None, None],       # k = 30
        [88.06, 89.50, 86.99, 88.54, None],   # k = 50
        [None, None, None, None, None]        # k = 100
    ],

    Models.CTABL: [
        # https://ieeexplore.ieee.org/abstract/document/8476227
        [77.63, 76.95, 78.44, 84.70, None],  # k = 10
        [66.93, 67.18, 66.94, 73.74, None],  # k = 20
        [None, None, None, None, None],      # k = 30
        [78.44, 79.05, 77.04, 79.87, None],  # k = 50
        [None, None, None, None, None]       # k = 100
    ],

    Models.DEEPLOBATT: [
        # https://arxiv.org/pdf/2105.10430.pdf
        [82.37, 82.50, 83.28, 83.28, None],  # k = 10
        [73.73, 74.31, 75.25, 75.25, None],  # k = 20
        [76.94, 77.32, 77.59, 77.59, None],  # k = 30
        [79.38, 79.51, 79.49, 79.49, None],  # k = 50
        [81.49, 81.62, 81.45, 81.45, None]   # k = 100
    ],

    Models.ATNBoF: [
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9762322
        [67.88, None, None, None, None],  # k = 10
        [60.04, None, None, None, None],  # k = 20
        [None, None, None, None, None],   # k = 30
        [73.40, None, None, None, None],  # k = 50
        [None, None, None, None, None]    # k = 100
    ],

    Models.AXIALLOB: [
        # https://arxiv.org/pdf/2212.01807.pdf
        [85.14, 84.93, 85.43, None, None],  # k = 10
        [75.78, 76.32, 76.98, None, None],  # k = 20
        [80.08, 80.54, 80.69, None, None],  # k = 30
        [83.27, 83.31, 83.38, None, None],  # k = 50
        [85.93, 86.04, 85.92, None, None]   # k = 100
    ],

    Models.DLA: [
        # https://link.springer.com/content/pdf/10.1007/s13369-022-07197-3.pdf
        [77.76, 74.34, 79.71, 82.64, None],  # k = 10
        [None, None, None, None, None],  # k = 20
        [79.38, 78.88, 80.16, 80.94, None],  # k = 30
        [78.96, 78.89, 79.20, 79.40, None],  # k = 50
        [None, None, None, None, None]   # k = 100
    ],

    Models.TLONBoF: [
        # https://www.sciencedirect.com/science/article/pii/S0167865520302245
        # Anchored evaluation setup
        [52.98, 50.20, 58.19, None, None],  # k = 10
        [None, None, None, None, None],  # k = 20
        [None, None, None, None, None],  # k = 30
        [None, None, None, None, None],  # k = 50
        [None, None, None, None, None]   # k = 100
    ]
}

# metrics_name = ['F1 Score (%)', 'Precision (%)', 'Recall (%)', 'Accuracy (%)', 'MCC']

MODELS_YEAR_DICT = {
    Models.MLP: 2017,
    Models.LSTM: 2017,
    Models.CNN1: 2017,
    Models.CTABL: 2018,
    Models.DEEPLOB: 2019,
    Models.DAIN: 2019,
    Models.CNNLSTM: 2020,
    Models.CNN2: 2020,
    Models.TRANSLOB: 2020,
    Models.TLONBoF: 2020,
    Models.BINCTABL: 2021,
    Models.DEEPLOBATT: 2021,
    Models.DLA: 2022,
    Models.ATNBoF: 2022,
    Models.AXIALLOB: 2022
}
MODELS_YEAR_DICT = {k: v for k, v in sorted(MODELS_YEAR_DICT.items(), key=lambda a: a[1])}

