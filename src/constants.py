from enum import Enum
import torch
import numpy as np


'''
Backward: 1      Forward: 1      Alfa: 1e-06
train:   0.18    0.63    0.19
val:     0.19    0.62    0.19
test:    0.21    0.59    0.2

Backward: 1      Forward: 2      Alfa: 1e-06
train:   0.24    0.5     0.25
val:     0.25    0.5     0.25
test:    0.27    0.47    0.27

Backward: 1      Forward: 3      Alfa: 1e-06
train:   0.28    0.43    0.29
val:     0.28    0.43    0.29
test:    0.3     0.4     0.3

Backward: 1      Forward: 5      Alfa: 1e-06
train:   0.32    0.36    0.33
val:     0.32    0.35    0.33
test:    0.34    0.33    0.33

Backward: 1      Forward: 10     Alfa: 1e-06
train:   0.37    0.25    0.38
val:     0.37    0.25    0.38
test:    0.38    0.23    0.38

'''


'''
################### NFLX ###################
Backward: 1      Forward: 5      Alfa: 2.5e-05
train:   0.33    0.33    0.34
val:     0.31    0.37    0.32
test:    0.37    0.26    0.37
'''

'''
################### SOFI ###################
Backward: 1      Forward: 5      Alfa: 0.00011
train:   0.35    0.32    0.34
val:     0.35    0.31    0.34
test:    0.36    0.3     0.35
'''

ALFA_NFLX = 25e-6
ALFA_SOFI = 11e-5

# ALFA = ALFA_NFLX
# ALFA = ALFA_SOFI
ALFA = 1e-6


class LearningHyperParameter(str, Enum):
    OPTIMIZER = "optimizer_name"
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
    VALIDATION_EPOCH = "validation-epoch-last"
    VALIDATION_MODEL = "validation-model"
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

    EVENTS1 = 1
    EVENTS2 = 2
    EVENTS3 = 3
    EVENTS5 = 5
    EVENTS10 = 10

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

    METALOB = "MetaLOB"
    MAJORITY = "Majority"


class DatasetFamily(str, Enum):
    FI = "FI"
    LOBSTER = "Lobster"
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


def map_id_win(win):
    if win in [WinSize.EVENTS1, FI_Horizons.K1]:
        return 0
    elif win in [WinSize.EVENTS2, FI_Horizons.K2]:
        return 1
    elif win in [WinSize.EVENTS3, FI_Horizons.K3]:
        return 2
    elif win in [WinSize.EVENTS5, FI_Horizons.K5]:
        return 3
    elif win in [WinSize.EVENTS10, FI_Horizons.K10]:
        return 4


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
        'first_day': '2020-03-02', 'last_day': '2020-04-03',
        'train': ('2020-03-02', '2020-03-20'),
        'val': ('2020-03-23', '2020-03-27'),
        'test': ('2020-03-30', '2020-04-03'),
    }

    JULY2021 = {
        'first_day': '2021-07-01', 'last_day': '2021-08-06',
        'train': ('2021-07-01', '2021-07-08'),  # 'train': ('2021-07-01', '2021-07-22'),
        'val':   ('2021-07-09', '2021-07-12'),  # 'val': ('2021-07-23', '2021-07-29'),
        'test':  ('2021-07-13', '2021-07-15'),  # 'test': ('2021-07-30', '2021-08-06'),
    }

    FEBRUARY2022 = {
        'first_day': '2022-02-01', 'last_day': '2022-02-28',
        'train': ('2022-02-01', '2022-02-07'),
        'val': ('2022-02-08', '2022-02-11'),
        'test': ('2022-02-14', '2022-02-16'),
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
    Events1 = 1
    Events10 = 10



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
NUM_CLASSES = 3

DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = None if DEVICE_TYPE == 'cpu' else torch.cuda.device_count()

PROJECT_NAME = "LOB-CLASSIFIERS-({})"
DIR_EXPERIMENTS = "data/experiments/" + PROJECT_NAME
DIR_SAVED_MODEL = "data/saved_models/" + PROJECT_NAME

DATA_SOURCE = "data/"
DATASET_LOBSTER = "LOBSTER_6/unzipped/"
DATASET_FI = "FI-2010/BenchmarkDatasets/"
DATA_PICKLES = "data/pickles/"


WANDB_SWEEP_MAX_RUNS = 20


class Servers(Enum):
    ALIEN1 = 1
    ALIEN2 = 2
    FISSO1 = 3


hostname2server = {
    'novella-Alienware-h2': Servers.ALIEN1,
    'novella-Alienware-x15-R1': Servers.ALIEN2,
    'novella-pc': Servers.FISSO1
}
server2hostname = {v: k for k, v in hostname2server.items()}



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

# 15 models sorted as MODELS_YEAR_DICT 5 horizons (1,2,3,5,10)
# FI_2010_PERF = [[48.20959434, 44.0243892 , 47.16951297, 48.98859317, 51.59479558],
#                 [66.52923452, 58.84149884, 65.32833911, 66.94654334, 59.39128832],
#                 [49.2981125 , 46.13136848, 62.25851268, 65.81445807, 67.22525588],
#                 [69.50077035, 62.37239457, 70.41167476, 71.57144704, 73.92415787],
#                 [71.05508381, 62.35127564, 70.79542871, 75.44489737, 77.58435408],
#                 [53.93056855, 46.7289375 , 53.53170748, 61.20605517, 62.799441  ],
#                 [63.54372207, 49.11480102, 63.2928966 , 69.20978287, 70.97891493],
#                 [27.63152453, 35.39434307, 53.23187754, 67.89210732, 68.54819641],
#                 [61.38668962, 54.6851    , 59.78363276, 60.61423192, 60.45715238],
#                 [36.47748716, 51.72324997, 41.60755085, 52.39199265, 66.1857505 ],
#                 [81.06300774, 71.50503826, 80.77082955, 87.70825349, 92.10769365],
#                 [70.56721888, 54.76342387, 66.01384491, 73.56041705, 71.60695086],
#                 [79.38524762, 69.28080704, 78.93945331, 87.08451691, 52.21075607],
#                 [32.91944651, 34.17991529, 38.19156618, 48.07789058, 50.97924787],
#                 [73.15020001, 63.35440217, 72.84614792, 78.31442845, 79.1791612 ]]

# WEIGHTED
FI_2010_PERF = [[65.74377179, 56.14452946, 53.15137661, 50.31746373, 51.47770519],
                  [78.40785429, 68.44792813, 70.37974811, 68.85010655, 59.4167335 ],
                  [69.58885214, 60.32890785, 67.86601128, 67.86324748, 67.30060613],
                  [79.55667652, 70.06549521, 74.54480202, 73.42570824, 74.01284748],
                  [81.59580668, 71.32465338, 75.37239928, 77.21540053, 77.66519794],
                  [71.30514919, 59.59248009, 60.15242892, 62.50810158, 62.66051733],
                  [76.90810561, 60.67157714, 68.69422572, 70.9777472 , 71.03534574],
                  [58.54914632, 52.07809315, 59.64969677, 69.85694567, 68.60860028],
                  [73.67614854, 64.05925375, 65.44546187, 63.47067745, 60.61902107],
                  [53.66686466, 61.08706224, 47.72654493, 55.58775446, 66.21390711],
                  [86.98177076, 77.47037867, 83.1885721 , 88.16341976, 92.10187628],
                  [81.21714157, 67.23314782, 72.21800485, 75.76457718, 71.71897308],
                  [86.3330744 , 76.01299488, 81.62353532, 87.5662242 , 52.22851796],
                  [56.92774699, 51.43929883, 49.10522107, 54.06993253, 50.95347016],
                  [82.57120726, 71.95823014, 76.7334157 , 79.60485128, 79.23628572]]

# LOBSTER_JULY_PERF = [[48.25227797, 56.18452407, 58.22355265, 59.23443022, 55.39248188],
#                      [49.63839944, 58.00871409, 60.29796907, 60.62009442, 56.02019579],
#                      [52.50797566, 57.70744634, 60.23634193, 60.13117262, 56.67051702],
#                      [55.33279756, 60.71586914, 62.1936955 , 62.17906741, 57.84894123],
#                      [54.9734162 , 60.41654582, 62.67363393, 62.22340311, 57.40612576],
#                      [47.72452985, 56.61416299, 59.14995541, 60.00280448, 56.0512674 ],
#                      [51.83302159, 58.11894161, 59.89830414, 60.05484327, 55.31208034],
#                      [49.90113329, 55.91208795, 58.8963473 , 58.75261477, 54.041783  ],
#                      [53.80612441, 57.77913512, 60.66730062, 60.30672049, 55.78203722],
#                      [52.50035168, 57.9657403 , 60.05693988, 59.90070666, 55.98582226],
#                      [56.95137544, 62.40156297, 63.93490814, 63.49880684, 59.11217048],
#                      [54.76970002, 61.0935584 , 62.62186232, 62.76511042, 59.01475136],
#                      [51.23755834, 58.60629038, 61.2933899 , 60.3004019 , 57.05265239],
#                      [49.75273464, 53.07585724, 54.63441582, 57.16119287, 50.87468015],
#                      [54.04858666, 60.72775997, 62.5974736 , 62.35708977, 57.76763044]]
# WEIGHTED
LOBSTER_JULY_PERF = [[60.11047428, 60.75883992, 60.49401491, 59.17928698, 53.78360401],
                     [61.02505454, 62.4022689 , 62.45860611, 60.55873397, 54.54530985],
                     [62.91423034, 62.27166227, 62.44185993, 60.06461996, 54.97427781],
                     [63.12637956, 64.55029448, 64.184854  , 62.13010309, 55.95206386],
                     [64.83925729, 64.63819641, 64.74699468, 62.17003128, 55.64899231],
                     [59.21136751, 61.01965478, 61.2416941 , 59.94617996, 54.89830137],
                     [62.35127879, 62.56038328, 62.1080386 , 59.98450912, 53.44201768],
                     [61.05826857, 60.61549693, 61.17792632, 58.68578003, 52.37339388],
                     [62.03643652, 62.05236074, 62.74265109, 60.2395226 , 53.77742118],
                     [60.81292444, 62.10410679, 62.16567115, 59.83217683, 54.03786051],
                     [64.48558723, 66.00659904, 65.77162651, 63.44952825, 57.48771536],
                     [64.28288243, 65.09283957, 64.65595818, 62.71706234, 57.32329692],
                     [62.15089043, 63.03551681, 63.40065404, 60.22598694, 55.18553642],
                     [60.15844815, 58.32381671, 57.08584885, 57.09404231, 48.56630942],
                     [64.0353057 , 64.81642668, 64.62420283, 62.30420307, 56.17647777],
                     [62.18260287, 60.7267662 , 59.94937708, 58.32111319, 54.32811527],
                     [62.02177144, 60.59479102, 59.74870857, 58.02517253, 53.41501285],]


LOBSTER_FEB_PERF = [[60.11047428, 60.75883992, 60.49401491, 59.17928698, 53.78360401],
                     [61.02505454, 62.4022689 , 62.45860611, 60.55873397, 54.54530985],
                     [62.91423034, 62.27166227, 62.44185993, 60.06461996, 54.97427781],
                     [63.12637956, 64.55029448, 64.184854  , 62.13010309, 55.95206386],
                     [64.83925729, 64.63819641, 64.74699468, 62.17003128, 55.64899231],
                     [59.21136751, 61.01965478, 61.2416941 , 59.94617996, 54.89830137],
                     [62.35127879, 62.56038328, 62.1080386 , 59.98450912, 53.44201768],
                     [61.05826857, 60.61549693, 61.17792632, 58.68578003, 52.37339388],
                     [62.03643652, 62.05236074, 62.74265109, 60.2395226 , 53.77742118],
                     [60.81292444, 62.10410679, 62.16567115, 59.83217683, 54.03786051],
                     [64.48558723, 66.00659904, 65.77162651, 63.44952825, 57.48771536],
                     [64.28288243, 65.09283957, 64.65595818, 62.71706234, 57.32329692],
                     [62.15089043, 63.03551681, 63.40065404, 60.22598694, 55.18553642],
                     [60.15844815, 58.32381671, 57.08584885, 57.09404231, 48.56630942],
                     [64.0353057 , 64.81642668, 64.62420283, 62.30420307, 56.17647777],
                     [62.18260287, 60.7267662 , 59.94937708, 58.32111319, 54.32811527],
                     [62.02177144, 60.59479102, 59.74870857, 58.02517253, 53.41501285],]


FI_2010_PERF = np.array(FI_2010_PERF) / 100  # NORMALIZED
LOBSTER_JULY_PERF = np.array(LOBSTER_JULY_PERF) / 100  # NORMALIZED
LOBSTER_FEB_PERF = np.array(LOBSTER_FEB_PERF) / 100  # NORMALIZED

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
    Models.AXIALLOB: 2022,
    Models.METALOB: 2023,
    Models.MAJORITY: 2023
}

MODELS_YEAR_DICT = {k: v for k, v in sorted(MODELS_YEAR_DICT.items(), key=lambda a: a[1])}

MODELS_15 = list(set(list(Models))-{Models.METALOB, Models.MAJORITY})
MODELS_15 = [m for m in MODELS_YEAR_DICT if m in MODELS_15]

TRAINABLE_16 = list(set(list(Models))-{Models.MAJORITY})
TRAINABLE_16 = [m for m in MODELS_YEAR_DICT if m in TRAINABLE_16]

MODELS_17 = list(set(list(Models)))
MODELS_17 = [m for m in MODELS_YEAR_DICT if m in MODELS_17]


def model_dataset(model, bias="FI"):
    if model in [Models.METALOB]:
        return "Meta"
    return bias
