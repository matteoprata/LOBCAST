import numpy as np

import src.constants as cst
from src.config import Configuration
from src.models.model_executor import NNEngine
from src.models.nbof.nbof_centers import get_nbof_centers

# DATASETS
from src.data_preprocessing.FI.FIDataBuilder import FIDataBuilder
from src.data_preprocessing.FI.FIDataset import FIDataset
from src.data_preprocessing.DataModule import DataModule
from src.data_preprocessing.LOB.LOBDataset import LOBDataset

# MODELS
from src.models.mlp.mlp import MLP
from src.models.tabl.ctabl import CTABL
from src.models.translob.translob import TransLob
from src.models.cnn1.cnn1 import CNN1
from src.models.cnn2.cnn2 import CNN2
from src.models.cnnlstm.cnnlstm import CNNLSTM
from src.models.dain.dain import DAIN
from src.models.deeplob.deeplob import DeepLob
from src.models.lstm.lstm import LSTM
from src.models.binctabl.bin_tabl import BiN_CTABL
from src.models.deeplobatt.deeplobatt import DeepLobAtt
from src.models.dla.dla import DLA
from src.models.nbof.nbof import NBoF
from src.models.atnbof.atnbof import ATNBoF
from src.models.tlonbof.tlonbof import TLONBoF
from src.models.axial.axiallob import AxialLOB


def prepare_data_FI(config: Configuration):

    fi_train = FIDataBuilder(
        cst.DATA_SOURCE + cst.DATASET_FI,
        dataset_type=cst.DatasetType.TRAIN,
        horizon=config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
        window=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS],
        train_val_split=config.TRAIN_SPLIT_VAL,
        chosen_model=config.CHOSEN_MODEL
    )

    fi_val = FIDataBuilder(
        cst.DATA_SOURCE + cst.DATASET_FI,
        dataset_type=cst.DatasetType.VALIDATION,
        horizon=config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
        window=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS],
        train_val_split=config.TRAIN_SPLIT_VAL,
        chosen_model=config.CHOSEN_MODEL
    )

    # fi_test = FIDataBuilder(
    #     cst.DATA_SOURCE + cst.DATASET_FI,
    #     dataset_type=cst.DatasetType.TEST,
    #     horizon=config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
    #     window=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS],
    #     train_val_split=config.TRAIN_SPLIT_VAL
    # )

    fi_test = fi_val

    train_set = FIDataset(
        x=fi_train.get_samples_x(),
        y=fi_train.get_samples_y(),
        chosen_model=config.CHOSEN_MODEL
    )

    val_set = FIDataset(
        x=fi_val.get_samples_x(),
        y=fi_val.get_samples_y(),
        chosen_model=config.CHOSEN_MODEL
    )

    test_set = FIDataset(
        x=fi_test.get_samples_x(),
        y=fi_test.get_samples_y(),
        chosen_model=config.CHOSEN_MODEL
    )

    print()
    print("Samples in the splits:")
    print(len(train_set), len(val_set), len(test_set))
    print()

    fi_dm = DataModule(
        train_set, val_set, test_set,
        config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE],
        config.HYPER_PARAMETERS[cst.LearningHyperParameter.IS_SHUFFLE_TRAIN_SET]
    )

    return fi_dm


def prepare_data_LOBSTER(config: Configuration):

    train_set = LOBDataset(
        config=config,
        dataset_type=cst.DatasetType.TRAIN,
        stocks=config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value,
        start_end_trading_day=config.CHOSEN_PERIOD.value['train']
    )

    stockName2mu, stockName2sigma = train_set.stockName2mu, train_set.stockName2sigma

    val_set = LOBDataset(
        config=config,
        dataset_type=cst.DatasetType.VALIDATION,
        stocks=config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value,
        start_end_trading_day=config.CHOSEN_PERIOD.value['val'],
        stockName2mu=stockName2mu, stockName2sigma=stockName2sigma
    )

    test_set = LOBDataset(
        config=config,
        dataset_type=cst.DatasetType.TEST,
        stocks=config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value,
        start_end_trading_day=config.CHOSEN_PERIOD.value['test'],
        stockName2mu=stockName2mu, stockName2sigma=stockName2sigma
    )

    print()
    print("Samples in the splits:")
    print(len(train_set), len(val_set), len(test_set))
    print()

    lob_dm = DataModule(train_set, val_set, test_set, config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE],
                        config.HYPER_PARAMETERS[cst.LearningHyperParameter.IS_SHUFFLE_TRAIN_SET])
    return lob_dm


def pick_dataset(config: Configuration):
    if config.CHOSEN_DATASET == cst.DatasetFamily.LOBSTER:
        return prepare_data_LOBSTER(config)
    elif config.CHOSEN_DATASET == cst.DatasetFamily.FI:
        return prepare_data_FI(config)


def pick_model(config: Configuration, data_module):
    net_architecture = None
    loss_weights = None

    if config.CHOSEN_MODEL == cst.Models.MLP:
        net_architecture = MLP(
            num_features=np.prod(data_module.x_shape),  # 40 * wind
            num_classes=data_module.num_classes,
            hidden_layer_dim=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MLP_HIDDEN],
            p_dropout=config.HYPER_PARAMETERS[cst.LearningHyperParameter.P_DROPOUT],
        )

    elif config.CHOSEN_MODEL == cst.Models.CNN1:
        net_architecture = CNN1(
            num_features=data_module.x_shape[1],
            num_classes=data_module.num_classes,
        )

    elif config.CHOSEN_MODEL == cst.Models.CNN2:
        net_architecture = CNN2(
            num_features=data_module.x_shape[1],
            num_classes=data_module.num_classes,
        )

    elif config.CHOSEN_MODEL == cst.Models.LSTM:
        net_architecture = LSTM(
            x_shape=data_module.x_shape[1],  # 40, wind is the time
            num_classes=data_module.num_classes,
            hidden_layer_dim=config.HYPER_PARAMETERS[cst.LearningHyperParameter.RNN_HIDDEN],
            hidden_mlp=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MLP_HIDDEN],
            num_layers=config.HYPER_PARAMETERS[cst.LearningHyperParameter.RNN_N_HIDDEN],
            p_dropout=config.HYPER_PARAMETERS[cst.LearningHyperParameter.P_DROPOUT],
        )

    elif config.CHOSEN_MODEL == cst.Models.CNNLSTM:
        net_architecture = CNNLSTM(
            num_features=data_module.x_shape[1],
            num_classes=data_module.num_classes,
            batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE],
            seq_len=data_module.x_shape[0],
            hidden_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.RNN_HIDDEN],
            num_layers=config.HYPER_PARAMETERS[cst.LearningHyperParameter.RNN_N_HIDDEN],
            hidden_mlp=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MLP_HIDDEN],
            p_dropout=config.HYPER_PARAMETERS[cst.LearningHyperParameter.P_DROPOUT],
        )

    elif config.CHOSEN_MODEL == cst.Models.DAIN:
        net_architecture = DAIN(
            backward_window=data_module.x_shape[0],
            num_features=data_module.x_shape[1],
            num_classes=data_module.num_classes,
            mlp_hidden=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MLP_HIDDEN],
            p_dropout=config.HYPER_PARAMETERS[cst.LearningHyperParameter.P_DROPOUT],
            mode='adaptive_avg',
            mean_lr=1e-06,
            gate_lr=1e-02,
            scale_lr=1e-02
        )

    elif config.CHOSEN_MODEL == cst.Models.DEEPLOB:
        net_architecture = DeepLob(num_classes=data_module.num_classes)

    elif config.CHOSEN_MODEL == cst.Models.TRANSLOB:
        net_architecture = TransLob()

    elif config.CHOSEN_MODEL == cst.Models.CTABL:
        net_architecture = CTABL(60, 40, 10, 10, 120, 5, 3, 1)
        loss_weights = data_module.train_set.loss_weights

    elif config.CHOSEN_MODEL == cst.Models.BINCTABL:
        net_architecture = BiN_CTABL(60, 40, 10, 10, 120, 5, 3, 1)
        loss_weights = data_module.train_set.loss_weights

    elif config.CHOSEN_MODEL == cst.Models.DEEPLOBATT:
        net_architecture = DeepLobAtt()

    elif config.CHOSEN_MODEL == cst.Models.DLA:
        num_snapshots, num_features = data_module.x_shape
        net_architecture = DLA(
            num_snapshots=num_snapshots,
            num_features=num_features,
            hidden_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.RNN_HIDDEN]
        )

    # elif config.CHOSEN_MODEL == cst.Models.NBoF:
    #     raise AssertionError("Do not use this model!")
    #     num_snapshots, num_features = data_module.x_shape
    #     net_architecture = NBoF(
    #         num_snapshots=num_snapshots,
    #         num_features=num_features,
    #         num_rbf_neurons=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_RBF_NEURONS],
    #         hidden_mlp=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MLP_HIDDEN],
    #         centers=get_nbof_centers(data_module, k=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_RBF_NEURONS]),
    #         lr_W=0.01,
    #     )

    elif config.CHOSEN_MODEL == cst.Models.TLONBoF:
        num_snapshots, num_features = data_module.x_shape
        net_architecture = TLONBoF(window=num_snapshots, split_horizon=5, use_scaling=True)

    elif config.CHOSEN_MODEL == cst.Models.ATNBoF:
        num_snapshots, num_features = data_module.x_shape
        loss_weights = data_module.train_set.loss_weights
        net_architecture = ATNBoF(
            in_channels=1,
            series_length=num_snapshots*num_features,
            n_codeword=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MLP_HIDDEN],
            att_type='temporal',            # ['temporal', 'spatial']
            n_class=data_module.num_classes,
            dropout=config.HYPER_PARAMETERS[cst.LearningHyperParameter.P_DROPOUT]
        )

    elif config.CHOSEN_MODEL == cst.Models.AXIALLOB:
        num_snapshots, num_features = data_module.x_shape
        net_architecture = AxialLOB(
            W=40,
            H=40,
            c_in=32,
            c_out=32,
            c_final=4,
            n_heads=4,
            pool_kernel=(1, 4),
            pool_stride=(1, 4)
        )

    engine = NNEngine(
        config=config,
        model_type=config.CHOSEN_MODEL,
        neural_architecture=net_architecture,
        optimizer=config.HYPER_PARAMETERS[cst.LearningHyperParameter.OPTIMIZER],
        lr=config.HYPER_PARAMETERS[cst.LearningHyperParameter.LEARNING_RATE],
        weight_decay=config.HYPER_PARAMETERS[cst.LearningHyperParameter.WEIGHT_DECAY],
        eps=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPS],
        momentum=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MOMENTUM],
        loss_weights=loss_weights,
        remote_log=config.WANDB_INSTANCE,
        n_samples=len(data_module.train_set.x),
        n_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS_UB],
        n_batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE],
    ).to(cst.DEVICE_TYPE)

    return engine
