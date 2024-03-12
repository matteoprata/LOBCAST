import numpy as np

import src.constants as cst
from src.config import Configuration
from src.utils.util_training import NNEngine


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
from src.models.atnbof.atnbof import ATNBoF
from src.models.tlonbof.tlonbof import TLONBoF
from src.models.axial.axiallob import AxialLOB
from src.models.metalob.metalob import MetaLOB


def pick_model(config: Configuration, data_module):
    net_architecture = None
    loss_weights = None

    if config.PREDICTION_MODEL == cst.Models.MLP:
        net_architecture = MLP(
            num_features=np.prod(data_module.x_shape),  # 40 * wind
            num_classes=data_module.num_classes,
            hidden_layer_dim=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MLP_HIDDEN],
            p_dropout=config.HYPER_PARAMETERS[cst.LearningHyperParameter.P_DROPOUT],
        )

    elif config.PREDICTION_MODEL == cst.Models.CNN1:
        net_architecture = CNN1(
            num_features=data_module.x_shape[1],
            num_classes=data_module.num_classes,
        )

    elif config.PREDICTION_MODEL == cst.Models.CNN2:
        net_architecture = CNN2(
            num_features=data_module.x_shape[1],
            num_classes=data_module.num_classes,
        )

    elif config.PREDICTION_MODEL == cst.Models.LSTM:
        net_architecture = LSTM(
            x_shape=data_module.x_shape[1],  # 40, wind is the time
            num_classes=data_module.num_classes,
            hidden_layer_dim=config.HYPER_PARAMETERS[cst.LearningHyperParameter.RNN_HIDDEN],
            hidden_mlp=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MLP_HIDDEN],
            num_layers=config.HYPER_PARAMETERS[cst.LearningHyperParameter.RNN_N_HIDDEN],
            p_dropout=config.HYPER_PARAMETERS[cst.LearningHyperParameter.P_DROPOUT],
        )

    elif config.PREDICTION_MODEL == cst.Models.CNNLSTM:
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

    elif config.PREDICTION_MODEL == cst.Models.DAIN:
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

    elif config.PREDICTION_MODEL == cst.Models.DEEPLOB:
        net_architecture = DeepLob(num_classes=data_module.num_classes)

    elif config.PREDICTION_MODEL == cst.Models.TRANSLOB:
        net_architecture = TransLob(seq_len=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS])
        loss_weights = data_module.train_set.loss_weights

    elif config.PREDICTION_MODEL == cst.Models.CTABL:
        net_architecture = CTABL(60, 40, 10, 10, 120, 5, 3, 1)
        loss_weights = data_module.train_set.loss_weights

    elif config.PREDICTION_MODEL == cst.Models.BINCTABL:
        net_architecture = BiN_CTABL(60, 40, 10, 10, 120, 5, 3, 1)
        loss_weights = data_module.train_set.loss_weights

    elif config.PREDICTION_MODEL == cst.Models.DEEPLOBATT:
        net_architecture = DeepLobAtt()

    elif config.PREDICTION_MODEL == cst.Models.DLA:
        num_snapshots, num_features = data_module.x_shape
        net_architecture = DLA(
            num_snapshots=num_snapshots,
            num_features=num_features,
            hidden_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.RNN_HIDDEN]
        )

    elif config.PREDICTION_MODEL == cst.Models.TLONBoF:
        num_snapshots, num_features = data_module.x_shape
        net_architecture = TLONBoF(window=num_snapshots, split_horizon=5, use_scaling=True)
        loss_weights = data_module.train_set.loss_weights

    elif config.PREDICTION_MODEL == cst.Models.ATNBoF:
        num_snapshots, num_features = data_module.x_shape
        loss_weights = data_module.train_set.loss_weights
        net_architecture = ATNBoF(
            in_channels=1,
            series_length=num_snapshots * num_features,
            n_codeword=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MLP_HIDDEN],
            att_type='temporal',            # ['temporal', 'spatial']
            n_class=data_module.num_classes,
            dropout=config.HYPER_PARAMETERS[cst.LearningHyperParameter.P_DROPOUT]
        )

    elif config.PREDICTION_MODEL == cst.Models.AXIALLOB:
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

    elif config.PREDICTION_MODEL == cst.Models.METALOB:
        net_architecture = MetaLOB(
            mlp_hidden=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MLP_HIDDEN],
            chosen_models=[m for m in cst.MODELS_15 if m != cst.Models.DEEPLOBATT],
        )
        # net_architecture = MetaLOB2()

    engine = NNEngine(
        config=config,
        model_type=config.PREDICTION_MODEL,
        neural_architecture=net_architecture,
        optimizer=config.HYPER_PARAMETERS[cst.LearningHyperParameter.OPTIMIZER],
        lr=config.HYPER_PARAMETERS[cst.LearningHyperParameter.LEARNING_RATE],
        weight_decay=config.HYPER_PARAMETERS[cst.LearningHyperParameter.WEIGHT_DECAY],
        eps=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPS],
        momentum=config.HYPER_PARAMETERS[cst.LearningHyperParameter.MOMENTUM],
        loss_weights=loss_weights,
        remote_log=config.WANDB_INSTANCE,
        n_samples=len(data_module.train_set),
        n_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS_UB],
        n_batch_size=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE],
    ).to(cst.DEVICE_TYPE)

    return engine
