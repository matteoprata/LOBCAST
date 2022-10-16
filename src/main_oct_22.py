from src.data_preprocessing.LOBDataBuilder import LOBDataBuilder
from src.data_preprocessing.LOBDataModule import LOBDataModule
from src.data_preprocessing.LOBDataset import LOBDataset
import src.models.model_callbacks as cbk
import numpy as np

from pytorch_lightning import Trainer
from src.models.model_executor import NNEngine
from src.models.mlp.mlp_param_search import sweep_configuration_mlp
import src.config as co
import wandb

from src.models.mlp.mlp import MLP
from src.models.lstm.lstm import LSTM

# def parser_cl_arguments():
#     """ Parses the arguments for the command line. """
#
#     parser = argparse.ArgumentParser(description='PyTorch Training')
#
#     parser.add_argument('--model',    default=co.Models.MLP.value)
#     parser.add_argument('--data',     default=co.DatasetFamily.LOBSTER.value)
#     parser.add_argument('--is_sweep', default=True)
#     parser.add_argument('--back_win', default=co.BACKWARD_WINDOW)
#     parser.add_argument('--forw_win', default=co.FORWARD_WINDOW)
#
#     return parser


SWEEP_CONF_DICT = {co.Models.MLP:  sweep_configuration_mlp,
                   co.Models.LSTM: sweep_configuration_mlp}


def prepare_data():

    # train & validation
    lo_train = LOBDataBuilder(
        co.DATA_DIR,
        co.DatasetType.TRAIN,
        start_end_trading_day=("2022-03-01", "2022-03-07"),  # ("2022-03-01", "2022-03-07")
        is_data_preload=False,
        crop_trading_day_by=60 * 30
    )

    # use the same
    mu, sigma = lo_train.normalization_mean, lo_train.normalization_std

    # test
    lo_test = LOBDataBuilder(
        co.DATA_DIR,
        co.DatasetType.TEST,
        start_end_trading_day=("2022-03-08", "2022-03-12"),  # ("2022-03-02", "2022-03-03")
        is_data_preload=False,
        crop_trading_day_by=60 * 30,
        normalization_mean=mu,
        normalization_std=sigma
    )

    n_inst_train = int(len(lo_train.samples_x) * co.TRAIN_SPLIT_VAL)

    train_set = LOBDataset(x=lo_train.samples_x[:n_inst_train], y=lo_train.samples_y[:n_inst_train])
    val_set   = LOBDataset(x=lo_train.samples_x[n_inst_train:], y=lo_train.samples_y[n_inst_train:])
    test_set  = LOBDataset(x=lo_test.samples_x, y=lo_test.samples_y)

    print()
    print("Samples in the splits:")
    print(len(train_set), len(val_set), len(test_set))
    print()

    lob_dm = LOBDataModule(train_set, val_set, test_set, co.BATCH_SIZE)
    return lob_dm


def pick_model(chosen_model, data_module, remote_log):
    net_architecture = None
    if chosen_model == co.Models.MLP:
        net_architecture = MLP(x_shape=np.prod(data_module.x_shape),
                               y_shape=data_module.y_shape,
                               hidden_layer_dim=co.MLP_HIDDEN)

    elif chosen_model == co.Models.LSTM:
        net_architecture = LSTM(x_shape=data_module.x_shape[1],
                                y_shape=data_module.y_shape,
                                hidden_layer_dim=co.LSTM_HIDDEN,
                                num_layers=co.LSTM_N_HIDDEN)

    return NNEngine(net_architecture, lr=co.LEARNING_RATE, remote_log=remote_log)


def lunch_training():
    print("Lunching the execution of {}.".format(co.CHOSEN_MODEL))

    remote_log = None
    if co.IS_SWEEP:

        wandb.init()
        remote_log = wandb

        co.MLP_HIDDEN = wandb.config.hidden_mlp
        co.BATCH_SIZE = wandb.config.batch_size
        co.LEARNING_RATE = wandb.config.lr
        co.IS_SHUFFLE_INPUT = wandb.config.is_shuffle
        co.BACKWARD_WINDOW = wandb.config.window_size_backward
        co.FORWARD_WINDOW = co.BACKWARD_WINDOW
        co.LABELING_SIGMA_SCALER = wandb.config.labeling_sigma_scaler

    data_module = prepare_data()

    model = pick_model(co.CHOSEN_MODEL, data_module, remote_log)

    trainer = Trainer(gpus=co.DEVICE,
                      check_val_every_n_epoch=co.VALIDATE_EVERY,  # val_check_interval
                      max_epochs=co.EPOCHS,
                      callbacks=[cbk.callback_save_model(co.CHOSEN_MODEL.value),
                                 cbk.early_stopping()])

    trainer .fit(model, data_module)
    trainer.test(model, data_module)


def lunch_training_sweep():

    # 🐝 STEP: initialize sweep by passing in config
    sweep_id = wandb.sweep(sweep=SWEEP_CONF_DICT[co.CHOSEN_MODEL], project=co.PROJECT_NAME)
    wandb.agent(sweep_id, function=lunch_training)  # count=4 max trials
    # wandb agent -p lob-adversarial-attacks-22 -e matteoprata rygxo9ti


if __name__ == "__main__":

    if co.IS_SWEEP:
        lunch_training_sweep()
    else:
        lunch_training()
