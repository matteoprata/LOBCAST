
import os
import sys

from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

# UTILS
import argparse
import random
import numpy as np
import wandb
import traceback
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# TORCH
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
import torch.nn as nn
import src.constants as cst
import src.models.model_callbacks as cbk
from src.config import Configuration

# DATASETS
from src.data_preprocessing.LOB.lobster_param_search import HP_LOBSTER

# MODELS
from src.models.mlp.mlp_param_search import HP_MLP, HP_MLP_FI_FIXED
from src.models.tabl.tabl_param_search import HP_TABL, HP_TABL_FI_FIXED
from src.models.translob.tlb_param_search import HP_TRANS, HP_TRANS_FI_FIXED
from src.models.cnn1.cnn1_param_search import HP_CNN1, HP_CNN1_FI_FIXED
from src.models.cnn2.cnn2_param_search import HP_CNN2, HP_CNN2_FI_FIXED
from src.models.cnnlstm.cnnlstm_param_search import HP_CNNLSTM, HP_CNNLSTM_FI_FIXED
from src.models.dain.dain_param_search import HP_DAIN, HP_DAIN_FI_FIXED
from src.models.deeplob.dlb_param_search import HP_DEEP, HP_DEEP_FI_FIXED, HP_DEEP_LOBSTER_FIXED
from src.models.lstm.lstm_param_search import HP_LSTM, HP_LSTM_FI_FIXED
from src.models.binctabl.binctabl_param_search import HP_BINTABL, HP_BINTABL_FI_FIXED
from src.models.deeplobatt.dlbatt_param_search import HP_DEEPATT, HP_DEEPATT_FI_FIXED, HP_DEEPATT_LOBSTER_FIXED
from src.models.dla.dla_param_search import HP_DLA, HP_DLA_FI_FIXED
from src.models.axial.axiallob_param_search import HP_AXIALLOB, HP_AXIALLOB_FI_FIXED
from src.models.metalob import metalob
from src.models.nbof.nbof_param_search import HP_NBoF, HP_NBoF_FI_FIXED
from src.models.atnbof.atnbof_param_search import HP_ATNBoF, HP_ATNBoF_FI_FIXED
from src.models.tlonbof.tlonbof_param_search import HP_TLONBoF, HP_TLONBoF_FI_FIXED
from src.models.metalob.metalob_param_search import HP_META, HP_META_FIXED
from src.models.metalob.metalob import train_metaLOB, test_metaLOB, plot_correlation_vector
from src.utils.utilities import get_sys_mac
from src.main_helper import pick_model, pick_dataset
from collections import namedtuple


HPSearchTypes = namedtuple('HPSearchTypes', ("sweep", "fixed_fi", "fixed_lob"))
HPSearchTypes2 = namedtuple('HPSearchTypes', ("sweep", "fixed"))

HP_DICT_MODEL = {
    cst.Models.MLP:  HPSearchTypes(HP_MLP, HP_MLP_FI_FIXED, None),
    cst.Models.CNN1: HPSearchTypes(HP_CNN1, HP_CNN1_FI_FIXED, None),
    cst.Models.CNN2: HPSearchTypes(HP_CNN2, HP_CNN2_FI_FIXED, None),
    cst.Models.LSTM: HPSearchTypes(HP_LSTM, HP_LSTM_FI_FIXED, None),
    cst.Models.CNNLSTM: HPSearchTypes(HP_CNNLSTM, HP_CNNLSTM_FI_FIXED, None),
    cst.Models.DAIN: HPSearchTypes(HP_DAIN, HP_DAIN_FI_FIXED, None),
    cst.Models.DEEPLOB: HPSearchTypes(HP_DEEP, HP_DEEP_FI_FIXED, HP_DEEP_LOBSTER_FIXED),
    cst.Models.TRANSLOB: HPSearchTypes(HP_TRANS, HP_TRANS_FI_FIXED, None),
    cst.Models.CTABL: HPSearchTypes(HP_TABL, HP_TABL_FI_FIXED, None),
    cst.Models.BINCTABL: HPSearchTypes(HP_BINTABL, HP_BINTABL_FI_FIXED, None),
    cst.Models.DEEPLOBATT: HPSearchTypes(HP_DEEPATT, HP_DEEPATT_FI_FIXED, HP_DEEPATT_LOBSTER_FIXED),
    cst.Models.DLA: HPSearchTypes(HP_DLA, HP_DLA_FI_FIXED, None),
    cst.Models.AXIALLOB: HPSearchTypes(HP_AXIALLOB, HP_AXIALLOB_FI_FIXED, None),
    # cst.Models.NBoF: HPSearchTypes(HP_NBoF, HP_NBoF_FI_FIXED, None),
    cst.Models.ATNBoF: HPSearchTypes(HP_ATNBoF, HP_ATNBoF_FI_FIXED, None),
    cst.Models.TLONBoF: HPSearchTypes(HP_TLONBoF, HP_TLONBoF_FI_FIXED, None),
    cst.Models.METALOB: HPSearchTypes2(HP_META, HP_META_FIXED)
}

HP_DICT_DATASET = {
    cst.DatasetFamily.FI:  {},
    cst.DatasetFamily.LOBSTER:  HP_LOBSTER,
    cst.DatasetFamily.META: {},
}


# def save_models_on_wandb(config: Configuration):
#     """ Dumps all the models in the wandb run. """
#
#     if config.WANDB_INSTANCE:
#         path = cst.DIR_SAVED_MODEL + config.WANDB_SWEEP_NAME
#         for f in os.listdir(path):
#             art = wandb.Artifact(f.replace("=", "-"), type="model")
#             art.add_file(path + "/" + f, f)
#             config.WANDB_INSTANCE.log_artifact(art)


def _wandb_exe(config: Configuration):
    """ Either a single wandb run or a sweep. """

    run_name = None
    if not config.IS_TUNE_H_PARAMS:
        config.dynamic_config_setup()
        run_name = config.WANDB_SWEEP_NAME

    with wandb.init(project=cst.PROJECT_NAME, name=run_name) as wandb_instance:
        wandb_instance.log_code("src/")
        wandb_instance.log({"model": config.CHOSEN_MODEL.name})
        wandb_instance.log({"seed": config.SEED})

        if config.CHOSEN_DATASET == cst.DatasetFamily.FI:
            wandb_instance.log({"fi-k":  config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON]})

        elif config.CHOSEN_DATASET == cst.DatasetFamily.LOBSTER:
            wandb_instance.log({"back-win":  config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW]})
            wandb_instance.log({"fwrd-win":  config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW]})

        config.WANDB_RUN_NAME = wandb_instance.name
        config.WANDB_INSTANCE = wandb_instance

        params_dict = wandb_instance.config   # YES SWEEP
        if not config.IS_TUNE_H_PARAMS:       # NO SWEEP thus
            params_dict = None
        launch_single(config, params_dict)


def launch_single(config: Configuration, model_params=None):
    def core(model_params):

        # selects the parameters for the run
        if not config.IS_TUNE_H_PARAMS:
            assert model_params is None

            if config.CHOSEN_DATASET == cst.DatasetFamily.FI:
                model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed_fi
            elif config.CHOSEN_DATASET == cst.DatasetFamily.LOBSTER:
                model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed_lob
            elif config.CHOSEN_DATASET == cst.DatasetFamily.META:
                model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed

        print("Setting model parameters", model_params)

        for param in cst.LearningHyperParameter:
            if param.value in model_params:
                config.HYPER_PARAMETERS[param] = model_params[param.value]

        config.dynamic_config_setup()
        data_module = pick_dataset(config)

        nn_engine = pick_model(config, data_module)
        model = nn_engine.neural_architecture
        launch_model(nn_engine, model, data_module, config)

        if not config.IS_TUNE_H_PARAMS:
            config.METRICS_JSON.close()

    try:
        core(model_params)
    except:
        print("The following error was raised")
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)


def launch_model(engine, model, data_module, config):

    #extracting the hyperparameters
    lr = config.HYPER_PARAMETERS[cst.LearningHyperParameter.LEARNING_RATE]
    momentum = config.HYPER_PARAMETERS[cst.LearningHyperParameter.MOMENTUM]
    n_samples_train = data_module.x_shape[0]
    horizon = config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON]
    batch_size = config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE]
    n_epochs = config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS_UB]
    opt = engine.configure_optimizer()
    weights = data_module.train_set.loss_weights
    loss = nn.CrossEntropyLoss(weights)

    #training the model
    train(model, config, lr, momentum, n_samples_train, data_module)

    #loading the best model
    model = torch.load(f'data/saved_models/{str(type(model))[8:-2]}_k={horizon}.pt')

    #testing the model with the best model
    meta_predictions = test(model, config, data_module)


def train(model, config, n_epochs, opt, loss_function, data_module):
    best_val_loss = np.inf
    best_test_epoch = 0
    horizon = config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON]
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    device = cst.DEVICE_TYPE

    for it in tqdm(range(n_epochs)):

        model.train()
        t0 = datetime.now()
        train_loss = []

        for inputs, targets in train_loader:
            loss = training_step(model, inputs, targets, opt, loss_function, device)
            train_loss.append(loss)

        # Get mean train loss
        mean_train_loss = np.mean(train_loss)

        model.eval()
        val_loss = []
        for inputs, targets in val_loader:
            loss = val_step(model, inputs, targets, loss_function, device)
            val_loss.append(loss)

        mean_val_loss = np.mean(val_loss)

        # We save the best model
        if mean_val_loss < best_val_loss:
            torch.save(model, f'data/saved_models/{str(type(model))[8:-2]}k={horizon}.pt')
            best_val_loss = mean_val_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {it + 1}/{n_epochs}, Train Loss: {mean_train_loss:.4f}, \
          Validation Loss: {mean_val_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')


def test(model, config, data_module):
    n_correct = 0.
    n_total = 0.
    all_targets = []
    all_predictions = []
    test_loader = data_module.test_dataloader()
    device = cst.DEVICE_TYPE

    for inputs, targets in test_loader:
        predictions = test_step(model, inputs, targets, device)

        # update counts
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)

    #computing and plot the metrics
    print(classification_report(all_targets, all_predictions, digits=4))
    c = confusion_matrix(all_targets, all_predictions, normalize="true")
    disp = ConfusionMatrixDisplay(c)
    disp.plot()
    plt.show()

    return all_predictions


def training_step(model, inputs, targets, opt, loss, device):

    # move data to GPU
    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

    # zero the parameter gradients
    opt.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = loss(outputs, targets)

    # Backward and optimize
    loss.backward()
    opt.step()
    return loss.item()


def val_step(model, inputs, targets, loss, device):
    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
    outputs = model(inputs)
    loss = loss(outputs, targets)
    return loss.item()


def test_step(model, inputs, targets, device):

    # Move to GPU
    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

    # Forward pass
    outputs = model(inputs)

    # Get prediction
    _, predictions = torch.max(outputs, 1)

    return predictions


def launch_wandb(config: Configuration):
    # 🐝 STEP: initialize sweep by passing in config

    config.dynamic_config_setup()

    if config.IS_TUNE_H_PARAMS:
        # DO SWEEP
        sweep_id = wandb.sweep(
            sweep={
                'command': ["${env}", "python3", "${program}", "${args}"],
                'program': "src/main_single.py",
                'name':   config.WANDB_SWEEP_NAME,
                'method': config.SWEEP_METHOD,
                'metric': config.SWEEP_METRIC,
                'parameters': {
                    **HP_DICT_DATASET[config.CHOSEN_DATASET],
                    **HP_DICT_MODEL  [config.CHOSEN_MODEL].sweep
                }
            },
            project=cst.PROJECT_NAME
        )
        wandb.agent(sweep_id, function=lambda: _wandb_exe(config), count=cst.WANDB_SWEEP_MAX_RUNS)
    else:
        # NO SWEEP
        _wandb_exe(config)


def parser_cl_arguments(config: Configuration):
    """ Parses the arguments for the command line. """

    print("Setting arguments.")

    parser = argparse.ArgumentParser(description='Stock Price Trend Prediction arguments:')

    parser.add_argument('-iw',  '--is_wandb',   default=config.IS_WANDB, type=int)
    parser.add_argument('-ih',  '--is_tune',    default=config.IS_TUNE_H_PARAMS, type=int)
    parser.add_argument('-d',   '--data',        default=config.CHOSEN_DATASET.name)
    parser.add_argument('-m',   '--model',       default=config.CHOSEN_MODEL.name)
    parser.add_argument('-p',   '--period',      default=config.CHOSEN_PERIOD.name)
    parser.add_argument('-str', '--stock_train', default=config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)
    parser.add_argument('-ste', '--stock_test',  default=config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].name)

    for lp in cst.LearningHyperParameter:
        val = config.HYPER_PARAMETERS[lp]
        parser.add_argument('-{}'.format(lp.name), '--{}'.format(lp.name), default=val, type=type(val))

    # Parsing arguments from cli
    args = vars(parser.parse_args())

    # Setting args from cli in the config
    config.IS_WANDB = bool(args["is_wandb"])
    config.IS_TUNE_H_PARAMS = bool(args["is_tune"])

    config.CHOSEN_DATASET = cst.DatasetFamily[args["data"]]
    config.CHOSEN_PERIOD = cst.Periods[args["period"]]
    config.CHOSEN_MODEL = cst.Models[args["model"]]
    config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks[args["stock_train"]]
    config.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks[args["stock_test"]]

    for lp in cst.LearningHyperParameter:
        config.HYPER_PARAMETERS[lp] = args[lp.name]

    # STOCKS FI
    if config.CHOSEN_DATASET == cst.DatasetFamily.FI:
        config.CHOSEN_PERIOD = cst.Periods.FI
        config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.FI
        config.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.FI


def set_seeds(config: Configuration):
    # Reproducibility stuff
    seed_everything(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    config.RANDOM_GEN_DATASET = np.random.RandomState(config.SEED)


def set_configuration():
    cf = Configuration()
    parser_cl_arguments(cf)
    cf.dynamic_config_setup()
    return cf


def experiment_preamble(now, servers):
    if now is None:
        parser = argparse.ArgumentParser(description='Stock Price Experiment FI:')
        parser.add_argument('-now', '--now', default=None)
        args = vars(parser.parse_args())
        now = args["now"]

    mac = get_sys_mac()

    n_servers = len(cst.ServerMACIDs) if servers is None else len(servers)
    servers = cst.ServerMACIDs if servers is None else servers  # list of server
    servers_mac = [cst.ServerMACIDs[s] for s in servers]  # list of macs

    if mac in servers_mac:
        server_name = cst.MACIDsServer[mac]
        server_id = servers_mac.index(mac)
        print("Running on server", server_name.name)
    else:
        raise "This SERVER is not handled for the experiment."

    return now, server_name, server_id, n_servers


if __name__ == "__main__":

    cf = set_configuration()
    set_seeds(cf)
    if cf.IS_WANDB:
        launch_wandb(cf)
    else:
        launch_single(cf)

    # python src/main_single.py -iw 1 -ih 1 -d LOBSTER -m MLP -p JULY2021 -str NFLX -ste NFLX
    # python src/main_single.py -iw 0 -ih 0 -d FI -m MLP
    # python src/main_single.py -iw 0 -ih 0 -d FI -m MLP -FI_HORIZON 1
    # python src/main_single.py -iw 0 -ih 0 -d FI -m MLP -FI_HORIZON 2
    # python src/main_single.py -iw 0 -ih 0 -d FI -m MLP -FI_HORIZON 3
    # python src/main_single.py -iw 0 -ih 0 -d FI -m MLP -FI_HORIZON 5
    # python src/main_single.py -iw 0 -ih 0 -d FI -m MLP -FI_HORIZON 10
    # python -m src.main_single -iw 0 -ih 0 -d LOBSTER -p JULY2021 -m DEEPLOBATT -str ALL -ste ALL
