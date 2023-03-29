
import os
import sys

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
import time
import torch

# TORCH
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy


import src.constants as cst
import src.models.model_callbacks as cbk
from src.config import Configuration
from src.data_preprocessing.META.METADataBuilder import MetaDataBuilder
from src.metrics.metrics_learning import compute_metrics, compute_sk_cm
from src.data_preprocessing.FI.FIDataBuilder import FIDataBuilder

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

from src.models.nbof.nbof_param_search import HP_NBoF, HP_NBoF_FI_FIXED
from src.models.atnbof.atnbof_param_search import HP_ATNBoF, HP_ATNBoF_FI_FIXED
from src.models.tlonbof.tlonbof_param_search import HP_TLONBoF, HP_TLONBoF_FI_FIXED
from src.models.metalob.metalob_param_search import HP_META, HP_META_FIXED
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
    cst.Models.METALOB: HPSearchTypes2(HP_META, HP_META_FIXED),
    cst.Models.MAJORITY: HPSearchTypes2(HP_META, HP_META_FIXED)
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


def launch_single(config: Configuration, model_params=None, is_baseline=False):
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

        trainer = Trainer(
            accelerator=cst.DEVICE_TYPE,
            devices=cst.NUM_GPUS,
            check_val_every_n_epoch=config.VALIDATE_EVERY,
            max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS_UB],
            callbacks=[
                cbk.callback_save_model(config, config.WANDB_RUN_NAME),
                cbk.early_stopping(config)
            ],
        )
        trainer.fit(nn_engine, data_module)

        nn_engine.testing_mode = cst.ModelSteps.VALIDATION_MODEL
        trainer.test(nn_engine, dataloaders=data_module.val_dataloader(), ckpt_path="best")

        nn_engine.testing_mode = cst.ModelSteps.TESTING
        trainer.test(nn_engine, dataloaders=data_module.test_dataloader(), ckpt_path="best")

        if not config.IS_TUNE_H_PARAMS:
            config.METRICS_JSON.close()
    try:
        core(model_params)
    except:
        print("The following error was raised")
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)





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
