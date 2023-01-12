
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

# TORCH
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything

import src.constants as cst
import src.models.model_callbacks as cbk
from src.config import Configuration

# DATASETS
from src.data_preprocessing.FI.fi_param_search import hyperparameters_fi
from src.data_preprocessing.LOB.lobster_param_search import hyperparameters_lobster

# MODELS
from src.models.mlp.mlp_param_search import hyperparameters_mlp
from src.models.tabl.tabl_param_search import hyperparameters_tabl
from src.models.translob.tlb_param_search import hyperparameters_tlb
from src.models.cnn1.cnn1_param_search import hyperparameters_cnn1
from src.models.cnn2.cnn2_param_search import hyperparameters_cnn2
from src.models.cnnlstm.cnnlstm_param_search import hyperparameters_cnnlstm
from src.models.dain.dain_param_search import hyperparameters_dain
from src.models.deeplob.dlb_param_search import hyperparameters_dlb
from src.models.lstm.lstm_param_search import hyperparameters_lstm


from src.main_model_data import pick_model, pick_dataset


SWEEP_CONF_DICT_MODEL = {
    cst.Models.MLP:  hyperparameters_mlp,
    cst.Models.CNN1: hyperparameters_cnn1,
    cst.Models.CNN2: hyperparameters_cnn2,
    cst.Models.LSTM: hyperparameters_lstm,
    cst.Models.CNNLSTM: hyperparameters_cnnlstm,
    cst.Models.DAIN: hyperparameters_dain,
    cst.Models.DEEPLOB: hyperparameters_dlb,
    cst.Models.TRANSLOB: hyperparameters_tlb,
    cst.Models.CTABL: hyperparameters_tabl,
}

SWEEP_CONF_DICT_DATA = {
    cst.DatasetFamily.FI:  hyperparameters_fi,
    cst.DatasetFamily.LOBSTER:  hyperparameters_lobster,
}


def _wandb_exe(config: Configuration):

    with wandb.init() as wandb_instance:
        config.WANDB_RUN_NAME = wandb_instance.name
        config.WANDB_INSTANCE = wandb_instance

        for param in cst.LearningHyperParameter:
            if param.value in wandb_instance.config:
                config.HYPER_PARAMETERS[param] = wandb_instance.config[param.value]

        launch_single(config)


def launch_single(config: Configuration):

    try:
        config.dynamic_config_setup()  # todo before launch

        data_module = pick_dataset(config)
        model = pick_model(config, data_module)

        trainer = Trainer(
            accelerator=cst.DEVICE_TYPE,
            devices=cst.NUM_GPUS,
            check_val_every_n_epoch=config.VALIDATE_EVERY,
            max_epochs=config.HYPER_PARAMETERS[cst.LearningHyperParameter.EPOCHS_UB],
            callbacks=[
                cbk.callback_save_model(config, config.CHOSEN_DATASET, config.CHOSEN_MODEL.value, config.WANDB_RUN_NAME),
                cbk.early_stopping(config)
            ]
        )
        trainer.fit (model, data_module)
        trainer.test(model, data_module, ckpt_path="best")

    except:
        print("The following error was raised")
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)

def launch_wandb(config: Configuration):
    # 🐝 STEP: initialize sweep by passing in config

    sweep_id = wandb.sweep(
        sweep={
            'command': ["${env}", "python3", "${program}", "${args}"],
            'program': "src/main.py",
            'name':   config.SWEEP_NAME,
            'method': config.SWEEP_METHOD,
            'metric': config.SWEEP_METRIC,
            'parameters': {
                **SWEEP_CONF_DICT_DATA[config.CHOSEN_DATASET],
                **SWEEP_CONF_DICT_MODEL[config.CHOSEN_MODEL]
            }
        },
        project=cst.PROJECT_NAME
    )
    wandb.agent(sweep_id, function=lambda: _wandb_exe(config))


def parser_cl_arguments(config: Configuration):
    """ Parses the arguments for the command line. """

    print("Setting arguments.")

    parser = argparse.ArgumentParser(description='Stock Price Trend Prediction arguments:')

    # python src/main.py -iw 0 -d LOBSTER -m MLP -p JULY2021 -str NFLX -ste NFLX
    # python src/main.py -iw 1 -d FI -m MLP

    parser.add_argument('-iw',  '--is_wandb',    default=config.IS_WANDB, type=int)
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

    config.CHOSEN_DATASET = cst.DatasetFamily[args["data"]]
    config.CHOSEN_PERIOD = cst.Periods[args["period"]]
    config.CHOSEN_MODEL = cst.Models[args["model"]]
    config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks[args["stock_train"]]
    config.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks[args["stock_test"]]

    for lp in cst.LearningHyperParameter:
        config.HYPER_PARAMETERS[lp] = args[lp.name]

    # STOCKS FI
    if config.CHOSEN_DATASET == cst.DatasetFamily.FI:
        config.CHOSEN_PERIOD = None
        config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.FI
        config.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.FI


def set_seeds(config: Configuration):
    # Reproducibility stuff
    seed_everything(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    config.RANDOM_GEN_DATASET = np.random.RandomState(config.SEED)


if __name__ == "__main__":

    cf = Configuration()
    parser_cl_arguments(cf)
    set_seeds(cf)
    cf.dynamic_config_setup()

    if cf.IS_WANDB:
        launch_wandb(cf)
    else:
        launch_single(cf)

