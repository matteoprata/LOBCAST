
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
from pytorch_lightning.loggers import WandbLogger

import src.constants as cst
import src.models.model_callbacks as cbk
from src.config import Configuration

# DATASETS
from src.data_preprocessing.FI.fi_param_search import HP_FI
from src.data_preprocessing.LOB.lobster_param_search import HP_LOBSTER

# MODELS
from src.models.mlp.mlp_param_search import HP_MLP, HP_MLP_FI_FIXED
from src.models.tabl.tabl_param_search import HP_TABL, HP_TABL_FI_FIXED
from src.models.translob.tlb_param_search import HP_TRANS, HP_TRANS_FI_FIXED
from src.models.cnn1.cnn1_param_search import HP_CNN1, HP_CNN1_FI_FIXED
from src.models.cnn2.cnn2_param_search import HP_CNN2, HP_CNN2_FI_FIXED
from src.models.cnnlstm.cnnlstm_param_search import HP_CNNLSTM, HP_CNNLSTM_FI_FIXED
from src.models.dain.dain_param_search import HP_DAIN, HP_DAIN_FI_FIXED
from src.models.deeplob.dlb_param_search import HP_DEEP, HP_DEEP_FI_FIXED
from src.models.lstm.lstm_param_search import HP_LSTM, HP_LSTM_FI_FIXED


from src.main_helper import pick_model, pick_dataset

from collections import namedtuple
HPSearchTypes = namedtuple('HPSearchTypes', ("sweep", "fixed_fi", "fixed_lob"))

HP_DICT_MODEL = {
    cst.Models.MLP:  HPSearchTypes(HP_MLP, HP_MLP_FI_FIXED, None),
    cst.Models.CNN1: HPSearchTypes(HP_CNN1, HP_CNN1_FI_FIXED, None),
    cst.Models.CNN2: HPSearchTypes(HP_CNN2, HP_CNN2_FI_FIXED, None),
    cst.Models.LSTM: HPSearchTypes(HP_LSTM, HP_LSTM_FI_FIXED, None),
    cst.Models.CNNLSTM: HPSearchTypes(HP_CNNLSTM, HP_CNNLSTM_FI_FIXED, None),
    cst.Models.DAIN: HPSearchTypes(HP_DAIN, HP_DAIN_FI_FIXED, None),
    cst.Models.DEEPLOB: HPSearchTypes(HP_DEEP, HP_DEEP_FI_FIXED, None),
    cst.Models.TRANSLOB: HPSearchTypes(HP_TRANS, HP_TRANS_FI_FIXED, None),
    cst.Models.CTABL: HPSearchTypes(HP_TABL, HP_TABL_FI_FIXED, None),
}

SWEEP_CONF_DICT_DATA = {
    cst.DatasetFamily.FI:  HP_FI,
    cst.DatasetFamily.LOBSTER:  HP_LOBSTER,
}


def save_models_on_wandb(config: Configuration):
    if config.WANDB_INSTANCE:
        path = cst.SAVED_MODEL_DIR + "/" + config.SWEEP_NAME
        for f in os.listdir(path):
            art = config.WANDB_INSTANCE.Artifact(f, type="model")
            art.add_file(path + "/" + f, f)
            config.WANDB_INSTANCE.log_artifact(art)


def _wandb_exe(config: Configuration):

    with wandb.init() as wandb_instance:
        wandb_instance.log_code("src/")
        config.WANDB_RUN_NAME = wandb_instance.name
        config.WANDB_INSTANCE = wandb_instance

        params_dict = wandb_instance.config   # YES SWEEP
        if not config.IS_TUNE_H_PARAMS:       # NO SWEEP thus
            params_dict = None
        launch_single(config, params_dict)


def launch_single(config: Configuration, model_params=None):
    def core(model_params):

        if not config.IS_TUNE_H_PARAMS:
            assert model_params is None
            if config.CHOSEN_DATASET == cst.DatasetFamily.FI:
                model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed_fi
            elif config.CHOSEN_DATASET == cst.DatasetFamily.LOBSTER:
                model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed_lob

        for param in cst.LearningHyperParameter:
            if param.value in model_params:
                config.HYPER_PARAMETERS[param] = model_params[param.value]

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
        trainer.fit(model, data_module)
        trainer.test(model, data_module, ckpt_path="best")
        print("CLOSING.....")
        config.METRICS_JSON.close()
        save_models_on_wandb(config)
    try:
        core(model_params)
    except:
        print("The following error was raised")
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)


def launch_wandb(config: Configuration):
    # 🐝 STEP: initialize sweep by passing in config

    if config.IS_TUNE_H_PARAMS:
        # DO SWEEP
        sweep_id = wandb.sweep(
            sweep={
                'command': ["${env}", "python3", "${program}", "${args}"],
                'program': "src/main.py",
                'name':   config.SWEEP_NAME,
                'method': config.SWEEP_METHOD,
                'metric': config.SWEEP_METRIC,
                'parameters': {
                    **SWEEP_CONF_DICT_DATA[config.CHOSEN_DATASET].sweep,
                    **HP_DICT_MODEL[config.CHOSEN_MODEL].sweep
                }
            },
            project=cst.PROJECT_NAME
        )
        wandb.agent(sweep_id, function=lambda: _wandb_exe(config))
    else:
        # NO SWEEP
        _wandb_exe(config)


def parser_cl_arguments(config: Configuration):
    """ Parses the arguments for the command line. """

    print("Setting arguments.")

    parser = argparse.ArgumentParser(description='Stock Price Trend Prediction arguments:')

    parser.add_argument('-iw',  '--is_wandb',    default=config.IS_WANDB, type=int)
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

    # python src/main.py -iw 1 -ih 1 -d LOBSTER -m MLP -p JULY2021 -str NFLX -ste NFLX
    # python src/main.py -iw 0 -ih 0 -d FI -m MLP

    # for mod in cst.Models:
    #     for k in cst.FI_Horizons:

    # python src/main.py -iw 0 -ih 0 -d FI -m MLP -FI_HORIZON 1
    # python src/main.py -iw 0 -ih 0 -d FI -m MLP -FI_HORIZON 2
    # python src/main.py -iw 0 -ih 0 -d FI -m MLP -FI_HORIZON 3
    # python src/main.py -iw 0 -ih 0 -d FI -m MLP -FI_HORIZON 5
    # python src/main.py -iw 0 -ih 0 -d FI -m MLP -FI_HORIZON 10
