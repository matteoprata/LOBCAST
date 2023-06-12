
import sys

import argparse
import random
import numpy as np
import wandb
import traceback
import socket


# TORCH
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything


import src.constants as cst
import src.models.model_callbacks as cbk
from src.config import Configuration

# MODELS
from src.models.mlp.mlp_param_search import HP_MLP, HP_MLP_FI_FIXED, HP_MLP_LOBSTER_FIXED
from src.models.tabl.tabl_param_search import HP_TABL, HP_TABL_FI_FIXED, HP_TABL_LOBSTER_FIXED
from src.models.translob.tlb_param_search import HP_TRANS, HP_TRANS_FI_FIXED, HP_TRANS_LOBSTER_FIXED
from src.models.cnn1.cnn1_param_search import HP_CNN1, HP_CNN1_FI_FIXED, HP_CNN1_LOBSTER_FIXED
from src.models.cnn2.cnn2_param_search import HP_CNN2, HP_CNN2_FI_FIXED, HP_CNN2_LOBSTER_FIXED
from src.models.cnnlstm.cnnlstm_param_search import HP_CNNLSTM, HP_CNNLSTM_FI_FIXED, HP_CNNLSTM_LOBSTER_FIXED
from src.models.dain.dain_param_search import HP_DAIN, HP_DAIN_FI_FIXED, HP_DAIN_LOBSTER_FIXED
from src.models.deeplob.dlb_param_search import HP_DEEP, HP_DEEP_FI_FIXED, HP_DEEP_LOBSTER_FIXED
from src.models.lstm.lstm_param_search import HP_LSTM, HP_LSTM_FI_FIXED, HP_LSTM_LOBSTER_FIXED
from src.models.binctabl.binctabl_param_search import HP_BINTABL, HP_BINTABL_FI_FIXED, HP_BINTABL_LOBSTER_FIXED
from src.models.deeplobatt.dlbatt_param_search import HP_DEEPATT, HP_DEEPATT_FI_FIXED, HP_DEEPATT_LOBSTER_FIXED
from src.models.dla.dla_param_search import HP_DLA, HP_DLA_FI_FIXED, HP_DLA_LOBSTER_FIXED
from src.models.axial.axiallob_param_search import HP_AXIALLOB, HP_AXIALLOB_FI_FIXED, HP_AXIALLOB_LOBSTER_FIXED
from src.models.atnbof.atnbof_param_search import HP_ATNBoF, HP_ATNBoF_FI_FIXED, HP_ATNBoF_LOBSTER_FIXED
from src.models.tlonbof.tlonbof_param_search import HP_TLONBoF, HP_TLONBoF_FI_FIXED, HP_TLONBoF_LOBSTER_FIXED
from src.models.metalob.metalob_param_search import HP_META, HP_META_FIXED

from src.utils.utils_dataset import pick_dataset
from src.utils.utils_models import pick_model
from collections import namedtuple

HPSearchTypes  = namedtuple('HPSearchTypes', ("sweep", "fixed_fi", "fixed_lob"))
HPSearchTypes2 = namedtuple('HPSearchTypes', ("sweep", "fixed"))

# MAPS every model to 3 dictionaries of parameters:
#
# HPSearchTypes.sweep:     for the hyperparameters sweep
# HPSearchTypes.fixed_fi:  fixed parameters for the FI dataset
# HPSearchTypes.fixed_lob: fixed parameters for the LOBSTER dataset
#
HP_DICT_MODEL = {
    cst.Models.MLP:  HPSearchTypes(HP_MLP, HP_MLP_FI_FIXED, HP_MLP_LOBSTER_FIXED),
    cst.Models.CNN1: HPSearchTypes(HP_CNN1, HP_CNN1_FI_FIXED, HP_CNN1_LOBSTER_FIXED),
    cst.Models.CNN2: HPSearchTypes(HP_CNN2, HP_CNN2_FI_FIXED, HP_CNN2_LOBSTER_FIXED),
    cst.Models.LSTM: HPSearchTypes(HP_LSTM, HP_LSTM_FI_FIXED, HP_LSTM_LOBSTER_FIXED),
    cst.Models.CNNLSTM: HPSearchTypes(HP_CNNLSTM, HP_CNNLSTM_FI_FIXED, HP_CNNLSTM_LOBSTER_FIXED),
    cst.Models.DAIN: HPSearchTypes(HP_DAIN, HP_DAIN_FI_FIXED, HP_DAIN_LOBSTER_FIXED),
    cst.Models.DEEPLOB: HPSearchTypes(HP_DEEP, HP_DEEP_FI_FIXED, HP_DEEP_LOBSTER_FIXED),
    cst.Models.TRANSLOB: HPSearchTypes(HP_TRANS, HP_TRANS_FI_FIXED, HP_TRANS_LOBSTER_FIXED),
    cst.Models.CTABL: HPSearchTypes(HP_TABL, HP_TABL_FI_FIXED, HP_TABL_LOBSTER_FIXED),
    cst.Models.BINCTABL: HPSearchTypes(HP_BINTABL, HP_BINTABL_FI_FIXED, HP_BINTABL_LOBSTER_FIXED),
    cst.Models.DEEPLOBATT: HPSearchTypes(HP_DEEPATT, HP_DEEPATT_FI_FIXED, HP_DEEPATT_LOBSTER_FIXED),
    cst.Models.DLA: HPSearchTypes(HP_DLA, HP_DLA_FI_FIXED, HP_DLA_LOBSTER_FIXED),
    cst.Models.AXIALLOB: HPSearchTypes(HP_AXIALLOB, HP_AXIALLOB_FI_FIXED, HP_AXIALLOB_LOBSTER_FIXED),
    cst.Models.ATNBoF: HPSearchTypes(HP_ATNBoF, HP_ATNBoF_FI_FIXED, HP_ATNBoF_LOBSTER_FIXED),
    cst.Models.TLONBoF: HPSearchTypes(HP_TLONBoF, HP_TLONBoF_FI_FIXED, HP_TLONBoF_LOBSTER_FIXED),

    cst.Models.METALOB: HPSearchTypes2(HP_META, HP_META_FIXED),
    cst.Models.MAJORITY: HPSearchTypes2(HP_META, HP_META_FIXED)
}


def __run_training_loop(config: Configuration, model_params=None):
    """ Set the model parameters and lunch the training loop. """

    def core(config, model_params):

        # if no hyperparameter tuning must be done, use the fixed parameters
        if not config.IS_TUNE_H_PARAMS:
            assert model_params is None

            if config.CHOSEN_DATASET == cst.DatasetFamily.FI:
                model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed_fi

            elif config.CHOSEN_DATASET == cst.DatasetFamily.LOB:
                model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed_lob

            elif config.CHOSEN_DATASET == cst.DatasetFamily.META:
                model_params = HP_DICT_MODEL[config.CHOSEN_MODEL].fixed

        print("Setting model parameters", model_params)

        # SET hyperparameter in the config object
        for param in cst.LearningHyperParameter:
            if param.value in model_params:
                config.HYPER_PARAMETERS[param] = model_params[param.value]

        config.dynamic_config_setup()

        # vvv TRAINING LOOP vvv

        data_module = pick_dataset(config)    # load the data
        nn = pick_model(config, data_module)  # load the model

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

        # TRAINING STEP
        trainer.fit(nn, data_module)

        # FINAL VALIDATION STEP
        nn.testing_mode = cst.ModelSteps.VALIDATION_MODEL
        trainer.test(nn, dataloaders=data_module.val_dataloader(), ckpt_path="best")

        # TEST STEP
        nn.testing_mode = cst.ModelSteps.TESTING
        trainer.test(nn, dataloaders=data_module.test_dataloader(), ckpt_path="best")

        if not config.IS_TUNE_H_PARAMS:
            config.METRICS_JSON.close()

    try:
        core(config, model_params)
    except:
        print("The following error was raised:")
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)


def run(config: Configuration):
    """ Build a WANDB sweep from a configuration object. """

    def _wandb_exe(config: Configuration):
        """ LOG on WANDB console. """

        run_name = None
        if not config.IS_TUNE_H_PARAMS:
            config.dynamic_config_setup()
            run_name = config.WANDB_SWEEP_NAME

        with wandb.init(project=cst.PROJECT_NAME, name=run_name) as wandb_instance:
            # log simulation details in WANDB console

            wandb_instance.log_code("src/")
            wandb_instance.log({"model": config.CHOSEN_MODEL.name})
            wandb_instance.log({"seed": config.SEED})
            wandb_instance.log({"stock_train": config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name})
            wandb_instance.log({"stock_test": config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].name})
            wandb_instance.log({"period": config.CHOSEN_PERIOD.name})
            wandb_instance.log({"alpha": cst.ALPHA})

            if config.CHOSEN_DATASET in [cst.DatasetFamily.FI, cst.DatasetFamily.META]:
                wandb_instance.log({"fi-k": config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON]})

            wandb_instance.log({"back-win": config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW]})
            wandb_instance.log({"fwrd-win": config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW]})

            config.WANDB_RUN_NAME = wandb_instance.name
            config.WANDB_INSTANCE = wandb_instance

            params_dict = wandb_instance.config  # chosen parameters from WANDB search
            if not config.IS_TUNE_H_PARAMS:
                params_dict = None

            __run_training_loop(config, params_dict)

    # 🐝 STEP: initialize sweep by passing in cf
    config.dynamic_config_setup()  # initializes the simulation

    if config.IS_TUNE_H_PARAMS:
        sweep_id = wandb.sweep(
            sweep={
                'command': ["${env}", "python3", "${program}", "${args}"],
                'program': "src/utils_training_loop.py",
                'name':    config.WANDB_SWEEP_NAME,
                'method':  config.SWEEP_METHOD,
                'metric':  config.SWEEP_METRIC,
                'parameters': {
                    **HP_DICT_MODEL[config.CHOSEN_MODEL].sweep
                }
            },
            project=cst.PROJECT_NAME
        )
        wandb.agent(sweep_id, function=lambda: _wandb_exe(config), count=cst.WANDB_SWEEP_MAX_RUNS)
    else:
        # NO SWEEP
        _wandb_exe(config)


def set_seeds(config: Configuration):
    """ Sets the random seed to all the random generators. """
    seed_everything(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    config.RANDOM_GEN_DATASET = np.random.RandomState(config.SEED)


def experiment_preamble(run_name_prefix, servers):
    """
    Returns the run_name_prefix, server name, server id and number of servers.
    This function is used to run the same code on different machines identified by their hostname.
    Each machine will execute the model defined in the execution plan.
    """

    if run_name_prefix is None:
        parser = argparse.ArgumentParser(description='Stock Price Experiment FI:')
        parser.add_argument('-run_name_prefix', '--run_name_prefix', default=None)
        args = vars(parser.parse_args())
        run_name_prefix = args["run_name_prefix"]

    hostname = socket.gethostname()

    n_servers = len(cst.server2hostname) if servers is None else len(servers)
    servers = cst.server2hostname if servers is None else servers  # list of server
    servers_hostname = [cst.server2hostname[s] for s in servers]   # list of hostnames

    if hostname in servers_hostname:
        server_name = cst.hostname2server[hostname]
        server_id = servers_hostname.index(hostname)

    elif cst.Servers.ANY in servers:
        server_name = cst.Servers.ANY
        server_id = 0

    else:
        raise "This SERVER is not handled for the experiment."

    print("Running on server", server_name.name)
    return run_name_prefix, server_name, server_id, n_servers
