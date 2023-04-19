import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *
from src.utils.utilities import get_upper_diagonal_windows

DEFAULT_SEEDS = set(range(500, 505))
DEFAULT_MODELS = cst.MODELS_15


def experiment_lobster(execution_plan, dataset, now=None, servers=None, is_debug=False):

    now, server_name, server_id, n_servers = experiment_preamble(now, servers)
    lunches_server = execution_plan[server_name]

    for instance in lunches_server:
        seeds = instance['seeds']
        seeds = DEFAULT_SEEDS if seeds == 'all' else seeds

        for see in seeds:
            models = instance['models']
            models = DEFAULT_MODELS if models == 'all' else models

            for mod in models:
                for window_backward, window_forward in zip(instance['k-'], instance['k+']):

                        print(f"Running LOBSTER experiment: model={mod}, bw={window_backward}, fw={window_forward}, seed={see}")

                        try:
                            cf: Configuration = Configuration(now)
                            cf.SEED = see

                            set_seeds(cf)

                            cf.CHOSEN_DATASET = dataset
                            if mod == cst.Models.METALOB:
                                cf.CHOSEN_DATASET = cst.DatasetFamily.META

                            cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.ALL
                            cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.ALL
                            cf.CHOSEN_PERIOD = cst.Periods.JULY2021

                            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = window_backward.value
                            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = window_forward.value

                            cf.CHOSEN_MODEL = mod

                            cf.IS_WANDB = int(not is_debug)
                            cf.IS_TUNE_H_PARAMS = True

                            launch_wandb(cf)

                        except KeyboardInterrupt:
                            print("There was a problem running on", server_name.name, "LOBSTER experiment on {}, with K-={}, K+={}".format(mod, window_backward, window_forward))
                            sys.exit()


servers = [cst.Servers.ALIEN1, cst.Servers.ALIEN2, cst.Servers.FISSO1]

# backwards, forwards = get_upper_diagonal_windows(windows=[cst.WinSize.SEC10, cst.WinSize.SEC50, cst.WinSize.SEC100])
backwards = [cst.WinSize.SEC100, cst.WinSize.SEC100, cst.WinSize.SEC100, cst.WinSize.SEC50, cst.WinSize.SEC50, cst.WinSize.SEC10]
forwards  = [cst.WinSize.SEC100, cst.WinSize.SEC50,  cst.WinSize.SEC10,  cst.WinSize.SEC50, cst.WinSize.SEC10, cst.WinSize.SEC10]

execution_plan = {

    cst.Servers.FISSO1: [
        {
            'seeds': [500, 501, 502],
            'models': [cst.Models.MLP, cst.Models.BINCTABL, cst.Models.CTABL, cst.Models.DLA],
            'k-': backwards,
            'k+': forwards,
        }
    ],

    cst.Servers.ALIEN1: [
        {
            'seeds': [500, 502],
            'models': [cst.Models.CNN1, cst.Models.LSTM, cst.Models.DAIN, cst.Models.DEEPLOB],
            'k-': backwards,
            'k+': forwards,
        },
    ],

    cst.Servers.ALIEN2: [
        {
            'seeds': [500, 502],
            'models': [cst.Models.CNNLSTM, cst.Models.CNN2, cst.Models.TLONBoF, cst.Models.DEEPLOBATT],
            'k-': backwards,
            'k+': forwards,
        },
    ],
}

now = 'LOBSTER-DEFINITIVE-14-04-2023'
experiment_lobster(execution_plan, dataset=cst.DatasetFamily.LOBSTER, now=now, servers=servers, is_debug=False)

