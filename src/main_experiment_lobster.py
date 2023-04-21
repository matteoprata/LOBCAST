import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *

DEFAULT_SEEDS = set(range(500, 505))
DEFAULT_MODELS = cst.MODELS_15
DEFAULT_FORWARD_WINDOWS = [
    # cst.WinSize.EVENTS1,
    # cst.WinSize.EVENTS2,
    # cst.WinSize.EVENTS3,
    cst.WinSize.EVENTS5,
    # cst.WinSize.EVENTS10
]

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
                for window_forward in DEFAULT_FORWARD_WINDOWS:

                        print(f"Running LOBSTER experiment: model={mod}, fw={window_forward.value}, seed={see}")

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

                            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = cst.WinSize.EVENTS1.value

                            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = window_forward.value

                            cf.CHOSEN_MODEL = mod

                            cf.IS_WANDB = int(not is_debug)
                            cf.IS_TUNE_H_PARAMS = int(not is_debug)

                            launch_wandb(cf)

                        except KeyboardInterrupt:
                            print("There was a problem running on", server_name.name, "LOBSTER experiment on {}, with K+={}".format(mod, window_forward))
                            sys.exit()


servers = [cst.Servers.ALIEN1, cst.Servers.ALIEN2, cst.Servers.FISSO1]

execution_plan = {

    cst.Servers.FISSO1: [
        {
            'seeds': [500],
            'models': [cst.Models.BINCTABL, cst.Models.CTABL, cst.Models.DLA],
        }
    ],

    cst.Servers.ALIEN1: [
        {
            'seeds': [500],
            'models': [cst.Models.CNN1, cst.Models.LSTM, cst.Models.CNNLSTM],
        },
    ],

    cst.Servers.ALIEN2: [
        {
            'seeds': [500],
            'models': [cst.Models.CNN2, cst.Models.TLONBoF, cst.Models.DEEPLOBATT],
        },
    ],
}

now = 'LOBSTER-DEFINITIVE-EVENTS-2023-04-20'
experiment_lobster(execution_plan, dataset=cst.DatasetFamily.LOBSTER, now=now, servers=servers, is_debug=False)
