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
    cst.WinSize.EVENTS1,
    cst.WinSize.EVENTS2,
    cst.WinSize.EVENTS3,
    cst.WinSize.EVENTS5,
    cst.WinSize.EVENTS10
]


def experiment_lobster(execution_plan, dataset, now=None, servers=None, is_debug=False, json_dir=None, target_dataset_meta=None, peri=None):

    now, server_name, server_id, n_servers = experiment_preamble(now, servers)
    lunches_server = execution_plan[server_name]

    for instance in lunches_server:
        seeds = instance['seeds']
        seeds = DEFAULT_SEEDS if seeds == 'all' else seeds

        for seed in seeds:
            models = instance['models']
            models = DEFAULT_MODELS if models == 'all' else models

            for mod in models:
                forward_windows = instance['forward_windows']
                forward_windows = DEFAULT_FORWARD_WINDOWS if forward_windows == 'all' else forward_windows

                for window_forward in forward_windows:

                        print(f"Running LOBSTER experiment: model={mod}, fw={window_forward.value}, seed={seed}")

                        try:
                            cf: Configuration = Configuration(now)
                            cf.SEED = seed

                            set_seeds(cf)

                            cf.CHOSEN_DATASET = dataset
                            if mod == cst.Models.METALOB:
                                cf.CHOSEN_DATASET = cst.DatasetFamily.META
                                cf.TARGET_DATASET_META_MODEL = target_dataset_meta
                                cf.JSON_DIRECTORY = json_dir

                            cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.ALL
                            cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.ALL
                            cf.CHOSEN_PERIOD = peri

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
    cst.Servers.ALIEN2: [
        {
            'seeds': [500],
            'forward_windows': 'all',
            'models': cst.MODELS_15,
        },
    ],
    # cst.Servers.ALIEN1: [
    #     {
    #         'seeds': [500],
    #         'forward_windows': [cst.WinSize.EVENTS5],
    #         'models': [cst.Models.DLA, cst.Models.TLONBoF, cst.Models.CTABL, cst.Models.DEEPLOBATT],
    #     },
    # ],
    # cst.Servers.ALIEN2: [
    #     {
    #         'seeds': [500],
    #         'forward_windows': [cst.WinSize.EVENTS5],
    #         'models': [cst.Models.BINCTABL, cst.Models.LSTM, cst.Models.CNN2, cst.Models.DEEPLOB],
    #     },
    # ],
}

now = 'FEB-META'
jsons_dir = "final_data/LOBSTER-FEB-TESTS/jsons/"
target_dataset_meta = cst.DatasetFamily.LOBSTER

experiment_lobster(
    execution_plan,
    dataset=cst.DatasetFamily.LOBSTER,
    now=now,
    servers=servers,
    is_debug=False,
    json_dir=jsons_dir,
    target_dataset_meta=target_dataset_meta,
    peri=cst.Periods.FEBRUARY2022
)
