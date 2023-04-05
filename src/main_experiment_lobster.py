import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *
from src.utils.utilities import get_upper_diagonal_windows

DEFAULT_SEEDS = set(range(500, 505))
DEFAULT_HORIZONS = set(cst.FI_Horizons)


def experiment_lobster(models_todo, dataset, now=None, servers=None, is_debug=False):

    now, server_name, server_id, n_servers = experiment_preamble(now, servers)
    lunches_server = models_todo[server_name]

    for mod, plan in lunches_server:
        seeds = plan['seed']
        seeds = DEFAULT_SEEDS if seeds == 'all' else seeds

        for see in seeds:
            horizons_wf = plan['k+']
            horizons_wb = plan['k-']

            for kp, km in zip(horizons_wb, horizons_wf):

                    print(f"Running LOBSTER experiment: model={mod}, bw={km}, fw={kp}, seed={see}")

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

                        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = kp.value
                        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = km.value

                        cf.CHOSEN_MODEL = mod

                        cf.IS_WANDB = 1 if not is_debug else 0
                        cf.IS_TUNE_H_PARAMS = 0

                        launch_wandb(cf)

                    except KeyboardInterrupt:
                        print("There was a problem running on", server_name.name, "LOBSTER experiment on {}, with K-={}, K+={}".format(mod, km, kp))
                        sys.exit()


servers = [cst.Servers.ALIEN1, cst.Servers.ALIEN2, cst.Servers.FISSO1]

backwards, forwards = get_upper_diagonal_windows(windows=[cst.WinSize.SEC10, cst.WinSize.SEC50, cst.WinSize.SEC100])

models_todo = {
    cst.Servers.FISSO1: [
        (cst.Models.DLA, {
            'seed': [500],
            'k-': backwards,
            'k+': forwards,
        }),
        (cst.Models.CTABL, {
            'seed': [500],
            'k-': backwards,
            'k+': forwards,
        }),
        (cst.Models.BINCTABL, {
            'seed': [500],
            'k-': backwards,
            'k+': forwards,
        }),
        (cst.Models.LSTM, {
            'seed': [500],
            'k-': backwards,
            'k+': forwards,
        }),
        (cst.Models.CNNLSTM, {
            'seed': [500],
            'k-': backwards,
            'k+': forwards,
        }),
    ],
    cst.Servers.ALIEN1: [
        (cst.Models.DAIN, {
            'seed': [500],
            'k-': backwards,
            'k+': forwards,
        }),
        (cst.Models.MLP, {
            'seed': [500],
            'k-': backwards,
            'k+': forwards,
        }),
        (cst.Models.CNN2, {
            'seed': [500],
            'k-': backwards,
            'k+': forwards,
        }),
    ],
    cst.Servers.ALIEN2: [
        (cst.Models.CNN1, {
            'seed': [500],
            'k-': backwards,
            'k+': forwards,
        }),
        (cst.Models.TLONBoF, {
            'seed': [500],
            'k-': backwards,
            'k+': forwards,
        }),
        (cst.Models.DEEPLOB, {
            'seed': [500],
            'k-': backwards,
            'k+': forwards,
        }),
    ]
}

now = "LOBSTER-05-04-2023"
experiment_lobster(models_todo, dataset=cst.DatasetFamily.LOBSTER, now=now, servers=servers, is_debug=False)
