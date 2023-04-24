
import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *

DEFAULT_SEEDS = set(range(500, 505))

def experiment_lobster(model, forward_window, seed, now=None):

    print(f"Running LOBSTER experiment: model={model}, fw={forward_window}, seed={seed}")

    try:
        cf: Configuration = Configuration(now)
        cf.SEED = seed

        set_seeds(cf)

        cf.CHOSEN_DATASET = cst.DatasetFamily.LOBSTER
        cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.ALL
        cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.ALL
        cf.CHOSEN_PERIOD = cst.Periods.JULY2021

        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = cst.WinSize.EVENTS1.value
        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = forward_window.value

        cf.CHOSEN_MODEL = model

        cf.IS_WANDB = True
        cf.IS_TUNE_H_PARAMS = True

        launch_wandb(cf)

    except KeyboardInterrupt:
        print("There was a problem running on cluster LOBSTER experiment on {} with K+={}".format(
            model,
            forward_window
        ))
        sys.exit()


now = 'LOBSTER-DEFINITIVE-EVENTS-2023-04-20'
wandb.login(key="54775690baa838985ad1ce959fd2d5dcc8b23b8b")
experiment_lobster(
    model=cst.Models[sys.argv[1]],
    forward_window=cst.WinSize[sys.argv[2]],
    seed=int(sys.argv[3]),
    now=now,
)