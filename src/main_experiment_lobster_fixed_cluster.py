
import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *

DEFAULT_SEEDS = set(range(500, 505))


def experiment_lobster(model, backward_window, forward_window, seed, now=None):

    print(f"Running LOBSTER experiment: model={model}, bw={backward_window}, fw={forward_window}, seed={seed}")

    try:
        cf: Configuration = Configuration(now)
        cf.SEED = seed

        set_seeds(cf)

        cf.CHOSEN_DATASET = cst.DatasetFamily.LOBSTER
        cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.ALL
        cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.ALL
        cf.CHOSEN_PERIOD = cst.Periods.JULY2021

        cf.CHOSEN_MODEL = model

        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = backward_window.value
        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = forward_window.value

        cf.IS_WANDB = 1
        cf.IS_TUNE_H_PARAMS = False

        launch_wandb(cf)

    except KeyboardInterrupt:
        print("There was a problem running on cluster LOBSTER experiment on {}, with K-={}, K+={}".format(
            model,
            backward_window,
            forward_window
        ))
        sys.exit()


now = "LOBSTER-31-03-2023"
wandb.login(key="54775690baa838985ad1ce959fd2d5dcc8b23b8b")
experiment_lobster(
    model=cst.Models[sys.argv[1]],
    backward_window=cst.WinSize[sys.argv[2]],
    forward_window=cst.WinSize[sys.argv[3]],
    seed=int(sys.argv[4]),
    now=now,
)
