
import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *


def experiment_FI(model, k, now=None):

    print("Running FI experiment on {}, with K={}".format(model.name, k))

    try:
        cf: Configuration = Configuration(now)
        set_seeds(cf)

        cf.CHOSEN_DATASET = cst.DatasetFamily.FI
        cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.FI
        cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.FI
        cf.CHOSEN_PERIOD = cst.Periods.FI
        cf.CHOSEN_MODEL = model

        cf.IS_WANDB = 1
        cf.IS_TUNE_H_PARAMS = True

        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON] = k.value
        launch_wandb(cf)

    except KeyboardInterrupt:
        print("There was a problem running on cluster FI experiment on {}, with K={}".format(model.name, k))
        sys.exit()


now = "FI-CI-SIAMO-FRA"
wandb.login(key="54775690baa838985ad1ce959fd2d5dcc8b23b8b")
experiment_FI(model=cst.Models[sys.argv[1]], k=cst.FI_Horizons[sys.argv[2]], now=now)
