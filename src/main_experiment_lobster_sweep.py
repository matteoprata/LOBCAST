
import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *


def experiment_lobster(models_todo, bwin, now=None, servers=None):

    now, server_name, server_id, n_servers = experiment_preamble(now, servers)
    for mod in models_todo[server_name]:
        for bk in bwin:
            for fk in bwin[bk]:
                print("Running LOBSTER experiment on {}, with BK={}, FK={}".format(mod, bk, fk))

                try:
                    cf: Configuration = Configuration(now)
                    set_seeds(cf)

                    cf.CHOSEN_DATASET = cst.DatasetFamily.LOBSTER
                    cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.ALL
                    cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.ALL

                    cf.HYPER_PARAMETERS[cst.LearningHyperParameter.LABELING_SIGMA_SCALER] = .4
                    cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = bk
                    cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = fk

                    cf.CHOSEN_PERIOD = cst.Periods.JULY2021
                    cf.CHOSEN_MODEL = mod

                    cf.IS_WANDB = 1
                    cf.IS_TUNE_H_PARAMS = True

                    launch_wandb(cf)

                except KeyboardInterrupt:
                    print("There was a problem running on", server_name.name, "LOBSTER experiment on {}, with BK={}, FK={}".format(mod, bk, fk))
                    print(traceback.print_exc(), file=sys.stderr)
                    sys.exit()


cst.WANDB_SWEEP_MAX_RUNS = 3
servers = [cst.Servers.ALIEN1, cst.Servers.ALIEN2, cst.Servers.FISSO1]

models_todo = {
    cst.Servers.ALIEN1: [cst.Models.MLP],
    cst.Servers.ALIEN2: [cst.Models.BINCTABL],
}

bwin = {
    cst.WinSize.SEC10.value:  [cst.WinSize.SEC10.value],
    cst.WinSize.SEC50.value:  [cst.WinSize.SEC50.value],
    cst.WinSize.SEC100.value: [cst.WinSize.SEC100.value]
}

now = "LOBSTER-SWEEP-ALL-FINAL"
experiment_lobster(models_todo, bwin, now=now, servers=servers)
