
import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *


def experiment_FI(models_todo, now=None, servers=None, is_debug=False):

    now, server_name, server_id, n_servers = experiment_preamble(now, servers)
    lunches_server = models_todo[server_name]

    for see in lunches_server['seed']:
        for mod in lunches_server['mod']:
            for k in lunches_server['k']:
                print("Running FI experiment on {}, with K={}".format(mod, k))

                try:
                    cf: Configuration = Configuration(now)
                    cf.SEED = see

                    set_seeds(cf)

                    if mod == cst.Models.METALOB:
                        cf.CHOSEN_DATASET = cst.DatasetFamily.META
                    else:
                        cf.CHOSEN_DATASET = cst.DatasetFamily.FI

                    cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.FI
                    cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.FI
                    cf.CHOSEN_PERIOD = cst.Periods.FI
                    cf.CHOSEN_MODEL = mod

                    cf.IS_WANDB = 1 if not is_debug else 0
                    cf.IS_TUNE_H_PARAMS = not is_debug

                    cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON] = k.value
                    launch_wandb(cf)

                except KeyboardInterrupt:
                    print("There was a problem running on", server_name.name, "FI experiment on {}, with K={}".format(mod, k))
                    sys.exit()


servers = [cst.Servers.ALIEN1, cst.Servers.ALIEN2, cst.Servers.FISSO1]

models_todo = {
    cst.Servers.ALIEN1: {'mod': [cst.Models.TRANSLOB, cst.Models.ATNBoF], 'k': [cst.FI_Horizons.K1, cst.FI_Horizons.K2, cst.FI_Horizons.K3], 'seed': list(range(110, 111))},  # without weights
    cst.Servers.ALIEN2: {'mod': [cst.Models.TRANSLOB, cst.Models.ATNBoF], 'k': [cst.FI_Horizons.K5, cst.FI_Horizons.K10], 'seed': list(range(110, 111))},  # with weights
    cst.Servers.FISSO1: {'mod': [cst.Models.TLONBoF], 'k': cst.FI_Horizons, 'seed': list(range(110, 111))},  # with weights
}


now = "FI-HARD-ONES"
experiment_FI(models_todo, now=now, servers=servers, is_debug=False)

