
import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *
from src.utils.utilities import get_sys_mac


def experiment_FI(now=None, models=None, servers=None, horizons=None, seeds=None):

    now, server_name, server_id, n_servers = experiment_preamble(now, servers)

    models = list(cst.Models)[server_id::len(servers)] if models is None else models
    horizons = cst.FI_Horizons if horizons is None else horizons
    seeds = [1] if seeds is None else seeds

    for mod in models:
        for k in horizons:
            for s in seeds:
                print("Running FI experiment on {}, with K={}".format(mod, k))

                try:
                    cf: Configuration = Configuration(now)
                    cf.SEED = s
                    set_seeds(cf)

                    cf.CHOSEN_DATASET = cst.DatasetFamily.FI
                    cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.FI
                    cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.FI
                    cf.CHOSEN_PERIOD = cst.Periods.FI
                    cf.CHOSEN_MODEL = mod

                    cf.IS_WANDB = 1
                    cf.IS_TUNE_H_PARAMS = False

                    cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON] = k.value
                    launch_wandb(cf)

                except KeyboardInterrupt:
                    print("There was a problem running on", server_name.name, "FI experiment on {}, with K={}".format(mod, k))
                    sys.exit()


seeds = range(50, 56)
servers = [cst.Servers.ALIEN2]
horizons = [cst.FI_Horizons.K1]
models = [cst.Models.ATNBoF]
now = "deb"  # "02-02-2023-FI-FINAL"
experiment_FI(now=now, servers=servers, horizons=horizons, models=models, seeds=None)


