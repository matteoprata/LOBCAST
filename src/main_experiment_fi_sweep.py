
import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *


def experiment_FI(models_todo, kset=None, now=None, servers=None):

    now, server_name, server_id, n_servers = experiment_preamble(now, servers)
    kset = kset if kset is not None else cst.FI_Horizons

    for mod in models_todo[server_name]:
        for k in kset:
            print("Running FI experiment on {}, with K={}".format(mod, k))

            try:
                cf: Configuration = Configuration(now)
                set_seeds(cf)

                cf.CHOSEN_DATASET = cst.DatasetFamily.FI
                cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.FI
                cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.FI
                cf.CHOSEN_PERIOD = cst.Periods.FI
                cf.CHOSEN_MODEL = mod

                cf.IS_WANDB = 1
                cf.IS_TUNE_H_PARAMS = True

                cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON] = k.value
                launch_wandb(cf)

            except KeyboardInterrupt:
                print("There was a problem running on", server_name.name, "FI experiment on {}, with K={}".format(mod, k))
                sys.exit()


servers = [cst.Servers.ALIEN1, cst.Servers.ALIEN2, cst.Servers.FISSO1]

models_todo = {cst.Servers.ALIEN1: [cst.Models.CNNLSTM, cst.Models.CNN1, cst.Models.AXIALLOB],
               cst.Servers.ALIEN2: [cst.Models.MLP],  # [cst.Models.DEEPLOB, cst.Models.DAIN, cst.Models.DEEPLOBATT, cst.Models.ATNBoF],  # DLA, MLP, CNN2
               cst.Servers.FISSO1: [cst.Models.BINCTABL, cst.Models.TLONBoF, cst.Models.LSTM, cst.Models.CTABL, cst.Models.TRANSLOB]}

kset = [cst.FI_Horizons.K10]

now = "FI-2010-SWEEP-ALL-FINAL-190223"
experiment_FI(models_todo, kset=kset, now=now, servers=servers)
