
import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main import *


def experiment_FI():

    mac = get_sys_mac()
    if mac in cst.ServerMACIDs.keys():
        server = cst.ServerMACIDs[mac]
        print("Running on server", server.name)
    else:
        print("This SERVER is not handled for the experiment.")
        exit()

    for mod in list(cst.Models)[server.value::len(cst.ServersMAC)]:
        for k in cst.FI_Horizons:
            print("Running FI experiment on {}, with K={}".format(mod, k))
            try:
                cf: Configuration = Configuration()
                set_seeds(cf)

                cf.CHOSEN_DATASET = cst.DatasetFamily.FI
                cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.FI
                cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.FI
                cf.CHOSEN_MODEL = mod
                cf.IS_WANDB = 1
                cf.IS_TUNE_H_PARAMS = False

                cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON] = k.value
                launch_wandb(cf)
                # launch_single(cf)
            except KeyboardInterrupt:
                print("There was a problem running on", server.name, "FI experiment on {}, with K={}".format(mod, k))
                sys.exit()


experiment_FI()
