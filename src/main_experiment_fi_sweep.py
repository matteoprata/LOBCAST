
import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *
from src.utils.utilities import get_sys_mac


def experiment_FI(sweep_time=None, models_todo=None):

    if sweep_time is None:
        parser = argparse.ArgumentParser(description='Stock Price Trend Prediction Sweep FI:')
        parser.add_argument('-now', '--now', default=None)
        args = vars(parser.parse_args())

    mac = get_sys_mac()

    if mac in cst.ServerMACIDs:
        server = cst.ServerMACIDs[mac]
        print("Running on server", server.name, mac)
    else:
        print("This SERVER is not handled for the experiment ({})".format(mac))
        exit()

    models_todo = models_todo if models_todo is not None else cst.Models
    for mod in list(models_todo)[server.value::len(cst.ServersMAC)]:
        k = cst.FI_Horizons.K3
        print("Running FI experiment on {}, with K={}".format(mod, k))

        try:
            now = args["now"] if sweep_time is None else sweep_time
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
            print("There was a problem running on", server.name, "FI experiment on {}, with K={}".format(mod, k))
            sys.exit()


now = "DEBUG-SWEEP"
models_todo = [cst.Models.ATNBoF]
experiment_FI(now, models_todo)
