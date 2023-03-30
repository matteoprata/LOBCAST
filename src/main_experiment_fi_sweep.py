
import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *

DEFAULT_SEEDS = set(range(500, 505))
DEFAULT_HORIZONS = set(cst.FI_Horizons)


def experiment_FI(models_todo, now=None, servers=None, is_debug=False):

    now, server_name, server_id, n_servers = experiment_preamble(now, servers)
    lunches_server = models_todo[server_name]

    for mod, plan in lunches_server:
        seeds = plan['seed']
        seeds = DEFAULT_SEEDS if seeds == 'all' else seeds

        for see in seeds:
            horizons = plan['k']
            horizons = DEFAULT_HORIZONS if horizons == 'all' else horizons

            for k in horizons:
                print("Running FI experiment on {}, with K={}".format(mod, k))

                try:
                    cf: Configuration = Configuration(now)
                    cf.SEED = see

                    set_seeds(cf)

                    if mod == cst.Models.METALOB:
                        cf.CHOSEN_DATASET = cst.DatasetFamily.META
                    else:
                        cf.CHOSEN_DATASET = cst.DatasetFamily.FI

                    cf.CHOSEN_MODEL = mod

                    cf.IS_WANDB = 1 if not is_debug else 0
                    cf.IS_TUNE_H_PARAMS = not is_debug

                    launch_wandb(cf)

                except KeyboardInterrupt:
                    print("There was a problem running on", server_name.name, "FI experiment on {}, with K={}".format(mod, k))
                    sys.exit()


servers = [cst.Servers.ALIEN1, cst.Servers.ALIEN2, cst.Servers.FISSO1]

models_todo = {

    cst.Servers.ALIEN1: [(cst.Models.METALOB, {'k': 'all', 'seed': [500, 501, 502, 503, 504]})],
    # cst.Servers.ALIEN2: [(cst.Models.AXIALLOB, {'k': [cst.FI_Horizons.K1], 'seed': [503]})],

    # cst.Servers.ALIEN2: {'mod': [cst.Models.DLA,
    #                              cst.Models.LSTM,
    #                              cst.Models.DAIN,
    #                              cst.Models.CNNLSTM,
    #                              cst.Models.DEEPLOBATT
    
    # ], 'k': cst.FI_Horizons, 'seed': list(range(500, 510))},
    #
    # cst.Servers.ALIEN1: {'mod': [cst.Models.DEEPLOB,
    #                              cst.Models.CNN1,
    #                              cst.Models.AXIALLOB,
    #                              cst.Models.TRANSLOB,
    #                              cst.Models.ATNBoF
    # ], 'k': cst.FI_Horizons, 'seed': list(range(500, 510))},
}


now = "FI-DEFINITIVE-23-03-23"
experiment_FI(models_todo, now=now, servers=servers, is_debug=False)

