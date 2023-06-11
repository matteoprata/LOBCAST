import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *

DEFAULT_SEEDS = set(range(500, 505))
DEFAULT_HORIZONS = set(cst.FI_Horizons)


def experiment_fi(execution_plan, run_name_prefix=None, servers=None, is_debug=False):
    """ Sets the experiment configuration based on the execution plan and runs the simulation. """

    run_name_prefix, server_name, _, _ = experiment_preamble(run_name_prefix, servers)
    lunches_server = execution_plan[server_name]  # the execution plan for this machine

    # iterates over the models and the plans assigned to this machine (i.e, seeds and horizons)
    for mod, plan in lunches_server:
        seeds = plan['seed']
        seeds = DEFAULT_SEEDS if seeds == 'all' else seeds

        for see in seeds:
            horizons = plan['k']
            horizons = DEFAULT_HORIZONS if horizons == 'all' else horizons

            for k in horizons:
                print("Running FI experiment on {}, with K={}".format(mod, k))

                try:
                    cf: Configuration = Configuration(run_name_prefix)
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


if __name__ == '__main__':

    # EXE_PLAN is an execution plan. It is a list of all the desired sequential lunches to do, having the format:
    # KEY: server name in src.constants.Servers enum, VALUE: list of tuples
    # s.t. (tuple[0] = model name in src.constants.FI_Horizons enum,
    #       tuple[1] = dictionary s.t. {'k':    'all' or list of src.constants.FI_Horizons,
    #                                   'seed': 'all' or list of integer representing the random seed })

    EXE_PLAN = {
        cst.Servers.ANY: [
            (cst.Models.MLP, {'k': [cst.FI_Horizons.K5],
                              'seed': [500]}),

            (cst.Models.BINCTABL, {'k': [cst.FI_Horizons.K5],
                                   'seed': 'all'})
        ],
    }

    PREFIX = "FI-DEFINITIVE-23-03-23"
    SERVERS = [server for server in EXE_PLAN.keys()]

    experiment_fi(EXE_PLAN, run_name_prefix=PREFIX, servers=SERVERS, is_debug=False)
