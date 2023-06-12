
import src.utils.utils_training_loop as tlu
import src.constants as cst
import sys

from src.config import Configuration

DEFAULT_SEEDS = set(range(500, 505))
DEFAULT_HORIZONS = set(cst.FI_Horizons)


def experiment_fi(execution_plan, run_name_prefix="FI-EXPERIMENTS", servers=None):
    """ Sets the experiment configuration object based on the execution plan and runs the simulation. """

    servers = [server for server in execution_plan.keys()] if servers is None else servers

    run_name_prefix, server_name, _, _ = tlu.experiment_preamble(run_name_prefix, servers)
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
                    # creates the configuration object to be used thought all the simulation
                    cf: Configuration = Configuration(run_name_prefix)
                    cf.SEED = see

                    tlu.set_seeds(cf)

                    if mod == cst.Models.METALOB:
                        cf.CHOSEN_DATASET = cst.DatasetFamily.META
                    else:
                        cf.CHOSEN_DATASET = cst.DatasetFamily.FI

                    cf.CHOSEN_MODEL = mod
                    cf.IS_WANDB = True
                    cf.IS_TUNE_H_PARAMS = True

                    tlu.run(cf)

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
            (cst.Models.MLP,      {'k': [cst.FI_Horizons.K5], 'seed': [500]}),
            (cst.Models.BINCTABL, {'k': [cst.FI_Horizons.K5], 'seed': 'all'})
        ],
    }

    experiment_fi(EXE_PLAN)
