import os
import sys

import src.utils.utils_training_loop as tlu
from src.config import Configuration
import src.constants as cst


DEFAULT_SEEDS = set(range(500, 505))

DEFAULT_FORWARD_WINDOWS = [
    cst.WinSize.EVENTS1,
    cst.WinSize.EVENTS2,
    cst.WinSize.EVENTS3,
    cst.WinSize.EVENTS5,
    cst.WinSize.EVENTS10
]


def experiment_lobster(execution_plan, dataset, PREFIX=None, is_debug=False, json_dir=None, target_dataset_meta=None, peri=None):

    servers = [server for server in execution_plan.keys()]
    PREFIX, server_name, server_id, n_servers = tlu.experiment_preamble(PREFIX, servers)
    lunches_server = execution_plan[server_name]

    for mod, plan in lunches_server:
        seeds = plan['seed']
        seeds = DEFAULT_SEEDS if seeds == 'all' else seeds

        for see in seeds:
            forward_windows = plan['forward_windows']
            forward_windows = DEFAULT_FORWARD_WINDOWS if forward_windows == 'all' else forward_windows

            for window_forward in forward_windows:

                    print(f"Running LOB experiment: model={mod}, fw={window_forward.value}, seed={see}")

                    try:
                        cf: Configuration = Configuration(PREFIX)
                        cf.SEED = see

                        tlu.set_seeds(cf)

                        cf.DATASET_NAME = dataset
                        if mod == cst.Models.METALOB:
                            cf.DATASET_NAME = cst.DatasetFamily.META
                            cf.TARGET_DATASET_META_MODEL = target_dataset_meta
                            cf.JSON_DIRECTORY = json_dir

                        cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.ALL
                        cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.ALL
                        cf.CHOSEN_PERIOD = peri

                        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = cst.WinSize.EVENTS1.value
                        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = window_forward.value

                        cf.PREDICTION_MODEL = mod

                        cf.IS_WANDB = int(not is_debug)
                        cf.IS_HPARAM_SEARCH = int(not is_debug)

                        tlu.run(cf)

                    except KeyboardInterrupt:
                        print("There was a problem running on", server_name.name, "LOB experiment on {}, with K+={}".format(mod, window_forward))
                        sys.exit()


if __name__ == '__main__':

    EXE_PLAN = {
        cst.Servers.ANY: [
            (cst.Models.MLP, {'forward_windows': [cst.WinSize.EVENTS5], 'seed': [500]})
        ]
    }

    experiment_lobster(
        EXE_PLAN,
        dataset=cst.DatasetFamily.LOB,
        PREFIX='LOBSTER-EXPERIMENT',
        is_debug=False,
        json_dir="final_data/LOB-FEB-TESTS/jsons/",
        target_dataset_meta=cst.DatasetFamily.LOB,
        peri=cst.Periods.JULY2021
    )

