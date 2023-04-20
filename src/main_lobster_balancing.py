import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *
from src.utils.utilities import get_upper_diagonal_windows

DEFAULT_SEEDS = set(range(500, 505))
DEFAULT_MODELS = cst.MODELS_15

from src.utils.utilities import write_data

import numpy as np
from scipy.optimize import differential_evolution


servers = [cst.Servers.ALIEN1, cst.Servers.ALIEN2, cst.Servers.FISSO1]

# backwards, forwards = get_upper_diagonal_windows(windows=[cst.WinSize.SEC10, cst.WinSize.SEC50, cst.WinSize.SEC100])
backwards = [cst.WinSize.SEC100, cst.WinSize.SEC100, cst.WinSize.SEC100, cst.WinSize.SEC50, cst.WinSize.SEC50, cst.WinSize.SEC10]
forwards  = [cst.WinSize.SEC100, cst.WinSize.SEC50,  cst.WinSize.SEC10,  cst.WinSize.SEC50, cst.WinSize.SEC10, cst.WinSize.SEC10]

execution_plan = {

    cst.Servers.FISSO1: [
        {
            'seeds': [500],
            'models': [cst.Models.MLP],
            'k-': backwards,
            'k+': forwards,
        }
    ],

    cst.Servers.ALIEN1: [
        {
            'seeds': [500, 502],
            'models': [cst.Models.CNN1, cst.Models.LSTM, cst.Models.DAIN, cst.Models.DEEPLOB],
            'k-': backwards,
            'k+': forwards,
        },
    ],

    cst.Servers.ALIEN2: [
        {
            'seeds': [500, 502],
            'models': [cst.Models.CNNLSTM, cst.Models.CNN2, cst.Models.TLONBoF, cst.Models.DEEPLOBATT],
            'k-': backwards,
            'k+': forwards,
        },
    ],
}

now = 'LOBSTER-DEFINITIVE-14-04-2023'

def experiment_lobster(alfa):

    cst.ALFA = alfa

    now, server_name, server_id, n_servers = experiment_preamble('okgrazieciao', servers)
    lunches_server = execution_plan[server_name]


    for instance in lunches_server:
        seeds = instance['seeds']
        seeds = DEFAULT_SEEDS if seeds == 'all' else seeds

        for see in seeds:
            models = instance['models']
            models = DEFAULT_MODELS if models == 'all' else models

            for mod in models:

                l = list()
                for window_backward, window_forward in zip(instance['k-'], instance['k+']):

                        print(f"Running LOBSTER experiment: model={mod}, bw={window_backward}, fw={window_forward}, seed={see}")

                        try:
                            cf: Configuration = Configuration(now)
                            cf.SEED = see

                            set_seeds(cf)

                            cf.CHOSEN_DATASET = cst.DatasetFamily.LOBSTER
                            if mod == cst.Models.METALOB:
                                cf.CHOSEN_DATASET = cst.DatasetFamily.META

                            cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.ALL
                            cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.ALL
                            cf.CHOSEN_PERIOD = cst.Periods.JULY2021

                            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = window_backward.value
                            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = window_forward.value

                            cf.CHOSEN_MODEL = mod

                            cf.IS_WANDB = int(not True)
                            cf.IS_TUNE_H_PARAMS = False

                            model_params = HP_DICT_MODEL[cf.CHOSEN_MODEL].fixed_lob

                            print("Setting model parameters", model_params)

                            for param in cst.LearningHyperParameter:
                                if param.value in model_params:
                                    cf.HYPER_PARAMETERS[param] = model_params[param.value]

                            cf.dynamic_config_setup()
                            data_module = pick_dataset(cf)

                            ys_occurrences_train = data_module.train_set.ys_occurrences
                            ys_occurrences_val = data_module.val_set.ys_occurrences
                            ys_occurrences_test = data_module.test_set.ys_occurrences

                            ys_occurrences_train = np.asarray([ys_occurrences_train[0.0], ys_occurrences_train[1.0], ys_occurrences_train[2.0]])
                            ys_occurrences_val = np.asarray([ys_occurrences_val[0.0], ys_occurrences_val[1.0], ys_occurrences_val[2.0]])
                            ys_occurrences_test = np.asarray([ys_occurrences_test[0.0], ys_occurrences_test[1.0], ys_occurrences_test[2.0]])

                            ys_occurrences_train = ys_occurrences_train / np.sum(ys_occurrences_train)
                            ys_occurrences_val = ys_occurrences_val / np.sum(ys_occurrences_val)
                            ys_occurrences_test = ys_occurrences_test / np.sum(ys_occurrences_test)

                            ys_occurrences = np.asarray([ys_occurrences_train, ys_occurrences_val, ys_occurrences_test])
                            ys_occurrences = np.round(ys_occurrences, 2)

                            l.append(ys_occurrences)

                        except KeyboardInterrupt:
                            print("There was a problem running on", server_name.name, "LOBSTER experiment on {}, with K-={}, K+={}".format(mod, window_backward, window_forward))
                            sys.exit()

    l = np.asarray(l)

    # for back, forw, arr in zip(backwards, forwards, l):
    #     print(f'Backward:{back}\tForward:{forw}')

    #     train, val, test = arr
    #     print(f'{train[0]}\t{train[1]}\t{train[2]}')
    #     print(f'{val[0]}\t{val[1]}\t{val[2]}')
    #     print(f'{test[0]}\t{test[1]}\t{test[2]}')

    #     print()
    # print()

    # print("Train\tVal\tTest")
    # print(l.min(axis=2))

    print()
    m = l.min(axis=(2, 0))
    print(m)

    return m



def optimize_balancing():
    def target_function(x):
        val = experiment_lobster(x[0])
        err = np.sum(np.square(val - np.array([0.33, 0.33, 0.33])))  # minimize the squared distance
        print(x[0], '\t', val, '\t', err)
        return err

    result = differential_evolution(target_function, [(0.00001, .0001)], workers=1, tol=.00001)

    if result.success:
        print(f"Optimal x: {result.x[0]}, f(x): {experiment_lobster(result.x[0])}")
    else:
        print("Optimization failed.")
    return result.x[0], experiment_lobster(result.x[0])









out = optimize_balancing()

