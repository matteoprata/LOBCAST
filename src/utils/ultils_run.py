import wandb
import src.constants as cst
import itertools


def grid_search_configurations(tunable_variables, n_steps=3):
    """ Given a set of parameters to tune of the form
    {
        p1: {"values": [v1, v2, v3]},
        p2: {"max": 1, "min": 0}, ...
    }
    returns the configurations associated with a grid search in the form:
    [ {p1:v1, p2:v1}, {p1:v1, v2}, ... ]
    """
    all_domains = []
    for name, domain in tunable_variables.items():
        # continuous variable
        if 'min' in domain:
            step = (domain['max'] - domain['min']) / n_steps
            all_domains += [[domain['min'] + step * i for i in range(n_steps)]]
            print(f"Warning! Param {name} domain {domain} was discretized! In {n_steps} steps as {all_domains}.")

        # discrete variable
        elif 'values' in domain:
            all_domains += [domain['values']]
    configurations_tuples = itertools.product(*all_domains)

    # from tuples [(v1, v2, v3)] to [{p1: v1}, ...]
    configurations_dicts = [{k: v for k, v in zip(tunable_variables.keys(), selected_values)} for selected_values in configurations_tuples]
    return configurations_dicts
