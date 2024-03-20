import wandb
import src.constants as cst
import itertools

def wandb_init(sim):
    def wandb_lunch(sim):  # runs multiple instances
        with wandb.init() as wandb_instance:
            sim.update_hyper_parameters(wandb_instance.config)
            sim.end_setup(wandb_instance)

            wandb_instance.log({k: str(v) for k, v in sim.SETTINGS.__dict__.items()})
            sim.run()

    sweep_id = wandb.sweep(project=cst.PROJECT_NAME_VERSION, sweep={
        'method': sim.SETTINGS.WANDB_SWEEP_METHOD,
        "metric": {"goal": "maximize", "name": cst.VALIDATION_METRIC},
        'parameters': sim.TUNABLE_H_PRAM.__dict__,
        'description': str(sim.SETTINGS) + str(sim.TUNABLE_H_PRAM),
    })
    return sweep_id, wandb_lunch


def grid_search_configurations(tunable_variables, n_steps=3):
    """ Given a set of parameters to tune of the form

    { p1: {"values": [v1, v2, v3]},
      p2: {"max": 1, "min": 0}, ... }

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


class ExecutionPlan:
    def __init__(self, plan, constraints):
        self.plan = plan
        self.constraints = constraints

    def configurations(self):
        """
        Generate configurations based on the execution plan and constraints.
        Returns: list: A list of dictionaries representing configurations for LOBCAST Settings,
                       where keys are variable names and values are the corresponding values.
        """
        all_domains = [list(dom) for dom in self.plan.values()]
        configurations_attempts = list(itertools.product(*all_domains))

        chosen_configurations = set()
        for fixed_var, fixed_value in self.constraints.items():
            for configuration in configurations_attempts:
                vf_index = list(self.plan.keys()).index(fixed_var)
                if configuration[vf_index] == fixed_value:
                    chosen_configurations |= {configuration}

        out_con = []
        for co_tup in chosen_configurations:
            co_dic = {k.value: co_tup[i] for i, k in enumerate(self.plan.keys())}
            out_con.append(co_dic)
        return out_con
