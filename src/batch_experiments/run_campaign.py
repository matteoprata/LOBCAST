

import wandb
from src.lobcast import LOBCAST
from src.utils.ultils_run import grid_search_configurations, ExecutionPlan, wandb_init
from src.settings import SettingsExp
import src.constants as cst

from src.batch_experiments import setup01
from src.run import run_simulation


def main():
    sim = LOBCAST()

    # for multiple experiments
    ep = ExecutionPlan(setup01.INDEPENDENT_VARIABLES,
                       setup01.INDEPENDENT_VARIABLES_CONSTRAINTS)

    setting_confs = ep.configurations()

    print("Running the following configurations:")
    print(setting_confs)

    for setting_conf in setting_confs:
        sim.update_settings(setting_conf)
        run_simulation(sim)
        print("done:", setting_conf)


if __name__ == '__main__':
    main()


# python -m src.run
