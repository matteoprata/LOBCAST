

import src.constants as cst
import wandb
from src.lobcast import LOBCAST
from src.utils.ultils_run import grid_search_configurations, wandb_init
from src.settings import SettingsExp


def run_simulation(sim):
    if not sim.SETTINGS.IS_WANDB:
        # generates runs based on a grid search of the hyper prams
        hparams_configs = grid_search_configurations(sim.TUNABLE_H_PRAM.__dict__)
        for hparams_config in hparams_configs:
            sim.update_hyper_parameters(hparams_config)
            sim.end_setup()
            sim.run()
    else:
        # hyper params search is handled by wandb
        sweep_id, wandb_lunch = wandb_init(sim)
        wandb.agent(sweep_id, function=lambda: wandb_lunch(sim))


def main():
    sim = LOBCAST()

    setting_conf = sim.parse_cl_arguments()
    sim.update_settings(setting_conf)
    run_simulation(sim)


if __name__ == '__main__':
    main()


# python -m src.run --PREDICTION_MODEL MLP
