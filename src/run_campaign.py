

import src.constants as cst
import wandb
from src.lobcast import LOBCAST
from src.utils.ultils_run import grid_search_configurations, ExecutionPlan, wandb_init
from src.settings import SettingsExp


INDEPENDENT_VARIABLES = {
    SettingsExp.SEED: [0],
    SettingsExp.PREDICTION_MODEL: [cst.ModelsClass.MLP, cst.ModelsClass.CNN1],
    SettingsExp.PREDICTION_HORIZON_FUTURE: [5],
    SettingsExp.PREDICTION_HORIZON_PAST: [1],
    SettingsExp.OBSERVATION_PERIOD: [10, 100]
}

INDEPENDENT_VARIABLES_CONSTRAINTS = {
    SettingsExp.PREDICTION_MODEL: cst.ModelsClass.MLP,
    SettingsExp.PREDICTION_HORIZON_FUTURE: 10
}


def main():
    sim = LOBCAST()

    # for multiple experiments
    ep = ExecutionPlan(INDEPENDENT_VARIABLES, INDEPENDENT_VARIABLES_CONSTRAINTS)
    confs = ep.configurations()
    print("Running the following configurations:")
    print(confs)

    for configuration in confs:
        sim.update_settings(configuration)

        if not sim.SETTINGS.IS_WANDB:
            # generates runs based on a grid search of the hyper prams
            runs = grid_search_configurations(sim.TUNABLE_H_PRAM.__dict__)
            for config in runs:
                sim.update_hyper_parameters(config)
                sim.end_setup()
                sim.run()
        else:
            # hyper params search is handled by wandb
            sweep_id, wandb_lunch = wandb_init(sim)
            wandb.agent(sweep_id, function=lambda: wandb_lunch(sim))


if __name__ == '__main__':
    main()


# python -m src.run
