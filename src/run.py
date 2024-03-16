
from pytorch_lightning import Trainer

import src.constants as cst
import wandb
from src.LOBCAST import LOBCAST
from src.models.model_callbacks import callback_save_model
from src.utils.utils_dataset import pick_dataset
from src.utils.utils_models import pick_model
from src.utils.ultils_run import grid_search_configurations


def run_single(sim):
    """ Given a simulation, settings and hyper params, it runs the training loop. """
    data_module = pick_dataset(sim)
    nets_module = pick_model(sim, data_module, sim.METRICS)

    trainer = Trainer(
        accelerator=sim.SETTINGS.DEVICE,
        devices=sim.SETTINGS.N_GPUs,
        check_val_every_n_epoch=sim.SETTINGS.VALIDATION_EVERY,
        max_epochs=sim.SETTINGS.EPOCHS_UB,
        callbacks=[
            callback_save_model(sim.SETTINGS.DIR_EXPERIMENTS, cst.VALIDATION_METRIC, top_k=3)
        ],
    )

    model_path = sim.SETTINGS.TEST_MODEL_PATH if sim.SETTINGS.IS_TEST_ONLY else "best"

    if not sim.SETTINGS.IS_TEST_ONLY:
        trainer.fit(nets_module, data_module)
        sim.METRICS.dump_metrics(cst.METRICS_RUNNING_FILE_NAME)
        sim.METRICS.reset_stats()

        trainer.validate(nets_module, data_module, ckpt_path=model_path)

    trainer.test(nets_module, data_module, ckpt_path=model_path)
    sim.METRICS.dump_metrics("metrics_best.json")
    print('Completed.')


def wandb_init(sim):
    def wandb_lunch(sim):  # runs multiple instances
        with wandb.init() as wandb_instance:
            sim.update_hyper_parameters(wandb_instance.config)
            sim.end_setup(wandb_instance)

            wandb_instance.log({k: str(v) for k, v in sim.SETTINGS.__dict__.items()})
            run_single(sim)

    sweep_id = wandb.sweep(project=cst.PROJECT_NAME_VERSION, sweep={
        'method': sim.SETTINGS.SWEEP_METHOD,
        "metric": {"goal": "maximize", "name": cst.VALIDATION_METRIC},
        'parameters': sim.TUNABLE_H_PRAM.__dict__,
        'description': str(sim.SETTINGS) + str(sim.TUNABLE_H_PRAM),
    })
    return sweep_id, wandb_lunch


def main():
    sim = LOBCAST()
    sim.update_settings()

    if not sim.SETTINGS.IS_WANDB:
        # generates runs based on a grid search of the hyper prams
        runs = grid_search_configurations(sim.TUNABLE_H_PRAM.__dict__)
        for config in runs:
            sim.update_hyper_parameters(config)
            sim.end_setup()
            run_single(sim)
    else:
        # hyper params search is handled by wandb
        sweep_id, wandb_lunch = wandb_init(sim)
        wandb.agent(sweep_id, function=lambda: wandb_lunch(sim))


if __name__ == '__main__':
    main()


# TODO allow to set settings at runtime
# MODEL
# SEED
# PREDICTION_HORIZON_FUTURE
# PREDICTION_HORIZON_PAST
# OBSERVATION_PERIOD


# python -m src.run --PREDICTION_MODEL MLP
