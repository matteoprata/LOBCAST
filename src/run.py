
from pytorch_lightning import Trainer

import src.constants as cst
import wandb
from src.config import LOBCASTSetupRun
from src.models.model_callbacks import callback_save_model
from src.utils.utils_dataset import pick_dataset
from src.utils.utils_models import pick_model


def run_single(cf):
    data_module = pick_dataset(cf)
    nets_module = pick_model(cf, data_module, cf.METRICS)

    trainer = Trainer(
        accelerator=cf.SETTINGS.DEVICE,
        devices=cf.SETTINGS.N_GPUs,
        check_val_every_n_epoch=cf.SETTINGS.VALIDATION_EVERY,
        max_epochs=cf.SETTINGS.EPOCHS_UB,
        callbacks=[
            callback_save_model(cf.SETTINGS.DIR_EXPERIMENTS, cst.VALIDATION_METRIC, top_k=3)
        ],
    )

    model_path = cf.SETTINGS.TEST_MODEL_PATH if cf.SETTINGS.IS_TEST_ONLY else "best"

    if not cf.SETTINGS.IS_TEST_ONLY:
        trainer.fit(nets_module, data_module)
        cf.METRICS.dump_metrics(cst.METRICS_RUNNING_FILE_NAME)
        cf.METRICS.reset_stats()

        trainer.validate(nets_module, data_module, ckpt_path=model_path)

    trainer.test(nets_module, data_module, ckpt_path=model_path)
    cf.METRICS.dump_metrics("metrics_best.json")


def main():
    # create a wandb sweep
    cf = LOBCASTSetupRun()

    if cf.SETTINGS.IS_WANDB:

        def wandb_lunch(cf):  # runs multiple instances
            with wandb.init() as wandb_instance:
                cf.end_setup(wandb_instance.configuration, wandb_instance)
                run_single(cf)

        sweep_id = wandb.sweep(project=cst.PROJECT_NAME_VERSION,
                               sweep={
                'method': cf.SETTINGS.SWEEP_METHOD,
                "metric": {"goal": "maximize", "name": cst.VALIDATION_METRIC},
                'parameters': cf.TUNABLE_H_PRAM.__dict__,
                'description': str(cf.SETTINGS) + str(cf.TUNABLE_H_PRAM),
            })

        # create a wandb agent
        wandb.agent(sweep_id, function=lambda: wandb_lunch(cf), count=cst.WANDB_SWEEP_MAX_RUNS)
    else:
        import itertools

        def grid_search_configurations(tunable_variable, n_steps=3):
            """ Given a set of parameters to tune in the form
            {
                p1: {"values": [v1, v2, v3]},
                p2: {"max": 1, "min": 0}, ...
            }
            returns the configurations associated with a grid search in the form:
            [ {p1:v1, p2:v1}, {p1:v1, v2}, ... ]
            """
            all_domains = []
            for param_domains in tunable_variable.values():
                # continuous variable
                if 'min' in param_domains:
                    step = (param_domains['max'] - param_domains['min']) / n_steps
                    all_domains += [[param_domains['min'] + step * i for i in range(n_steps)]]
                # discrete variable
                elif 'values' in param_domains:
                    all_domains += [param_domains['values']]
            print(all_domains)
            configurations_tuples = itertools.product(*all_domains)

            # from tuples [(v1, v2, v3)] to [{p1: v1}, ...]
            configurations_dicts = [{k: v for k, v in zip(tunable_variable.keys(), selected_values)} for selected_values in configurations_tuples]
            return configurations_dicts

        confs = grid_search_configurations(cf.TUNABLE_H_PRAM.__dict__)
        for con in confs:
            print("\n\n\n\n HOLA!")
            cf.end_setup(con)
            run_single(cf)


if __name__ == '__main__':
    main()

# python -m src.run --PREDICTION_MODEL MLP
