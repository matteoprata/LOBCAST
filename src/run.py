
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
        check_val_every_n_epoch=3,
        max_epochs=cf.SETTINGS.EPOCHS_UB,
        callbacks=[
            callback_save_model(cf.SETTINGS.DIR_EXPERIMENTS, cst.VALIDATION_METRIC, top_k=3)
        ],
    )

    model_path = cf.SETTINGS.TEST_MODEL_PATH if cf.SETTINGS.IS_TEST_ONLY else "best"

    if not cf.SETTINGS.IS_TEST_ONLY:
        trainer.fit(nets_module, data_module)
        cf.METRICS.dump_metrics(cf.SETTINGS.DIR_EXPERIMENTS, "metrics_train.json")
        cf.METRICS.reset_stats()

        trainer.validate(nets_module, data_module, ckpt_path=model_path)

    trainer.test(nets_module, data_module, ckpt_path=model_path)
    cf.METRICS.dump_metrics(cf.SETTINGS.DIR_EXPERIMENTS, "metrics_best.json")


def main():
    # create a wandb sweep
    cf = LOBCASTSetupRun()

    if cf.SETTINGS.IS_WANDB:

        def wandb_lunch(cf):  # runs multiple instances
            with wandb.init() as wandb_instance:
                cf.end_setup(wandb_instance)
                run_single(cf)

        sweep_id = wandb.sweep(
            sweep={
                # 'command': ["${env}", "python3", "${program}", "${args}"],
                # 'program': "src/utils_training_loop.py",
                'method': cf.SETTINGS.SWEEP_METHOD,
                "metric": {"goal": "maximize", "name": cst.VALIDATION_METRIC},
                'parameters': cf.TUNABLE_H_PRAM.__dict__,
                'description': str({**cf.SETTINGS.__dict__, **cf.TUNABLE_H_PRAM.__dict__}),
            },
            project=cst.PROJECT_NAME.format("v" + str(cst.VERSION))
        )

        # create a wandb agent
        wandb.agent(sweep_id, function=lambda: wandb_lunch(cf), count=cst.WANDB_SWEEP_MAX_RUNS)
    else:
        cf.end_setup()
        run_single(cf)


if __name__ == '__main__':
    main()

# python -m src.run --PREDICTION_MODEL MLP
