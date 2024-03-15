
import src.constants as cst

from src.config import LOBCASTSetupRun
import src.utils.utils_training_loop as tru

from src.utils.utils_dataset import pick_dataset
from src.utils.utils_models import pick_model
from src.metrics.metrics_log import Metrics
from src.config import Settings, ConfigHPTunable, ConfigHPTuned
from src.models.model_callbacks import callback_save_model
from pytorch_lightning import Trainer


def run_instance():

    cf = LOBCASTSetupRun()

    data_module = pick_dataset(cf)
    nets_module = pick_model  (cf, data_module, cf.METRICS)

    target_halt_metric = "{}_{}".format(cst.ModelSteps.VALIDATION.value, cst.Metrics.F1.value)

    trainer = Trainer(
        accelerator=cf.SETTINGS.DEVICE,
        devices=cf.SETTINGS.N_GPUs,
        check_val_every_n_epoch=3,
        max_epochs=cf.SETTINGS.EPOCHS_UB,
        callbacks=[
            callback_save_model(cf.SETTINGS.DIR_EXPERIMENTS, target_halt_metric, top_k=3)
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



# class Instance:
#     def __init__(self, model: cst.Models, horizons, seeds):
#         self.model = model
#         self.horizons = horizons
#         self.seeds = seeds

# def run(instances):
#     for instance in instances:
#         for seed in instance.seeds:
#             for win in instance.horizons:
#
#                 print(f"Running LOB experiment: model={instance.model}, fw={win}, seed={seed}")
#
#                 try:
#                     cf: Configuration = Configuration()
#                     cf.SEED = seed
#                     cf.PREDICTION_MODEL = instance.model
#                     cf.CHOSEN_PERIOD = instance.horizons
#
#                     tru.set_seeds(cf)
#
#                     cf.DATASET_NAME = dataset
#                     if mod == cst.Models.METALOB:
#                         cf.DATASET_NAME = cst.DatasetFamily.META
#                         cf.TARGET_DATASET_META_MODEL = target_dataset_meta
#                         cf.JSON_DIRECTORY = json_dir
#
#                     cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.ALL
#                     cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.ALL
#                     cf.CHOSEN_PERIOD = peri
#
#                     cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = cst.WinSize.EVENTS1.value
#                     cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = window_forward.value
#
#                     cf.PREDICTION_MODEL = mod
#
#                     cf.IS_WANDB = int(not is_debug)
#                     cf.IS_HPARAM_SEARCH = int(not is_debug)
#
#                     tru.run(cf)
#
#                 except KeyboardInterrupt:
#                     print("There was a problem running on", server_name.name, "LOB experiment on {}, with K+={}".format(mod, window_forward))
#                     sys.exit()


if __name__ == '__main__':
    # models = [cst.Models.MLP, cst.Models.CNN1]
    # HORIZONS = [10, 20, 30, 40, 50, 60, 70, 80]
    # SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    #
    # instances = []
    # instances += [Instance(model, HORIZONS, SEEDS) for model in models]
    # instances += [Instance(cst.Models.MLP , [10], [1])]
    # instances += [Instance(cst.Models.CNN2, [10], [1])]

    run_instance()

# ARGUMENTS SINGLE:
#
# model
# horizon +
# horizon -
# seed
# dataset name (whole period, just to split)
# is_hyper_search  (else read hps from json)
# chosen stock
#
