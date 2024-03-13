import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils.utils_training_loop import *
from src.utils.utils_dataset import pick_dataset
from src.utils.utils_models import pick_model
from src.utils.utils_generic import make_dir


def core_test(seed, model, dataset, src_data, out_data, horizon=None, win_back=None, win_forward=None, target_dataset_meta=cst.DatasetFamily.LOB):
    cf: Configuration = Configuration()
    cf.SEED = seed

    set_seeds(cf)

    cf.IS_TEST_ONLY = True

    cf.DATASET_NAME = dataset
    if model == cst.Models.METALOB:
        cf.DATASET_NAME = cst.DatasetFamily.META
        cf.TARGET_DATASET_META_MODEL = target_dataset_meta
        cf.JSON_DIRECTORY = out_data

    if win_back is not None:
        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = win_back.value
        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = win_forward.value

    if horizon is not None:
        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON] = horizon.value

    cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.FI if dataset == cst.DatasetFamily.FI else cst.Stocks.ALL
    cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.FI if dataset == cst.DatasetFamily.FI else cst.Stocks.ALL

    cf.CHOSEN_PERIOD = cst.Periods.FI if dataset == cst.DatasetFamily.FI else cst.Periods.JULY2021
    cf.IS_WANDB = 0
    cf.IS_HPARAM_SEARCH = False

    cf.PREDICTION_MODEL = model

    # set to cst.FI_Horizons.K10.value when lobster (unused)

    # OPEN DIR
    dir_name = "model={}-seed={}-trst={}-test={}-data={}-peri={}-bw={}-fw={}-fiw={}/".format(
        cf.PREDICTION_MODEL.name,
        cf.SEED,
        cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name,
        cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST].name,
        cf.DATASET_NAME.value,
        cf.CHOSEN_PERIOD.name,
        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW],
        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW],
        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
    )

    # OPEN FILE
    files = [f for f in os.listdir(src_data + dir_name) if not f.startswith('.')]
    assert len(
        files) == 1, 'We expect that in the folder there is only the checkpoint with the highest F1-score:\n{}'.format(
        files)

    print("OK")
    print(seed, model, horizon, dir_name)
    # return

    file_name = files[0]

    # Settings configuration hps
    model_params = HP_DICT_MODEL[cf.PREDICTION_MODEL].sweep

    for param in cst.LearningHyperParameter:
        if param.value in model_params:
            values = model_params[param.value]['values']
            assert len(values) == 1
            cf.HYPER_PARAMETERS[param] = values[0]

    datamodule = pick_dataset(cf)
    model = pick_model(cf, datamodule)

    # Loading the model
    max_predict_batches = 500
    trainer = Trainer(accelerator=cst.DEVICE_TYPE, devices=cst.NUM_GPUS, limit_predict_batches=max_predict_batches)
    checkpoint_file_path = src_data + dir_name + file_name
    print("opening", checkpoint_file_path)
    trainer.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_file_path)

    print("done testing")
    print("start eval ")
    # measure inference time of the best model
    datamodule.batch_size = 2  #
    prediction_time = trainer.predict(model, dataloaders=datamodule.test_dataloader(), ckpt_path=checkpoint_file_path)
    prediction_time_mean, prediction_time_std = np.mean(prediction_time), np.std(prediction_time)
    cf.METRICS_JSON.update_metrics(cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, {'inference_mean': prediction_time_mean,
                                                                               'inference_std': prediction_time_std})

    make_dir(out_data)
    cf.METRICS_JSON.close(out_data)


def launch_lobster_test(seeds, model_todo, models_to_avoid, dataset_type, backwards, forwards, src_data, out_data, target_dataset_meta=None):
    for s in seeds:
        for model in model_todo:
            for i in range(len(backwards)):
                assert len(backwards) == len(forwards)
                km, kp = backwards[i], forwards[i]

                if model in set(model_todo) - set(models_to_avoid):
                    core_test(s, model, dataset_type, src_data, out_data, win_back=km, win_forward=kp, target_dataset_meta=cst.DatasetFamily.LOB)


def launch_FI_test(seeds, model_todo, models_to_avoid, dataset_type, kset, src_data, out_data):
    for s in seeds:
        for k in kset:
            for model in model_todo:
                if model in set(model_todo) - set(models_to_avoid):
                    core_test(s, model, dataset_type, src_data, out_data, target_dataset_meta=cst.DatasetFamily.FI)


def lobster_testing(src_data, out_data):
    # kset, mset = cst.FI_Horizons, cst.Models
    dataset_type = cst.DatasetFamily.LOB  # "FI"

    # src_data = "all_models_25_04_23/"         # "all_models_28_03_23/"
    # out_data = "all_models_25_04_23/jsons/"  # "data/experiments/all_models_28_03_23/"

    target_dataset_meta = cst.DatasetFamily.LOB

    model_todo = [cst.Models.BINCTABL]
    models_to_avoid = []
    seeds = [500]

    # LOB
    backwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1]
    forwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS2, cst.WinSize.EVENTS3, cst.WinSize.EVENTS5, cst.WinSize.EVENTS10]

    launch_lobster_test(seeds, model_todo, models_to_avoid, dataset_type, backwards, forwards, src_data, out_data, target_dataset_meta)


def fi_testing(src_data, out_data):
    dataset_type = cst.DatasetFamily.FI  # "FI"

    model_todo = [cst.Models.BINCTABL]
    models_to_avoid = []
    seeds = [500]
    kset = [cst.FI_Horizons.K5]

    launch_FI_test(seeds, model_todo, models_to_avoid, dataset_type, kset, src_data, out_data)


if __name__ == "__main__":
    src_data, out_data = "data/saved_models/", "data/saved_models/"
    fi_testing(src_data, out_data)
