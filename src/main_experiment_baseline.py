import os
import sys
import torch

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.config import Configuration
import src.constants as cst
from src.main_single import *
from src.main_helper import pick_dataset, pick_model
from src.models.model_executor import NNEngine

kset, mset = cst.FI_Horizons, cst.Models

out_data = "data/experiments/all_models_28_03_23/"


def launch_test():

    for k in kset:
        cf = set_configuration()
        cf.SEED = 502

        set_seeds(cf)

        cf.IS_TEST_ONLY = True
        cf.CHOSEN_DATASET = cst.DatasetFamily.META

        cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.FI
        cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.FI
        cf.CHOSEN_PERIOD = cst.Periods.FI
        cf.IS_WANDB = 0
        cf.IS_TUNE_H_PARAMS = False

        cf.CHOSEN_MODEL = cst.Models.MAJORITY
        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON] = k.value

        # Setting configuration parameters
        model_params = HP_DICT_MODEL[cf.CHOSEN_MODEL].fixed
        for param in cst.LearningHyperParameter:
            if param.value in model_params:
                cf.HYPER_PARAMETERS[param] = model_params[param.value]

        logits, _ = MetaDataBuilder.load_predictions_from_jsons(cst.MODELS_15, cf.SEED, cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON], is_raw=True)

        horizons = [horizon.value for horizon in cst.FI_Horizons]
        h = horizons.index(k.value)

        logits_weighted = logits * cst.FI_2010_PERF[:, h]
        sum_softmax = np.sum(logits_weighted, axis=2)
        preds = np.argmax(sum_softmax, axis=1)

        # now truth

        databuilder_test = FIDataBuilder(
            cst.DATA_SOURCE + cst.DATASET_FI,
            dataset_type=cst.DatasetType.TEST,
            horizon=cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
            window=cf.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS],
            chosen_model=cf.CHOSEN_MODEL
        )

        truths = databuilder_test.samples_y[99:-1]

        val_dict = compute_metrics(truths, preds, cst.ModelSteps.TESTING, [], cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)  # dict to log
        cf.METRICS_JSON.update_metrics(cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, val_dict)

        cm = compute_sk_cm(truths, preds)
        cf.METRICS_JSON.update_cfm(cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, cm)

        cf.METRICS_JSON.close(out_data)


if __name__ == "__main__":
    launch_test()
