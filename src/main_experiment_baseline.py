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

out_data = "all_models_25_04_23/jsons/"
jsons_dir = "all_models_25_04_23/jsons/"

from src.data_preprocessing.LOB.LOBDataset import LOBDataset


def launch_test_FI():

    for k in kset:
        cf = set_configuration()
        cf.SEED = 502

        set_seeds(cf)

        cf.IS_TEST_ONLY = True
        cf.CHOSEN_DATASET = cst.DatasetFamily.IF

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

        # def load_predictions_from_jsons(in_dir, models, seed, horizon, trst="FI", test="FI", peri="FI", bw=None, fw=None, is_raw=False, n_instances=139487):
        logits, _ = MetaDataBuilder.load_predictions_from_jsons(jsons_dir, cf.CHOSEN_DATASET.value, cst.MODELS_15, cf.SEED, cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON], is_raw=True)

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
        print(val_dict)
        exit()
        cf.METRICS_JSON.update_metrics(cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, val_dict)

        cm = compute_sk_cm(truths, preds)
        cf.METRICS_JSON.update_cfm(cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, cm)

        cf.METRICS_JSON.close(out_data)


def launch_test_LOBSTER(seeds, kset, period):

    for s in seeds:
        for iw, (wb, wf) in enumerate(kset):
            cf = set_configuration()
            cf.SEED = s

            set_seeds(cf)

            cf.IS_TEST_ONLY = True
            cf.CHOSEN_DATASET = cst.DatasetFamily.LOBSTER

            cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.ALL
            cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.ALL
            cf.CHOSEN_PERIOD = period
            cf.IS_WANDB = 0
            cf.IS_TUNE_H_PARAMS = False

            cf.CHOSEN_MODEL = cst.Models.MAJORITY
            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON] = 10
            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = wf.value
            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = wb.value

            # Setting configuration parameters
            model_params = HP_DICT_MODEL[cf.CHOSEN_MODEL].fixed
            for param in cst.LearningHyperParameter:
                if param.value in model_params:
                    cf.HYPER_PARAMETERS[param] = model_params[param.value]

            N_INSTANCES = cst.n_test_instances(cf.CHOSEN_DATASET, cf.CHOSEN_PERIOD)
            # def load_predictions_from_jsons(in_dir, models, seed, horizon, trst="FI", test="FI", peri="FI", bw=None, fw=None, is_raw=False, n_instances=139487):
            logits, _ = MetaDataBuilder.load_predictions_from_jsons(jsons_dir,
                                                                    cf.CHOSEN_DATASET.value,
                                                                    cst.MODELS_15,
                                                                    cf.SEED,
                                                                    cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
                                                                    trst=cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name,
                                                                    test=cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST].name,
                                                                    peri=cf.CHOSEN_PERIOD.name,
                                                                    bw=wb.value,
                                                                    fw=wf.value,
                                                                    is_raw=True,
                                                                    n_instances=N_INSTANCES)

            # horizons = [horizon.value for horizon in cst.WinSize]
            # h = horizons.index(k.value)
            # exit()

            logits_weighted = logits * cst.LOBSTER_JULY_PERF[:, iw]
            sum_softmax = np.sum(logits_weighted, axis=2)
            preds = np.argmax(sum_softmax, axis=1)

            # now truth

            train_set = LOBDataset(
                config=cf,
                dataset_type=cst.DatasetType.TRAIN,
                stocks_list=cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value,
                start_end_trading_day=cf.CHOSEN_PERIOD.value['train']
            )

            vol_price_mu, vol_price_sig = train_set.vol_price_mu, train_set.vol_price_sig

            test_set = LOBDataset(
                config=cf,
                dataset_type=cst.DatasetType.TEST,
                stocks_list=cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value,
                start_end_trading_day=cf.CHOSEN_PERIOD.value['test'],
                vol_price_mu=vol_price_mu, vol_price_sig=vol_price_sig
            )

            truths = test_set.y[-N_INSTANCES:]

            val_dict = compute_metrics(truths, preds, cst.ModelSteps.TESTING, [], cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)  # dict to log

            cf.METRICS_JSON.update_metrics(cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, val_dict)

            cm = compute_sk_cm(truths, preds)
            cf.METRICS_JSON.update_cfm(cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, cm)

            cf.METRICS_JSON.close(out_data)
            print("done 1")
        print("done 2")
    print("done 3")


if __name__ == "__main__":
    seeds = [500, 501, 502, 503, 504]
    backwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1, cst.WinSize.EVENTS1]
    forwards = [cst.WinSize.EVENTS1, cst.WinSize.EVENTS2, cst.WinSize.EVENTS3, cst.WinSize.EVENTS5, cst.WinSize.EVENTS10]

    kset = list(zip(backwards, forwards))
    period = cst.Periods.JULY2021
    launch_test_LOBSTER(seeds, kset, period=period)
