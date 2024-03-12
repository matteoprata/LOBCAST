import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils.utils_training_loop import *

kset, mset = cst.FI_Horizons, cst.Models

out_data = "final_data/FI-2010-TESTS/jsons/" #"final_data/FI-2010-TESTS/jsons/" #"final_data/LOB-FEB-TESTS/jsons/"
jsons_dir = out_data

from src.data_preprocessing.LOB.LOBDataset import LOBDataset


def launch_test_FI(seeds_set):

    for s in seeds_set:
        for k in kset:
            cf: Configuration = Configuration()
            cf.SEED = s

            set_seeds(cf)

            cf.IS_TEST_ONLY = True
            cf.DATASET_NAME = cst.DatasetFamily.FI

            cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.FI
            cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.FI
            cf.CHOSEN_PERIOD = cst.Periods.FI
            cf.IS_WANDB = 0
            cf.IS_HPARAM_SEARCH = False

            cf.PREDICTION_MODEL = cst.Models.MAJORITY
            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON] = k.value

            # Setting configuration parameters
            model_params = HP_DICT_MODEL[cf.PREDICTION_MODEL].fixed
            for param in cst.LearningHyperParameter:
                if param.value in model_params:
                    cf.HYPER_PARAMETERS[param] = model_params[param.value]

            # def load_predictions_from_jsons(in_dir, models, seed, horizon, trst="FI", test="FI", peri="FI", bw=None, fw=None, is_raw=False, n_instances=139487):
            logits, _ = MetaDataBuilder.load_predictions_from_jsons(jsons_dir,
                                                                    cst.DatasetFamily.FI,
                                                                    cst.MODELS_15,
                                                                    cf.SEED,
                                                                    cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
                                                                    trst=cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name,
                                                                    test=cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST].name,
                                                                    peri=cf.CHOSEN_PERIOD.name,
                                                                    bw=None,
                                                                    fw=None,
                                                                    is_raw=True,
                                                                    is_ignore_deeplobatt=False)

            logits = logits[int(len(logits)*.15):, :, :]

            horizons = [horizon.value for horizon in cst.FI_Horizons]
            h = horizons.index(k.value)

            # N x 3 x 15
            logits_weighted = logits * cst.FI_2010_PERF[:, h]
            sum_softmax = np.sum(logits_weighted, axis=2)
            preds = np.argmax(sum_softmax, axis=1)

            # run_name_prefix truth
            databuilder_test = FIDataBuilder(
                cst.DATA_SOURCE + cst.DATASET_FI,
                dataset_type=cst.DatasetType.TEST,
                horizon=cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
                window=cf.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS],
                chosen_model=cf.PREDICTION_MODEL
            )
            truths = databuilder_test.samples_y[100:]
            truths = truths[int(len(truths)*.15):]

            val_dict = compute_metrics(truths, preds, cst.ModelSteps.TESTING, [], cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)  # dict to log
            print(val_dict)

            cf.METRICS_JSON.update_metrics(cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, val_dict)
            cm = compute_sk_cm(truths, preds)
            cf.METRICS_JSON.update_cfm(cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, cm)

            cf.METRICS_JSON.close(out_data)


def launch_test_LOBSTER(seeds, kset, period, json_dir):

    for s in seeds:
        for iw, (wb, wf) in enumerate(kset):
            cf = Configuration()
            cf.SEED = s

            set_seeds(cf)

            cf.IS_TEST_ONLY = True
            cf.DATASET_NAME = cst.DatasetFamily.LOB

            cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.ALL
            cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.ALL
            cf.CHOSEN_PERIOD = period
            cf.IS_WANDB = 0
            cf.IS_HPARAM_SEARCH = False

            cf.PREDICTION_MODEL = cst.Models.MAJORITY
            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON] = 10
            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = wf.value
            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = wb.value
            cf.TARGET_DATASET_META_MODEL = cst.DatasetFamily.LOB
            cf.JSON_DIRECTORY = json_dir

            # Setting configuration parameters
            model_params = HP_DICT_MODEL[cf.PREDICTION_MODEL].fixed
            for param in cst.LearningHyperParameter:
                if param.value in model_params:
                    cf.HYPER_PARAMETERS[param] = model_params[param.value]

            # def load_predictions_from_jsons(in_dir, models, seed, horizon, trst="FI", test="FI", peri="FI", bw=None, fw=None, is_raw=False, n_instances=139487):
            logits, _ = MetaDataBuilder.load_predictions_from_jsons(jsons_dir,
                                                                    cf.DATASET_NAME.value,
                                                                    cst.MODELS_15,
                                                                    cf.SEED,
                                                                    cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
                                                                    trst=cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name,
                                                                    test=cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST].name,
                                                                    peri=cf.CHOSEN_PERIOD.name,
                                                                    bw=wb.value,
                                                                    fw=wf.value,
                                                                    is_raw=True,
                                                                    is_ignore_deeplobatt=False)

            # horizons = [horizon.value for horizon in cst.WinSize]
            # h = horizons.index(k.value)
            # exit()

            # LOBSTER_FEB_PERF
            logits_weighted = logits * cst.LOBSTER_FEB_PERF[:len(cst.MODELS_15), iw]
            sum_softmax = np.sum(logits_weighted, axis=2)
            preds = np.argmax(sum_softmax, axis=1)

            # run_name_prefix truth

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

            truths = test_set.y[100:]   # TODO CHECK
            cut = min(truths.shape[0], preds.shape[0])
            print(truths.shape, preds.shape, logits.shape)
            # TODO check cut the end
            truths = truths[:cut]
            preds = preds[:cut]

            # # NEW CODE
            # print(logits.shape, type(logits))
            # print(truths.shape, type(truths))
            #
            # splitter = int(len(logits) * .85)
            # logits = np.reshape(logits, newshape=(-1, 15 * 3))
            # logits_train, logits_test = logits[:splitter, :], logits[splitter:, :]
            # truths_train, truths_test = truths[:splitter], truths[splitter:]
            #
            # print(logits_train.shape, logits_test.shape)
            # clf = BaggingClassifier(n_estimators=100, verbose=True, n_jobs=11).fit(logits_train, truths_train)
            # print("RUN_NAME_PREFIX PREDICT")
            # preds_test = clf.predict(logits_test)
            # print("RUN_NAME_PREFIX METRICS")
            # print(preds_test.shape)
            # val_dict = compute_metrics(truths_test, preds_test, cst.ModelSteps.TESTING, [],
            #                            cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)  # dict to log
            # print(val_dict)
            # # NEW CODE

            val_dict = compute_metrics(truths, preds, cst.ModelSteps.TESTING, [], cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)  # dict to log

            cf.METRICS_JSON.update_metrics(cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, val_dict)

            cm = compute_sk_cm(truths, preds)
            cf.METRICS_JSON.update_cfm(cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, cm)
            cf.METRICS_JSON.close(out_data)


if __name__ == "__main__":
    # seeds = [500]
    # backwards = [cst.WinSize.EVENTS1]
    # forwards = [cst.WinSize.EVENTS5]
    #
    # kset = list(zip(backwards, forwards))
    # period = cst.Periods.JULY2021
    #
    # launch_test_LOBSTER(seeds, kset, period=period, json_dir=jsons_dir)

    seeds = [500, 501, 502, 503, 504]
    kset = cst.FI_Horizons
    launch_test_FI(seeds)
