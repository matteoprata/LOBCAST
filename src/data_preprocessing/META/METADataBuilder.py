import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import src.constants as cst
from src.config import Configuration
import numpy as np


class MetaDataBuilder:
    def __init__(self, truth_y, config: Configuration):

        self.split_percentages = config.META_TRAIN_VAL_TEST_SPLIT
        # truth_y.shape = [n_samples]

        self.seed = config.SEED
        self.trst = config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name
        self.test = config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].name
        self.chosen_dataset = cst.DatasetFamily.FI if config.CHOSEN_PERIOD == cst.Periods.FI else cst.DatasetFamily.LOBSTER
        self.peri = config.CHOSEN_PERIOD.name
        self.bw  = config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW]
        self.fw  = config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW]
        self.fiw = config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON]

        self.generic_file_name = config.cf_name_format(ext='.json')

        # KEY call, generates the dataset

        self.truth_y = truth_y
        # N x 3 x 15  logits
        # N x 1  truths, preds
        self.logits, self.preds = MetaDataBuilder.load_predictions_from_jsons(
            config.JSON_DIRECTORY,
            config.TARGET_DATASET_META_MODEL,
            cst.MODELS_15,
            config.SEED,
            config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
            trst=config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name,
            test=config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].name,
            peri=config.CHOSEN_PERIOD.name,
            bw=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW],
            fw=config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW],
            is_raw=True
        )

        # shuffle
        print(self.truth_y.shape, self.logits.shape, self.preds.shape)

        self.n_samples = min(self.truth_y.shape[0], self.logits.shape[0])
        indexes = np.arange(self.n_samples)
        np.random.shuffle(indexes)

        print(indexes.shape)
        self.truth_y = self.truth_y[indexes]
        self.logits = self.logits[indexes]
        self.preds = self.preds[indexes]

        # logits.shape = [n_samples, n_classes*n_models]
        # preds.shape = [n_samples, n_models]
        print(self.truth_y.shape, self.logits.shape, self.preds.shape)
        self.n_models = self.preds.shape[1]

    @staticmethod
    def load_predictions_from_jsons(in_dir, dataset, models, seed, horizon, trst="FI", test="FI", peri="FI", bw=None, fw=None, is_raw=False, is_ignore_deeplobatt=True):

        MODEL_STATS = dict()
        min_len_model = np.inf
        for model in models:

            if model == cst.Models.DEEPLOBATT and is_ignore_deeplobatt:  # shape did not match
                continue

            file_name = "model={}-seed={}-trst={}-test={}-data={}-peri={}-bw={}-fw={}-fiw={}.json".format(
                model.name,
                seed,
                trst,
                test,
                cst.model_dataset(model, dataset),
                peri,
                bw,
                fw,
                horizon
            )
            print("opening", in_dir, file_name)
            if os.path.exists(in_dir + file_name):
                with open(in_dir + file_name, "r") as f:
                    d = json.loads(f.read())

                    logits_str = d['LOGITS']
                    logits_ = np.array(json.loads(logits_str))

                    MODEL_STATS[model] = d
                    if min_len_model > logits_.shape[0]:
                        min_len_model = logits_.shape[0]

        logits = list()
        for model in MODEL_STATS:
            stats = MODEL_STATS[model]
            logits_str = stats['LOGITS']
            logits_ = np.array(json.loads(logits_str))

            scarto = logits_.shape[0] - min_len_model
            logits_ = logits_[scarto:]

            if model == cst.Models.DEEPLOBATT:
                if dataset == cst.DatasetFamily.FI:
                    horizons = [horizon.value for horizon in cst.FI_Horizons]
                    h = horizons.index(horizon)

                elif dataset == cst.DatasetFamily.LOBSTER:
                    horizons = [cst.WinSize.EVENTS1.value, cst.WinSize.EVENTS2.value, cst.WinSize.EVENTS3.value,
                                cst.WinSize.EVENTS5.value, cst.WinSize.EVENTS10.value]
                    h = horizons.index(fw)
                logits_ = logits_[:, :, h]

            print(logits_.shape)

            # print(model, logits_.shape)
            logits.append(logits_)

        logits = np.dstack(logits)
        # logits.shape = [n_samples, n_classes, n_models]

        preds = np.argmax(logits, axis=1)
        # preds.shape = [n_samples, n_models]

        if is_raw:
            return logits, preds

        return logits, preds

    def get_samples_train(self):
        # s = slice()
        s = 0
        e = int(self.n_samples * self.split_percentages[0])
        return self.logits[s:e], self.truth_y[s:e]

    def get_samples_val(self):
        # s = slice(
        #     int(self.n_samples * self.split_percentages[0]),
        #     int(self.n_samples * (self.split_percentages[0] + self.split_percentages[1])),
        # )
        s = int(self.n_samples * self.split_percentages[0])
        e = int(self.n_samples * self.split_percentages[0]) + int(self.n_samples * self.split_percentages[1])
        return self.logits[s:e], self.truth_y[s:e]

    def get_samples_test(self):
        # s = slice(
        #     int(self.n_samples * (self.split_percentages[0] + self.split_percentages[1])),
        #     -1
        # )
        # s = int(self.n_samples * self.split_percentages[0])
        s = int(self.n_samples * self.split_percentages[0]) + int(self.n_samples * self.split_percentages[1])
        # e = int(self.n_samples * self.split_percentages[0]) + int(self.n_samples * self.split_percentages[1]) + int(self.n_samples * self.split_percentages[1])
        return self.logits[s:], self.truth_y[s:]
