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
        self.truth_y = truth_y
        # truth_y.shape = [n_samples]
        self.n_samples = self.truth_y.shape[0]

        self.seed = config.SEED
        self.trst = config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name
        self.test = config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].name
        self.chosen_dataset = cst.DatasetFamily.FI if config.CHOSEN_PERIOD == cst.Periods.FI else cst.DatasetFamily.LOBSTER
        self.peri = config.CHOSEN_PERIOD.name
        self.bw = config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW]
        self.fw = config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW]
        self.fiw = config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON]

        self.generic_file_name = config.cf_name_format(ext='.json')

        # KEY call, generates the dataset
        self.logits, self.preds = self.load_predictions_from_jsons(cst.MODELS_15, self.seed, self.fiw)
        # logits.shape = [n_samples, n_classes*n_models]
        # preds.shape = [n_samples, n_models]

        self.n_models = self.preds.shape[1]

    @staticmethod
    def load_predictions_from_jsons(models, seed, horizon, is_raw=False, n_instances=139487):
        logits = list()

        for model in models:

            file_name = "model={}-seed={}-trst=FI-test=FI-data={}-peri=FI-bw=None-fw=None-fiw={}.json".format(model.name, seed, cst.model_dataset(model), horizon)

            if os.path.exists(cst.DIR_FI_FINAL_JSONS + file_name):
                with open(cst.DIR_FI_FINAL_JSONS + file_name, "r") as f:
                    d = json.loads(f.read())

                    logits_str = d['LOGITS']
                    logits_ = np.array(json.loads(logits_str))

                    # there are models for which the predictions are more because of the smaller len window, so we have to cut them
                    assert logits_.shape[0] >= n_instances
                    # cut = logits_.shape[0] - n_instances
                    logits_ = logits_[-n_instances:]

                    if (model == cst.Models.DEEPLOBATT):
                        horizons = [horizon.value for horizon in cst.FI_Horizons]
                        h = horizons.index(horizon)
                        logits_ = logits_[:, :, h]

                    logits.append(logits_)
            else:
                print("problem with file", cst.DIR_FI_FINAL_JSONS + file_name)
                exit()

        logits = np.dstack(logits)
        # logits.shape = [n_samples, n_classes, n_models]

        preds = np.argmax(logits, axis=1)
        # preds.shape = [n_samples, n_models]

        if is_raw:
            return logits, preds

        return logits, preds

    def get_samples_train(self):
        s = slice(int(
            self.n_samples * self.split_percentages[0]
        ))
        return self.logits[s], self.truth_y[s]

    def get_samples_val(self):
        s = slice(
            int(self.n_samples * self.split_percentages[0]),
            int(self.n_samples * (self.split_percentages[0] + self.split_percentages[1])),
        )
        return self.logits[s], self.truth_y[s]

    def get_samples_test(self):
        s = slice(
            int(self.n_samples * (self.split_percentages[0] + self.split_percentages[1])),
            -1
        )
        return self.logits[s], self.truth_y[s]
