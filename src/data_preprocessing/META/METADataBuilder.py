import json
import os
import pandas as pd
import seaborn
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
        self.logits, self.preds = self.__load_predictions_from_jsons()
        # logits.shape = [n_samples, n_classes*n_models]
        # preds.shape = [n_samples, n_models]

        self.n_models = self.preds.shape[1]

        self.__plot_agreement_matrix()
        self.__plot_corr_matrix()



    def __load_predictions_from_jsons(self):
        logits = list()

        for model in cst.Models:

            file_name = self.generic_file_name.format(
                model.name,
                self.seed,
                self.trst,
                self.test,
                self.chosen_dataset,
                self.peri,
                self.bw,
                self.fw,
                self.fiw,
            )

            if os.path.exists(self.project_dir + cst.DIR_FI_FINAL_JSONS + file_name):
                with open(self.project_dir + cst.DIR_FI_FINAL_JSONS + file_name, "r") as f:
                    d = json.loads(f.read())
                    logits_str = d['LOGITS']
                    logits_ = np.array(json.loads(logits_str))
                    assert logits_.shape[0] == self.n_samples, f"For model {model}, n_samples != number of predictions in the json"
                    if (model == cst.Models.DEEPLOBATT):
                        horizons = [horizon.value for horizon in cst.FI_Horizons]
                        h = horizons.index(self.fiw)
                        logits_ = logits_[:, :, h]
                    logits.append(logits_)

        logits = np.dstack(logits)
        # logits.shape = [n_samples, n_classes, n_models]

        preds = np.argmax(logits, axis=1)
        # preds.shape = [n_samples, n_models]

        n_samples, n_classes, n_models = logits.shape
        logits = logits.reshape(n_samples, n_classes*n_models)
        # logits.shape = [n_samples, n_classes*n_models]

        return logits, preds

    def get_samples_train(self):
        s = slice(int(
                self.n_samples*self.split_percentages[0]
        ))
        return self.logits[s], self.truth_y[s]

    def get_samples_val(self):
        s = slice(
            int(self.n_samples*self.split_percentages[0]),
            int(self.n_samples*(self.split_percentages[0]+self.split_percentages[1])),
        )
        return self.logits[s], self.truth_y[s]

    def get_samples_test(self):
        s = slice(
            int(self.n_samples*(self.split_percentages[0]+self.split_percentages[1])),
            -1
        )
        return self.logits[s], self.truth_y[s]

    def __plot_corr_matrix(self):
         # collect data
         models = sorted([model.name for model in cst.Models if (model.name != "METALOB")])

         #we swap the order of DeepLOBATT and DeepLOB, because in the json there is DEEPLOBATT first
         models[8], models[9] = models[9], models[8]
         data = {}
         for i, model in enumerate(models):
             data[model] = self.preds[:, i]

         # form dataframe
         dataframe = pd.DataFrame(data, columns=list(data.keys()))

         # form correlation matrix
         corr_matrix = dataframe.corr()

         heatmap = seaborn.heatmap(corr_matrix, annot=True, fmt=".2f")
         heatmap.set(title="Correlation matrix")

         heatmap.figure.set_size_inches(10, 7)
         plt.show()


    def __plot_agreement_matrix(self):

        # collect data
        models = sorted([model.name for model in cst.Models if (model.name != "METALOB")])

        # we swap the order of DeepLOBATT and DeepLOB, because in the json there is DEEPLOBATT first
        models[8], models[9] = models[9], models[8]
        data = {}
        for i, model in enumerate(models):
            data[model] = self.preds[:, i]

        agreement_matrix = np.zeros((self.n_models, self.n_models))
        list_names = list(data.keys())
        for i in range(self.n_models):
            for j in range(self.n_models):
                agr = 0
                for pred in range(self.n_samples):
                    if self.preds[pred, i] == self.preds[pred, j]:
                       agr += 1
                agreement_matrix[i, j] = agr / self.n_samples

        fig, ax = plt.subplots()
        ax.matshow(agreement_matrix, cmap=plt.cm.Reds)
        ax.set_title('Agreement matrix', fontsize=12)

        #Set number of ticks for x-axis
        ax.set_xticks(np.arange(0, self.n_models, 1))
        #Set ticks labels for x-axis
        ax.set_xticklabels(list_names, fontsize=8, rotation=90, ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        #Set number of ticks for x-axis
        ax.set_yticks(np.arange(0, self.n_models, 1))
        #Set ticks labels for x-axis
        ax.set_yticklabels(list_names, fontsize=8)

        for i in range(self.n_models):
            for j in range(self.n_models):
                c = agreement_matrix[j, i]
                ax.text(i, j, str(round(c, 2)), va='center', ha='center', fontsize=8)

        plt.show()