import json
import os

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
        pass
        # if (all_predictions.shape[1] != n_models):
        #     raise Exception("dimensions of all_predictions are wrong. They have to be [n_models, n_instances]")

        # # collect data
        # data = {
        #     "ATNBoF": all_predictions[:, 0],
        #     "AXIAL-LOB": all_predictions[:, 1],
        #     'BiN-CTABL': all_predictions[:, 2],
        #     'CNN1': all_predictions[:, 3],
        #     'CNN2': all_predictions[:, 4],
        #     'CNN-LSTM': all_predictions[:, 5],
        #     'CTABL': all_predictions[:, 6],
        #     'DAIN-MLP': all_predictions[:, 7],
        #     'DeepLOBATT': all_predictions[:, 8],
        #     'DeepLOB': all_predictions[:, 9],
        #     'DLA': all_predictions[:, 10],
        #     'LSTM': all_predictions[:, 11],
        #     'MLP': all_predictions[:, 12],
        #     'TLONBoF': all_predictions[:, 13],
        #     'TransLOB': all_predictions[:, 14]
        # }

        # # form dataframe
        # dataframe = pd.DataFrame(data, columns=list(data.keys()))

        # # form correlation matrix
        # corr_matrix = dataframe.corr()

        # heatmap = seaborn.heatmap(corr_matrix, annot=True)
        # heatmap.set(title="Correlation matrix")
        # plt.show()


    def __plot_agreement_matrix(self):
        pass
        # all_predictions.permute(1, 0)
        # n_predictions = all_predictions.shape[1]

        # data = {
        #     "ATNBoF": all_predictions[:, 0],
        #     "AXIAL-LOB": all_predictions[:, 1],
        #     'BiN-CTABL': all_predictions[:, 2],
        #     'CNN1': all_predictions[:, 3],
        #     'CNN2': all_predictions[:, 4],
        #     'CNN-LSTM': all_predictions[:, 5],
        #     'CTABL': all_predictions[:, 6],
        #     'DAIN-MLP': all_predictions[:, 7],
        #     'DeepLOBATT': all_predictions[:, 8],
        #     'DeepLOB': all_predictions[:, 9],
        #     'DLA': all_predictions[:, 10],
        #     'LSTM': all_predictions[:, 11],
        #     'MLP': all_predictions[:, 12],
        #     'TLONBoF': all_predictions[:, 13],
        #     'TransLOB': all_predictions[:, 14]
        # }

        # agreement_matrix = np.zeros((n_models, n_models))
        # list_names = list(data.keys())
        # for i in range(n_models):
        #     for j in range(n_models):
        #         agr = 0
        #         for pred in range(n_predictions):
        #             if all_predictions[i, pred] == all_predictions[j, pred]:
        #                 agr += 1
        #         agreement_matrix[i, j] = agr / n_predictions

        # fig, ax = plt.subplots()
        # print("Agreement_Matrix is : ")
        # ax.matshow(agreement_matrix, cmap=plt.cm.Reds)

        # Set number of ticks for x-axis
        # ax.set_xticks(np.arange(0, n_models, 1))
        # Set ticks labels for x-axis
        # ax.set_xticklabels(list_names, fontsize=18)

        # Set number of ticks for x-axis
        # ax.set_yticks(np.arange(0, n_models, 1))
        # Set ticks labels for x-axis
        # ax.set_yticklabels(list_names, fontsize=18)

        # for i in range(n_models):
        #     for j in range(n_models):
        #         c = agreement_matrix[j, i]
        #         ax.text(i, j, str(round(c, 5)), va='center', ha='center')
