import collections
from collections import Counter

import src.constants as cst
import numpy as np
import tqdm
from src.data_preprocessing.METACLASS.meta_preprocessing import load_predictions_fi
from pprint import pprint


class MetaDataBuilder:
    def __init__(
            self,
            fi_data_dir,
            dataset_type,
            horizon=10,
            train_val_split=None,
            chosen_model=None,
            auction=False,
            normalization_type=cst.NormalizationType.Z_SCORE,
            pred_data_path=None
    ):

        assert horizon in (1, 2, 3, 5, 10)

        self.dataset_type = dataset_type
        self.train_val_split = train_val_split
        self.chosen_model = chosen_model
        self.fi_data_dir = fi_data_dir
        self.auction = auction
        self.normalization_type = normalization_type
        self.horizon = horizon
        self.pred_data_path = pred_data_path
        # KEY call, generates the dataset
        self.data, self.samples_x, self.samples_y = None, None, None
        self.__prepare_dataset()

    def __read_dataset(self):
        """ Reads the dataset from the FI files. """

        AUCTION = 'Auction' if self.auction else 'NoAuction'
        N = '1.' if self.normalization_type == cst.NormalizationType.Z_SCORE else '2.' if self.normalization_type == cst.NormalizationType.MINMAX else '3.'
        NORMALIZATION = 'Zscore' if self.normalization_type == cst.NormalizationType.Z_SCORE else 'MinMax' if self.normalization_type == cst.NormalizationType.MINMAX else 'DecPre'
        DATASET_TYPE = 'Training' if self.dataset_type == cst.DatasetType.TRAIN or self.dataset_type == cst.DatasetType.VALIDATION else 'Testing'
        DIR = self.fi_data_dir + \
              "/{}".format(AUCTION) + \
              "/{}{}_{}".format(N, AUCTION, NORMALIZATION) + \
              "/{}_{}_{}".format(AUCTION, NORMALIZATION, DATASET_TYPE)

        NORMALIZATION = 'ZScore' if self.normalization_type == cst.NormalizationType.Z_SCORE else 'MinMax' if self.normalization_type == cst.NormalizationType.MINMAX else 'DecPre'
        DATASET_TYPE = 'Train' if self.dataset_type == cst.DatasetType.TRAIN or self.dataset_type == cst.DatasetType.VALIDATION else 'Test'

        F_EXTENSION = '.txt'

        # if it is training time, we open the 7-days training file
        # if it is testing time, we open the 3 test files
        if self.dataset_type == cst.DatasetType.TRAIN or self.dataset_type == cst.DatasetType.VALIDATION:

            F_NAME = DIR + \
                     '/{}_Dst_{}_{}_CF_7'.format(DATASET_TYPE, AUCTION, NORMALIZATION) + \
                     F_EXTENSION

            out_df = np.loadtxt(F_NAME)

            n_samples_train = int(np.floor(out_df.shape[1] * self.train_val_split))
            if self.dataset_type == cst.DatasetType.TRAIN:
                out_df = out_df[:, :n_samples_train]
            elif self.dataset_type == cst.DatasetType.VALIDATION:
                out_df = out_df[:, n_samples_train:]

        else:

            F_NAMES = [
                DIR + \
                '/{}_Dst_{}_{}_CF_{}'.format(DATASET_TYPE, AUCTION, NORMALIZATION, i) + \
                F_EXTENSION
                for i in range(7, 10)
            ]

            out_df = np.hstack(
                [np.loadtxt(F_NAME) for F_NAME in F_NAMES]
            )

        self.data = out_df

    def __prepareX(self):
        self.samples_x, self.all_pred = load_predictions_fi(num_classes=cst.NUM_CLASSES,
                                                            n_models=cst.N_MODELS,
                                                            n_samples=self.n_samples,
                                                            data_path=self.pred_data_path,
                                                            k=self.horizon)

    def __prepareY(self):
        """ gets the labels """
        # the last five elements in self.data contain the labels
        # they are based on the possible horizon values [1, 2, 3, 5, 10]
        self.samples_y = self.data[cst.HORIZONS_MAPPINGS_FI[self.horizon], :]
        self.samples_y -= 1
        self.n_samples = self.samples_y.shape[0]

    def __prepare_dataset(self):
        """ Crucial call! """
        self.__read_dataset()
        self.__prepareY()
        self.__prepareX()

    def get_data(self, first_half_split=1):
        return self.data

    def get_samples_x(self, first_half_split=1):
        return self.samples_x

    def get_samples_y(self, first_half_split=1):
        return self.samples_y

    def get_n_samples(self):
        return self.n_samples

    def get_all_pred(self):
        return self.all_pred
