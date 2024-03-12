import collections
from collections import Counter

import src.constants as cst
import numpy as np
import tqdm
import torch
from pprint import pprint
from torch.utils import data


class FIDataset(data.Dataset):
    def __init__(
        self,
        dataset_path,
        dataset_type,
        horizon,
        observation_length,
        train_val_split,
        n_trends,
        auction=False,
        normalization_type=cst.NormalizationType.Z_SCORE,
    ):
        assert horizon in [1, 2, 3, 5, 10]

        self.fi_data_dir = dataset_path
        self.dataset_type = dataset_type
        self.train_val_split = train_val_split
        self.auction = auction
        self.normalization_type = normalization_type
        self.horizon = horizon
        self.observation_length = observation_length
        self.num_classes = n_trends

        # KEY call, generates the dataset
        self.data, self.samples_X, self.samples_y = None, None, None
        self.__prepare_dataset()

        _, occs = self.__class_balancing(self.samples_y)
        # LOSS_WEIGHTS_DICT = {m: 1e6 for m in cst.Models}
        LOSS_WEIGHT = 1e6
        self.loss_weights = torch.Tensor(LOSS_WEIGHT / occs)

        self.samples_X = torch.from_numpy(self.samples_X).type(torch.FloatTensor)  # torch.Size([203800, 40])
        self.samples_y = torch.from_numpy(self.samples_y).type(torch.LongTensor)   # torch.Size([203800])
        self.x_shape = (self.observation_length, self.samples_X.shape[1])          # shape of a single sample

    def __len__(self):
        """ Denotes the total number of samples. """
        return self.samples_X.shape[0] - self.observation_length

    def __getitem__(self, index):
        """ Generates samples of data. """
        sample = self.samples_X[index: index + self.observation_length], self.samples_y[index + self.observation_length - 1]
        return sample

    @staticmethod
    def __class_balancing(y):
        ys_occurrences = collections.Counter(y)
        occs = np.array([ys_occurrences[k] for k in sorted(ys_occurrences)])
        return ys_occurrences, occs

    def __parse_dataset(self):
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

            F_NAME = DIR + '/{}_Dst_{}_{}_CF_7'.format(DATASET_TYPE, AUCTION, NORMALIZATION) + F_EXTENSION
            out_df = np.loadtxt(F_NAME)

            n_samples_train = int(np.floor(out_df.shape[1] * self.train_val_split))
            if self.dataset_type == cst.DatasetType.TRAIN:
                out_df = out_df[:, :n_samples_train]

            elif self.dataset_type == cst.DatasetType.VALIDATION:
                out_df = out_df[:, n_samples_train:]

        else:
            F_NAMES = [DIR + '/{}_Dst_{}_{}_CF_{}'.format(DATASET_TYPE, AUCTION, NORMALIZATION, i) + F_EXTENSION for i in range(7, 10)]
            out_df = np.hstack([np.loadtxt(F_NAME) for F_NAME in F_NAMES])

        self.data = out_df

    def __prepare_X(self):
        """ we only consider the first 40 features, i.e. the 10 levels of the LOB"""
        LOB_TEN_LEVEL_FEATURES = 40
        self.samples_X = self.data[:LOB_TEN_LEVEL_FEATURES, :].transpose()

    def __prepare_y(self):
        """ gets the labels """
        # the last five elements in self.data contain the labels
        # they are based on the possible horizon values [1, 2, 3, 5, 10]
        self.samples_y = self.data[cst.HORIZONS_MAPPINGS_FI[self.horizon], :]
        self.samples_y -= 1

    def __prepare_dataset(self):
        """ Crucial call! """

        self.__parse_dataset()

        self.__prepare_X()
        self.__prepare_y()

        print("> Dataset type:", self.dataset_type, " - normalization:", self.normalization_type)
        occs, occs_vec = self.__class_balancing(self.samples_y)

        perc = ["{}%".format(round(i, 2)) for i in (occs_vec / np.sum(occs_vec)) * 100]
        print(">> balancing", occs, "or even", perc)
        print()
