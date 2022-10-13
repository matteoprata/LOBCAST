

import os
import pickle
import src.data_preprocessing.preprocessing_utils as ppu
import src.utils.utilities as util
import src.utils.lob_util as lbu
import src.config as co
from enum import Enum
import tqdm
import collections
import numpy as np


class NormalizationType(Enum):
    STATIC = 0
    DYNAMIC = 1
    NONE = 2


class WIN_SIZE(Enum):
    SEC10 = 10
    SEC20 = 20
    SEC30 = 30

    MIN01 = 60
    MIN05 = 60 * 5
    MIN10 = 60 * 10
    MIN20 = 60 * 20


class Predictions(Enum):
    UPWARD = 2
    DOWNWARD = 0
    STATIONARY = 1


class LOBDataBuilder:
    def __init__(self, lobster_data_dir, dataset_type, n_lob_levels=10, normalization_type=NormalizationType.STATIC,
                 normalization_mean=None, normalization_std=None,
                 start_end_trading_day=("1990-01-01", "2100-01-01"), crop_trading_day_by=0,
                 window_size_forward=WIN_SIZE.MIN01.value, window_size_backward=WIN_SIZE.MIN01.value, label_threshold=.001,
                 data_granularity=co.Granularity.Sec1, is_data_preload=True):

        self.dataset_type = dataset_type
        self.lobster_data_dir = lobster_data_dir
        self.n_lob_levels = n_lob_levels
        self.normalization_type = normalization_type
        self.data_granularity = data_granularity
        self.is_data_preload = is_data_preload
        self.start_end_trading_day = start_end_trading_day
        self.crop_trading_day_by = crop_trading_day_by

        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std

        self.window_size_forward = window_size_forward
        self.window_size_backward = window_size_backward
        self.label_threshold = label_threshold

        # KEY call, generates the dataset
        self.data, self.samples_x, self.samples_y = None, None, None
        self.__prepare_dataset()

    def __read_dataset(self):
        """ Reads the dataset from the pickle (generates it in case). """

        F_EXTENSION = 'dat.pickle'
        F_NAME = self.lobster_data_dir + "_{}_{}".format(self.dataset_type.value, F_EXTENSION)

        if self.is_data_preload and os.path.exists(self.lobster_data_dir + F_EXTENSION):
            out_df = util.read_data(F_NAME)
        else:
            out_df = lbu.from_folder_to_unique_df(self.lobster_data_dir,
                                                  level=self.n_lob_levels,
                                                  granularity=self.data_granularity,
                                                  first_date=self.start_end_trading_day[0],
                                                  last_date=self.start_end_trading_day[0],
                                                  boundaries_purge=self.crop_trading_day_by)
            if not os.path.exists(self.lobster_data_dir + F_EXTENSION):
                util.write_data(out_df, F_NAME)

        out_df = out_df.fillna(method="ffill")
        self.data = out_df

    def __normalize_dataset(self):
        """ Does normalization. """
        if self.normalization_type == NormalizationType.STATIC:
            self.data = ppu.stationary_normalize_data(self.data, self.normalization_mean, self.normalization_std)
        elif self.normalization_type == NormalizationType.NONE:
            pass

        # needed to update the mid-prices columns, after the normalization, mainly for visualization purposes
        self.data = ppu.add_midprices_columns(self.data, self.window_size_forward, self.window_size_backward)

    def __label_dataset(self, label_threshold=None):
        self.label_threshold = label_threshold if label_threshold is not None else self.label_threshold
        self.data = ppu.add_lob_labels(self.data, self.window_size_forward, self.window_size_backward, self.label_threshold)

    def __snapshotting(self):
        """ This creates 4 X n_levels X window_size_backward -> prediction. """
        relevant_columns = [c for c in self.data.columns if "sell" in c or "buy" in c]

        X, Y = [], []
        for st in tqdm.tqdm(range(0, self.data.shape[0]-self.window_size_backward)):
            X += [self.data.iloc[st:st+self.window_size_backward, :].loc[:, relevant_columns]]
            Y += [self.data.iloc[st+self.window_size_backward, :][ppu.DataCols.PREDICTION.value]]
        self.samples_x, self.samples_y = np.asarray(X), np.asarray(Y)

    def __under_sampling(self):
        """ Discard instances of the majority class. """
        occurrences = collections.Counter(self.samples_y)
        i_min_occ = min(occurrences, key=occurrences.get)  # index of the class with the least instances
        n_min_occ = occurrences[i_min_occ]                 # number of occurrences of the minority class

        indexes_chosen = []
        for i in [Predictions.UPWARD.value, Predictions.STATIONARY.value, Predictions.DOWNWARD.value]:
            indexes = np.where(self.samples_y == i)[0]
            indexes_chosen += list(co.RANDOM_GEN_DATASET.choice(indexes, n_min_occ, replace=False))

        self.samples_x = self.samples_x[indexes_chosen]
        self.samples_y = self.samples_y[indexes_chosen]

    def __plot_dataset(self):
        ppu.plot_dataframe_stats(self.data, self.label_threshold)

    def __prepare_dataset(self):
        """ Crucial call! """
        self.__read_dataset()
        self.__label_dataset()
        self.__normalize_dataset()
        self.__snapshotting()

        if self.dataset_type == co.DatasetType.TRAIN:
            self.__under_sampling()

        # self.__plot_dataset()


