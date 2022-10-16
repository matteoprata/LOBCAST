

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


class LOBDataBuilder:
    def __init__(
            self,
            lobster_data_dir,
            dataset_type,
            n_lob_levels=co.N_LOB_LEVELS,
            normalization_type=co.NormalizationType.STATIC,
            normalization_mean=None,
            normalization_std=None,
            start_end_trading_day=("1990-01-01", "2100-01-01"),
            crop_trading_day_by=0,
            window_size_forward=co.FORWARD_WINDOW,
            window_size_backward=co.BACKWARD_WINDOW,
            label_threshold=co.LABELING_THRESHOLD,
            label_dynamic_scaler=co.LABELING_SIGMA_SCALER,
            data_granularity=co.Granularity.Sec1,
            is_data_preload=True):

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
        self.label_dynamic_scaler = label_dynamic_scaler
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
            out_df = lbu.from_folder_to_unique_df(
                self.lobster_data_dir,
                level=self.n_lob_levels,
                granularity=self.data_granularity,
                first_date=self.start_end_trading_day[0],
                last_date=self.start_end_trading_day[1],
                boundaries_purge=self.crop_trading_day_by)

            if not os.path.exists(self.lobster_data_dir + F_EXTENSION):
                util.write_data(out_df, F_NAME)

        out_df = out_df.fillna(method="ffill")
        self.data = out_df

    def __normalize_dataset(self):
        """ Does normalization. """
        if self.normalization_type == co.NormalizationType.STATIC:
            self.data = ppu.stationary_normalize_data(self.data, self.normalization_mean, self.normalization_std)
        elif self.normalization_type == co.NormalizationType.NONE:
            pass

        # needed to update the mid-prices columns, after the normalization, mainly for visualization purposes
        self.data = ppu.add_midprices_columns(self.data, self.window_size_forward, self.window_size_backward)

    def __label_dataset(self):
        self.data, self.label_threshold = ppu.add_lob_labels(self.data, self.window_size_forward, self.window_size_backward, self.label_threshold, self.label_dynamic_scaler)

    def __snapshotting(self, do_shuffle=False):  # TODO implement
        """ This creates 4 X n_levels X window_size_backward -> prediction. """
        relevant_columns = [c for c in self.data.columns if "sell" in c or "buy" in c]

        X, Y = [], []
        print("Snapshotting... (data has", self.data.shape[0], "rows)")
        for st in tqdm.tqdm(range(0, self.data.shape[0]-self.window_size_backward)):
            x_snap = self.data.iloc[st:st+self.window_size_backward, :].loc[:, relevant_columns]
            y_snap = self.data.iloc[st+self.window_size_backward, :][ppu.DataCols.PREDICTION.value]
            X.append(x_snap)
            Y.append(y_snap)

        self.samples_x, self.samples_y = np.asarray(X), np.asarray(Y)

        if do_shuffle:
            index = co.RANDOM_GEN_DATASET.randint(0, self.samples_x.shape[0], size=self.samples_x.shape[0])
            self.samples_x = self.samples_x[index]
            self.samples_y = self.samples_y[index]

    def __under_sampling(self):
        """ Discard instances of the majority class. """
        print("Doing under-sampling...")
        occurrences = collections.Counter(self.samples_y)
        i_min_occ = min(occurrences, key=occurrences.get)  # index of the class with the least instances
        n_min_occ = occurrences[i_min_occ]                 # number of occurrences of the minority class

        indexes_chosen = []
        for i in [co.Predictions.UPWARD.value, co.Predictions.STATIONARY.value, co.Predictions.DOWNWARD.value]:
            indexes = np.where(self.samples_y == i)[0]
            if len(indexes) < co.INSTANCES_LOWERBOUND:
                print("The instance is not well formed, there are less than {} instances fo the class {} ({})."
                      .format(co.INSTANCES_LOWERBOUND, i, len(indexes)))
                self.__abort_generation()
                return

            indexes_chosen += list(co.RANDOM_GEN_DATASET.choice(indexes, n_min_occ, replace=False))

        indexes_chosen = np.sort(indexes_chosen)
        self.samples_x = self.samples_x[indexes_chosen]
        self.samples_y = self.samples_y[indexes_chosen]

    def __plot_dataset(self):
        ppu.plot_dataframe_stats(self.data, self.label_threshold)

    def __abort_generation(self):
        self.data, self.samples_x, self.samples_y = None, None, None

    def __prepare_dataset(self):
        """ Crucial call! """

        self.__read_dataset()
        self.__label_dataset()
        self.__normalize_dataset()
        self.__snapshotting(do_shuffle=co.IS_SHUFFLE_INPUT)
        self.__under_sampling()  # if self.dataset_type == co.DatasetType.TRAIN:
        # self.__plot_dataset()



