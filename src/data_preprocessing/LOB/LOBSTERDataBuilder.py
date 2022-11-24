

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
from datetime import datetime


class LOBSTERDataBuilder:
    def __init__(
            self,
            lobster_data_dir,
            dataset_type,
            n_lob_levels=co.N_LOB_LEVELS,
            normalization_type=co.NormalizationType.Z_SCORE,
            normalization_mean=None,
            normalization_std=None,
            start_end_trading_day=("1990-01-01", "2100-01-01"),
            crop_trading_day_by=0,
            window_size_forward=co.FORWARD_WINDOW,
            window_size_backward=co.BACKWARD_WINDOW,
            label_threshold_pos=None,
            label_threshold_neg=None,
            label_dynamic_scaler=None,
            data_granularity=co.Granularity.Sec1,
            is_data_preload=False
    ):

        self.dataset_type = dataset_type
        self.lobster_dataset_name = lobster_data_dir
        self.n_lob_levels = n_lob_levels
        self.normalization_type = normalization_type
        self.data_granularity = data_granularity
        self.is_data_preload = is_data_preload
        self.start_end_trading_day = start_end_trading_day
        self.crop_trading_day_by = crop_trading_day_by

        self.normalization_means = normalization_mean
        self.normalization_stds = normalization_std

        self.window_size_forward = window_size_forward
        self.window_size_backward = window_size_backward
        self.label_dynamic_scaler = label_dynamic_scaler

        self.label_threshold_pos = label_threshold_pos
        self.label_threshold_neg = label_threshold_neg

        # to store the datasets
        self.STOCK_NAME = self.lobster_dataset_name.split("_")[0]  # AVXL_2022-03-01_2022-03-31_10
        self.F_NAME_PICKLE = "{}_{}_{}_{}_{}_{}_{}.pickle".format(
            self.STOCK_NAME,
            self.start_end_trading_day[0],
            self.start_end_trading_day[1],
            self.dataset_type.value,
            self.window_size_backward,
            self.window_size_forward,
            self.label_dynamic_scaler
        )

        self.__data, self.__samples_x, self.__samples_y = None, None, None
        self.__data_init()

    def __read_dataset(self):
        """ Reads the dataset from the pickle (generates it in case). """

        out_df = lbu.from_folder_to_unique_df(
            co.DATA_SOURCE + self.lobster_dataset_name,
            level=self.n_lob_levels,
            granularity=self.data_granularity,
            first_date=self.start_end_trading_day[0],
            last_date=self.start_end_trading_day[1],
            boundaries_purge=self.crop_trading_day_by
        )

        out_df = out_df.fillna(method="ffill")
        self.__data = out_df

    def __normalize_dataset(self):
        """ Does normalization. """
        if self.normalization_type == co.NormalizationType.Z_SCORE:
            # returns the mean and std, both for the price and for the volume
            self.__data, means_dicts, stds_dicts = ppu.stationary_normalize_data(
                self.__data,
                self.normalization_means,
                self.normalization_stds)

            # the training dataset shares its normalization with the others
            if self.dataset_type == co.DatasetType.TRAIN:
                self.normalization_means = means_dicts
                self.normalization_stds = stds_dicts

        elif self.normalization_type == co.NormalizationType.NONE:
            pass

        # needed to update the mid-prices columns, after the normalization, mainly for visualization purposes
        self.__data = ppu.add_midprices_columns(self.__data, self.window_size_forward, self.window_size_backward)

    def __label_dataset(self):
        self.__data, self.label_threshold_pos, self.label_threshold_neg = ppu.add_lob_labels(
            self.__data,
            self.window_size_forward,
            self.window_size_backward,
            self.label_threshold_pos,
            self.label_threshold_neg,
            self.label_dynamic_scaler
        )

    def __snapshotting(self):
        """ This creates 4 X n_levels X window_size_backward -> prediction. """
        print("Snapshotting... (__data has", self.__data.shape[0], "rows)")

        relevant_columns = [c for c in self.__data.columns if "sell" in c or "buy" in c]

        X, Y = [], []
        for st in tqdm.tqdm(range(0, self.__data.shape[0] - self.window_size_backward)):
            x_snap = self.__data.iloc[st:st + self.window_size_backward, :].loc[:, relevant_columns]
            y_snap = self.__data.iloc[st + self.window_size_backward - 1, :][ppu.DataCols.PREDICTION.value]
            X.append(x_snap)
            Y.append(y_snap)

        self.__samples_x, self.__samples_y = np.asarray(X), np.asarray(Y)

    def __under_sampling(self):
        """ Discard instances of the majority class. """
        print("Doing under-sampling...")

        occurrences = collections.Counter(self.__samples_y)
        i_min_occ = min(occurrences, key=occurrences.get)  # index of the class with the least instances
        n_min_occ = occurrences[i_min_occ]                 # number of occurrences of the minority class

        indexes_chosen = []
        for i in [co.Predictions.UPWARD.value, co.Predictions.STATIONARY.value, co.Predictions.DOWNWARD.value]:
            indexes = np.where(self.__samples_y == i)[0]
            if len(indexes) < co.INSTANCES_LOWERBOUND:
                print("The instance is not well formed, there are less than {} instances fo the class {} ({})."
                      .format(co.INSTANCES_LOWERBOUND, i, len(indexes)))
                self.__abort_generation()
                return

            indexes_chosen += list(co.RANDOM_GEN_DATASET.choice(indexes, n_min_occ, replace=False))

        indexes_chosen = np.sort(indexes_chosen)
        self.__samples_x = self.__samples_x[indexes_chosen]
        self.__samples_y = self.__samples_y[indexes_chosen]

    def plot_dataset(self):
        ppu.plot_dataframe_stats(
            self.__data,
            self.label_threshold_pos,
            self.label_threshold_neg,
            self.dataset_type
        )

    def __abort_generation(self):
        self.__data, self.__samples_x, self.__samples_y = None, None, None

    def __serialize_dataset(self):
        if not os.path.exists(co.DATA_PICKLES + self.F_NAME_PICKLE):
            print("Serialization...", self.F_NAME_PICKLE)
            util.write_data((self.__data, self.__samples_x, self.__samples_y), co.DATA_PICKLES, self.F_NAME_PICKLE)

    def __deserialize_dataset(self):
        if os.path.exists(co.DATA_PICKLES + self.F_NAME_PICKLE):
            print("Deserialization...", self.F_NAME_PICKLE)
            out = util.read_data(co.DATA_PICKLES + self.F_NAME_PICKLE)
        else:
            out = None
        return out

    def __prepare_dataset(self):
        """ Crucial call! """
        print("Generating dataset", self.dataset_type)

        self.__read_dataset()
        self.__label_dataset()
        self.__normalize_dataset()
        self.__snapshotting()

        occurrences = collections.Counter(self.__samples_y)
        # print("Before undersampling:", self.dataset_type, occurrences)

        if not self.dataset_type == co.DatasetType.TEST:
            self.__under_sampling()

        # occurrences = collections.Counter(self.__samples_y)
        # print("After undersampling:", self.dataset_type, occurrences)

        # self.plot_dataset()

    def __data_init(self):
        """ This method serializes and deserializes __data."""

        if self.is_data_preload:
            # TODO serialize based on the hash of the values of the parameters that this class uses
            data = self.__deserialize_dataset()
            if data is not None:
                print("Reloaded, not recomputed, nice!")
                self.__data, self.__samples_x, self.__samples_y = data
            else:
                self.__prepare_dataset()  # KEY call, generates the dataset
                self.__serialize_dataset()
        else:
            self.__prepare_dataset()

    def get_samples_x(self, split=None):
        return self.__samples_x if split is None else np.split(self.__samples_x, split)

    def get_samples_y(self, split=None):
        return self.__samples_y if split is None else np.split(self.__samples_y, split)
