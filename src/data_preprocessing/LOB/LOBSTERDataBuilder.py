

import os
import src.data_preprocessing.preprocessing_utils as ppu
import src.utils.utils_generic as util
import src.utils.utilis_lobster_datasource as lbu

import numpy as np
from src.config import Configuration
import src.constants as cst


class LOBSTERDataBuilder:
    def __init__(
        self,
        stock_name,
        lobster_data_dir,
        config: Configuration,
        dataset_type,
        n_lob_levels=None,
        normalization_type=cst.NormalizationType.Z_SCORE,
        normalization_mean=None,
        normalization_std=None,
        start_end_trading_day=("1990-01-01", "2100-01-01"),
        crop_trading_day_by=0,
        window_size_forward=None,
        window_size_backward=None,
        num_snapshots=100,
        label_threshold_pos=None,
        label_threshold_neg=None,
        label_dynamic_scaler=None,
        data_granularity=cst.Granularity.Events10,
        is_data_preload=None,
    ):
        self.config = config
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

        self.num_snapshots = num_snapshots

        self.label_threshold_pos = label_threshold_pos
        self.label_threshold_neg = label_threshold_neg

        # to store the datasets
        self.STOCK_NAME = stock_name

        self.F_NAME_PICKLE = "{}_{}_{}_{}_{}_{}.pickle".format(
            self.config.CHOSEN_PERIOD.name,
            self.STOCK_NAME,
            self.start_end_trading_day[0],
            self.start_end_trading_day[1],
            self.dataset_type.value,
            self.data_granularity
        )

        self.__data_un_gathered = None
        self.__data, self.__samples_x, self.__samples_y = None, None, None   # NX40, MX100X40, MX1
        self.__prepare_dataset()  # KEY CALL

    def __read_dataset(self):
        """ Reads the dataset from the pickle if they exist (generates it in case). Otherwise, it opens the trading files."""
        exists = os.path.exists(cst.DATA_PICKLES + self.F_NAME_PICKLE)

        if self.is_data_preload and exists:
            # print("Reloaded, not recomputed, nice!")
            self.__data = self.__deserialize_dataset()
        else:
            out_df = lbu.from_folder_to_unique_df(
                cst.DATA_SOURCE + self.lobster_dataset_name,
                level=self.n_lob_levels,
                granularity=self.data_granularity,
                first_date=self.start_end_trading_day[0],
                last_date=self.start_end_trading_day[1],
                boundaries_purge=self.crop_trading_day_by
            )
            out_df = out_df.fillna(method="ffill")
            out_df = out_df.drop(
                out_df.index[
                    (np.where((out_df.index > '2021-08-03') & (out_df.index < '2021-08-05')))[0]
                ]
            )

            days = list()
            for date in out_df.index:
                date = str(date)
                yyyymmdd = date.split()[0]
                day = yyyymmdd.split('-')[2]
                days.append(int(day))

            days = set(days)
            print()
            print()
            print(self.dataset_type, '\t', self.STOCK_NAME, '\tdays:', sorted(days))

            # out_df_ung = lbu.from_folder_to_unique_df(
            #     cst.DATA_SOURCE + self.lobster_dataset_name,
            #     level=self.n_lob_levels,
            #     granularity=cst.Granularity.Events1,
            #     first_date=self.start_end_trading_day[0],
            #     last_date=self.start_end_trading_day[1],
            #     boundaries_purge=self.crop_trading_day_by
            # )
            #
            # self.__data_un_gathered = out_df_ung

            self.__data = out_df

        if self.is_data_preload and not exists:
            self.__serialize_dataset()

    # def __normalize_dataset(self):
    #     """ Does normalization. """
    #     if self.normalization_type == cst.NormalizationType.Z_SCORE:
    #         # returns the mean and std, both for the price and for the volume
    #         self.__data, means_dicts, stds_dicts = ppu.stationary_normalize_data(
    #             self.__data,
    #             self.normalization_means,
    #             self.normalization_stds
    #         )
    #
    #         # the training dataset shares its normalization with the others
    #         if self.dataset_type == cst.DatasetType.TRAIN:
    #             self.normalization_means = means_dicts
    #             self.normalization_stds = stds_dicts
    #
    #     elif self.normalization_type == cst.NormalizationType.NONE:
    #         pass
    #
    #     # needed to update the mid-prices columns, after the normalization, mainly for visualization purposes
    #     self.__data = ppu.add_midprices_columns(self.__data, self.window_size_forward, self.window_size_backward)

    def __label_dataset(self):
        if self.config.PREDICTION_MODEL == cst.Models.DEEPLOBATT:
            for winsize in cst.FI_Horizons:
                self.__data = ppu.add_lob_labels_march_2023(self.__data, winsize.value, self.window_size_backward, cst.ALPHA)
                self.__data = self.__data.rename(columns={'y': f'y{winsize.value}'})
            self.__data['y'] = self.__data[[f'y{winsize.value}' for winsize in cst.FI_Horizons]].values.tolist()
            self.__data = self.__data.drop([f'y{winsize.value}' for winsize in cst.FI_Horizons], axis=1)
        else:
            self.__data = ppu.add_lob_labels_march_2023(
                self.__data,
                self.window_size_forward,
                self.window_size_backward,
                cst.ALPHA
            )

    # def plot_dataset(self):
    #     ppu.plot_dataframe_stats(
    #         self.__data,
    #         self.label_threshold_pos,
    #         self.label_threshold_neg,
    #         self.dataset_type
    #     )

    def __serialize_dataset(self):
        if not os.path.exists(cst.DATA_PICKLES + self.F_NAME_PICKLE):
            # print("Serialization...", self.F_NAME_PICKLE)
            util.write_data(self.__data, cst.DATA_PICKLES, self.F_NAME_PICKLE)

    def __deserialize_dataset(self):
        # print("Deserialization...", self.F_NAME_PICKLE)
        out = util.read_data(cst.DATA_PICKLES + self.F_NAME_PICKLE)
        return out

    def __prepare_dataset(self):
        """ Crucial call! """
        # print("Generating dataset", self.dataset_type)

        self.__read_dataset()
        self.__label_dataset()
        # self.__normalize_dataset()

        # TOO MUCH MEMORY! AVOID
        # self.__snapshotting()
        #
        # occurrences, _ = self.__compute_occurrences()
        # print("Before undersampling:", self.dataset_type, occurrences)
        #
        # if not self.dataset_type == cst.DatasetType.TEST:
        #     self.__under_sampling()
        #
        # occurrences, _ = self.__compute_occurrences()
        # print("After undersampling:", self.dataset_type, occurrences)

        # self.plot_dataset()

    def get_X_nx40(self):
        return self.__data.iloc[:, :-5]  # in the last 5 columns there are predictions and shifts

    def get_Xung_nx40(self):
        return self.__data_un_gathered.iloc[:, :]

    def get_Y_n(self):
        if self.config.PREDICTION_MODEL == cst.Models.DEEPLOBATT:
            return np.asarray(self.__data[ppu.DataCols.PREDICTION.value].values.tolist())
        else:
            return self.__data[ppu.DataCols.PREDICTION.value]
