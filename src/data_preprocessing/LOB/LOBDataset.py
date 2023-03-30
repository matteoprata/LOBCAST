

from torch.utils import data
import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
import src.constants as cst
import collections
import src.data_preprocessing.preprocessing_utils as ppu

from src.data_preprocessing.LOB.LOBSTERDataBuilder import LOBSTERDataBuilder
from src.config import Configuration


class LOBDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """

    def __init__(
            self,
            config: Configuration,
            dataset_type,
            stocks_list,
            start_end_trading_day,
            vol_price_mu=None,
            vol_price_sig=None,
            num_classes=cst.NUM_CLASSES
    ):
        self.config = config
        self.start_end_trading_day = start_end_trading_day
        self.num_classes = num_classes

        self.vol_price_mu, self.vol_price_sig = vol_price_mu, vol_price_sig
        self.sample_size = self.config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS]  # 100

        map_stock_databuilder = dict()

        # Choose the stock names to open to build the specific dataset.
        # No need to open all for test set, because mu/sig are pre-computed when prev opened train and dev
        stocks_open = None
        if dataset_type == cst.DatasetType.TRAIN:
            # we open also the TEST stock(s) to determine mu and sigma for normalization, needed for all
            stocks_open = list(set(config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value + config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value))  # = [LYFT, NVDA]

        elif dataset_type == cst.DatasetType.VALIDATION:
            stocks_open = config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value  # = [LYFT]

        elif dataset_type == cst.DatasetType.TEST:
            stocks_open = config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value   # = [NVDA]

        for stock in stocks_open:
            path = cst.DATASET_LOBSTER + f'_data_dwn_48_332__{stock}_{config.CHOSEN_PERIOD.value["train"][0]}_{config.CHOSEN_PERIOD.value["test"][1]}_10'

            # normalization_mean = self.vol_price_mu[stock] if stock in self.vol_price_mu else None
            # normalization_std = self.vol_price_sig[stock] if stock in self.vol_price_sig else None

            print()
            print(dataset_type, stocks_list, stock, start_end_trading_day, path, sep="\n")
            print()

            databuilder = LOBSTERDataBuilder(
                stock,
                path,
                config=config,
                n_lob_levels=cst.N_LOB_LEVELS,
                dataset_type=dataset_type,
                start_end_trading_day=start_end_trading_day,
                crop_trading_day_by=60*30,
                window_size_forward=config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW],
                window_size_backward=config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW],
                # normalization_mean=normalization_mean,
                # normalization_std=normalization_std,
                num_snapshots=self.sample_size,
                label_dynamic_scaler=config.HYPER_PARAMETERS[cst.LearningHyperParameter.LABELING_SIGMA_SCALER],
                is_data_preload=config.IS_DATA_PRELOAD
            )

            # self.vol_price_mu[stock], self.vol_price_sig[stock] = databuilder.normalization_means, databuilder.normalization_stds
            map_stock_databuilder[stock] = databuilder  # STOCK: databuilder

        # print('vol_price_mu:', self.vol_price_mu)
        # print('vol_price_sig:', self.vol_price_sig)

        Xs, Ys, Ss, ignore_indices_len = list(), list(), list(), [0]
        for stock in stocks_list:
            print("Handling", stock, "for dataset", dataset_type)
            databuilder = map_stock_databuilder[stock]

            data_x, data_y = databuilder.get_X_nx40(), databuilder.get_Y_n()
            Xs.append(data_x)
            Ys.append(data_y)
            Ss.extend([stock]*len(data_y))
            ignore_indices_len.append(len(data_y))

        # removes the indices that are the first sample_size
        ignore_indices = []
        ind_sf = 0
        for iv in range(len(ignore_indices_len)-1):
            p = ind_sf + ignore_indices_len[iv]
            ignore_indices += list(range(p, p+self.sample_size))  # [range(0, 100), range(1223, 1323), ]
            ind_sf = p

        # X and Y ready to go
        self.x = pd.concat(Xs, axis=0)
        self.x, self.vol_price_mu, self.vol_price_sig = self.__stationary_normalize_data(self.x, self.vol_price_mu, self.vol_price_sig)
        self.x = torch.from_numpy(self.x.values).type(torch.FloatTensor)

        self.y = np.concatenate(Ys, axis=0).astype(int)
        self.stock_sym_name = Ss

        # self.indexes_chosen = self.__under_sampling(self.y, ignore_indices)
        self.x_shape = (self.sample_size, self.x.shape[1])

    def __len__(self):
        """ Denotes the total number of samples. """
        return len(self.y)  # len(self.indexes_chosen)

    def __getitem__(self, index):
        """ Generates samples of data. """
        id_sample = index  # self.indexes_chosen[index]
        x, y, s = self.x[id_sample-self.sample_size:id_sample, :], self.y[id_sample], self.stock_sym_name[id_sample]
        return x, y, s

    def __under_sampling(self, y, ignore_indices):
        """ Discard instances of the majority class. """
        print("Doing under-sampling...")

        y_without_snap = [y[i] for i in range(len(y)) if i not in ignore_indices]  # removes the indices of the first sample for each stock

        occurrences = self.compute_occurrences(y_without_snap)
        i_min_occ = min(occurrences, key=occurrences.get)  # index of the class with the least instances
        n_min_occ = occurrences[i_min_occ]                 # number of occurrences of the minority class

        # if we balance the classes, loss_weights is not useuful anymore
        # occs = np.array([occurrences[k] for k in sorted(occurrences)])
        # self.loss_weights = torch.Tensor(occs / np.sum(occs))

        indexes_ignore = set(ignore_indices)
        indexes_chosen = []
        for i in [cst.Predictions.UPWARD.value, cst.Predictions.STATIONARY.value, cst.Predictions.DOWNWARD.value]:
            indexes = np.where(y == i)[0]
            indexes = np.array(list(set(indexes) - indexes_ignore))  # the indices of the first sample for each stock

            assert len(indexes) >= self.config.INSTANCES_LOWER_BOUND, "The instance is not well formed, there are less than {} instances for the class {} ({}).".format(self.config.INSTANCES_LOWER_BOUND, i, len(indexes))
            indexes_chosen += list(self.config.RANDOM_GEN_DATASET.choice(indexes, n_min_occ, replace=False))

        indexes_chosen = np.sort(indexes_chosen)
        return indexes_chosen

    def __stationary_normalize_data(self, data, normalization_mean=None, normalization_std=None):
        """ DONE: remember to use the mean/std of the training set, to z-normalize the test set. """

        col_choice = {"volumes": ppu.get_volume_column_name(data.columns),
                      "prices":  ppu.get_price_column_name(data.columns)}

        print("Normalization... (using means", normalization_mean, "and stds", normalization_std, ")")

        means_dict, stds_dict = dict(), dict()
        for col_name in col_choice:
            cols = col_choice[col_name]

            if normalization_mean is None and normalization_std is None:
                means_dict[col_name] = data.loc[:, cols].stack().mean()
                stds_dict[col_name] = data.loc[:, cols].stack().std()

            elif normalization_mean is not None and normalization_std is not None:
                means_dict[col_name] = normalization_mean[col_name]
                stds_dict[col_name] = normalization_std[col_name]

            data.loc[:, cols] = (data.loc[:, cols] - means_dict[col_name]) / stds_dict[col_name]
            data.loc[:, cols] = data.loc[:, cols]

            # TODO: volumes and prices can be negative, add min value
            # + abs(data.loc[:, cols].stack().min())  # scale positive
            # data.loc[:, cols].stack().plot.hist(bins=200, alpha=0.5, title=col_name)
            # plt.show()
        # data = data.fillna(method="bfill")
        # data = data.fillna(method="ffill")

        return data, means_dict, stds_dict

    @staticmethod
    def compute_occurrences(y):
        occurrences = collections.Counter(y)
        return occurrences
