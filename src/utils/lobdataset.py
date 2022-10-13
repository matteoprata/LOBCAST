import os 
import torch 
import requests
import pickle
import pandas as pd
import numpy as np 

from abc import ABCMeta
from torch.utils import data

from matplotlib import pyplot as plt
from src import config
import src.utils.lob_util as lbu
import src.generator.baseapproach as baseapproach
from src.costpredictor.markovian_cost_lobster import add_outstanding_volume, add_volatility
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch.nn.functional as F


class Dataset(data.Dataset):
    """ Characterizes a dataset for PyTorch """

    def __init__(self, df_lob, horizon, num_classes=3, goal_col="y", one_hot_encoding=False):
        """Initialization"""
        self.df_lob = df_lob
        self.num_classes = num_classes
        self.horizon = horizon
        self.goal_col = goal_col

        x, y = self.__prepare_observations()

        self.reset_x_y(x, y)

        if one_hot_encoding:
            self.y = F.one_hot(self.y.to(torch.int64), num_classes=self.num_classes)

    def reset_x_y(self, x, y):
        print("resetting x, y")
        self.length = len(x)
        self.unique, self.counts = np.unique(y, return_counts=True)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __prepare_observations(self):
        """ preprocessing data, so to split X and Y """
        relevant_columns = [c for c in self.df_lob.columns if "sell" in c or "buy" in c or "volatility" in c]

        X = self.df_lob[relevant_columns].values 
        Y = self.df_lob[self.goal_col].values

        [rows, cols] = X.shape
        dX, dY = np.array(X), np.array(Y)

        # TODO: removes the first HORIZON rows, but why here?
        dataY = dY[self.horizon - 1:]

        # rows - self.horizon + 1: number of rows minus the horizon rows
        # crates the mapping: window size rows -> class
        dataX = np.zeros((rows - self.horizon + 1, self.horizon, cols))

        for i in range(rows - self.horizon + 1):
            observation = dX[self.horizon*i:self.horizon*(i+1), :]
            if observation.shape[0] == self.horizon:
                dataX[i, :, :] = observation  # the i-th observation
        return dataX, dataY

    def __len__(self):
        """Denotes the total number of samples"""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]


class RawDataset(metaclass=ABCMeta):
    """ Split dataset. """

    # The only accepted horizon that we can use in our system.
    ACCEPTED_HORIZONS = [10, 20, 50, 100, 3600]

    def __init__(self, horizon : int, n_levels=10, sign_threshold=0.002, train_val_test_split=(0.5, 0.25, 0.25), one_hot_encoding=True, history_T : int = 100):
        """ The dataset class load the files to be used in the framework.
            Both lobster and deeplob data are supported. 
        Args:
            horizon (int): The lenght of historical orderbook snapshots to use in the prediction
            history_T (int) : The lenght of historical snapshot of the orderbook to use as input for the prediction
            n_levels (int, optional): The level of orderbook to use. Defaults to 10.
            sign_threshold (float, optional): How the define a future stock trend (increase, decrease, stable price). Defaults to 0.002.
        """
        self.horizon = horizon
        self.n_levels = n_levels
        self.train_val_test_split = train_val_test_split   #
        self.sign_threshold = sign_threshold
        self.one_hot_encoding = one_hot_encoding
        self.all_df_data = None
        assert n_levels == 10, "Level != 10 not supported for now"
        # assert horizon in RawDataset.ACCEPTED_HORIZONS, "The horizon {} as input is not supported.".format(horizon)
    
    def min_volume(self):
        cols_vols = [col for col in self.all_df_data.columns  if "v" in col]
        min_cols_vols = self.all_df_data[cols_vols].stack().min()
        return min_cols_vols

    def split_train_data(self, torch_dataset=False):
        """ return only the training data
        
            perc_training (float) : How much percentage of all the data we have to return, for training.
        """
        stop = int(len(self.all_df_data)*self.train_val_test_split[0])  # first 50%
        data_df = self.all_df_data.iloc[:stop]
        if torch_dataset:
            return Dataset(data_df, horizon=self.horizon, one_hot_encoding=self.one_hot_encoding)
        return data_df

    def split_test_data(self, torch_dataset=False):
        """ return only the testing data 
        
            perc_testing (float) : How much percentage of all the data we have to return, for testing.
        
        """
        start = int(len(self.all_df_data)*(1-self.train_val_test_split[2]))  # last 25%
        data_df = self.all_df_data[start:]
        if torch_dataset:
            return Dataset(data_df, horizon=self.horizon, one_hot_encoding=self.one_hot_encoding)
        return data_df

    def split_val_data(self, torch_dataset=False):
        """ return only the validation data
        
            perc_validation (float) : How much percentage of all the data we have to return, for validation.
        """
        start = int(len(self.all_df_data)*(self.train_val_test_split[0]))
        end = int(len(self.all_df_data)*(1-self.train_val_test_split[1]))
        data_df = self.all_df_data[start:end]
        if torch_dataset:
            return Dataset(data_df, horizon=self.horizon, one_hot_encoding=self.one_hot_encoding)
        return data_df


class LOBDataset(RawDataset):

    def __init__(self, lobster_data_dir : str, horizon : int, n_levels=10, sign_threshold=0.002,
                    normalize : bool = True, one_hot_encoding : bool = False, 
                    train_val_test_split=(0.6, 0.2, 0.2), time_norm='5D',
                    volatility_window : str = "1h", sell_size : bool = True,
                    add_costpredictor_columns : bool = False, history_T : int = 100,
                    ratio_rolling_window : int = 3600, last_date_data : str = "2100-01-01"):
        """ The dataset class load the files to be used in the framework.
            Both lobster and deeplob data are supported. 
        Args:
            lobster_data_dir (str): The directory with orderbook.CSV and messages.CSV data to load.
            horizon (int): The lenght of future data to use in the prediction
            n_levels (int, optional): The level of orderbook to use. Defaults to 10.
            normalize (bool) : if normalize or not the lob data
            sign_threshold (float, optional): How the define a future stock trend (increase, decrease, stable price). Defaults to 0.002.
            history_T (int) : The lenght of historical snapshot of the orderbook to use as input for the prediction
            time_norm (str): Define the time for the rolling mean and std.
            volatility_window (str) : the volatility window to use in the cost prediction method
            merge_cost_df (bool) : if we want to merge the cost_df along the all_df data that will be used by GAN
            sell_size (bool) : if we want to work (on perturbation) only on the sell side
            last_date_data (str) : the last date to load, to shorten the time-series
            ratio_rolling_window (int) : the seconds to compute mean and std of mid-price ratio and therefore the classes (-1, 0, 1) distribution. Put -1 to disable it.
        """
        super().__init__(horizon, n_levels, sign_threshold, train_val_test_split, one_hot_encoding=one_hot_encoding)
        self.time_norm = time_norm
        self.normalize = normalize
        self.add_costpredictor_columns = add_costpredictor_columns
        self.volatility_window = volatility_window
        self.sell_size = sell_size
        self.last_date_data = last_date_data
        self.lobster_data_dir = lobster_data_dir
        self.train_val_test_split = train_val_test_split
        self.ratio_rolling_window = ratio_rolling_window

        # The dataframe below has these cols:
        # ['psell1', 'vsell1', 'pbuy1', 'vbuy1', ... ,
        #  'midprice', 'm+t', 'ratio_y', 'ratio_y_rolled_std', 'ratio_y_rolled_mean', 'y']

        # KEY CALL
        self.all_df_data, self.cost_df_data = self.__read_dataset()
        print(self.all_df_data.columns)
        exit()

        if self.normalize:
            self.all_df_data = self.__normalize_data()

        # self.all_df_data = lbu.add_lob_labels_rolling(self.all_df_data, self.horizon, self.ratio_rolling_window)
        self.all_df_data = lbu.add_lob_labels(self.all_df_data,
                                              rolling_tu=self.horizon,
                                              sign_threshold=self.sign_threshold)

        self.plot_dataframe_stats()


        # merge cost_df to all_df_data
        if self.add_costpredictor_columns:
            self.__merge_datasets()

    def plot_dataframe_stats(self):
        out_df = self.all_df_data.reset_index()
        ax = out_df[["midprice", "m+t"]].plot()
        yticks, _ = plt.yticks()

        ax2 = ax.twinx()
        ax2 = out_df[["ratio_y"]].plot(ax=ax)
        ax2.set_ylim([-1, max(yticks)])
        ax2.axhline(y=self.sign_threshold, color='r', linestyle=':')
        ax2.axhline(y=-self.sign_threshold, color='r', linestyle=':')
        out_df[["y"]].plot()
        plt.show()

    def __merge_datasets(self):
        """ merge cost_df and all_df """

        all_cost_df_columns = [f +"_cost" for f in self.cost_df_data.columns]
        self.cost_df_data.columns = all_cost_df_columns
        self.cost_df_data = self.cost_df_data[all_cost_df_columns]
        self.all_df_data = self.all_df_data.join(self.cost_df_data)

    def __read_dataset(self, plot=False, preload=True):
        """ actually read the data as a df in the needed format """
        if preload:
            if os.path.exists(self.lobster_data_dir + 'dat.pickle'):
                with open(self.lobster_data_dir + 'dat.pickle', 'rb') as handle:
                    out_df = pickle.load(handle)
            else:
                out_df = lbu.from_folder_to_unique_df(self.lobster_data_dir, plot=plot, level=self.n_levels, last_date=self.last_date_data)
                with open(self.lobster_data_dir + 'dat.pickle', 'wb') as handle:
                    pickle.dump(out_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            out_df = lbu.from_folder_to_unique_df(self.lobster_data_dir, plot=plot, level=self.n_levels, last_date=self.last_date_data)

        out_df = out_df.fillna(method="ffill")
        if self.add_costpredictor_columns:
            cost_df = self.__dataset_for_cost_gan(self.volatility_window)  # [,,,,,]
            # drop last rolling_tu items, as they are not predictable.
            cost_df = cost_df[0:len(cost_df) - (self.horizon-1)]
        else:
            cost_df = None

        # add label to our dataset 
        # old code with static thresholds

        return out_df, cost_df

    def __dataset_for_cost_gan(self, volatility_window):
        """
            This method is used to create a parallel dataset with outstanding volumes and volatility that
                will be used in the gan training process/evalution to understand the expected cost of the attacks
            out_df : is lobster raw data at tick level
        """
        df = lbu.from_folder_to_unique_df(self.lobster_data_dir, plot=False, level=self.n_levels,
                                          granularity=None, add_messages=True)

        df = df.fillna(method="ffill")

        df = add_outstanding_volume(df)
        # add mid-price for matches
        df["mid_price"] = (df["psell1"] + df["pbuy1"]) / 2
        #Add volatility feature to the dataframe
        df = add_volatility(df, volatility_window=volatility_window)

        df = df.reset_index()
        # pick the correct granularity that we want
        df.set_index("date", inplace=True)
        df = df.resample(config.Granularity.Sec1.value).first()

        # keep only needed columns
        if self.sell_size:
            needed_columns = [f for f in df.columns if "volatility" in f or "vsell" in f]
        else:
            needed_columns = [f for f in df.columns if "volatility" in f or "vbuy" in f]

        assert len(needed_columns) == 11, "We expect 10 volumes + 1 volatility"
        df = df[needed_columns]
        return df

    def __stationary_normalize_data(self):
        out_df = self.all_df_data

        columns_volumes_to_normalize = [f for f in out_df.columns if "vbuy" in f or "vsell" in f]
        columns_prices_to_normalize = [f for f in out_df.columns if "pbuy" in f or "psell" in f]

        col_choice = {"volumes": columns_volumes_to_normalize, "prices": columns_prices_to_normalize}

        print("Normalization...")
        for col_name in col_choice:
            cols = col_choice[col_name]
            mean_out = out_df.loc[:, cols].stack().mean()
            std_out = out_df.loc[:, cols].stack().std()

            # print(out_df.loc[:, cols].head(100).to_string())
            # print()

            out_df.loc[:, cols] = (out_df.loc[:, cols] - mean_out) / std_out
            out_df.loc[:, cols] = out_df.loc[:, cols] + abs(out_df.loc[:, cols].stack().min())  # scale positive

            # mat_stacked = out_df.loc[:, cols].stack()
            # print("old global avg", col_name, mean_out, "new avg", mat_stacked.mean())
            # print("old global std", col_name, std_out, "new std", mat_stacked.std())
            # print("min", mat_stacked.min(), "max", mat_stacked.max())

            # ax = out_df.loc[:, cols].stack().plot.hist(bins=50, alpha=0.5, title=col_name)
            # plt.show()

        out_df = out_df.fillna(method="bfill")
        out_df = out_df.fillna(method="ffill")
        return out_df

    def __normalize_data(self, new_normalization_stationary=True):
        if new_normalization_stationary:
            return self.__stationary_normalize_data()
        else:
            return self.__no_stationary_normalize_data()

    def __no_stationary_normalize_data_OLD(self):
        out_df = self.all_df_data
        # out_df['date'] = pd.to_datetime(out_df['date'])
        # out_df = out_df.set_index('date')
        columns_to_normalize = [f for f in out_df.columns if "buy" in f or "sell" in f]

        # print(out_df[:10].to_string())

        for col in columns_to_normalize:
            out_df[col] = ((out_df[col] - out_df[col].rolling(self.time_norm).mean().shift())
                           / out_df[col].rolling(self.time_norm).std().shift())
            out_df[out_df == np.inf] = np.nan
            out_df[out_df == -np.inf] = np.nan

            out_df[col] = out_df[col] + np.abs(np.nanmin(out_df[col]))

        out_df.reset_index(inplace=True)
        out_df = out_df.fillna(method="bfill")

        # print(out_df[:10].to_string())
        # exit()
        return out_df

    def __no_stationary_normalize_data(self):
        out_df = self.all_df_data
        columns_volumes_to_normalize = [f for f in out_df.columns if "vbuy" in f or "vsell" in f]
        columns_prices_to_normalize = [f for f in out_df.columns if "pbuy" in f or "psell" in f]

        col_choice = {"volumes": columns_volumes_to_normalize, "prices": columns_prices_to_normalize}

        for col_name in col_choice:
            cols = col_choice[col_name]
            out_df_2 = out_df.copy()
            df_stacked = pd.concat([out_df_2[[col]].rename(columns={col: col_name}) for col in cols], axis=0).sort_index()
            mean_out = df_stacked.rolling(self.time_norm).mean()
            std_out = df_stacked.rolling(self.time_norm).std()

            is_duplicate = df_stacked.index.duplicated(keep="last")
            mean_out = mean_out[~is_duplicate].values
            std_out = std_out[~is_duplicate].values

            out_df.loc[:, cols] = (out_df.loc[:, cols] - mean_out) / std_out
            # out_df.loc[:, cols] = out_df.loc[:, cols] + abs(out_df.loc[:, cols].stack().min())  # scale positive

        out_df = out_df.fillna(method="bfill")
        out_df = out_df.fillna(method="ffill")

        # ax = out_df.loc[:, columns_prices_to_normalize].stack().plot.hist(bins=1000, alpha=0.5, title=col_name)
        # plt.show()
        # exit()
        return out_df

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.all_df_data)

    def all_data(self, torch_dataset=False):
        """ return all data togheter, train + test """
        if torch_dataset:
            return Dataset(self.all_df_data, horizon=self.horizon)
        return self.all_df_data
    

class DEEPDataset(RawDataset):

    URL_RESOURCE = "https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip"
    UNZIP_COMMMAND = "unzip -n {} -d {}"

    def __init__(self, deplob_data_dir : str, horizon : int,
                 n_levels=10, one_hot_encoding = False, sign_threshold=0.002,
                 add_costpredictor_columns=False,
                 sell_size: bool = True,
                 volatility_window : int = 3600):
        """ The dataset class load the files to be used in the framework.
            Both lobster and deeplob data are supported. 
        Args:
            deplob_data_dir (str): The directory with deeplob data (e.g., Test_Dst_NoAuction_DecP*).
            horizon (int): The lenght of historical orderbook snapshots to use in the prediction
            n_levels (int, optional): The level of orderbook to use. Defaults to 10.
            sign_threshold (float, optional): How the define a future stock trend (increase, decrease, stable price). Defaults to 0.002.
        """
        super().__init__(horizon, n_levels, sign_threshold, None, one_hot_encoding=one_hot_encoding)
        self.add_costpredictor_columns = add_costpredictor_columns
        self.sell_size = sell_size
        self.deplob_data_dir = deplob_data_dir
        self.len_test = None
        self.len_train = None
        self.volatility_window = volatility_window
        self.all_df_data, self.cost_df_data = self.__read_dataset()  # normal lob | outstanding volumes + volatility

        # TODO: self.all_df_data, self.cost_df_data may not match in number of rows
        # merge cost_df to all_df_data
        if self.add_costpredictor_columns:
            self.__merge_datasets()

    def __merge_datasets(self):
        """ merge cost_df and all_df """
        all_cost_df_columns = [f + "_cost" for f in self.cost_df_data.columns]
        self.cost_df_data.columns = all_cost_df_columns

        self.all_df_data = self.all_df_data.reset_index(drop=True)
        self.cost_df_data = self.cost_df_data.reset_index(drop=True)
        self.all_df_data[all_cost_df_columns] = self.cost_df_data[all_cost_df_columns]

    def __read_dataset(self):
        """ actually read the data as a df in the needed format """
        test_files, train_files = DEEPDataset.dataset_files(self.deplob_data_dir)
        df_test = pd.concat([lbu.f1_file_dataset_to_lob_df(f) for f in test_files])
        df_train = pd.concat([lbu.f1_file_dataset_to_lob_df(f) for f in train_files])

        self.len_test = len(df_test)
        self.len_train = len(df_train)

        out_df = pd.concat([df_train, df_test])

        if self.add_costpredictor_columns:
            cost_df = self.__dataset_for_cost_gan(out_df)  # [,,,,,]
            # drop last rolling_tu items, as they are not predictable.
            cost_df = cost_df[1:len(cost_df) - (self.horizon-1)]
            out_df = out_df[1:]
        else:
            cost_df = None

        return out_df, cost_df

    def __dataset_for_cost_gan(self, out_df):
        """
            This method is used to create a parallel dataset with outstanding volumes and volatility that
                will be used in the gan training process/evalution to understand the expected cost of the attacks
            out_df : is lobster raw data at tick level
        """
        df = out_df.fillna(method="ffill")
        df = add_outstanding_volume(df)

        # add mid-price for matches
        df["mid_price"] = (df["psell1"] + df["pbuy1"]) / 2
        # Add volatility feature to the dataframe
        df["volatility"] = df["mid_price"].rolling(self.volatility_window, min_periods=1).std()
        df["volatility"] = df["volatility"].fillna(method="ffill")

        # keep only needed columns
        if self.sell_size:
            needed_columns = [f for f in df.columns if "volatility" in f or "vsell" in f]
        else:
            needed_columns = [f for f in df.columns if "volatility" in f or "vbuy" in f]

        assert len(needed_columns) == 11, "We expect 10 volumes + 1 volatility"
        df = df[needed_columns]
        return df

    def all_data(self, torch_dataset=False):
        """ return all data togheter, train + test """
        if torch_dataset:
            return Dataset(self.all_df_data, horizon=self.horizon)
        return self.all_df_data 

    # TODO: fix, return a training set!
    def train_data(self, torch_dataset=False):
        """ return only the training data """
        if torch_dataset:
            return Dataset(self.all_df_data, horizon=self.horizon)
        return self.all_df_data

    # TODO: fix, return a training set, for now it is not defined for FI
    def test_data(self, torch_dataset=False):
        """ return only the testing data """
        if torch_dataset:
            return Dataset(self.df_test, horizon=self.horizon)
        return self.df_test 

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.all_df_data)

    @staticmethod
    def check_and_download_deeplob_data():
        """  This method if the deeplob files exists, otherwise they will be downloaded """
        data_dir = DEEPDataset.base_deeplob_dir()
        os.makedirs(data_dir, exist_ok=True)
        test_files, train_files = DEEPDataset.dataset_files(data_dir)
        if len(test_files + train_files) == 0:
            r = requests.get(DEEPDataset.URL_RESOURCE, allow_redirects=True)
            open(data_dir + 'data.zip', 'wb').write(r.content)
            os.system(DEEPDataset.UNZIP_COMMMAND.format(data_dir + "data.zip", data_dir))
            print('data downloaded.')

    @staticmethod
    def base_deeplob_dir():
        """ return the common folder where deeplob data are stored """
        return "indata/FI-2010/"

    @staticmethod
    def dataset_files(data_dir):
        test_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if "Test" in f]
        train_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if "Train" in f]
        return test_files, train_files

    def split_train_data(self, torch_dataset=False):
        """ return only the training data
        
            perc_training (float) : How much percentage of all the data we have to return, for training.
        """
        len_train = int(np.floor(self.len_train * 0.8))       
        data_df = self.all_df_data[:len_train] 
        if torch_dataset:
            return Dataset(data_df, horizon=self.horizon, one_hot_encoding=self.one_hot_encoding)
        return data_df

    def split_test_data(self, torch_dataset=False):
        """ return only the testing data
            perc_testing (float) : How much percentage of all the data we have to return, for testing.
        """
        end_index = int(np.floor(self.len_train))       
        data_df = self.all_df_data[end_index:]
        if torch_dataset:
            return Dataset(data_df, horizon=self.horizon, one_hot_encoding=self.one_hot_encoding)
        return data_df

    def split_val_data(self, torch_dataset=False):
        """ return only the validation data
            perc_validation (float) : How much percentage of all the data we have to return, for validation.
        """
        start_index = int(np.floor(self.len_train * 0.8)) 
        end_index = int(np.floor(self.len_train))       
        data_df = self.all_df_data[start_index:end_index]
        if torch_dataset:
            return Dataset(data_df, horizon=self.horizon, one_hot_encoding=self.one_hot_encoding)
        return data_df


if __name__ == "__main__":
    DEEPDataset.check_and_download_deeplob_data()
    # Create object
    # Example:
    # Lobster datas

    # threshold --> 1std of label y ratio * 0.44 to have 0.33 in each tail
    # old class label method
    # dt = LOBDataset("indata/MSFT_2020_1/", horizon=100, sign_threshold=0.000301488)
    # new method 
    dt = LOBDataset("indata/MSFT_2020_1/", horizon=100, ratio_rolling_window=-1) 
    print("Test_data")
    print(dt.split_test_data())
    print("ML Dataset last entry")
    print(dt.split_test_data(torch_dataset=True)[-1])
    print("Manual Pollution")
    print(baseapproach.manual_pollution(dt.split_test_data()[0:4000]))

    # Deep lob data
    # Download DATA
    DEEPDataset.check_and_download_deeplob_data()
    # Create object
    dt = DEEPDataset("indata/FI-2010/", horizon=10)
    print("Test_data")
    print(dt.test_data()) 
    print("ML Dataset last entry")   
    print(dt.test_data(torch_dataset=True)[-1])
    print("Manual Pollution")
    print(baseapproach.manual_pollution(dt.test_data()[0:4000]))