

from torch.utils import data
import torch.nn.functional as F
import numpy as np
import torch
import src.constants as cst

from src.data_preprocessing.LOB.LOBSTERDataBuilder import LOBSTERDataBuilder
from src.config import Configuration


class LOBDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """

    def __init__(
            self,
            config: Configuration,
            dataset_type,
            stocks,
            start_end_trading_day,
            stockName2mu=dict(),
            stockName2sigma=dict(),
            num_classes=3,
            num_snapshots=100,
            one_hot_encoding=False
    ):
        self.dataset_type = dataset_type
        self.stocks = stocks
        self.start_end_trading_day = start_end_trading_day
        self.num_snapshots = num_snapshots
        self.num_classes = num_classes

        self.stockName2mu, self.stockName2sigma = stockName2mu, stockName2sigma

        stockName2databuilder = dict()

        # Choose the stock names to open to build the specific dataset.
        # No need to open all for test set, because mu/sig are pre-computed when prev opened train and dev
        stocksToOpen = None
        if dataset_type == cst.DatasetType.TRAIN:
            # we open also the TEST stock(s) to determine mu and sigma for normalization, needed for all
            stocksToOpen = list(set(config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value + config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value))  # = [LYFT, NVDA]
        elif dataset_type == cst.DatasetType.VALIDATION:
            stocksToOpen = config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value  # = [LYFT]
        elif dataset_type == cst.DatasetType.TEST:
            stocksToOpen = config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value   # = [NVDA]

        for stock in stocksToOpen:
            path = cst.DATASET_LOBSTER + f'_data_dwn_48_332__{stock}_{config.CHOSEN_PERIOD.value["train"][0]}_{config.CHOSEN_PERIOD.value["test"][1]}_10'

            normalization_mean = stockName2mu[stock] if stock in stockName2mu else None
            normalization_std = stockName2sigma[stock] if stock in stockName2sigma else None

            print(dataset_type, '\t', stocks, '\t', stock, '\t', start_end_trading_day, '\t', normalization_mean, '\t', normalization_std, '\t', path)

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
                normalization_mean=normalization_mean,
                normalization_std=normalization_std,
                num_snapshots=num_snapshots,
                label_dynamic_scaler=config.HYPER_PARAMETERS[cst.LearningHyperParameter.LABELING_SIGMA_SCALER],
                is_data_preload=config.IS_DATA_PRELOAD
            )

            self.stockName2mu[stock], self.stockName2sigma[stock] = databuilder.normalization_means, databuilder.normalization_stds
            stockName2databuilder[stock] = databuilder

        print('stockName2mu:', self.stockName2mu)
        print('stockName2sigma:', self.stockName2sigma)

        self.stock2orderNlen = dict()
        self.x, self.y, self.stock_sym_name = list(), list(), list()
        for stock in self.stocks:
            print("Handling", stock, "for dataset", dataset_type)
            databuilder = stockName2databuilder[stock]
            samplesX, samplesY = databuilder.get_samples_x(), databuilder.get_samples_y()
            self.x.extend(samplesX)
            self.y.extend(samplesY)
            self.stock_sym_name.extend([stock]*len(samplesY))

        self.x = torch.from_numpy(np.array(self.x)).type(torch.FloatTensor)
        self.y = torch.from_numpy(np.array(self.y)).type(torch.LongTensor)

        self.x_shape = tuple(self.x[0].shape)

        # print(len(self.x), len(self.y), len(self.stock_sym_name))
        print()
        print()

        if one_hot_encoding:
            self.y = F.one_hot(self.y.to(torch.int64), num_classes=self.num_classes)

    def __len__(self):
        """ Denotes the total number of samples. """
        return self.x.shape[0]

    def __getitem__(self, index):
        """ Generates samples of data. """
        return self.x[index], self.y[index], self.stock_sym_name[index]
