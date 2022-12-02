

from torch.utils import data
import torch.nn.functional as F
import numpy as np
import torch
import src.config as co

from src.data_preprocessing.LOB.LOBSTERDataBuilder import LOBSTERDataBuilder

class LOBDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """

    def __init__(
            self,
            dataset_type,
            stocks,
            start_end_trading_day,
            stockName2mu=dict(),
            stockName2sigma=dict(),
            num_classes=3,
            one_hot_encoding=False
    ):
        self.dataset_type = dataset_type
        self.stocks = stocks
        self.start_end_trading_day = start_end_trading_day
        self.num_classes = num_classes

        self.stockName2mu, self.stockName2sigma = stockName2mu, stockName2sigma

        stockName2databuilder = dict()
        for stock in list(set(co.CHOSEN_STOCKS['train'].value + co.CHOSEN_STOCKS['test'].value)):

            path = \
                co.DATASET_LOBSTER + \
                f'_data_dwn_48_332__{stock}_{co.CHOSEN_PERIOD.value["train"][0]}_{co.CHOSEN_PERIOD.value["test"][1]}_10'

            normalization_mean = stockName2mu[stock] if stock in stockName2mu else None
            normalization_std = stockName2sigma[stock] if stock in stockName2sigma else None

            print(dataset_type, '\t', stocks, '\t', stock, '\t', start_end_trading_day, '\t', normalization_mean, '\t', normalization_std, '\t', path)

            databuilder = LOBSTERDataBuilder(
                stock,
                path,
                dataset_type=dataset_type,
                start_end_trading_day=start_end_trading_day,
                crop_trading_day_by=60*30,
                window_size_forward=co.FORWARD_WINDOW,
                window_size_backward=co.BACKWARD_WINDOW,
                normalization_mean=normalization_mean,
                normalization_std=normalization_std,
                label_dynamic_scaler=co.LABELING_SIGMA_SCALER,
                is_data_preload=co.IS_DATA_PRELOAD
            )

            self.stockName2mu[stock], self.stockName2sigma[stock] = databuilder.normalization_means, databuilder.normalization_stds
            stockName2databuilder[stock] = databuilder

        print('stockName2mu:', self.stockName2mu)
        print('stockName2sigma:', self.stockName2sigma)

        self.stock2orderNlen = dict()
        self.x, self.y = list(), list()
        for i, stock in enumerate(self.stocks):
            databuilder = stockName2databuilder[stock]
            samplesX, samplesY = databuilder.get_samples_x(), databuilder.get_samples_y()
            self.stock2orderNlen[stock] = (i, len(samplesX))
            self.x.extend(samplesX)
            self.y.extend(samplesY)

        self.x = torch.from_numpy(np.array(self.x)).type(torch.FloatTensor)
        self.y = torch.from_numpy(np.array(self.y)).type(torch.LongTensor)

        self.x_shape = tuple(self.x[0].shape)

        print('stock2orderNlen:', self.stock2orderNlen)


        print()
        print()


        if one_hot_encoding:
            self.y = F.one_hot(self.y.to(torch.int64), num_classes=self.num_classes)


    def __len__(self):
        """ Denotes the total number of samples. """
        return self.x.shape[0]

    def __getitem__(self, index):
        """ Generates samples of data. """
        return self.x[index], self.y[index]
