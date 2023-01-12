

from torch.utils import data
import torch.nn.functional as F
import torch
import src.constants as cst
import numpy as np


class FIDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """

    def __init__(self, x, y, num_classes=3, one_hot_encoding=False):
        self.num_classes = num_classes

        self.x = torch.from_numpy(x).type(torch.FloatTensor)
        self.y = torch.from_numpy(y).type(torch.LongTensor)

        if one_hot_encoding:
            self.y = F.one_hot(self.y.to(torch.int64), num_classes=self.num_classes)

        self.x_shape = tuple(self.x[0].shape)

    def __len__(self):
        """ Denotes the total number of samples. """
        return self.x.shape[0]

    def __getitem__(self, index):
        """ Generates samples of data. """
        return self.x[index], self.y[index], cst.Stocks.FI.value[0]
