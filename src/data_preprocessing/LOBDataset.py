

from torch.utils import data
import torch.nn.functional as F
import torch
import numpy as np


class LOBDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """

    def __init__(self, x, y, num_classes=3, one_hot_encoding=True):
        self.num_classes = num_classes

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

        if one_hot_encoding:
            self.y = F.one_hot(self.y.to(torch.int64), num_classes=self.num_classes)

        self.x_shape, self.y_shape = np.prod(list(self.x[0].shape)), np.prod(list(self.y[0].shape))

    def __len__(self):
        """ Denotes the total number of samples. """
        return self.x.shape[0]

    def __getitem__(self, index):
        """ Generates samples of data. """
        return self.x[index], self.y[index]
