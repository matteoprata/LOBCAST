
from torch.utils import data
import torch.nn.functional as F
import torch
import src.constants as cst
import numpy as np
import collections

LOSS_WEIGHTS_DICT = {m: 1e6 for m in cst.Models}
# LOSS_WEIGHTS_DICT[cst.Models.ATNBoF] = 1e3


class FIDataset(data.Dataset):
    """ Characterizes a dataset for PyTorch. """

    def __init__(self, x, y, chosen_model=None, num_classes=3, num_snapshots=100, one_hot_encoding=False):
        self.num_snapshots = num_snapshots
        self.num_classes = num_classes
        self.chosen_model = chosen_model
        self.loss_weights = None

        self.x = torch.from_numpy(x).type(torch.FloatTensor)  # torch.Size([203800, 40])
        self.y = torch.from_numpy(y).type(torch.LongTensor)  # torch.Size([203800])

        # DEEPLOBATT, unlike the others models, does the predictions for all the horizons, using an encoder-decoder
        if chosen_model == cst.Models.DEEPLOBATT:
            self.y = F.one_hot(self.y.to(torch.int64), num_classes=self.num_classes).float()
            self.y = torch.permute(self.y, (0, 2, 1))
            # y.shape = (n_samples, num_classes, num_horizons)
        else:
            self.ys_occurrences = collections.Counter(y)
            occs = np.array([self.ys_occurrences[k] for k in sorted(self.ys_occurrences)])
            self.loss_weights = torch.Tensor(LOSS_WEIGHTS_DICT[chosen_model] / occs)

        self.x_shape = (self.num_snapshots, self.x.shape[1])

    def __len__(self):
        """ Denotes the total number of samples. """
        return self.x.shape[0]-self.num_snapshots

    def __getitem__(self, index):
        """ Generates samples of data. """
        return self.x[index: index+self.num_snapshots], self.y[index+self.num_snapshots-1], cst.Stocks.FI.value[0]
