#
# from torch.utils import data
# import torch.nn.functional as F
# import torch
# import src.constants as cst
# import numpy as np
# import collections
#
# # LOSS_WEIGHTS_DICT[cst.Models.ATNBoF] = 1e3
#
#
# class FIDataset(data.Dataset):
#     """ Characterizes a dataset for PyTorch. """
#
#     def __init__(self, x, y, chosen_model=None, num_classes=3, observation_length=100, one_hot_encoding=False):
#         self.observation_length = observation_length
#         self.num_classes = num_classes
#         self.chosen_model = chosen_model
#         self.loss_weights = None
#
#         self.x = torch.from_numpy(x).type(torch.FloatTensor)  # torch.Size([203800, 40])
#         self.y = torch.from_numpy(y).type(torch.LongTensor)  # torch.Size([203800])
#
#         self.ys_occurrences = collections.Counter(y)
#         occs = np.array([self.ys_occurrences[k] for k in sorted(self.ys_occurrences)])
#
#         LOSS_WEIGHTS_DICT = {m: 1e6 for m in cst.Models}
#         self.loss_weights = torch.Tensor(LOSS_WEIGHTS_DICT[chosen_model] / occs)
#
#         self.x_shape = (self.observation_length, self.x.shape[1])  # shape of a single sample
#
#     def __len__(self):
#         """ Denotes the total number of samples. """
#         return self.x.shape[0]-self.observation_length
#
#     def __getitem__(self, index):
#         """ Generates samples of data. """
#         sample = self.x[index: index+self.observation_length], self.y[index + self.observation_length - 1]
#         stock_id = cst.Stocks.FI.value[0]
#         return sample, stock_id
