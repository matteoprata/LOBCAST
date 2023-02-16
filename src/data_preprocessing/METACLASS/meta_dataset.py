from torch.utils import data
import torch


class MetaDataset(data.Dataset):
    # x need to be of shape -> [n_samples, n_models*num_classes]
    def __init__(self, x, y, num_classes):
        """Initialization"""
        self.x = x
        self.y = y
        self.num_classes = num_classes
        self.length = x.shape[0]
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, i):
        return self.x[i], self.y[i]
