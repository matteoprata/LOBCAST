from torch.utils import data
import torch


class MetaDataset(data.Dataset):

    def __init__(self, samples, dataset_type, num_classes=3):
        """Initialization"""
        x, y = samples[0], samples[1]
        # x.shape = [n_samples, n_classes*n_models]
        # y.shape = [n_samples]

        self.dataset_type = dataset_type
        self.num_classes = num_classes
        self.n_samples = x.shape[0]
        self.x_shape = x.shape

        self.x = torch.from_numpy(x).type(torch.FloatTensor)
        self.y = torch.from_numpy(y).type(torch.LongTensor)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]
