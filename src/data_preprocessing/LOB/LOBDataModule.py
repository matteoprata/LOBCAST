
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class LOBDataModule(pl.LightningDataModule):
    """ Splits the datasets in TRAIN, VALIDATION, TEST. """

    def __init__(self, train_set, val_set, test_set, batch_size, is_shuffle_train=True):
        super().__init__()

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.batch_size = batch_size
        self.is_shuffle_train = is_shuffle_train

        self.x_shape = self.train_set.x_shape
        self.y_shape = self.train_set.y_shape

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.is_shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
