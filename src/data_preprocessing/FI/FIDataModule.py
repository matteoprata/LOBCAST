
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class FIDataModule(pl.LightningDataModule):
    """ Splits the datasets in TRAIN, VALIDATION, TEST. """

    def __init__(self, train_set, val_set, batch_size):
        super().__init__()

        self.train_set = train_set
        self.val_set = val_set

        self.batch_size = batch_size

        self.x_shape = self.train_set.x_shape
        self.y_shape = self.train_set.y_shape

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)
