
import pytorch_lightning as pl


class LOBDataModule(pl.LightningDataModule):
    def __init__(self, train_set, test_set, batch_size: int = 32):
        super().__init__()
        self.train_set = train_set
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            pass
        elif stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)