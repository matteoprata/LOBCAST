import pytorch_lightning as pl

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score
import numpy as np
import src.config as co

from mlp import MLPModel

class MLP(pl.LightningModule):
    """ Multi layer perceptron. """

    def __init__(
            self,
            x_shape,
            y_shape,
            lr,
            hidden_layer_dim=128,
            remote_log=None
    ):

        super().__init__()
        self.lr = lr
        self.remote_log = remote_log

        self.x_shape = x_shape
        self.y_shape = y_shape

        self.hidden_layer_dim = hidden_layer_dim

        self.MLPModel = MLPModel(
            x_shape=self.x_shape,
            y_shape=self.y_shape,
            hidden_layer_dim=self.hidden_layer_dim
        )

    def forward(self, x):  # [batch_size x 40 x window]
        x = x.view(x.size(0), -1).float()
        return self.MLPModel(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = F.cross_entropy(prediction, y.float())
        return loss

    def validation_step(self, batch, batch_idx):
        prediction_ind, y_ind, loss_val = self.__validation_and_testing(batch)
        return prediction_ind, y_ind, loss_val

    def test_step(self, batch, batch_idx):
        prediction_ind, y_ind, loss_val = self.__validation_and_testing(batch)
        return prediction_ind, y_ind, loss_val

    def validation_epoch_end(self, validation_step_outputs):
        self.__validation_and_testing_end(validation_step_outputs, co.ModelSteps.VALIDATION)

    def test_epoch_end(self, validation_step_outputs):
        self.__validation_and_testing_end(validation_step_outputs, co.ModelSteps.TESTING)

    # COMMON
    def __validation_and_testing(self, batch):
        x, y = batch
        prediction = self(x)
        loss_val = F.cross_entropy(prediction, y.float())

        # deriving prediction from softmax probs
        print(y)
        prediction_ind = torch.argmax(prediction, dim=1)
        y_ind = torch.argmax(y, dim=1)
        print(y_ind)

        return prediction_ind, y_ind, loss_val

    def __validation_and_testing_end(self, validation_step_outputs, model_step):

        predictions, ys, loss_vals = [], [], []
        for prediction, y, loss_val in validation_step_outputs:
            predictions += prediction.tolist()
            ys += y.tolist()
            loss_vals += [loss_val.item()]

        precision, recall, f1score, _ = prfs(predictions, ys, average="macro", zero_division=0)
        accuracy = accuracy_score(predictions, ys)

        val_dict = {
            model_step.value + co.Metrics.LOSS.value:      float(np.sum(loss_vals)),
            model_step.value + co.Metrics.F1.value:        float(f1score),
            model_step.value + co.Metrics.PRECISION.value: float(precision),
            model_step.value + co.Metrics.RECALL.value:    float(recall),
            model_step.value + co.Metrics.ACCURACY.value:  float(accuracy)
            }

        # print(model_step)
        # print(predictions)
        # print(ys)
        # print(val_dict)

        # for saving best model
        self.log(model_step.value + co.Metrics.F1.value, f1score, prog_bar=True)

        if self.remote_log is not None:  # log to wandb
            self.remote_log.log(val_dict)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
