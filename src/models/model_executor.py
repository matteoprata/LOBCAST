import pytorch_lightning as pl

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score
import numpy as np
import src.config as co
import wandb
from collections import Counter

class NNEngine(pl.LightningModule):
    """ Multi layer perceptron. """

    def __init__(self,  neural_architecture, lr, remote_log=None ):

        super().__init__()
        self.lr = lr
        self.remote_log = remote_log
        self.neural_architecture = neural_architecture

    def forward(self, x):
        return self.neural_architecture(x)

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

    def training_epoch_end(self, validation_step_outputs):
        losses   = [el["loss"].item() for el in validation_step_outputs]
        sum_losses = float(np.sum(losses))
        self.log(co.ModelSteps.TRAINING.value + co.Metrics.LOSS.value, sum_losses, prog_bar=True)

        if self.remote_log is not None:
            self.remote_log.log({co.ModelSteps.TRAINING.value + co.Metrics.LOSS.value: sum_losses})

    def validation_epoch_end(self, validation_step_outputs):
        self.__validation_and_testing_end(validation_step_outputs, model_step=co.ModelSteps.VALIDATION)

    def test_epoch_end(self, validation_step_outputs):
        self.__validation_and_testing_end(validation_step_outputs, model_step=co.ModelSteps.TESTING)

    # COMMON
    def __validation_and_testing(self, batch):
        x, y = batch
        prediction = self(x)
        loss_val = F.cross_entropy(prediction, y.float())

        # deriving prediction from softmax probs
        prediction_ind = torch.argmax(prediction, dim=1)
        y_ind = torch.argmax(y, dim=1)

        return prediction_ind, y_ind, loss_val

    def __validation_and_testing_end(self, validation_step_outputs, model_step):

        predictions, ys, loss_vals = [], [], []
        for prediction, y, loss_val in validation_step_outputs:
            predictions += prediction.tolist()
            ys += y.tolist()
            loss_vals += [loss_val.item()]

        # TODO: recode
        counts_class = {k: 1/v for k, v in Counter(ys).items()}
        yweights = np.ones(shape=len(ys)) * counts_class[0]
        yweights[ys == 2] = counts_class[2]
        yweights[ys == 1] = counts_class[1]

        precision, recall, f1score, _ = prfs(predictions, ys, average="weighted", sample_weight=yweights, zero_division=0)
        accuracy = accuracy_score(predictions, ys)

        val_dict = {
            model_step.value + co.Metrics.LOSS.value:      float(np.sum(loss_vals)),
            model_step.value + co.Metrics.F1.value:        float(f1score),
            model_step.value + co.Metrics.PRECISION.value: float(precision),
            model_step.value + co.Metrics.RECALL.value:    float(recall),
            model_step.value + co.Metrics.ACCURACY.value:  float(accuracy)
            }

        # for saving best model
        self.log(model_step.value + co.Metrics.F1.value, f1score, prog_bar=True)

        if self.remote_log is not None:  # log to wandb
            self.remote_log.log(val_dict)
            self.remote_log.log({model_step.value + "_conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=ys, preds=predictions,
                class_names=co.CLASS_NAMES,
                title=model_step.value + "_conf_mat")})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
